//------------------------------------------------------
// YOLO26 (Ultralytics) - anchor-free, DFL-free head
//
// Supported exported output layouts (auto-detected at runtime):
//   [E2E]    1 tensor  (6, N)          x1,y1,x2,y2,score,class in input pixels, NMS-free
//   [CONCAT] 1 tensor  (A, 4+nc)       A = sum of grid*grid over strides 8/16/32
//   [SCALE]  3 tensors (W, H, 4+nc)    per-stride raw head, W==H==input/stride
//   [SPLIT]  6 tensors (W, H, 4)+(W, H, nc) box/cls separated per stride
//
// Box regression is direct distance (l,t,r,b) from the cell center in
// grid units (no DFL). Class scores pass through sigmoid unless the
// graph already contains it (detected from quantization params).
//------------------------------------------------------
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "img_process/img_process.h"
#include "nn_utils/sigmoid.h"
#include "nn_utils/iou.h"
#include "nn_utils/nms.h"
#include "nn_utils/quantize.h"
#include "nn_utils/tensor.h"
#include "mmf2_module.h"
#include "module_vipnn.h"
#include "hal_cache.h"

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

// Set to 0 if the exported graph already ends with a sigmoid on the
// class branch and the auto-detection gets it wrong.
#define YOLO26_AUTO_DETECT_CLS_ACT   1
#define YOLO26_APPLY_SIGMOID_DEFAULT 1

// One-to-many head still benefits from NMS; E2E layout skips it.
#define YOLO26_USE_NMS               1

static box_t res_box[MAX_DETECT_OBJ_NUM];
static box_t *p_res_box[MAX_DETECT_OBJ_NUM];
static int box_idx;

static float yolo26_confidence_thresh = 0.5;
static float yolo26_nms_thresh = 0.3;

static int yolo26_in_width, yolo26_in_height;

void *yolo26_get_network_filename_init(void)
{
	return (void *)"NN_MDL/yolo26.nb";	// fix name for NN model binary
}

int yolo26_preprocess(void *data_in, nn_data_param_t *data_param, void *tensor_in, nn_tensor_param_t *tensor_param)
{
	void **tensor = (void **)tensor_in;
	rect_t *roi = &data_param->img.roi;

	img_t img_in, img_out;

	img_in.width  = data_param->img.width;
	img_in.height = data_param->img.height;
	img_out.width  = tensor_param->dim[0].size[0];
	img_out.height = tensor_param->dim[0].size[1];
	yolo26_in_width  = tensor_param->dim[0].size[0];
	yolo26_in_height = tensor_param->dim[0].size[1];

	img_in.data   = (unsigned char *)data_in;
	img_out.data  = (unsigned char *)tensor[0];

	if (img_in.width == img_out.width && img_in.height == img_out.height) {
		img_dma_copy(img_out.data, img_in.data, img_out.width * img_out.height * 3);
	} else {
		img_resize_planar(&img_in, roi, &img_out);
	}

	dcache_clean_by_addr((uint32_t *)img_out.data, img_out.width * img_out.height * 3);

	return 0;
}

//------------------------------------------------------
// layout detection
//------------------------------------------------------
typedef enum {
	YOLO26_LAYOUT_UNKNOWN = 0,
	YOLO26_LAYOUT_E2E,
	YOLO26_LAYOUT_CONCAT,
	YOLO26_LAYOUT_SCALE,
	YOLO26_LAYOUT_SPLIT,
} yolo26_layout_t;

static yolo26_layout_t yolo26_layout = YOLO26_LAYOUT_UNKNOWN;
static int yolo26_last_count = -1;
static int yolo26_nc = 0;             // number of classes
static int yolo26_cls_is_prob = 0;    // 1: graph already applied sigmoid
static int yolo26_layout_printed = 0;

// number of valid dims (acuity pads size[] with 1s)
static int tensor_rank(nn_tensor_dim_t *dim)
{
	int r = dim->num;
	if (r <= 0 || r > 6) {
		r = 6;
	}
	while (r > 1 && dim->size[r - 1] == 1) {
		r--;
	}
	return r;
}

// guess whether class scores are already probabilities from the
// quantization parameters: an asymmetric uint8 tensor spanning only
// [0, ~1] cannot hold logits.
static int cls_tensor_is_prob(nn_tensor_format_t *fmt)
{
#if YOLO26_AUTO_DETECT_CLS_ACT
	switch (fmt->buf_type) {
	case VIP_BUFFER_FORMAT_UINT8: {
		float lo = (0 - fmt->zero_point) * fmt->scale;
		float hi = (255 - fmt->zero_point) * fmt->scale;
		return (lo >= -0.01f && hi <= 1.5f);
	}
	case VIP_BUFFER_FORMAT_INT8: {
		float hi = 127.0f / (float)(1 << fmt->fix_point_pos);
		return (hi <= 1.5f);
	}
	case VIP_BUFFER_FORMAT_INT16: {
		float hi = 32767.0f / (float)(1 << fmt->fix_point_pos);
		return (hi <= 1.5f);
	}
	default:
		return 0;   // float/bf16: assume logits, apply sigmoid
	}
#else
	return !YOLO26_APPLY_SIGMOID_DEFAULT;
#endif
}

static void yolo26_detect_layout(nn_tensor_param_t *param)
{
	int in_w = yolo26_in_width > 0 ? yolo26_in_width : 416;

	yolo26_layout = YOLO26_LAYOUT_UNKNOWN;

	if (param->count == 1) {
		nn_tensor_dim_t *d = &param->dim[0];
		int r = tensor_rank(d);
		// (6, N) or (N, 6): end-to-end output
		if (r >= 2 && (d->size[0] == 6 || d->size[1] == 6)) {
			yolo26_layout = YOLO26_LAYOUT_E2E;
			yolo26_nc = 0; // class id carried in the tensor
			return;
		}
		// (A, 4+nc): concatenated raw head, A = sum of grid^2
		int a_expect = 0;
		for (int s = 8; s <= 32; s *= 2) {
			a_expect += (in_w / s) * (in_w / s);
		}
		if (r >= 2 && (int)d->size[0] == a_expect && d->size[1] > 4) {
			yolo26_layout = YOLO26_LAYOUT_CONCAT;
			yolo26_nc = d->size[1] - 4;
			yolo26_cls_is_prob = cls_tensor_is_prob(&param->format[0]);
			return;
		}
		if (r >= 2 && (int)d->size[1] == a_expect && d->size[0] > 4) {
			// channel-first variant (4+nc, A): treat via CONCAT with swap
			yolo26_layout = YOLO26_LAYOUT_CONCAT;
			yolo26_nc = d->size[0] - 4;
			yolo26_cls_is_prob = cls_tensor_is_prob(&param->format[0]);
			return;
		}
	} else if (param->count == 3) {
		// per-scale (W, H, 4+nc)
		nn_tensor_dim_t *d = &param->dim[0];
		if (d->size[2] > 4) {
			yolo26_layout = YOLO26_LAYOUT_SCALE;
			yolo26_nc = d->size[2] - 4;
			yolo26_cls_is_prob = cls_tensor_is_prob(&param->format[0]);
			return;
		}
	} else if (param->count == 6) {
		// split box(4)/cls(nc) per scale: find a cls tensor to get nc
		for (int n = 0; n < 6; n++) {
			if (param->dim[n].size[2] != 4) {
				yolo26_layout = YOLO26_LAYOUT_SPLIT;
				yolo26_nc = param->dim[n].size[2];
				yolo26_cls_is_prob = cls_tensor_is_prob(&param->format[n]);
				return;
			}
		}
	}
}

static void yolo26_print_layout(nn_tensor_param_t *param)
{
	if (yolo26_layout_printed) {
		return;
	}
	yolo26_layout_printed = 1;

	static const char *names[] = {"UNKNOWN", "E2E", "CONCAT", "SCALE", "SPLIT"};
	printf("[YOLO26] input %dx%d, output tensors: %d\n\r", yolo26_in_width, yolo26_in_height, param->count);
	for (int n = 0; n < param->count && n < 16; n++) {
		nn_tensor_dim_t *d = &param->dim[n];
		nn_tensor_format_t *f = &param->format[n];
		printf("[YOLO26]   out[%d] dims(", n);
		for (int k = 0; k < d->num && k < 6; k++) {
			printf("%s%d", k ? "," : "", (int)d->size[k]);
		}
		printf(") buf_type %d scale %f zp %d fixpos %d\n\r", f->buf_type, f->scale, f->zero_point, f->fix_point_pos);
	}
	printf("[YOLO26] layout: %s, classes: %d, cls already prob: %d\n\r",
		   names[yolo26_layout], yolo26_nc, yolo26_cls_is_prob);
	if (yolo26_layout == YOLO26_LAYOUT_UNKNOWN) {
		printf("[YOLO26] ERROR: unsupported output layout, please check the exported model\n\r");
	}
}

//------------------------------------------------------
// decode helpers
//------------------------------------------------------
static float yolo26_cls_score(float raw)
{
	return yolo26_cls_is_prob ? raw : sigmoid(raw);
}

static void yolo26_emit_box(float x1, float y1, float x2, float y2, float score, int class_idx)
{
	if (box_idx >= MAX_DETECT_OBJ_NUM) {
		return;
	}

	x1 = max(x1, 0);
	y1 = max(y1, 0);
	x2 = min(x2, 1);
	y2 = min(y2, 1);
	if (x2 <= x1 || y2 <= y1) {
		return;
	}

	res_box[box_idx].class_idx = class_idx;
	res_box[box_idx].prob = score;
	res_box[box_idx].x = x1;
	res_box[box_idx].y = y1;
	res_box[box_idx].w = x2 - x1;
	res_box[box_idx].h = y2 - y1;
	box_idx++;
}

// decode one cell of a raw head tensor laid out (W, H, C) with
// idx = c*H*W + y*W + x  (same indexing as model_yolo.c)
static void yolo26_decode_cell(void *data, nn_tensor_format_t *fmt, int w, int h, int i, int j, int stride, int nc)
{
	int hw = h * w;
	int base = j * w + i;

	// find best class first (cheap threshold before reading box regs)
	int best_c = -1;
	float best_s = 0;
	for (int c = 0; c < nc; c++) {
		float s = yolo26_cls_score(get_tensor_value(data, (4 + c) * hw + base, fmt));
		if (s > best_s) {
			best_s = s;
			best_c = c;
		}
	}
	if (best_c < 0 || best_s < yolo26_confidence_thresh) {
		return;
	}

	float l = get_tensor_value(data, 0 * hw + base, fmt);
	float t = get_tensor_value(data, 1 * hw + base, fmt);
	float r = get_tensor_value(data, 2 * hw + base, fmt);
	float b = get_tensor_value(data, 3 * hw + base, fmt);

	// distances are in grid units around the cell center
	float cx = i + 0.5f;
	float cy = j + 0.5f;
	float x1 = (cx - l) * stride / (float)yolo26_in_width;
	float y1 = (cy - t) * stride / (float)yolo26_in_height;
	float x2 = (cx + r) * stride / (float)yolo26_in_width;
	float y2 = (cy + b) * stride / (float)yolo26_in_height;

	yolo26_emit_box(x1, y1, x2, y2, best_s, best_c);
}

// decode the E2E tensor: rows of [x1,y1,x2,y2,score,class] in input pixels
static void yolo26_decode_e2e(void *data, nn_tensor_format_t *fmt, nn_tensor_dim_t *dim)
{
	int ndet, elem_stride, det_stride;
	if (dim->size[0] == 6) {          // (6, N): element index fastest
		ndet = dim->size[1];
		elem_stride = 1;
		det_stride = 6;
	} else {                          // (N, 6)
		ndet = dim->size[0];
		elem_stride = ndet;
		det_stride = 1;
	}

	for (int a = 0; a < ndet; a++) {
		float score = get_tensor_value(data, a * det_stride + 4 * elem_stride, fmt);
		if (score < yolo26_confidence_thresh) {
			continue;
		}
		float x1 = get_tensor_value(data, a * det_stride + 0 * elem_stride, fmt) / (float)yolo26_in_width;
		float y1 = get_tensor_value(data, a * det_stride + 1 * elem_stride, fmt) / (float)yolo26_in_height;
		float x2 = get_tensor_value(data, a * det_stride + 2 * elem_stride, fmt) / (float)yolo26_in_width;
		float y2 = get_tensor_value(data, a * det_stride + 3 * elem_stride, fmt) / (float)yolo26_in_height;
		int cls = (int)get_tensor_value(data, a * det_stride + 5 * elem_stride, fmt);
		yolo26_emit_box(x1, y1, x2, y2, score, cls);
	}
}

// decode the concatenated raw head (A, 4+nc) / (4+nc, A);
// anchors ordered stride 8 grid, then 16, then 32
static void yolo26_decode_concat(void *data, nn_tensor_format_t *fmt, nn_tensor_dim_t *dim)
{
	int a_total, c_total, a_stride, c_stride;
	int in_w = yolo26_in_width;
	int a_expect = 0;
	for (int s = 8; s <= 32; s *= 2) {
		a_expect += (in_w / s) * (in_w / s);
	}

	if ((int)dim->size[0] == a_expect) {  // (A, C): anchor index fastest
		a_total = dim->size[0];
		c_total = dim->size[1];
		a_stride = 1;
		c_stride = a_total;
	} else {                              // (C, A)
		a_total = dim->size[1];
		c_total = dim->size[0];
		a_stride = c_total;
		c_stride = 1;
	}
	int nc = c_total - 4;

	int a0 = 0;
	for (int stride = 8; stride <= 32; stride *= 2) {
		int gw = yolo26_in_width / stride;
		int gh = yolo26_in_height / stride;
		for (int j = 0; j < gh; j++) {
			for (int i = 0; i < gw; i++) {
				int a = a0 + j * gw + i;

				int best_c = -1;
				float best_s = 0;
				for (int c = 0; c < nc; c++) {
					float s = yolo26_cls_score(get_tensor_value(data, a * a_stride + (4 + c) * c_stride, fmt));
					if (s > best_s) {
						best_s = s;
						best_c = c;
					}
				}
				if (best_c < 0 || best_s < yolo26_confidence_thresh) {
					continue;
				}

				float l = get_tensor_value(data, a * a_stride + 0 * c_stride, fmt);
				float t = get_tensor_value(data, a * a_stride + 1 * c_stride, fmt);
				float r = get_tensor_value(data, a * a_stride + 2 * c_stride, fmt);
				float b = get_tensor_value(data, a * a_stride + 3 * c_stride, fmt);

				float cx = i + 0.5f;
				float cy = j + 0.5f;
				float x1 = (cx - l) * stride / (float)yolo26_in_width;
				float y1 = (cy - t) * stride / (float)yolo26_in_height;
				float x2 = (cx + r) * stride / (float)yolo26_in_width;
				float y2 = (cy + b) * stride / (float)yolo26_in_height;

				yolo26_emit_box(x1, y1, x2, y2, best_s, best_c);
			}
		}
		a0 += gw * gh;
	}
}

//------------------------------------------------------
// postprocess entry
//------------------------------------------------------
int yolo26_postprocess(void *tensor_out, nn_tensor_param_t *param, void *res)
{
	void **tensor = (void **)tensor_out;
	objdetect_res_t *od_res = (objdetect_res_t *)res;

	yolo26_last_count = param->count;
	box_idx = 0;
	memset(res_box, 0, sizeof(res_box));

	if (yolo26_layout == YOLO26_LAYOUT_UNKNOWN) {
		yolo26_detect_layout(param);
		yolo26_print_layout(param);
		if (yolo26_layout == YOLO26_LAYOUT_UNKNOWN) {
			return 0;
		}
	}

	switch (yolo26_layout) {
	case YOLO26_LAYOUT_E2E:
		yolo26_decode_e2e(tensor[0], &param->format[0], &param->dim[0]);
		break;

	case YOLO26_LAYOUT_CONCAT:
		yolo26_decode_concat(tensor[0], &param->format[0], &param->dim[0]);
		break;

	case YOLO26_LAYOUT_SCALE:
		for (int n = 0; n < param->count; n++) {
			nn_tensor_dim_t *d = &param->dim[n];
			int w = d->size[0];
			int h = d->size[1];
			int stride = yolo26_in_width / w;
			for (int j = 0; j < h; j++) {
				for (int i = 0; i < w; i++) {
					yolo26_decode_cell(tensor[n], &param->format[n], w, h, i, j, stride, yolo26_nc);
				}
			}
		}
		break;

	case YOLO26_LAYOUT_SPLIT:
		// pair each box tensor (C==4) with the cls tensor of the same grid
		for (int n = 0; n < param->count; n++) {
			nn_tensor_dim_t *db = &param->dim[n];
			if (db->size[2] != 4) {
				continue;
			}
			int w = db->size[0];
			int h = db->size[1];
			int stride = yolo26_in_width / w;
			int m_cls = -1;
			for (int m = 0; m < param->count; m++) {
				if (m != n && (int)param->dim[m].size[0] == w && param->dim[m].size[2] != 4) {
					m_cls = m;
					break;
				}
			}
			if (m_cls < 0) {
				continue;
			}
			int hw = h * w;
			for (int j = 0; j < h; j++) {
				for (int i = 0; i < w; i++) {
					int base = j * w + i;

					int best_c = -1;
					float best_s = 0;
					for (int c = 0; c < yolo26_nc; c++) {
						float s = yolo26_cls_score(get_tensor_value(tensor[m_cls], c * hw + base, &param->format[m_cls]));
						if (s > best_s) {
							best_s = s;
							best_c = c;
						}
					}
					if (best_c < 0 || best_s < yolo26_confidence_thresh) {
						continue;
					}

					float l = get_tensor_value(tensor[n], 0 * hw + base, &param->format[n]);
					float t = get_tensor_value(tensor[n], 1 * hw + base, &param->format[n]);
					float r = get_tensor_value(tensor[n], 2 * hw + base, &param->format[n]);
					float b = get_tensor_value(tensor[n], 3 * hw + base, &param->format[n]);

					float cx = i + 0.5f;
					float cy = j + 0.5f;
					yolo26_emit_box((cx - l) * stride / (float)yolo26_in_width,
									(cy - t) * stride / (float)yolo26_in_height,
									(cx + r) * stride / (float)yolo26_in_width,
									(cy + b) * stride / (float)yolo26_in_height,
									best_s, best_c);
				}
			}
		}
		break;

	default:
		return 0;
	}

#if YOLO26_USE_NMS
	if (yolo26_layout != YOLO26_LAYOUT_E2E) {
		int classes = yolo26_nc > 0 ? yolo26_nc : 80;
		do_nms(classes, box_idx, yolo26_nms_thresh, res_box, p_res_box, DIOU);
	}
#endif

	int od_num = 0;
	for (int i = 0; i < box_idx; i++) {
		box_t *obj = &res_box[i];

		if (obj->invalid == 0) {
			od_res[od_num].result[0] = obj->class_idx;
			od_res[od_num].result[1] = obj->prob;
			od_res[od_num].result[2] = obj->x;	// top_x
			od_res[od_num].result[3] = obj->y;	// top_y
			od_res[od_num].result[4] = obj->x + obj->w; // bottom_x
			od_res[od_num].result[5] = obj->y + obj->h; // bottom_y
			od_num++;
		}
	}

	return od_num;
}

void yolo26_set_confidence_thresh(void *confidence_thresh)
{
	yolo26_confidence_thresh = *(float *)confidence_thresh;
	printf("set yolo26 confidence thresh to %f\n\r", *(float *)confidence_thresh);
}

void yolo26_set_nms_thresh(void *nms_thresh)
{
	yolo26_nms_thresh = *(float *)nms_thresh;
	printf("set yolo26 NMS thresh to %f\n\r", *(float *)nms_thresh);
}

void yolo26_set_init_info(void *m)
{
	nnmodel_t *model = (nnmodel_t *)m;
	yolo26_in_width  = model->input_param.dim[0].size[0];
	yolo26_in_height = model->input_param.dim[0].size[1];
	printf("[YOLO26] init: input %dx%d\n\r", yolo26_in_width, yolo26_in_height);
}

//------------------------------------------------------
// diagnostics: expose decode state for on-screen display
// (serial console is not always available on a deployed board)
//------------------------------------------------------
void yolo26_get_debug_info(char *buf, int len)
{
	static const char *names[] = { "UNKNOWN", "E2E", "CONCAT", "SCALE", "SPLIT" };
	snprintf(buf, len, "Y26 %s t%d nc%d p%d in%d conf%d",
			 names[yolo26_layout], yolo26_last_count, yolo26_nc,
			 yolo26_cls_is_prob, yolo26_in_width,
			 (int)(yolo26_confidence_thresh * 100));
}

nnmodel_t yolo26 = {
	.nb 			= yolo26_get_network_filename_init,
	.preprocess 	= yolo26_preprocess,
	.postprocess 	= yolo26_postprocess,
	.model_src 		= MODEL_SRC_FILE,
	.set_init_info   = yolo26_set_init_info,
	.set_confidence_thresh   = yolo26_set_confidence_thresh,
	.set_nms_thresh     = yolo26_set_nms_thresh,

	.name = "YOLO26"
};

