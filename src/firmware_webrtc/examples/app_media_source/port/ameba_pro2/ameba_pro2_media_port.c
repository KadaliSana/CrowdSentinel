/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "logging.h"
#include "demo_config.h"
#include "ameba_pro2_media_port.h"
#include "platform_opts.h"

#include "mmf2_link.h"
#include "mmf2_siso.h"
#include "mmf2_miso.h"

#include "module_video.h"
#include "module_audio.h"
#include "module_g711.h"
#include "module_opusc.h"
#include "module_opusd.h"
#include "opus_defines.h"

#include "avcodec.h"

#include "video_api.h"

#if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
#include "module_vipnn.h"
#include "nn_utils/class_name.h"
#include "osd_render.h"

/* Compile-time assert that works under -std=gnu99 (no _Static_assert). */
#define MEDIA_PORT_ASSERT_CONCAT_( a, b ) a##b
#define MEDIA_PORT_ASSERT_CONCAT( a, b )  MEDIA_PORT_ASSERT_CONCAT_( a, b )
#define MEDIA_PORT_STATIC_ASSERT( cond ) \
    typedef char MEDIA_PORT_ASSERT_CONCAT( media_port_static_assert_, __LINE__ )[ ( cond ) ? 1 : -1 ]

/*=============================================================================
 * NN MODEL SELECTION - SINGLE SOURCE OF TRUTH
 *
 * Swapping the NN model means keeping FOUR separate things in sync. They live in
 * three different files, they have drifted apart before, and each mismatch fails
 * in a way that does NOT point at the real cause:
 *
 *   (1) MEDIA_PORT_NN_MODEL      - here; the nnmodel_t passed to CMD_VIPNN_SET_MODEL
 *   (2) scenario.cmake           - which model_*.c decoder gets compiled
 *   (3) amebapro2_fwfs_nn_models.json -> FWFS.files - which .nb is packed into the
 *                                  image. NOTE: it is hardcoded there; the
 *                                  auto_model_cfg step does NOT rewrite it.
 *   (4) MEDIA_PORT_NN_WIDTH/HEIGHT - the ISP RGB channel that feeds the NPU
 *
 * Failure modes actually observed on this board:
 *   (3) wrong -> the .nb is absent, vipnn cannot open it, deploy fails.
 *   (2) not matching (1) -> the model loads but the decoder misparses its
 *       tensors and you get plausible-looking garbage boxes.
 *   (4) wrong -> "VOE cmd 0x206 ACK timeout" / "VOE_OPEN_CMD command fail" /
 *       "hal_video_open fail", i.e. NO VIDEO AT ALL, and the NN then looks
 *       broken because it never receives a frame. 576x320 is the only size
 *       this board has been observed to accept; 416x416 and 640x640 both
 *       killed VOE. This is why the guard below is a whitelist, not a formula
 *       - the constraint is empirical, so a new size MUST be tested on device.
 *
 * Also required: the model .nb must load through fwfs. A .nb whose size is an
 * exact multiple of 32 used to hard-fault on load (Usage Fault, UNALIGNED).
 * That was a real SDK defect - see the memcpy32 comment in
 * component/file_system/fwfs/fwfs.c. Preserve that patch across SDK updates;
 * without it, any 32-byte-aligned model brings the board down at deploy time.
 *
 * Define exactly ONE MEDIA_PORT_NN_MODEL_* below.
 *===========================================================================*/
#define MEDIA_PORT_NN_MODEL_SCRFD 1

#if defined( MEDIA_PORT_NN_MODEL_SCRFD ) && ( MEDIA_PORT_NN_MODEL_SCRFD == 1 )
    #include "model_scrfd.h"
    #define MEDIA_PORT_NN_MODEL      scrfd_fwfs      /* (1) */
    #define MEDIA_PORT_NN_DECODER_C  "model_scrfd.c" /* (2) keep scenario.cmake in step */
    #define MEDIA_PORT_NN_FWFS_ENTRY "scrfd320p"     /* (3) keep FWFS.files in step */
    #define MEDIA_PORT_NN_WIDTH      576             /* (4) tested-good ISP size */
    #define MEDIA_PORT_NN_HEIGHT     320
    /* SCRFD returns facedetect_res_t (= objdetect_res_t + landmark_t). It is the
       LARGER struct, so CMD_VIPNN_SET_RES_SIZE and the callback cast must both
       use this typedef, or pRes[i] strides wrong for every result after the
       first and every box but one lands in the wrong place. */
    typedef facedetect_res_t media_port_nn_res_t;
#else
    #error "Define exactly one MEDIA_PORT_NN_MODEL_* (see the block above)."
#endif

/* Empirical whitelist - see (4) above. Changing the NN channel size without
   testing reproduces the silent "no video" VOE failure, so this deliberately
   breaks the build instead of the board. */
MEDIA_PORT_STATIC_ASSERT( ( MEDIA_PORT_NN_WIDTH == 576 ) && ( MEDIA_PORT_NN_HEIGHT == 320 ) );

#define LIMIT(x, lower, upper) if(x<lower) x=lower; else if(x>upper) x=upper;
#endif /* ENABLE_NN_OBJECT_DETECTION */

#include "FreeRTOS.h"
#include "networking_utils.h"

#if METRIC_PRINT_ENABLED
#include "metric.h"
#endif

/* used to monitor skb resource */
extern int skbbuf_used_num;
extern int skbdata_used_num;
extern int max_local_skb_num;
extern int max_skb_buf_num;

#define MEDIA_PORT_SKB_BUFFER_THRESHOLD ( 64 )
#define MEDIA_PORT_WEBRTC_AUDIO_FRAME_SIZE ( 256 )

#define VIDEO_QCIF  0
#define VIDEO_CIF   1
#define VIDEO_WVGA  2
#define VIDEO_VGA   3
#define VIDEO_D1    4
#define VIDEO_HD    5
#define VIDEO_FHD   6
#define VIDEO_3M    7
#define VIDEO_5M    8
#define VIDEO_2K    9

/*****************************************************************************
* ISP channel : 0
* Video type  : H264/HEVC
*****************************************************************************/
#define MEDIA_PORT_V1_CHANNEL 0
#define MEDIA_PORT_V1_RESOLUTION VIDEO_HD
#define MEDIA_PORT_V1_FPS 30
#define MEDIA_PORT_V1_GOP 30
#define MEDIA_PORT_V1_BPS 512 * 1024
#define MEDIA_PORT_V1_RCMODE 2 // 1: CBR, 2: VBR

#if USE_VIDEO_CODEC_H265
#define MEDIA_PORT_VIDEO_TYPE VIDEO_HEVC
#define MEDIA_PORT_VIDEO_CODEC AV_CODEC_ID_H265
#else
#define MEDIA_PORT_VIDEO_TYPE VIDEO_H264
#define MEDIA_PORT_VIDEO_CODEC AV_CODEC_ID_H264
#endif

#if MEDIA_PORT_V1_RESOLUTION == VIDEO_VGA
#define MEDIA_PORT_V1_WIDTH 640
#define MEDIA_PORT_V1_HEIGHT 480
#elif MEDIA_PORT_V1_RESOLUTION == VIDEO_HD
#define MEDIA_PORT_V1_WIDTH 1280
#define MEDIA_PORT_V1_HEIGHT 720
#elif MEDIA_PORT_V1_RESOLUTION == VIDEO_FHD
#define MEDIA_PORT_V1_WIDTH 1920
#define MEDIA_PORT_V1_HEIGHT 1080
#endif

static mm_context_t * pVideoContext = NULL;
static mm_context_t * pAudioContext = NULL;
#if ( AUDIO_G711_MULAW || AUDIO_G711_ALAW )
#if MEDIA_PORT_ENABLE_AUDIO_RECV
static mm_context_t * pG711dContext = NULL;
#endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */
static mm_context_t * pG711eContext = NULL;
#endif /* ( AUDIO_G711_MULAW || AUDIO_G711_ALAW ) */
#if ( AUDIO_OPUS )
static mm_context_t * pOpuscContext = NULL;
#if MEDIA_PORT_ENABLE_AUDIO_RECV
static mm_context_t * pOpusdContext = NULL;
#endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */
#endif /* AUDIO_OPUS */
static mm_context_t * pWebrtcMmContext = NULL;

static mm_siso_t * pSisoAudioA1 = NULL;
static mm_miso_t * pMisoWebrtc = NULL;
#if MEDIA_PORT_ENABLE_AUDIO_RECV
static mm_siso_t * pSisoWebrtcA2 = NULL;
static mm_siso_t * pSisoAudioA2 = NULL;
#endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */

static video_params_t videoParams = {
    .stream_id = MEDIA_PORT_V1_CHANNEL,
    .type = MEDIA_PORT_VIDEO_TYPE,
    .resolution = MEDIA_PORT_V1_RESOLUTION,
    .width = MEDIA_PORT_V1_WIDTH,
    .height = MEDIA_PORT_V1_HEIGHT,
    .bps = MEDIA_PORT_V1_BPS,
    .fps = MEDIA_PORT_V1_FPS,
    .gop = MEDIA_PORT_V1_GOP,
    .rc_mode = MEDIA_PORT_V1_RCMODE,
    .use_static_addr = 1
};

#if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
/*****************************************************************************
* ISP channel : 4
* Video type  : RGB -> NPU (YOLO26 object detection)
*****************************************************************************/
#define MEDIA_PORT_NN_CHANNEL 4
/* MEDIA_PORT_NN_WIDTH / _HEIGHT come from the model-selection block near the top
   of this file - do not redefine them here, that is how they drifted out of sync
   with the packed model before. */
#define MEDIA_PORT_NN_FPS     15

#define MEDIA_PORT_SENSOR_MAX_WIDTH  1920
#define MEDIA_PORT_SENSOR_MAX_HEIGHT 1080

static video_params_t videoNnParams = {
    .stream_id = MEDIA_PORT_NN_CHANNEL,
    .type = VIDEO_RGB,
    .width = MEDIA_PORT_NN_WIDTH,
    .height = MEDIA_PORT_NN_HEIGHT,
    .fps = MEDIA_PORT_NN_FPS,
    .gop = MEDIA_PORT_NN_FPS,
    .direct_output = 0,
    .use_static_addr = 1,
    .use_roi = 1,
    .roi = {
        .xmin = 0,
        .ymin = 0,
        .xmax = MEDIA_PORT_SENSOR_MAX_WIDTH,
        .ymax = MEDIA_PORT_SENSOR_MAX_HEIGHT,
    }
};

static nn_data_param_t nnRoi = {
    .img = {
        .width = MEDIA_PORT_NN_WIDTH,
        .height = MEDIA_PORT_NN_HEIGHT,
        .rgb = 0,
        .roi = {
            .xmin = 0,
            .ymin = 0,
            .xmax = MEDIA_PORT_NN_WIDTH,
            .ymax = MEDIA_PORT_NN_HEIGHT,
        }
    },
    .codec_type = AV_CODEC_ID_RGB888
};

/* 0.35 chosen from a threshold sweep of the quantised model on the validation
   set: recall 0.148 -> 0.201 (30 -> 44 detections/image) with precision still
   0.84. uint8 quantisation makes the scores discrete, so 0.5 and 0.45 behave
   identically, as do 0.40 and 0.35. */
static float nnConfidenceThresh = 0.35;
static float nnNmsThresh = 0.3;

static mm_context_t * pVideoNnContext = NULL;
static mm_context_t * pVipnnContext = NULL;
static mm_siso_t * pSisoNnVipnn = NULL;

static void NnDetectionResultCallback( void * p,
                                       void * img_param )
{
    vipnn_out_buf_t * pOut = ( vipnn_out_buf_t * ) p;
    /* Result type comes from the model-selection block; see the note there on
       why using the wrong struct mis-strides every result after the first. */
    media_port_nn_res_t * pRes;
    int i;

    if( ( p == NULL ) || ( img_param == NULL ) )
    {
        return;
    }

    pRes = ( media_port_nn_res_t * ) &( pOut->res[ 0 ] );

    canvas_create_bitmap( MEDIA_PORT_V1_CHANNEL, 0, RTS_OSD2_BLK_FMT_1BPP );

    /* periodic decode diagnostic over the serial console (the OSD font table
       only contains digits and a handful of letters, so text there is useless) */
    {
        static int diagFrame = 0;
        if( ( diagFrame++ % 30 ) == 0 )
        {
            const char * dbg = "SCRFD 576x320";
            if( pOut->res_cnt > 0 )
            {
                int wmin = 9999, wmax = -9999, hmin = 9999, hmax = -9999, d;
                for( d = 0; d < pOut->res_cnt; d++ )
                {
                    int bw = ( int ) ( ( pRes[ d ].result[ 4 ] - pRes[ d ].result[ 2 ] ) * 1000 );
                    int bh = ( int ) ( ( pRes[ d ].result[ 5 ] - pRes[ d ].result[ 3 ] ) * 1000 );
                    if( bw < wmin ) { wmin = bw; }
                    if( bw > wmax ) { wmax = bw; }
                    if( bh < hmin ) { hmin = bh; }
                    if( bh > hmax ) { hmax = bh; }
                }
                printf( "[DIAG] %s n=%d w=%d-%d h=%d-%d\n\r",
                        dbg, ( int ) pOut->res_cnt, wmin, wmax, hmin, hmax );
            }
            else
            {
                printf( "[DIAG] %s n=0 (no detections)\n\r", dbg );
            }
        }
    }

    if( pOut->res_cnt > 0 )
    {
        LogInfo( ( "[SCRFD] face num = %d", pOut->res_cnt ) );

        for( i = 0; i < pOut->res_cnt; i++ )
        {
            int classId = ( int ) pRes[ i ].result[ 0 ];
            /* SCRFD is single-class: it writes result[0] == 0 for every face */
            if( classId == 0 )
            {
                int im_w = MEDIA_PORT_V1_WIDTH;
                int im_h = MEDIA_PORT_V1_HEIGHT;
                
                float ratio_w = (float)im_w / (float)MEDIA_PORT_NN_WIDTH;
                float ratio_h = (float)im_h / (float)MEDIA_PORT_NN_HEIGHT;
                int roi_h, roi_w, roi_x, roi_y;
                
                if ( videoNnParams.use_roi == 1 ) { //resize
                    roi_w = (int)((nnRoi.img.roi.xmax - nnRoi.img.roi.xmin) * ratio_w);
                    roi_h = (int)((nnRoi.img.roi.ymax - nnRoi.img.roi.ymin) * ratio_h);
                    roi_x = (int)(nnRoi.img.roi.xmin * ratio_w);
                    roi_y = (int)(nnRoi.img.roi.ymin * ratio_h);
                } else {  //crop
                    float ratio = ratio_h < ratio_w ? ratio_h : ratio_w;
                    roi_w = (int)((nnRoi.img.roi.xmax - nnRoi.img.roi.xmin) * ratio);
                    roi_h = (int)((nnRoi.img.roi.ymax - nnRoi.img.roi.ymin) * ratio);
                    roi_x = (int)(nnRoi.img.roi.xmin * ratio + (im_w - roi_w) / 2);
                    roi_y = (int)(nnRoi.img.roi.ymin * ratio + (im_h - roi_h) / 2);
                }

                int xmin = (int)(pRes[ i ].result[ 2 ] * roi_w) + roi_x;
                int ymin = (int)(pRes[ i ].result[ 3 ] * roi_h) + roi_y;
                int xmax = (int)(pRes[ i ].result[ 4 ] * roi_w) + roi_x;
                int ymax = (int)(pRes[ i ].result[ 5 ] * roi_h) + roi_y;

                LIMIT(xmin, 0, im_w);
                LIMIT(xmax, 0, im_w);
                LIMIT(ymin, 0, im_h);
                LIMIT(ymax, 0, im_h);

                LogInfo( ( "[SCRFD] %d: %s %d%% (%d,%d)-(%d,%d)",
                           i,
                           coco_name_get_by_id( classId ),
                           ( int ) ( pRes[ i ].result[ 1 ] * 100 ),
                           xmin, ymin, xmax, ymax ) );

                canvas_set_rect( MEDIA_PORT_V1_CHANNEL, 0, xmin, ymin, xmax, ymax, 3, COLOR_WHITE );
                char text_str[20];
                snprintf( text_str, sizeof(text_str), "%s %d", coco_name_get_by_id( classId ), ( int ) ( pRes[ i ].result[ 1 ] * 100 ) );
                canvas_set_text( MEDIA_PORT_V1_CHANNEL, 0, xmin, ymin - 32, text_str, COLOR_CYAN );
            }
        }
    }
    
    canvas_update( MEDIA_PORT_V1_CHANNEL, 0, 1 );
}
#endif /* ENABLE_NN_OBJECT_DETECTION */

#if !USE_DEFAULT_AUDIO_SET
static audio_params_t audioParams = {
    .sample_rate = ASR_8KHZ,
    .word_length = WL_16BIT,
    .mic_gain = MIC_0DB,
    .dmic_l_gain = DMIC_BOOST_24DB,
    .dmic_r_gain = DMIC_BOOST_24DB,
    .use_mic_type = USE_AUDIO_AMIC,
    .channel = 1,
    .mix_mode = 0,
    .enable_record = 0
};
#endif

#if ( AUDIO_G711_MULAW || AUDIO_G711_ALAW )
static g711_params_t g711eParams = {
    .codec_id = AV_CODEC_ID_PCMU,
    .buf_len = 2048,
    .mode = G711_ENCODE
};

#if MEDIA_PORT_ENABLE_AUDIO_RECV
static g711_params_t g711dParams = {
    .codec_id = AV_CODEC_ID_PCMU,
    .buf_len = 2048,
    .mode = G711_DECODE
};
#endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */
#endif /* ( AUDIO_G711_MULAW || AUDIO_G711_ALAW ) */

#if ( AUDIO_OPUS )
static opusc_params_t opuscParams = {
    .sample_rate = 8000, // 16000
    .channel = 1,
    .bit_length = 16,    // 16 recommand
    .complexity = 5,     // 0~10
    .bitrate = 25000,    // default 25000
    .use_framesize = 40, // 10 // needs to the same or bigger than AUDIO_DMA_PAGE_SIZE/(sample_rate/1000)/2 but less than 60
    .enable_vbr = 1,
    .vbr_constraint = 0,
    .packetLossPercentage = 0,
    .opus_application = OPUS_APPLICATION_AUDIO
};

#if MEDIA_PORT_ENABLE_AUDIO_RECV
static opusd_params_t opusdParams = {
    .sample_rate = 8000, // 16000
    .channel = 1,
    .bit_length = 16,         // 16 recommand
    .frame_size_in_msec = 10, // will not be uused
    .with_opus_enc = 1,       // enable semaphore if the application with opus encoder
    .opus_application = OPUS_APPLICATION_AUDIO
};
#endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */
#endif /* AUDIO_OPUS */

static int HandleModuleFrameHook( void * p,
                                  void * input,
                                  void * output );
static int ControlModuleHook( void * p,
                              int cmd,
                              int arg );
static void * DestroyModuleHook( void * p );
static void * CreateModuleHook( void * parent );
static void * NewModuleItemHook( void * p );
static void * DeleteModuleItemHook( void * p,
                                    void * d );

mm_module_t webrtcMmModule = {
    .create = CreateModuleHook,
    .destroy = DestroyModuleHook,
    .control = ControlModuleHook,
    .handle = HandleModuleFrameHook,

    .new_item = NewModuleItemHook,
    .del_item = DeleteModuleItemHook,

    .output_type = MM_TYPE_ASINK, // output for audio sink
    .module_type = MM_TYPE_AVSINK, // module type is video algorithm
    .name = "KVS_WebRTC"
};

static int HandleModuleFrameHook( void * p,
                                  void * input,
                                  void * output )
{
    int ret = 0;
    MediaModuleContext_t * pCtx = ( MediaModuleContext_t * )p;
    MediaFrame_t frame;
    mm_queue_item_t * pInputItem = ( mm_queue_item_t * )input;

    ( void ) output;

    if( pCtx->mediaStart != 0 )
    {
        do
        {
            /* Set SKB buffer threshold to manage memory allocation. Reference:
             * https://github.com/Freertos-kvs-LTS/freertos-kvs-LTS/blob/bd0702130e0b8dfa386e011644ce1bc7e0d7fd09/component/example/kvs_webrtc_mmf/webrtc_app_src/AppMediaSrc_AmebaPro2.c#L86-L88 */
            if( ( skbdata_used_num > ( max_skb_buf_num - MEDIA_PORT_SKB_BUFFER_THRESHOLD ) ) ||
                ( skbbuf_used_num > ( max_local_skb_num - MEDIA_PORT_SKB_BUFFER_THRESHOLD ) ) )
            {
                ret = -1;
                break; //skip this frame and wait for skb resource release.
            }

            frame.size = pInputItem->size;
            frame.pData = ( uint8_t * ) pvPortMalloc( frame.size );
            if( !frame.pData )
            {
                LogWarn( ( "Fail to allocate memory for webrtc media frame, size: %lu", frame.size ) );
                ret = -1;
                break;
            }

            memcpy( frame.pData,
                    ( uint8_t * )pInputItem->data_addr,
                    frame.size );
            frame.freeData = 1;
            frame.timestampUs = NetworkingUtils_GetCurrentTimeUs( &pInputItem->timestamp );

            if( ( pInputItem->type == AV_CODEC_ID_H264 ) || ( pInputItem->type == AV_CODEC_ID_H265 ) )
            {
                if( pCtx->onVideoFrameReadyToSendFunc )
                {
                    frame.trackKind = TRANSCEIVER_TRACK_KIND_VIDEO;
                    ( void ) pCtx->onVideoFrameReadyToSendFunc( pCtx->pOnVideoFrameReadyToSendCustomContext,
                                                                &frame );
                }
                else
                {
                    LogError( ( "No available ready to send callback function pointer for video." ) );
                    vPortFree( frame.pData );
                    ret = -1;
                }
            }
            else if( ( pInputItem->type == AV_CODEC_ID_OPUS ) ||
                     ( pInputItem->type == AV_CODEC_ID_PCMU ) )
            {
                if( pCtx->onAudioFrameReadyToSendFunc )
                {
                    frame.trackKind = TRANSCEIVER_TRACK_KIND_AUDIO;
                    ( void ) pCtx->onAudioFrameReadyToSendFunc( pCtx->pOnAudioFrameReadyToSendCustomContext,
                                                                &frame );
                }
                else
                {
                    LogError( ( "No available ready to send callback function pointer for audio." ) );
                    vPortFree( frame.pData );
                    ret = -1;
                }
            }
            else
            {
                LogWarn( ( "Input type cannot be handled: %ld", pInputItem->type ) );
                vPortFree( frame.pData );
                ret = -1;
            }
        } while( pdFALSE );
    }

    return ret;
}

static int ControlModuleHook( void * p,
                              int cmd,
                              int arg )
{
    MediaModuleContext_t * pCtx = ( MediaModuleContext_t * )p;

    switch( cmd )
    {
        case CMD_KVS_WEBRTC_START:
            /* If loopback is enabled, we don't need the camera to provide frames.
             * Instead, we loopback the received frames. */
            #ifdef ENABLE_STREAMING_LOOPBACK
            pCtx->mediaStart = 0;
            #else
            pCtx->mediaStart = 1;
            #endif
            break;
        case CMD_KVS_WEBRTC_STOP:
            pCtx->mediaStart = 0;
            break;
        case CMD_KVS_WEBRTC_REG_VIDEO_SEND_CALLBACK:
            pCtx->onVideoFrameReadyToSendFunc = ( OnFrameReadyToSend_t ) arg;
            break;
        case CMD_KVS_WEBRTC_REG_VIDEO_SEND_CALLBACK_CUSTOM_CONTEXT:
            pCtx->pOnVideoFrameReadyToSendCustomContext = ( void * ) arg;
            break;
        case CMD_KVS_WEBRTC_REG_AUDIO_SEND_CALLBACK:
            pCtx->onAudioFrameReadyToSendFunc = ( OnFrameReadyToSend_t ) arg;
            break;
        case CMD_KVS_WEBRTC_REG_AUDIO_SEND_CALLBACK_CUSTOM_CONTEXT:
            pCtx->pOnAudioFrameReadyToSendCustomContext = ( void * ) arg;
            break;
        default:
            LogWarn( ( "Unknown module command: %d", cmd ) );
            break;
    }
    return 0;
}

static void * DestroyModuleHook( void * p )
{
    MediaModuleContext_t * ctx = ( MediaModuleContext_t * )p;
    if( ctx )
    {
        vPortFree( ctx );
    }
    return NULL;
}

static void * CreateModuleHook( void * parent )
{
    MediaModuleContext_t * ctx = pvPortMalloc( sizeof( MediaModuleContext_t ) );

    if( ctx )
    {
        memset( ctx,
                0,
                sizeof( MediaModuleContext_t ) );
        ctx->pParent = parent;
    }

    return ctx;
}

static void * NewModuleItemHook( void * p )
{
    void * pBuffer = pvPortMalloc( MEDIA_PORT_WEBRTC_AUDIO_FRAME_SIZE * 2 );

    ( void ) p;

    if( pBuffer == NULL )
    {
        LogError( ( "Fail to allocate buffer for module item." ) );
    }

    return pBuffer;
}

static void * DeleteModuleItemHook( void * p,
                                    void * d )
{
    ( void ) p;

    if( d != NULL )
    {
        vPortFree( d );
    }

    return NULL;
}

void AppMediaSourcePort_Destroy( void )
{
    #if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
    osd_render_task_stop();
    osd_render_dev_deinit_all();
    #endif
    // Pause Linkers
    siso_pause( pSisoAudioA1 );
    miso_pause( pMisoWebrtc,
                MM_OUTPUT );
    #if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
    siso_pause( pSisoNnVipnn );
    #endif
    #if MEDIA_PORT_ENABLE_AUDIO_RECV
    siso_pause( pSisoWebrtcA2 );
    siso_pause( pSisoAudioA2 );
    #endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */

    // Stop modules
    mm_module_ctrl( pWebrtcMmContext,
                    CMD_KVS_WEBRTC_STOP,
                    0 );
    mm_module_ctrl( pVideoContext,
                    CMD_VIDEO_STREAM_STOP,
                    MEDIA_PORT_V1_CHANNEL );
    #if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
    mm_module_ctrl( pVideoNnContext,
                    CMD_VIDEO_STREAM_STOP,
                    MEDIA_PORT_NN_CHANNEL );
    #endif
    mm_module_ctrl( pAudioContext,
                    CMD_AUDIO_SET_TRX,
                    0 );

    // Delete linkers
    pSisoAudioA1 = siso_delete( pSisoAudioA1 );
    pMisoWebrtc = miso_delete( pMisoWebrtc );
    #if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
    pSisoNnVipnn = siso_delete( pSisoNnVipnn );
    #endif
    #if MEDIA_PORT_ENABLE_AUDIO_RECV
    pSisoWebrtcA2 = siso_delete( pSisoWebrtcA2 );
    pSisoAudioA2 = siso_delete( pSisoAudioA2 );
    #endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */

    // Close modules
    pWebrtcMmContext = mm_module_close( pWebrtcMmContext );
    pVideoContext = mm_module_close( pVideoContext );
    #if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
    pVipnnContext = mm_module_close( pVipnnContext );
    pVideoNnContext = mm_module_close( pVideoNnContext );
    #endif
    pAudioContext = mm_module_close( pAudioContext );
    #if ( AUDIO_G711_MULAW || AUDIO_G711_ALAW )
    pG711eContext = mm_module_close( pG711eContext );
    #if MEDIA_PORT_ENABLE_AUDIO_RECV
    pG711dContext = mm_module_close( pG711dContext );
    #endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */
    #elif AUDIO_OPUS
    pOpuscContext = mm_module_close( pOpuscContext );
    #if MEDIA_PORT_ENABLE_AUDIO_RECV
    pOpusdContext = mm_module_close( pOpusdContext );
    #endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */
    #endif

    // Video Deinit
    video_deinit();
}

int32_t AppMediaSourcePort_Init( void )
{
    int32_t ret = 0;
    int voe_heap_size;

    pWebrtcMmContext = mm_module_open( &webrtcMmModule );
    if( pWebrtcMmContext )
    {
        mm_module_ctrl( pWebrtcMmContext,
                        MM_CMD_SET_QUEUE_LEN,
                        6 );
        mm_module_ctrl( pWebrtcMmContext,
                        MM_CMD_INIT_QUEUE_ITEMS,
                        MMQI_FLAG_STATIC );
        mm_module_ctrl( pWebrtcMmContext, CMD_KVS_WEBRTC_SET_APPLY, 0 );
    }
    else
    {
        LogError( ( "KVS open fail" ) );
        ret = -1;
    }

    if( ret == 0 )
    {
        #if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
        voe_heap_size = video_voe_presetting( 1, MEDIA_PORT_V1_WIDTH, MEDIA_PORT_V1_HEIGHT, MEDIA_PORT_V1_BPS, 0,
                                              0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0,
                                              1, MEDIA_PORT_NN_WIDTH, MEDIA_PORT_NN_HEIGHT );
        #else
        voe_heap_size = video_voe_presetting( 1, MEDIA_PORT_V1_WIDTH, MEDIA_PORT_V1_HEIGHT, MEDIA_PORT_V1_BPS, 0,
                                              0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0,
                                              0, 0, 0 );
        #endif /* ENABLE_NN_OBJECT_DETECTION */
        ( void ) voe_heap_size;
        LogInfo( ( "voe heap size = %d", voe_heap_size ) );
    }

    if( ret == 0 )
    {
        pVideoContext = mm_module_open( &video_module );
        if( pVideoContext )
        {
            mm_module_ctrl( pVideoContext,
                            CMD_VIDEO_SET_PARAMS,
                            ( int )&videoParams );
            mm_module_ctrl( pVideoContext,
                            MM_CMD_SET_QUEUE_LEN,
                            MEDIA_PORT_V1_FPS * 3 );
            mm_module_ctrl( pVideoContext,
                            MM_CMD_INIT_QUEUE_ITEMS,
                            MMQI_FLAG_DYNAMIC );

            #if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
            int ch_enable[3] = {1, 0, 0};
            int char_resize_w[3] = {16, 0, 0}, char_resize_h[3] = {32, 0, 0};
            int ch_width[3] = {MEDIA_PORT_V1_WIDTH, 0, 0}, ch_height[3] = {MEDIA_PORT_V1_HEIGHT, 0, 0};
            osd_render_dev_init(ch_enable, char_resize_w, char_resize_h);
            osd_render_task_start(ch_enable, ch_width, ch_height);
            #endif

            mm_module_ctrl( pVideoContext,
                            CMD_VIDEO_APPLY,
                            MEDIA_PORT_V1_CHANNEL ); // start channel 0
        }
        else
        {
            LogError( ( "video open fail" ) );
            ret = -1;
        }
    }

    #if defined( ENABLE_NN_OBJECT_DETECTION ) && ( ENABLE_NN_OBJECT_DETECTION == 1 )
    if( ret == 0 )
    {
        pVideoNnContext = mm_module_open( &video_module );
        if( pVideoNnContext )
        {
            mm_module_ctrl( pVideoNnContext,
                            CMD_VIDEO_SET_PARAMS,
                            ( int ) &videoNnParams );
            mm_module_ctrl( pVideoNnContext,
                            MM_CMD_SET_QUEUE_LEN,
                            2 );
            mm_module_ctrl( pVideoNnContext,
                            MM_CMD_INIT_QUEUE_ITEMS,
                            MMQI_FLAG_DYNAMIC );
        }
        else
        {
            LogError( ( "NN RGB video open fail" ) );
            ret = -1;
        }
    }

    if( ret == 0 )
    {
        pVipnnContext = mm_module_open( &vipnn_module );
        if( pVipnnContext )
        {
            mm_module_ctrl( pVipnnContext,
                            CMD_VIPNN_SET_MODEL,
                            ( int ) &MEDIA_PORT_NN_MODEL );
            mm_module_ctrl( pVipnnContext,
                            CMD_VIPNN_SET_IN_PARAMS,
                            ( int ) &nnRoi );
            mm_module_ctrl( pVipnnContext,
                            CMD_VIPNN_SET_DISPPOST,
                            ( int ) NnDetectionResultCallback );
            mm_module_ctrl( pVipnnContext,
                            CMD_VIPNN_SET_CONFIDENCE_THRES,
                            ( int ) &nnConfidenceThresh );
            mm_module_ctrl( pVipnnContext,
                            CMD_VIPNN_SET_NMS_THRES,
                            ( int ) &nnNmsThresh );
            mm_module_ctrl( pVipnnContext,
                            CMD_VIPNN_SET_RES_SIZE,
                            sizeof( media_port_nn_res_t ) );
            mm_module_ctrl( pVipnnContext,
                            CMD_VIPNN_SET_RES_MAX_CNT,
                            MAX_DETECT_OBJ_NUM );
            mm_module_ctrl( pVipnnContext,
                            CMD_VIPNN_APPLY,
                            0 );
        }
        else
        {
            LogError( ( "VIPNN open fail" ) );
            ret = -1;
        }
    }

    if( ret == 0 )
    {
        pSisoNnVipnn = siso_create();
        if( pSisoNnVipnn )
        {
            #if defined( configENABLE_TRUSTZONE ) && ( configENABLE_TRUSTZONE == 1 )
            siso_ctrl( pSisoNnVipnn,
                       MMIC_CMD_SET_SECURE_CONTEXT,
                       1,
                       0 );
            #endif
            siso_ctrl( pSisoNnVipnn,
                       MMIC_CMD_ADD_INPUT,
                       ( uint32_t ) pVideoNnContext,
                       0 );
            siso_ctrl( pSisoNnVipnn,
                       MMIC_CMD_SET_STACKSIZE,
                       ( uint32_t ) 1024 * 64,
                       0 );
            siso_ctrl( pSisoNnVipnn,
                       MMIC_CMD_SET_TASKPRIORITY,
                       3,
                       0 );
            siso_ctrl( pSisoNnVipnn,
                       MMIC_CMD_ADD_OUTPUT,
                       ( uint32_t ) pVipnnContext,
                       0 );
            siso_start( pSisoNnVipnn );

            mm_module_ctrl( pVideoNnContext,
                            CMD_VIDEO_APPLY,
                            MEDIA_PORT_NN_CHANNEL );
            mm_module_ctrl( pVideoNnContext,
                            CMD_VIDEO_YUV,
                            2 );
        }
        else
        {
            LogError( ( "pSisoNnVipnn open fail" ) );
            ret = -1;
        }
    }
    #endif /* ENABLE_NN_OBJECT_DETECTION */

    if( ret == 0 )
    {
        pAudioContext = mm_module_open( &audio_module );
        if( pAudioContext )
        {
            #if !USE_DEFAULT_AUDIO_SET
            mm_module_ctrl( pAudioContext,
                            CMD_AUDIO_SET_PARAMS,
                            ( int )&audioParams );
            #endif
            mm_module_ctrl( pAudioContext,
                            MM_CMD_SET_QUEUE_LEN,
                            6 );
            mm_module_ctrl( pAudioContext,
                            MM_CMD_INIT_QUEUE_ITEMS,
                            MMQI_FLAG_STATIC );
            mm_module_ctrl( pAudioContext,
                            CMD_AUDIO_APPLY,
                            0 );
        }
        else
        {
            LogError( ( "Audio open fail" ) );
            ret = -1;
        }
    }

    if( ret == 0 )
    {
        #if ( AUDIO_G711_MULAW || AUDIO_G711_ALAW )
        pG711eContext = mm_module_open( &g711_module );
        if( pG711eContext )
        {
            mm_module_ctrl( pG711eContext,
                            CMD_G711_SET_PARAMS,
                            ( int )&g711eParams );
            mm_module_ctrl( pG711eContext,
                            MM_CMD_SET_QUEUE_LEN,
                            6 );
            mm_module_ctrl( pG711eContext,
                            MM_CMD_INIT_QUEUE_ITEMS,
                            MMQI_FLAG_STATIC );
            mm_module_ctrl( pG711eContext,
                            CMD_G711_APPLY,
                            0 );
        }
        else
        {
            LogError( ( "G711 open fail" ) );
            ret = -1;
        }
        #elif AUDIO_OPUS
        pOpuscContext = mm_module_open( &opusc_module );
        if( pOpuscContext )
        {
            mm_module_ctrl( pOpuscContext,
                            CMD_OPUSC_SET_PARAMS,
                            ( int )&opuscParams );
            mm_module_ctrl( pOpuscContext,
                            MM_CMD_SET_QUEUE_LEN,
                            6 );
            mm_module_ctrl( pOpuscContext,
                            MM_CMD_INIT_QUEUE_ITEMS,
                            MMQI_FLAG_STATIC );
            mm_module_ctrl( pOpuscContext,
                            CMD_OPUSC_APPLY,
                            0 );
        }
        else
        {
            LogError( ( "OPUSC open fail" ) );
            ret = -1;
        }
        #endif
    }

    if( ret == 0 )
    {
        pSisoAudioA1 = siso_create();
        if( pSisoAudioA1 )
        {
            siso_ctrl( pSisoAudioA1,
                       MMIC_CMD_ADD_INPUT,
                       ( uint32_t )pAudioContext,
                       0 );
            #if ( AUDIO_G711_MULAW || AUDIO_G711_ALAW )
            siso_ctrl( pSisoAudioA1,
                       MMIC_CMD_ADD_OUTPUT,
                       ( uint32_t )pG711eContext,
                       0 );
            #elif AUDIO_OPUS
            siso_ctrl( pSisoAudioA1,
                       MMIC_CMD_ADD_OUTPUT,
                       ( uint32_t )pOpuscContext,
                       0 );
            siso_ctrl( pSisoAudioA1,
                       MMIC_CMD_SET_STACKSIZE,
                       24 * 1024,
                       0 );
            #endif
            siso_start( pSisoAudioA1 );
        }
        else
        {
            LogError( ( "pSisoAudioA1 open fail" ) );
            ret = -1;
        }
    }

    if( ret == 0 )
    {
        pMisoWebrtc = miso_create();
        if( pMisoWebrtc )
        {
            #if defined( configENABLE_TRUSTZONE ) && ( configENABLE_TRUSTZONE == 1 )
            miso_ctrl( pMisoWebrtc,
                       MMIC_CMD_SET_SECURE_CONTEXT,
                       1,
                       0 );
            #endif
            miso_ctrl( pMisoWebrtc,
                       MMIC_CMD_ADD_INPUT0,
                       ( uint32_t )pVideoContext,
                       0 );
            #if ( AUDIO_G711_MULAW || AUDIO_G711_ALAW )
            miso_ctrl( pMisoWebrtc,
                       MMIC_CMD_ADD_INPUT1,
                       ( uint32_t )pG711eContext,
                       0 );
            #elif AUDIO_OPUS
            miso_ctrl( pMisoWebrtc,
                       MMIC_CMD_ADD_INPUT1,
                       ( uint32_t )pOpuscContext,
                       0 );
            #endif
            miso_ctrl( pMisoWebrtc,
                       MMIC_CMD_ADD_OUTPUT,
                       ( uint32_t )pWebrtcMmContext,
                       0 );
            miso_start( pMisoWebrtc );
        }
        else
        {
            LogError( ( "pMisoWebrtc open fail" ) );
            ret = -1;
        }
    }

    #if MEDIA_PORT_ENABLE_AUDIO_RECV
    if( ret == 0 )
    {
        #if ( AUDIO_G711_MULAW || AUDIO_G711_ALAW )
        pG711dContext = mm_module_open( &g711_module );
        if( pG711dContext )
        {
            mm_module_ctrl( pG711dContext,
                            CMD_G711_SET_PARAMS,
                            ( int )&g711dParams );
            mm_module_ctrl( pG711dContext,
                            MM_CMD_SET_QUEUE_LEN,
                            6 );
            mm_module_ctrl( pG711dContext,
                            MM_CMD_INIT_QUEUE_ITEMS,
                            MMQI_FLAG_STATIC );
            mm_module_ctrl( pG711dContext,
                            CMD_G711_APPLY,
                            0 );
        }
        else
        {
            LogError( ( "G711 open fail" ) );
            ret = -1;
        }
        #elif AUDIO_OPUS
        pOpusdContext = mm_module_open( &opusd_module );
        if( pOpusdContext )
        {
            mm_module_ctrl( pOpusdContext,
                            CMD_OPUSD_SET_PARAMS,
                            ( int )&opusdParams );
            mm_module_ctrl( pOpusdContext,
                            MM_CMD_SET_QUEUE_LEN,
                            6 );
            mm_module_ctrl( pOpusdContext,
                            MM_CMD_INIT_QUEUE_ITEMS,
                            MMQI_FLAG_STATIC );
            mm_module_ctrl( pOpusdContext,
                            CMD_OPUSD_APPLY,
                            0 );
        }
        else
        {
            LogError( ( "OPUSD open fail" ) );
            ret = -1;
        }
        #endif
    }

    if( ret == 0 )
    {
        pSisoWebrtcA2 = siso_create();
        if( pSisoWebrtcA2 )
        {
            siso_ctrl( pSisoWebrtcA2,
                       MMIC_CMD_ADD_INPUT,
                       ( uint32_t )pWebrtcMmContext,
                       0 );
            #if ( AUDIO_G711_MULAW || AUDIO_G711_ALAW )
            siso_ctrl( pSisoWebrtcA2,
                       MMIC_CMD_ADD_OUTPUT,
                       ( uint32_t )pG711dContext,
                       0 );
            #elif AUDIO_OPUS
            siso_ctrl( pSisoWebrtcA2,
                       MMIC_CMD_ADD_OUTPUT,
                       ( uint32_t )pOpusdContext,
                       0 );
            siso_ctrl( pSisoWebrtcA2,
                       MMIC_CMD_SET_STACKSIZE,
                       24 * 1024,
                       0 );
            #endif
            siso_start( pSisoWebrtcA2 );
        }
        else
        {
            LogError( ( "pSisoWebrtcA2 open fail" ) );
            ret = -1;
        }
    }

    if( ret == 0 )
    {
        pSisoAudioA2 = siso_create();
        if( pSisoAudioA2 )
        {
            #if ( AUDIO_G711_MULAW || AUDIO_G711_ALAW )
            siso_ctrl( pSisoAudioA2,
                       MMIC_CMD_ADD_INPUT,
                       ( uint32_t )pG711dContext,
                       0 );
            #elif AUDIO_OPUS
            siso_ctrl( pSisoAudioA2,
                       MMIC_CMD_ADD_INPUT,
                       ( uint32_t )pOpusdContext,
                       0 );
            #endif
            siso_ctrl( pSisoAudioA2,
                       MMIC_CMD_ADD_OUTPUT,
                       ( uint32_t )pAudioContext,
                       0 );
            siso_start( pSisoAudioA2 );
        }
        else
        {
            LogError( ( "pSisoAudioA2 open fail" ) );
            ret = -1;
        }
    }
    #endif /* MEDIA_PORT_ENABLE_AUDIO_RECV */

    return ret;
}

int32_t AppMediaSourcePort_Start( OnFrameReadyToSend_t onVideoFrameReadyToSendFunc,
                                  void * pOnVideoFrameReadyToSendCustomContext,
                                  OnFrameReadyToSend_t onAudioFrameReadyToSendFunc,
                                  void * pOnAudioFrameReadyToSendCustomContext )
{
    int32_t ret = 0;

    #if METRIC_PRINT_ENABLED
    Metric_StartEvent( METRIC_EVENT_MEDIA_PORT_START );
    #endif
    mm_module_ctrl( pWebrtcMmContext,
                    CMD_KVS_WEBRTC_REG_VIDEO_SEND_CALLBACK,
                    ( int ) onVideoFrameReadyToSendFunc );
    mm_module_ctrl( pWebrtcMmContext,
                    CMD_KVS_WEBRTC_REG_VIDEO_SEND_CALLBACK_CUSTOM_CONTEXT,
                    ( int ) pOnVideoFrameReadyToSendCustomContext );
    mm_module_ctrl( pWebrtcMmContext,
                    CMD_KVS_WEBRTC_REG_AUDIO_SEND_CALLBACK,
                    ( int ) onAudioFrameReadyToSendFunc );
    mm_module_ctrl( pWebrtcMmContext,
                    CMD_KVS_WEBRTC_REG_AUDIO_SEND_CALLBACK_CUSTOM_CONTEXT,
                    ( int ) pOnAudioFrameReadyToSendCustomContext );
    mm_module_ctrl( pWebrtcMmContext,
                    CMD_KVS_WEBRTC_START,
                    0 );
    #if METRIC_PRINT_ENABLED
    Metric_EndEvent( METRIC_EVENT_MEDIA_PORT_START );
    #endif

    return ret;
}

void AppMediaSourcePort_Stop( void )
{
    #if METRIC_PRINT_ENABLED
    Metric_StartEvent( METRIC_EVENT_MEDIA_PORT_STOP );
    #endif
    mm_module_ctrl( pWebrtcMmContext,
                    CMD_KVS_WEBRTC_STOP,
                    0 );
    #if METRIC_PRINT_ENABLED
    Metric_EndEvent( METRIC_EVENT_MEDIA_PORT_STOP );
    #endif
}

void AppMediaSourcePort_PlayAudioFrame( MediaFrame_t * pFrame )
{
    uint8_t skipProcess = 0U;
    mm_queue_item_t *output_item;

    if( pFrame == NULL )
    {
        LogError( ( "Invalid input, pFrame: %p", pFrame ) );
        skipProcess = 1U;
    }
    else if( pFrame->trackKind != TRANSCEIVER_TRACK_KIND_AUDIO )
    {
        LogError( ( "Dropping non-audio frame, track kind: %d", pFrame->trackKind ) );
        skipProcess = 1U;
    }
    else
    {
        /* Empty else marker. */
    }

    if( skipProcess == 0U )
    {
        LogDebug( ( "Playing audio frame with length: %lu", pFrame->size ) );

        if( xQueueReceive( pWebrtcMmContext->output_recycle, &output_item, 0xFFFFFFFF) == pdTRUE )
        {
            memcpy( ( void * )output_item->data_addr, ( void * ) pFrame->pData, pFrame->size );

            #if AUDIO_G711_MULAW
                output_item->type = AV_CODEC_ID_PCMU;
            #elif AUDIO_G711_ALAW
                output_item->type = AV_CODEC_ID_PCMA;
            #elif AUDIO_OPUS
                output_item->type = AV_CODEC_ID_OPUS;
            #else
                #error "Audio codec is not configured."
            #endif

            output_item->size = pFrame->size;
            output_item->timestamp = pFrame->timestampUs;
            xQueueSend( pWebrtcMmContext->output_ready, (void *)&output_item, 0xFFFFFFFF );
        }
        else
        {
            LogWarn( ( "No free output queue item for frame type: %d", AV_CODEC_ID_OPUS ) );
        }
    }
}
