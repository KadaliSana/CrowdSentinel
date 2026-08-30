# YOLO26 → AMB Pro2 (.nb) Conversion Guide

This document covers the end-to-end process of converting a trained YOLO26 PyTorch model into a Neural Binary (.nb) file for deployment on the Realtek Ameba Pro 2 (RTL8735B) NPU.

## Overview

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  best.pt │───▶│ best.onnx│───▶│ .json/.data│───▶│ .quantize│───▶│ yolo26.nb│
│ (PyTorch)│    │  (ONNX)  │    │  (Acuity)  │    │  (uint8) │    │  (NPU)   │
└──────────┘    └──────────┘    └───────────┘    └──────────┘    └──────────┘
  export_to_onnx.py              convert_to_nb.sh (Steps 1-3)
```

## Prerequisites

- **Python 3.8+** with `ultralytics` and `onnx` packages
- **Acuity Toolchain** — either:
  - Native install with `ACUITY_PATH` set, or
  - Docker: `ameba-ai-offline-toolkit/Docker/Linux/`
- Trained `.pt` weights (default: `models/weights/best.pt`)

## Step 1: Export PyTorch → ONNX

```bash
cd Yan/models/conversion

# Basic export (uses defaults: best.pt, 416x416, opset=12)
python3 export_to_onnx.py

# With calibration dataset generation
python3 export_to_onnx.py \
    --weights "../models/weights/best.pt" \
    --imgsz 416 \
    --gen-dataset \
    --dataset-dir "../dataset/valid/images" \
    --output-dir ./conversion_workspace

# Custom model name
python3 export_to_onnx.py --weights /path/to/model.pt --model-name yolo26_crowd
```

**Output**: `conversion_workspace/best.onnx` + `best_inputmeta.yml` + `dataset.txt`

### Key Export Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `opset` | 12 | AMB Pro2 NPU maximum supported |
| `simplify` | True | Removes ops the NPU can't handle |
| `dynamic` | False | NPU requires fixed input size |
| `imgsz` | 416 | Must match `MEDIA_PORT_NN_WIDTH`/`_HEIGHT` in `ameba_pro2_media_port.c` (see Step 5) |

## Step 2: ONNX → .nb (Acuity Pipeline)

### Option A: Using the conversion script (recommended)

```bash
cd Yan/models/conversion

# Default uint8 quantization
./convert_to_nb.sh --name best --workspace ./conversion_workspace

# With custom settings
./convert_to_nb.sh \
    --name best \
    --workspace ./conversion_workspace \
    --qtype uint8 \
    --imgsz 416
```

### Option B: Using Docker

```bash
# 1. Start the Acuity Docker container
cd "models/acuity/Docker/Linux"
docker run -it --rm -v $(pwd)/acuity_examples_c901149:/workspace acuity-toolkit

# 2. Inside the container, copy your ONNX and run conversion
cp /path/to/best.onnx /workspace/
cp /path/to/dataset.txt /workspace/
cp /path/to/best_inputmeta.yml /workspace/
cd /workspace
./convert.sh
```

### Option C: Manual steps

```bash
# 1. Import
pegasus import onnx \
    --model best.onnx \
    --output-model best.json \
    --output-data best.data

# 2. Quantize
pegasus quantize \
    --model best.json \
    --model-data best.data \
    --device CPU \
    --with-input-meta best_inputmeta.yml \
    --compute-entropy \
    --rebuild \
    --model-quantize best_uint8.quantize \
    --quantizer asymmetric_affine \
    --qtype uint8

# 3. Export
pegasus export ovxlib \
    --model best.json \
    --model-data best.data \
    --dtype quantized \
    --model-quantize best_uint8.quantize \
    --target-ide-project 'linux64' \
    --with-input-meta best_inputmeta.yml \
    --output-path ./output/best_uint8 \
    --optimize 'VIP8000NANONI_PID0XAD' \
    --pack-nbg-unify \
    --viv-sdk '/opt/acuity/Vivante_IDE/VivanteIDE5.8.1.1/cmdtools'
```

**Output**: `output/yolo26.nb`

## Step 3: Review the inputmeta.yml

After `pegasus import`, the auto-generated `_inputmeta.yml` may need the `lid` field updated. Check the generated `best.json` for the actual input layer ID:

```bash
# Find the input layer name
grep -o '"lid":"[^"]*"' best.json | head -1
```

Update `best_inputmeta.yml` to match:
```yaml
ports:
- lid: images_392    # ← must match the actual layer ID from best.json
```

## Step 4: Flash to Device

### Using SD Card
1. Copy `yolo26.nb` to the SD card under `NN_MDL/yolo26.nb`
2. The firmware will load it via `yolo26_get_network_filename_init()` which returns `"NN_MDL/yolo26.nb"`

### Using FWFS (Firmware File System)
1. Put the `.nb` in the SDK's model directory,
   `src/firmware_webrtc/libraries/ambpro2_sdk/project/realtek_amebapro2_v0_example/src/test_model/model_nb/`
   (this is `NN_MODEL_PATH` in `GCC-RELEASE/CMakeLists.txt`).
2. Name it in `FWFS.files` in `amebapro2_fwfs_nn_models.json`. `files` is a list of *keys*, each
   resolved by a sibling entry; the build ships `"scrfd320p"` and a `"yolo26"` entry already exists:
   ```json
   {
     "FWFS": { "files": [ "yolo26" ] },
     "yolo26": { "name": "yolo26.nb", "source": "binary", "file": "yolo26.nb" }
   }
   ```
   Edit the copy under
   `libraries/ambpro2_sdk/project/realtek_amebapro2_v0_example/GCC-RELEASE/mp/` — the build copies it
   over `build/amebapro2_fwfs_nn_models.json` on every run, so editing `build/` is lost. The
   `auto_model_cfg` step does *not* rewrite `FWFS.files`.
3. Rebuild with the `flash_nn` target (`flash` alone leaves the old model on the NN partition) and flash.

## Step 5: Firmware Integration

There is one firmware application — the KVS WebRTC app at
`src/firmware_webrtc/project/realtek_amebapro2_webrtc_application` — and it currently selects
**SCRFD**, not yolo26. The yolo26 decoder is still present in the vendored SDK:

- **Model code**: `src/firmware_webrtc/libraries/ambpro2_sdk/project/realtek_amebapro2_v0_example/src/test_model/model_yolo26.c`
- **Header**: the matching `model_yolo26.h` beside it

Deploying a yolo26 `.nb` therefore means switching four things that must agree. They are documented
together in the **NN MODEL SELECTION** block at the top of
`src/firmware_webrtc/examples/app_media_source/port/ameba_pro2/ameba_pro2_media_port.c`, which is the
single source of truth: (1) `MEDIA_PORT_NN_MODEL` there, (2) the `model_*.c` compiled by
`scenario.cmake`, (3) the `FWFS.files` entry (Step 4), and (4) `MEDIA_PORT_NN_WIDTH`/`_HEIGHT`.

**(4) is the trap:** the ISP on this board has only been observed to accept **576x320** for the NN
channel — 416x416 and 640x640 both kill VOE and produce no video at all — and the block enforces
that with a compile-time whitelist assert. `export_to_onnx.py` emits square inputs only, so a yolo26
export at the default 416 cannot currently satisfy both (4) and the "`--imgsz` must equal the NN
channel size" rule above.

The model registers itself as:
```c
nnmodel_t yolo26 = {
    .nb              = yolo26_get_network_filename_init,  // "NN_MDL/yolo26.nb"
    .preprocess      = yolo26_preprocess,                 // RGB resize + cache clean
    .postprocess     = yolo26_postprocess,                // auto-detect layout + decode
    .model_src       = MODEL_SRC_FILE,
    .set_init_info         = yolo26_set_init_info,
    .set_confidence_thresh = yolo26_set_confidence_thresh,
    .set_nms_thresh        = yolo26_set_nms_thresh,
    .name = "YOLO26"
};
```

### Supported Output Layouts (auto-detected)

| Layout | Tensors | Shape | Description |
|--------|---------|-------|-------------|
| E2E | 1 | (6, N) | End-to-end: x1,y1,x2,y2,score,class in pixels |
| CONCAT | 1 | (A, 4+nc) | Concatenated raw head across strides |
| SCALE | 3 | (W, H, 4+nc) | Per-stride raw head |
| SPLIT | 6 | (W,H,4)+(W,H,nc) | Box/cls separated per stride |

## Troubleshooting

### "ERROR: unsupported output layout"
The ONNX export may have included DFL (Distribution Focal Loss) layers. Re-export with `simplify=True` and ensure you're using YOLO26 (not YOLOv8 which uses DFL).

### Quantization accuracy drop
- Increase calibration images (use `--max-images 500`)
- Try `int16` quantization instead of `uint8`: `./convert_to_nb.sh --qtype int16`

### ONNX import fails
- Ensure opset is 12 (not higher): `python3 export_to_onnx.py --opset 12`
- Check input/output tensor names using Netron: `pip install netron && netron best.onnx`

### Model file too large
- Use `yolo26n.pt` (nano) instead of `yolo26s.pt` (small) for a smaller model
- The NPU has limited memory; typical `.nb` files are 1-5 MB
