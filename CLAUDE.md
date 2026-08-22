# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CrowdSentinel: crowd-density / stampede-risk detection running a YOLO26 model on the NPU of a
Realtek AmebaPro2 (RTL8735B) camera board, streamed to either RTSP or AWS KVS WebRTC, with a
Flask dashboard for viewing/analysis.

Three top-level folders — `flash/` (device tooling + images), `src/` (all source), `models/`
(weights, dataset, conversion):

| Tree | Role |
|---|---|
| `src/dashboard/` | Flask + Ultralytics dashboard (host side). Pulls RTSP from the board, runs YOLO on the host, serves SSE counts. |
| `models/dataset/`, `models/weights/` | YOLO26 training (Roboflow "crowd density" dataset, `nc:1`, class `person`) + ONNX experiments in `models/weights/`. |
| `models/conversion/` | `best.pt` → ONNX → Acuity/pegasus → `yolo26.nb` (NPU binary). See `models/conversion/README_conversion.md`. |
| `src/firmware_rtsp/`, `src/firmware_webrtc/` | Two AmebaPro2 firmware applications (below). |

### The two firmware applications

Both compile the same on-device NN stack (`module_vipnn.c` + `model_yolo26.c` + `nn_utils/`) but
differ in transport. Neither is "the" app — they are alternates.

- **RTSP app** — `src/firmware_rtsp/project/realtek_amebapro2_v0_example`, example
  `src/mmfv2_video_example/mmf2_video_example_vipnn_rtsp_init.c`. VGA 640x480 H.264 over RTSP,
  `NN_MODEL_OBJ = yolo26`. This is what `src/dashboard/server.py` consumes.
- **KVS WebRTC app** — `src/firmware_webrtc/project/realtek_amebapro2_webrtc_application`. NN wiring lives in
  `src/firmware_webrtc/examples/app_media_source/port/ameba_pro2/ameba_pro2_media_port.c` (opens `vipnn_module`
  with `&yolo26`, renders boxes via OSD).

### Which SDK copy a firmware edit lands in — read before editing SDK files

There are **two populated copies of the AmebaPro2 SDK** and they are NOT the same files:

- `src/firmware_rtsp/` is its own git repo (remote `Freertos-kvs-LTS/ambpro2_sdk`). It is what the RTSP app
  builds. It always shows as `?? src/firmware_rtsp/` in the root repo's status; root commits never capture it.
- `src/firmware_webrtc/libraries/ambpro2_sdk/` is a vendored copy with no `.git`. The WebRTC app's
  `CMakeLists.txt:25` sets `sdk_root` to it, so **`scenario.cmake` references like
  `${sdk_prj_example_root}/src/test_model/model_yolo26.c` resolve here, not into `src/firmware_rtsp/`.**

**The two copies have diverged — verify before assuming a change applies to both:**

- `model_yolo26.c`: the firmware_rtsp copy has `set_desired_class` (class filtering) and no `set_init_info`;
  the firmware_webrtc copy has `set_init_info` and no class filter. Decode logic is otherwise identical.
- `yolo26.nb`: the two projects ship **different models**. The firmware_webrtc one is a valid raw-head export
  (6 outputs: 52/26/13 grids x [4 box + 1 cls], input scale 1/255 -> SPLIT layout). The firmware_rtsp one is
  a broken graph cut with a single `(169,32,2)` output taken from an attention block
  (`/model.10/m/m.0/attn/Split_output_0`) and input scale 0.937744 instead of 1/255 — it decodes to
  `LAYOUT_UNKNOWN` and produces no detections at all.

Inspect any `.nb` before trusting it:

```bash
cd "models/acuity/Verisilicon_SW_NBInfo_1.2.17_20230412"
bash build_nbinfo.sh && ./nbinfo -in -out /path/to/yolo26.nb
```

The model embedded in an already-built flash image can be extracted the same way — find the `VPMN`
magic (`grep -abo VPMN flash_ntz.nn.bin`) and `dd` from that offset.

## Commands

No test suite exists anywhere in this repo.

### Dashboard
```bash
cd src/dashboard && python3 server.py          # http://localhost:5000
```

### Firmware build (either app)
```bash
export PATH=<sdk>/tools/asdk-10.3.0/linux/newlib/bin:$PATH   # src/firmware_rtsp/tools/ has the split tarball
cd <project>/GCC-RELEASE && mkdir -p build && cd build
cmake .. -G"Unix Makefiles" -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake
cmake --build . --target flash        # -> flash_ntz.bin       (app only)
cmake --build . --target flash_nn     # -> flash_ntz.nn.bin    (app + NN model partition)
```
Use `flash_nn` whenever the `.nb` model or `amebapro2_fwfs_nn_models.json` changed — `flash` alone
leaves the old model on the NN partition.

### Flash
```bash
./uartfwburn.linux -p /dev/ttyUSB0 -f flash_ntz.nn.bin -b 2000000 -U
```
Enter program mode first: hold Reset, press Program, release Reset, release Program.
Tool: `flash/Pro2_PG_tool_v1.4.3/`. (v1.3.0 was removed — it handshakes but then fails
with `fw_buf is NULL` on this image; recoverable from git history if ever needed.)

### Model pipeline
```bash
cd models/dataset && python3 train.py
cd models/conversion && python3 export_to_onnx.py --imgsz 416 --gen-dataset
./convert_to_nb.sh --name best --workspace ./conversion_workspace --qtype uint8
```

## Coupled constants in the model pipeline

These must be changed together or detection silently degrades:

- **416** — `export_to_onnx.py --imgsz` must equal `NN_WIDTH`/`NN_HEIGHT` in
  `mmf2_video_example_vipnn_rtsp_init.c` (~L95) **and** `MEDIA_PORT_NN_WIDTH`/`_HEIGHT` in
  `ameba_pro2_media_port.c` (now set in the NN MODEL SELECTION block near the top, not further down).
  Training `imgsz` (640) is independent of this. Note the ISP constraint below: the NN channel size
  is not free to match any model — see "wrong NN channel size kills VOE".
- **opset 12** — the NPU's maximum; higher opsets fail Acuity import.
- **`lid:` in `*_inputmeta.yml`** must match the input layer id in the generated `best.json`
  (`grep -o '"lid":"[^"]*"' best.json | head -1`).
- **`"yolo26"` in `amebapro2_fwfs_nn_models.json`'s `FWFS.files`** — controls whether the `.nb` is
  packed into the image at all. The build **copies this file from
  `libraries/ambpro2_sdk/project/realtek_amebapro2_v0_example/GCC-RELEASE/mp/`** over the one in
  `build/` on every run (`CMakeLists.txt:142`), so edit the `mp/` copy — editing `build/` is lost.
  `auto_model_cfg` does *not* rewrite `FWFS.files`; it is hardcoded there.

`model_yolo26.c` auto-detects the head layout (E2E / CONCAT / SCALE / SPLIT); an "unsupported output
layout" log usually means the ONNX export kept DFL layers.

### WebRTC app: NN model selection is centralised

`ameba_pro2_media_port.c` has a **NN MODEL SELECTION** block near the top that is the single source
of truth for the four things that must agree (model struct, decoder `.c` in `scenario.cmake`,
`FWFS.files` entry, and the ISP NN channel size). It ends in a compile-time whitelist assert on
`MEDIA_PORT_NN_WIDTH/_HEIGHT`, so an untested channel size **breaks the build instead of the board**.
Read that block before changing models — it records the failure mode of each mismatch.

## Known defect: `memcpy32` hard-faults on 32-byte-aligned models

**Symptom:** board resets right after `Deploy <model>` with `Usage Fault` /
`SCB Configurable Fault Status Reg = 0x01000000` (UNALIGNED), and a stack-scan backtrace pointing
into `fwfs.c`.

**Cause:** `component/file_system/fwfs/fwfs.c` defined `memcpy32()` as an ordinary C function whose
body is a bare `__asm` block ending in its own `pop {r0-r12}` + `bx lr`. GCC **inlined** it into
`nor_copy_read()`/`nor_pfw_read()`, so that `bx lr` returned from the *caller*, skipping its epilogue,
the `device_mutex_unlock`, the 4-byte-addr exit and `return size` — and branching through an `lr`
already clobbered by `bl device_mutex_lock`.

`nor_copy_read` only took that path when **dst, src and len were all 32-byte aligned**, so it
depended on the model's file size:

| `.nb` | size | `% 32` | result |
|---|---|---|---|
| `yolo26.nb` | 2868240 | 16 | safe `memcpy` — never hit the bug |
| `nanodet_plus_m_416_uint8.nb` | 1959040 | 0 | hard fault |
| `scrfd_500m_bnkps_576x320_u8.nb` | 583232 | 0 | hard fault |

**Padding a `.nb` by 16 bytes only dodges the guard** and must be redone per model — it was the old
workaround and it kept getting lost. The real fix (in the vendored SDK) drops the `memcpy32` fast
path and marks the function `__attribute__((naked, noinline))`. Verify after any SDK refresh:

```bash
arm-none-eabi-objdump -d application/application.ntz.axf --disassemble=nor_pfw_read | grep -c 'bx\slr'
# must print 0 — a bare `bx lr` means the defect is back
```

The same defect still exists **unpatched** in the `src/firmware_rtsp/` SDK copy.

## Known defect: wrong NN channel size kills VOE (no video at all)

**Symptom:** `VOE cmd 0x206 ACK timeout` → `VOE_OPEN_CMD command fail` → `hal_video_open fail`,
preceded by a VOE dump full of `A5A5A5A5`, then `[VID Err]Please check sensor id first, the id is 2`.
The NN then looks broken because it never receives a frame.

`MEDIA_PORT_NN_WIDTH/_HEIGHT` must be a size the ISP actually accepts. **576x320 is the only size
observed working on this board**; 416x416 and 640x640 both killed VOE. The constraint is empirical,
hence the whitelist assert rather than a formula — a new size must be tested on device.

Ruled out by partition-diffing flash images (don't re-test these): `voe.bin` is byte-identical
everywhere; the `fcsdata` partition is identical; the `iq` partition differs only by a 3-byte build
timestamp; and the forked `sensor.bin`/`sensor_f37.bin` were already present in the known-good image.
"`the id is 2`" means `sensor_sets[2]` (= F37), **not** `SENSOR_GC2053` (`0x02` in `inc/sensor.h`) —
the `IQ_OFFSET`/`SENSOR_OFFSET` values in the log settle it.

Compare images partition-by-partition with the offsets from
`GCC-RELEASE/build/amebapro2_partitiontable.json` (`fw1`=0x60000/0x400000, `iq`=0x460000/0xC0000,
`nn`=0x920000/0x5E0000, `fcsdata`=0x8000/0x1000), stripping trailing `0xff`.

Reference images: `flash/reference/known_good_1626.nn.bin` is the same WebRTC app built with yolo26
and working VOE; `flash/Pro2_PG_tool_v1.4.3/flash_ntz.nn.bin` is the last image actually flashed.
`/home/sana/Yan-backup/Yan/webrtc/` is a **full source tree** of an earlier working SCRFD build and is
the fastest way to diff against a known-good configuration.

## Known defect: uint8 class-head quantization is too coarse

The shipped `crowd26` uint8 `.nb` calibrates the class head `cv3.0` output to
`[-101.284, +1.208]` (scale `0.402`). Real logits span about `-6..+1`, so the decision
region (logit -1..+1 = confidence 27%-73%) gets **5 quantization levels**. On-device this
pins nearly every detection to exactly 40%, 50% or 60% confidence. The `-101` end is a
background activation carrying no information (sigmoid(-101) == sigmoid(-6) == 0 for any
practical purpose) yet eats 96% of the range.

The box head is NOT affected the same way: `cv2.0` calibrates to `[-1.288, 18.590]`
(scale `0.078`), and real box distances span 0-7 grid units, giving ~90 levels at 0.62px.

Fix: re-quantize as int16 with `models/acuity/Docker/Linux/acuity_examples_c901149/convert_crowd_rawhead_int16.sh`
(dynamic_fixed_point, 200 iterations). Needs the Acuity container, which needs `sudo docker`.
Note the `i32` entries in a `.quantize` file with ranges of +/-1e6 are **bias accumulators and
are normal** - do not mistake them for a defect.

## Credentials

Configured directly in headers, not env:

- `src/firmware_webrtc/examples/demo_config/demo_config.h` — gitignored at the root.
- `src/firmware_rtsp/component/example/kvs_producer_mmf/sample_config.h` — **inside the nested repo**, so the
  root `.gitignore` does not protect it. It contains a live-looking AWS access key. Treat it as
  compromised and rotate it; do not copy the values into new files.

## Dashboard (`src/dashboard/`)

`src/dashboard/server.py` is the only dashboard. Config is environment-driven via `src/dashboard/.env`:
`RTSP_URL`, `MODEL_PATH` (default `src/dashboard/best.pt`), `CONF_THRESH`, `AI_WIDTH`/`AI_HEIGHT`.
A missing model now raises at startup instead of silently substituting `yolov8n.pt` —
that fallback used to produce odd detections that read like a firmware bug.

`README.md` is stale: it describes `src/models/best_m.pt` and a folder layout that does not
exist. Trust this file over the README.

## Searching this repo

Repo-wide grep is unusable without exclusions. Exclude at minimum:
`models/dataset/train/`, `models/dataset/valid/`, `models/dataset/test/`, `models/acuity/`, `*/GCC-RELEASE/build*/`,
`src/firmware_rtsp/component/`, `src/firmware_webrtc/libraries/` (unless you specifically mean the vendored SDK).

## Housekeeping

The tree was pruned from 7.2 GB to ~3.9 GB by removing duplicated acuity workspaces, extracted
archives, stock toolkit example models, build outputs, and the `runs/` training checkpoints.
What remains is load-bearing: the `asdk-10.3.0` toolchain (firmware builds), the dataset under
`models/dataset/{train,valid,test}`, the calibration `images/` in the acuity
workspace (needed to re-quantize), and both vendored SDKs. Build directories are regenerable
with `make flash_nn`; the last flashed image is kept at
`flash/Pro2_PG_tool_v1.4.3/flash_ntz.nn.bin`.
