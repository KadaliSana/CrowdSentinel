# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CrowdSentinel: crowd-density / stampede-risk detection on a Realtek AmebaPro2 (RTL8735B) camera
board. The board runs a detector on its NPU (currently SCRFD) and streams H.264 to AWS KVS WebRTC;
each on-device inference also goes out as JSON on that connection's data channel. Host side is the
`sfcpi` pipeline, which the Flask dashboard now runs in-process: one WebRTC session drives both the
video panel and the analytics. **The alarm is on crowd pressure, not headcount** — see "The risk
model" below before touching anything that thresholds.

**WebRTC is the only transport.** The RTSP firmware app and its SDK clone (`src/firmware_rtsp/`)
were deleted from this tree, and RTSP is no longer compiled into the WebRTC app.

Three top-level folders — `flash/` (device tooling + images), `src/` (all source), `models/`
(weights, dataset, conversion):

| Tree | Role |
|---|---|
| `src/firmware_webrtc/` | The AmebaPro2 firmware application + the vendored SDK it builds against (below). |
| `src/sfcpi/` | Host pipeline: KVS WebRTC ingest, board-metadata parsing, risk/alerting, CLI. |
| `src/dashboard/` | Flask dashboard (host side). Consumes the KVS WebRTC stream and runs the `sfcpi` pipeline in-process; the count comes from the board, the risk level from the pipeline. |
| `models/dataset/`, `models/weights/` | YOLO26 training (Roboflow "crowd density" dataset, `nc:1`, class `person`) + ONNX experiments in `models/weights/`. |
| `models/conversion/` | `best.pt` → ONNX → Acuity/pegasus → `yolo26.nb` (NPU binary). See `models/conversion/README_conversion.md`. |

### The firmware application

One app: the **KVS WebRTC application** at
`src/firmware_webrtc/project/realtek_amebapro2_webrtc_application`. It compiles the on-device NN
stack (`module_vipnn.c` + the selected `model_*.c` + `nn_utils/`). NN wiring lives in
`src/firmware_webrtc/examples/app_media_source/port/ameba_pro2/ameba_pro2_media_port.c`: it opens
`vipnn_module` with the model named in the NN MODEL SELECTION block, renders boxes via OSD, and
builds the per-inference JSON that `examples/master/master.c` pushes down the WebRTC data channel
(`PeerConnectionSCTP_DataChannelSend`).

RTSP was removed from this app's build: `scenario.cmake` no longer compiles `module_rtsp2.c`, and
`GCC-RELEASE/application/application.cmake` no longer compiles
`component/network/rtsp/{rtp_api,rtsp_api,sdp}.c`. The firmware has been rebuilt without those
sources and flashed successfully — this is the only firmware on the board.

#### Detection metadata on the WebRTC data channel

Every inference is also published as JSON to the viewers, so the host does not have to re-detect
what the board already saw:

```json
{"t":<ms>,"w":1280,"h":720,"model":"scrfd","d":[[x,y,w,h,conf_pct], ...],"n":<true count>,"trunc":0|1}
```

`n` is authoritative. `d` is capped at `MEDIA_PORT_METADATA_MAX_DETECTIONS` (64), and `trunc` is 1
when that cap bites — so on a dense crowd `len(d) < n`, and counting `d` under-reports exactly the
crowds this system exists to detect. A message is emitted on **every** inference, `n=0` included:
silence means the board stopped, not that the scene is empty.

Path: the NN callback in `ameba_pro2_media_port.c` builds `metaJson` →
`AppMediaSourcePort_RegisterMetadataSink` → `OnMetadataSinkHook` in `examples/master/master.c` →
`PeerConnectionSCTP_DataChannelSend`. Host side: `src/sfcpi/webrtc/metadata.py`
(`parse_detection_message`) and `sfcpi.detect.BoardDetector`.

### The vendored SDK — read before editing SDK files

There is exactly **one** populated copy of the AmebaPro2 SDK left:
`src/firmware_webrtc/libraries/ambpro2_sdk/`, vendored with no `.git`. The app's
`GCC-RELEASE/CMakeLists.txt:25` sets `sdk_root` to it, so `scenario.cmake` references like
`${sdk_prj_example_root}/src/test_model/model_scrfd.c` resolve there.

The second copy, `src/firmware_rtsp/` — its own git repo (remote `Freertos-kvs-LTS/ambpro2_sdk`),
which built the RTSP app — was deleted. It is recoverable from that remote, but nothing here builds
against it any more, and it carried the `memcpy32` defect below **unpatched**.

Because the surviving copy is vendored rather than pinned, **an SDK refresh silently reverts the
local patches in it** — above all the `memcpy32` fix in `component/file_system/fwfs/fwfs.c`. Re-run
the check in that section after any refresh.

The `yolo26.nb` under the vendored copy
(`project/realtek_amebapro2_v0_example/src/test_model/model_nb/`) is a valid raw-head export
(6 outputs: 52/26/13 grids x [4 box + 1 cls], input scale 1/255 -> SPLIT layout). Its
`model_yolo26.c` has `set_init_info` and **no** class filter (`set_desired_class` existed only in
the deleted copy) — worth knowing if an old note or diff mentions class filtering.

Inspect any `.nb` before trusting it:

```bash
cd "models/acuity/Verisilicon_SW_NBInfo_1.2.17_20230412"
bash build_nbinfo.sh && ./nbinfo -in -out /path/to/yolo26.nb
```

The model embedded in an already-built flash image can be extracted the same way — find the `VPMN`
magic (`grep -abo VPMN flash_ntz.nn.bin`) and `dd` from that offset.

## Commands

Python tests are a `pytest` suite under `tests/` (`pytest.ini`: `testpaths = tests`,
`pythonpath = src`). Run `pytest` from the repo root. The firmware has no test suite.

### Host pipeline / dashboard
```bash
python3 -m sfcpi.cli live                    # live KVS WebRTC ingest + risk assessment
cd src/dashboard && python3 server.py        # http://localhost:5000
```
`--channel` / `--region` are optional. They default to `$KVS_CHANNEL_NAME` / `$AWS_REGION` (then
`$AWS_DEFAULT_REGION`), falling back to `camstream` / `ap-south-1` — deliberately the same env
names `src/dashboard/server.py` reads, so the CLI and the dashboard cannot be aimed at different
boards by accident. (The `AWS_DEFAULT_REGION` fallback is CLI-only; the dashboard reads
`AWS_REGION` alone.) There is one board on one channel, so these are defaults, not flags you are
expected to pass.

### Firmware build
```bash
export PATH=/home/sana/tools/asdk-10.3.0/linux/newlib/bin:$PATH   # toolchain lives outside the repo
cd src/firmware_webrtc/project/realtek_amebapro2_webrtc_application/GCC-RELEASE && mkdir -p build && cd build
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

- **input size** — `export_to_onnx.py --imgsz` must equal `MEDIA_PORT_NN_WIDTH`/`_HEIGHT` in
  `ameba_pro2_media_port.c` (set in the NN MODEL SELECTION block near the top). Training `imgsz`
  (640) is independent of this. Note the ISP constraint below: the NN channel size is not free to
  match any model — see "wrong NN channel size kills VOE". That constraint and this script are
  currently at odds: the board only accepts 576x320, while `export_to_onnx.py` takes one `--imgsz`
  and emits square inputs only (default 416).
- **opset 12** — the NPU's maximum; higher opsets fail Acuity import.
- **`lid:` in `*_inputmeta.yml`** must match the input layer id in the generated `best.json`
  (`grep -o '"lid":"[^"]*"' best.json | head -1`).
- **the `FWFS.files` entry in `amebapro2_fwfs_nn_models.json`** (currently `"scrfd320p"`) — controls
  whether the `.nb` is packed into the image at all. The build **copies this file from
  `libraries/ambpro2_sdk/project/realtek_amebapro2_v0_example/GCC-RELEASE/mp/`** over the one in
  `build/` on every run (`GCC-RELEASE/CMakeLists.txt:142`), so edit the `mp/` copy — editing `build/` is lost.
  `auto_model_cfg` does *not* rewrite `FWFS.files`; it is hardcoded there.

`model_yolo26.c` auto-detects the head layout (E2E / CONCAT / SCALE / SPLIT); an "unsupported output
layout" log usually means the ONNX export kept DFL layers.

### NN model selection is centralised

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

## Known defect: wrong NN channel size kills VOE (no video at all)

**Symptom:** `VOE cmd 0x206 ACK timeout` → `VOE_OPEN_CMD command fail` → `hal_video_open fail`,
preceded by a VOE dump full of `A5A5A5A5`, then `[VID Err]Please check sensor id first, the id is 2`.
The NN then looks broken because it never receives a frame.

`MEDIA_PORT_NN_WIDTH/_HEIGHT` must be a size the ISP actually accepts. **576x320 is the only size
observed working on this board**; 416x416 and 640x640 both killed VOE. The constraint is empirical
(all three are multiples of 16, so there is no formula to check) — hence the whitelist assert. A new
size must be tested on device.

Confirmed working at 576x320: VOE opens, `Deploy SCRFD` completes, and the H.264 stream reaches AWS
KVS WebRTC end to end.

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

## The risk model: the alarm is on PRESSURE, not headcount

The single most misunderstood thing in this repo. The board reports a person count, and the
dashboard shows it, but **nothing alerts on it**. The alarm quantity is crowd pressure
(`src/sfcpi/metrics/pressure.py`):

```
P = (count / cell_area_px) * fps^2 * Var(velocity_px_per_frame)      # units s^-2
```

`Var(v)` is the variance of the velocity **vectors** (`Var(vx) + Var(vy)`), not of speed
magnitudes — the magnitude variance is 0 for perfect counterflow (+5, -5), i.e. exactly the
turbulent regime the metric exists to catch. `P` carries no length dimension, so the unknown
metres-per-pixel scale cancels and pixel-space inputs give the real value, given only the rate.

Headcount enters **only** through the density term and is *multiplied* by velocity variance.
So: 200 people standing still is NORMAL (Var(v) ≈ 0), while 15 people shoving in a doorway can
be CRITICAL. Any change that reintroduces a headcount threshold is a regression.

Default thresholds (`Thresholds` in `src/sfcpi/risk/levels.py`), rise/fall in s^-2:

| Level | rise | fall |
|---|---|---|
| ELEVATED | 0.010 | 0.008 |
| HIGH | 0.020 | 0.016 |
| CRITICAL | 0.040 | 0.032 |

The fall values give hysteresis; `Thresholds.__post_init__` rejects any set where a fall is not
strictly below its rise, or where the tiers are out of order across each other. The `0.020`
figure traces to Johansson/Helbing's crowd-turbulence onset — a default, not a constant (it is
R-dependent). A non-finite or absent pressure classifies as `RiskLevel.UNKNOWN`, never NORMAL —
an absent measurement is not a calm scene. As a *held* state UNKNOWN carries no severity (it is
deliberately unordered), so once finite readings resume the level re-enters at NORMAL rather than
inheriting whatever it was before the sensor went blind.

`RiskStateMachine` (`src/sfcpi/risk/machine.py`) adds dwell, re-alert and blind-sensor gating on
top of `classify()`.

## Warning: pressure numbers from before the dt fix are not comparable

`Pipeline.run` used to compute pressure with the **nominal** fps. `FrameBridge` is drop-oldest,
so on a live stream consecutive frames handed to optical flow are routinely many multiples of
`1/fps` apart — and pressure goes as rate², so the error squares. A measured 0.201 s median gap
against a nominal 29.41 fps inflated peak pressure **~35x** and fired a false CRITICAL alert on
an empty corridor.

`Pipeline.run` now derives the rate per frame pair from `frame.timestamp - prev.timestamp`, and
falls back to the nominal fps only when that gap is unusable (zero, backwards, or non-finite).
Tests: `tests/sfcpi/test_pipeline_dt.py`.

**Any pressure number, JSONL metrics file or threshold tuned before this fix is not comparable
with one after it.** Do not carry old numbers forward as a baseline.

## Known open issue: OSD boxes are burned into the analytics stream

The firmware draws detection rectangles as OSD into the same H.264 stream the host consumes, so
those boxes appear as image content on the host. They move whenever a detection moves — which
**injects motion the crowd did not make** into the optical-flow / pressure computation in `sfcpi`.
The analytics path wants the OSD off, or the NN overlay on a separate channel from the frames used
for flow. **Still unresolved** — the data-channel metadata above removed the need to *count* from
pixels, but not the need to measure motion from clean ones. Every pressure number currently
produced carries this contamination.

## Known open issue: the host pipeline processes only a fraction of the frames

Optical flow cannot keep up with 30 fps at 1280x704 (720 cropped down to a multiple of the 64 px
cell), so `FrameBridge` drops frames to keep latency bounded. Observed on live runs: **roughly
450-750 frames dropped per few hundred processed** (reported from live runs; no committed log
captures these figures). Correctness is protected — the dt fix above means each surviving pair is
scaled by its own real interval, not the nominal fps — but **coverage is partial**: a burst
shorter than the gap between two processed frames is not seen at all. Unresolved. Reducing
`SFCPI_FLOW_DOWNSCALE` trades flow resolution for coverage; `source.dropped` is the counter to
watch (the dashboard surfaces it as `dropped`, and `sfcpi live` prints it on exit).

## Credentials

Configured directly in headers, not env:

- `src/firmware_webrtc/examples/demo_config/demo_config.h` — gitignored at the root.
- The deleted `src/firmware_rtsp/` tree contained a live-looking AWS access key in
  `component/example/kvs_producer_mmf/sample_config.h`. Deleting the tree does not un-leak it — it
  survives in that clone's own git history. Treat the key as compromised and rotate it; do not copy
  the values into new files.

## Host side: dashboard (`src/dashboard/`) and pipeline (`src/sfcpi/`)

`src/dashboard/server.py` is the only dashboard. It consumes the live KVS WebRTC stream — it does
not pull RTSP, and it does not run host-side YOLO. **The person count comes from the board**: the
AmebaPro2 runs SCRFD on its NPU and sends each inference as JSON over the WebRTC data channel.

**The dashboard is no longer a passive viewer — it runs the SF-CPI pipeline in-process.** One
WebRTC session feeds both the video panel and the analytics: each frame is published for
`/video_feed` and then handed to `Pipeline`, which computes crowd pressure; a `RiskStateMachine`
turns that into a level and alert events, which go to the configured alert sinks and out on the
`/count_feed` SSE (`pressure`, `max_pressure`, `level`, `coverage`, `last_alert` alongside
`count`). **The browser does not compute risk at all** — the old `calcRisk(count)` headcount
heuristic in `static/main.js` is deleted, and `main.js` only renders the server's `level` and
`pressure`. (`CONFIG.CROWD_THRESHOLD = 50` survives there, but purely as a display scale for the
count bar and the chart's reference line — it decides nothing.)

Dashboard env knobs beyond `KVS_CHANNEL_NAME` / `AWS_REGION` / `KVS_CONNECT_TIMEOUT` /
`METADATA_MAX_AGE` / `PORT`:

| Variable | Default | Meaning |
|---|---|---|
| `SFCPI_CELL_SIZE` | `64` | grid cell edge in px; the frame is cropped down to a multiple of it |
| `SFCPI_FLOW_DOWNSCALE` | `0.5` | optical-flow input scale (lower = faster, fewer dropped frames) |
| `SFCPI_SCORE_FIELD` | `global_max_pressure` | which metric the state machine thresholds |
| `SFCPI_MIN_DWELL` | `2.0` | seconds a level must hold before it alerts |
| `SFCPI_MIN_REALERT` | `60.0` | seconds between repeat alerts at the same level |
| `SFCPI_BLIND_ALERT` | `30.0` | seconds of UNKNOWN before alerting that the sensor is blind |
| `SFCPI_MIN_COVERAGE` | `0.0` | minimum `sensing_confidence` for a reading to count |
| `SNS_TOPIC_ARN` | *(empty)* | SNS topic for alerts |
| `SNS_ENABLE` | `0` | must be `1` **and** a topic set before SNS is used at all |
| `SNS_DRY_RUN` | `1` | dry run unless explicitly `0` |

SNS is double-gated on purpose: paging real people is not a default. `SNS_ENABLE=1` with an empty
`SNS_TOPIC_ARN` logs a warning and alerts nowhere.

Only two viewers can attach to the board at once (`AWS_MAX_VIEWER_NUM = 2`), and a stale session
is reclaimed only after 30 s (`PEER_CONNECTION_INACTIVE_CONNECTION_TIMEOUT_MS`). If the dashboard
and the AWS console viewer are both connected, `sfcpi live` times out during ICE/media
negotiation — that is contention, not a bug. See `src/sfcpi/webrtc/README.md`.

- `src/sfcpi/webrtc/metadata.py` — `parse_detection_message` parses one data-channel message. `n` is
  the authoritative count; `d` is capped by the firmware (`MEDIA_PORT_METADATA_MAX_DETECTIONS`), so
  on a dense crowd `len(d) < n` and `trunc` is 1. Counting `d` under-reports exactly the crowds this
  system exists to detect.
- `sfcpi.detect.BoardDetector` — holds the most recent message. "Nothing received yet" and "message
  too old" are deliberately errors, not empty lists: an empty list would read as a calm scene.
- `python3 -m sfcpi.cli live` — live ingest plus risk assessment, headless. Same defaults as the
  dashboard (see Commands above); `--channel`/`--region` only if you need to override them.

See `src/sfcpi/webrtc/README.md` for the KVS signalling/SigV4 details, the three live-only
protocol defects (keepalives, trickle ICE, and the DTLS fingerprint bug below) and the
two-viewer limit.

### The DTLS fingerprint bug — why the viewer connected but saw nothing

Worth knowing even from the firmware side. aiortc advertises **three** `a=fingerprint:` lines
(sha-256 = 95 chars, sha-384 = 143, sha-512 = 191). The KVS C SDK master on the AmebaPro2 sizes
its fingerprint buffer for sha-256 and rejects the sha-512 line, failing certificate verification
and destroying the session:

```
[ERROR] DTLS_VerifyRemoteCertificateFingerprint: ... CERTIFICATE_FINGERPRINT_LENGTH < fingerprintMaxLen(191)
[ERROR] OnDtlsHandshakeComplete: Fail to DTLS_VerifyRemoteCertificateFingerprint with return 255
```

This happens **after** the viewer's own DTLS handshake reports success, so the host sees
`connectionState = connected` and simply never receives RTP — which for a long time read as a
board-side "master sends no media" fault. Browsers send a single sha-256 line, which is why the
AWS console viewer worked against the same board throughout. Fix: `keep_single_fingerprint()` in
`src/sfcpi/webrtc/signaling.py`, applied to the offer before it is sent; tests in
`tests/sfcpi/test_webrtc_fingerprint.py`. **The live viewer path now works end to end.**

## Searching this repo

Repo-wide grep is unusable without exclusions. Exclude at minimum:
`models/dataset/train/`, `models/dataset/valid/`, `models/dataset/test/`, `models/acuity/`, `*/GCC-RELEASE/build*/`,
`src/firmware_webrtc/libraries/` (unless you specifically mean the vendored SDK).

## Housekeeping

The tree was pruned from 7.2 GB to ~3.9 GB by removing duplicated acuity workspaces, extracted
archives, stock toolkit example models, build outputs, and the `runs/` training checkpoints; the
`src/firmware_rtsp/` tree (RTSP app + its SDK clone + the toolchain tarball) went on top of that.
What remains is load-bearing: the dataset under `models/dataset/{train,valid,test}`, the calibration
`images/` in the acuity workspace (needed to re-quantize), and the vendored SDK under
`src/firmware_webrtc/libraries/`. The ARM toolchain now lives outside the repo at
`/home/sana/tools/asdk-10.3.0`. Build directories are regenerable with `make flash_nn`; the last
flashed image is kept at `flash/Pro2_PG_tool_v1.4.3/flash_ntz.nn.bin`.
