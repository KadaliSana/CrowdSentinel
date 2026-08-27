# Cycle 2 — Live WebRTC Ingest (Design)

Date: 2026-08-28
Status: design proposal
Depends on: cycle 1 (`src/sfcpi/sources.py` FrameSource protocol)

## 1. Purpose

Let the pipeline consume the AmebaPro2's live H.264 stream from AWS KVS WebRTC, behind
the same `FrameSource` protocol the file replay already implements — so nothing
downstream changes.

## 2. The constraint that shapes everything

AWS ships KVS WebRTC SDKs for C, browser JS, Android and iOS. **There is no official
Python viewer SDK.** So this cycle writes two things AWS would otherwise provide:

1. **Signaling** — discover the channel endpoints, then hold a SigV4-signed WSS
   connection carrying SDP offer/answer and ICE candidates as JSON.
2. **A media bridge** — `aiortc` delivers frames on an asyncio event loop; `FrameSource`
   is a synchronous iterator. Something must cross that boundary safely.

Dependencies are installed and verified: aiortc 1.15.0, av 17.1.0, websockets 16.1.1, boto3 1.42.

## 3. Honest scope on verification

The signaling client and the frame bridge are unit-testable with fakes and recorded
message shapes. **End-to-end correctness cannot be established without the board
streaming to a real channel.** This spec does not pretend otherwise: the exit criterion
for the live path is a manual smoke test, and the automated suite covers only the parts
that can be honestly faked. A green suite here means "the plumbing is correct", not
"live ingest works".

## 4. Architecture

    webrtc/bridge.py      FrameBridge: async producer -> bounded sync queue -> iterator
    webrtc/signaling.py   KvsSignalingClient: endpoint discovery, SigV4 WSS URL, SDP/ICE
    webrtc/source.py      WebRTCSource: FrameSource over a live KVS channel
    webrtc/__init__.py

`WebRTCSource` satisfies the existing `FrameSource` protocol structurally — it exposes
`fps` and yields `Frame(index, timestamp, image)`. **No edit to `sources.py` is required
or permitted**, which is the whole point of having used a Protocol.

## 5. The bridge is where the bugs will be

Crossing async->sync is the part most likely to deadlock or leak, so it gets explicit rules:

- **Bounded queue.** A slow consumer must not grow memory without limit.
- **Drop-oldest, not block.** This is live video: a stalled consumer should lose old
  frames, never stall the event loop. Dropped frames are COUNTED and exposed, because a
  silently-lossy pipeline is a measurement error.
- **Sentinel-terminated.** Producer death must end the consumer's iteration, not hang it.
- **Idempotent close**, safe from either side.
- **Timeout on `next()`** so a dead connection surfaces as an error rather than a hang.

## 6. fps

`FrameSource.fps` must be correct BEFORE iteration (cycle 1 learned this the hard way:
fps enters crowd pressure as `fps^2`, so a silent default is a quadratic error).
A live stream has no declared fps, so it is MEASURED from frame timestamps over a short
warm-up window, and `fps` is unavailable until warm-up completes. Reading it early raises
rather than returning a plausible default.

## 7. Error handling

- Missing/invalid AWS credentials -> fail at construction, naming the missing setting.
- Channel not found -> explicit error naming the channel.
- Signaling socket drop -> surface as end-of-stream; reconnection is out of scope.
- ICE failure / no media within `connect_timeout_s` -> raise naming the phase that timed out.
- Any producer exception is captured and re-raised on the consumer side, never swallowed
  into a silent empty stream (an empty stream reads as "calm crowd").

## 8. Testing

- **Bridge:** fully unit-tested with a fake async producer — ordering, bounded capacity,
  drop-oldest accounting, sentinel termination, timeout, idempotent close.
- **Signaling:** endpoint-discovery and URL-signing tested against a stubbed boto3 client;
  SDP/ICE message parsing tested against recorded JSON shapes. No network.
- **Source:** tested with a fake track producing synthetic frames, asserting the
  `FrameSource` contract holds (indices monotonic, timestamps increasing, fps measured).
- **Live:** manual smoke test only, documented as such.

## 9. Out of scope

Reconnection/backoff, audio, sending media back, TURN credential refresh, the dashboard
UI swap (a follow-up once this source is proven).
