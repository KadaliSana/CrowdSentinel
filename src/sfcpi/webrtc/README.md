# sfcpi.webrtc

Live ingest from an AWS Kinesis Video Streams (KVS) WebRTC signalling channel, wrapped to
look like the file-replay `FrameSource`s in `sfcpi.sources` so the rest of the pipeline
consumes both identically.

```
signaling.py  KvsSignalingClient + the SDP/ICE message envelope (boto3-based, no aiortc)
bridge.py     FrameBridge -- bounded, drop-oldest async(producer) -> sync(consumer) handoff
source.py     WebRTCSource -- FrameSource-shaped: .fps, __iter__, .connect()
```

## Status: plumbing verified, live path NOT verified against a real channel

Everything in this package has an automated test, and those tests are real (they exercise
actual code paths, not mocks-all-the-way-down) -- but **no test in this repo, and no run by
the author of this README, has ever moved a real frame from a real AWS KVS channel through
`WebRTCSource`.** A green `pytest` run here means *the wiring is correct*: precondition
checks fire before any I/O, the async/thread/queue plumbing doesn't deadlock or leak, and
errors reach the consumer instead of vanishing into a silent (and therefore falsely
reassuring) empty stream. It does not mean live ingest works end to end.

What has been checked, and how:

- The two required unit tests (`tests/sfcpi/test_webrtc_connect.py`) run with no network
  access: pre-flight WSS validation, and the phase-named timeout path.
- Manually, outside the automated suite (not committed -- there is no real KVS channel to
  point it at): a full negotiation was run against a *local* fake signalling server plus a
  second local `aiortc` peer acting as the KVS "master" -- real SDP offer/answer exchange,
  real ICE gathering, a real received video track, `feed()` producing `Frame`s, `.fps`
  settling after warm-up, and a clean `close()` (idempotent, callable from another thread,
  no leaked tasks or "coroutine was never awaited" warnings). This confirms the aiortc/
  websockets/signalling wiring is mechanically correct. It does **not** confirm anything
  about real AWS network paths, real STUN/TURN behaviour, or real KVS's exact message
  timing.
- The phase-timeout behaviour was checked against both a host that fails to resolve at all
  and a live-but-silent signalling peer (one that accepts the connection but never answers)
  -- both correctly raise `TimeoutError` naming the phase, neither hangs.

**Known gap: the WSS URL is used unsigned.** `KvsSignalingClient.endpoints()` returns the
raw endpoint AWS reports; real KVS requires the WebSocket handshake itself to carry SigV4
*query-string* authentication (a WebSocket handshake can't carry a normal `Authorization`
header, so AWS's viewer/master SDKs sign the URL's query string the way a presigned S3 URL
is signed). `WebRTCSource.connect()` does **not** do this signing -- it was kept out of this
task's scope to match `signaling.py`'s own scope (task 2) and to keep the two required tests
network-free and deterministic (adding a `describe()`/credentials round trip ahead of the
socket open would defeat the fake signalling client the tests use). **Practically, this
means `connect()` will fail authentication against a real AWS channel today.** Signing the
WSS URL (SigV4 query auth, service `kinesisvideo`, using the credentials already on the
injected `kinesisvideo` boto3 client) is the next piece of work before a live smoke test can
even reach the signalling handshake.

Do the live smoke test (once the signing gap above is closed) with the board actually
streaming to the channel before trusting this path in anger.

## Required IAM permissions

The viewer identity (the credentials behind the `kinesisvideo` boto3 client passed into
`KvsSignalingClient`) needs, at minimum:

- `kinesisvideo:DescribeSignalingChannel`
- `kinesisvideo:GetSignalingChannelEndpoint`
- `kinesisvideo:GetIceServerConfig` -- needed to fetch TURN credentials via
  `KvsSignalingClient.ice_servers(...)`; without it you fall back to STUN-only ICE
  (`connect(ice_servers=[])`), which will not traverse a symmetric NAT.

`GetIceServerConfig` is called against a *separate* `kinesis-video-signaling` boto3 client
(scoped to the channel's signalling endpoint), not the `kinesisvideo` client used for
discovery -- see `KvsSignalingClient.ice_servers`'s docstring.

## Environment variables

Nothing in this package reads environment variables itself -- `WebRTCSource` and
`KvsSignalingClient` take already-constructed objects (dependency injection, deliberately,
so both are testable without real AWS credentials or network access). The variables below
are what a caller's own entry-point script is expected to read; they are a convention for
this repo, not something enforced by this package:

| Variable | Used for |
|---|---|
| `AWS_REGION` (or `AWS_DEFAULT_REGION`) | `region=` passed to `KvsSignalingClient` and to `boto3.client("kinesisvideo", ...)` |
| `KVS_CHANNEL_NAME` | `channel_name=` passed to `KvsSignalingClient` |
| `KVS_ROLE` | `role=` passed to `KvsSignalingClient`; defaults to `"VIEWER"`, which is what this package implements (it never sends outbound media -- see Deferred below) |
| Standard AWS credential env vars (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`, or an instance/role profile) | picked up by `boto3.client(...)` the normal way |

## Usage

```python
import boto3
from sfcpi.webrtc.signaling import KvsSignalingClient
from sfcpi.webrtc.source import WebRTCSource

kinesisvideo = boto3.client("kinesisvideo", region_name="ap-south-1")
signaling = KvsSignalingClient(
    channel_name="camstream",
    region="ap-south-1",
    kinesisvideo_client=kinesisvideo,
    role="VIEWER",
)

source = WebRTCSource(signaling, warmup_frames=10, connect_timeout_s=15.0)
source.connect()  # blocks until the remote video track is flowing, or raises

try:
    for frame in source:          # sfcpi.sources.Frame: .index, .timestamp, .image (BGR ndarray)
        ...                       # hand off to the pipeline, exactly like a file-replay source
    print(f"stream ended cleanly; fps={source.fps}, dropped={source.dropped}")
except TimeoutError:
    # no frame for FrameBridge's own idle window (default 10s) -- the producer
    # went silent or died; NOT the same as connect()'s own connect-phase timeout.
    raise
finally:
    source.close()
```

`connect()` raises before returning if anything is wrong, so the loop above never silently
runs with zero frames:

- `RuntimeError` (mentions `"WSS"`) -- the channel has no WSS endpoint; checked before any
  thread or socket is created, so a bad channel name or missing IAM permission fails
  immediately rather than hanging for `connect_timeout_s`.
- `TimeoutError`, message names the phase -- `"signalling connect"` (couldn't open/complete
  the WebSocket handshake in time: DNS, network path, or the signing gap above) vs.
  `"ICE/media negotiation"` (WebSocket connected, SDP exchanged, but no video track arrived
  in time: ICE/NAT/TURN problem, or the far end never started sending).
- Any other exception the negotiation or media loop raises reaches the caller the same way,
  via `self.fail(exc)` -> `FrameBridge` re-raising it out of `__iter__` once buffered frames
  drain. This package never turns a producer failure into a quietly-empty stream.

## Deferred (explicitly out of scope for this cycle)

- SigV4-signing the WSS URL (see "Known gap" above) -- blocks the live smoke test.
- Reconnection/backoff on a dropped connection.
- TURN credential refresh (KVS ICE server credentials expire; nothing here renews them).
- Outbound media / audio -- this is a receive-only (`recvonly`) viewer.
- Wiring `WebRTCSource` into `src/dashboard/server.py` in place of the current RTSP source --
  a follow-up once the live smoke test passes, not before.
