# sfcpi.webrtc

Live ingest from an AWS Kinesis Video Streams (KVS) WebRTC signalling channel, wrapped to
look like the file-replay `FrameSource`s in `sfcpi.sources` so the rest of the pipeline
consumes both identically.

```
signaling.py  KvsSignalingClient + the SDP/ICE message envelope + sign_wss_url (SigV4 query
              signing, via botocore.auth/botocore.awsrequest -- no aiortc, no boto3 import)
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

**SigV4 WSS signing is now implemented** (`sign_wss_url` in `signaling.py`, wired into
`WebRTCSource.connect()`). `KvsSignalingClient.endpoints()` used to return the raw AWS
endpoint and `connect()` used it as-is; real KVS requires the WebSocket handshake itself to
carry SigV4 *query-string* authentication (a WebSocket handshake can't carry a normal
`Authorization` header, so AWS's viewer/master SDKs sign the URL's query string the way a
presigned S3 URL is signed). `connect()` now signs the WSS URL before opening it, whenever
signing is actually possible and needed:

- `KvsSignalingClient` caches the channel's `ChannelARN` (as `.channel_arn`) the moment
  `describe()` runs -- which `endpoints()` already does internally -- so signing costs no
  extra network round trip over what `connect()` was already doing.
- `connect()` now takes an optional `credentials=` argument (a botocore-style credentials
  object: `.access_key`, `.secret_key`, optional `.token`), **injected**, never fetched
  inside this package by default. If omitted, `connect()` falls back lazily to
  `boto3.Session().get_credentials()` -- but only when there is a `channel_arn` to sign
  against in the first place, so the network-free tests (whose fake signalling objects have
  no `channel_arn`) neither pay for nor depend on that lookup: signing is skipped and the
  raw URL is used, exactly as before this feature existed. Against a real
  `KvsSignalingClient` where `channel_arn` is set but no credentials can be found anywhere,
  `connect()` raises `RuntimeError` before opening any socket, rather than silently
  attempting an unsigned (and certain-to-fail) connection.
- Signing itself is delegated to `botocore.auth.SigV4QueryAuth.add_auth()` against a
  `botocore.awsrequest.AWSRequest` (service `kinesisvideo`) -- not hand-rolled HMAC --
  because canonicalisation and percent-encoding are easy to get subtly wrong by hand.
  `add_auth()` is deliberately the library's **public** entry point, not its internal
  signing steps (`_modify_request_before_signing` / `canonical_request` / `string_to_sign`
  / `signature` / `_inject_signature_to_request`) called directly: those carry no
  compatibility guarantee, and production code depending on them purely to make tests
  deterministic would mean a botocore upgrade could silently break real signing while CI
  stayed green. The signature covers the request's host and path, not its scheme:
  `sign_wss_url` signs the `https://` form of the endpoint and returns the result with the
  `wss://` scheme restored.

**Verified by `tests/sfcpi/test_webrtc_signing.py` (network-free, no real credentials --
uses `botocore.credentials.Credentials("AKIDEXAMPLE", ...)`):** the
signed URL keeps the `wss://` scheme, host and path; all six presign params
(`X-Amz-Algorithm`, `X-Amz-Credential`, `X-Amz-Date`, `X-Amz-Expires`, `X-Amz-SignedHeaders`,
`X-Amz-Signature`) are present; `X-Amz-ChannelARN` is present and correctly percent-encoded;
`X-Amz-ClientId` appears iff a `client_id` is given; signing twice under a frozen clock is
byte-identical (determinism); **changing the channel ARN with everything else fixed changes
`X-Amz-Signature`** (proof the signature actually covers the request, not a constant); a
session token on the credentials appears as `X-Amz-Security-Token`; and an empty
`wss_endpoint` or `channel_arn` raises `ValueError` naming the missing field. Determinism
comes from an autouse fixture that freezes the clock `add_auth()` itself reads
(`botocore.auth.get_current_datetime`, monkeypatched) rather than from a timestamp parameter
threaded through `sign_wss_url` -- `sign_wss_url` has no such parameter, so production always
signs against the real current time, and the one remaining dependency on a botocore internal
lives in the test file, guarded by its own canary test (`test_canary_botocore_auth_still_exposes_get_current_datetime`)
so a rename/removal fails loudly in CI instead of silently in production. Also manually
verified (not committed, no assertions, just a wiring smoke check) that
`WebRTCSource.connect()` actually passes the *signed* URL to `websockets.connect(...)`
against a fake `kinesisvideo` client, and that omitting credentials against a real
`channel_arn` with no AWS credentials configured anywhere raises the `RuntimeError` above
rather than connecting unsigned.

**Still NOT verified: nothing has been tested against a real AWS KVS channel.** Whether a
real KVS signalling service actually accepts a `sign_wss_url(...)`-produced URL --
correct region, correct scope, correct treatment of `Role: VIEWER` vs `MASTER` in KVS's
own validation, TURN-credential-adjacent edge cases -- is unverified. Do the live smoke
test with the board actually streaming to the channel before trusting this path in anger.

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
# credentials= is optional -- omitted here, connect() falls back lazily to
# boto3.Session().get_credentials() itself (env vars, shared config, instance/role
# profile, ...). Pass it explicitly for testability or if the WebRTCSource's caller
# and the kinesisvideo client above should not share the ambient credential chain.
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
- `RuntimeError` (mentions `"credentials"`) -- the channel needs a SigV4-signed WSS
  connection (its `ChannelARN` is known) but no AWS credentials were found anywhere, neither
  passed as `credentials=` nor discoverable by `boto3.Session().get_credentials()`; also
  checked before any thread or socket is created.
- `TimeoutError`, message names the phase -- `"signalling connect"` (couldn't open/complete
  the WebSocket handshake in time: DNS, network path, or bad/expired signing) vs.
  `"ICE/media negotiation"` (WebSocket connected, SDP exchanged, but no video track arrived
  in time: ICE/NAT/TURN problem, or the far end never started sending).
- Any other exception the negotiation or media loop raises reaches the caller the same way,
  via `self.fail(exc)` -> `FrameBridge` re-raising it out of `__iter__` once buffered frames
  drain. This package never turns a producer failure into a quietly-empty stream.

## Deferred (explicitly out of scope for this cycle)

- The live smoke test itself (SigV4 signing is now implemented -- see above -- but nothing
  has actually connected to a real AWS KVS channel with it yet).
- Reconnection/backoff on a dropped connection.
- TURN credential refresh (KVS ICE server credentials expire; nothing here renews them).
- Outbound media / audio -- this is a receive-only (`recvonly`) viewer.
- Wiring `WebRTCSource` into `src/dashboard/server.py` in place of the current RTSP source --
  a follow-up once the live smoke test passes, not before.
