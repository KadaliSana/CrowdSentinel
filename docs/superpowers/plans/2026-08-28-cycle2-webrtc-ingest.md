# Cycle 2 — Live WebRTC Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Consume the AmebaPro2's live KVS WebRTC stream behind the existing `FrameSource` protocol.

**Architecture:** A new self-contained `src/sfcpi/webrtc/` package: a bounded async→sync `FrameBridge`, a KVS signalling client, and a `WebRTCSource` that structurally satisfies `FrameSource`. **No existing module is edited.**

**Tech Stack:** Python 3.10, aiortc 1.15.0, av 17.1.0, websockets 16.1.1, boto3 1.42, numpy, pytest 6.2.

**Spec:** `docs/superpowers/specs/2026-08-28-cycle2-webrtc-ingest-design.md`

## Global Constraints

- **Create only new files under `src/sfcpi/webrtc/` and `tests/sfcpi/`.** Do NOT edit `sources.py`, `cli.py`, `pipeline.py`, or anything under `risk/` or `alerts/` — another agent is working in `risk/`, `alerts/` and `cli.py` right now. `FrameSource` is a structural Protocol, so no edit to `sources.py` is needed.
- `WebRTCSource` must expose `fps: float` and yield `Frame(index, timestamp, image)` with `image` a BGR numpy array, matching `sfcpi.sources.Frame`.
- **fps must be correct before it is read.** A live stream declares none, so it is MEASURED over a warm-up window; reading `fps` before warm-up RAISES rather than returning a plausible default. Frame rate enters crowd pressure as `fps**2`, so a silent default is a quadratic error.
- **Never swallow a producer error into an empty stream.** An empty stream reads as "calm crowd" downstream. Capture and re-raise on the consumer side.
- No test may touch the network or require AWS credentials. Stub boto3 and fake the media track.
- Package root `src/sfcpi/`, tests `tests/sfcpi/`. Run `python3 -m pytest`.
- Stage only your own paths; never `git add -A` (an untracked `webrtc` symlink at the repo root must NEVER be committed — note the unfortunate name collision, be careful).
- Commit after every task.

---

### Task 1: FrameBridge — bounded async→sync handoff

**Files:** Create `src/sfcpi/webrtc/__init__.py`, `src/sfcpi/webrtc/bridge.py`; Test `tests/sfcpi/test_webrtc_bridge.py`

**Interfaces produced:**
- `FrameBridge(maxsize=8, timeout_s=10.0)`
- `.put(item)` — called from the producer thread/loop; drops the OLDEST item when full and increments `.dropped`
- `.close()` / `.fail(exc)` — idempotent; `fail` stores an exception to re-raise on the consumer side
- `.__iter__()` — yields items until closed; raises `TimeoutError` if nothing arrives within `timeout_s`; re-raises a stored producer exception
- attributes `.dropped: int`, `.closed: bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_webrtc_bridge.py
import threading
import pytest
from sfcpi.webrtc.bridge import FrameBridge

def test_items_arrive_in_order():
    b = FrameBridge()
    for i in range(5):
        b.put(i)
    b.close()
    assert list(b) == [0, 1, 2, 3, 4]

def test_close_terminates_iteration():
    b = FrameBridge()
    b.put("x"); b.close()
    assert list(b) == ["x"]
    assert b.closed is True

def test_bounded_queue_drops_oldest_and_counts():
    """Live video: a stalled consumer must lose OLD frames, never stall the producer."""
    b = FrameBridge(maxsize=3)
    for i in range(10):
        b.put(i)
    b.close()
    got = list(b)
    assert len(got) == 3
    assert got == [7, 8, 9]          # newest retained
    assert b.dropped == 7

def test_producer_exception_is_reraised_not_swallowed():
    """An empty stream reads as 'calm crowd' downstream -- errors must surface."""
    b = FrameBridge()
    b.put(1)
    b.fail(RuntimeError("ice failed"))
    it = iter(b)
    assert next(it) == 1
    with pytest.raises(RuntimeError, match="ice failed"):
        next(it)

def test_timeout_when_producer_goes_silent():
    b = FrameBridge(timeout_s=0.2)
    with pytest.raises(TimeoutError):
        next(iter(b))

def test_close_is_idempotent():
    b = FrameBridge()
    b.close(); b.close()
    assert list(b) == []

def test_put_after_close_is_ignored():
    b = FrameBridge()
    b.close()
    b.put("late")
    assert list(b) == []

def test_works_across_threads():
    b = FrameBridge(maxsize=100)
    def produce():
        for i in range(50):
            b.put(i)
        b.close()
    threading.Thread(target=produce, daemon=True).start()
    assert list(b) == list(range(50))
```

- [ ] **Step 2: Run test to verify it fails** — `python3 -m pytest tests/sfcpi/test_webrtc_bridge.py -v`

- [ ] **Step 3: Implement**

```python
# src/sfcpi/webrtc/__init__.py
"""Live KVS WebRTC ingest. Import-light: heavy deps load inside submodules."""
from .bridge import FrameBridge

__all__ = ["FrameBridge"]
```

```python
# src/sfcpi/webrtc/bridge.py
"""Bounded async->sync handoff.

aiortc delivers frames on an asyncio loop; FrameSource is a synchronous iterator.
This is the boundary, and it is where deadlocks and leaks live, so the rules are
explicit: bounded, drop-oldest (never block the loop), sentinel-terminated,
idempotent close, and a consumer-side timeout so a dead connection surfaces as an
error instead of a hang.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Iterator, Optional

_SENTINEL = object()


class FrameBridge:
    def __init__(self, maxsize: int = 8, timeout_s: float = 10.0) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._closed = False
        self._error: Optional[BaseException] = None
        self.dropped = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def put(self, item: Any) -> None:
        """Never blocks. When full, discards the OLDEST item and counts it."""
        if self._closed:
            return
        while True:
            try:
                self._q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:  # pragma: no cover - racy drain
                    pass

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
                self._q.put_nowait(_SENTINEL)
            except (queue.Empty, queue.Full):  # pragma: no cover
                pass

    def __iter__(self) -> Iterator[Any]:
        while True:
            try:
                item = self._q.get(timeout=self._timeout_s)
            except queue.Empty:
                raise TimeoutError(
                    f"no frame within {self._timeout_s}s; the producer is silent or dead"
                ) from None
            if item is _SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            yield item
```

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit** — `git add src/sfcpi/webrtc tests/sfcpi/test_webrtc_bridge.py && git commit -m "feat(sfcpi): bounded async-to-sync frame bridge"`

---

### Task 2: KVS signalling client

**Files:** Create `src/sfcpi/webrtc/signaling.py`; Test `tests/sfcpi/test_webrtc_signaling.py`

**Interfaces produced:**
- `KvsSignalingClient(channel_name, region, kinesisvideo_client, role="VIEWER")`
- `.describe() -> dict` (channel ARN)
- `.endpoints() -> dict` mapping `"WSS"`/`"HTTPS"` to URLs
- `.ice_servers(signaling_client) -> list[dict]`
- `encode_sdp_offer(client_id, sdp) -> str` / `decode_message(raw) -> tuple[str, str, dict]`
  returning `(message_type, sender_client_id, payload)`; payload is base64-decoded JSON

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_webrtc_signaling.py
import base64
import json
import pytest
from sfcpi.webrtc.signaling import KvsSignalingClient, encode_sdp_offer, decode_message

ARN = "arn:aws:kinesisvideo:ap-south-1:123456789012:channel/camstream/1"

class _StubKV:
    def __init__(self, missing=False):
        self.missing = missing
        self.calls = []
    def describe_signaling_channel(self, **kw):
        self.calls.append(("describe", kw))
        if self.missing:
            from botocore.exceptions import ClientError
            raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "Describe")
        return {"ChannelInfo": {"ChannelARN": ARN, "ChannelName": "camstream"}}
    def get_signaling_channel_endpoint(self, **kw):
        self.calls.append(("endpoint", kw))
        return {"ResourceEndpointList": [
            {"Protocol": "WSS", "ResourceEndpoint": "wss://example.kinesisvideo/"},
            {"Protocol": "HTTPS", "ResourceEndpoint": "https://example.kinesisvideo/"},
        ]}

def _client(**kw):
    kw.setdefault("channel_name", "camstream")
    kw.setdefault("region", "ap-south-1")
    kw.setdefault("kinesisvideo_client", _StubKV())
    return KvsSignalingClient(**kw)

def test_describe_returns_channel_arn():
    assert _client().describe()["ChannelARN"] == ARN

def test_missing_channel_names_the_channel():
    c = _client(kinesisvideo_client=_StubKV(missing=True))
    with pytest.raises(LookupError, match="camstream"):
        c.describe()

def test_endpoints_are_keyed_by_protocol():
    eps = _client().endpoints()
    assert eps["WSS"].startswith("wss://")
    assert eps["HTTPS"].startswith("https://")

def test_viewer_role_is_passed_through():
    stub = _StubKV()
    _client(kinesisvideo_client=stub).endpoints()
    _, kw = stub.calls[-1]
    assert kw["SingleMasterChannelEndpointConfiguration"]["Role"] == "VIEWER"

def test_empty_channel_name_rejected_at_construction():
    with pytest.raises(ValueError, match="channel_name"):
        _client(channel_name="")

def test_encode_sdp_offer_is_base64_json():
    raw = encode_sdp_offer("viewer-1", "v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\n")
    msg = json.loads(raw)
    assert msg["action"] == "SDP_OFFER"
    assert msg["recipientClientId"] == "viewer-1"
    decoded = json.loads(base64.b64decode(msg["messagePayload"]))
    assert decoded["type"] == "offer" and decoded["sdp"].startswith("v=0")

def test_decode_message_round_trips_an_answer():
    payload = base64.b64encode(json.dumps({"type": "answer", "sdp": "v=0\r\n"}).encode()).decode()
    raw = json.dumps({"messageType": "SDP_ANSWER", "senderClientId": "master",
                      "messagePayload": payload})
    mtype, sender, body = decode_message(raw)
    assert mtype == "SDP_ANSWER" and sender == "master" and body["type"] == "answer"

def test_decode_message_rejects_malformed_payload():
    raw = json.dumps({"messageType": "SDP_ANSWER", "senderClientId": "m",
                      "messagePayload": "!!!not-base64!!!"})
    with pytest.raises(ValueError, match="payload"):
        decode_message(raw)
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement `signaling.py`.** Use the stub-shaped boto3 calls above: `describe_signaling_channel(ChannelName=...)` and `get_signaling_channel_endpoint(ChannelARN=..., SingleMasterChannelEndpointConfiguration={"Protocols": ["WSS", "HTTPS"], "Role": role})`. Translate `ResourceNotFoundException` into a `LookupError` naming the channel. `encode_sdp_offer` builds `{"action": "SDP_OFFER", "recipientClientId": ..., "messagePayload": base64(json({"type":"offer","sdp":...}))}`. `decode_message` reverses it and raises `ValueError` mentioning "payload" on malformed base64/JSON. Do NOT construct a boto3 client inside this module — it is injected.

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 3: WebRTCSource

**Files:** Create `src/sfcpi/webrtc/source.py`; Test `tests/sfcpi/test_webrtc_source.py`

**Interfaces produced:** `WebRTCSource(signaling, bridge=None, warmup_frames=10, connect_timeout_s=15.0)` with `.fps` and `__iter__`, plus `.feed(av_frame_like, timestamp)` used by the media task and by tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/sfcpi/test_webrtc_source.py
import numpy as np
import pytest
from sfcpi.sources import Frame
from sfcpi.webrtc.source import WebRTCSource

class _FakeSignaling:
    channel_name = "camstream"

def _src(**kw):
    kw.setdefault("signaling", _FakeSignaling())
    kw.setdefault("warmup_frames", 3)
    return WebRTCSource(**kw)

def _img(v=0):
    return np.full((8, 8, 3), v, np.uint8)

def test_fps_before_warmup_raises_not_a_default():
    """A silent fps default is a quadratic error downstream."""
    s = _src()
    with pytest.raises(RuntimeError, match="fps"):
        _ = s.fps

def test_fps_is_measured_from_timestamps():
    s = _src()
    for i in range(3):
        s.feed(_img(i), timestamp=i * 0.1)     # 10 fps
    s.close()
    assert s.fps == pytest.approx(10.0, rel=0.05)

def test_yields_frames_matching_the_framesource_contract():
    s = _src()
    for i in range(3):
        s.feed(_img(i), timestamp=i * 0.1)
    s.close()
    frames = list(s)
    assert all(isinstance(f, Frame) for f in frames)
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[1].timestamp == pytest.approx(0.1)
    assert frames[0].image.shape == (8, 8, 3)

def test_producer_error_surfaces_to_the_consumer():
    s = _src()
    s.feed(_img(), timestamp=0.0)
    s.fail(RuntimeError("ice failed"))
    with pytest.raises(RuntimeError, match="ice failed"):
        list(s)

def test_dropped_frames_are_counted_and_visible():
    s = _src(maxsize=2)
    for i in range(6):
        s.feed(_img(i), timestamp=i * 0.1)
    s.close()
    list(s)
    assert s.dropped > 0
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement `source.py`.** `feed()` converts the incoming frame to a BGR numpy array (accept a numpy array directly, and an `av.VideoFrame` via `.to_ndarray(format="bgr24")` when present), stamps an increasing index, and `put`s a `Frame` onto an internal `FrameBridge`. Measure fps from the first `warmup_frames` timestamps (median delta); `fps` raises `RuntimeError` naming fps until warm-up completes. Expose `.dropped` from the bridge, `.close()`, `.fail(exc)`. Import `aiortc` LAZILY inside the connect path only, so these tests never need it. Accept `maxsize` and forward it to the bridge.

- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 4: Live connect path and documentation

**Files:** Modify `src/sfcpi/webrtc/source.py`; Create `src/sfcpi/webrtc/README.md`; Test `tests/sfcpi/test_webrtc_connect.py`

- [ ] **Step 1: Write the failing test** (connection wiring only — no network)

```python
# tests/sfcpi/test_webrtc_connect.py
import pytest
from sfcpi.webrtc.source import WebRTCSource

class _Sig:
    channel_name = "camstream"
    def __init__(self, endpoints=None): self._eps = endpoints or {}
    def endpoints(self): return self._eps

def test_connect_requires_a_wss_endpoint():
    s = WebRTCSource(signaling=_Sig(endpoints={}), warmup_frames=1)
    with pytest.raises(RuntimeError, match="WSS"):
        s.connect(timeout_s=0.1)

def test_connect_timeout_names_the_phase():
    s = WebRTCSource(signaling=_Sig(endpoints={"WSS": "wss://example/"}), warmup_frames=1)
    with pytest.raises(TimeoutError) as exc:
        s.connect(timeout_s=0.1)
    assert "signal" in str(exc.value).lower() or "connect" in str(exc.value).lower()
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement `connect()`** — start a background thread running an asyncio loop; inside it open the WSS connection, build an `aiortc.RTCPeerConnection`, send the SDP offer via `encode_sdp_offer`, apply the answer, and attach an `on("track")` handler whose recv loop calls `self.feed(...)`. Any exception inside the loop must call `self.fail(exc)` so it reaches the consumer. Validate the WSS endpoint is present before starting and raise `RuntimeError` mentioning "WSS" if not. Raise `TimeoutError` naming the phase that timed out. Import aiortc/websockets lazily inside this method.

- [ ] **Step 4: Write `src/sfcpi/webrtc/README.md`** documenting: the required IAM permissions (`kinesisvideo:DescribeSignalingChannel`, `GetSignalingChannelEndpoint`, `GetIceServerConfig`), the env vars, a usage snippet, and — stated plainly — that the automated suite covers the plumbing only and the live path requires a manual smoke test with the board streaming.

- [ ] **Step 5: Run the full suite** — `python3 -m pytest tests/sfcpi/ -q`
- [ ] **Step 6: Commit**

---

## Deferred

- Reconnection/backoff, TURN credential refresh, audio, outbound media.
- Swapping `src/dashboard/server.py` from RTSP to this source — a follow-up once the live smoke test passes.
- Live end-to-end verification: **requires the board streaming to a real KVS channel.**
