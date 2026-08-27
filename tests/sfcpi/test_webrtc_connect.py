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
