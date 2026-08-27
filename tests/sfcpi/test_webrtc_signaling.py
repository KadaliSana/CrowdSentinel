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
