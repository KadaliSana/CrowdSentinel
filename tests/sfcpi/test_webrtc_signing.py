from datetime import datetime
from urllib.parse import parse_qs, urlsplit

import botocore.auth
import pytest
from botocore.credentials import Credentials

from sfcpi.webrtc.signaling import sign_wss_url

WSS_ENDPOINT = "wss://m-1234abcd.kinesisvideo.ap-south-1.amazonaws.com"
CHANNEL_ARN = (
    "arn:aws:kinesisvideo:ap-south-1:123456789012:channel/my-channel/1234567890123"
)
FROZEN_TIMESTAMP = datetime(2026, 1, 1, 12, 0, 0)

REQUIRED_PRESIGN_PARAMS = {
    "X-Amz-Algorithm",
    "X-Amz-Credential",
    "X-Amz-Date",
    "X-Amz-Expires",
    "X-Amz-SignedHeaders",
    "X-Amz-Signature",
}


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch):
    """Freeze the clock that botocore.auth.SigV4Auth.add_auth() reads.

    sign_wss_url() now calls the PUBLIC SigV4QueryAuth.add_auth() entry
    point in production, which always timestamps from the real current time
    (there is no timestamp override parameter on the public API). Determinism
    for tests comes from patching the clock botocore itself reads --
    `get_current_datetime`, imported into the `botocore.auth` module
    namespace from `botocore.compat` and looked up there as a module global
    at call time -- rather than from a parameter threaded through
    sign_wss_url. This keeps the internals dependency in the TEST, where a
    botocore upgrade that renames/removes it breaks loudly in CI instead of
    silently in production.
    """
    monkeypatch.setattr(
        botocore.auth, "get_current_datetime", lambda *a, **k: FROZEN_TIMESTAMP
    )


def _creds(token=None):
    return Credentials("AKIDEXAMPLE", "secret", token)


def _query(url):
    """Parse the query string of a signed URL into {key: single-value}."""
    parts = urlsplit(url)
    parsed = parse_qs(parts.query, keep_blank_values=True)
    return {k: v[0] for k, v in parsed.items()}


def test_keeps_wss_scheme_and_original_host_and_path():
    signed = sign_wss_url(
        WSS_ENDPOINT, CHANNEL_ARN, "ap-south-1", _creds()
    )
    original = urlsplit(WSS_ENDPOINT)
    result = urlsplit(signed)
    assert result.scheme == "wss"
    assert result.netloc == original.netloc
    # A bare host with no path signs/round-trips as "/".
    assert result.path in ("", "/")


def test_all_six_presign_params_present():
    signed = sign_wss_url(
        WSS_ENDPOINT, CHANNEL_ARN, "ap-south-1", _creds()
    )
    params = _query(signed)
    missing = REQUIRED_PRESIGN_PARAMS - params.keys()
    assert not missing, f"missing presign params: {missing}"


def test_channel_arn_present_and_correctly_percent_encoded():
    signed = sign_wss_url(
        WSS_ENDPOINT, CHANNEL_ARN, "ap-south-1", _creds()
    )
    # The raw query string must NOT contain the ARN's literal ':' or '/' --
    # both must have been percent-encoded (%3A / %2F).
    raw_query = urlsplit(signed).query
    assert "X-Amz-ChannelARN=" in raw_query
    arn_field = [p for p in raw_query.split("&") if p.startswith("X-Amz-ChannelARN=")][0]
    assert "%3A" in arn_field
    assert "%2F" in arn_field
    assert ":" not in arn_field
    # parse_qs decodes percent-encoding back to the original value.
    params = _query(signed)
    assert params["X-Amz-ChannelARN"] == CHANNEL_ARN


def test_client_id_present_when_given():
    signed = sign_wss_url(
        WSS_ENDPOINT,
        CHANNEL_ARN,
        "ap-south-1",
        _creds(),
        client_id="viewer-abc",
    )
    params = _query(signed)
    assert params.get("X-Amz-ClientId") == "viewer-abc"


def test_client_id_absent_when_not_given():
    signed = sign_wss_url(
        WSS_ENDPOINT, CHANNEL_ARN, "ap-south-1", _creds()
    )
    params = _query(signed)
    assert "X-Amz-ClientId" not in params


def test_signing_is_deterministic_for_a_frozen_timestamp():
    first = sign_wss_url(
        WSS_ENDPOINT,
        CHANNEL_ARN,
        "ap-south-1",
        _creds(),
        client_id="viewer-abc",
    )
    second = sign_wss_url(
        WSS_ENDPOINT,
        CHANNEL_ARN,
        "ap-south-1",
        _creds(),
        client_id="viewer-abc",
    )
    assert first == second


def test_changing_channel_arn_changes_the_signature():
    """The test that proves this isn't just returning a fixed string."""
    other_arn = (
        "arn:aws:kinesisvideo:ap-south-1:123456789012:channel/other-channel/999"
    )
    signed_a = sign_wss_url(
        WSS_ENDPOINT, CHANNEL_ARN, "ap-south-1", _creds()
    )
    signed_b = sign_wss_url(
        WSS_ENDPOINT, other_arn, "ap-south-1", _creds()
    )
    sig_a = _query(signed_a)["X-Amz-Signature"]
    sig_b = _query(signed_b)["X-Amz-Signature"]
    assert sig_a != sig_b


def test_session_token_appears_as_security_token_param():
    signed = sign_wss_url(
        WSS_ENDPOINT,
        CHANNEL_ARN,
        "ap-south-1",
        _creds(token="FwoGZXIvYXdzEB..."),
    )
    params = _query(signed)
    assert params.get("X-Amz-Security-Token") == "FwoGZXIvYXdzEB..."


def test_no_session_token_means_no_security_token_param():
    signed = sign_wss_url(
        WSS_ENDPOINT, CHANNEL_ARN, "ap-south-1", _creds()
    )
    params = _query(signed)
    assert "X-Amz-Security-Token" not in params


def test_empty_wss_endpoint_raises_value_error_naming_the_field():
    with pytest.raises(ValueError, match="wss_endpoint"):
        sign_wss_url("", CHANNEL_ARN, "ap-south-1", _creds())


def test_empty_channel_arn_raises_value_error_naming_the_field():
    with pytest.raises(ValueError, match="channel_arn"):
        sign_wss_url(WSS_ENDPOINT, "", "ap-south-1", _creds())


def test_canary_botocore_auth_still_exposes_get_current_datetime():
    """Guardrail for the ONE botocore internal this test suite still touches.

    Production code (sign_wss_url) only calls the PUBLIC add_auth() now. The
    `frozen_clock` fixture above is the sole remaining dependency on a
    botocore internal (`botocore.auth.get_current_datetime`, used to freeze
    add_auth()'s timestamp for deterministic tests). If a botocore upgrade
    renames or removes it, this fails loudly and specifically here instead of
    every other signing test failing with a confusing non-deterministic
    diff.
    """
    assert hasattr(botocore.auth, "get_current_datetime")
    assert callable(botocore.auth.get_current_datetime)
