"""KVS WebRTC signalling client.

AWS ships KVS WebRTC SDKs for C, browser JS, Android and iOS but no official
Python viewer SDK. This module writes the piece that SDK would otherwise
provide: channel endpoint discovery (via the injected boto3 `kinesisvideo`
client), the SDP/ICE message envelope used on the signalling channel's
WebSocket connection, and SigV4 query-string signing of that WebSocket URL.

The boto3 client is always injected by the caller -- this module never
constructs one, which is what makes it testable without network access or
AWS credentials. `sign_wss_url` uses `botocore.auth`/`botocore.awsrequest`
directly (both pure, no network) rather than importing boto3 itself, for the
same reason.
"""
from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from botocore.auth import SIGV4_TIMESTAMP, SigV4QueryAuth
from botocore.awsrequest import AWSRequest


class KvsSignalingClient:
    """Discovers a KVS signalling channel's ARN and protocol endpoints."""

    def __init__(self, channel_name, region, kinesisvideo_client, role="VIEWER"):
        if not channel_name:
            raise ValueError("channel_name must be a non-empty string")
        self.channel_name = channel_name
        self.region = region
        self._client = kinesisvideo_client
        self.role = role
        self._channel_arn: str | None = None

    @property
    def channel_arn(self) -> str | None:
        """The `ChannelARN` from the most recent `describe()` call, cached.

        `None` until `describe()` (directly, or indirectly via `endpoints()`)
        has actually run -- this is a cache of a real lookup, not a
        rediscovery of the ARN, so it never makes its own network call.
        """
        return self._channel_arn

    def describe(self) -> dict:
        """Return the `ChannelInfo` dict (contains `ChannelARN`)."""
        try:
            response = self._client.describe_signaling_channel(
                ChannelName=self.channel_name
            )
        except Exception as exc:
            code = _error_code(exc)
            if code == "ResourceNotFoundException":
                raise LookupError(
                    f"KVS signalling channel {self.channel_name!r} not found"
                ) from exc
            raise
        info = response["ChannelInfo"]
        self._channel_arn = info.get("ChannelARN")
        return info

    def endpoints(self) -> dict:
        """Return endpoint URLs keyed by protocol, e.g. {"WSS": ..., "HTTPS": ...}."""
        arn = self.describe()["ChannelARN"]
        response = self._client.get_signaling_channel_endpoint(
            ChannelARN=arn,
            SingleMasterChannelEndpointConfiguration={
                "Protocols": ["WSS", "HTTPS"],
                "Role": self.role,
            },
        )
        return {
            entry["Protocol"]: entry["ResourceEndpoint"]
            for entry in response["ResourceEndpointList"]
        }

    def ice_servers(self, signaling_client) -> list[dict]:
        """Return the ICE server list from a KVS `signaling_client`.

        `signaling_client` is expected to be a boto3
        `kinesis-video-signaling` client (a separate service/client from the
        `kinesisvideo` client used for discovery), exposing
        `get_ice_server_config(ChannelARN=...)`.
        """
        arn = self.describe()["ChannelARN"]
        response = signaling_client.get_ice_server_config(ChannelARN=arn)
        return response.get("IceServerList", [])


def _error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    return response.get("Error", {}).get("Code")


def encode_sdp_offer(client_id: str, sdp: str) -> str:
    """Build the SDP_OFFER message envelope sent over the signalling WebSocket."""
    payload = json.dumps({"type": "offer", "sdp": sdp}).encode("utf-8")
    message = {
        "action": "SDP_OFFER",
        "recipientClientId": client_id,
        "messagePayload": base64.b64encode(payload).decode("ascii"),
    }
    return json.dumps(message)


def decode_message(raw: str) -> tuple[str, str, dict]:
    """Reverse the signalling message envelope.

    Returns (messageType, senderClientId, decoded_payload_dict). Raises
    ValueError (mentioning "payload") if the outer JSON, the base64
    encoding, or the inner JSON payload is malformed.
    """
    try:
        envelope = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"malformed signalling message payload (outer JSON): {exc}") from exc

    message_type = envelope.get("messageType")
    sender_client_id = envelope.get("senderClientId")
    raw_payload = envelope.get("messagePayload", "")

    try:
        decoded_bytes = base64.b64decode(raw_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"malformed base64 in message payload: {exc}") from exc

    try:
        payload = json.loads(decoded_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed JSON in message payload: {exc}") from exc

    return message_type, sender_client_id, payload


def sign_wss_url(
    wss_endpoint: str,
    channel_arn: str,
    region: str,
    credentials: Any,
    client_id: str | None = None,
    expires: int = 299,
    timestamp: datetime | None = None,
) -> str:
    """SigV4 *query-string* (presigned) sign a KVS viewer WebSocket URL.

    A WebSocket handshake can't carry a normal `Authorization` header, so KVS
    (like a presigned S3 URL) authenticates the connection via signed query
    parameters instead: `X-Amz-ChannelARN` (and, for a VIEWER, also
    `X-Amz-ClientId`) are carried as parameters to sign, and presigning adds
    `X-Amz-Algorithm`, `X-Amz-Credential`, `X-Amz-Date`, `X-Amz-Expires`,
    `X-Amz-SignedHeaders` and `X-Amz-Signature` (plus `X-Amz-Security-Token`
    if `credentials` carries a session token).

    Signing is delegated to `botocore.auth.SigV4QueryAuth` -- canonicalisation
    and percent-encoding are easy to get subtly wrong by hand. The signature
    covers the request's host and path, not its scheme, so this signs the
    `https://` form of `wss_endpoint` and returns the result with the `wss://`
    scheme restored.

    `credentials` is a botocore-style credentials object (`.access_key`,
    `.secret_key`, and optional `.token`) -- always injected, never fetched
    here. `timestamp` is injectable (defaults to now, UTC) so callers/tests
    get a deterministic, reproducible signature.
    """
    if not wss_endpoint:
        raise ValueError("wss_endpoint must be a non-empty string")
    if not channel_arn:
        raise ValueError("channel_arn must be a non-empty string")

    parts = urlsplit(wss_endpoint)
    https_url = urlunsplit(("https", parts.netloc, parts.path or "/", "", ""))

    params = {"X-Amz-ChannelARN": channel_arn}
    if client_id:
        params["X-Amz-ClientId"] = client_id

    request = AWSRequest(method="GET", url=https_url, params=params)
    ts = timestamp if timestamp is not None else datetime.utcnow()
    # SigV4Auth.add_auth() (the usual entry point) always stamps
    # request.context['timestamp'] from datetime.now() itself, with no way to
    # override it -- so this calls the same steps add_auth() would, in order,
    # but with an injected timestamp, to keep signing deterministic for tests.
    request.context["timestamp"] = ts.strftime(SIGV4_TIMESTAMP)

    auth = SigV4QueryAuth(credentials, "kinesisvideo", region, expires=expires)
    auth._modify_request_before_signing(request)
    canonical_request = auth.canonical_request(request)
    string_to_sign = auth.string_to_sign(request, canonical_request)
    signature = auth.signature(string_to_sign, request)
    auth._inject_signature_to_request(request, signature)

    signed = urlsplit(request.url)
    return urlunsplit(("wss", signed.netloc, signed.path, signed.query, ""))
