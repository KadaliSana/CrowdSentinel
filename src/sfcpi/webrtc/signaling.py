"""KVS WebRTC signalling client.

AWS ships KVS WebRTC SDKs for C, browser JS, Android and iOS but no official
Python viewer SDK. This module writes the piece that SDK would otherwise
provide: channel endpoint discovery (via the injected boto3 `kinesisvideo`
client) and the SDP/ICE message envelope used on the signalling channel's
WebSocket connection.

The boto3 client is always injected by the caller -- this module never
constructs one, which is what makes it testable without network access or
AWS credentials.
"""
from __future__ import annotations

import base64
import binascii
import json


class KvsSignalingClient:
    """Discovers a KVS signalling channel's ARN and protocol endpoints."""

    def __init__(self, channel_name, region, kinesisvideo_client, role="VIEWER"):
        if not channel_name:
            raise ValueError("channel_name must be a non-empty string")
        self.channel_name = channel_name
        self.region = region
        self._client = kinesisvideo_client
        self.role = role

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
        return response["ChannelInfo"]

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
