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
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from botocore.auth import SigV4QueryAuth
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
    """Build the SDP_OFFER envelope a VIEWER sends over the signalling socket.

    NO `recipientClientId`. The AWS JS SDK documents it as "Required for
    'MASTER' role. Should not be present for 'VIEWER' role" and enforces that
    in `validateRecipientClientId()`, which THROWS if a viewer supplies one;
    `examples/viewer.js` calls `sendSdpOffer(localDescription)` with no
    recipient. We previously sent our OWN client id there, which addresses the
    offer to ourselves. The viewer identifies itself in the signed WSS URL
    (`X-Amz-ClientId`), so `client_id` is accepted here only to keep the call
    signature stable.
    """
    del client_id  # identity travels in the signed URL, not the envelope
    payload = json.dumps({"type": "offer", "sdp": sdp}).encode("utf-8")
    message = {
        "action": "SDP_OFFER",
        "messagePayload": base64.b64encode(payload).decode("ascii"),
    }
    return json.dumps(message)


def ice_servers_to_rtc(entries: list) -> list[dict]:
    """Convert a KVS `IceServerList` into aiortc `RTCIceServer(**kwargs)` dicts.

    KVS names the fields `Uris` / `Username` / `Password` / `Ttl`; aiortc
    wants `urls` / `username` / `credential`. Absent credentials are OMITTED
    rather than passed as None -- a STUN entry with `username=None,
    credential=None` is not the same thing to aiortc as a STUN entry with
    neither.

    `Ttl` is dropped: these credentials expire, and nothing in this package
    renews them (see the README's Deferred list).
    """
    out = []
    for entry in entries or []:
        item = {"urls": entry.get("Uris", [])}
        if entry.get("Username"):
            item["username"] = entry["Username"]
        if entry.get("Password"):
            item["credential"] = entry["Password"]
        out.append(item)
    return out


def keep_single_fingerprint(sdp: str, algorithm: str = "sha-256") -> str:
    """Drop every `a=fingerprint:` line except `algorithm`, per m= section.

    aiortc advertises three fingerprints (sha-256, sha-384, sha-512). The KVS
    C SDK master on the AmebaPro2 sizes its fingerprint buffer for sha-256 and
    REJECTS the sha-512 line, which is 191 characters:

        DTLS_VerifyRemoteCertificateFingerprint: invalid input, ...
            CERTIFICATE_FINGERPRINT_LENGTH < fingerprintMaxLen(191)
        OnDtlsHandshakeComplete: Fail to ... with return 255

    The failure is silent from the viewer's side -- OUR DTLS handshake
    completes, so the connection reports `connected`, and only the board
    knows it then failed certificate verification and tore the session down.
    No media and no data channel ever arrive. A browser sends one sha-256
    line, which is why AWS's own console viewer worked against the same board.

    Dropping the extra lines is safe: they are alternative digests of the
    SAME certificate, and RFC 8122 requires only that the peer be able to
    verify one of them.

    Raises ValueError if no line for `algorithm` exists -- an offer with no
    fingerprint at all would fail DTLS far more confusingly downstream.
    """
    prefix = f"a=fingerprint:{algorithm} "
    if prefix not in sdp:
        raise ValueError(
            f"SDP carries no {algorithm} fingerprint; refusing to send an "
            f"offer whose DTLS certificate cannot be verified"
        )

    out = []
    seen_in_section = False
    for line in sdp.split("\n"):
        stripped = line.rstrip("\r")
        if stripped.startswith("m="):
            seen_in_section = False
        if stripped.startswith("a=fingerprint:"):
            # Keep the first `algorithm` line of each m= section and drop the
            # rest; a BUNDLE offer with a data channel carries one set per
            # section, and each section needs its own.
            if stripped.startswith(prefix) and not seen_in_section:
                seen_in_section = True
            else:
                continue
        out.append(line)
    return "\n".join(out)


def is_keepalive(raw) -> bool:
    """True if `raw` is a KVS signalling keepalive rather than a message.

    Observed against a real KVS channel: the signalling service sends
    zero-length WebSocket frames -- the first one arrives immediately after
    the SDP_OFFER, before the answer, and more arrive during an idle
    session. They carry no envelope, so `decode_message` correctly rejects
    them as malformed; the recv loops must skip them instead of feeding
    them in.

    Deliberately narrow: ONLY an empty-or-whitespace frame counts. A
    non-empty frame that merely fails to parse is NOT a keepalive -- it is
    corruption, and it must keep reaching `decode_message`'s ValueError
    rather than being silently discarded, which is precisely the failure
    this predicate must not become a blanket except-clause for. Accepts
    `str` or `bytes` because a WebSocket peer chooses the frame type.
    """
    if raw is None:
        return True
    if isinstance(raw, (bytes, bytearray)):
        return not raw.strip()
    return not str(raw).strip()


@dataclass(frozen=True)
class SdpCandidate:
    """One `a=candidate:` line, with the m= section it belongs to."""

    candidate: str
    sdp_mid: str | None
    sdp_mline_index: int


def sdp_candidate_lines(sdp: str) -> list[SdpCandidate]:
    """Pull the ICE candidates out of a local SDP, per m= section.

    A candidate is attributed to the m= section it FOLLOWS -- sending them
    all under `sdpMLineIndex: 0` would misattribute every candidate of every
    section after the first. The leading `a=` is stripped: it is SDP
    framing, and the signalling payload carries the bare `candidate:...`
    string (the same form the far side sends us).
    """
    out: list[SdpCandidate] = []
    index = -1
    mid: str | None = None
    for line in sdp.splitlines():
        line = line.strip()
        if line.startswith("m="):
            index += 1
            mid = None
        elif line.startswith("a=mid:"):
            mid = line[len("a=mid:"):].strip()
        elif line.startswith("a=candidate:"):
            if index < 0:
                # A candidate before any m= section is malformed SDP; there
                # is no section to attribute it to, so skip rather than
                # invent index 0.
                continue
            out.append(SdpCandidate(line[2:], mid, index))
    return out


def encode_ice_candidate(
    candidate: str, sdp_mid: str | None, sdp_mline_index: int
) -> str:
    """Build the ICE_CANDIDATE message a viewer sends to the master.

    Required, not optional: the KVS C SDK master does not adopt candidates
    from the offer SDP alone. Without these messages it forms no valid
    candidate pair, never sends the DTLS ClientHello (it answers
    `a=setup:active`, making it the DTLS client), and drops the session
    after ~30s -- verified live against a real channel.

    Outgoing messages use `action`; incoming ones use `messageType` (see
    `decode_message`). A viewer omits `recipientClientId`: the master is the
    implicit recipient, which is how the offer itself is routed.
    """
    payload = json.dumps({
        "candidate": candidate,
        "sdpMid": sdp_mid,
        "sdpMLineIndex": sdp_mline_index,
    }).encode("utf-8")
    return json.dumps({
        "action": "ICE_CANDIDATE",
        "messagePayload": base64.b64encode(payload).decode("ascii"),
    })


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

    # STATUS_RESPONSE carries a `statusResponse` object and NO messagePayload.
    # The SDK dispatches it as a first-class message type (SignalingClient.ts);
    # treating it as corruption would discard the service's own error reports,
    # which are exactly what you want when a session misbehaves.
    if envelope.get("statusResponse") is not None and not envelope.get("messagePayload"):
        return message_type, sender_client_id, dict(envelope["statusResponse"])

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
) -> str:
    """SigV4 *query-string* (presigned) sign a KVS viewer WebSocket URL.

    A WebSocket handshake can't carry a normal `Authorization` header, so KVS
    (like a presigned S3 URL) authenticates the connection via signed query
    parameters instead: `X-Amz-ChannelARN` (and, for a VIEWER, also
    `X-Amz-ClientId`) are carried as parameters to sign, and presigning adds
    `X-Amz-Algorithm`, `X-Amz-Credential`, `X-Amz-Date`, `X-Amz-Expires`,
    `X-Amz-SignedHeaders` and `X-Amz-Signature` (plus `X-Amz-Security-Token`
    if `credentials` carries a session token).

    Signing is delegated to `botocore.auth.SigV4QueryAuth.add_auth()` -- the
    library's PUBLIC entry point, not its internal signing steps.
    Canonicalisation and percent-encoding are easy to get subtly wrong by
    hand, and calling internal helpers (`_modify_request_before_signing`,
    `canonical_request`, `string_to_sign`, `signature`,
    `_inject_signature_to_request`) directly ties this module to botocore
    internals that carry no compatibility guarantee -- a botocore upgrade
    could change or reorder them and this would break silently in production
    (the one path with no test coverage), while any test built against the
    same internals would happily stay green. `add_auth()` is public and
    stable; the only friction it adds is that it always timestamps the
    request from the real current time with no override parameter. Tests get
    a deterministic signature by freezing the clock botocore itself reads
    (patch `botocore.auth.get_current_datetime`), not by threading a
    timestamp through this function -- see `tests/sfcpi/test_webrtc_signing.py`.

    The signature covers the request's host and path, not its scheme, so
    this signs the `https://` form of `wss_endpoint` and returns the result
    with the `wss://` scheme restored.

    `credentials` is a botocore-style credentials object (`.access_key`,
    `.secret_key`, and optional `.token`) -- always injected, never fetched
    here.
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
    auth = SigV4QueryAuth(credentials, "kinesisvideo", region, expires=expires)
    auth.add_auth(request)

    signed = urlsplit(request.url)
    return urlunsplit(("wss", signed.netloc, signed.path, signed.query, ""))
