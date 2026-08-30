"""The offer must advertise exactly ONE DTLS fingerprint, sha-256.

aiortc puts three `a=fingerprint:` lines in its SDP (sha-256, sha-384,
sha-512). The KVS C SDK master on the AmebaPro2 sizes its fingerprint buffer
for sha-256 and rejects the sha-512 one, which is 191 characters:

    DTLS_VerifyRemoteCertificateFingerprint: invalid input, ...
        CERTIFICATE_FINGERPRINT_LENGTH < fingerprintMaxLen(191)
    OnDtlsHandshakeComplete: Fail to DTLS_VerifyRemoteCertificateFingerprint

The viewer's own DTLS completes, so it looks connected; the board then fails
certificate verification and destroys the connection, and NO media or data
channel ever arrives. Browsers send a single sha-256 line, which is why AWS's
own console viewer worked against the same board throughout.

Verified live against the board, 2026-08-28.
"""
import pytest

from sfcpi.webrtc.signaling import keep_single_fingerprint

THREE = """v=0
o=- 1 2 IN IP4 127.0.0.1
s=-
t=0 0
m=video 9 UDP/TLS/RTP/SAVPF 101
a=mid:0
a=fingerprint:sha-256 AA:BB
a=fingerprint:sha-384 CC:DD
a=fingerprint:sha-512 EE:FF
a=setup:actpass
"""


def test_keeps_only_the_sha256_fingerprint():
    out = keep_single_fingerprint(THREE)
    lines = [l for l in out.splitlines() if l.startswith("a=fingerprint")]
    assert lines == ["a=fingerprint:sha-256 AA:BB"]


def test_everything_else_is_untouched():
    out = keep_single_fingerprint(THREE)
    kept = [l for l in out.splitlines() if not l.startswith("a=fingerprint")]
    original = [l for l in THREE.splitlines() if not l.startswith("a=fingerprint")]
    assert kept == original


def test_line_endings_are_preserved():
    """SDP is CRLF on the wire; rewriting it must not strip the CR."""
    crlf = THREE.replace("\n", "\r\n")
    out = keep_single_fingerprint(crlf)
    # every LF still preceded by a CR, and exactly the two dropped lines gone
    assert out.count("\n") == out.count("\r\n")
    assert out.count("\r\n") == crlf.count("\r\n") - 2


def test_a_single_sha256_offer_is_unchanged():
    one = THREE.replace("a=fingerprint:sha-384 CC:DD\n", "").replace(
        "a=fingerprint:sha-512 EE:FF\n", "")
    assert keep_single_fingerprint(one) == one


def test_per_m_section_fingerprints_each_keep_one():
    """With BUNDLE plus a data channel there is a fingerprint per m= section;
    each section must keep its own sha-256, not lose one to the other."""
    two = THREE + """m=application 9 UDP/DTLS/SCTP webrtc-datachannel
a=mid:1
a=fingerprint:sha-256 11:22
a=fingerprint:sha-512 33:44
"""
    lines = [l for l in keep_single_fingerprint(two).splitlines()
             if l.startswith("a=fingerprint")]
    assert lines == ["a=fingerprint:sha-256 AA:BB", "a=fingerprint:sha-256 11:22"]


def test_raises_when_no_sha256_is_present():
    """Silently sending an offer with no fingerprint would fail DTLS in a far
    more confusing way than an explicit error here."""
    none = THREE.replace("a=fingerprint:sha-256 AA:BB\n", "")
    with pytest.raises(ValueError, match="sha-256"):
        keep_single_fingerprint(none)
