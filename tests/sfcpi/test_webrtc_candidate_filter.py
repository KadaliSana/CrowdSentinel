"""Drop remote host candidates we could never reach.

Observed live:

    aioice.stun.TransactionFailed: STUN transaction failed (403 - Forbidden IP)
      ... TurnClientMixin.send_data -> channel_bind

AWS's TURN service refuses to open a relay channel to an unroutable peer
address. The board advertises a private host candidate (10.193.209.62) while
this host sits on 172.16.x -- different subnets, so that candidate is useless
to us, and handing it to TURN produces the 403 plus an unretrieved-task
traceback in the log.

The AWS JS SDK treats candidate filtering as a first-class control
(shouldSendIceCandidate: sendHostCandidates / sendSrflxCandidates /
sendRelayCandidates), so filtering is supported practice, not a hack.

The filter is deliberately NARROW: only private HOST candidates on a subnet we
have no local address in. A same-LAN board keeps its host candidate, which is
the fastest path available and must not be thrown away.
"""
import pytest

from sfcpi.webrtc.source import is_reachable_candidate

LOCAL = ["172.16.144.16", "192.168.122.1", "127.0.0.1"]


def _cand(ip, typ="host"):
    return f"candidate:1 1 udp 2130706431 {ip} 5000 typ {typ}"


def test_private_host_candidate_on_a_foreign_subnet_is_dropped():
    """The exact case that produced 403 Forbidden IP."""
    assert is_reachable_candidate(_cand("10.193.209.62"), LOCAL) is False


def test_private_host_candidate_on_our_own_subnet_is_kept():
    """A same-LAN board: this is the FASTEST path and must survive."""
    assert is_reachable_candidate(_cand("172.16.144.99"), LOCAL) is True


def test_srflx_candidates_are_always_kept():
    assert is_reachable_candidate(_cand("182.66.218.119", "srflx"), LOCAL) is True


def test_relay_candidates_are_always_kept():
    assert is_reachable_candidate(_cand("13.235.50.14", "relay"), LOCAL) is True


def test_public_host_candidates_are_kept():
    """A publicly routable host address is reachable; only private ones are
    subnet-checked."""
    assert is_reachable_candidate(_cand("3.110.158.91"), LOCAL) is True


def test_an_unparseable_candidate_is_kept():
    """Never drop something we merely failed to understand -- ICE can afford a
    useless candidate far better than a missing one."""
    assert is_reachable_candidate("total nonsense", LOCAL) is True


def test_no_local_addresses_known_keeps_everything():
    assert is_reachable_candidate(_cand("10.193.209.62"), []) is True
