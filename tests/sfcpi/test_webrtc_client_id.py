"""The viewer's client id must be STABLE across reconnects.

The board keys its session table on the remote client id
(app_common.c, AppCommon_GetPeerConnectionSession):

    if( appSessions[i].remoteClientIdLength == remoteClientIdLength &&
        strncmp( appSessions[i].remoteClientId, pRemoteClientId, ... ) == 0 )
        /* Found existing session. */

A fresh uuid4() per connect therefore presents as a NEW viewer every time and
consumes another of the board's AWS_MAX_VIEWER_NUM (=2) slots, which are only
released by its own close timer. Reconnect a few times and both slots are held
by ghosts of this same process: signalling still answers, but no session is
free to carry media.
"""
from sfcpi.webrtc.source import default_client_id


def test_client_id_is_stable_across_calls():
    assert default_client_id() == default_client_id()


def test_client_id_is_stable_across_processes():
    """It must survive a restart, or a crashed run still burns a slot."""
    import subprocess
    import sys

    code = ("import sys; sys.path.insert(0, 'src'); "
            "from sfcpi.webrtc.source import default_client_id; "
            "print(default_client_id())")
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=".").stdout.strip() for _ in range(2)}
    assert len(runs) == 1 and runs != {""}


def test_client_id_is_kvs_safe():
    """KVS client ids allow letters, digits, underscore, hyphen and period."""
    import re

    assert re.fullmatch(r"[A-Za-z0-9_.-]{1,256}", default_client_id())


def test_distinct_hosts_get_distinct_ids():
    """Two machines viewing the same board must not collide onto one slot."""
    from sfcpi.webrtc import source

    a = source.default_client_id(host="alpha")
    b = source.default_client_id(host="beta")
    assert a != b
