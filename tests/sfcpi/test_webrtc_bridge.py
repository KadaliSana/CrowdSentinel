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
