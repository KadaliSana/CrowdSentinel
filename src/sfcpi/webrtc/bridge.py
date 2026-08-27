"""Bounded async->sync handoff.

aiortc delivers frames on an asyncio loop; FrameSource is a synchronous iterator.
This is the boundary, and it is where deadlocks and leaks live, so the rules are
explicit: bounded, drop-oldest (never block the loop), sentinel-terminated,
idempotent close, and a consumer-side timeout so a dead connection surfaces as an
error instead of a hang.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Iterator, Optional

_SENTINEL = object()


class FrameBridge:
    def __init__(self, maxsize: int = 8, timeout_s: float = 10.0) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        # Internal capacity is maxsize+1: the extra slot is reserved for the
        # close() sentinel so shutdown never has to evict a real data item
        # (and therefore never over-counts .dropped) to make room for it.
        self._maxsize = maxsize
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize + 1)
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._closed = False
        self._error: Optional[BaseException] = None
        self.dropped = 0

    @property
    def closed(self) -> bool:
        return self._closed

    def put(self, item: Any) -> None:
        """Never blocks. When full, discards the OLDEST item and counts it."""
        if self._closed:
            return
        while True:
            try:
                if self._q.qsize() >= self._maxsize:
                    raise queue.Full
                self._q.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self.dropped += 1
                except queue.Empty:  # pragma: no cover - racy drain
                    pass

    def fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
                self._q.put_nowait(_SENTINEL)
            except (queue.Empty, queue.Full):  # pragma: no cover
                pass

    def __iter__(self) -> Iterator[Any]:
        while True:
            try:
                item = self._q.get(timeout=self._timeout_s)
            except queue.Empty:
                raise TimeoutError(
                    f"no frame within {self._timeout_s}s; the producer is silent or dead"
                ) from None
            if item is _SENTINEL:
                if self._error is not None:
                    raise self._error
                return
            yield item
