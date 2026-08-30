"""Detection metadata sent by the board over the WebRTC data channel.

The AmebaPro2 runs SCRFD on its NPU and draws the boxes into the outgoing
H.264 stream as OSD rectangles. Those are pixels, not numbers -- so the
firmware (`ameba_pro2_media_port.c` -> `master.c`) also emits each inference
as a JSON object on the data channel:

    {"t":<ms>,"w":1280,"h":720,"model":"scrfd",
     "d":[[x,y,w,h,conf_pct], ...], "n":<true count>, "trunc":0|1}

`n` is authoritative. `d` is capped by the firmware
(MEDIA_PORT_METADATA_MAX_DETECTIONS), so on a dense crowd `len(d) < n` and
`trunc` is 1 -- counting the array instead of reading `n` would under-report
exactly the crowds this system exists to detect.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from sfcpi.detect import Detection


@dataclass(frozen=True)
class BoardDetections:
    """One on-device inference."""

    timestamp_ms: int
    frame_width: int
    frame_height: int
    count: int
    detections: List[Detection]
    truncated: bool
    model: str = ""

    @property
    def boxes_are_complete(self) -> bool:
        return not self.truncated and len(self.detections) == self.count


def parse_detection_message(raw) -> BoardDetections:
    """Parse one data-channel message.

    Raises ValueError on anything malformed. It must never degrade to an
    empty result: a parse failure reported as "zero people" is the single
    most dangerous outcome this whole system can produce.
    """
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"detection message is not UTF-8: {exc}") from exc
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty detection message")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed detection message: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"detection message must be a JSON object, got {type(payload).__name__}"
        )

    for key in ("t", "w", "h", "n", "d"):
        if key not in payload:
            raise ValueError(f"detection message missing required field {key!r}")

    count = int(payload["n"])
    if count < 0:
        raise ValueError(f"detection count must not be negative, got {count}")

    raw_boxes = payload["d"]
    if not isinstance(raw_boxes, list):
        raise ValueError("detection field 'd' must be a list")

    detections: List[Detection] = []
    for item in raw_boxes:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            raise ValueError(f"malformed detection entry: {item!r}")
        x, y, w, h, conf = (float(v) for v in item[:5])
        # The firmware sends x,y,w,h in video-frame pixels and confidence as
        # a whole percent; the host works in corners and 0..1.
        detections.append(Detection(x1=x, y1=y, x2=x + w, y2=y + h,
                                    score=conf / 100.0))

    return BoardDetections(
        timestamp_ms=int(payload["t"]),
        frame_width=int(payload["w"]),
        frame_height=int(payload["h"]),
        count=count,
        detections=detections,
        truncated=bool(payload.get("trunc", 0)),
        model=str(payload.get("model", "")),
    )
