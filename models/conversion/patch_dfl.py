"""Export a YOLO26 model with its detection head left RAW (undecoded).

Why this exists
---------------
A stock `model.export(format="onnx")` on YOLO26 emits an already-decoded head:
boxes in input pixels, either (1, 4+nc, A) or (1, 300, 6). The AMB Pro2 firmware
decoder in `model_yolo26.c` does NOT want that. It expects the raw per-stride
head and applies the decode itself:

    x1 = (i + 0.5 - l) * stride / input_width      # l,t,r,b in GRID units

so it needs the six raw tensors, which it auto-detects as the SPLIT layout:

    cv2.{0,1,2} -> (4, g, g)   box distances, grid units    g = 52, 26, 13
    cv3.{0,1,2} -> (nc, g, g)  class logits (sigmoid applied on-device)

Bypassing `Detect.forward` is what produces that shape. Note the filename is a
slight misnomer: YOLO26 is already DFL-free, so `cv2[i]` emits 4 channels
directly rather than the 64 (4 x reg_max=16) a YOLOv8 head would. The point is
skipping the decode + concat, not removing DFL.

Verified: feeding these tensors through the firmware's formula reproduces the
decoded reference export's boxes exactly.
"""

import argparse
import os

from ultralytics import YOLO

# models/weights/best.pt is the canonical model: stripped yolo26n, mAP@.5 = 0.336 on
# the validation set vs 0.285 for the older YOLO26_Training run (since removed).
DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "weights", "best.pt"
)


def export_raw_head(weights: str, imgsz: int, opset: int) -> None:
    model = YOLO(weights)
    detect_head = model.model.model[-1]

    def raw_forward(x):
        """Return the box/cls branches per stride, skipping decode and concat."""
        out = []
        for i in range(detect_head.nl):
            out.append(detect_head.cv2[i](x[i]))  # box distances
            out.append(detect_head.cv3[i](x[i]))  # class logits
        return tuple(out)

    detect_head.forward = raw_forward

    print(f"[1/1] Exporting raw-head ONNX from {weights} (imgsz={imgsz}, opset={opset})")
    path = model.export(format="onnx", opset=opset, simplify=True, dynamic=False, imgsz=imgsz)
    print(f"[OK] Wrote {path}")
    print("     Expect 6 outputs: (4,52,52) (nc,52,52) (4,26,26) (nc,26,26) (4,13,13) (nc,13,13)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="path to the .pt checkpoint")
    # imgsz must match NN_WIDTH/NN_HEIGHT in the firmware, and opset 12 is the NPU ceiling.
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()
    export_raw_head(args.weights, args.imgsz, args.opset)
