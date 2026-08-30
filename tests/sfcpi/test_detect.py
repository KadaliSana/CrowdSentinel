import numpy as np
import pytest
from sfcpi.detect import Detection, FixedDetector, YoloDetector, counts_per_cell
from sfcpi.grid import CellGrid

def test_centroid():
    d = Detection(x1=0, y1=0, x2=10, y2=20, score=0.9)
    assert d.centroid == (5.0, 10.0)

def test_counts_land_in_the_right_cells():
    grid = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    dets = [
        Detection(0, 0, 2, 2, 0.9),     # centroid (1,1)   -> cell (0,0)
        Detection(2, 0, 4, 2, 0.9),     # centroid (3,1)   -> cell (0,1)
        Detection(2, 2, 4, 4, 0.9),     # centroid (3,3)   -> cell (1,1)
    ]
    counts = counts_per_cell(dets, grid)
    assert counts.tolist() == [[1.0, 1.0], [0.0, 1.0]]

def test_counts_shape_and_empty_case():
    grid = CellGrid(frame_width=8, frame_height=4, cell_size=2)
    counts = counts_per_cell([], grid)
    assert counts.shape == (2, 4)
    assert counts.sum() == 0.0

def test_detections_outside_frame_are_ignored():
    grid = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    counts = counts_per_cell([Detection(100, 100, 110, 110, 0.9)], grid)
    assert counts.sum() == 0.0

def test_fixed_detector_returns_what_it_was_given():
    dets = [Detection(0, 0, 1, 1, 0.5)]
    assert FixedDetector(dets).detect(np.zeros((4, 4, 3), np.uint8)) == dets

def test_centroid_on_right_edge_is_dropped_not_clamped():
    grid = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    # x1=x2=4 -> centroid x == frame_width exactly (on the right edge)
    counts = counts_per_cell([Detection(4, 0, 4, 2, 0.9)], grid)
    assert counts.sum() == 0.0

def test_centroid_on_bottom_edge_is_dropped_not_clamped():
    grid = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    # y1=y2=4 -> centroid y == frame_height exactly (on the bottom edge)
    counts = counts_per_cell([Detection(0, 4, 2, 4, 0.9)], grid)
    assert counts.sum() == 0.0


class _Arr:
    """Minimal stand-in for a torch tensor: supports .cpu().numpy() and len()."""

    def __init__(self, data):
        self._data = np.asarray(data, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def __len__(self):
        return len(self._data)


class _Boxes:
    def __init__(self, xyxy=None, conf=None):
        self.xyxy = _Arr(xyxy) if xyxy is not None else None
        self.conf = _Arr(conf) if conf is not None else None


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _StubModel:
    def __init__(self, results):
        self._results = results

    def __call__(self, image, conf=None, classes=None, verbose=False):
        return self._results


def _detector_with_stub(results):
    det = YoloDetector(model_path="unused.pt")
    det._model = _StubModel(results)
    return det


def test_yolo_detector_raises_on_empty_results():
    det = _detector_with_stub([])
    with pytest.raises(RuntimeError):
        det.detect(np.zeros((4, 4, 3), np.uint8))


def test_yolo_detector_raises_when_boxes_is_none():
    det = _detector_with_stub([_Result(boxes=None)])
    with pytest.raises(RuntimeError):
        det.detect(np.zeros((4, 4, 3), np.uint8))


def test_yolo_detector_raises_when_xyxy_is_none():
    det = _detector_with_stub([_Result(boxes=_Boxes(xyxy=None))])
    with pytest.raises(RuntimeError):
        det.detect(np.zeros((4, 4, 3), np.uint8))


def test_yolo_detector_empty_xyxy_array_is_a_legitimate_zero():
    # A real, well-formed result with zero rows means "model ran, found nobody" --
    # this must NOT raise, unlike the malformed cases above.
    boxes = _Boxes(xyxy=np.zeros((0, 4)))
    det = _detector_with_stub([_Result(boxes=boxes)])
    result = det.detect(np.zeros((4, 4, 3), np.uint8))
    assert result == []


def test_yolo_detector_returns_detection_for_one_box():
    boxes = _Boxes(xyxy=np.array([[1.0, 2.0, 3.0, 4.0]]), conf=np.array([0.75]))
    det = _detector_with_stub([_Result(boxes=boxes)])
    result = det.detect(np.zeros((4, 4, 3), np.uint8))
    assert result == [Detection(1.0, 2.0, 3.0, 4.0, 0.75)]
