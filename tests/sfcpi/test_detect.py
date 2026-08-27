import numpy as np
import pytest
from sfcpi.detect import Detection, FixedDetector, counts_per_cell
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
