import numpy as np
import pytest
from sfcpi.grid import CellGrid

def test_grid_dimensions():
    g = CellGrid(frame_width=640, frame_height=480, cell_size=160)
    assert (g.n_cols, g.n_rows) == (4, 3)
    assert g.cell_area_px == pytest.approx(160 * 160)

def test_rejects_non_dividing_cell_size():
    with pytest.raises(ValueError, match="cell_size"):
        CellGrid(frame_width=640, frame_height=480, cell_size=150)

def test_iter_cells_covers_every_pixel_once():
    g = CellGrid(frame_width=8, frame_height=4, cell_size=2)
    canvas = np.zeros((4, 8), dtype=int)
    for _r, _c, ys, xs in g.iter_cells():
        canvas[ys, xs] += 1
    assert (canvas == 1).all()

def test_aggregate_shapes_and_uniform_motion():
    g = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    flow = np.full((4, 4, 2), 2.0)
    counts = np.full((2, 2), 5.0)
    out = g.aggregate(flow, counts, fps=10.0)
    for key in ("pressure", "mean_speed", "speed_variance", "density"):
        assert out[key].shape == (2, 2)
    assert out["speed_variance"] == pytest.approx(0.0)
    assert out["pressure"] == pytest.approx(0.0)
    assert out["mean_speed"] == pytest.approx(np.sqrt(8.0))

def test_aggregate_isolates_cells():
    """Motion in one cell must not leak into its neighbours."""
    g = CellGrid(frame_width=4, frame_height=2, cell_size=2)
    flow = np.zeros((2, 4, 2))
    flow[0, 0] = (6.0, 0.0)          # single moving pixel in cell (0,0)
    counts = np.ones((1, 2))
    out = g.aggregate(flow, counts, fps=10.0)
    assert out["speed_variance"][0, 0] > 0
    assert out["speed_variance"][0, 1] == pytest.approx(0.0)

def test_counts_shape_must_match_grid():
    g = CellGrid(frame_width=4, frame_height=4, cell_size=2)
    with pytest.raises(ValueError, match="counts"):
        g.aggregate(np.zeros((4, 4, 2)), np.ones((3, 3)), fps=10.0)
