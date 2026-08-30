import numpy as np
import pytest
from sfcpi.flow import FlowEstimator

def _textured(h=64, w=64, shift=0):
    """Deterministic texture so Farneback has gradients to track."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, size=(h, w + 32), dtype=np.uint8)
    crop = base[:, shift:shift + w]
    return np.repeat(crop[:, :, None], 3, axis=2)

def test_flow_shape_matches_frame():
    est = FlowEstimator()
    flow = est.estimate(_textured(), _textured(shift=2))
    assert flow.shape == (64, 64, 2)
    assert flow.dtype == np.float32

def test_static_scene_gives_near_zero_flow():
    est = FlowEstimator()
    img = _textured()
    flow = est.estimate(img, img)
    assert np.abs(flow).max() < 0.1

def test_horizontal_shift_detected_with_correct_sign():
    est = FlowEstimator()
    flow = est.estimate(_textured(shift=0), _textured(shift=4))
    mean_dx = float(np.mean(flow[..., 0]))
    assert mean_dx < -1.0     # content moved left in image coords

def test_downscale_rescales_vectors_back_to_full_resolution():
    full = FlowEstimator(downscale=1.0).estimate(_textured(), _textured(shift=4))
    half = FlowEstimator(downscale=0.5).estimate(_textured(), _textured(shift=4))
    assert half.shape == (64, 64, 2)
    assert float(np.mean(half[..., 0])) == pytest.approx(float(np.mean(full[..., 0])), abs=1.5)

def test_mismatched_shapes_raise():
    est = FlowEstimator()
    with pytest.raises(ValueError, match="same shape"):
        est.estimate(_textured(64, 64), _textured(32, 32))
