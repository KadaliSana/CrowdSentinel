import numpy as np
import pytest
from sfcpi.sources import Frame, FileSource, SyntheticSource

def test_synthetic_source_yields_indexed_frames():
    imgs = [np.zeros((4, 4, 3), np.uint8) for _ in range(3)]
    src = SyntheticSource(imgs, fps=5.0)
    frames = list(src)
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[1].timestamp == pytest.approx(0.2)
    assert isinstance(frames[0], Frame)
    assert src.fps == 5.0

def test_synthetic_source_rejects_bad_fps():
    with pytest.raises(ValueError, match="fps"):
        SyntheticSource([np.zeros((2, 2, 3), np.uint8)], fps=0.0)

def test_file_source_missing_file_names_the_path():
    with pytest.raises(FileNotFoundError, match="no_such_video.mp4"):
        list(FileSource("no_such_video.mp4"))

def test_file_source_reads_written_video(tmp_path):
    import cv2
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(5):
        writer.write(np.full((32, 32, 3), i * 20, np.uint8))
    writer.release()

    src = FileSource(path)
    frames = list(src)
    assert len(frames) == 5
    assert src.fps == pytest.approx(10.0)
    assert frames[0].image.shape == (32, 32, 3)

def test_file_source_respects_max_frames(tmp_path):
    import cv2
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(10):
        writer.write(np.full((32, 32, 3), i * 10, np.uint8))
    writer.release()
    assert len(list(FileSource(path, max_frames=3))) == 3

def test_fps_is_correct_before_iteration(tmp_path):
    """Reading .fps before iterating must not return the 25.0 default."""
    import cv2
    path = str(tmp_path / "clip.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(3):
        writer.write(np.full((32, 32, 3), i * 20, np.uint8))
    writer.release()

    src = FileSource(path)          # note: NOT iterated
    assert src.fps == pytest.approx(10.0)
