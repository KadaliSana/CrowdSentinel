"""Root conftest.

The installed pytest (6.2.5) predates the `pythonpath` ini option (added in
pytest 7.0), so `pytest.ini`'s `pythonpath = src` is silently ignored (with a
warning) rather than honoured. Insert `src` on sys.path directly so the
src-layout package is importable regardless of pytest version.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
