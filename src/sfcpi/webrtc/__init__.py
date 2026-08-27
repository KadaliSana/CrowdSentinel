"""Live KVS WebRTC ingest. Import-light: heavy deps load inside submodules."""
from .bridge import FrameBridge

__all__ = ["FrameBridge"]
