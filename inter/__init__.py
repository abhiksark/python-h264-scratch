# h264/inter/__init__.py
"""Inter prediction module for P and B frame decoding."""

from inter.reference import ReferenceFrame, ReferenceFrameBuffer

__all__ = [
    "ReferenceFrame",
    "ReferenceFrameBuffer",
]
