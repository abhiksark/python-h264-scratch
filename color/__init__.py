# h264/color/__init__.py
"""Color space conversion module.

Handles YCbCr (4:2:0) to RGB conversion for H.264 decoded frames.
"""

from .yuv_to_rgb import (
    ColorMatrix,
    ycbcr_to_rgb,
    rgb_to_ycbcr,
    upsample_chroma,
    subsample_chroma,
)

__all__ = [
    "ColorMatrix",
    "ycbcr_to_rgb",
    "rgb_to_ycbcr",
    "upsample_chroma",
    "subsample_chroma",
]
