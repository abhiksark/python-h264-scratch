# h264/intra/__init__.py
"""Intra prediction module for H.264 I-frames.

Handles 16x16 and 4x4 luma prediction, and 8x8 chroma prediction.
"""

from .intra_16x16 import (
    Intra16x16Mode,
    DEFAULT_PIXEL_VALUE,
    intra_16x16_vertical,
    intra_16x16_horizontal,
    intra_16x16_dc,
    intra_16x16_plane,
    predict_intra_16x16,
    get_neighbors_for_macroblock,
    validate_prediction_mode,
)

from .intra_4x4 import (
    Intra4x4Mode,
    predict_intra_4x4,
    intra_4x4_vertical,
    intra_4x4_horizontal,
    intra_4x4_dc,
    intra_4x4_diagonal_down_left,
    intra_4x4_diagonal_down_right,
    intra_4x4_vertical_right,
    intra_4x4_horizontal_down,
    intra_4x4_vertical_left,
    intra_4x4_horizontal_up,
)

__all__ = [
    # 16x16 prediction
    "Intra16x16Mode",
    "DEFAULT_PIXEL_VALUE",
    "intra_16x16_vertical",
    "intra_16x16_horizontal",
    "intra_16x16_dc",
    "intra_16x16_plane",
    "predict_intra_16x16",
    "get_neighbors_for_macroblock",
    "validate_prediction_mode",
    # 4x4 prediction
    "Intra4x4Mode",
    "predict_intra_4x4",
    "intra_4x4_vertical",
    "intra_4x4_horizontal",
    "intra_4x4_dc",
    "intra_4x4_diagonal_down_left",
    "intra_4x4_diagonal_down_right",
    "intra_4x4_vertical_right",
    "intra_4x4_horizontal_down",
    "intra_4x4_vertical_left",
    "intra_4x4_horizontal_up",
]
