# h264/intra/__init__.py
"""Intra prediction module for H.264 I-frames.

Handles 16x16, 8x8 (High profile), and 4x4 luma prediction.
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

from .intra_8x8 import (
    Intra8x8Mode,
    predict_intra_8x8,
    intra_8x8_vertical,
    intra_8x8_horizontal,
    intra_8x8_dc,
    intra_8x8_diagonal_down_left,
    intra_8x8_diagonal_down_right,
    intra_8x8_vertical_right,
    intra_8x8_horizontal_down,
    intra_8x8_vertical_left,
    intra_8x8_horizontal_up,
    # Lowpass filtering (High profile)
    lowpass_filter_8x8,
    # Filtered diagonal modes
    intra_8x8_diagonal_down_left_filtered,
    intra_8x8_diagonal_down_right_filtered,
    # Availability-safe variants
    intra_8x8_vertical_safe,
    intra_8x8_horizontal_safe,
    intra_8x8_diagonal_down_right_safe,
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
    # 8x8 prediction (High profile)
    "Intra8x8Mode",
    "predict_intra_8x8",
    "intra_8x8_vertical",
    "intra_8x8_horizontal",
    "intra_8x8_dc",
    "intra_8x8_diagonal_down_left",
    "intra_8x8_diagonal_down_right",
    "intra_8x8_vertical_right",
    "intra_8x8_horizontal_down",
    "intra_8x8_vertical_left",
    "intra_8x8_horizontal_up",
    # 8x8 lowpass filtering
    "lowpass_filter_8x8",
    # 8x8 filtered diagonal modes
    "intra_8x8_diagonal_down_left_filtered",
    "intra_8x8_diagonal_down_right_filtered",
    # 8x8 availability-safe variants
    "intra_8x8_vertical_safe",
    "intra_8x8_horizontal_safe",
    "intra_8x8_diagonal_down_right_safe",
]
