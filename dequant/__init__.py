# h264/dequant/__init__.py
"""Inverse quantization (dequantization) module.

Handles scaling of transform coefficients based on QP.
Supports both 4x4 and 8x8 blocks.
"""

from .dequant import (
    LEVEL_SCALE,
    LEVEL_SCALE_8x8,
    POSITION_TYPE,
    POSITION_TYPE_8x8,
    get_scale_matrix,
    get_scale_matrix_8x8,
    dequant_4x4,
    dequant_4x4_simple,
    dequant_8x8,
    dequant_dc_4x4,
    dequant_dc_2x2,
    get_chroma_qp,
    qp_to_qstep,
)

__all__ = [
    "LEVEL_SCALE",
    "LEVEL_SCALE_8x8",
    "POSITION_TYPE",
    "POSITION_TYPE_8x8",
    "get_scale_matrix",
    "get_scale_matrix_8x8",
    "dequant_4x4",
    "dequant_4x4_simple",
    "dequant_8x8",
    "dequant_dc_4x4",
    "dequant_dc_2x2",
    "get_chroma_qp",
    "qp_to_qstep",
]
