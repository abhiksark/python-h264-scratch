# h264/dequant/__init__.py
"""Inverse quantization (dequantization) module.

Handles scaling of transform coefficients based on QP.
"""

from .dequant import (
    LEVEL_SCALE,
    POSITION_TYPE,
    get_scale_matrix,
    dequant_4x4,
    dequant_4x4_simple,
    dequant_dc_4x4,
    dequant_dc_2x2,
    get_chroma_qp,
    qp_to_qstep,
)

__all__ = [
    "LEVEL_SCALE",
    "POSITION_TYPE",
    "get_scale_matrix",
    "dequant_4x4",
    "dequant_4x4_simple",
    "dequant_dc_4x4",
    "dequant_dc_2x2",
    "get_chroma_qp",
    "qp_to_qstep",
]
