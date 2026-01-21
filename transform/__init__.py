# h264/transform/__init__.py
"""Transform module for H.264 inverse transforms.

Handles 4x4 and 8x8 inverse DCT and Hadamard transforms.
"""

from .idct_4x4 import (
    idct_4x4,
    idct_4x4_matrix,
    idct_1d,
    hadamard_4x4,
    hadamard_2x2,
    inverse_hadamard_4x4,
    inverse_hadamard_2x2,
    forward_4x4,
    process_dc_block_i16x16,
    process_dc_block_chroma,
    IDCT_CORE,
)
from .idct_8x8 import (
    idct_8x8,
    idct_1d_8,
    forward_8x8,
    forward_1d_8,
    idct_8x8_batch,
    ZIGZAG_SCAN_8x8,
    FIELD_SCAN_8x8,
    TRANSFORM_MATRIX_8x8,
    SCALING_FACTORS_8x8,
)

__all__ = [
    # 4x4 transforms
    "idct_4x4",
    "idct_4x4_matrix",
    "idct_1d",
    "forward_4x4",
    "hadamard_4x4",
    "hadamard_2x2",
    "inverse_hadamard_4x4",
    "inverse_hadamard_2x2",
    "process_dc_block_i16x16",
    "process_dc_block_chroma",
    "IDCT_CORE",
    # 8x8 transforms
    "idct_8x8",
    "idct_1d_8",
    "forward_8x8",
    "forward_1d_8",
    "idct_8x8_batch",
    "ZIGZAG_SCAN_8x8",
    "FIELD_SCAN_8x8",
    "TRANSFORM_MATRIX_8x8",
    "SCALING_FACTORS_8x8",
]
