# h264/entropy/__init__.py
"""Entropy decoding module for H.264.

Handles CAVLC (Context-Adaptive Variable-Length Coding) used in
Baseline profile for decoding transform coefficients.

H.264 Spec Reference:
- Section 9.2: CAVLC parsing process
"""

from .cavlc import (
    CAVLCBlock,
    decode_coeff_token,
    decode_trailing_ones_signs,
    decode_levels,
    decode_total_zeros,
    decode_run_before,
    decode_residual_block,
    decode_residual_4x4,
    decode_residual_8x8,
    decode_chroma_dc,
    decode_luma_dc_16x16,
    calculate_nC,
)

from .tables import (
    ZIGZAG_4x4,
    ZIGZAG_2x2,
    ZIGZAG_4x4_INV,
    ZIGZAG_8x8,
    ZIGZAG_8x8_INV,
)

__all__ = [
    # CAVLC decoding
    "CAVLCBlock",
    "decode_coeff_token",
    "decode_trailing_ones_signs",
    "decode_levels",
    "decode_total_zeros",
    "decode_run_before",
    "decode_residual_block",
    "decode_residual_4x4",
    "decode_residual_8x8",
    "decode_chroma_dc",
    "decode_luma_dc_16x16",
    "calculate_nC",
    # Tables
    "ZIGZAG_4x4",
    "ZIGZAG_2x2",
    "ZIGZAG_4x4_INV",
    "ZIGZAG_8x8",
    "ZIGZAG_8x8_INV",
]
