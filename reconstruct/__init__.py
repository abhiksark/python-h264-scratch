# h264/reconstruct/__init__.py
"""Macroblock reconstruction module for H.264.

Combines prediction, entropy decoding, dequantization, and inverse
transform to reconstruct macroblock pixels.

H.264 Spec Reference:
- Section 8.5: Transform coefficient decoding
- Section 8.3: Intra prediction
"""

from .macroblock import (
    MBType,
    CodedBlockPattern,
    MacroblockData,
    decode_i16x16_mb_type,
    decode_cbp_intra,
    decode_macroblock,
    reconstruct_i16x16_luma,
    reconstruct_chroma,
)

__all__ = [
    "MBType",
    "CodedBlockPattern",
    "MacroblockData",
    "decode_i16x16_mb_type",
    "decode_cbp_intra",
    "decode_macroblock",
    "reconstruct_i16x16_luma",
    "reconstruct_chroma",
]
