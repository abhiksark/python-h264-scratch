# h264/decoder/__init__.py
"""H.264 Baseline profile decoder.

Main decoder module that orchestrates all decoding stages.

H.264 Spec Reference:
- Section 7: Syntax and semantics
- Section 8: Decoding process
"""

from .decoder import (
    H264Decoder,
    DecodedFrame,
    DecoderState,
    decode_h264_file,
    decode_h264_bytes,
)

from .i8x8 import (
    is_i8x8_macroblock,
    validate_i8x8_profile,
    can_use_i8x8_transform,
    predict_i8x8_mode,
    decode_i8x8_pred_modes,
    reconstruct_i8x8_block,
    get_i8x8_block_neighbors,
    reconstruct_i8x8_luma,
)

__all__ = [
    "H264Decoder",
    "DecodedFrame",
    "DecoderState",
    "decode_h264_file",
    "decode_h264_bytes",
    # I_8x8 support
    "is_i8x8_macroblock",
    "validate_i8x8_profile",
    "can_use_i8x8_transform",
    "predict_i8x8_mode",
    "decode_i8x8_pred_modes",
    "reconstruct_i8x8_block",
    "get_i8x8_block_neighbors",
    "reconstruct_i8x8_luma",
]
