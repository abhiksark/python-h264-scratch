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

from .poc import (
    POCCalculator,
    calculate_poc,
)

from . import poc  # For 'from decoder import poc' style imports

from .i8x8 import (
    is_i8x8_macroblock,
    validate_i8x8_profile,
    can_use_i8x8_transform,
    predict_i8x8_mode,
    decode_i8x8_pred_modes,
    reconstruct_i8x8_block,
    get_i8x8_block_neighbors,
    reconstruct_i8x8_luma,
    # Constants
    I8X8_BLOCK_POSITIONS,
    I8X8_SCAN_ORDER,
    TRANSFORM_8X8_MODE_FLAG_CTX_IDX,
    # Neighbor functions
    get_i8x8_mb_neighbors,
    get_i8x8_block_top_right_availability,
    get_i8x8_constrained_neighbors,
    get_i8x8_neighbors_constrained,
    get_cross_transform_neighbors,
    # Residual decoding
    decode_i8x8_residual,
    dequant_i8x8_block,
    parse_i8x8_cbp,
    decode_i8x8_cbp_luma,
    decode_i8x8_cbp_chroma,
    # Chroma
    get_i8x8_chroma_pred_mode,
    decode_i8x8_chroma_residual,
    # CABAC/CAVLC
    init_i8x8_cabac_contexts,
    decode_cavlc_residual_8x8,
    calculate_nc_for_8x8,
    parse_transform_8x8_mode_flag,
    parse_transform_8x8_mode_flag_cabac,
    parse_intra8x8_pred_modes,
    derive_intra8x8_mode,
    # Filtering
    filter_reference_samples_8x8,
    should_filter_8x8,
    # QP
    apply_qp_delta_i8x8,
    calculate_i8x8_qp,
    # Deblocking
    get_i8x8_deblock_edges,
    calculate_bs_i8x8,
    # Macroblock/frame decode
    decode_i8x8_macroblock,
    decode_i8x8_macroblock_with_scaling,
    get_i8x8_scaling_list,
    decode_mixed_intra_frame,
    decode_i8x8_frame,
    decode_mixed_frame,
)

__all__ = [
    "H264Decoder",
    "DecodedFrame",
    "DecoderState",
    "decode_h264_file",
    "decode_h264_bytes",
    # POC Calculator
    "poc",
    "POCCalculator",
    "calculate_poc",
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
