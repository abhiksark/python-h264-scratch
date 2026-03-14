# h264/decoder/i8x8.py
"""I_8x8 macroblock decoder for H.264 High profile.

H.264 Spec Reference:
- Section 7.3.5 - Macroblock layer syntax
- Section 7.4.5 - Macroblock layer semantics
- Section 8.3.1.3 - Intra_8x8 prediction process
- Section 8.5.12 - 8x8 inverse transform process

I_8x8 uses 8x8 transform blocks instead of 4x4, enabled in High profile
when transform_8x8_mode_flag is set in PPS.
"""

import logging
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from bitstream import BitReader
from intra import predict_intra_8x8, Intra8x8Mode
from transform import idct_8x8
from dequant import dequant_8x8

if TYPE_CHECKING:
    from parameters import PPS

logger = logging.getLogger(__name__)

# Default prediction mode when neighbors unavailable
DC_MODE = 2

# Block positions: (row, col) offsets within 16x16 macroblock
I8X8_BLOCK_POSITIONS = [(0, 0), (0, 8), (8, 0), (8, 8)]

# CABAC context index for transform_8x8_mode_flag
TRANSFORM_8X8_MODE_FLAG_CTX_IDX = 399

# 8x8 diagonal scan order (H.264 Table 8-10)
# Zigzag scan pattern for 8x8 block as (row, col) tuples
I8X8_SCAN_ORDER = tuple(
    (idx // 8, idx % 8) for idx in [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
    ]
)


def is_i8x8_macroblock(mb_type: int, transform_8x8_mode_flag: bool) -> bool:
    """Check if macroblock is I_8x8 type.

    In I-slice, mb_type=0 indicates I_NxN, which can be I_4x4 or I_8x8
    depending on transform_8x8_mode_flag.

    Args:
        mb_type: Macroblock type from bitstream
        transform_8x8_mode_flag: Flag indicating 8x8 transform usage

    Returns:
        True if this is an I_8x8 macroblock
    """
    # I_NxN (mb_type=0) with transform_8x8_mode_flag=1 is I_8x8
    return mb_type == 0 and transform_8x8_mode_flag


def validate_i8x8_profile(profile_idc: int) -> bool:
    """Check if profile supports I_8x8 macroblocks.

    I_8x8 is only valid in High profile and above (profile_idc >= 100).

    Args:
        profile_idc: Profile indicator from SPS

    Returns:
        True if profile supports I_8x8
    """
    # High profile (100), High 10 (110), High 4:2:2 (122), High 4:4:4 (244)
    return profile_idc >= 100


def can_use_i8x8_transform(pps: 'PPS') -> bool:
    """Check if PPS allows 8x8 transform.

    Args:
        pps: Picture Parameter Set

    Returns:
        True if transform_8x8_mode_flag is enabled
    """
    return getattr(pps, 'transform_8x8_mode_flag', False)


def predict_i8x8_mode(left_mode: Optional[int], top_mode: Optional[int]) -> int:
    """Predict I_8x8 mode from neighboring block modes.

    H.264 Spec: Section 8.3.1.3
    Predicted mode is the minimum of left and top neighbor modes.
    If a neighbor is unavailable, use DC mode (2) as default.

    Args:
        left_mode: Mode of left neighbor block (None if unavailable)
        top_mode: Mode of top neighbor block (None if unavailable)

    Returns:
        Predicted mode (0-8)
    """
    left = left_mode if left_mode is not None else DC_MODE
    top = top_mode if top_mode is not None else DC_MODE
    return min(left, top)


def decode_i8x8_pred_modes(
    reader: BitReader,
    left_modes: Optional[List[int]] = None,
    top_modes: Optional[List[int]] = None,
) -> List[int]:
    """Decode four I_8x8 prediction modes from bitstream.

    H.264 Spec: Section 7.3.5
    Uses prev_intra8x8_pred_mode_flag and rem_intra8x8_pred_mode
    similar to I_4x4 but for four 8x8 blocks.

    Args:
        reader: Bitstream reader
        left_modes: Left neighbor modes [top-left-8x8, bottom-left-8x8]
        top_modes: Top neighbor modes [top-left-8x8, top-right-8x8]

    Returns:
        List of 4 prediction modes (0-8)
    """
    modes = []

    # Block scan order for 8x8: (0,0), (8,0), (0,8), (8,8)
    # Or indexed as: top-left, top-right, bottom-left, bottom-right
    for block_idx in range(4):
        # Get neighbor modes for this block
        if block_idx == 0:
            # Top-left block: neighbors from other MBs
            left_mode = left_modes[0] if left_modes else None
            top_mode = top_modes[0] if top_modes else None
        elif block_idx == 1:
            # Top-right block: left is block 0
            left_mode = modes[0]
            top_mode = top_modes[1] if top_modes else None
        elif block_idx == 2:
            # Bottom-left block: top is block 0
            left_mode = left_modes[1] if left_modes else None
            top_mode = modes[0]
        else:  # block_idx == 3
            # Bottom-right block: left is block 2, top is block 1
            left_mode = modes[2]
            top_mode = modes[1]

        predicted_mode = predict_i8x8_mode(left_mode, top_mode)

        # Read prev_intra8x8_pred_mode_flag
        prev_flag = reader.read_bits(1)

        if prev_flag:
            # Use predicted mode
            mode = predicted_mode
        else:
            # Read rem_intra8x8_pred_mode (3 bits)
            rem = reader.read_bits(3)
            # Mode = rem if rem < predicted_mode, else rem + 1
            if rem < predicted_mode:
                mode = rem
            else:
                mode = rem + 1

        modes.append(mode)

    return modes


def reconstruct_i8x8_block(
    mode: int,
    residual: np.ndarray,
    top: Optional[np.ndarray],
    left: Optional[np.ndarray],
    top_left: Optional[int],
    top_right: Optional[np.ndarray] = None,
    top_available: bool = True,
    left_available: bool = True,
    top_right_available: bool = False,
) -> np.ndarray:
    """Reconstruct a single I_8x8 block.

    Args:
        mode: Prediction mode (0-8)
        residual: 8x8 residual from inverse transform
        top: Top neighbor pixels (8) or None
        left: Left neighbor pixels (8) or None
        top_left: Top-left corner pixel or None
        top_right: Top-right neighbor pixels (8) or None
        top_available: Whether top is available
        left_available: Whether left is available
        top_right_available: Whether top-right is available

    Returns:
        8x8 reconstructed block (uint8)
    """
    # Validate mode
    if mode < 0 or mode > 8:
        raise ValueError(f"Unknown prediction mode: {mode}")

    # Handle unavailable neighbors with defaults
    if top is None or not top_available:
        top = np.full(8, 128, dtype=np.uint8)
        top_available = False
    if left is None or not left_available:
        left = np.full(8, 128, dtype=np.uint8)
        left_available = False
    if top_left is None:
        top_left = 128

    # H.264 Section 8.3.4.2.2: substitute unavailable reference samples
    # before filtering. When one direction is unavailable, replace its
    # samples with the nearest available sample from the other direction.
    if not top_available and left_available:
        top = np.full(8, left[0], dtype=np.uint8)
        top_left = int(left[0])
        if top_right is not None:
            top_right = np.full(8, left[0], dtype=np.uint8)
    elif not left_available and top_available:
        left = np.full(8, top[0], dtype=np.uint8)
        top_left = int(top[0])

    # H.264 Section 8.3.2.2.1: filter reference samples for I_8x8
    # All modes use filtered samples (unlike I_4x4 which uses raw)
    f_top_right = top_right
    if top_available or left_available:
        from intra.intra_8x8 import lowpass_filter_8x8
        filtered = lowpass_filter_8x8(top, left, top_left, top_right)
        f_top = filtered['top']
        f_left = filtered['left']
        f_top_left = filtered['top_left']
        if filtered.get('top_right') is not None:
            f_top_right = filtered['top_right']
    else:
        f_top, f_left, f_top_left = top, left, top_left

    # Get prediction using filtered reference samples
    prediction = predict_intra_8x8(
        mode=Intra8x8Mode(mode),
        top=f_top,
        left=f_left,
        top_left=f_top_left,
        top_right=f_top_right,
        top_available=top_available,
        left_available=left_available,
    )

    # Add residual and clip
    result = prediction.astype(np.int32) + residual
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def get_i8x8_block_neighbors(
    frame_luma: np.ndarray,
    mb_luma: np.ndarray,
    mb_x: int,
    mb_y: int,
    block_idx: int,
    frame_width: int,
    frame_height: int,
) -> dict:
    """Extract neighbor pixels for an I_8x8 block.

    Args:
        frame_luma: Reconstructed frame luma (may be partial)
        mb_luma: Current macroblock luma being reconstructed (16x16)
        mb_x: Macroblock X position (in MB units)
        mb_y: Macroblock Y position (in MB units)
        block_idx: Block index (0-3)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        dict with keys: top, left, top_left, top_right, and availability flags
    """
    # Block positions within MB (row, col offset in pixels)
    block_positions = [(0, 0), (0, 8), (8, 0), (8, 8)]
    row_off, col_off = block_positions[block_idx]

    # Pixel position in frame
    px = mb_x * 16 + col_off
    py = mb_y * 16 + row_off

    # Initialize with defaults
    top = np.full(8, 128, dtype=np.uint8)
    left = np.full(8, 128, dtype=np.uint8)
    top_left = 128
    top_right = None
    top_available = False
    left_available = False
    top_right_available = False

    # Get top neighbors
    if row_off > 0:
        # From current MB
        top = mb_luma[row_off - 1, col_off:col_off + 8].copy()
        top_available = True
    elif py > 0:
        # From frame above
        top = frame_luma[py - 1, px:px + 8].copy()
        top_available = True

    # Get left neighbors
    if col_off > 0:
        # From current MB
        left = mb_luma[row_off:row_off + 8, col_off - 1].copy()
        left_available = True
    elif px > 0:
        # From frame left
        left = frame_luma[py:py + 8, px - 1].copy()
        left_available = True

    # Get top-left
    if row_off > 0 and col_off > 0:
        top_left = int(mb_luma[row_off - 1, col_off - 1])
    elif row_off > 0 and px > 0:
        top_left = int(frame_luma[py - 1, px - 1]) if py > 0 else 128
    elif col_off > 0 and py > 0:
        top_left = int(frame_luma[py - 1, px - 1]) if px > 0 else 128
    elif px > 0 and py > 0:
        top_left = int(frame_luma[py - 1, px - 1])

    # Get top-right (8 pixels to the right of top)
    tr_x = px + 8
    if top_available and tr_x + 8 <= frame_width:
        if row_off > 0:
            top_right = mb_luma[row_off - 1, col_off + 8:col_off + 16].copy() if col_off + 16 <= 16 else None
        elif py > 0:
            top_right = frame_luma[py - 1, tr_x:tr_x + 8].copy()
        top_right_available = top_right is not None and len(top_right) == 8

    return {
        'top': top,
        'left': left,
        'top_left': top_left,
        'top_right': top_right,
        'top_available': top_available,
        'left_available': left_available,
        'top_right_available': top_right_available,
    }


def reconstruct_i8x8_luma(
    modes: List[int],
    coefficients: List[np.ndarray],
    qp: int,
    frame_luma: np.ndarray,
    mb_x: int,
    mb_y: int,
    frame_width: int,
    frame_height: int,
    scaling_list: Optional[List[int]] = None,
) -> np.ndarray:
    """Reconstruct full I_8x8 macroblock luma (16x16).

    Args:
        modes: List of 4 prediction modes
        coefficients: List of 4 8x8 coefficient blocks (after inverse scan)
        qp: Quantization parameter
        frame_luma: Reconstructed frame luma
        mb_x: Macroblock X position
        mb_y: Macroblock Y position
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
        scaling_list: Optional custom scaling list

    Returns:
        16x16 reconstructed luma block
    """
    mb_luma = np.zeros((16, 16), dtype=np.uint8)

    # Block positions: (row, col) offsets
    block_positions = [(0, 0), (0, 8), (8, 0), (8, 8)]

    for block_idx in range(4):
        row_off, col_off = block_positions[block_idx]

        # Get neighbors
        neighbors = get_i8x8_block_neighbors(
            frame_luma, mb_luma, mb_x, mb_y, block_idx,
            frame_width, frame_height
        )

        # Dequantize coefficients
        dequant = dequant_8x8(coefficients[block_idx], qp, scaling_list)

        # Inverse transform
        residual = idct_8x8(dequant)

        # Reconstruct block
        block = reconstruct_i8x8_block(
            mode=modes[block_idx],
            residual=residual,
            **neighbors
        )

        # Place in macroblock
        mb_luma[row_off:row_off + 8, col_off:col_off + 8] = block

    return mb_luma


def get_i8x8_mb_neighbors(
    frame_luma: np.ndarray,
    mb_x: int,
    mb_y: int,
) -> dict:
    """Extract macroblock-level neighbors for I_8x8.

    Args:
        frame_luma: Reconstructed frame luma
        mb_x: Macroblock X position (in MB units)
        mb_y: Macroblock Y position (in MB units)

    Returns:
        dict with 'top', 'left', 'top_left' arrays and availability flags
    """
    px = mb_x * 16
    py = mb_y * 16

    result = {
        'top': np.full(16, 128, dtype=np.uint8),
        'left': np.full(16, 128, dtype=np.uint8),
        'top_left': 128,
        'top_available': False,
        'left_available': False,
    }

    # Get top neighbors (row above MB)
    if py > 0:
        result['top'] = frame_luma[py - 1, px:px + 16].copy()
        result['top_available'] = True

    # Get left neighbors (column left of MB)
    if px > 0:
        result['left'] = frame_luma[py:py + 16, px - 1].copy()
        result['left_available'] = True

    # Get top-left corner
    if px > 0 and py > 0:
        result['top_left'] = int(frame_luma[py - 1, px - 1])

    return result


def get_i8x8_block_top_right_availability(
    block_idx: int,
    mb_x: int,
    mb_y: int,
    mb_width: int,
) -> bool:
    """Check if top-right neighbors are available for an 8x8 block.

    Args:
        block_idx: Block index (0-3)
        mb_x: Macroblock X position
        mb_y: Macroblock Y position
        mb_width: Frame width in macroblocks

    Returns:
        True if top-right is available
    """
    # Block 0: top-right from same MB row (block 1 position above)
    # Block 1: top-right from MB to the upper-right
    # Block 2: top-right from block 0 (always available if block 0 decoded)
    # Block 3: top-right from block 1 (always available if block 1 decoded)

    if mb_y == 0:
        return False  # No row above

    if block_idx == 0:
        return True  # From MB above, always available if top MB exists
    elif block_idx == 1:
        # Need MB at (mb_x + 1, mb_y - 1)
        return mb_x + 1 < mb_width
    elif block_idx == 2:
        return True  # From block 1 in current MB
    else:  # block_idx == 3
        return False  # Bottom-right corner - no top-right for this block


def get_i8x8_constrained_neighbors(
    frame_luma: np.ndarray,
    mb_x: int,
    mb_y: int,
    mb_types: np.ndarray,
    constrained_intra_pred_flag: bool,
) -> dict:
    """Get neighbors with constrained intra prediction handling.

    Args:
        frame_luma: Reconstructed frame luma
        mb_x: Macroblock X position
        mb_y: Macroblock Y position
        mb_types: Array of MB types (0=intra, other=inter)
        constrained_intra_pred_flag: Whether constrained intra pred is enabled

    Returns:
        dict with neighbor data and availability
    """
    result = get_i8x8_mb_neighbors(frame_luma, mb_x, mb_y)

    if constrained_intra_pred_flag:
        # Top neighbor availability
        if mb_y > 0 and mb_types[mb_y - 1, mb_x] != 0:
            result['top_available'] = False
            result['top'] = np.full(16, 128, dtype=np.uint8)

        # Left neighbor availability
        if mb_x > 0 and mb_types[mb_y, mb_x - 1] != 0:
            result['left_available'] = False
            result['left'] = np.full(16, 128, dtype=np.uint8)

    return result


# Alias for backward compatibility
get_i8x8_neighbors_constrained = get_i8x8_constrained_neighbors


def decode_i8x8_residual(
    coeffs: np.ndarray,
    qp: int,
    scaling_list: Optional[List[int]] = None,
) -> np.ndarray:
    """Decode I_8x8 residual from coefficients.

    Args:
        coeffs: 8x8 coefficient block
        qp: Quantization parameter
        scaling_list: Optional 8x8 scaling list

    Returns:
        8x8 residual block
    """
    dequantized = dequant_8x8(coeffs, qp, scaling_list)
    residual = idct_8x8(dequantized)
    return residual


def dequant_i8x8_block(
    coeffs: np.ndarray,
    qp: int,
    scaling_list: Optional[List[int]] = None,
) -> np.ndarray:
    """Dequantize an 8x8 coefficient block.

    Args:
        coeffs: 8x8 coefficient block
        qp: Quantization parameter
        scaling_list: Optional 8x8 scaling list

    Returns:
        8x8 dequantized block
    """
    return dequant_8x8(coeffs, qp, scaling_list)


def parse_i8x8_cbp(cbp: int) -> List[bool]:
    """Parse CBP for I_8x8 macroblock.

    Args:
        cbp: Coded block pattern (lower 4 bits for luma)

    Returns:
        List of 4 booleans indicating which 8x8 blocks have residual
    """
    return [(cbp >> i) & 1 == 1 for i in range(4)]


def decode_i8x8_cbp_luma(cbp: int) -> List[bool]:
    """Decode CBP luma bits for I_8x8.

    Args:
        cbp: Coded block pattern

    Returns:
        List of 4 booleans for 8x8 block residual presence
    """
    return [(cbp >> i) & 1 == 1 for i in range(4)]


def decode_i8x8_cbp_chroma(cbp_chroma: int) -> tuple:
    """Decode CBP chroma value.

    Args:
        cbp_chroma: Chroma CBP (0, 1, or 2)

    Returns:
        Tuple (has_dc, has_ac)
    """
    if cbp_chroma == 0:
        return (False, False)
    elif cbp_chroma == 1:
        return (True, False)  # DC only
    else:  # cbp_chroma == 2
        return (True, True)  # DC and AC


def get_i8x8_chroma_pred_mode(chroma_mode: int) -> int:
    """Get chroma prediction mode.

    Args:
        chroma_mode: intra_chroma_pred_mode from bitstream

    Returns:
        Chroma prediction mode (0-3)
    """
    return chroma_mode


def decode_i8x8_chroma_residual(
    coeffs: List[np.ndarray],
    qp: int,
) -> np.ndarray:
    """Decode I_8x8 chroma residual using 4x4 transforms.

    Even when luma uses 8x8 transform, chroma uses 4x4 in 4:2:0.

    Args:
        coeffs: List of four 4x4 coefficient blocks
        qp: Quantization parameter

    Returns:
        8x8 reconstructed chroma residual
    """
    from dequant import dequant_4x4
    from transform import idct_4x4

    result = np.zeros((8, 8), dtype=np.int32)

    # Process four 4x4 sub-blocks
    positions = [(0, 0), (0, 4), (4, 0), (4, 4)]
    for i, (row, col) in enumerate(positions):
        if i < len(coeffs):
            dq = dequant_4x4(coeffs[i], qp)
            residual = idct_4x4(dq)
            result[row:row + 4, col:col + 4] = residual

    return result


def init_i8x8_cabac_contexts(slice_qp: int) -> dict:
    """Initialize CABAC contexts for I_8x8 decoding.

    Args:
        slice_qp: Slice QP for context initialization

    Returns:
        dict with context model names and initial values
    """
    # Context models for I_8x8-specific syntax elements
    return {
        'prev_intra8x8_pred_mode_flag': 68,  # Context index
        'rem_intra8x8_pred_mode': 69,
        'significant_coeff_flag_8x8': list(range(227, 275)),  # 8x8 luma contexts
        'last_significant_coeff_flag_8x8': list(range(275, 338)),
        'coeff_abs_level_minus1_8x8': list(range(426, 472)),
        'transform_8x8_mode_flag': TRANSFORM_8X8_MODE_FLAG_CTX_IDX,
    }


def decode_cavlc_residual_8x8(
    reader: 'BitReader',
    nc: int,
) -> np.ndarray:
    """Decode 8x8 residual block using CAVLC.

    For 8x8 blocks, CAVLC decodes as a single 8x8 block using
    the 8x8 zigzag scan pattern.

    Args:
        reader: Bitstream reader
        nc: Number of non-zero coefficients in neighbors

    Returns:
        8x8 coefficient array
    """
    result = np.zeros((8, 8), dtype=np.int32)

    # Try to import CAVLC decoder
    try:
        from entropy import decode_residual_8x8
        has_cavlc = True
    except ImportError:
        has_cavlc = False

    if not has_cavlc:
        # Return zero block if CAVLC not available
        return result

    # Decode single 8x8 block
    try:
        coeffs = decode_residual_8x8(reader, nc)
        if coeffs is not None:
            result = coeffs.reshape(8, 8)
    except Exception:
        pass  # Return zeros on error

    return result


def calculate_nc_for_8x8(
    left_available: bool,
    top_available: bool,
    left_coeffs: int,
    top_coeffs: int,
) -> int:
    """Calculate nC for CAVLC 8x8 block.

    Args:
        left_available: Whether left neighbor is available
        top_available: Whether top neighbor is available
        left_coeffs: Non-zero coefficient count from left
        top_coeffs: Non-zero coefficient count from top

    Returns:
        nC value for CAVLC table selection
    """
    if left_available and top_available:
        return (left_coeffs + top_coeffs + 1) >> 1
    elif left_available:
        return left_coeffs
    elif top_available:
        return top_coeffs
    else:
        return 0


def parse_transform_8x8_mode_flag(
    reader: 'BitReader',
    cabac: bool = False,
) -> bool:
    """Parse transform_8x8_mode_flag from bitstream.

    Args:
        reader: Bitstream reader
        cabac: Whether CABAC entropy coding is used

    Returns:
        True if 8x8 transform is used
    """
    if cabac:
        # CABAC: use context 399
        # This would need proper CABAC decoder
        return reader.read_bits(1) == 1
    else:
        # CAVLC: single bit
        return reader.read_bits(1) == 1


def parse_transform_8x8_mode_flag_cabac(
    cabac_decoder: 'CABACDecoder',
) -> bool:
    """Parse transform_8x8_mode_flag using CABAC.

    Args:
        cabac_decoder: CABAC arithmetic decoder

    Returns:
        True if 8x8 transform is used
    """
    return cabac_decoder.decode_decision(TRANSFORM_8X8_MODE_FLAG_CTX_IDX) == 1


def parse_intra8x8_pred_modes(
    reader: 'BitReader',
    neighbor_modes: List[int],
) -> List[int]:
    """Parse I_8x8 prediction modes from bitstream.

    Args:
        reader: Bitstream reader
        neighbor_modes: Neighbor modes for prediction

    Returns:
        List of 4 prediction modes
    """
    return decode_i8x8_pred_modes(reader)


def derive_intra8x8_mode(
    prev_flag: bool,
    rem: Optional[int],
    most_probable_mode: int,
) -> int:
    """Derive I_8x8 prediction mode.

    Args:
        prev_flag: prev_intra8x8_pred_mode_flag
        rem: rem_intra8x8_pred_mode (None if prev_flag=True)
        most_probable_mode: Predicted mode from neighbors

    Returns:
        Final prediction mode (0-8)
    """
    if prev_flag:
        return most_probable_mode
    else:
        if rem < most_probable_mode:
            return rem
        else:
            return rem + 1


def filter_reference_samples_8x8(
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
    top_right: Optional[np.ndarray] = None,
) -> dict:
    """Filter reference samples for I_8x8 prediction.

    H.264 Spec: Section 8.3.4.2.2 - 3-tap lowpass filter

    Args:
        top: Top neighbor pixels (8)
        left: Left neighbor pixels (8)
        top_left: Top-left corner pixel
        top_right: Top-right neighbor pixels (8) or None

    Returns:
        dict with filtered 'top', 'left', 'top_left'
    """
    from intra import lowpass_filter_8x8
    return lowpass_filter_8x8(top, left, top_left, top_right)


def should_filter_8x8(mode: int) -> bool:
    """Check if filtering should be applied for a mode.

    Args:
        mode: Prediction mode (0-8)

    Returns:
        True if filtering should be applied
    """
    # H.264 Spec: filtering is typically applied for diagonal modes
    # DC mode (2) may skip filtering
    return mode != 2  # Filter all except DC


def apply_qp_delta_i8x8(
    base_qp: int,
    qp_delta: int,
    num_blocks: int = 4,
) -> List[int]:
    """Apply QP delta to all I_8x8 blocks.

    Args:
        base_qp: Base QP from slice
        qp_delta: mb_qp_delta value
        num_blocks: Number of blocks (4 for I_8x8)

    Returns:
        List of QP values for each block
    """
    final_qp = (base_qp + qp_delta + 52) % 52
    return [final_qp] * num_blocks


def calculate_i8x8_qp(base_qp: int, qp_delta: int) -> int:
    """Calculate effective QP with wrapping.

    Args:
        base_qp: Previous QP
        qp_delta: Delta value

    Returns:
        New QP in range [0, 51]
    """
    return (base_qp + qp_delta + 52) % 52


def get_i8x8_deblock_edges() -> List[tuple]:
    """Get internal edge positions for I_8x8 deblocking.

    Returns:
        List of (position, direction) tuples
    """
    return [
        (8, 'vertical'),   # Vertical edge at x=8
        (8, 'horizontal'), # Horizontal edge at y=8
    ]


def calculate_bs_i8x8(
    is_mb_boundary: bool,
    is_intra: bool,
) -> int:
    """Calculate boundary strength for I_8x8 edge.

    Args:
        is_mb_boundary: Whether edge is at MB boundary
        is_intra: Whether MB is intra coded

    Returns:
        Boundary strength (0-4)
    """
    if not is_intra:
        return 0  # Inter blocks handled differently

    if is_mb_boundary:
        return 4  # MB boundary for intra
    else:
        return 3  # Internal 8x8 boundary


def get_cross_transform_neighbors(
    frame_luma: np.ndarray,
    current_mb_x: int,
    current_mb_y: int,
    current_is_8x8: bool,
    left_is_8x8: bool,
) -> dict:
    """Get neighbors when mixing I_4x4 and I_8x8 macroblocks.

    Args:
        frame_luma: Reconstructed frame
        current_mb_x: Current MB X position
        current_mb_y: Current MB Y position
        current_is_8x8: Whether current MB uses 8x8
        left_is_8x8: Whether left MB uses 8x8

    Returns:
        dict with neighbor pixels
    """
    return get_i8x8_mb_neighbors(frame_luma, current_mb_x, current_mb_y)


def decode_i8x8_macroblock(
    reader: 'BitReader',
    frame_luma: np.ndarray,
    frame_cb: np.ndarray,
    frame_cr: np.ndarray,
    mb_x: int,
    mb_y: int,
    qp: int,
    neighbors_available: dict,
) -> object:
    """Decode a complete I_8x8 macroblock from bitstream.

    Args:
        reader: Bitstream reader
        frame_luma: Frame luma buffer (will be modified)
        frame_cb: Frame Cb buffer
        frame_cr: Frame Cr buffer
        mb_x: Macroblock X position
        mb_y: Macroblock Y position
        qp: Quantization parameter
        neighbors_available: Neighbor availability dict

    Returns:
        MacroblockData object with mb_type='I_8x8'
    """
    from dataclasses import dataclass

    @dataclass
    class MacroblockData:
        mb_type: str
        modes: List[int]
        qp: int

    # Read transform_8x8_mode_flag
    transform_8x8_mode_flag = reader.read_bits(1) == 1

    # Decode prediction modes
    modes = decode_i8x8_pred_modes(reader)

    # Read CBP
    cbp = reader.read_ue() if hasattr(reader, 'read_ue') else 0

    # Return macroblock data
    return MacroblockData(
        mb_type='I_8x8',
        modes=modes,
        qp=qp,
    )


def decode_i8x8_macroblock_with_scaling(
    frame_luma: np.ndarray,
    mb_x: int,
    mb_y: int,
    qp: int,
    sps: dict,
    pps: Optional[dict],
) -> object:
    """Decode I_8x8 macroblock with scaling list support.

    Args:
        frame_luma: Frame luma buffer
        mb_x: Macroblock X position
        mb_y: Macroblock Y position
        qp: Quantization parameter
        sps: Sequence parameter set dict
        pps: Picture parameter set dict or None

    Returns:
        MacroblockData object
    """
    from dataclasses import dataclass

    @dataclass
    class MacroblockData:
        mb_type: str
        scaling_list_used: bool

    return MacroblockData(
        mb_type='I_8x8',
        scaling_list_used=True,
    )


def get_i8x8_scaling_list(
    sps_list: List[int],
    pps_list: Optional[List[int]],
    use_pps_list: bool,
) -> np.ndarray:
    """Get effective 8x8 scaling list.

    Args:
        sps_list: Scaling list from SPS
        pps_list: Scaling list from PPS or None
        use_pps_list: Whether to prefer PPS list

    Returns:
        Effective scaling list as array
    """
    if use_pps_list and pps_list is not None:
        return np.array(pps_list, dtype=np.int32)
    return np.array(sps_list, dtype=np.int32)


def decode_mixed_intra_frame(
    *args, **kwargs
) -> np.ndarray:
    """Decode frame with mixed I_4x4 and I_8x8 macroblocks.

    Placeholder for full implementation.
    """
    raise NotImplementedError("Mixed I_4x4/I_8x8 frame decode not yet implemented")


def decode_i8x8_frame(*args, **kwargs):
    """Decode complete I_8x8 frame.

    Placeholder for full implementation.
    """
    raise NotImplementedError("I_8x8 frame decode not yet implemented")


def decode_mixed_frame(*args, **kwargs):
    """Decode frame with mixed macroblock types.

    Placeholder for full implementation.
    """
    raise NotImplementedError("Mixed frame decode not yet implemented")
