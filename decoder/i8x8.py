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
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
    top_right: Optional[np.ndarray] = None,
    top_available: bool = True,
    left_available: bool = True,
    top_right_available: bool = False,
) -> np.ndarray:
    """Reconstruct a single I_8x8 block.

    Args:
        mode: Prediction mode (0-8)
        residual: 8x8 residual from inverse transform
        top: Top neighbor pixels (8)
        left: Left neighbor pixels (8)
        top_left: Top-left corner pixel
        top_right: Top-right neighbor pixels (8) or None
        top_available: Whether top is available
        left_available: Whether left is available
        top_right_available: Whether top-right is available

    Returns:
        8x8 reconstructed block (uint8)
    """
    # Get prediction
    prediction = predict_intra_8x8(
        mode=Intra8x8Mode(mode),
        top=top,
        left=left,
        top_left=top_left,
        top_right=top_right,
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
