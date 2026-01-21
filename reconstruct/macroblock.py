# h264/reconstruct/macroblock.py
"""Macroblock reconstruction for H.264 I-frames.

This module combines prediction, entropy decoding, dequantization,
and inverse transform to reconstruct macroblock pixels.

H.264 Spec Reference:
- Section 7.3.5: Macroblock layer syntax
- Section 7.4.5: Macroblock layer semantics
- Section 8.3: Intra prediction
- Section 8.5: Transform coefficient decoding

Macroblock Types for I-slices (Table 7-11):
- I_NxN (0): 4x4 or 8x8 intra prediction
- I_16x16_x_x_x (1-24): 16x16 intra prediction with coded pattern
- I_PCM (25): Raw samples, no prediction/transform

I_16x16 type encoding:
- mb_type = 1 + intra16x16_pred_mode + 4*cbp_chroma + 12*(cbp_luma != 0)
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, List

import numpy as np

from bitstream import BitReader
from parameters import SPS, PPS
from dequant import dequant_4x4, dequant_dc_4x4, dequant_dc_2x2, get_chroma_qp
from transform import idct_4x4, hadamard_4x4, hadamard_2x2
from intra import predict_intra_16x16, Intra16x16Mode, predict_intra_4x4
from entropy import decode_residual_block, calculate_nC, ZIGZAG_4x4

logger = logging.getLogger(__name__)


class MBType(IntEnum):
    """Macroblock type for I-slices."""
    I_NxN = 0      # I_4x4 in baseline (no 8x8 transform)
    I_16x16 = 1    # I_16x16 (types 1-24 encode prediction mode + CBP)
    I_PCM = 25     # Raw samples


@dataclass
class CodedBlockPattern:
    """Coded block pattern indicating non-zero coefficients.

    Attributes:
        luma: 4-bit pattern for 8x8 luma blocks (bit i = block i has coeffs)
        chroma: 0=none, 1=DC only, 2=DC+AC
    """
    luma: int = 0
    chroma: int = 0

    @property
    def has_luma_dc(self) -> bool:
        """Check if any luma block has coefficients."""
        return self.luma != 0

    @property
    def has_chroma_dc(self) -> bool:
        """Check if chroma has DC coefficients."""
        return self.chroma >= 1

    @property
    def has_chroma_ac(self) -> bool:
        """Check if chroma has AC coefficients."""
        return self.chroma >= 2

    def has_luma_8x8(self, block_idx: int) -> bool:
        """Check if 8x8 block has coefficients."""
        return (self.luma >> block_idx) & 1 == 1


@dataclass
class MacroblockData:
    """Decoded macroblock data.

    Attributes:
        mb_type: Macroblock type
        intra_16x16_pred_mode: Prediction mode for I_16x16 (0-3)
        intra_4x4_pred_modes: Prediction modes for I_4x4 (16 modes)
        intra_chroma_pred_mode: Chroma prediction mode (0-3)
        cbp: Coded block pattern
        mb_qp_delta: QP adjustment for this MB
        luma: Reconstructed 16x16 luma samples
        cb: Reconstructed 8x8 Cb samples
        cr: Reconstructed 8x8 Cr samples
    """
    mb_type: int = 0
    intra_16x16_pred_mode: int = 0
    intra_4x4_pred_modes: List[int] = field(default_factory=list)
    intra_chroma_pred_mode: int = 0
    cbp: CodedBlockPattern = field(default_factory=CodedBlockPattern)
    mb_qp_delta: int = 0

    # Reconstructed samples
    luma: np.ndarray = field(default_factory=lambda: np.zeros((16, 16), dtype=np.uint8))
    cb: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=np.uint8))
    cr: np.ndarray = field(default_factory=lambda: np.zeros((8, 8), dtype=np.uint8))

    # Coefficient counts for CAVLC context (16 luma + 4 Cb + 4 Cr)
    nz_counts: np.ndarray = field(default_factory=lambda: np.zeros(24, dtype=np.int32))


def decode_i16x16_mb_type(mb_type: int) -> Tuple[int, int, int]:
    """Decode I_16x16 macroblock type into components.

    Args:
        mb_type: Raw mb_type value (1-24)

    Returns:
        Tuple of (intra_16x16_pred_mode, cbp_luma, cbp_chroma)

    H.264 Spec: Table 7-11
    mb_type = 1 + pred_mode + 4*cbp_chroma + 12*(cbp_luma != 0)
    """
    if mb_type < 1 or mb_type > 24:
        raise ValueError(f"Invalid I_16x16 mb_type: {mb_type}")

    code = mb_type - 1

    # cbp_luma: 0 or 15 (all blocks)
    cbp_luma = 15 if code >= 12 else 0
    code = code % 12

    # cbp_chroma: 0, 1 (DC only), or 2 (DC+AC)
    cbp_chroma = code // 4

    # pred_mode: 0-3
    pred_mode = code % 4

    return pred_mode, cbp_luma, cbp_chroma


# CBP lookup table for Intra macroblocks (H.264 Table 9-4)
# Maps codeNum from me(v) to (cbp_luma, cbp_chroma)
# cbp_luma: 4-bit pattern (0-15) indicating which 8x8 blocks have coefficients
# cbp_chroma: 0=none, 1=DC only, 2=DC+AC
#
# The table is designed so common patterns (all/no coefficients) have shortest codes.
# Combined cbp = (chroma << 4) | luma
CBP_INTRA_TABLE = [
    # codeNum 0-3: highest probability patterns
    (15, 2),  # 0: cbp=47, all luma + chroma DC+AC
    (15, 1),  # 1: cbp=31, all luma + chroma DC only
    (15, 0),  # 2: cbp=15, all luma, no chroma
    (0, 0),   # 3: cbp=0, no coefficients

    # codeNum 4-7: 3 of 4 luma blocks + chroma DC
    (7, 1),   # 4: cbp=23
    (11, 1),  # 5: cbp=27
    (13, 1),  # 6: cbp=29
    (14, 1),  # 7: cbp=30

    # codeNum 8-11: 3 of 4 luma blocks, no chroma
    (7, 0),   # 8: cbp=7
    (11, 0),  # 9: cbp=11
    (13, 0),  # 10: cbp=13
    (14, 0),  # 11: cbp=14

    # codeNum 12-15: 3 of 4 luma blocks + chroma DC+AC
    (7, 2),   # 12: cbp=39
    (11, 2),  # 13: cbp=43
    (13, 2),  # 14: cbp=45
    (14, 2),  # 15: cbp=46

    # codeNum 16: no luma, chroma DC only
    (0, 1),   # 16: cbp=16

    # codeNum 17-20: 2 of 4 luma blocks, no chroma
    (3, 0),   # 17: cbp=3
    (5, 0),   # 18: cbp=5
    (10, 0),  # 19: cbp=10
    (12, 0),  # 20: cbp=12

    # codeNum 21-24: 2 of 4 luma blocks + chroma DC
    (3, 1),   # 21: cbp=19
    (5, 1),   # 22: cbp=21
    (10, 1),  # 23: cbp=26
    (12, 1),  # 24: cbp=28

    # codeNum 25-28: 2 of 4 luma blocks + chroma DC+AC
    (3, 2),   # 25: cbp=35
    (5, 2),   # 26: cbp=37
    (10, 2),  # 27: cbp=42
    (12, 2),  # 28: cbp=44

    # codeNum 29-32: 1 of 4 luma blocks, no chroma
    (1, 0),   # 29: cbp=1
    (2, 0),   # 30: cbp=2
    (4, 0),   # 31: cbp=4
    (8, 0),   # 32: cbp=8

    # codeNum 33-36: 1 of 4 luma blocks + chroma DC
    (1, 1),   # 33: cbp=17
    (2, 1),   # 34: cbp=18
    (4, 1),   # 35: cbp=20
    (8, 1),   # 36: cbp=24

    # codeNum 37-40: 2 adjacent luma blocks
    (6, 0),   # 37: cbp=6
    (9, 0),   # 38: cbp=9
    (6, 1),   # 39: cbp=22
    (9, 1),   # 40: cbp=25

    # codeNum 41-45: no luma or 1 luma + chroma DC+AC
    (0, 2),   # 41: cbp=32
    (1, 2),   # 42: cbp=33
    (2, 2),   # 43: cbp=34
    (4, 2),   # 44: cbp=36
    (8, 2),   # 45: cbp=40

    # codeNum 46-47: 2 adjacent luma + chroma DC+AC
    (6, 2),   # 46: cbp=38
    (9, 2),   # 47: cbp=41
]


def decode_cbp_intra(coded_value: int) -> Tuple[int, int]:
    """Decode coded_block_pattern for intra macroblocks.

    Args:
        coded_value: ue(v) coded value

    Returns:
        Tuple of (cbp_luma, cbp_chroma)
    """
    if coded_value < len(CBP_INTRA_TABLE):
        return CBP_INTRA_TABLE[coded_value]
    return 0, 0


def _get_4x4_block_position(block_idx: int) -> Tuple[int, int]:
    """Get top-left position of 4x4 block within 16x16 MB.

    Block layout (raster scan within 8x8, then 8x8 blocks):
    0  1  4  5
    2  3  6  7
    8  9  12 13
    10 11 14 15
    """
    # 8x8 block index
    block_8x8 = block_idx // 4
    # 4x4 within 8x8
    sub_idx = block_idx % 4

    # 8x8 position
    row_8x8 = (block_8x8 // 2) * 8
    col_8x8 = (block_8x8 % 2) * 8

    # 4x4 position within 8x8
    row_4x4 = (sub_idx // 2) * 4
    col_4x4 = (sub_idx % 2) * 4

    return row_8x8 + row_4x4, col_8x8 + col_4x4


def _clip_pixel(value: int) -> int:
    """Clip pixel value to valid range [0, 255]."""
    return max(0, min(255, value))


def _clip_block(block: np.ndarray) -> np.ndarray:
    """Clip block values to valid range."""
    return np.clip(block, 0, 255).astype(np.uint8)


# Block scan order for 4x4 blocks within 16x16 macroblock
# Uses raster scan within 8x8 blocks, then 8x8 blocks in raster order
# Block layout:
#   0  1  4  5
#   2  3  6  7
#   8  9  12 13
#   10 11 14 15
BLOCK_SCAN_ORDER = [
    (0, 0), (0, 4), (4, 0), (4, 4),       # 8x8 block 0 (top-left)
    (0, 8), (0, 12), (4, 8), (4, 12),     # 8x8 block 1 (top-right)
    (8, 0), (8, 4), (12, 0), (12, 4),     # 8x8 block 2 (bottom-left)
    (8, 8), (8, 12), (12, 8), (12, 12),   # 8x8 block 3 (bottom-right)
]


def decode_intra4x4_pred_modes(
    reader: BitReader,
    neighbor_modes: Optional[List[int]] = None,
) -> List[int]:
    """Decode 16 Intra_4x4 prediction modes from bitstream.

    H.264 Spec 7.3.5.1, 8.3.1.1:
    For each 4x4 block:
    - prev_intra4x4_pred_mode_flag: if 1, use predicted mode
    - rem_intra4x4_pred_mode: if flag=0, provides mode (3 bits)

    The predicted mode is min(modeA, modeB) where A and B are
    left and top neighbor block modes.

    Args:
        reader: Bit reader positioned at prediction mode data
        neighbor_modes: Modes from neighboring MBs (for blocks on edge)

    Returns:
        List of 16 prediction modes (0-8)
    """
    modes = []
    # Track modes as we decode for prediction
    decoded_modes = {}  # block_idx -> mode

    for block_idx in range(16):
        # Get predicted mode from neighbors
        row, col = BLOCK_SCAN_ORDER[block_idx]

        # Find left neighbor mode
        if col > 0:
            # Left neighbor is within this MB
            left_idx = _find_block_at_position(row, col - 4)
            mode_a = decoded_modes.get(left_idx, 2)  # Default DC
        else:
            # Left neighbor from external MB (or unavailable)
            mode_a = 2  # DC as default

        # Find top neighbor mode
        if row > 0:
            # Top neighbor is within this MB
            top_idx = _find_block_at_position(row - 4, col)
            mode_b = decoded_modes.get(top_idx, 2)  # Default DC
        else:
            # Top neighbor from external MB (or unavailable)
            mode_b = 2  # DC as default

        predicted_mode = min(mode_a, mode_b)

        # Read flag
        prev_flag = reader.read_bits(1)

        if prev_flag == 1:
            mode = predicted_mode
        else:
            rem = reader.read_bits(3)
            if rem < predicted_mode:
                mode = rem
            else:
                mode = rem + 1

        modes.append(mode)
        decoded_modes[block_idx] = mode

    return modes


def _find_block_at_position(row: int, col: int) -> int:
    """Find block index for given (row, col) position."""
    for idx, (r, c) in enumerate(BLOCK_SCAN_ORDER):
        if r == row and c == col:
            return idx
    return -1


def get_4x4_block_neighbors(
    frame: np.ndarray,
    block_row: int,
    block_col: int,
    mb_row: int,
    mb_col: int,
    frame_width_mbs: int = 0,
) -> dict:
    """Get neighbor pixels for a 4x4 block.

    Args:
        frame: Reconstructed frame so far (or current MB buffer)
        block_row: Row position of block within MB (0, 4, 8, or 12)
        block_col: Column position of block within MB (0, 4, 8, or 12)
        mb_row: Macroblock row in frame
        mb_col: Macroblock column in frame
        frame_width_mbs: Frame width in macroblocks

    Returns:
        Dict with:
        - top: 4 pixels above (or None)
        - left: 4 pixels to left (or None)
        - top_left: corner pixel (or None)
        - top_right: 4 pixels above-right (or None)
        - top_available, left_available, top_right_available: booleans
    """
    result = {
        'top': None,
        'left': None,
        'top_left': None,
        'top_right': None,
        'top_available': False,
        'left_available': False,
        'top_right_available': False,
    }

    frame_height, frame_width = frame.shape
    # Absolute position in frame
    abs_row = mb_row * 16 + block_row
    abs_col = mb_col * 16 + block_col

    # Top neighbors
    if abs_row > 0:
        result['top'] = frame[abs_row - 1, abs_col:abs_col + 4].copy()
        result['top_available'] = True

    # Left neighbors
    if abs_col > 0:
        result['left'] = frame[abs_row:abs_row + 4, abs_col - 1].copy()
        result['left_available'] = True

    # Top-left neighbor
    if abs_row > 0 and abs_col > 0:
        result['top_left'] = int(frame[abs_row - 1, abs_col - 1])

    # Top-right neighbors (4 pixels above and to the right)
    if abs_row > 0 and abs_col + 4 < frame_width:
        # Check if top-right block is available
        # Top-right is not available for rightmost block in each 8x8
        if block_col < 12 or (block_col == 12 and block_row >= 4):
            # More complex availability check needed per spec
            result['top_right'] = frame[abs_row - 1, abs_col + 4:abs_col + 8].copy()
            if len(result['top_right']) == 4:
                result['top_right_available'] = True

    return result


def reconstruct_i4x4_block(
    mode: int,
    residual: np.ndarray,
    top: Optional[np.ndarray],
    left: Optional[np.ndarray],
    top_left: Optional[int],
    top_right: Optional[np.ndarray],
    top_available: bool,
    left_available: bool,
    top_right_available: bool,
) -> np.ndarray:
    """Reconstruct a single 4x4 block.

    Args:
        mode: Intra 4x4 prediction mode (0-8)
        residual: 4x4 residual block (from IDCT)
        top, left, top_left, top_right: Neighbor pixels
        *_available: Neighbor availability flags

    Returns:
        Reconstructed 4x4 block (uint8)
    """
    from intra import predict_intra_4x4

    # Generate prediction
    prediction = predict_intra_4x4(
        mode=mode,
        top=top,
        left=left,
        top_left=top_left,
        top_right=top_right,
        top_available=top_available,
        left_available=left_available,
        top_right_available=top_right_available,
    )

    # Add residual
    result = prediction.astype(np.int32) + residual

    # Clip to valid range
    return _clip_block(result)


def reconstruct_i4x4_luma(
    modes: List[int],
    residuals: List[np.ndarray],
    neighbors_top: Optional[np.ndarray],
    neighbors_left: Optional[np.ndarray],
    neighbor_top_left: Optional[int],
) -> np.ndarray:
    """Reconstruct I_4x4 (I_NxN) luma macroblock.

    Processes 16 4x4 blocks in scan order, using previously
    reconstructed blocks as neighbors for later blocks.

    Args:
        modes: 16 prediction modes (one per 4x4 block)
        residuals: 16 residual blocks (4x4 each)
        neighbors_top: Top neighbor row (16 pixels) or None
        neighbors_left: Left neighbor column (16 pixels) or None
        neighbor_top_left: Top-left corner pixel or None

    Returns:
        Reconstructed 16x16 luma block
    """
    from intra import predict_intra_4x4

    # Initialize output buffer
    luma = np.zeros((16, 16), dtype=np.uint8)
    DEFAULT_VALUE = 128

    for block_idx in range(16):
        row, col = BLOCK_SCAN_ORDER[block_idx]
        mode = modes[block_idx]
        residual = residuals[block_idx]

        # Determine neighbor availability and values
        # Top neighbors
        if row == 0:
            # Use external top neighbors
            if neighbors_top is not None:
                top = neighbors_top[col:col + 4].copy()
                top_available = True
            else:
                top = None
                top_available = False
        else:
            # Use from reconstructed luma buffer
            top = luma[row - 1, col:col + 4].copy()
            top_available = True

        # Left neighbors
        if col == 0:
            # Use external left neighbors
            if neighbors_left is not None:
                left = neighbors_left[row:row + 4].copy()
                left_available = True
            else:
                left = None
                left_available = False
        else:
            # Use from reconstructed luma buffer
            left = luma[row:row + 4, col - 1].copy()
            left_available = True

        # Top-left neighbor
        if row == 0 and col == 0:
            top_left = neighbor_top_left
        elif row == 0:
            # Top-left is from external top
            if neighbors_top is not None and col > 0:
                top_left = int(neighbors_top[col - 1])
            else:
                top_left = None
        elif col == 0:
            # Top-left is from external left
            if neighbors_left is not None and row > 0:
                top_left = int(neighbors_left[row - 1])
            else:
                top_left = None
        else:
            # Top-left is from reconstructed buffer
            top_left = int(luma[row - 1, col - 1])

        # Top-right neighbors
        top_right = None
        top_right_available = False
        if row == 0 and col + 4 <= 12:
            # External top-right
            if neighbors_top is not None:
                top_right = neighbors_top[col + 4:col + 8].copy()
                if len(top_right) == 4:
                    top_right_available = True
        elif row > 0 and col + 4 < 16:
            # Internal top-right - but need to check if that block
            # is already decoded (it's to the right, so not yet)
            # Top-right comes from row above
            if _is_top_right_available(block_idx):
                top_right = luma[row - 1, col + 4:col + 8].copy()
                if len(top_right) == 4:
                    top_right_available = True

        # Generate prediction
        prediction = predict_intra_4x4(
            mode=mode,
            top=top,
            left=left,
            top_left=top_left,
            top_right=top_right,
            top_available=top_available,
            left_available=left_available,
            top_right_available=top_right_available,
        )

        # Add residual and clip
        result = prediction.astype(np.int32) + residual
        luma[row:row + 4, col:col + 4] = _clip_block(result)

    return luma


def _is_top_right_available(block_idx: int) -> bool:
    """Check if top-right neighbor is available for a block.

    H.264 Spec 6.4.4: Top-right is not available for:
    - Blocks 3, 7, 11, 15 (rightmost in each row of 8x8)
    - Block 13 (special case in bottom-right 8x8)
    """
    # Blocks where top-right is NOT available
    unavailable = {3, 7, 11, 13, 15}
    return block_idx not in unavailable


def _get_i4x4_neighbors(
    frame_luma: np.ndarray,
    mb_luma: np.ndarray,
    mb_x: int,
    mb_y: int,
    row: int,
    col: int,
    block_idx: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int], Optional[np.ndarray]]:
    """Get neighbor pixels for I_4x4 prediction.

    Combines pixels from:
    - Frame buffer (for neighbors outside this MB)
    - Already-reconstructed blocks in current MB

    Args:
        frame_luma: Full frame luma buffer
        mb_luma: Current macroblock's partially reconstructed luma
        mb_x, mb_y: Macroblock position
        row, col: 4x4 block position within MB (0,4,8,12)
        block_idx: Block index (0-15)

    Returns:
        Tuple of (top_4, left_4, top_left, top_right_4)
        Each may be None if unavailable
    """
    luma_y = mb_y * 16
    luma_x = mb_x * 16

    # Absolute position in frame
    abs_y = luma_y + row
    abs_x = luma_x + col

    # Top neighbors (4 pixels above this block)
    if row > 0:
        # Inside MB - use already reconstructed
        top = mb_luma[row - 1, col:col + 4].copy()
    elif mb_y > 0:
        # Use frame buffer
        top = frame_luma[abs_y - 1, abs_x:abs_x + 4].copy()
    else:
        top = None

    # Left neighbors (4 pixels to the left)
    if col > 0:
        # Inside MB
        left = mb_luma[row:row + 4, col - 1].copy()
    elif mb_x > 0:
        # Use frame buffer
        left = frame_luma[abs_y:abs_y + 4, abs_x - 1].copy()
    else:
        left = None

    # Top-left corner
    if row > 0 and col > 0:
        # Inside MB
        top_left = int(mb_luma[row - 1, col - 1])
    elif row > 0 and col == 0 and mb_x > 0:
        # Left edge of MB, top-left is in frame buffer
        top_left = int(frame_luma[abs_y - 1, abs_x - 1])
    elif row == 0 and col > 0 and mb_y > 0:
        # Top edge of MB, top-left is in frame buffer
        top_left = int(frame_luma[abs_y - 1, abs_x - 1])
    elif row == 0 and col == 0 and mb_y > 0 and mb_x > 0:
        # Corner of MB
        top_left = int(frame_luma[abs_y - 1, abs_x - 1])
    else:
        top_left = None

    # Top-right neighbors (4 pixels above and to the right)
    # Availability depends on block position (H.264 6.4.4)
    top_right = None
    if _is_top_right_available(block_idx):
        if row > 0 and col + 4 < 16:
            # Inside MB
            top_right = mb_luma[row - 1, col + 4:col + 8].copy()
        elif row == 0 and mb_y > 0:
            # Top edge - check if we can get from frame
            if abs_x + 4 < frame_luma.shape[1]:
                top_right = frame_luma[abs_y - 1, abs_x + 4:abs_x + 8].copy()

    # If top_right not available but top is, extend with last top pixel
    if top_right is None and top is not None:
        top_right = np.full(4, top[-1], dtype=top.dtype)

    return top, left, top_left, top_right


def reconstruct_i16x16_luma(
    reader: BitReader,
    pred_mode: int,
    cbp_luma: int,
    qp: int,
    neighbors_top: Optional[np.ndarray],
    neighbors_left: Optional[np.ndarray],
    neighbor_top_left: Optional[int],
    nz_counts: np.ndarray
) -> np.ndarray:
    """Reconstruct I_16x16 luma macroblock.

    Args:
        reader: Bit reader for coefficient data
        pred_mode: Intra 16x16 prediction mode (0-3)
        cbp_luma: Luma CBP (0 or 15)
        qp: Quantization parameter
        neighbors_top: Top neighbor pixels (16,) or None
        neighbors_left: Left neighbor pixels (16,) or None
        neighbor_top_left: Top-left corner pixel or None
        nz_counts: Array to store non-zero counts for context

    Returns:
        Reconstructed 16x16 luma block
    """
    # Step 1: Generate prediction
    # Infer availability from whether arrays are provided
    top_available = neighbors_top is not None
    left_available = neighbors_left is not None

    prediction = predict_intra_16x16(
        pred_mode,
        neighbors_top,
        neighbors_left,
        neighbor_top_left,
        top_available=top_available,
        left_available=left_available
    )

    # Step 2: Decode and reconstruct residual
    residual = np.zeros((16, 16), dtype=np.int32)

    # H.264 Spec 7.3.5.3: For I_16x16, DC block is ALWAYS coded
    # AC blocks are only coded when cbp_luma bits are set

    # Decode DC coefficients (Hadamard-coded 4x4 block) - ALWAYS for I_16x16
    dc_nC = calculate_nC(None, None)  # Simplified - no neighbor context
    dc_block = decode_residual_block(reader, dc_nC, max_coeffs=16)
    nz_counts[0] = dc_block.total_coeff  # Store for context

    # Arrange DC coefficients in 4x4 for inverse Hadamard
    # CAVLC decode_residual_block() already places coefficients in spatial order
    # using run_before values - no zigzag reordering needed here
    dc_4x4 = np.zeros((4, 4), dtype=np.int32)
    for i in range(16):
        row, col = i // 4, i % 4
        dc_4x4[row, col] = dc_block.coefficients[i]

    # Inverse Hadamard transform
    # Note: No division here - normalization is in dequant and IDCT
    dc_transformed = hadamard_4x4(dc_4x4)

    # Dequantize DC coefficients
    dc_dequant = dequant_dc_4x4(dc_transformed, qp)

    # Decode AC coefficients for each 4x4 block (only when cbp_luma != 0)
    if cbp_luma != 0:
        for block_idx in range(16):
            row, col = _get_4x4_block_position(block_idx)

            # Check if this 8x8 region has AC coefficients
            block_8x8_idx = block_idx // 4  # Which 8x8 quadrant
            if not (cbp_luma & (1 << block_8x8_idx)):
                continue

            # Decode AC (15 coefficients, DC is separate)
            ac_nC = calculate_nC(None, None)
            ac_block = decode_residual_block(reader, ac_nC, max_coeffs=15)
            nz_counts[block_idx] = ac_block.total_coeff

            # Build coefficient block with DC from Hadamard
            # DC array is in raster order: dc[i,j] corresponds to block at pixel (i*4, j*4)
            # Use spatial position, not block_idx scan order
            coeffs = np.zeros((4, 4), dtype=np.int32)
            dc_row, dc_col = row // 4, col // 4
            coeffs[0, 0] = dc_dequant[dc_row, dc_col]

            # Fill AC coefficients (zigzag order, starting from position 1)
            for i, pos in enumerate(ZIGZAG_4x4[1:]):  # Skip DC
                if i < len(ac_block.coefficients):
                    r, c = pos // 4, pos % 4
                    coeffs[r, c] = ac_block.coefficients[i]

            # Dequantize AC
            coeffs_dequant = dequant_4x4(coeffs, qp)
            coeffs_dequant[0, 0] = coeffs[0, 0]  # DC already dequantized

            # Inverse transform
            block_residual = idct_4x4(coeffs_dequant)

            # Place in residual
            residual[row:row+4, col:col+4] = block_residual

    # When cbp_luma=0: DC block was read (possibly all zeros), but no AC
    # Apply DC contribution to residual even when cbp_luma=0
    if cbp_luma == 0:
        for block_idx in range(16):
            row, col = _get_4x4_block_position(block_idx)
            # DC array is in raster order: dc[i,j] corresponds to block at pixel (i*4, j*4)
            dc_row, dc_col = row // 4, col // 4

            if dc_dequant[dc_row, dc_col] != 0:
                # Build coefficient block with only DC
                coeffs = np.zeros((4, 4), dtype=np.int32)
                coeffs[0, 0] = dc_dequant[dc_row, dc_col]

                # Inverse transform
                block_residual = idct_4x4(coeffs)
                residual[row:row+4, col:col+4] = block_residual

    # Step 3: Add residual to prediction and clip
    reconstructed = prediction.astype(np.int32) + residual
    return _clip_block(reconstructed)


def reconstruct_chroma(
    reader: BitReader,
    pred_mode: int,
    cbp_chroma: int,
    qp_chroma: int,
    neighbors_top: Optional[np.ndarray],
    neighbors_left: Optional[np.ndarray],
    neighbor_top_left: Optional[int],
    nz_counts: np.ndarray,
    nz_offset: int
) -> np.ndarray:
    """Reconstruct 8x8 chroma component.

    Args:
        reader: Bit reader
        pred_mode: Intra chroma prediction mode (0-3)
        cbp_chroma: Chroma CBP (0=none, 1=DC, 2=DC+AC)
        qp_chroma: Chroma QP
        neighbors_top: Top neighbor pixels (8,)
        neighbors_left: Left neighbor pixels (8,)
        neighbor_top_left: Top-left corner pixel
        nz_counts: Non-zero count array
        nz_offset: Offset into nz_counts for this component

    Returns:
        Reconstructed 8x8 chroma block
    """
    # Generate prediction (using DC mode for simplicity)
    # Full chroma prediction would use pred_mode
    if neighbors_top is not None and neighbors_left is not None:
        prediction = np.full((8, 8),
            (np.sum(neighbors_top) + np.sum(neighbors_left) + 8) >> 4,
            dtype=np.int32)
    elif neighbors_top is not None:
        prediction = np.full((8, 8), (np.sum(neighbors_top) + 4) >> 3, dtype=np.int32)
    elif neighbors_left is not None:
        prediction = np.full((8, 8), (np.sum(neighbors_left) + 4) >> 3, dtype=np.int32)
    else:
        prediction = np.full((8, 8), 128, dtype=np.int32)

    residual = np.zeros((8, 8), dtype=np.int32)

    if cbp_chroma >= 1:
        # Decode DC coefficients (2x2 block)
        dc_block = decode_residual_block(reader, nC=-1, max_coeffs=4)

        # Arrange and inverse Hadamard
        dc_2x2 = np.zeros((2, 2), dtype=np.int32)
        for i in range(min(4, len(dc_block.coefficients))):
            dc_2x2[i // 2, i % 2] = dc_block.coefficients[i]

        dc_transformed = hadamard_2x2(dc_2x2)
        dc_dequant = dequant_dc_2x2(dc_transformed, qp_chroma)

        if cbp_chroma >= 2:
            # Decode AC for each 4x4 block
            for block_idx in range(4):
                row = (block_idx // 2) * 4
                col = (block_idx % 2) * 4

                ac_nC = calculate_nC(None, None)
                ac_block = decode_residual_block(reader, ac_nC, max_coeffs=15)
                nz_counts[nz_offset + block_idx] = ac_block.total_coeff

                # Build coefficient block
                coeffs = np.zeros((4, 4), dtype=np.int32)
                coeffs[0, 0] = dc_dequant[block_idx // 2, block_idx % 2]

                for i, pos in enumerate(ZIGZAG_4x4[1:]):
                    if i < len(ac_block.coefficients):
                        r, c = pos // 4, pos % 4
                        coeffs[r, c] = ac_block.coefficients[i]

                coeffs_dequant = dequant_4x4(coeffs, qp_chroma)
                coeffs_dequant[0, 0] = coeffs[0, 0]

                block_residual = idct_4x4(coeffs_dequant)
                residual[row:row+4, col:col+4] = block_residual
        else:
            # DC only - place DC values
            for block_idx in range(4):
                row = (block_idx // 2) * 4
                col = (block_idx % 2) * 4
                dc_val = dc_dequant[block_idx // 2, block_idx % 2]
                residual[row:row+4, col:col+4] = dc_val

    reconstructed = prediction + residual
    return _clip_block(reconstructed)


def decode_macroblock(
    reader: BitReader,
    sps: SPS,
    pps: PPS,
    slice_qp: int,
    mb_x: int,
    mb_y: int,
    frame_luma: np.ndarray,
    frame_cb: np.ndarray,
    frame_cr: np.ndarray,
    is_i_slice: bool = True
) -> MacroblockData:
    """Decode and reconstruct a complete macroblock.

    Args:
        reader: Bit reader positioned at MB data
        sps: Sequence parameter set
        pps: Picture parameter set
        slice_qp: Current slice QP
        mb_x: MB column index
        mb_y: MB row index
        frame_luma: Frame luma buffer to write to
        frame_cb: Frame Cb buffer
        frame_cr: Frame Cr buffer
        is_i_slice: Whether this is an I-slice

    Returns:
        Decoded macroblock data
    """
    mb = MacroblockData()

    # Parse mb_type
    mb.mb_type = reader.read_ue()

    logger.debug(f"Decoding MB ({mb_x}, {mb_y}): type={mb.mb_type}")

    # Handle I_PCM specially
    if mb.mb_type == MBType.I_PCM:
        reader.byte_align()
        # Read raw samples
        for y in range(16):
            for x in range(16):
                mb.luma[y, x] = reader.read_bits(8)
        for y in range(8):
            for x in range(8):
                mb.cb[y, x] = reader.read_bits(8)
        for y in range(8):
            for x in range(8):
                mb.cr[y, x] = reader.read_bits(8)
        return mb

    # Determine MB properties based on type
    if mb.mb_type == MBType.I_NxN:
        # I_4x4: Parse prediction modes using proper neighbor-based prediction
        # decode_intra4x4_pred_modes handles:
        #   - prev_intra4x4_pred_mode_flag and rem_intra4x4_pred_mode parsing
        #   - predicted mode calculation from neighbors (min of left and top)
        #   - mode mapping formula: rem < pred ? rem : rem + 1
        mb.intra_4x4_pred_modes = decode_intra4x4_pred_modes(reader, neighbor_modes=None)

        # Parse coded_block_pattern
        cbp_coded = reader.read_ue()
        cbp_luma, cbp_chroma = decode_cbp_intra(cbp_coded)
        mb.cbp = CodedBlockPattern(luma=cbp_luma, chroma=cbp_chroma)

    else:
        # I_16x16: Extract pred_mode and CBP from mb_type
        pred_mode, cbp_luma, cbp_chroma = decode_i16x16_mb_type(mb.mb_type)
        mb.intra_16x16_pred_mode = pred_mode
        mb.cbp = CodedBlockPattern(luma=cbp_luma, chroma=cbp_chroma)

    # Parse intra_chroma_pred_mode
    mb.intra_chroma_pred_mode = reader.read_ue()

    # Parse mb_qp_delta if any coded coefficients OR if I_16x16
    # Per H.264 spec 7.3.5: mb_qp_delta is present if:
    #   CodedBlockPatternLuma > 0 OR CodedBlockPatternChroma > 0 OR
    #   MbPartPredMode(mb_type,0) == Intra_16x16
    # For I_16x16, mb_qp_delta is ALWAYS coded, even with cbp=0
    is_i16x16 = 1 <= mb.mb_type <= 24
    if mb.cbp.has_luma_dc or mb.cbp.has_chroma_dc or is_i16x16:
        mb.mb_qp_delta = reader.read_se()

    # Calculate QP
    qp = slice_qp + mb.mb_qp_delta
    qp = max(0, min(51, qp))
    qp_chroma = get_chroma_qp(qp)

    logger.debug(f"MB QP: {qp}, chroma QP: {qp_chroma}, CBP: luma={mb.cbp.luma}, chroma={mb.cbp.chroma}")

    # Get neighbor pixels for prediction
    luma_y = mb_y * 16
    luma_x = mb_x * 16
    chroma_y = mb_y * 8
    chroma_x = mb_x * 8

    # Top neighbors
    neighbors_top_luma = frame_luma[luma_y - 1, luma_x:luma_x + 16] if mb_y > 0 else None
    neighbors_top_cb = frame_cb[chroma_y - 1, chroma_x:chroma_x + 8] if mb_y > 0 else None
    neighbors_top_cr = frame_cr[chroma_y - 1, chroma_x:chroma_x + 8] if mb_y > 0 else None

    # Left neighbors
    neighbors_left_luma = frame_luma[luma_y:luma_y + 16, luma_x - 1] if mb_x > 0 else None
    neighbors_left_cb = frame_cb[chroma_y:chroma_y + 8, chroma_x - 1] if mb_x > 0 else None
    neighbors_left_cr = frame_cr[chroma_y:chroma_y + 8, chroma_x - 1] if mb_x > 0 else None

    # Top-left corner
    neighbor_tl_luma = frame_luma[luma_y - 1, luma_x - 1] if mb_y > 0 and mb_x > 0 else None
    neighbor_tl_cb = frame_cb[chroma_y - 1, chroma_x - 1] if mb_y > 0 and mb_x > 0 else None
    neighbor_tl_cr = frame_cr[chroma_y - 1, chroma_x - 1] if mb_y > 0 and mb_x > 0 else None

    # Reconstruct luma
    if mb.mb_type >= 1 and mb.mb_type <= 24:
        # I_16x16
        mb.luma = reconstruct_i16x16_luma(
            reader,
            mb.intra_16x16_pred_mode,
            mb.cbp.luma,
            qp,
            neighbors_top_luma,
            neighbors_left_luma,
            neighbor_tl_luma,
            mb.nz_counts
        )
    else:
        # I_4x4: Full reconstruction with 9 prediction modes per block
        # Process blocks in scan order, each using previously reconstructed neighbors
        mb.luma = np.zeros((16, 16), dtype=np.uint8)

        for block_idx in range(16):
            row, col = BLOCK_SCAN_ORDER[block_idx]

            # Get neighbor pixels for this 4x4 block
            # Must use reconstructed pixels from: frame buffer + already-done blocks in this MB
            top, left, top_left, top_right = _get_i4x4_neighbors(
                frame_luma, mb.luma, mb_x, mb_y, row, col, block_idx
            )

            # Generate prediction using the decoded mode
            mode = mb.intra_4x4_pred_modes[block_idx]
            pred = predict_intra_4x4(mode, top, left, top_left, top_right)

            # Check if this 8x8 region has coefficients
            block_8x8_idx = block_idx // 4  # 0-3 (which quadrant)
            if mb.cbp.has_luma_8x8(block_8x8_idx):
                # Decode residual coefficients (all 16, no separate DC Hadamard for I_4x4)
                nC = calculate_nC(None, None)  # TODO: use proper neighbor nC
                block = decode_residual_block(reader, nC, max_coeffs=16)
                mb.nz_counts[block_idx] = block.total_coeff

                # Dequantize and inverse transform
                if block.total_coeff > 0:
                    # CAVLC already placed coefficients in scan order
                    # Apply zigzag to convert scan position -> raster position
                    coeffs_2d = np.zeros((4, 4), dtype=np.int32)
                    for scan_idx in range(16):
                        raster_pos = ZIGZAG_4x4[scan_idx]
                        r, c = raster_pos // 4, raster_pos % 4
                        coeffs_2d[r, c] = block.coefficients[scan_idx]
                    dequant = dequant_4x4(coeffs_2d, qp)
                    residual = idct_4x4(dequant)
                else:
                    residual = np.zeros((4, 4), dtype=np.int32)
            else:
                residual = np.zeros((4, 4), dtype=np.int32)
                mb.nz_counts[block_idx] = 0

            # Add prediction + residual and clip
            result = pred.astype(np.int32) + residual
            result = np.clip(result, 0, 255).astype(np.uint8)

            # Store in macroblock luma
            mb.luma[row:row+4, col:col+4] = result

    # Reconstruct chroma
    mb.cb = reconstruct_chroma(
        reader,
        mb.intra_chroma_pred_mode,
        mb.cbp.chroma,
        qp_chroma,
        neighbors_top_cb,
        neighbors_left_cb,
        neighbor_tl_cb,
        mb.nz_counts,
        16  # Cb offset
    )

    mb.cr = reconstruct_chroma(
        reader,
        mb.intra_chroma_pred_mode,
        mb.cbp.chroma,
        qp_chroma,
        neighbors_top_cr,
        neighbors_left_cr,
        neighbor_tl_cr,
        mb.nz_counts,
        20  # Cr offset
    )

    # Write to frame buffers
    frame_luma[luma_y:luma_y + 16, luma_x:luma_x + 16] = mb.luma
    frame_cb[chroma_y:chroma_y + 8, chroma_x:chroma_x + 8] = mb.cb
    frame_cr[chroma_y:chroma_y + 8, chroma_x:chroma_x + 8] = mb.cr

    logger.debug(f"Reconstructed MB ({mb_x}, {mb_y}): luma mean={np.mean(mb.luma):.1f}")

    return mb
