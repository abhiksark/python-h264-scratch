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
from intra import predict_intra_16x16, Intra16x16Mode
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

    # Arrange DC coefficients in 4x4 and apply inverse Hadamard
    # Use raster order for coefficient placement
    # NOTE: This works for uniform-color MBs. Complex gradients may need investigation.
    dc_4x4 = np.zeros((4, 4), dtype=np.int32)
    for i in range(min(16, len(dc_block.coefficients))):
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
        # I_4x4: Parse 4x4 prediction modes
        # Each block has prev_intra4x4_pred_mode_flag, and optionally rem_intra4x4_pred_mode
        mb.intra_4x4_pred_modes = []
        for _ in range(16):
            prev_flag = reader.read_bit()
            if prev_flag:
                # Use predicted mode (DC for now since we don't track neighbors)
                mb.intra_4x4_pred_modes.append(2)  # DC mode
            else:
                # Read remaining prediction mode (3 bits)
                rem_mode = reader.read_bits(3)
                mb.intra_4x4_pred_modes.append(rem_mode)

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
        # I_4x4 - simplified: decode coefficients to keep sync, use DC prediction
        prediction = np.full((16, 16), 128, dtype=np.int32)
        if neighbors_top_luma is not None:
            prediction[:8, :] = int(np.mean(neighbors_top_luma))
        if neighbors_left_luma is not None:
            prediction[:, :8] = int(np.mean(neighbors_left_luma))

        # Read coefficients for each 4x4 block to keep bitstream in sync
        # I_4x4 has all 16 coefficients per block (DC included, no separate Hadamard)
        for block_idx in range(16):
            # Check if this 8x8 region has coefficients
            block_8x8_idx = (block_idx // 4)  # 0-3
            if mb.cbp.has_luma_8x8(block_8x8_idx):
                # Read the 4x4 block coefficients (all 16, DC included)
                nC = calculate_nC(None, None)  # Simplified
                block = decode_residual_block(reader, nC, max_coeffs=16)
                mb.nz_counts[block_idx] = block.total_coeff

        mb.luma = _clip_block(prediction)

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
