# h264/dequant/dequant.py
"""Inverse quantization (dequantization) for H.264.

H.264 uses a position-dependent scaling approach where different positions
in the 4x4 block have different scaling factors. This is combined with
the QP (Quantization Parameter) to determine final scaling.

H.264 Spec Reference: Section 8.5.11 - Scaling and transformation process
Table 8-13: LevelScale lookup table

QP ranges from 0 to 51:
- Lower QP = less compression, higher quality
- Higher QP = more compression, lower quality
- QP increases by 6 doubles the quantization step size
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _rshift_toward_zero(x: np.ndarray, n: int) -> np.ndarray:
    """Right shift with truncation toward zero (C-style semantics).

    Python's >> truncates toward -∞, C truncates toward 0.
    H.264 spec uses C semantics.
    """
    return np.where(x >= 0, x >> n, -(-x >> n))


# LevelScale lookup table from H.264 spec Table 8-13
# Indexed by [qp % 6][position_type]
# position_type: 0 = (0,0), (0,2), (2,0), (2,2) - corners
#                1 = (1,1), (1,3), (3,1), (3,3) - center
#                2 = other positions
LEVEL_SCALE = np.array([
    [10, 16, 13],  # qp % 6 == 0
    [11, 18, 14],  # qp % 6 == 1
    [13, 20, 16],  # qp % 6 == 2
    [14, 23, 18],  # qp % 6 == 3
    [16, 25, 20],  # qp % 6 == 4
    [18, 29, 23],  # qp % 6 == 5
], dtype=np.int32)


# Position type map for 4x4 block
# 0 = corner, 1 = center diagonal, 2 = other
POSITION_TYPE = np.array([
    [0, 2, 0, 2],
    [2, 1, 2, 1],
    [0, 2, 0, 2],
    [2, 1, 2, 1],
], dtype=np.int32)


# 8x8 LevelScale lookup table (H.264 Table 8-14)
# Indexed by [qp % 6][position_type] where position_type is 0-5 for 8x8
LEVEL_SCALE_8x8 = np.array([
    [20, 18, 32, 19, 25, 24],  # qp % 6 == 0
    [22, 19, 35, 21, 28, 26],  # qp % 6 == 1
    [26, 23, 42, 24, 33, 31],  # qp % 6 == 2
    [28, 25, 45, 26, 35, 33],  # qp % 6 == 3
    [32, 28, 51, 30, 40, 38],  # qp % 6 == 4
    [36, 32, 58, 34, 46, 43],  # qp % 6 == 5
], dtype=np.int32)


# Position type map for 8x8 block (H.264 Table 8-15)
# 6 position types for 8x8 blocks
POSITION_TYPE_8x8 = np.array([
    [0, 3, 4, 3, 0, 3, 4, 3],
    [3, 1, 5, 1, 3, 1, 5, 1],
    [4, 5, 2, 5, 4, 5, 2, 5],
    [3, 1, 5, 1, 3, 1, 5, 1],
    [0, 3, 4, 3, 0, 3, 4, 3],
    [3, 1, 5, 1, 3, 1, 5, 1],
    [4, 5, 2, 5, 4, 5, 2, 5],
    [3, 1, 5, 1, 3, 1, 5, 1],
], dtype=np.int32)


def get_scale_matrix(qp: int) -> np.ndarray:
    """Get the 4x4 scaling matrix for a given QP.

    Args:
        qp: Quantization parameter (0-51)

    Returns:
        4x4 scaling matrix (int32)
    """
    qp_mod_6 = qp % 6
    scales = LEVEL_SCALE[qp_mod_6]

    # Build scale matrix based on position types
    scale_matrix = np.zeros((4, 4), dtype=np.int32)
    for i in range(4):
        for j in range(4):
            pos_type = POSITION_TYPE[i, j]
            scale_matrix[i, j] = scales[pos_type]

    return scale_matrix


def dequant_4x4(
    coeffs: np.ndarray,
    qp: int,
    is_intra: bool = True
) -> np.ndarray:
    """Dequantize a 4x4 coefficient block.

    Applies the inverse quantization scaling to transform coefficients.

    Args:
        coeffs: 4x4 quantized coefficients (int32)
        qp: Quantization parameter (0-51)
        is_intra: Whether this is an intra block (affects scaling)

    Returns:
        4x4 dequantized coefficients (int32)

    H.264 Spec: Section 8.5.11
    Formula: dequant[i,j] = (coeffs[i,j] * LevelScale[qp%6][pos] * scale) >> shift
    where scale depends on qp // 6
    """
    if coeffs.shape != (4, 4):
        raise ValueError(f"Expected 4x4 block, got {coeffs.shape}")

    qp = max(0, min(51, qp))  # Clamp QP to valid range

    qp_div_6 = qp // 6
    qp_mod_6 = qp % 6

    logger.debug(f"Dequantizing with QP={qp} (div6={qp_div_6}, mod6={qp_mod_6})")
    logger.debug(f"Input coefficients:\n{coeffs}")

    # Get scale matrix
    scale_matrix = get_scale_matrix(qp)

    # Apply dequantization
    # H.264 Spec Section 8.5.12.1: For default flat scaling list (qmatrix=16),
    # the formula simplifies to: d[i][j] = c[i][j] * V[i][j] * 2^(qp/6)
    # where the >>6 normalization happens in the IDCT step
    dequant = coeffs * scale_matrix
    if qp_div_6 > 0:
        dequant = dequant << qp_div_6

    logger.debug(f"Dequantized output:\n{dequant}")

    return dequant.astype(np.int32)


def dequant_4x4_simple(
    coeffs: np.ndarray,
    qp: int
) -> np.ndarray:
    """Simplified dequantization matching common implementations.

    This version uses the standard dequantization formula that's
    commonly found in reference decoders.

    Args:
        coeffs: 4x4 quantized coefficients
        qp: Quantization parameter (0-51)

    Returns:
        4x4 dequantized coefficients

    Formula: d[i,j] = c[i,j] * scale[qp%6][pos] << (qp // 6)
    """
    qp = max(0, min(51, qp))
    qp_div_6 = qp // 6
    scale_matrix = get_scale_matrix(qp)

    # Simple scaling with left shift
    dequant = (coeffs * scale_matrix) << qp_div_6

    return dequant.astype(np.int32)


def dequant_4x4_with_scaling_list(
    coeffs: np.ndarray,
    qp: int,
    scaling_list: list,
) -> np.ndarray:
    """Dequantize a 4x4 block using a custom scaling list.

    This function applies position-dependent scaling from a custom
    scaling list, as used in High profile for custom quantization.

    Args:
        coeffs: 4x4 quantized coefficients (int32)
        qp: Quantization parameter (0-51)
        scaling_list: 16-element scaling list in zigzag scan order

    Returns:
        4x4 dequantized coefficients (int32)

    H.264 Spec: Section 8.5.9, 8.5.11
    Formula: d[i,j] = c[i,j] * LevelScale[qp%6][pos] * ScalingList[k] >> shift
    where k = zigzag index of position (i, j)
    """
    if coeffs.shape != (4, 4):
        raise ValueError(f"Expected 4x4 block, got {coeffs.shape}")

    if len(scaling_list) != 16:
        raise ValueError(f"Scaling list must have 16 elements, got {len(scaling_list)}")

    qp = max(0, min(51, qp))  # Clamp QP to valid range

    qp_div_6 = qp // 6
    qp_mod_6 = qp % 6

    # Zigzag scan order for 4x4 block (maps linear index to (row, col))
    zigzag_4x4 = [
        (0, 0), (0, 1), (1, 0), (2, 0),
        (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (3, 1), (2, 2),
        (1, 3), (2, 3), (3, 2), (3, 3)
    ]

    # Get base scale matrix
    scale_matrix = get_scale_matrix(qp)

    # Apply scaling list (position-dependent)
    dequant = np.zeros((4, 4), dtype=np.int32)
    for k, (i, j) in enumerate(zigzag_4x4):
        # Combine level scale with custom scaling list
        # H.264 normative: scale by scaling_list[k] / 16
        combined_scale = (scale_matrix[i, j] * scaling_list[k]) // 16
        dequant[i, j] = coeffs[i, j] * combined_scale

    # Apply QP-based shift
    if qp_div_6 >= 1:
        dequant = dequant << (qp_div_6 - 1)

    logger.debug(f"Dequant 4x4 with scaling list: QP={qp}, max_coeff={np.max(np.abs(dequant))}")

    return dequant.astype(np.int32)


def dequant_dc_4x4(
    dc_coeffs: np.ndarray,
    qp: int
) -> np.ndarray:
    """Dequantize 4x4 DC coefficients for Intra16x16 luma.

    DC coefficients from the 4x4 Hadamard transform of 16x16 macroblock
    DC values have special scaling rules.

    Args:
        dc_coeffs: 4x4 DC coefficients after Hadamard transform
        qp: Quantization parameter

    Returns:
        4x4 dequantized DC coefficients

    H.264 Spec: Section 8.5.11.1
    """
    qp = max(0, min(51, qp))
    qp_div_6 = qp // 6
    qp_mod_6 = qp % 6

    # For DC coefficients, use scale from position (0,0)
    scale = LEVEL_SCALE[qp_mod_6, 0]

    logger.debug(f"Dequantizing DC with QP={qp}, scale={scale}")

    if qp_div_6 >= 2:
        # For qp >= 12: shift left by (qp/6 - 2)
        dequant = (dc_coeffs * scale) << (qp_div_6 - 2)
    else:
        # For qp < 12: shift right by (2 - qp/6) with rounding
        # The extra /4 vs AC dequant absorbs the Hadamard *16 factor,
        # since the IDCT >>6 only provides /64 normalization
        rounding = 1 << (1 - qp_div_6)
        dequant = (dc_coeffs * scale + rounding) >> (2 - qp_div_6)

    return dequant.astype(np.int32)


def dequant_dc_2x2(
    dc_coeffs: np.ndarray,
    qp: int
) -> np.ndarray:
    """Dequantize 2x2 DC coefficients for chroma.

    Chroma DC coefficients from the 2x2 Hadamard transform.

    Args:
        dc_coeffs: 2x2 DC coefficients after Hadamard transform
        qp: Quantization parameter (chroma QP)

    Returns:
        2x2 dequantized DC coefficients

    H.264 Spec: Section 8.5.11.2
    """
    if dc_coeffs.shape != (2, 2):
        raise ValueError(f"Expected 2x2 block, got {dc_coeffs.shape}")

    qp = max(0, min(51, qp))
    qp_div_6 = qp // 6
    qp_mod_6 = qp % 6

    # For chroma DC, use scale from position (0,0)
    scale = LEVEL_SCALE[qp_mod_6, 0]

    logger.debug(f"Dequantizing chroma DC with QP={qp}, scale={scale}")

    if qp_div_6 >= 1:
        dequant = (dc_coeffs * scale) << (qp_div_6 - 1)
    else:
        # Use C-style truncation toward zero
        dequant = _rshift_toward_zero(dc_coeffs * scale + 1, 1)

    return dequant.astype(np.int32)


def get_scale_matrix_8x8(qp: int) -> np.ndarray:
    """Get the 8x8 scaling matrix for a given QP.

    Args:
        qp: Quantization parameter (0-51)

    Returns:
        8x8 scaling matrix (int32)
    """
    qp_mod_6 = qp % 6
    scales = LEVEL_SCALE_8x8[qp_mod_6]

    # Build scale matrix based on position types
    scale_matrix = np.zeros((8, 8), dtype=np.int32)
    for i in range(8):
        for j in range(8):
            pos_type = POSITION_TYPE_8x8[i, j]
            scale_matrix[i, j] = scales[pos_type]

    return scale_matrix


def dequant_8x8(
    coeffs: np.ndarray,
    qp: int,
    scaling_list: list = None,
) -> np.ndarray:
    """Dequantize an 8x8 coefficient block (High profile).

    Applies the inverse quantization scaling to transform coefficients.

    Args:
        coeffs: 8x8 quantized coefficients (int32)
        qp: Quantization parameter (0-51)
        scaling_list: Optional 64-element scaling list from SPS/PPS

    Returns:
        8x8 dequantized coefficients (int32)

    H.264 Spec: Section 8.5.11
    Formula: d[i,j] = c[i,j] * LevelScale8x8[qp%6][pos] * ScalingList[k] << (qp // 6)
    where k = zigzag index of position (i, j)
    """
    if coeffs.shape != (8, 8):
        raise ValueError(f"Expected 8x8 block, got {coeffs.shape}")

    qp = max(0, min(51, qp))  # Clamp QP to valid range

    qp_div_6 = qp // 6
    qp_mod_6 = qp % 6

    logger.debug(f"Dequantizing 8x8 with QP={qp} (div6={qp_div_6}, mod6={qp_mod_6})")

    # Get scale matrix
    scale_matrix = get_scale_matrix_8x8(qp)

    # Apply scaling list if provided (H.264 Section 8.5.12.1)
    if scaling_list is not None:
        if len(scaling_list) != 64:
            raise ValueError(f"Scaling list must have 64 elements, got {len(scaling_list)}")
        # Scaling lists are in zigzag scan order; convert to raster for
        # position-dependent scaling (H.264 Section 8.5.9)
        from entropy.tables import ZIGZAG_8x8
        sl_raster = np.zeros(64, dtype=np.int32)
        for scan_pos in range(64):
            sl_raster[int(ZIGZAG_8x8[scan_pos])] = scaling_list[scan_pos]
        scaling_matrix = sl_raster.reshape(8, 8)
        # LevelScale8x8 = normAdjust * weightScale (full precision, no //16)
        level_scale = scale_matrix * scaling_matrix

        # H.264 Section 8.5.12.1: exact spec formula
        # d[i][j] = (c * LevelScale) << (qpPer - 6)  if qpPer >= 6
        # d[i][j] = (c * LevelScale + 2^(5-qpPer)) >> (6 - qpPer)  if qpPer < 6
        if qp_div_6 >= 6:
            dequant = (coeffs * level_scale) << (qp_div_6 - 6)
        else:
            rounding = 1 << (5 - qp_div_6)
            dequant = (coeffs * level_scale + rounding) >> (6 - qp_div_6)
    else:
        # No scaling list: normAdjust only (equivalent to FLAT weightScale=16)
        # d = c * normAdjust * 16 << (qpPer - 6) = c * normAdjust << (qpPer - 2)
        if qp_div_6 >= 2:
            dequant = (coeffs * scale_matrix) << (qp_div_6 - 2)
        else:
            rounding = 1 << (1 - qp_div_6)
            dequant = (coeffs * scale_matrix + rounding) >> (2 - qp_div_6)

    logger.debug(f"Dequantized 8x8 output: max={np.max(np.abs(dequant))}")

    return dequant.astype(np.int32)


def get_chroma_qp(luma_qp: int) -> int:
    """Convert luma QP to chroma QP.

    H.264 uses a mapping table for chroma QP to reduce chroma artifacts
    at high QP values.

    Args:
        luma_qp: Luma quantization parameter (0-51)

    Returns:
        Chroma quantization parameter

    H.264 Spec: Table 8-15
    """
    # Chroma QP mapping table (for QP >= 30, chroma QP is lower)
    CHROMA_QP_TABLE = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34,
        35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39
    ]

    luma_qp = max(0, min(51, luma_qp))
    return CHROMA_QP_TABLE[luma_qp]


def qp_to_qstep(qp: int) -> float:
    """Convert QP to quantization step size.

    Useful for understanding the relationship between QP and quantization.

    Args:
        qp: Quantization parameter (0-51)

    Returns:
        Quantization step size (approximate)

    Note:
        QP increases by 6 → step size doubles
        QP=0: step ≈ 0.625
        QP=6: step ≈ 1.25
        QP=12: step ≈ 2.5
        ...
    """
    return 0.625 * (2 ** (qp / 6))
