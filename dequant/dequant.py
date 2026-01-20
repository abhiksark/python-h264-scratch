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
    # For qp >= 6: multiply then shift left
    # For qp < 6: multiply, add rounding, shift right
    if qp_div_6 >= 1:
        # Shift left by (qp_div_6 - 1)
        # Note: actual scaling depends on normative transform
        dequant = coeffs * scale_matrix
        dequant = dequant << (qp_div_6 - 1)
    else:
        # For qp < 6, scale is just the level_scale value
        dequant = coeffs * scale_matrix

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
        # For qp >= 12: shift left
        dequant = (dc_coeffs * scale) << (qp_div_6 - 2)
    elif qp_div_6 == 1:
        # For 6 <= qp < 12: just multiply
        dequant = dc_coeffs * scale
    else:
        # For qp < 6: shift right with rounding
        dequant = (dc_coeffs * scale + 1) >> 1

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
        dequant = (dc_coeffs * scale + 1) >> 1

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
