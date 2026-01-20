# h264/transform/idct_4x4.py
"""H.264 4x4 Integer Inverse Transform and Hadamard Transforms.

H.264 uses an integer approximation of the DCT that allows for exact
(bit-accurate) reconstruction. The transform is separable and uses
simple integer operations.

H.264 Spec Reference:
- Section 8.5.12: Inverse transform process
- Section 8.5.12.1: Inverse 4x4 luma DC Hadamard transform
- Section 8.5.12.2: Inverse 2x2 chroma DC Hadamard transform

The inverse transform is:
1. Apply 1D inverse transform to columns
2. Apply 1D inverse transform to rows
3. Normalize by shifting right by 6 (divide by 64)
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


# 1D inverse transform matrix (transposed for column operation)
# H.264 uses a modified DCT with basis vectors scaled for integer arithmetic
# The transform matrix is: [1  1  1  1]
#                          [1  1/2 -1/2 -1]
#                          [1 -1 -1  1]
#                          [1/2 -1 1 -1/2]
# Scaled by 2 to avoid fractions (the 1/2 becomes 1 when we account for shift)

# For inverse transform, we use this core matrix:
IDCT_CORE = np.array([
    [1, 1, 1, 1],
    [1, 1, -1, -1],
    [1, -1, -1, 1],
    [1, -1, 1, -1]
], dtype=np.int32)


def idct_1d(x: np.ndarray) -> np.ndarray:
    """Apply 1D inverse transform to a 4-element vector.

    Args:
        x: Input vector of 4 elements

    Returns:
        Transformed vector of 4 elements

    This implements the butterfly operation for H.264 inverse transform.
    """
    # Even part
    e0 = x[0] + x[2]
    e1 = x[0] - x[2]

    # Odd part (with H.264's specific scaling)
    # x[1] is scaled by 1, x[3] is scaled by 1/2
    o0 = (x[1] >> 1) - x[3]
    o1 = x[1] + (x[3] >> 1)

    # Combine
    return np.array([
        e0 + o1,
        e1 + o0,
        e1 - o0,
        e0 - o1
    ], dtype=np.int32)


def idct_4x4(coeffs: np.ndarray) -> np.ndarray:
    """Apply 4x4 integer inverse transform (IDCT).

    This is the core transform used in H.264 to convert frequency-domain
    coefficients back to spatial-domain residuals.

    Args:
        coeffs: 4x4 transform coefficients (int32)

    Returns:
        4x4 spatial residuals (int32), normalized

    H.264 Spec: Section 8.5.12

    The process is:
    1. Column transform with post-multiply (add 32, shift right 6)
    2. Row transform
    3. Final normalization

    Note: The scaling is integrated - input is dequantized coefficients,
    output is pixel residuals ready to add to prediction.
    """
    if coeffs.shape != (4, 4):
        raise ValueError(f"Expected 4x4 block, got {coeffs.shape}")

    logger.debug(f"IDCT input:\n{coeffs}")

    # Work with int32 to avoid overflow
    temp = coeffs.astype(np.int32)

    # Step 1: Apply 1D transform to each column
    col_result = np.zeros((4, 4), dtype=np.int32)
    for j in range(4):
        col_result[:, j] = idct_1d(temp[:, j])

    # Step 2: Apply 1D transform to each row
    row_result = np.zeros((4, 4), dtype=np.int32)
    for i in range(4):
        row_result[i, :] = idct_1d(col_result[i, :])

    # Step 3: Normalize (divide by 64 with rounding)
    # Add 32 for rounding, then shift right by 6
    result = (row_result + 32) >> 6

    logger.debug(f"IDCT output:\n{result}")

    return result


def idct_4x4_matrix(coeffs: np.ndarray) -> np.ndarray:
    """Alternative matrix-based IDCT implementation.

    Uses explicit matrix multiplication for clarity.
    Result should match idct_4x4().

    Args:
        coeffs: 4x4 transform coefficients

    Returns:
        4x4 spatial residuals, normalized
    """
    # Transform matrix (scaled version of DCT-II inverse)
    C = np.array([
        [1, 1, 1, 1],
        [1, 0.5, -0.5, -1],
        [1, -1, -1, 1],
        [0.5, -1, 1, -0.5]
    ])

    # 2D transform: C^T * coeffs * C
    # But H.264 uses specific integer rounding, so we use the butterfly version
    # for spec compliance

    result = C.T @ coeffs.astype(np.float64) @ C

    # Normalize and convert to int
    return np.round(result / 4).astype(np.int32)


def hadamard_4x4(dc_coeffs: np.ndarray) -> np.ndarray:
    """Apply 4x4 Hadamard transform to luma DC coefficients.

    Used for I16x16 macroblocks where all 16 blocks share a common
    DC transform. The Hadamard transform is its own inverse (up to scaling).

    Args:
        dc_coeffs: 4x4 DC coefficients from I16x16 macroblock

    Returns:
        4x4 transformed DC coefficients

    H.264 Spec: Section 8.5.12.1

    The 4x4 Hadamard matrix is:
    H = [1  1  1  1]
        [1  1 -1 -1]
        [1 -1 -1  1]
        [1 -1  1 -1]
    """
    if dc_coeffs.shape != (4, 4):
        raise ValueError(f"Expected 4x4 block, got {dc_coeffs.shape}")

    logger.debug(f"Hadamard 4x4 input:\n{dc_coeffs}")

    H = np.array([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
        [1, -1, 1, -1]
    ], dtype=np.int32)

    # Apply: H * dc_coeffs * H^T
    # Since H is symmetric and orthogonal (up to scale), H^T = H
    temp = dc_coeffs.astype(np.int32)

    # Row transform
    temp = H @ temp

    # Column transform
    temp = temp @ H.T

    # No normalization needed for inverse (done in dequant)
    logger.debug(f"Hadamard 4x4 output:\n{temp}")

    return temp


def hadamard_2x2(dc_coeffs: np.ndarray) -> np.ndarray:
    """Apply 2x2 Hadamard transform to chroma DC coefficients.

    Used for chroma DC coefficients (4:2:0 format has 2x2 chroma DCs).

    Args:
        dc_coeffs: 2x2 DC coefficients

    Returns:
        2x2 transformed DC coefficients

    H.264 Spec: Section 8.5.12.2

    The 2x2 Hadamard matrix is:
    H = [1  1]
        [1 -1]
    """
    if dc_coeffs.shape != (2, 2):
        raise ValueError(f"Expected 2x2 block, got {dc_coeffs.shape}")

    logger.debug(f"Hadamard 2x2 input:\n{dc_coeffs}")

    H = np.array([
        [1, 1],
        [1, -1]
    ], dtype=np.int32)

    temp = dc_coeffs.astype(np.int32)

    # Apply: H * dc_coeffs * H^T
    temp = H @ temp @ H.T

    logger.debug(f"Hadamard 2x2 output:\n{temp}")

    return temp


def inverse_hadamard_4x4(dc_coeffs: np.ndarray) -> np.ndarray:
    """Inverse 4x4 Hadamard transform.

    For decoding, we apply the inverse Hadamard. Since Hadamard is self-inverse
    up to scaling, we just apply Hadamard again and normalize.

    Args:
        dc_coeffs: 4x4 Hadamard-transformed DC coefficients

    Returns:
        4x4 inverse-transformed coefficients

    Note: Normalization factor is 1/16 for 4x4 Hadamard
    """
    # Apply Hadamard (self-inverse property)
    result = hadamard_4x4(dc_coeffs)

    # Normalize: divide by 16 (4x4 elements) with rounding
    # In H.264, this is typically done with (result + 8) >> 4
    # But the actual normalization depends on the dequant step

    return result


def inverse_hadamard_2x2(dc_coeffs: np.ndarray) -> np.ndarray:
    """Inverse 2x2 Hadamard transform.

    Args:
        dc_coeffs: 2x2 Hadamard-transformed DC coefficients

    Returns:
        2x2 inverse-transformed coefficients

    Note: Normalization factor is 1/4 for 2x2 Hadamard
    """
    # Apply Hadamard (self-inverse property)
    result = hadamard_2x2(dc_coeffs)

    # Normalization typically done in dequant step
    return result


def forward_4x4(spatial: np.ndarray) -> np.ndarray:
    """Forward 4x4 transform (for testing/encoding).

    Converts spatial residuals to frequency-domain coefficients.
    This is the inverse operation of idct_4x4.

    Args:
        spatial: 4x4 spatial residuals

    Returns:
        4x4 transform coefficients

    Note: This is primarily for testing - a decoder doesn't need this.
    """
    if spatial.shape != (4, 4):
        raise ValueError(f"Expected 4x4 block, got {spatial.shape}")

    # Forward transform matrix
    Cf = np.array([
        [1, 1, 1, 1],
        [2, 1, -1, -2],
        [1, -1, -1, 1],
        [1, -2, 2, -1]
    ], dtype=np.int32)

    temp = spatial.astype(np.int32)

    # Apply: Cf * spatial * Cf^T
    result = Cf @ temp @ Cf.T

    return result


def process_dc_block_i16x16(
    dc_coeffs: np.ndarray,
    qp: int
) -> np.ndarray:
    """Process DC block for I16x16 macroblock.

    Complete processing pipeline for I16x16 luma DC coefficients:
    1. Inverse Hadamard
    2. Dequantization (with DC-specific scaling)

    Args:
        dc_coeffs: 4x4 DC coefficients
        qp: Quantization parameter

    Returns:
        4x4 processed DC values
    """
    # Import here to avoid circular dependency
    from dequant import dequant_dc_4x4

    # Inverse Hadamard
    dc_transformed = inverse_hadamard_4x4(dc_coeffs)

    # Dequantize (DC-specific scaling)
    dc_dequant = dequant_dc_4x4(dc_transformed, qp)

    return dc_dequant


def process_dc_block_chroma(
    dc_coeffs: np.ndarray,
    qp: int
) -> np.ndarray:
    """Process DC block for chroma.

    Complete processing pipeline for chroma DC coefficients:
    1. Inverse Hadamard (2x2)
    2. Dequantization

    Args:
        dc_coeffs: 2x2 DC coefficients
        qp: Quantization parameter (chroma QP)

    Returns:
        2x2 processed DC values
    """
    from dequant import dequant_dc_2x2

    dc_transformed = inverse_hadamard_2x2(dc_coeffs)
    dc_dequant = dequant_dc_2x2(dc_transformed, qp)

    return dc_dequant
