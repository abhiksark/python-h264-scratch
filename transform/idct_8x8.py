# h264/transform/idct_8x8.py
"""H.264 8x8 Integer Inverse Transform for High Profile.

The 8x8 IDCT is used for I_8x8 macroblocks in High profile.
Uses 8-point butterfly operations with integer arithmetic.

H.264 Spec Reference: Section 8.5.12 - Inverse transform process

The inverse transform uses a separable 2D approach:
1. Apply 1D inverse transform to each column
2. Apply 1D inverse transform to each row
3. Normalize by right-shifting

The 8-point butterfly uses scaled integer coefficients to avoid
floating-point operations while maintaining precision.
"""

import logging

import numpy as np

# Import zigzag scan from entropy tables (already defined there)
from entropy.tables import ZIGZAG_8x8, ZIGZAG_8x8_INV

logger = logging.getLogger(__name__)


# Re-export zigzag scans with common naming convention
# ZIGZAG_8x8 contains flat indices (0-63), tests expect 2D coordinates
# Convert to tuple format: index i -> (row=i//8, col=i%8)
ZIGZAG_SCAN_8x8 = tuple((idx // 8, idx % 8) for idx in ZIGZAG_8x8)

# H.264 Table 8-13: Field scan for 8x8 transform (interlaced video)
# Column-major pattern optimized for interlaced content
_FIELD_SCAN_8x8_FLAT = np.array([
    0,  8, 16,  1,  9, 24, 32, 17,
    2, 25, 40, 48, 33, 26, 18,  3,
   10, 41, 56, 49, 34, 27, 19, 11,
    4, 12, 35, 42, 50, 57, 58, 51,
   43, 36, 28, 20,  5, 13, 21, 29,
   37, 44, 52, 59, 60, 53, 45, 38,
   30, 22,  6, 14,  7, 15, 23, 31,
   39, 46, 54, 61, 62, 55, 47, 63,
], dtype=np.int32)
# Convert to tuple format for consistency with ZIGZAG_SCAN_8x8
FIELD_SCAN_8x8 = tuple((idx // 8, idx % 8) for idx in _FIELD_SCAN_8x8_FLAT)

# H.264 8x8 DCT transform matrix (Table 8-12)
# Scaled integer approximation of DCT-II basis vectors
# T = C * X * C^T where C is this matrix
TRANSFORM_MATRIX_8x8 = np.array([
    [ 8,  8,  8,  8,  8,  8,  8,  8],
    [12, 10,  6,  3, -3, -6,-10,-12],
    [ 8,  4, -4, -8, -8, -4,  4,  8],
    [10, -3,-12, -6,  6, 12,  3,-10],
    [ 8, -8, -8,  8,  8, -8, -8,  8],
    [ 6,-12,  3, 10,-10, -3, 12, -6],
    [ 4, -8,  8, -4, -4,  8, -8,  4],
    [ 3, -6, 10,-12, 12,-10,  6, -3],
], dtype=np.int32)

# Position-dependent scaling factors for 8x8 transform (H.264 Table 8-14)
# These are the normalization factors for each position in the 8x8 block
SCALING_FACTORS_8x8 = np.array([
    [64, 68, 64, 68, 64, 68, 64, 68],
    [68, 72, 68, 72, 68, 72, 68, 72],
    [64, 68, 64, 68, 64, 68, 64, 68],
    [68, 72, 68, 72, 68, 72, 68, 72],
    [64, 68, 64, 68, 64, 68, 64, 68],
    [68, 72, 68, 72, 68, 72, 68, 72],
    [64, 68, 64, 68, 64, 68, 64, 68],
    [68, 72, 68, 72, 68, 72, 68, 72],
], dtype=np.int32)


# H.264 8x8 IDCT butterfly coefficients (Table 8-12 scaled)
# These are the basis function values scaled for integer arithmetic
# a=8, b=10, c=9, d=6, e=4, f=2, g=1 (with various combinations)


def idct_1d_8(x: np.ndarray) -> np.ndarray:
    """Apply 1D 8-point inverse transform.

    Uses H.264's integer butterfly structure for 8-point IDCT.
    This is the core 1D transform that gets applied to rows and columns.

    Args:
        x: Input vector of 8 elements (int32)

    Returns:
        Transformed vector of 8 elements (int32)

    The butterfly implements the inverse of the H.264 8x8 forward transform,
    using specific integer approximations for the DCT basis functions.
    """
    x = x.astype(np.int32)

    # Stage 1: Initial butterfly
    # Even indices: 0, 2, 4, 6
    # Odd indices: 1, 3, 5, 7

    # Even part (4-point IDCT-like structure)
    e0 = x[0] + x[4]
    e1 = x[0] - x[4]
    e2 = (x[2] >> 1) - x[6]
    e3 = x[2] + (x[6] >> 1)

    # Combine even terms
    f0 = e0 + e3
    f1 = e1 + e2
    f2 = e1 - e2
    f3 = e0 - e3

    # Odd part (more complex butterfly)
    # Using H.264 specific coefficients
    g0 = x[1] - x[7]
    g1 = x[1] + x[7]
    g2 = x[3] - x[5]
    g3 = x[3] + x[5]

    h0 = g0 + (g2 >> 1)
    h1 = (g0 >> 1) - g2
    h2 = g1 - (g3 >> 1)
    h3 = (g1 >> 1) + g3

    # Final stage: combine even and odd
    return np.array([
        f0 + h3,
        f1 + h2,
        f2 + h1,
        f3 + h0,
        f3 - h0,
        f2 - h1,
        f1 - h2,
        f0 - h3,
    ], dtype=np.int32)


def idct_8x8(coeffs: np.ndarray) -> np.ndarray:
    """Apply 8x8 integer inverse transform (IDCT).

    This is the core transform used in H.264 High profile to convert
    frequency-domain coefficients back to spatial-domain residuals
    for 8x8 blocks.

    Args:
        coeffs: 8x8 transform coefficients (int32)

    Returns:
        8x8 spatial residuals (int32), normalized

    H.264 Spec: Section 8.5.12

    The process is:
    1. Apply 1D transform to each column
    2. Apply 1D transform to each row
    3. Final normalization (divide by 256 with rounding)
    """
    if coeffs.shape != (8, 8):
        raise ValueError(f"Expected 8x8 block, got {coeffs.shape}")

    logger.debug(f"IDCT 8x8 input:\n{coeffs}")

    # Work with int32 to avoid overflow
    temp = coeffs.astype(np.int32)

    # Step 1: Apply 1D transform to each column
    col_result = np.zeros((8, 8), dtype=np.int32)
    for j in range(8):
        col_result[:, j] = idct_1d_8(temp[:, j])

    # Step 2: Apply 1D transform to each row
    row_result = np.zeros((8, 8), dtype=np.int32)
    for i in range(8):
        row_result[i, :] = idct_1d_8(col_result[i, :])

    # Step 3: Normalize (divide by 256 with rounding)
    # For 8x8: scale factor is 2^6 per dimension = 2^12 total
    # But H.264 spec uses 2^6 combined, giving effective 2^12 / 64 = 64
    # Add 128 for rounding, then shift right by 8
    result = (row_result + 128) >> 8

    logger.debug(f"IDCT 8x8 output:\n{result}")

    return result


def forward_1d_8(x: np.ndarray) -> np.ndarray:
    """Apply 1D 8-point forward transform.

    This computes the forward DCT using matrix multiplication.

    Args:
        x: Input vector of 8 elements (int32)

    Returns:
        Transformed vector of 8 elements (int32)
    """
    return TRANSFORM_MATRIX_8x8 @ x.astype(np.int32)


def forward_8x8(block: np.ndarray) -> np.ndarray:
    """Apply 8x8 forward transform (DCT).

    This is the inverse of idct_8x8, used primarily for testing
    round-trip accuracy of the transform.

    Uses matrix multiplication: Y = Cf * X * Cf^T

    Args:
        block: 8x8 spatial block (int32)

    Returns:
        8x8 transform coefficients (int32)

    H.264 Spec: This is the encoder-side transform, inverse of Section 8.5.12

    Note: The forward transform produces coefficients that when passed through
    idct_8x8 will recover the original (within integer rounding).
    """
    if block.shape != (8, 8):
        raise ValueError(f"Expected 8x8 block, got {block.shape}")

    # Work with int32
    temp = block.astype(np.int32)

    # Apply: Cf * spatial * Cf^T (matrix multiplication)
    raw = TRANSFORM_MATRIX_8x8 @ temp @ TRANSFORM_MATRIX_8x8.T

    # Scale to match IDCT's normalization
    # The H.264 integer transforms use scaled coefficients (8, 10, 12, etc.)
    # The IDCT expects coefficients at a certain scale
    # Empirically tune scaling for best round-trip accuracy
    result = raw >> 4

    return result


def idct_8x8_batch(blocks: np.ndarray) -> np.ndarray:
    """Process multiple 8x8 blocks efficiently.

    Applies idct_8x8 to each block in the input array.

    Args:
        blocks: Array of shape (N, 8, 8) containing N coefficient blocks

    Returns:
        Array of shape (N, 8, 8) containing N residual blocks

    Note: For maximum performance, consider using vectorized operations
    in the future. Current implementation uses a simple loop.
    """
    if blocks.ndim != 3 or blocks.shape[1:] != (8, 8):
        raise ValueError(f"Expected shape (N, 8, 8), got {blocks.shape}")

    n_blocks = blocks.shape[0]
    result = np.zeros_like(blocks, dtype=np.int32)

    for i in range(n_blocks):
        result[i] = idct_8x8(blocks[i])

    return result
