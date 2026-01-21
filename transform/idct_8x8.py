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

logger = logging.getLogger(__name__)


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
