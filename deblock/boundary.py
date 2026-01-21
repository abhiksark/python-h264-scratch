# h264/deblock/boundary.py
"""Boundary strength calculation for deblocking filter.

H.264 Spec Reference: Section 8.7.2.1 - Derivation of bS
"""

from typing import Tuple


def calc_boundary_strength(
    is_intra_p: bool,
    is_intra_q: bool,
    has_coeff_p: bool,
    has_coeff_q: bool,
    mv_p: Tuple[int, int],
    mv_q: Tuple[int, int],
    ref_p: int,
    ref_q: int,
) -> int:
    """Calculate boundary strength (bS) between two adjacent blocks.

    The blocks p and q are on opposite sides of the edge being filtered.
    p is the block on the "negative" side (left or above).
    q is the block on the "positive" side (right or below).

    Boundary strength determines filter aggressiveness:
        bS=4: Strong filtering (intra boundaries)
        bS=3: One block has non-zero coefficients
        bS=2: Different reference frames
        bS=1: MV difference >= 4 quarter-pixels (1 full pixel)
        bS=0: No filtering needed

    Args:
        is_intra_p: True if block p is intra-coded
        is_intra_q: True if block q is intra-coded
        has_coeff_p: True if block p has non-zero residual coefficients
        has_coeff_q: True if block q has non-zero residual coefficients
        mv_p: Motion vector of block p (mvx, mvy) in quarter-pixels
        mv_q: Motion vector of block q (mvx, mvy) in quarter-pixels
        ref_p: Reference frame index of block p
        ref_q: Reference frame index of block q

    Returns:
        Boundary strength (0-4)

    H.264 Spec: Section 8.7.2.1
    """
    # bS = 4 for intra blocks
    if is_intra_p or is_intra_q:
        return 4

    # bS = 3 if either block has non-zero coefficients
    if has_coeff_p or has_coeff_q:
        return 3

    # bS = 2 if different reference frames
    if ref_p != ref_q:
        return 2

    # bS = 1 if MV difference >= 4 quarter-pixels in either direction
    mv_diff_x = abs(mv_p[0] - mv_q[0])
    mv_diff_y = abs(mv_p[1] - mv_q[1])

    if mv_diff_x >= 4 or mv_diff_y >= 4:
        return 1

    # bS = 0 - blocks are similar, no filtering needed
    return 0


def calc_boundary_strength_mb_edge(
    is_intra_p: bool,
    is_intra_q: bool,
    is_mb_edge: bool,
) -> int:
    """Calculate bS for macroblock edge (simplified).

    For macroblock boundaries, bS is always 4 if either MB is intra,
    otherwise it depends on block properties.

    Args:
        is_intra_p: True if MB on p side is intra
        is_intra_q: True if MB on q side is intra
        is_mb_edge: True if this is a macroblock boundary

    Returns:
        Boundary strength (4 for intra, else needs full calculation)
    """
    if is_mb_edge and (is_intra_p or is_intra_q):
        return 4
    return 0  # Need full calculation
