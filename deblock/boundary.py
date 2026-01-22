# h264/deblock/boundary.py
"""Boundary strength calculation for deblocking filter.

H.264 Spec Reference: Section 8.7.2.1 - Derivation of bS
"""

from typing import Tuple, Dict, List


def calc_boundary_strength(
    is_intra_p: bool,
    is_intra_q: bool,
    has_coeff_p: bool,
    has_coeff_q: bool,
    mv_p: Tuple[int, int],
    mv_q: Tuple[int, int],
    ref_p: int,
    ref_q: int,
    is_pcm_p: bool = False,
    is_pcm_q: bool = False,
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
        bS=0: No filtering needed (including I_PCM edges)

    Args:
        is_intra_p: True if block p is intra-coded
        is_intra_q: True if block q is intra-coded
        has_coeff_p: True if block p has non-zero residual coefficients
        has_coeff_q: True if block q has non-zero residual coefficients
        mv_p: Motion vector of block p (mvx, mvy) in quarter-pixels
        mv_q: Motion vector of block q (mvx, mvy) in quarter-pixels
        ref_p: Reference frame index of block p
        ref_q: Reference frame index of block q
        is_pcm_p: True if block p is I_PCM
        is_pcm_q: True if block q is I_PCM

    Returns:
        Boundary strength (0-4)

    H.264 Spec: Section 8.7.2.1
    """
    # bS = 0 for I_PCM blocks (no filtering at PCM boundaries)
    if is_pcm_p or is_pcm_q:
        return 0

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


def calc_mb_boundary_strength(
    current_mb: Dict,
    neighbor_mb: Dict,
    is_mb_edge: bool = True,
) -> int:
    """Calculate boundary strength at macroblock boundary.

    Args:
        current_mb: Info for current macroblock
        neighbor_mb: Info for neighbor macroblock
        is_mb_edge: True if this is an MB boundary

    Returns:
        Boundary strength (0-4)
    """
    is_intra_p = neighbor_mb.get('is_intra', False)
    is_intra_q = current_mb.get('is_intra', False)

    # bS=4 if either block is intra
    if is_intra_p or is_intra_q:
        return 4

    has_coeff_p = neighbor_mb.get('has_coeff', False)
    has_coeff_q = current_mb.get('has_coeff', False)

    # bS=3 if either has non-zero coefficients
    if has_coeff_p or has_coeff_q:
        return 3

    ref_p = neighbor_mb.get('ref', 0)
    ref_q = current_mb.get('ref', 0)

    # bS=2 if different references
    if ref_p != ref_q:
        return 2

    mv_p = neighbor_mb.get('mv', (0, 0))
    mv_q = current_mb.get('mv', (0, 0))

    # bS=1 if MV difference >= 4 quarter-pixels
    if abs(mv_p[0] - mv_q[0]) >= 4 or abs(mv_p[1] - mv_q[1]) >= 4:
        return 1

    return 0


def calc_boundary_strength_8x8(
    is_intra_p: bool,
    is_intra_q: bool,
    transform_8x8_p: bool,
    transform_8x8_q: bool,
    has_coeff_8x8_p: bool,
    has_coeff_8x8_q: bool,
    mv_p: Tuple[int, int],
    mv_q: Tuple[int, int],
    ref_p: int,
    ref_q: int,
) -> int:
    """Calculate boundary strength (bS) for 8x8 transform blocks.

    H.264 Spec: Section 8.7.2.1, extended for 8x8 transforms
    When transform_size_8x8_flag=1, coefficient checks use entire 8x8 block.

    Args:
        is_intra_p: True if block p is intra-coded
        is_intra_q: True if block q is intra-coded
        transform_8x8_p: True if p uses 8x8 transform
        transform_8x8_q: True if q uses 8x8 transform
        has_coeff_8x8_p: True if 8x8 block p has any non-zero coefficients
        has_coeff_8x8_q: True if 8x8 block q has any non-zero coefficients
        mv_p: Motion vector of block p (mvx, mvy) in quarter-pixels
        mv_q: Motion vector of block q (mvx, mvy) in quarter-pixels
        ref_p: Reference frame index of block p
        ref_q: Reference frame index of block q

    Returns:
        Boundary strength (0-4)
    """
    # bS = 4 for intra blocks
    if is_intra_p or is_intra_q:
        return 4

    # bS = 3 if either 8x8 block has non-zero coefficients
    if has_coeff_8x8_p or has_coeff_8x8_q:
        return 3

    # bS = 2 if different reference frames
    if ref_p != ref_q:
        return 2

    # bS = 1 if MV difference >= 4 quarter-pixels
    mv_diff_x = abs(mv_p[0] - mv_q[0])
    mv_diff_y = abs(mv_p[1] - mv_q[1])
    if mv_diff_x >= 4 or mv_diff_y >= 4:
        return 1

    # bS = 0 - blocks are similar
    return 0


def calc_boundary_strength_mixed_transform(
    is_intra_p: bool,
    is_intra_q: bool,
    transform_8x8_p: bool,
    transform_8x8_q: bool,
    has_coeff_p: bool = False,
    has_coeff_q: bool = False,
    mv_p: Tuple[int, int] = (0, 0),
    mv_q: Tuple[int, int] = (0, 0),
    ref_p: int = 0,
    ref_q: int = 0,
) -> int:
    """Calculate bS at boundary between blocks with different transform sizes.

    H.264 Spec: When transform sizes differ, bS calculation still applies
    but coefficient flags may come from different granularities.

    Args:
        is_intra_p, is_intra_q: Intra flags for p and q blocks
        transform_8x8_p, transform_8x8_q: Transform size flags
        has_coeff_p, has_coeff_q: Coefficient flags (8x8 or 4x4 as appropriate)
        mv_p, mv_q: Motion vectors
        ref_p, ref_q: Reference indices

    Returns:
        Boundary strength (0-4)
    """
    # Same rules apply regardless of transform size mismatch
    if is_intra_p or is_intra_q:
        return 4

    if has_coeff_p or has_coeff_q:
        return 3

    if ref_p != ref_q:
        return 2

    mv_diff_x = abs(mv_p[0] - mv_q[0])
    mv_diff_y = abs(mv_p[1] - mv_q[1])
    if mv_diff_x >= 4 or mv_diff_y >= 4:
        return 1

    return 0


def calc_boundary_strength_8x8_bipred(
    is_intra_p: bool,
    is_intra_q: bool,
    transform_8x8_p: bool,
    transform_8x8_q: bool,
    has_coeff_8x8_p: bool,
    has_coeff_8x8_q: bool,
    mv_l0_p: Tuple[int, int],
    mv_l0_q: Tuple[int, int],
    mv_l1_p: Tuple[int, int],
    mv_l1_q: Tuple[int, int],
    ref_l0_p: int,
    ref_l0_q: int,
    ref_l1_p: int,
    ref_l1_q: int,
) -> int:
    """Calculate bS for bi-predicted 8x8 blocks.

    H.264 Spec: Section 8.7.2.1 for B-slices with bi-prediction.
    Both L0 and L1 motion vectors and references are considered.

    Args:
        is_intra_p, is_intra_q: Intra flags
        transform_8x8_p, transform_8x8_q: Transform size flags
        has_coeff_8x8_p, has_coeff_8x8_q: Coefficient flags
        mv_l0_p, mv_l0_q: L0 motion vectors
        mv_l1_p, mv_l1_q: L1 motion vectors
        ref_l0_p, ref_l0_q: L0 reference indices
        ref_l1_p, ref_l1_q: L1 reference indices

    Returns:
        Boundary strength (0-4)
    """
    # bS=4 for intra blocks
    if is_intra_p or is_intra_q:
        return 4

    # bS=3 if either has coefficients
    if has_coeff_8x8_p or has_coeff_8x8_q:
        return 3

    # Check if references differ
    # For bi-prediction, check both L0 and L1
    l0_refs_differ = ref_l0_p != ref_l0_q
    l1_refs_differ = ref_l1_p != ref_l1_q

    # bS=2 if any reference pair differs
    if l0_refs_differ or l1_refs_differ:
        return 2

    # Check MV differences for both lists
    mv_l0_diff_x = abs(mv_l0_p[0] - mv_l0_q[0])
    mv_l0_diff_y = abs(mv_l0_p[1] - mv_l0_q[1])
    mv_l1_diff_x = abs(mv_l1_p[0] - mv_l1_q[0])
    mv_l1_diff_y = abs(mv_l1_p[1] - mv_l1_q[1])

    # bS=1 if any MV difference >= 4 quarter-pixels
    if (mv_l0_diff_x >= 4 or mv_l0_diff_y >= 4 or
        mv_l1_diff_x >= 4 or mv_l1_diff_y >= 4):
        return 1

    return 0


def calc_boundary_strength_8x8_direct(
    is_intra_p: bool,
    is_intra_q: bool,
    transform_8x8_p: bool,
    transform_8x8_q: bool,
    has_coeff_8x8_p: bool,
    has_coeff_8x8_q: bool,
    is_direct_p: bool,
    is_direct_q: bool,
    collocated_ref_p: int = 0,
    collocated_ref_q: int = 0,
) -> int:
    """Calculate bS for direct mode 8x8 blocks.

    H.264 Spec: Direct mode uses co-located MVs from reference list.
    Both temporal and spatial direct modes are supported.

    Args:
        is_intra_p, is_intra_q: Intra flags
        transform_8x8_p, transform_8x8_q: Transform size flags
        has_coeff_8x8_p, has_coeff_8x8_q: Coefficient flags
        is_direct_p, is_direct_q: True if block uses direct mode
        collocated_ref_p, collocated_ref_q: Co-located reference indices

    Returns:
        Boundary strength (0-4)
    """
    # bS=4 for intra blocks
    if is_intra_p or is_intra_q:
        return 4

    # bS=3 if either has coefficients
    if has_coeff_8x8_p or has_coeff_8x8_q:
        return 3

    # bS=2 if collocated refs differ
    if collocated_ref_p != collocated_ref_q:
        return 2

    # For direct mode without coefficients and same collocated ref, bS=0
    return 0


def get_4x4_blocks_for_8x8(block_8x8_idx: int) -> List[int]:
    """Get the 4x4 block indices that compose an 8x8 block.

    8x8 block layout in H.264 raster order:
        Block 0: 4x4 blocks 0, 1, 2, 3
        Block 1: 4x4 blocks 4, 5, 6, 7
        Block 2: 4x4 blocks 8, 9, 10, 11
        Block 3: 4x4 blocks 12, 13, 14, 15

    Args:
        block_8x8_idx: 8x8 block index (0-3)

    Returns:
        List of four 4x4 block indices
    """
    if block_8x8_idx < 0 or block_8x8_idx > 3:
        raise ValueError(f"Invalid 8x8 block index: {block_8x8_idx}")

    # Map 8x8 block to its constituent 4x4 blocks
    # 8x8 block 0 (top-left):     4x4 blocks 0, 1, 2, 3
    # 8x8 block 1 (top-right):    4x4 blocks 4, 5, 6, 7
    # 8x8 block 2 (bottom-left):  4x4 blocks 8, 9, 10, 11
    # 8x8 block 3 (bottom-right): 4x4 blocks 12, 13, 14, 15
    base = block_8x8_idx * 4
    return [base, base + 1, base + 2, base + 3]


def get_boundary_block_pairs(direction: str, edge_pos: int) -> List[Tuple[int, int]]:
    """Get pairs of 4x4 block indices for MB boundary filtering.

    Args:
        direction: 'vertical' or 'horizontal'
        edge_pos: Edge position (0 for MB boundary)

    Returns:
        List of (neighbor_block_idx, current_block_idx) pairs
    """
    pairs = []

    if direction == 'vertical' and edge_pos == 0:
        # Left MB boundary: right edge of left MB to left edge of current MB
        # Left MB right edge blocks: 5, 7, 13, 15 (in H.264 scan order)
        # Current MB left edge blocks: 0, 2, 8, 10
        neighbor_blocks = [5, 7, 13, 15]
        current_blocks = [0, 2, 8, 10]
        pairs = list(zip(neighbor_blocks, current_blocks))

    elif direction == 'horizontal' and edge_pos == 0:
        # Top MB boundary: bottom edge of top MB to top edge of current MB
        # Top MB bottom edge blocks: 10, 11, 14, 15
        # Current MB top edge blocks: 0, 1, 4, 5
        neighbor_blocks = [10, 11, 14, 15]
        current_blocks = [0, 1, 4, 5]
        pairs = list(zip(neighbor_blocks, current_blocks))

    return pairs
