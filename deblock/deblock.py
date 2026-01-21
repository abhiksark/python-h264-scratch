# h264/deblock/deblock.py
"""Main deblocking filter orchestration.

H.264 Spec Reference: Section 8.7 - Deblocking filter process
"""

from typing import List, Tuple, Dict
import numpy as np

from deblock.boundary import calc_boundary_strength
from deblock.thresholds import get_alpha, get_beta, get_tc0
from deblock.filter import should_filter_edge, filter_luma_edge_normal


def get_edge_filter_order() -> List[str]:
    """Get the order in which edges are filtered.

    Per H.264 spec, vertical edges are filtered first,
    then horizontal edges.

    Returns:
        List of edge directions in filter order
    """
    return ['vertical', 'horizontal']


def get_luma_edge_positions(direction: str) -> List[Tuple[int, int]]:
    """Get positions of edges to filter in a luma macroblock.

    Args:
        direction: 'vertical' or 'horizontal'

    Returns:
        List of (x, y) positions for edge filtering.
        For vertical edges, x is the column position.
        For horizontal edges, y is the row position.
    """
    positions = []

    if direction == 'vertical':
        # Vertical edges at x = 0, 4, 8, 12
        # For each edge, filter 4 rows of 4 pixels
        for x in [0, 4, 8, 12]:
            for y_block in range(4):
                positions.append((x, y_block * 4))
    else:
        # Horizontal edges at y = 0, 4, 8, 12
        for y in [0, 4, 8, 12]:
            for x_block in range(4):
                positions.append((x_block * 4, y))

    return positions


def deblock_macroblock(
    luma: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    block_info: Dict,
    qp: int,
    alpha_offset: int = 0,
    beta_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply deblocking filter to a macroblock.

    Args:
        luma: 16x16 luma samples
        cb: 8x8 Cb samples
        cr: 8x8 Cr samples
        block_info: Dictionary with per-4x4-block info:
            - is_intra: bool
            - has_coeff: bool array (16,)
            - mvs: int array (16, 2)
            - refs: int array (16,)
        qp: Quantization parameter
        alpha_offset: Slice alpha offset
        beta_offset: Slice beta offset

    Returns:
        Tuple of filtered (luma, cb, cr)
    """
    luma_out = luma.copy()
    cb_out = cb.copy()
    cr_out = cr.copy()

    # Get thresholds
    index_a = max(0, min(51, qp + alpha_offset))
    index_b = max(0, min(51, qp + beta_offset))
    alpha = get_alpha(index_a)
    beta = get_beta(index_b)

    is_intra = block_info['is_intra']
    has_coeff = block_info['has_coeff']
    mvs = block_info['mvs']
    refs = block_info['refs']

    # Filter vertical edges then horizontal
    for direction in get_edge_filter_order():
        _filter_luma_edges(
            luma_out, direction, is_intra, has_coeff, mvs, refs,
            alpha, beta, index_a
        )

    return luma_out, cb_out, cr_out


def _filter_luma_edges(
    luma: np.ndarray,
    direction: str,
    is_intra: bool,
    has_coeff: np.ndarray,
    mvs: np.ndarray,
    refs: np.ndarray,
    alpha: int,
    beta: int,
    index_a: int,
) -> None:
    """Filter luma edges in one direction (in-place).

    Args:
        luma: 16x16 luma array to filter in-place
        direction: 'vertical' or 'horizontal'
        is_intra: True if macroblock is intra-coded
        has_coeff: Per-block coefficient flags
        mvs: Per-block motion vectors
        refs: Per-block reference indices
        alpha, beta: Filter thresholds
        index_a: Index for tc0 lookup
    """
    # For simplicity, skip filtering if bS would be 0 everywhere
    # (all inter, no coeffs, same MVs and refs)
    if not is_intra and not has_coeff.any():
        # Check if all MVs and refs are the same
        if np.all(mvs == mvs[0]) and np.all(refs == refs[0]):
            return  # bS=0 everywhere, no filtering needed

    # Full filtering implementation would iterate over edges
    # and apply filter based on bS at each edge
    # For now, this is a stub that returns without modification
    pass


def _get_4x4_block_index(x: int, y: int) -> int:
    """Get 4x4 block index from pixel position.

    Args:
        x, y: Pixel position within macroblock

    Returns:
        Block index (0-15)
    """
    bx = x // 4
    by = y // 4

    # Convert to raster scan order within 8x8 blocks
    # Block layout:
    #  0  1  4  5
    #  2  3  6  7
    #  8  9 12 13
    # 10 11 14 15

    # Determine which 8x8 quadrant
    qx = bx // 2
    qy = by // 2
    quadrant = qy * 2 + qx

    # Position within 8x8 quadrant
    lx = bx % 2
    ly = by % 2
    local_idx = ly * 2 + lx

    # Map quadrant and local index to block index
    quadrant_bases = [0, 4, 8, 12]
    local_offsets = [0, 1, 2, 3]

    return quadrant_bases[quadrant] + local_offsets[local_idx]
