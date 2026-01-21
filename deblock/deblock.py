# h264/deblock/deblock.py
"""Main deblocking filter orchestration.

H.264 Spec Reference: Section 8.7 - Deblocking filter process
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

from deblock.boundary import calc_boundary_strength
from deblock.thresholds import get_alpha, get_beta, get_tc0, get_chroma_qp
from deblock.filter import (
    should_filter_edge,
    filter_luma_edge_normal,
    filter_chroma_edge,
)


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

    # Calculate bS for this MB (simplified - use bS=4 if intra, else check coeffs)
    bs = 4 if is_intra else 3 if has_coeff.any() else 0

    if bs == 0:
        return

    # Get tc0 for this bS and QP
    tc0 = get_tc0(index_a, bs)
    tc = tc0 + 1  # tc = tc0 + (ap < beta ? 1 : 0) + (aq < beta ? 1 : 0)

    # Filter internal edges (4x4 block boundaries at positions 4, 8, 12)
    if direction == 'vertical':
        for edge_x in [4, 8, 12]:
            for y in range(16):
                p0 = int(luma[y, edge_x - 1])
                p1 = int(luma[y, edge_x - 2])
                q0 = int(luma[y, edge_x])
                q1 = int(luma[y, edge_x + 1])

                # Check thresholds
                if abs(p0 - q0) >= alpha:
                    continue
                if abs(p1 - p0) >= beta:
                    continue
                if abs(q1 - q0) >= beta:
                    continue

                # Apply normal filter
                delta = np.clip((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3, -tc, tc)
                luma[y, edge_x - 1] = np.clip(p0 + delta, 0, 255)
                luma[y, edge_x] = np.clip(q0 - delta, 0, 255)
    else:
        for edge_y in [4, 8, 12]:
            for x in range(16):
                p0 = int(luma[edge_y - 1, x])
                p1 = int(luma[edge_y - 2, x])
                q0 = int(luma[edge_y, x])
                q1 = int(luma[edge_y + 1, x])

                # Check thresholds
                if abs(p0 - q0) >= alpha:
                    continue
                if abs(p1 - p0) >= beta:
                    continue
                if abs(q1 - q0) >= beta:
                    continue

                # Apply normal filter
                delta = np.clip((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3, -tc, tc)
                luma[edge_y - 1, x] = np.clip(p0 + delta, 0, 255)
                luma[edge_y, x] = np.clip(q0 - delta, 0, 255)


def deblock_frame(
    luma: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    mb_info: Optional[Dict],
    qp: int,
    alpha_offset: int = 0,
    beta_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply deblocking filter to entire frame.

    Processes macroblocks in raster scan order, filtering vertical
    edges first, then horizontal edges within each MB.

    Args:
        luma: Full frame luma (H x W)
        cb: Full frame Cb (H/2 x W/2)
        cr: Full frame Cr (H/2 x W/2)
        mb_info: Per-MB info dictionary or None
        qp: Base quantization parameter
        alpha_offset: Slice alpha offset
        beta_offset: Slice beta offset

    Returns:
        Tuple of filtered (luma, cb, cr)
    """
    height, width = luma.shape
    mb_height = height // 16
    mb_width = width // 16

    luma_out = luma.copy()
    cb_out = cb.copy()
    cr_out = cr.copy()

    # Process MBs in raster order
    for mb_y, mb_x in get_deblock_mb_order(mb_width, mb_height):
        # Extract MB region
        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        mb_luma = luma_out[ly:ly + 16, lx:lx + 16]
        mb_cb = cb_out[cy:cy + 8, cx:cx + 8]
        mb_cr = cr_out[cy:cy + 8, cx:cx + 8]

        # Default block info if not provided
        # Assume intra blocks when no info available to ensure filtering happens
        if mb_info is None:
            block_info = {
                'is_intra': True,  # Default to intra to ensure filtering
                'has_coeff': np.zeros(16, dtype=bool),
                'mvs': np.zeros((16, 2), dtype=np.int32),
                'refs': np.zeros(16, dtype=np.int32),
            }
        else:
            mb_idx = mb_y * mb_width + mb_x
            block_info = mb_info.get(mb_idx, {
                'is_intra': False,
                'has_coeff': np.zeros(16, dtype=bool),
                'mvs': np.zeros((16, 2), dtype=np.int32),
                'refs': np.zeros(16, dtype=np.int32),
            })

        # Deblock luma (internal edges)
        filtered_luma, _, _ = deblock_macroblock(
            mb_luma, mb_cb, mb_cr, block_info, qp, alpha_offset, beta_offset
        )

        # Deblock chroma planes
        filtered_cb = deblock_chroma_mb(mb_cb, block_info, qp, alpha_offset, beta_offset)
        filtered_cr = deblock_chroma_mb(mb_cr, block_info, qp, alpha_offset, beta_offset)

        # Write back
        luma_out[ly:ly + 16, lx:lx + 16] = filtered_luma
        cb_out[cy:cy + 8, cx:cx + 8] = filtered_cb
        cr_out[cy:cy + 8, cx:cx + 8] = filtered_cr

        # Filter MB boundaries
        left_info = None
        top_info = None
        if mb_x > 0:
            left_info = {'is_intra': True}  # Assume intra when no info
        if mb_y > 0:
            top_info = {'is_intra': True}

        if left_info or top_info:
            deblock_macroblock_with_neighbors(
                luma_out, mb_x, mb_y, block_info,
                left_info, top_info, qp, alpha_offset, beta_offset
            )

    return luma_out, cb_out, cr_out


def deblock_macroblock_in_frame(
    frame_luma: np.ndarray,
    frame_cb: np.ndarray,
    frame_cr: np.ndarray,
    mb_x: int,
    mb_y: int,
    block_info: Dict,
    neighbor_info: Optional[Dict],
    qp: int,
    alpha_offset: int = 0,
    beta_offset: int = 0,
) -> None:
    """Apply deblocking filter to a macroblock in-place within frame.

    Uses neighbor information for filtering MB boundary edges.

    Args:
        frame_luma: Full frame luma array (modified in-place)
        frame_cb: Full frame Cb array (modified in-place)
        frame_cr: Full frame Cr array (modified in-place)
        mb_x, mb_y: Macroblock position
        block_info: Info for current MB
        neighbor_info: Info for left and top neighbor MBs
        qp: Quantization parameter
        alpha_offset, beta_offset: Slice filter offsets
    """
    ly, lx = mb_y * 16, mb_x * 16
    cy, cx = mb_y * 8, mb_x * 8

    mb_luma = frame_luma[ly:ly + 16, lx:lx + 16].copy()
    mb_cb = frame_cb[cy:cy + 8, cx:cx + 8].copy()
    mb_cr = frame_cr[cy:cy + 8, cx:cx + 8].copy()

    filtered_luma, filtered_cb, filtered_cr = deblock_macroblock(
        mb_luma, mb_cb, mb_cr, block_info, qp, alpha_offset, beta_offset
    )

    frame_luma[ly:ly + 16, lx:lx + 16] = filtered_luma
    frame_cb[cy:cy + 8, cx:cx + 8] = filtered_cb
    frame_cr[cy:cy + 8, cx:cx + 8] = filtered_cr


def get_deblock_mb_order(width_mbs: int, height_mbs: int) -> List[Tuple[int, int]]:
    """Get macroblock processing order for deblocking.

    Per H.264 spec, MBs are processed in raster scan order.

    Args:
        width_mbs: Frame width in macroblocks
        height_mbs: Frame height in macroblocks

    Returns:
        List of (mb_y, mb_x) tuples in processing order
    """
    order = []
    for mb_y in range(height_mbs):
        for mb_x in range(width_mbs):
            order.append((mb_x, mb_y))
    return order


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


def get_chroma_bs_from_luma(luma_bs: np.ndarray) -> np.ndarray:
    """Get chroma boundary strength from luma bS values.

    Chroma 4x4 blocks correspond to 8x8 luma regions.
    Chroma bS is the max of the corresponding luma block bS values.

    Args:
        luma_bs: 4x4 array of luma bS values (one per 4x4 block)

    Returns:
        2x2 array of chroma bS values
    """
    chroma_bs = np.zeros((2, 2), dtype=np.int32)

    # Each chroma 4x4 corresponds to a 2x2 group of luma 4x4 blocks
    for cy in range(2):
        for cx in range(2):
            ly = cy * 2
            lx = cx * 2
            chroma_bs[cy, cx] = np.max(luma_bs[ly:ly + 2, lx:lx + 2])

    return chroma_bs


def deblock_chroma_mb(
    chroma: np.ndarray,
    block_info: Dict,
    qp: int,
    alpha_offset: int = 0,
    beta_offset: int = 0,
) -> np.ndarray:
    """Apply deblocking filter to a chroma macroblock plane.

    Args:
        chroma: 8x8 chroma samples (Cb or Cr)
        block_info: Dictionary with block info
        qp: Luma quantization parameter
        alpha_offset: Slice alpha offset
        beta_offset: Slice beta offset

    Returns:
        Filtered 8x8 chroma block
    """
    result = chroma.copy()

    # Get chroma QP
    qpc = get_chroma_qp(qp)
    index_a = max(0, min(51, qpc + alpha_offset))
    index_b = max(0, min(51, qpc + beta_offset))
    alpha = get_alpha(index_a)
    beta = get_beta(index_b)

    is_intra = block_info.get('is_intra', False)
    has_coeff = block_info.get('has_coeff', np.zeros(4, dtype=bool))

    # Calculate bS
    bs = 4 if is_intra else 3 if np.any(has_coeff) else 0

    if bs == 0:
        return result

    tc0 = get_tc0(index_a, min(bs, 3))
    tc = tc0 + 1

    # Filter vertical edge at x=4
    for y in range(8):
        p0 = int(result[y, 3])
        p1 = int(result[y, 2])
        q0 = int(result[y, 4])
        q1 = int(result[y, 5])

        if not should_filter_edge(bs, alpha, beta, p0, p1, q0, q1):
            continue

        delta = np.clip((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3, -tc, tc)
        result[y, 3] = np.clip(p0 + delta, 0, 255)
        result[y, 4] = np.clip(q0 - delta, 0, 255)

    # Filter horizontal edge at y=4
    for x in range(8):
        p0 = int(result[3, x])
        p1 = int(result[2, x])
        q0 = int(result[4, x])
        q1 = int(result[5, x])

        if not should_filter_edge(bs, alpha, beta, p0, p1, q0, q1):
            continue

        delta = np.clip((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3, -tc, tc)
        result[3, x] = np.clip(p0 + delta, 0, 255)
        result[4, x] = np.clip(q0 - delta, 0, 255)

    return result


def calc_mb_boundary_strength(
    current_mb: Dict,
    neighbor_mb: Dict,
    is_mb_edge: bool = True,
) -> int:
    """Calculate boundary strength at macroblock boundary.

    Args:
        current_mb: Info for current macroblock
        neighbor_mb: Info for neighbor macroblock
        is_mb_edge: True if this is an MB boundary (always True for this function)

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


def filter_mb_boundary(
    neighbor_luma: np.ndarray,
    current_luma: np.ndarray,
    direction: str,
    bs: int,
    alpha: int,
    beta: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter edge at macroblock boundary.

    Args:
        neighbor_luma: 16x16 luma of neighbor MB
        current_luma: 16x16 luma of current MB
        direction: 'vertical' or 'horizontal'
        bs: Boundary strength
        alpha: Alpha threshold
        beta: Beta threshold

    Returns:
        Tuple of (filtered_neighbor, filtered_current)
    """
    neighbor_out = neighbor_luma.copy()
    current_out = current_luma.copy()

    if bs == 0:
        return neighbor_out, current_out

    tc0 = get_tc0(26, min(bs, 3))  # Use QP=26 as default
    tc = tc0 + 1

    if direction == 'vertical':
        # Filter right edge of neighbor (col 15) against left edge of current (col 0)
        for y in range(16):
            p0 = int(neighbor_luma[y, 15])
            p1 = int(neighbor_luma[y, 14])
            q0 = int(current_luma[y, 0])
            q1 = int(current_luma[y, 1])

            if not should_filter_edge(bs, alpha, beta, p0, p1, q0, q1):
                continue

            delta = np.clip((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3, -tc, tc)
            neighbor_out[y, 15] = np.clip(p0 + delta, 0, 255)
            current_out[y, 0] = np.clip(q0 - delta, 0, 255)

    else:  # horizontal
        # Filter bottom edge of neighbor (row 15) against top edge of current (row 0)
        for x in range(16):
            p0 = int(neighbor_luma[15, x])
            p1 = int(neighbor_luma[14, x])
            q0 = int(current_luma[0, x])
            q1 = int(current_luma[1, x])

            if not should_filter_edge(bs, alpha, beta, p0, p1, q0, q1):
                continue

            delta = np.clip((((q0 - p0) << 2) + (p1 - q1) + 4) >> 3, -tc, tc)
            neighbor_out[15, x] = np.clip(p0 + delta, 0, 255)
            current_out[0, x] = np.clip(q0 - delta, 0, 255)

    return neighbor_out, current_out


def deblock_macroblock_with_neighbors(
    frame_luma: np.ndarray,
    mb_x: int,
    mb_y: int,
    current_info: Dict,
    left_info: Optional[Dict],
    top_info: Optional[Dict],
    qp: int,
    alpha_offset: int = 0,
    beta_offset: int = 0,
) -> None:
    """Deblock macroblock including MB boundary edges (in-place).

    Args:
        frame_luma: Full frame luma (modified in-place)
        mb_x, mb_y: Macroblock position
        current_info: Info for current MB
        left_info: Info for left neighbor MB (or None)
        top_info: Info for top neighbor MB (or None)
        qp: Quantization parameter
        alpha_offset, beta_offset: Filter offsets
    """
    ly, lx = mb_y * 16, mb_x * 16
    index_a = max(0, min(51, qp + alpha_offset))
    alpha = get_alpha(index_a)
    beta = get_beta(max(0, min(51, qp + beta_offset)))

    # Filter left MB boundary if left neighbor exists
    if left_info is not None and mb_x > 0:
        bs = calc_mb_boundary_strength(current_info, left_info)
        if bs > 0:
            left_lx = (mb_x - 1) * 16
            neighbor = frame_luma[ly:ly + 16, left_lx:left_lx + 16]
            current = frame_luma[ly:ly + 16, lx:lx + 16]

            filtered_neighbor, filtered_current = filter_mb_boundary(
                neighbor, current, 'vertical', bs, alpha, beta
            )

            frame_luma[ly:ly + 16, left_lx:left_lx + 16] = filtered_neighbor
            frame_luma[ly:ly + 16, lx:lx + 16] = filtered_current

    # Filter top MB boundary if top neighbor exists
    if top_info is not None and mb_y > 0:
        bs = calc_mb_boundary_strength(current_info, top_info)
        if bs > 0:
            top_ly = (mb_y - 1) * 16
            neighbor = frame_luma[top_ly:top_ly + 16, lx:lx + 16]
            current = frame_luma[ly:ly + 16, lx:lx + 16]

            filtered_neighbor, filtered_current = filter_mb_boundary(
                neighbor, current, 'horizontal', bs, alpha, beta
            )

            frame_luma[top_ly:top_ly + 16, lx:lx + 16] = filtered_neighbor
            frame_luma[ly:ly + 16, lx:lx + 16] = filtered_current


def get_mb_edge_filter_order() -> List[str]:
    """Get order for filtering MB edges.

    Returns:
        List of edge types in filter order
    """
    return ['left_boundary', 'internal_vertical', 'top_boundary', 'internal_horizontal']


def should_filter_mb_edge(mb_x: int, mb_y: int, direction: str) -> bool:
    """Check if an MB boundary edge should be filtered.

    Args:
        mb_x, mb_y: Macroblock position
        direction: 'left' or 'top'

    Returns:
        True if edge should be filtered
    """
    if direction == 'left':
        return mb_x > 0
    elif direction == 'top':
        return mb_y > 0
    return False
