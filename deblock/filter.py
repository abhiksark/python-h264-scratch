# h264/deblock/filter.py
"""Deblocking filter sample-level operations.

H.264 Spec Reference: Section 8.7.2.3 - Filtering process for edges
Formulas verified against JM reference: ~/JM/ldecod/src/loop_filter_normal.c
"""

from typing import Tuple
import numpy as np


def _clip(low: int, high: int, val: int) -> int:
    if val < low:
        return low
    if val > high:
        return high
    return val


def should_filter_edge(
    bs: int,
    alpha: int,
    beta: int,
    p0: int,
    p1: int,
    q0: int,
    q1: int,
) -> bool:
    """Determine if a sample row/column should be filtered.

    Section 8.7.2.3: filterSamplesFlag is true when all conditions met.

    Args:
        bs: Boundary strength (0-4)
        alpha, beta: Thresholds from tables
        p0, p1: Samples on p side (p0 adjacent to edge)
        q0, q1: Samples on q side (q0 adjacent to edge)

    Returns:
        True if this sample row/column should be filtered
    """
    if bs == 0:
        return False
    if abs(p0 - q0) >= alpha:
        return False
    if abs(p1 - p0) >= beta:
        return False
    if abs(q1 - q0) >= beta:
        return False
    return True


def filter_luma_strong_sample(
    p0: int, p1: int, p2: int, p3: int,
    q0: int, q1: int, q2: int, q3: int,
    alpha: int, beta: int,
) -> Tuple[int, int, int, int, int, int]:
    """Apply strong luma filter (bS=4) to one sample row/column.

    Section 8.7.2.3: Strong filtering with conditional paths.
    JM reference: loop_filter_normal.c lines 392-446.

    Args:
        p0-p3: Samples on p side (p0 adjacent to edge)
        q0-q3: Samples on q side (q0 adjacent to edge)
        alpha, beta: Thresholds

    Returns:
        (p0', p1', p2', q0', q1', q2') - filtered samples
    """
    p0n, p1n, p2n = p0, p1, p2
    q0n, q1n, q2n = q0, q1, q2

    if abs(p0 - q0) < ((alpha >> 2) + 2):
        # Strong condition met - can use full multi-tap filter
        if abs(p2 - p0) < beta:
            # Full strong filter on p side (3 pixels modified)
            p0n = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3
            p1n = (p2 + p1 + p0 + q0 + 2) >> 2
            p2n = (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3
        else:
            # Weak filter on p side (1 pixel modified)
            p0n = (2 * p1 + p0 + q1 + 2) >> 2

        if abs(q2 - q0) < beta:
            # Full strong filter on q side (3 pixels modified)
            q0n = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3
            q1n = (p0 + q0 + q1 + q2 + 2) >> 2
            q2n = (2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3
        else:
            # Weak filter on q side (1 pixel modified)
            q0n = (2 * q1 + q0 + p1 + 2) >> 2
    else:
        # Strong condition NOT met - only modify p0, q0
        p0n = (2 * p1 + p0 + q1 + 2) >> 2
        q0n = (2 * q1 + q0 + p1 + 2) >> 2

    return (
        _clip(0, 255, p0n), _clip(0, 255, p1n), _clip(0, 255, p2n),
        _clip(0, 255, q0n), _clip(0, 255, q1n), _clip(0, 255, q2n),
    )


def filter_luma_normal_sample(
    p0: int, p1: int, p2: int,
    q0: int, q1: int, q2: int,
    tc0: int, beta: int,
) -> Tuple[int, int, int, int]:
    """Apply normal luma filter (bS=1,2,3) to one sample row/column.

    Section 8.7.2.3: Normal filtering with optional p1/q1 adjustment.
    JM reference: loop_filter_normal.c lines 454-533.

    Args:
        p0-p2: Samples on p side
        q0-q2: Samples on q side
        tc0: Base clipping threshold from TC0_TABLE
        beta: Beta threshold

    Returns:
        (p0', p1', q0', q1') - filtered samples
    """
    ap = 1 if abs(p2 - p0) < beta else 0
    aq = 1 if abs(q2 - q0) < beta else 0
    tc = tc0 + ap + aq

    delta = _clip(-tc, tc, (4 * (q0 - p0) + (p1 - q1) + 4) >> 3)
    p0n = _clip(0, 255, p0 + delta)
    q0n = _clip(0, 255, q0 - delta)

    p1n = p1
    q1n = q1
    if ap and tc0 > 0:
        p1n = p1 + _clip(-tc0, tc0, (p2 + ((p0 + q0 + 1) >> 1) - 2 * p1) >> 1)
        p1n = _clip(0, 255, p1n)
    if aq and tc0 > 0:
        q1n = q1 + _clip(-tc0, tc0, (q2 + ((p0 + q0 + 1) >> 1) - 2 * q1) >> 1)
        q1n = _clip(0, 255, q1n)

    return p0n, p1n, q0n, q1n


def filter_chroma_sample(
    p0: int, p1: int,
    q0: int, q1: int,
    tc0: int,
) -> Tuple[int, int]:
    """Apply normal chroma filter (bS=1,2,3) to one sample row/column.

    Section 8.7.2.3: Chroma filtering only modifies p0, q0.
    tc = tc0 + 1 (always, no ap/aq for chroma).

    Args:
        p0, p1: Samples on p side
        q0, q1: Samples on q side
        tc0: Base clipping threshold from TC0_TABLE

    Returns:
        (p0', q0') - filtered samples
    """
    tc = tc0 + 1
    delta = _clip(-tc, tc, (4 * (q0 - p0) + (p1 - q1) + 4) >> 3)
    p0n = _clip(0, 255, p0 + delta)
    q0n = _clip(0, 255, q0 - delta)
    return p0n, q0n


def filter_chroma_strong_sample(
    p0: int, p1: int,
    q0: int, q1: int,
) -> Tuple[int, int]:
    """Apply strong chroma filter (bS=4) to one sample row/column.

    Section 8.7.2.3: Simple 3-tap averaging, no tc clipping.
    JM reference: loop_filter_normal.c, Strng == 4 chroma path.

    Args:
        p0, p1: Samples on p side
        q0, q1: Samples on q side

    Returns:
        (p0', q0') - filtered samples
    """
    p0n = (2 * p1 + p0 + q1 + 2) >> 2
    q0n = (2 * q1 + q0 + p1 + 2) >> 2
    return _clip(0, 255, p0n), _clip(0, 255, q0n)


# ---------------------------------------------------------------------------
# Legacy wrappers (used by existing tests)
# ---------------------------------------------------------------------------

def filter_luma_edge_strong(
    p: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy wrapper: strong filter with array interface.

    Args:
        p: Array [p2, p1, p0]
        q: Array [q0, q1, q2]
    """
    p2, p1, p0 = int(p[0]), int(p[1]), int(p[2])
    q0, q1, q2 = int(q[0]), int(q[1]), int(q[2])
    # Use p3=p2, q3=q2 as fallback (no p3/q3 in this interface)
    p0n, p1n, p2n, q0n, q1n, q2n = filter_luma_strong_sample(
        p0, p1, p2, p2, q0, q1, q2, q2, alpha=255, beta=255,
    )
    return (
        np.array([p2n, p1n, p0n], dtype=np.int32),
        np.array([q0n, q1n, q2n], dtype=np.int32),
    )


def filter_luma_edge_normal(
    p: np.ndarray,
    q: np.ndarray,
    bs: int,
    tc: int,
) -> Tuple[int, int]:
    """Legacy wrapper: normal filter returning only (p0', q0')."""
    p1, p0 = int(p[0]), int(p[1])
    q0, q1 = int(q[0]), int(q[1])
    delta = _clip(-tc, tc, (4 * (q0 - p0) + (p1 - q1) + 4) >> 3)
    return _clip(0, 255, p0 + delta), _clip(0, 255, q0 - delta)


def filter_chroma_edge(
    p: np.ndarray,
    q: np.ndarray,
    bs: int,
    tc: int,
) -> Tuple[int, int]:
    """Legacy wrapper: chroma filter with array interface."""
    if bs == 0:
        return int(p[1]), int(q[0])
    p1, p0 = int(p[0]), int(p[1])
    q0, q1 = int(q[0]), int(q[1])
    delta = _clip(-tc, tc, (4 * (q0 - p0) + (p1 - q1) + 4) >> 3)
    return _clip(0, 255, p0 + delta), _clip(0, 255, q0 - delta)


def filter_chroma_vertical_edge(
    chroma: np.ndarray,
    edge_x: int,
    bs: int,
    alpha: int,
    beta: int,
    tc: int,
) -> np.ndarray:
    """Legacy wrapper: filter vertical chroma edge."""
    result = chroma.copy()
    if bs == 0 or edge_x < 1 or edge_x >= chroma.shape[1]:
        return result
    for y in range(chroma.shape[0]):
        p0 = int(chroma[y, edge_x - 1])
        p1 = int(chroma[y, edge_x - 2]) if edge_x >= 2 else p0
        q0 = int(chroma[y, edge_x])
        q1 = int(chroma[y, edge_x + 1]) if edge_x + 1 < chroma.shape[1] else q0
        if not should_filter_edge(bs, alpha, beta, p0, p1, q0, q1):
            continue
        p0n, q0n = filter_chroma_sample(p0, p1, q0, q1, tc - 1)
        result[y, edge_x - 1] = p0n
        result[y, edge_x] = q0n
    return result


def filter_chroma_horizontal_edge(
    chroma: np.ndarray,
    edge_y: int,
    bs: int,
    alpha: int,
    beta: int,
    tc: int,
) -> np.ndarray:
    """Legacy wrapper: filter horizontal chroma edge."""
    result = chroma.copy()
    if bs == 0 or edge_y < 1 or edge_y >= chroma.shape[0]:
        return result
    for x in range(chroma.shape[1]):
        p0 = int(chroma[edge_y - 1, x])
        p1 = int(chroma[edge_y - 2, x]) if edge_y >= 2 else p0
        q0 = int(chroma[edge_y, x])
        q1 = int(chroma[edge_y + 1, x]) if edge_y + 1 < chroma.shape[0] else q0
        if not should_filter_edge(bs, alpha, beta, p0, p1, q0, q1):
            continue
        p0n, q0n = filter_chroma_sample(p0, p1, q0, q1, tc - 1)
        result[edge_y - 1, x] = p0n
        result[edge_y, x] = q0n
    return result
