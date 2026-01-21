# h264/deblock/filter.py
"""Deblocking filter operations.

H.264 Spec Reference: Section 8.7.2.3 - Filtering process
"""

from typing import Tuple
import numpy as np


def should_filter_edge(
    bs: int,
    alpha: int,
    beta: int,
    p0: int,
    p1: int,
    q0: int,
    q1: int,
) -> bool:
    """Determine if an edge should be filtered.

    Filtering is applied only when:
    - bS > 0
    - |p0 - q0| < alpha
    - |p1 - p0| < beta
    - |q1 - q0| < beta

    Args:
        bs: Boundary strength
        alpha: Alpha threshold
        beta: Beta threshold
        p0, p1: Samples on p side (p0 adjacent to edge)
        q0, q1: Samples on q side (q0 adjacent to edge)

    Returns:
        True if edge should be filtered
    """
    if bs == 0:
        return False

    # Check thresholds
    if abs(p0 - q0) >= alpha:
        return False

    if abs(p1 - p0) >= beta:
        return False

    if abs(q1 - q0) >= beta:
        return False

    return True


def filter_luma_edge_strong(
    p: np.ndarray,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply strong filtering (bS=4) to luma samples.

    Strong filter modifies p0, p1, p2 and q0, q1, q2.

    Args:
        p: Array [p2, p1, p0] - samples on p side
        q: Array [q0, q1, q2] - samples on q side

    Returns:
        Tuple of (p_new, q_new) with filtered samples
    """
    p2, p1, p0 = int(p[0]), int(p[1]), int(p[2])
    q0, q1, q2 = int(q[0]), int(q[1]), int(q[2])

    # Strong filter formulas from H.264 spec
    p0_new = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3
    p1_new = (p2 + p1 + p0 + q0 + 2) >> 2
    p2_new = (2 * p2 + 3 * p1 + p0 + q0 + 4) >> 3

    q0_new = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3
    q1_new = (p0 + q0 + q1 + q2 + 2) >> 2
    q2_new = (p0 + q0 + 3 * q1 + 2 * q2 + 4) >> 3

    p_new = np.array([p2_new, p1_new, p0_new], dtype=np.int32)
    q_new = np.array([q0_new, q1_new, q2_new], dtype=np.int32)

    # Clip to valid range
    p_new = np.clip(p_new, 0, 255)
    q_new = np.clip(q_new, 0, 255)

    return p_new, q_new


def filter_luma_edge_normal(
    p: np.ndarray,
    q: np.ndarray,
    bs: int,
    tc: int,
) -> Tuple[int, int]:
    """Apply normal filtering (bS=1,2,3) to luma samples.

    Normal filter modifies only p0 and q0.

    Args:
        p: Array [p1, p0] - samples on p side
        q: Array [q0, q1] - samples on q side
        bs: Boundary strength (1-3)
        tc: Clipping threshold (tc0 + 1 if ap < beta else tc0)

    Returns:
        Tuple of (p0_new, q0_new)
    """
    p1, p0 = int(p[0]), int(p[1])
    q0, q1 = int(q[0]), int(q[1])

    # Delta calculation
    delta = (4 * (q0 - p0) + (p1 - q1) + 4) >> 3
    delta = max(-tc, min(tc, delta))

    p0_new = np.clip(p0 + delta, 0, 255)
    q0_new = np.clip(q0 - delta, 0, 255)

    return int(p0_new), int(q0_new)


def filter_chroma_edge(
    p: np.ndarray,
    q: np.ndarray,
    bs: int,
    tc: int,
) -> Tuple[int, int]:
    """Apply filtering to chroma samples.

    Chroma filtering is similar to normal luma filtering but
    uses different tc calculation.

    Args:
        p: Array [p1, p0] - samples on p side
        q: Array [q0, q1] - samples on q side
        bs: Boundary strength
        tc: Clipping threshold

    Returns:
        Tuple of (p0_new, q0_new)
    """
    if bs == 0:
        return int(p[1]), int(q[0])

    p1, p0 = int(p[0]), int(p[1])
    q0, q1 = int(q[0]), int(q[1])

    # Same delta calculation as luma normal filter
    delta = (4 * (q0 - p0) + (p1 - q1) + 4) >> 3
    delta = max(-tc, min(tc, delta))

    p0_new = np.clip(p0 + delta, 0, 255)
    q0_new = np.clip(q0 - delta, 0, 255)

    return int(p0_new), int(q0_new)


def filter_chroma_vertical_edge(
    chroma: np.ndarray,
    edge_x: int,
    bs: int,
    alpha: int,
    beta: int,
    tc: int,
) -> np.ndarray:
    """Filter vertical edge in chroma plane.

    Args:
        chroma: 8x8 chroma block
        edge_x: X position of edge (column index)
        bs: Boundary strength
        alpha: Alpha threshold
        beta: Beta threshold
        tc: Clipping threshold

    Returns:
        Filtered chroma block
    """
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

        p = np.array([p1, p0], dtype=np.int32)
        q = np.array([q0, q1], dtype=np.int32)
        p0_new, q0_new = filter_chroma_edge(p, q, bs, tc)

        result[y, edge_x - 1] = p0_new
        result[y, edge_x] = q0_new

    return result


def filter_chroma_horizontal_edge(
    chroma: np.ndarray,
    edge_y: int,
    bs: int,
    alpha: int,
    beta: int,
    tc: int,
) -> np.ndarray:
    """Filter horizontal edge in chroma plane.

    Args:
        chroma: 8x8 chroma block
        edge_y: Y position of edge (row index)
        bs: Boundary strength
        alpha: Alpha threshold
        beta: Beta threshold
        tc: Clipping threshold

    Returns:
        Filtered chroma block
    """
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

        p = np.array([p1, p0], dtype=np.int32)
        q = np.array([q0, q1], dtype=np.int32)
        p0_new, q0_new = filter_chroma_edge(p, q, bs, tc)

        result[edge_y - 1, x] = p0_new
        result[edge_y, x] = q0_new

    return result
