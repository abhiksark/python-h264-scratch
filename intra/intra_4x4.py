# h264/intra/intra_4x4.py
"""Intra 4x4 prediction modes for H.264 I-frames.

H.264 Spec Reference: Section 8.3.1.2 - Intra_4x4 prediction

The nine 4x4 luma prediction modes use up to 13 neighboring pixels:
    M  A  B  C  D  E  F  G  H
    I [         4x4        ]
    J [       block        ]
    K [                    ]
    L [                    ]

Neighbor naming:
- top[0-3] = A, B, C, D (pixels above)
- top_right[0-3] = E, F, G, H (pixels above-right)
- left[0-3] = I, J, K, L (pixels to left)
- top_left = M (corner pixel)

Modes:
- 0: Vertical - extrapolate from top
- 1: Horizontal - extrapolate from left
- 2: DC - average of neighbors
- 3: Diagonal Down-Left - 45° from top-right
- 4: Diagonal Down-Right - 45° from top-left
- 5: Vertical-Right - 26.6° right of vertical
- 6: Horizontal-Down - 26.6° below horizontal
- 7: Vertical-Left - 26.6° left of vertical
- 8: Horizontal-Up - 26.6° above horizontal
"""

import logging
from enum import IntEnum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PIXEL_VALUE = 128


class Intra4x4Mode(IntEnum):
    """Intra 4x4 prediction modes."""
    VERTICAL = 0
    HORIZONTAL = 1
    DC = 2
    DIAGONAL_DOWN_LEFT = 3
    DIAGONAL_DOWN_RIGHT = 4
    VERTICAL_RIGHT = 5
    HORIZONTAL_DOWN = 6
    VERTICAL_LEFT = 7
    HORIZONTAL_UP = 8


def _clip(val: int) -> int:
    """Clip value to [0, 255]."""
    return max(0, min(255, val))


def _avg2(a: int, b: int) -> int:
    """Average of 2 values with rounding: (a + b + 1) >> 1."""
    return (int(a) + int(b) + 1) >> 1


def _avg3(a: int, b: int, c: int) -> int:
    """Weighted average: (a + 2*b + c + 2) >> 2."""
    return (int(a) + 2 * int(b) + int(c) + 2) >> 2


def intra_4x4_vertical(top: np.ndarray) -> np.ndarray:
    """Mode 0: Vertical prediction.

    H.264 Spec 8.3.1.2.1:
    pred[y, x] = top[x]

    Args:
        top: Top neighbors A, B, C, D (4 pixels)

    Returns:
        4x4 prediction block
    """
    pred = np.tile(top[:4].reshape(1, 4), (4, 1))
    return pred.astype(np.uint8)


def intra_4x4_horizontal(left: np.ndarray) -> np.ndarray:
    """Mode 1: Horizontal prediction.

    H.264 Spec 8.3.1.2.2:
    pred[y, x] = left[y]

    Args:
        left: Left neighbors I, J, K, L (4 pixels)

    Returns:
        4x4 prediction block
    """
    pred = np.tile(left[:4].reshape(4, 1), (1, 4))
    return pred.astype(np.uint8)


def intra_4x4_dc(
    top: Optional[np.ndarray],
    left: Optional[np.ndarray],
    top_available: bool = True,
    left_available: bool = True,
) -> np.ndarray:
    """Mode 2: DC prediction.

    H.264 Spec 8.3.1.2.3:
    - Both available: (sum(top) + sum(left) + 4) >> 3
    - Only top: (sum(top) + 2) >> 2
    - Only left: (sum(left) + 2) >> 2
    - Neither: 128

    Args:
        top: Top neighbors (4 pixels) or None
        left: Left neighbors (4 pixels) or None
        top_available: Whether top is available
        left_available: Whether left is available

    Returns:
        4x4 prediction block
    """
    if top_available and left_available:
        dc = (int(np.sum(top[:4])) + int(np.sum(left[:4])) + 4) >> 3
    elif top_available:
        dc = (int(np.sum(top[:4])) + 2) >> 2
    elif left_available:
        dc = (int(np.sum(left[:4])) + 2) >> 2
    else:
        dc = DEFAULT_PIXEL_VALUE

    return np.full((4, 4), dc, dtype=np.uint8)


def intra_4x4_diagonal_down_left(
    top: np.ndarray,
    top_right: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mode 3: Diagonal Down-Left prediction.

    H.264 Spec 8.3.1.2.4:
    Extrapolates at 45° from top-right toward bottom-left.

    Uses pixels: A B C D E F G H (top + top_right)
    If top_right unavailable, replicate D (top[3]).

    Args:
        top: Top neighbors A-D (4 pixels)
        top_right: Top-right neighbors E-H (4 pixels) or None

    Returns:
        4x4 prediction block
    """
    # Build extended top array [A B C D E F G H]
    t = np.zeros(8, dtype=np.int32)
    t[0:4] = top[:4]

    if top_right is not None:
        t[4:8] = top_right[:4]
    else:
        # Replicate D (top[3])
        t[4:8] = top[3]

    pred = np.zeros((4, 4), dtype=np.int32)

    for y in range(4):
        for x in range(4):
            if x == 3 and y == 3:
                # Special case: pred[3,3] = (t[6] + 3*t[7] + 2) >> 2
                pred[y, x] = (t[6] + 3 * t[7] + 2) >> 2
            else:
                # pred[y,x] = (t[x+y] + 2*t[x+y+1] + t[x+y+2] + 2) >> 2
                idx = x + y
                pred[y, x] = _avg3(t[idx], t[idx + 1], t[idx + 2])

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_4x4_diagonal_down_right(
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
) -> np.ndarray:
    """Mode 4: Diagonal Down-Right prediction.

    H.264 Spec 8.3.1.2.5:
    Extrapolates at 45° from top-left toward bottom-right.

    Uses pixels: M (top_left), A B C D (top), I J K L (left)

    Args:
        top: Top neighbors A-D (4 pixels)
        left: Left neighbors I-L (4 pixels)
        top_left: Corner pixel M

    Returns:
        4x4 prediction block
    """
    # Build arrays for easier indexing
    # p[-1] = M, p[0..3] = A B C D
    # q[-1] = M, q[0..3] = I J K L
    M = int(top_left)
    A, B, C, D = int(top[0]), int(top[1]), int(top[2]), int(top[3])
    I, J, K, L = int(left[0]), int(left[1]), int(left[2]), int(left[3])

    pred = np.zeros((4, 4), dtype=np.int32)

    for y in range(4):
        for x in range(4):
            if x > y:
                # Above diagonal: use top neighbors
                # pred[y,x] = avg3(top[x-y-2], top[x-y-1], top[x-y])
                idx = x - y - 1
                if idx == 0:
                    pred[y, x] = _avg3(M, A, B)
                elif idx == 1:
                    pred[y, x] = _avg3(A, B, C)
                elif idx == 2:
                    pred[y, x] = _avg3(B, C, D)
                else:  # idx == 3
                    pred[y, x] = _avg3(C, D, D)  # Replicate D
            elif x < y:
                # Below diagonal: use left neighbors
                idx = y - x - 1
                if idx == 0:
                    pred[y, x] = _avg3(M, I, J)
                elif idx == 1:
                    pred[y, x] = _avg3(I, J, K)
                elif idx == 2:
                    pred[y, x] = _avg3(J, K, L)
                else:  # idx == 3
                    pred[y, x] = _avg3(K, L, L)  # Replicate L
            else:
                # On diagonal (x == y): use corner
                pred[y, x] = _avg3(left[0], M, top[0])  # avg3(I, M, A)

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_4x4_vertical_right(
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
) -> np.ndarray:
    """Mode 5: Vertical-Right prediction.

    H.264 Spec 8.3.1.2.6:
    Extrapolates at 26.6° to the right of vertical.

    Args:
        top: Top neighbors A-D (4 pixels)
        left: Left neighbors I-L (4 pixels)
        top_left: Corner pixel M

    Returns:
        4x4 prediction block
    """
    # p[n, -1] for top: -1=M, 0=A, 1=B, 2=C, 3=D
    p_top = {-1: int(top_left), 0: int(top[0]), 1: int(top[1]),
             2: int(top[2]), 3: int(top[3])}
    # p[-1, n] for left: -1=M, 0=I, 1=J, 2=K, 3=L
    p_left = {-1: int(top_left), 0: int(left[0]), 1: int(left[1]),
              2: int(left[2]), 3: int(left[3])}

    pred = np.zeros((4, 4), dtype=np.int32)

    for y in range(4):
        for x in range(4):
            zVR = 2 * x - y
            if zVR >= 0 and zVR % 2 == 0:
                idx = x - (y >> 1)
                pred[y, x] = (p_top[idx - 1] + p_top[idx] + 1) >> 1
            elif zVR >= 0 and zVR % 2 == 1:
                idx = x - (y >> 1)
                pred[y, x] = (p_top[idx - 2] + 2 * p_top[idx - 1] + p_top[idx] + 2) >> 2
            elif zVR == -1:
                pred[y, x] = (p_left[0] + p_left[-1] + 1) >> 1
            else:  # zVR in {-2, -3}
                pred[y, x] = (p_left[y - 1] + 2 * p_left[y - 2] + p_left[y - 3] + 2) >> 2

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_4x4_horizontal_down(
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
) -> np.ndarray:
    """Mode 6: Horizontal-Down prediction.

    H.264 Spec 8.3.1.2.7:
    Extrapolates at 26.6° below horizontal.

    Args:
        top: Top neighbors A-D (4 pixels)
        left: Left neighbors I-L (4 pixels)
        top_left: Corner pixel M

    Returns:
        4x4 prediction block
    """
    # p[n, -1] for top: -1=M, 0=A, 1=B, 2=C, 3=D
    p_top = {-1: int(top_left), 0: int(top[0]), 1: int(top[1]),
             2: int(top[2]), 3: int(top[3])}
    # p[-1, n] for left: -1=M, 0=I, 1=J, 2=K, 3=L
    p_left = {-1: int(top_left), 0: int(left[0]), 1: int(left[1]),
              2: int(left[2]), 3: int(left[3])}

    pred = np.zeros((4, 4), dtype=np.int32)

    for y in range(4):
        for x in range(4):
            zHD = 2 * y - x
            if zHD >= 0 and zHD % 2 == 0:
                idx = y - (x >> 1)
                pred[y, x] = (p_left[idx - 1] + p_left[idx] + 1) >> 1
            elif zHD >= 0 and zHD % 2 == 1:
                idx = y - (x >> 1)
                pred[y, x] = (p_left[idx - 2] + 2 * p_left[idx - 1] + p_left[idx] + 2) >> 2
            elif zHD == -1:
                pred[y, x] = (p_top[0] + p_top[-1] + 1) >> 1
            else:  # zHD in {-2, -3}
                pred[y, x] = (p_top[x - 1] + 2 * p_top[x - 2] + p_top[x - 3] + 2) >> 2

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_4x4_vertical_left(
    top: np.ndarray,
    top_right: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mode 7: Vertical-Left prediction.

    H.264 Spec 8.3.1.2.8:
    Extrapolates at 26.6° to the left of vertical.

    Args:
        top: Top neighbors A-D (4 pixels)
        top_right: Top-right neighbors E-H (4 pixels) or None

    Returns:
        4x4 prediction block
    """
    # Build extended top array
    t = np.zeros(8, dtype=np.int32)
    t[0:4] = top[:4]
    if top_right is not None:
        t[4:8] = top_right[:4]
    else:
        t[4:8] = top[3]

    pred = np.zeros((4, 4), dtype=np.int32)

    for y in range(4):
        for x in range(4):
            if y % 2 == 0:
                # Even rows: avg2
                pred[y, x] = _avg2(t[x + (y >> 1)], t[x + (y >> 1) + 1])
            else:
                # Odd rows: avg3
                idx = x + (y >> 1)
                pred[y, x] = _avg3(t[idx], t[idx + 1], t[idx + 2])

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_4x4_horizontal_up(left: np.ndarray) -> np.ndarray:
    """Mode 8: Horizontal-Up prediction.

    H.264 Spec 8.3.1.2.9:
    Extrapolates upward and to the right from left neighbors.

    Args:
        left: Left neighbors I-L (4 pixels)

    Returns:
        4x4 prediction block
    """
    I, J, K, L = int(left[0]), int(left[1]), int(left[2]), int(left[3])
    l = [I, J, K, L, L, L, L, L]  # Extend with L for bounds

    pred = np.zeros((4, 4), dtype=np.int32)

    # zHU = x + 2*y
    for y in range(4):
        for x in range(4):
            zHU = x + 2 * y
            if zHU == 0:
                pred[y, x] = _avg2(I, J)
            elif zHU == 1:
                pred[y, x] = _avg3(I, J, K)
            elif zHU == 2:
                pred[y, x] = _avg2(J, K)
            elif zHU == 3:
                pred[y, x] = _avg3(J, K, L)
            elif zHU == 4:
                pred[y, x] = _avg2(K, L)
            elif zHU == 5:
                pred[y, x] = _avg3(K, L, L)
            else:  # zHU >= 6
                pred[y, x] = L

    return np.clip(pred, 0, 255).astype(np.uint8)


def predict_intra_4x4(
    mode: int,
    top: Optional[np.ndarray] = None,
    left: Optional[np.ndarray] = None,
    top_left: Optional[int] = None,
    top_right: Optional[np.ndarray] = None,
    top_available: bool = True,
    left_available: bool = True,
    top_right_available: bool = True,
) -> np.ndarray:
    """Generate 4x4 intra prediction block.

    Main entry point for 4x4 intra prediction.

    Args:
        mode: Prediction mode (0-8)
        top: Top neighbors (4 pixels)
        left: Left neighbors (4 pixels)
        top_left: Top-left corner pixel
        top_right: Top-right neighbors (4 pixels)
        top_available: Whether top is available
        left_available: Whether left is available
        top_right_available: Whether top-right is available

    Returns:
        4x4 prediction block (uint8)

    Raises:
        ValueError: If mode is invalid
    """
    logger.debug(f"Intra_4x4 prediction: mode={mode}")

    if mode == Intra4x4Mode.VERTICAL:
        if not top_available:
            return np.full((4, 4), DEFAULT_PIXEL_VALUE, dtype=np.uint8)
        return intra_4x4_vertical(top)

    elif mode == Intra4x4Mode.HORIZONTAL:
        if not left_available:
            return np.full((4, 4), DEFAULT_PIXEL_VALUE, dtype=np.uint8)
        return intra_4x4_horizontal(left)

    elif mode == Intra4x4Mode.DC:
        return intra_4x4_dc(top, left, top_available, left_available)

    elif mode == Intra4x4Mode.DIAGONAL_DOWN_LEFT:
        if not top_available:
            return np.full((4, 4), DEFAULT_PIXEL_VALUE, dtype=np.uint8)
        tr = top_right if top_right_available else None
        return intra_4x4_diagonal_down_left(top, tr)

    elif mode == Intra4x4Mode.DIAGONAL_DOWN_RIGHT:
        if not (top_available and left_available):
            return np.full((4, 4), DEFAULT_PIXEL_VALUE, dtype=np.uint8)
        tl = top_left if top_left is not None else DEFAULT_PIXEL_VALUE
        return intra_4x4_diagonal_down_right(top, left, tl)

    elif mode == Intra4x4Mode.VERTICAL_RIGHT:
        if not (top_available and left_available):
            return np.full((4, 4), DEFAULT_PIXEL_VALUE, dtype=np.uint8)
        tl = top_left if top_left is not None else DEFAULT_PIXEL_VALUE
        return intra_4x4_vertical_right(top, left, tl)

    elif mode == Intra4x4Mode.HORIZONTAL_DOWN:
        if not (top_available and left_available):
            return np.full((4, 4), DEFAULT_PIXEL_VALUE, dtype=np.uint8)
        tl = top_left if top_left is not None else DEFAULT_PIXEL_VALUE
        return intra_4x4_horizontal_down(top, left, tl)

    elif mode == Intra4x4Mode.VERTICAL_LEFT:
        if not top_available:
            return np.full((4, 4), DEFAULT_PIXEL_VALUE, dtype=np.uint8)
        tr = top_right if top_right_available else None
        return intra_4x4_vertical_left(top, tr)

    elif mode == Intra4x4Mode.HORIZONTAL_UP:
        if not left_available:
            return np.full((4, 4), DEFAULT_PIXEL_VALUE, dtype=np.uint8)
        return intra_4x4_horizontal_up(left)

    else:
        raise ValueError(f"Invalid Intra_4x4 mode: {mode}")
