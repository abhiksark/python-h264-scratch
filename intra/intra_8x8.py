# h264/intra/intra_8x8.py
"""Intra 8x8 prediction modes for H.264 High profile.

H.264 Spec Reference: Section 8.3.1.3 - Intra_8x8 prediction

The nine 8x8 luma prediction modes use up to 25 neighboring pixels:
    M  A0 A1 A2 A3 A4 A5 A6 A7  B0 B1 B2 B3 B4 B5 B6 B7
    I0 [                                              ]
    I1 [                                              ]
    I2 [                                              ]
    I3 [              8x8 block                       ]
    I4 [                                              ]
    I5 [                                              ]
    I6 [                                              ]
    I7 [                                              ]

Neighbor naming:
- top[0-7] = A0..A7 (8 pixels above)
- top_right[0-7] = B0..B7 (8 pixels above-right)
- left[0-7] = I0..I7 (8 pixels to left)
- top_left = M (corner pixel)

Modes are identical to 4x4 but scaled to 8x8:
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


class Intra8x8Mode(IntEnum):
    """Intra 8x8 prediction modes."""
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


def intra_8x8_vertical(top: np.ndarray) -> np.ndarray:
    """Mode 0: Vertical prediction.

    pred[y, x] = top[x]

    Args:
        top: Top neighbors (8 pixels)

    Returns:
        8x8 prediction block
    """
    pred = np.tile(top[:8].reshape(1, 8), (8, 1))
    return pred.astype(np.uint8)


def intra_8x8_horizontal(left: np.ndarray) -> np.ndarray:
    """Mode 1: Horizontal prediction.

    pred[y, x] = left[y]

    Args:
        left: Left neighbors (8 pixels)

    Returns:
        8x8 prediction block
    """
    pred = np.tile(left[:8].reshape(8, 1), (1, 8))
    return pred.astype(np.uint8)


def intra_8x8_dc(
    top: Optional[np.ndarray],
    left: Optional[np.ndarray],
    top_available: bool = True,
    left_available: bool = True,
) -> np.ndarray:
    """Mode 2: DC prediction.

    - Both available: (sum(top) + sum(left) + 8) >> 4
    - Only top: (sum(top) + 4) >> 3
    - Only left: (sum(left) + 4) >> 3
    - Neither: 128

    Args:
        top: Top neighbors (8 pixels) or None
        left: Left neighbors (8 pixels) or None
        top_available: Whether top is available
        left_available: Whether left is available

    Returns:
        8x8 prediction block
    """
    if top_available and top is not None and left_available and left is not None:
        # Both available
        dc_val = (int(np.sum(top[:8])) + int(np.sum(left[:8])) + 8) >> 4
    elif top_available and top is not None:
        # Only top
        dc_val = (int(np.sum(top[:8])) + 4) >> 3
    elif left_available and left is not None:
        # Only left
        dc_val = (int(np.sum(left[:8])) + 4) >> 3
    else:
        # Neither available
        dc_val = DEFAULT_PIXEL_VALUE

    dc_val = _clip(dc_val)
    pred = np.full((8, 8), dc_val, dtype=np.uint8)
    return pred


def intra_8x8_diagonal_down_left(
    top: np.ndarray,
    top_right: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mode 3: Diagonal Down-Left prediction.

    Extrapolates at 45° from top-right toward bottom-left.
    Uses 16 pixels: top[0-7] + top_right[0-7]

    Args:
        top: Top neighbors (8 pixels)
        top_right: Top-right neighbors (8 pixels) or None

    Returns:
        8x8 prediction block
    """
    # Build extended top array [A0..A7 B0..B7]
    t = np.zeros(16, dtype=np.int32)
    t[0:8] = top[:8]

    if top_right is not None:
        t[8:16] = top_right[:8]
    else:
        # Replicate top[7]
        t[8:16] = top[7]

    pred = np.zeros((8, 8), dtype=np.int32)

    for y in range(8):
        for x in range(8):
            if x == 7 and y == 7:
                # Special case: pred[7,7] = (t[14] + 3*t[15] + 2) >> 2
                pred[y, x] = (t[14] + 3 * t[15] + 2) >> 2
            else:
                idx = x + y
                pred[y, x] = _avg3(t[idx], t[idx + 1], t[idx + 2])

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_8x8_diagonal_down_right(
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
) -> np.ndarray:
    """Mode 4: Diagonal Down-Right prediction.

    Extrapolates at 45° from top-left toward bottom-right.

    Args:
        top: Top neighbors (8 pixels)
        left: Left neighbors (8 pixels)
        top_left: Corner pixel M

    Returns:
        8x8 prediction block
    """
    M = int(top_left)
    t = [int(top[i]) for i in range(8)]
    l = [int(left[i]) for i in range(8)]

    pred = np.zeros((8, 8), dtype=np.int32)

    for y in range(8):
        for x in range(8):
            if x > y:
                # Above diagonal: use top neighbors
                idx = x - y - 1
                if idx == 0:
                    pred[y, x] = _avg3(M, t[0], t[1])
                elif idx < 7:
                    pred[y, x] = _avg3(t[idx - 1], t[idx], t[idx + 1])
                else:
                    pred[y, x] = _avg3(t[6], t[7], t[7])
            elif x < y:
                # Below diagonal: use left neighbors
                idx = y - x - 1
                if idx == 0:
                    pred[y, x] = _avg3(M, l[0], l[1])
                elif idx < 7:
                    pred[y, x] = _avg3(l[idx - 1], l[idx], l[idx + 1])
                else:
                    pred[y, x] = _avg3(l[6], l[7], l[7])
            else:
                # On diagonal (x == y)
                pred[y, x] = _avg3(l[0], M, t[0])

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_8x8_vertical_right(
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
) -> np.ndarray:
    """Mode 5: Vertical-Right prediction.

    Extrapolates at 26.6° to the right of vertical.

    Args:
        top: Top neighbors (8 pixels)
        left: Left neighbors (8 pixels)
        top_left: Corner pixel M

    Returns:
        8x8 prediction block
    """
    M = int(top_left)
    t = [int(top[i]) for i in range(8)]
    l = [int(left[i]) for i in range(8)]

    pred = np.zeros((8, 8), dtype=np.int32)

    for y in range(8):
        for x in range(8):
            zVR = 2 * x - y
            if zVR >= 0:
                if zVR % 2 == 0:
                    idx = x - (y >> 1)
                    if idx == 0:
                        pred[y, x] = _avg2(M, t[0])
                    elif idx <= 7:
                        pred[y, x] = _avg2(t[idx - 1], t[idx])
                    else:
                        pred[y, x] = t[7]
                else:
                    idx = x - (y >> 1)
                    if idx == 0:
                        pred[y, x] = _avg3(l[0], M, t[0])
                    elif idx == 1:
                        pred[y, x] = _avg3(M, t[0], t[1])
                    elif idx <= 7:
                        pred[y, x] = _avg3(t[idx - 2], t[idx - 1], t[idx])
                    else:
                        pred[y, x] = t[7]
            else:
                # zVR < 0
                idx = y - 2 * x - 1
                if idx % 2 == 0:
                    i = idx >> 1
                    if i == 0:
                        pred[y, x] = _avg2(M, l[0])
                    elif i < 7:
                        pred[y, x] = _avg2(l[i - 1], l[i])
                    else:
                        pred[y, x] = l[7]
                else:
                    i = idx >> 1
                    if i == 0:
                        pred[y, x] = _avg3(t[0], M, l[0])
                    elif i == 1:
                        pred[y, x] = _avg3(M, l[0], l[1])
                    elif i < 7:
                        pred[y, x] = _avg3(l[i - 2], l[i - 1], l[i])
                    else:
                        pred[y, x] = l[7]

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_8x8_horizontal_down(
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
) -> np.ndarray:
    """Mode 6: Horizontal-Down prediction.

    Extrapolates at 26.6° below horizontal.

    Args:
        top: Top neighbors (8 pixels)
        left: Left neighbors (8 pixels)
        top_left: Corner pixel M

    Returns:
        8x8 prediction block
    """
    M = int(top_left)
    t = [int(top[i]) for i in range(8)]
    l = [int(left[i]) for i in range(8)]

    pred = np.zeros((8, 8), dtype=np.int32)

    for y in range(8):
        for x in range(8):
            zHD = 2 * y - x
            if zHD >= 0:
                if zHD % 2 == 0:
                    idx = y - (x >> 1)
                    if idx == 0:
                        pred[y, x] = _avg2(M, l[0])
                    elif idx <= 7:
                        pred[y, x] = _avg2(l[idx - 1], l[idx])
                    else:
                        pred[y, x] = l[7]
                else:
                    idx = y - (x >> 1)
                    if idx == 0:
                        pred[y, x] = _avg3(t[0], M, l[0])
                    elif idx == 1:
                        pred[y, x] = _avg3(M, l[0], l[1])
                    elif idx <= 7:
                        pred[y, x] = _avg3(l[idx - 2], l[idx - 1], l[idx])
                    else:
                        pred[y, x] = l[7]
            else:
                # zHD < 0
                idx = x - 2 * y - 1
                if idx % 2 == 0:
                    i = idx >> 1
                    if i == 0:
                        pred[y, x] = _avg2(M, t[0])
                    elif i < 7:
                        pred[y, x] = _avg2(t[i - 1], t[i])
                    else:
                        pred[y, x] = t[7]
                else:
                    i = idx >> 1
                    if i == 0:
                        pred[y, x] = _avg3(l[0], M, t[0])
                    elif i == 1:
                        pred[y, x] = _avg3(M, t[0], t[1])
                    elif i < 7:
                        pred[y, x] = _avg3(t[i - 2], t[i - 1], t[i])
                    else:
                        pred[y, x] = t[7]

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_8x8_vertical_left(
    top: np.ndarray,
    top_right: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mode 7: Vertical-Left prediction.

    Extrapolates at 26.6° to the left of vertical.

    Args:
        top: Top neighbors (8 pixels)
        top_right: Top-right neighbors (8 pixels) or None

    Returns:
        8x8 prediction block
    """
    # Build extended top array
    t = np.zeros(16, dtype=np.int32)
    t[0:8] = top[:8]

    if top_right is not None:
        t[8:16] = top_right[:8]
    else:
        t[8:16] = top[7]

    pred = np.zeros((8, 8), dtype=np.int32)

    for y in range(8):
        for x in range(8):
            if y % 2 == 0:
                idx = x + (y >> 1)
                pred[y, x] = _avg2(t[idx], t[idx + 1])
            else:
                idx = x + (y >> 1)
                pred[y, x] = _avg3(t[idx], t[idx + 1], t[idx + 2])

    return np.clip(pred, 0, 255).astype(np.uint8)


def intra_8x8_horizontal_up(left: np.ndarray) -> np.ndarray:
    """Mode 8: Horizontal-Up prediction.

    Extrapolates at 26.6° above horizontal.

    Args:
        left: Left neighbors (8 pixels)

    Returns:
        8x8 prediction block
    """
    l = [int(left[i]) for i in range(8)]

    pred = np.zeros((8, 8), dtype=np.int32)

    for y in range(8):
        for x in range(8):
            zHU = y + 2 * x
            if zHU < 13:
                if zHU % 2 == 0:
                    idx = y + (x >> 0)  # This is just y + x for even
                    # Actually: zHU = y + 2*x, so if even:
                    idx = (zHU >> 1)
                    if idx < 7:
                        pred[y, x] = _avg2(l[idx], l[idx + 1])
                    else:
                        pred[y, x] = l[7]
                else:
                    idx = (zHU >> 1)
                    if idx < 6:
                        pred[y, x] = _avg3(l[idx], l[idx + 1], l[idx + 2])
                    else:
                        pred[y, x] = l[7]
            elif zHU == 13:
                pred[y, x] = _avg3(l[6], l[7], l[7])
            else:
                # zHU >= 14
                pred[y, x] = l[7]

    return np.clip(pred, 0, 255).astype(np.uint8)


def predict_intra_8x8(
    mode: Intra8x8Mode,
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
    top_right: Optional[np.ndarray] = None,
    top_available: bool = True,
    left_available: bool = True,
) -> np.ndarray:
    """Dispatch to appropriate 8x8 intra prediction mode.

    Args:
        mode: Prediction mode (0-8)
        top: Top neighbors (8 pixels)
        left: Left neighbors (8 pixels)
        top_left: Corner pixel
        top_right: Top-right neighbors (8 pixels) or None
        top_available: Whether top neighbors are available
        left_available: Whether left neighbors are available

    Returns:
        8x8 prediction block
    """
    if mode == Intra8x8Mode.VERTICAL:
        return intra_8x8_vertical(top)
    elif mode == Intra8x8Mode.HORIZONTAL:
        return intra_8x8_horizontal(left)
    elif mode == Intra8x8Mode.DC:
        return intra_8x8_dc(top, left, top_available, left_available)
    elif mode == Intra8x8Mode.DIAGONAL_DOWN_LEFT:
        return intra_8x8_diagonal_down_left(top, top_right)
    elif mode == Intra8x8Mode.DIAGONAL_DOWN_RIGHT:
        return intra_8x8_diagonal_down_right(top, left, top_left)
    elif mode == Intra8x8Mode.VERTICAL_RIGHT:
        return intra_8x8_vertical_right(top, left, top_left)
    elif mode == Intra8x8Mode.HORIZONTAL_DOWN:
        return intra_8x8_horizontal_down(top, left, top_left)
    elif mode == Intra8x8Mode.VERTICAL_LEFT:
        return intra_8x8_vertical_left(top, top_right)
    elif mode == Intra8x8Mode.HORIZONTAL_UP:
        return intra_8x8_horizontal_up(left)
    else:
        raise ValueError(f"Unknown intra 8x8 mode: {mode}")
