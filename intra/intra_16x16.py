# h264/intra/intra_16x16.py
"""Intra 16x16 prediction modes for H.264 I-frames.

Intra prediction generates a prediction block based on neighboring pixels
that have already been decoded. The residual (prediction error) is then
added to get the final reconstructed pixels.

H.264 Spec Reference: Section 8.3.3 - Intra_16x16 prediction

The four 16x16 luma prediction modes are:
- Mode 0 (Intra_16x16_Vertical): Extrapolate from top neighbors
- Mode 1 (Intra_16x16_Horizontal): Extrapolate from left neighbors
- Mode 2 (Intra_16x16_DC): Average of available neighbors
- Mode 3 (Intra_16x16_Plane): Bi-linear interpolation
"""

import logging
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Intra16x16Mode(IntEnum):
    """Intra 16x16 prediction modes."""
    VERTICAL = 0
    HORIZONTAL = 1
    DC = 2
    PLANE = 3


# Default value when neighbors are not available
DEFAULT_PIXEL_VALUE = 128


def intra_16x16_vertical(
    top: np.ndarray,
    top_available: bool = True
) -> np.ndarray:
    """Intra_16x16_Vertical prediction (mode 0).

    Each column is filled with the corresponding top neighbor pixel.

    Args:
        top: Top neighbor pixels (16,), uint8
        top_available: Whether top neighbors are available

    Returns:
        16x16 prediction block (uint8)

    H.264 Spec: Section 8.3.3.1
    pred[y, x] = top[x] for all y
    """
    if not top_available:
        logger.warning("Vertical mode requires top neighbors, using default")
        return np.full((16, 16), DEFAULT_PIXEL_VALUE, dtype=np.uint8)

    if top.shape[0] < 16:
        raise ValueError(f"Top neighbors must have at least 16 pixels, got {top.shape}")

    logger.debug(f"Intra_16x16_Vertical: top={top[:16]}")

    # Repeat top row 16 times
    pred = np.tile(top[:16].reshape(1, 16), (16, 1))

    return pred.astype(np.uint8)


def intra_16x16_horizontal(
    left: np.ndarray,
    left_available: bool = True
) -> np.ndarray:
    """Intra_16x16_Horizontal prediction (mode 1).

    Each row is filled with the corresponding left neighbor pixel.

    Args:
        left: Left neighbor pixels (16,), uint8
        left_available: Whether left neighbors are available

    Returns:
        16x16 prediction block (uint8)

    H.264 Spec: Section 8.3.3.2
    pred[y, x] = left[y] for all x
    """
    if not left_available:
        logger.warning("Horizontal mode requires left neighbors, using default")
        return np.full((16, 16), DEFAULT_PIXEL_VALUE, dtype=np.uint8)

    if left.shape[0] < 16:
        raise ValueError(f"Left neighbors must have at least 16 pixels, got {left.shape}")

    logger.debug(f"Intra_16x16_Horizontal: left={left[:16]}")

    # Repeat left column 16 times
    pred = np.tile(left[:16].reshape(16, 1), (1, 16))

    return pred.astype(np.uint8)


def intra_16x16_dc(
    top: Optional[np.ndarray],
    left: Optional[np.ndarray],
    top_available: bool = True,
    left_available: bool = True
) -> np.ndarray:
    """Intra_16x16_DC prediction (mode 2).

    Fill block with average of available neighbors.

    Args:
        top: Top neighbor pixels (16,) or None
        left: Left neighbor pixels (16,) or None
        top_available: Whether top neighbors are available
        left_available: Whether left neighbors are available

    Returns:
        16x16 prediction block (uint8)

    H.264 Spec: Section 8.3.3.3

    DC value calculation:
    - Both available: (sum(top) + sum(left) + 16) >> 5
    - Only top: (sum(top) + 8) >> 4
    - Only left: (sum(left) + 8) >> 4
    - Neither: 128
    """
    if top_available and left_available:
        # Both available: average of all 32 neighbors
        dc_sum = int(np.sum(top[:16])) + int(np.sum(left[:16]))
        dc = (dc_sum + 16) >> 5  # Divide by 32 with rounding
        logger.debug(f"DC (both): sum={dc_sum}, dc={dc}")

    elif top_available:
        # Only top available
        dc_sum = int(np.sum(top[:16]))
        dc = (dc_sum + 8) >> 4  # Divide by 16 with rounding
        logger.debug(f"DC (top only): sum={dc_sum}, dc={dc}")

    elif left_available:
        # Only left available
        dc_sum = int(np.sum(left[:16]))
        dc = (dc_sum + 8) >> 4  # Divide by 16 with rounding
        logger.debug(f"DC (left only): sum={dc_sum}, dc={dc}")

    else:
        # Neither available (top-left macroblock)
        dc = DEFAULT_PIXEL_VALUE
        logger.debug(f"DC (neither): dc={dc}")

    # Fill entire block with DC value
    pred = np.full((16, 16), dc, dtype=np.uint8)

    return pred


def intra_16x16_plane(
    top: np.ndarray,
    left: np.ndarray,
    top_left: int,
    top_available: bool = True,
    left_available: bool = True
) -> np.ndarray:
    """Intra_16x16_Plane prediction (mode 3).

    Bi-linear interpolation based on neighbors. Most complex mode.

    Args:
        top: Top neighbor pixels (16,), uint8
        left: Left neighbor pixels (16,), uint8
        top_left: Top-left corner pixel
        top_available: Whether top neighbors are available
        left_available: Whether left neighbors are available

    Returns:
        16x16 prediction block (uint8)

    H.264 Spec: Section 8.3.3.4

    The plane prediction formula is:
    pred[y,x] = Clip1Y((a + b*(x-7) + c*(y-7) + 16) >> 5)

    where:
    - a = 16 * (top[15] + left[15])
    - b = (5*H + 32) >> 6
    - c = (5*V + 32) >> 6
    - H = sum((x+1) * (top[8+x] - top[6-x])) for x in [0,7]
    - V = sum((y+1) * (left[8+y] - left[6-y])) for y in [0,7]
    """
    if not (top_available and left_available):
        logger.warning("Plane mode requires both neighbors, falling back to DC")
        return intra_16x16_dc(top, left, top_available, left_available)

    logger.debug("Intra_16x16_Plane prediction")

    # Convert to int32 for computation
    top_i = top[:16].astype(np.int32)
    left_i = left[:16].astype(np.int32)
    top_left_i = int(top_left)

    # Calculate H (horizontal gradient)
    H = 0
    for x in range(8):
        # top[8+x] when x=0 is top[8], top[6-x] when x=0 is top[6]
        # For x=7: top[15] and top[-1] which is top_left
        if x < 7:
            H += (x + 1) * (top_i[8 + x] - top_i[6 - x])
        else:  # x == 7
            H += 8 * (top_i[15] - top_left_i)

    # Calculate V (vertical gradient)
    V = 0
    for y in range(8):
        if y < 7:
            V += (y + 1) * (left_i[8 + y] - left_i[6 - y])
        else:  # y == 7
            V += 8 * (left_i[15] - top_left_i)

    # Calculate coefficients
    a = 16 * (int(top_i[15]) + int(left_i[15]))
    b = (5 * H + 32) >> 6
    c = (5 * V + 32) >> 6

    logger.debug(f"Plane: a={a}, b={b}, c={c}, H={H}, V={V}")

    # Generate prediction
    pred = np.zeros((16, 16), dtype=np.int32)
    for y in range(16):
        for x in range(16):
            pred[y, x] = (a + b * (x - 7) + c * (y - 7) + 16) >> 5

    # Clip to [0, 255]
    pred = np.clip(pred, 0, 255).astype(np.uint8)

    return pred


def predict_intra_16x16(
    mode: int,
    top: Optional[np.ndarray],
    left: Optional[np.ndarray],
    top_left: Optional[int] = None,
    top_available: bool = True,
    left_available: bool = True
) -> np.ndarray:
    """Generate 16x16 intra prediction block.

    Main entry point for 16x16 intra prediction.

    Args:
        mode: Prediction mode (0-3)
        top: Top neighbor pixels (16,) or None
        left: Left neighbor pixels (16,) or None
        top_left: Top-left corner pixel (for plane mode)
        top_available: Whether top neighbors exist
        left_available: Whether left neighbors exist

    Returns:
        16x16 prediction block (uint8)

    Raises:
        ValueError: If mode is invalid
    """
    logger.debug(f"Intra_16x16 prediction: mode={mode}")

    if mode == Intra16x16Mode.VERTICAL:
        return intra_16x16_vertical(top, top_available)

    elif mode == Intra16x16Mode.HORIZONTAL:
        return intra_16x16_horizontal(left, left_available)

    elif mode == Intra16x16Mode.DC:
        return intra_16x16_dc(top, left, top_available, left_available)

    elif mode == Intra16x16Mode.PLANE:
        if top_left is None and top_available and left_available:
            # Default top_left to top[-1] which would be left of top[0]
            top_left = int(left[0]) if left_available else DEFAULT_PIXEL_VALUE
        elif top_left is None:
            top_left = DEFAULT_PIXEL_VALUE
        return intra_16x16_plane(top, left, top_left, top_available, left_available)

    else:
        raise ValueError(f"Invalid Intra_16x16 mode: {mode}")


def get_neighbors_for_macroblock(
    frame: np.ndarray,
    mb_x: int,
    mb_y: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int], bool, bool]:
    """Extract neighbor pixels for a macroblock from the frame.

    Args:
        frame: Reconstructed frame so far (H, W)
        mb_x: Macroblock X position (in MB units)
        mb_y: Macroblock Y position (in MB units)

    Returns:
        Tuple of (top, left, top_left, top_available, left_available)
    """
    height, width = frame.shape
    pixel_x = mb_x * 16
    pixel_y = mb_y * 16

    top_available = mb_y > 0
    left_available = mb_x > 0

    top = None
    left = None
    top_left = None

    if top_available:
        # Row above the macroblock
        top = frame[pixel_y - 1, pixel_x:pixel_x + 16].copy()

    if left_available:
        # Column to the left of the macroblock
        left = frame[pixel_y:pixel_y + 16, pixel_x - 1].copy()

    if top_available and left_available:
        top_left = int(frame[pixel_y - 1, pixel_x - 1])

    return top, left, top_left, top_available, left_available


def validate_prediction_mode(
    mode: int,
    top_available: bool,
    left_available: bool
) -> bool:
    """Check if a prediction mode is valid given neighbor availability.

    Args:
        mode: Requested prediction mode
        top_available: Whether top neighbors exist
        left_available: Whether left neighbors exist

    Returns:
        True if mode is valid for the given neighbors
    """
    if mode == Intra16x16Mode.VERTICAL:
        return top_available
    elif mode == Intra16x16Mode.HORIZONTAL:
        return left_available
    elif mode == Intra16x16Mode.DC:
        return True  # DC always works (uses default if no neighbors)
    elif mode == Intra16x16Mode.PLANE:
        return top_available and left_available
    else:
        return False
