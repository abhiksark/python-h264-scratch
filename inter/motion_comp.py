# h264/inter/motion_comp.py
"""Motion compensation for inter prediction.

Extracts prediction blocks from reference frames at motion vector positions.
Supports both integer and fractional (sub-pixel) positions.

H.264 Spec Reference: Section 8.4.2 - Fractional sample interpolation
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def clip_mv_to_frame(
    x: int,
    y: int,
    frame_width: int,
    frame_height: int
) -> Tuple[int, int]:
    """Clip motion vector position to frame boundaries.

    Args:
        x: Horizontal position (can be negative or past frame edge)
        y: Vertical position (can be negative or past frame edge)
        frame_width: Width of reference frame
        frame_height: Height of reference frame

    Returns:
        Tuple of (clipped_x, clipped_y) within [0, frame_size-1]
    """
    clipped_x = max(0, min(x, frame_width - 1))
    clipped_y = max(0, min(y, frame_height - 1))
    return clipped_x, clipped_y


def get_block_integer(
    ref_frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """Extract a block from reference frame at integer position.

    Handles boundary conditions by replicating edge pixels when the
    requested block extends outside the frame.

    Args:
        ref_frame: Reference frame (height x width), uint8
        x: Horizontal position of block top-left corner
        y: Vertical position of block top-left corner
        width: Block width
        height: Block height

    Returns:
        Block of size (height, width), uint8, as a copy

    H.264 Spec: Section 8.4.2.1 - Uses edge pixel replication
    """
    frame_height, frame_width = ref_frame.shape

    # Allocate output block
    block = np.zeros((height, width), dtype=np.uint8)

    for by in range(height):
        for bx in range(width):
            # Source position in reference frame
            src_x = x + bx
            src_y = y + by

            # Clip to frame boundaries (edge replication)
            src_x = max(0, min(src_x, frame_width - 1))
            src_y = max(0, min(src_y, frame_height - 1))

            block[by, bx] = ref_frame[src_y, src_x]

    return block


def _get_sample(ref_frame: np.ndarray, x: int, y: int) -> int:
    """Get single sample with edge replication.

    Args:
        ref_frame: Reference frame
        x: Horizontal position
        y: Vertical position

    Returns:
        Sample value at (x, y), clipped to frame boundaries
    """
    h, w = ref_frame.shape
    cx = max(0, min(x, w - 1))
    cy = max(0, min(y, h - 1))
    return int(ref_frame[cy, cx])


def interpolate_half_h(
    ref_frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """Interpolate half-pixel positions horizontally using 6-tap filter.

    Computes samples at position (x+0.5, y) for each pixel in block.
    Uses H.264 6-tap filter: [1, -5, 20, 20, -5, 1] / 32

    Args:
        ref_frame: Reference frame (height x width), uint8
        x: Integer horizontal position
        y: Integer vertical position
        width: Output block width
        height: Output block height

    Returns:
        Interpolated block of size (height, width), uint8

    H.264 Spec: Section 8.4.2.2.1 - Luma sample interpolation process
    """
    block = np.zeros((height, width), dtype=np.uint8)

    for by in range(height):
        for bx in range(width):
            px = x + bx
            py = y + by

            # 6-tap filter: samples at x-2, x-1, x, x+1, x+2, x+3
            a = _get_sample(ref_frame, px - 2, py)
            b = _get_sample(ref_frame, px - 1, py)
            c = _get_sample(ref_frame, px, py)
            d = _get_sample(ref_frame, px + 1, py)
            e = _get_sample(ref_frame, px + 2, py)
            f = _get_sample(ref_frame, px + 3, py)

            # Apply filter: (a - 5b + 20c + 20d - 5e + f + 16) >> 5
            result = a - 5 * b + 20 * c + 20 * d - 5 * e + f
            result = (result + 16) >> 5

            # Clip to [0, 255]
            block[by, bx] = max(0, min(255, result))

    return block


def interpolate_half_v(
    ref_frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """Interpolate half-pixel positions vertically using 6-tap filter.

    Computes samples at position (x, y+0.5) for each pixel in block.
    Uses H.264 6-tap filter: [1, -5, 20, 20, -5, 1] / 32

    Args:
        ref_frame: Reference frame (height x width), uint8
        x: Integer horizontal position
        y: Integer vertical position
        width: Output block width
        height: Output block height

    Returns:
        Interpolated block of size (height, width), uint8

    H.264 Spec: Section 8.4.2.2.1 - Luma sample interpolation process
    """
    block = np.zeros((height, width), dtype=np.uint8)

    for by in range(height):
        for bx in range(width):
            px = x + bx
            py = y + by

            # 6-tap filter: samples at y-2, y-1, y, y+1, y+2, y+3
            a = _get_sample(ref_frame, px, py - 2)
            b = _get_sample(ref_frame, px, py - 1)
            c = _get_sample(ref_frame, px, py)
            d = _get_sample(ref_frame, px, py + 1)
            e = _get_sample(ref_frame, px, py + 2)
            f = _get_sample(ref_frame, px, py + 3)

            # Apply filter: (a - 5b + 20c + 20d - 5e + f + 16) >> 5
            result = a - 5 * b + 20 * c + 20 * d - 5 * e + f
            result = (result + 16) >> 5

            # Clip to [0, 255]
            block[by, bx] = max(0, min(255, result))

    return block


def interpolate_half_hv(
    ref_frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """Interpolate half-pixel diagonal position using two-pass 6-tap filter.

    Computes samples at position (x+0.5, y+0.5). Per H.264 spec, this is done
    by first computing horizontal half-pixels, then applying vertical filter.

    Args:
        ref_frame: Reference frame (height x width), uint8
        x: Integer horizontal position
        y: Integer vertical position
        width: Output block width
        height: Output block height

    Returns:
        Interpolated block of size (height, width), uint8

    H.264 Spec: Section 8.4.2.2.1 - Position 'j' interpolation
    """
    # Need extra rows for vertical filter (3 above, 3 below)
    # First compute horizontal half-pixels for extended area
    h_height = height + 5  # y-2 to y+height+2
    h_samples = np.zeros((h_height, width), dtype=np.int32)

    for by in range(h_height):
        for bx in range(width):
            px = x + bx
            py = y + by - 2  # Start 2 rows above

            # Horizontal 6-tap filter
            a = _get_sample(ref_frame, px - 2, py)
            b = _get_sample(ref_frame, px - 1, py)
            c = _get_sample(ref_frame, px, py)
            d = _get_sample(ref_frame, px + 1, py)
            e = _get_sample(ref_frame, px + 2, py)
            f = _get_sample(ref_frame, px + 3, py)

            # Don't normalize yet - keep full precision for second pass
            h_samples[by, bx] = a - 5 * b + 20 * c + 20 * d - 5 * e + f

    # Now apply vertical filter to horizontal half-pixel values
    block = np.zeros((height, width), dtype=np.uint8)

    for by in range(height):
        for bx in range(width):
            # Vertical filter on h_samples (indices 0-5 map to y-2 to y+3)
            a = h_samples[by, bx]          # y - 2
            b = h_samples[by + 1, bx]      # y - 1
            c = h_samples[by + 2, bx]      # y
            d = h_samples[by + 3, bx]      # y + 1
            e = h_samples[by + 4, bx]      # y + 2
            f = h_samples[by + 5, bx] if by + 5 < h_height else h_samples[h_height - 1, bx]

            # Apply filter with proper rounding for two-stage
            # First stage didn't normalize, so we need (sum + 512) >> 10
            result = a - 5 * b + 20 * c + 20 * d - 5 * e + f
            result = (result + 512) >> 10

            # Clip to [0, 255]
            block[by, bx] = max(0, min(255, result))

    return block


def get_luma_block_fractional(
    ref_frame: np.ndarray,
    x: int,
    y: int,
    dx: int,
    dy: int,
    width: int,
    height: int
) -> np.ndarray:
    """Extract luma block at fractional (quarter-pixel) position.

    H.264 motion vectors have quarter-pixel precision. This function handles
    all 16 possible sub-pixel positions (dx, dy each in range 0-3).

    Position encoding:
        (0,0): integer position
        (2,0): half-pixel horizontal
        (0,2): half-pixel vertical
        (2,2): half-pixel diagonal
        (1,0): quarter between integer and half-h
        (3,0): quarter between half-h and next integer
        etc.

    Args:
        ref_frame: Reference frame (height x width), uint8
        x: Integer part of horizontal position
        y: Integer part of vertical position
        dx: Fractional horizontal offset (0-3, in quarter-pixels)
        dy: Fractional vertical offset (0-3, in quarter-pixels)
        width: Output block width
        height: Output block height

    Returns:
        Interpolated block of size (height, width), uint8

    H.264 Spec: Section 8.4.2.2.1 - Luma sample interpolation
    """
    # Integer position
    if dx == 0 and dy == 0:
        return get_block_integer(ref_frame, x, y, width, height)

    # Half-pixel horizontal
    if dx == 2 and dy == 0:
        return interpolate_half_h(ref_frame, x, y, width, height)

    # Half-pixel vertical
    if dx == 0 and dy == 2:
        return interpolate_half_v(ref_frame, x, y, width, height)

    # Half-pixel diagonal
    if dx == 2 and dy == 2:
        return interpolate_half_hv(ref_frame, x, y, width, height)

    # Quarter-pixel positions require averaging
    block = np.zeros((height, width), dtype=np.uint8)

    # Determine which samples to average based on dx, dy
    if dy == 0:
        # Horizontal quarter-pixels
        int_block = get_block_integer(ref_frame, x, y, width, height)
        half_h = interpolate_half_h(ref_frame, x, y, width, height)

        if dx == 1:
            # Average integer and half-h
            avg = (int_block.astype(np.int32) + half_h.astype(np.int32) + 1) >> 1
        else:  # dx == 3
            # Average half-h and next integer
            next_int = get_block_integer(ref_frame, x + 1, y, width, height)
            avg = (half_h.astype(np.int32) + next_int.astype(np.int32) + 1) >> 1
        block = avg.astype(np.uint8)

    elif dx == 0:
        # Vertical quarter-pixels
        int_block = get_block_integer(ref_frame, x, y, width, height)
        half_v = interpolate_half_v(ref_frame, x, y, width, height)

        if dy == 1:
            # Average integer and half-v
            avg = (int_block.astype(np.int32) + half_v.astype(np.int32) + 1) >> 1
        else:  # dy == 3
            # Average half-v and next integer
            next_int = get_block_integer(ref_frame, x, y + 1, width, height)
            avg = (half_v.astype(np.int32) + next_int.astype(np.int32) + 1) >> 1
        block = avg.astype(np.uint8)

    elif dx == 2:
        # Half-h with vertical quarter
        half_h = interpolate_half_h(ref_frame, x, y, width, height)
        half_hv = interpolate_half_hv(ref_frame, x, y, width, height)

        if dy == 1:
            avg = (half_h.astype(np.int32) + half_hv.astype(np.int32) + 1) >> 1
        else:  # dy == 3
            half_h_below = interpolate_half_h(ref_frame, x, y + 1, width, height)
            avg = (half_hv.astype(np.int32) + half_h_below.astype(np.int32) + 1) >> 1
        block = avg.astype(np.uint8)

    elif dy == 2:
        # Half-v with horizontal quarter
        half_v = interpolate_half_v(ref_frame, x, y, width, height)
        half_hv = interpolate_half_hv(ref_frame, x, y, width, height)

        if dx == 1:
            avg = (half_v.astype(np.int32) + half_hv.astype(np.int32) + 1) >> 1
        else:  # dx == 3
            half_v_right = interpolate_half_v(ref_frame, x + 1, y, width, height)
            avg = (half_hv.astype(np.int32) + half_v_right.astype(np.int32) + 1) >> 1
        block = avg.astype(np.uint8)

    else:
        # Quarter-pixel in both directions (dx=1 or 3, dy=1 or 3)
        # H.264 spec 8.4.2.2.1: average two nearest half-pel samples
        # e(1,1) = (b + h + 1) >> 1   b=half_h(x,y), h=half_v(x,y)
        # g(3,1) = (b + m + 1) >> 1   b=half_h(x,y), m=half_v(x+1,y)
        # p(1,3) = (h + s + 1) >> 1   h=half_v(x,y), s=half_h(x,y+1)
        # r(3,3) = (m + s + 1) >> 1   m=half_v(x+1,y), s=half_h(x,y+1)
        if dx == 1 and dy == 1:
            half_h = interpolate_half_h(ref_frame, x, y, width, height)
            half_v = interpolate_half_v(ref_frame, x, y, width, height)
            avg = (half_h.astype(np.int32) + half_v.astype(np.int32) + 1) >> 1
        elif dx == 3 and dy == 1:
            half_h = interpolate_half_h(ref_frame, x, y, width, height)
            half_v = interpolate_half_v(ref_frame, x + 1, y, width, height)
            avg = (half_h.astype(np.int32) + half_v.astype(np.int32) + 1) >> 1
        elif dx == 1 and dy == 3:
            half_v = interpolate_half_v(ref_frame, x, y, width, height)
            half_h = interpolate_half_h(ref_frame, x, y + 1, width, height)
            avg = (half_v.astype(np.int32) + half_h.astype(np.int32) + 1) >> 1
        else:  # dx == 3 and dy == 3
            half_v = interpolate_half_v(ref_frame, x + 1, y, width, height)
            half_h = interpolate_half_h(ref_frame, x, y + 1, width, height)
            avg = (half_v.astype(np.int32) + half_h.astype(np.int32) + 1) >> 1
        block = avg.astype(np.uint8)

    return block


def get_chroma_block_fractional(
    ref_frame: np.ndarray,
    x: int,
    y: int,
    dx: int,
    dy: int,
    width: int,
    height: int
) -> np.ndarray:
    """Extract chroma block at fractional (eighth-pixel) position.

    H.264 chroma motion compensation uses bilinear interpolation at
    1/8-pixel precision. The fractional position is specified by dx and dy
    in the range 0-7.

    Formula (per H.264 spec section 8.4.2.2.2):
        ((8-dx)*(8-dy)*A + dx*(8-dy)*B + (8-dx)*dy*C + dx*dy*D + 32) >> 6

    Where A, B, C, D are the four surrounding integer samples:
        A = (x, y)      B = (x+1, y)
        C = (x, y+1)    D = (x+1, y+1)

    Args:
        ref_frame: Reference chroma frame (height x width), uint8
        x: Integer part of horizontal position
        y: Integer part of vertical position
        dx: Fractional horizontal offset (0-7, in eighth-pixels)
        dy: Fractional vertical offset (0-7, in eighth-pixels)
        width: Output block width
        height: Output block height

    Returns:
        Interpolated block of size (height, width), uint8

    H.264 Spec: Section 8.4.2.2.2 - Chroma sample interpolation
    """
    # Integer position - direct copy
    if dx == 0 and dy == 0:
        return get_block_integer(ref_frame, x, y, width, height)

    block = np.zeros((height, width), dtype=np.uint8)

    # Precompute weights
    wx0 = 8 - dx  # Weight for left column
    wx1 = dx      # Weight for right column
    wy0 = 8 - dy  # Weight for top row
    wy1 = dy      # Weight for bottom row

    for by in range(height):
        for bx in range(width):
            px = x + bx
            py = y + by

            # Get four surrounding integer samples
            a = _get_sample(ref_frame, px, py)          # Top-left
            b = _get_sample(ref_frame, px + 1, py)      # Top-right
            c = _get_sample(ref_frame, px, py + 1)      # Bottom-left
            d = _get_sample(ref_frame, px + 1, py + 1)  # Bottom-right

            # Bilinear interpolation
            result = (wx0 * wy0 * a + wx1 * wy0 * b +
                      wx0 * wy1 * c + wx1 * wy1 * d + 32) >> 6

            block[by, bx] = result

    return block
