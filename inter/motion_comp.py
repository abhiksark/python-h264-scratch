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
