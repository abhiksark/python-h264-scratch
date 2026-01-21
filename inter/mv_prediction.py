# h264/inter/mv_prediction.py
"""Motion vector prediction for P-frames.

Implements spatial median prediction algorithm to derive MV predictors
from neighboring blocks.

H.264 Spec Reference: Section 8.4.1.3 - Derivation of motion vector predictors
"""

import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MVCache:
    """Cache for storing motion vectors of decoded macroblocks.

    Stores MVs at 4x4 block granularity for sub-macroblock partitions.
    Each macroblock has 16 4x4 blocks (4 columns x 4 rows).

    Coordinate system:
        - mb_x, mb_y: Macroblock position in frame
        - block_x, block_y: 4x4 block position within macroblock (0-3)
    """

    def __init__(self, width_in_mbs: int, height_in_mbs: int):
        """Initialize MV cache for frame.

        Args:
            width_in_mbs: Frame width in macroblocks
            height_in_mbs: Frame height in macroblocks
        """
        self.width_in_mbs = width_in_mbs
        self.height_in_mbs = height_in_mbs

        # Store MVs at 4x4 block granularity (16 blocks per MB)
        # Shape: (height_in_4x4, width_in_4x4, 2) for (mvx, mvy)
        width_in_4x4 = width_in_mbs * 4
        height_in_4x4 = height_in_mbs * 4
        self._mvs = np.zeros((height_in_4x4, width_in_4x4, 2), dtype=np.int32)

        # Track which blocks have been set (for availability check)
        self._available = np.zeros((height_in_4x4, width_in_4x4), dtype=bool)

    def _to_4x4_coords(
        self, mb_x: int, mb_y: int, block_x: int, block_y: int
    ) -> Tuple[int, int]:
        """Convert MB + block coords to absolute 4x4 block coords."""
        abs_x = mb_x * 4 + block_x
        abs_y = mb_y * 4 + block_y
        return abs_x, abs_y

    def is_available(
        self, mb_x: int, mb_y: int, block_x: int, block_y: int
    ) -> bool:
        """Check if block MV is available (in-frame and decoded).

        Args:
            mb_x: Macroblock X position
            mb_y: Macroblock Y position
            block_x: 4x4 block X within MB (0-3)
            block_y: 4x4 block Y within MB (0-3)

        Returns:
            True if MV is available for prediction
        """
        abs_x, abs_y = self._to_4x4_coords(mb_x, mb_y, block_x, block_y)

        # Check bounds
        if abs_x < 0 or abs_x >= self.width_in_mbs * 4:
            return False
        if abs_y < 0 or abs_y >= self.height_in_mbs * 4:
            return False

        return self._available[abs_y, abs_x]

    def get_mv(
        self, mb_x: int, mb_y: int, block_x: int, block_y: int
    ) -> Tuple[int, int]:
        """Get MV for a 4x4 block.

        Args:
            mb_x: Macroblock X position
            mb_y: Macroblock Y position
            block_x: 4x4 block X within MB (0-3)
            block_y: 4x4 block Y within MB (0-3)

        Returns:
            Tuple of (mvx, mvy). Returns (0, 0) if unavailable.
        """
        abs_x, abs_y = self._to_4x4_coords(mb_x, mb_y, block_x, block_y)

        # Out of bounds returns (0, 0)
        if abs_x < 0 or abs_x >= self.width_in_mbs * 4:
            return 0, 0
        if abs_y < 0 or abs_y >= self.height_in_mbs * 4:
            return 0, 0

        return int(self._mvs[abs_y, abs_x, 0]), int(self._mvs[abs_y, abs_x, 1])

    def set_mv(
        self,
        mb_x: int,
        mb_y: int,
        block_x: int,
        block_y: int,
        mvx: int,
        mvy: int
    ) -> None:
        """Set MV for a 4x4 block.

        Args:
            mb_x: Macroblock X position
            mb_y: Macroblock Y position
            block_x: 4x4 block X within MB (0-3)
            block_y: 4x4 block Y within MB (0-3)
            mvx: Horizontal motion vector component
            mvy: Vertical motion vector component
        """
        abs_x, abs_y = self._to_4x4_coords(mb_x, mb_y, block_x, block_y)

        self._mvs[abs_y, abs_x, 0] = mvx
        self._mvs[abs_y, abs_x, 1] = mvy
        self._available[abs_y, abs_x] = True

    def set_mv_16x16(self, mb_x: int, mb_y: int, mvx: int, mvy: int) -> None:
        """Set same MV for all 16 4x4 blocks in a macroblock.

        Used for P_L0_16x16 and P_Skip macroblocks.

        Args:
            mb_x: Macroblock X position
            mb_y: Macroblock Y position
            mvx: Horizontal motion vector
            mvy: Vertical motion vector
        """
        for by in range(4):
            for bx in range(4):
                self.set_mv(mb_x, mb_y, bx, by, mvx, mvy)

    def set_mv_16x8(
        self, mb_x: int, mb_y: int, partition: int, mvx: int, mvy: int
    ) -> None:
        """Set MV for 16x8 partition.

        Args:
            mb_x: Macroblock X position
            mb_y: Macroblock Y position
            partition: 0 for top, 1 for bottom
            mvx: Horizontal motion vector
            mvy: Vertical motion vector
        """
        start_y = 0 if partition == 0 else 2
        for by in range(start_y, start_y + 2):
            for bx in range(4):
                self.set_mv(mb_x, mb_y, bx, by, mvx, mvy)

    def set_mv_8x16(
        self, mb_x: int, mb_y: int, partition: int, mvx: int, mvy: int
    ) -> None:
        """Set MV for 8x16 partition.

        Args:
            mb_x: Macroblock X position
            mb_y: Macroblock Y position
            partition: 0 for left, 1 for right
            mvx: Horizontal motion vector
            mvy: Vertical motion vector
        """
        start_x = 0 if partition == 0 else 2
        for by in range(4):
            for bx in range(start_x, start_x + 2):
                self.set_mv(mb_x, mb_y, bx, by, mvx, mvy)

    def set_mv_8x8(
        self, mb_x: int, mb_y: int, sub_mb_idx: int, mvx: int, mvy: int
    ) -> None:
        """Set MV for 8x8 sub-macroblock partition.

        Args:
            mb_x: Macroblock X position
            mb_y: Macroblock Y position
            sub_mb_idx: Sub-macroblock index (0=TL, 1=TR, 2=BL, 3=BR)
            mvx: Horizontal motion vector
            mvy: Vertical motion vector
        """
        # Sub-MB layout:
        # 0 1
        # 2 3
        start_x = (sub_mb_idx % 2) * 2
        start_y = (sub_mb_idx // 2) * 2

        for by in range(start_y, start_y + 2):
            for bx in range(start_x, start_x + 2):
                self.set_mv(mb_x, mb_y, bx, by, mvx, mvy)


def _median(a: int, b: int, c: int) -> int:
    """Calculate median of three values."""
    return sorted([a, b, c])[1]


def _get_neighbor_mvs(
    cache: MVCache,
    mb_x: int,
    mb_y: int,
    block_x: int,
    block_y: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], bool, bool, bool]:
    """Get A, B, C/D neighbor MVs for prediction.

    A = left neighbor
    B = top neighbor
    C = top-right neighbor (or D = top-left if C unavailable)

    Args:
        cache: MV cache
        mb_x, mb_y: Current macroblock position
        block_x, block_y: Current 4x4 block within MB

    Returns:
        (mvA, mvB, mvC, a_avail, b_avail, c_avail)
    """
    # A: left neighbor
    if block_x > 0:
        # Within same MB
        a_avail = cache.is_available(mb_x, mb_y, block_x - 1, block_y)
        mvA = cache.get_mv(mb_x, mb_y, block_x - 1, block_y)
    else:
        # Previous MB
        a_avail = cache.is_available(mb_x - 1, mb_y, 3, block_y)
        mvA = cache.get_mv(mb_x - 1, mb_y, 3, block_y)

    # B: top neighbor
    if block_y > 0:
        # Within same MB
        b_avail = cache.is_available(mb_x, mb_y, block_x, block_y - 1)
        mvB = cache.get_mv(mb_x, mb_y, block_x, block_y - 1)
    else:
        # Previous row MB
        b_avail = cache.is_available(mb_x, mb_y - 1, block_x, 3)
        mvB = cache.get_mv(mb_x, mb_y - 1, block_x, 3)

    # C: top-right neighbor (preferred) or D: top-left
    # Try C first
    if block_y > 0:
        # Check within same MB
        if block_x < 3:
            c_avail = cache.is_available(mb_x, mb_y, block_x + 1, block_y - 1)
            mvC = cache.get_mv(mb_x, mb_y, block_x + 1, block_y - 1)
        else:
            # C is in next MB
            c_avail = cache.is_available(mb_x + 1, mb_y, 0, block_y - 1)
            mvC = cache.get_mv(mb_x + 1, mb_y, 0, block_y - 1)
    else:
        # Top row - C is in MB above
        if block_x < 3:
            c_avail = cache.is_available(mb_x, mb_y - 1, block_x + 1, 3)
            mvC = cache.get_mv(mb_x, mb_y - 1, block_x + 1, 3)
        else:
            # C is in top-right MB
            c_avail = cache.is_available(mb_x + 1, mb_y - 1, 0, 3)
            mvC = cache.get_mv(mb_x + 1, mb_y - 1, 0, 3)

    # If C unavailable, try D (top-left)
    if not c_avail:
        if block_y > 0:
            if block_x > 0:
                d_avail = cache.is_available(mb_x, mb_y, block_x - 1, block_y - 1)
                mvD = cache.get_mv(mb_x, mb_y, block_x - 1, block_y - 1)
            else:
                d_avail = cache.is_available(mb_x - 1, mb_y, 3, block_y - 1)
                mvD = cache.get_mv(mb_x - 1, mb_y, 3, block_y - 1)
        else:
            if block_x > 0:
                d_avail = cache.is_available(mb_x, mb_y - 1, block_x - 1, 3)
                mvD = cache.get_mv(mb_x, mb_y - 1, block_x - 1, 3)
            else:
                d_avail = cache.is_available(mb_x - 1, mb_y - 1, 3, 3)
                mvD = cache.get_mv(mb_x - 1, mb_y - 1, 3, 3)

        if d_avail:
            c_avail = True
            mvC = mvD

    return mvA, mvB, mvC, a_avail, b_avail, c_avail


def _predict_mv_median(
    cache: MVCache,
    mb_x: int,
    mb_y: int,
    block_x: int,
    block_y: int
) -> Tuple[int, int]:
    """Standard median prediction from A, B, C neighbors.

    Args:
        cache: MV cache
        mb_x, mb_y: Macroblock position
        block_x, block_y: 4x4 block position within MB

    Returns:
        Predicted (mvx, mvy)
    """
    mvA, mvB, mvC, a_avail, b_avail, c_avail = _get_neighbor_mvs(
        cache, mb_x, mb_y, block_x, block_y
    )

    # Count available neighbors
    avail_count = sum([a_avail, b_avail, c_avail])

    if avail_count == 0:
        # No neighbors available
        return 0, 0

    if avail_count == 1:
        # Only one available - use it directly
        if a_avail:
            return mvA
        if b_avail:
            return mvB
        return mvC

    # Two or three available - use median
    # For unavailable, use (0, 0)
    ax, ay = mvA if a_avail else (0, 0)
    bx, by = mvB if b_avail else (0, 0)
    cx, cy = mvC if c_avail else (0, 0)

    return _median(ax, bx, cx), _median(ay, by, cy)


def predict_mv_16x16(cache: MVCache, mb_x: int, mb_y: int) -> Tuple[int, int]:
    """Predict MV for 16x16 partition (P_L0_16x16).

    Uses standard median prediction from neighboring macroblocks.
    For 16x16, neighbors are at MB level:
        A = left MB (block 3, 0)
        B = top MB (block 0, 3)
        C = top-right MB (block 0, 3), or top-left if unavailable

    Args:
        cache: MV cache with decoded neighbor MVs
        mb_x: Current macroblock X position
        mb_y: Current macroblock Y position

    Returns:
        Predicted (mvx, mvy) for the macroblock

    H.264 Spec: Section 8.4.1.3.1
    """
    # A: left MB, rightmost column
    a_avail = cache.is_available(mb_x - 1, mb_y, 3, 0)
    mvA = cache.get_mv(mb_x - 1, mb_y, 3, 0)

    # B: top MB, bottom row
    b_avail = cache.is_available(mb_x, mb_y - 1, 0, 3)
    mvB = cache.get_mv(mb_x, mb_y - 1, 0, 3)

    # C: top-right MB, bottom-left block
    c_avail = cache.is_available(mb_x + 1, mb_y - 1, 0, 3)
    mvC = cache.get_mv(mb_x + 1, mb_y - 1, 0, 3)

    # If C unavailable, try D (top-left MB)
    if not c_avail:
        d_avail = cache.is_available(mb_x - 1, mb_y - 1, 3, 3)
        if d_avail:
            c_avail = True
            mvC = cache.get_mv(mb_x - 1, mb_y - 1, 3, 3)

    # Count available neighbors
    avail_count = sum([a_avail, b_avail, c_avail])

    if avail_count == 0:
        return 0, 0

    if avail_count == 1:
        if a_avail:
            return mvA
        if b_avail:
            return mvB
        return mvC

    # Median of three
    ax, ay = mvA if a_avail else (0, 0)
    bx, by = mvB if b_avail else (0, 0)
    cx, cy = mvC if c_avail else (0, 0)

    return _median(ax, bx, cx), _median(ay, by, cy)


def predict_mv_16x8(
    cache: MVCache, mb_x: int, mb_y: int, partition: int
) -> Tuple[int, int]:
    """Predict MV for 16x8 partition.

    partition 0 (top): Standard prediction like 16x16
    partition 1 (bottom): Use top partition as B, standard A/C

    Args:
        cache: MV cache
        mb_x, mb_y: Macroblock position
        partition: 0 for top, 1 for bottom

    Returns:
        Predicted (mvx, mvy)

    H.264 Spec: Section 8.4.1.3.1
    """
    if partition == 0:
        # Top partition - same as 16x16
        # A: left MB, rightmost column
        a_avail = cache.is_available(mb_x - 1, mb_y, 3, 0)
        mvA = cache.get_mv(mb_x - 1, mb_y, 3, 0)

        # B: top MB, bottom row
        b_avail = cache.is_available(mb_x, mb_y - 1, 0, 3)
        mvB = cache.get_mv(mb_x, mb_y - 1, 0, 3)

        # C: top-right MB
        c_avail = cache.is_available(mb_x + 1, mb_y - 1, 0, 3)
        mvC = cache.get_mv(mb_x + 1, mb_y - 1, 0, 3)

        if not c_avail:
            d_avail = cache.is_available(mb_x - 1, mb_y - 1, 3, 3)
            if d_avail:
                c_avail = True
                mvC = cache.get_mv(mb_x - 1, mb_y - 1, 3, 3)

        avail_count = sum([a_avail, b_avail, c_avail])
        if avail_count == 0:
            return 0, 0
        if avail_count == 1:
            if a_avail:
                return mvA
            if b_avail:
                return mvB
            return mvC

        ax, ay = mvA if a_avail else (0, 0)
        bx, by = mvB if b_avail else (0, 0)
        cx, cy = mvC if c_avail else (0, 0)
        return _median(ax, bx, cx), _median(ay, by, cy)
    else:
        # Bottom partition - B comes from top partition of same MB
        mvB = cache.get_mv(mb_x, mb_y, 0, 0)
        b_avail = cache.is_available(mb_x, mb_y, 0, 0)

        if b_avail:
            return mvB

        # Fall back to standard median
        return _predict_mv_median(cache, mb_x, mb_y, block_x=0, block_y=2)


def predict_mv_8x16(
    cache: MVCache, mb_x: int, mb_y: int, partition: int
) -> Tuple[int, int]:
    """Predict MV for 8x16 partition.

    partition 0 (left): Prefer A (left neighbor)
    partition 1 (right): Prefer C (top-right neighbor)

    Args:
        cache: MV cache
        mb_x, mb_y: Macroblock position
        partition: 0 for left, 1 for right

    Returns:
        Predicted (mvx, mvy)

    H.264 Spec: Section 8.4.1.3.1
    """
    if partition == 0:
        # Left partition - prefer A
        a_avail = cache.is_available(mb_x - 1, mb_y, 3, 0)
        if a_avail:
            return cache.get_mv(mb_x - 1, mb_y, 3, 0)

        # Fall back to median
        return _predict_mv_median(cache, mb_x, mb_y, block_x=0, block_y=0)
    else:
        # Right partition - prefer C (top-right of right half)
        # C for right partition is at block (2, 0), so top-right is...
        # Actually for right 8x16, C is the top-right neighbor MB's bottom-left
        c_avail = cache.is_available(mb_x + 1, mb_y - 1, 0, 3)
        if c_avail:
            return cache.get_mv(mb_x + 1, mb_y - 1, 0, 3)

        # Check if C within top of same frame
        c2_avail = cache.is_available(mb_x, mb_y - 1, 2, 3)
        if c2_avail:
            return cache.get_mv(mb_x, mb_y - 1, 2, 3)

        # Fall back to median
        return _predict_mv_median(cache, mb_x, mb_y, block_x=2, block_y=0)


def predict_mv_8x8(
    cache: MVCache, mb_x: int, mb_y: int, sub_mb_idx: int
) -> Tuple[int, int]:
    """Predict MV for 8x8 sub-macroblock partition.

    Sub-macroblock layout:
        0 1
        2 3

    For each 8x8, neighbors are determined by the partition edges,
    not individual 4x4 blocks.

    Args:
        cache: MV cache
        mb_x, mb_y: Macroblock position
        sub_mb_idx: Sub-macroblock index (0-3)

    Returns:
        Predicted (mvx, mvy)

    H.264 Spec: Section 8.4.1.3.1
    """
    # Block positions for each sub-MB
    # sub_mb 0: blocks (0,0)-(1,1), sub_mb 1: blocks (2,0)-(3,1)
    # sub_mb 2: blocks (0,2)-(1,3), sub_mb 3: blocks (2,2)-(3,3)
    block_x = (sub_mb_idx % 2) * 2  # 0 or 2
    block_y = (sub_mb_idx // 2) * 2  # 0 or 2

    # Determine neighbors based on sub-MB position
    if sub_mb_idx == 0:
        # Top-left 8x8: A from left MB, B from top MB, C from top-right MB
        a_avail = cache.is_available(mb_x - 1, mb_y, 3, 0)
        mvA = cache.get_mv(mb_x - 1, mb_y, 3, 0)

        b_avail = cache.is_available(mb_x, mb_y - 1, 0, 3)
        mvB = cache.get_mv(mb_x, mb_y - 1, 0, 3)

        # C: top-right of this 8x8 partition (block 2,3 of top MB or beyond)
        c_avail = cache.is_available(mb_x, mb_y - 1, 2, 3)
        mvC = cache.get_mv(mb_x, mb_y - 1, 2, 3)
        if not c_avail:
            # Try top-left MB
            c_avail = cache.is_available(mb_x - 1, mb_y - 1, 3, 3)
            mvC = cache.get_mv(mb_x - 1, mb_y - 1, 3, 3)

    elif sub_mb_idx == 1:
        # Top-right 8x8: A from sub-MB 0, B from top MB, C from top-right MB
        a_avail = cache.is_available(mb_x, mb_y, 1, 0)
        mvA = cache.get_mv(mb_x, mb_y, 1, 0)

        b_avail = cache.is_available(mb_x, mb_y - 1, 2, 3)
        mvB = cache.get_mv(mb_x, mb_y - 1, 2, 3)

        # C: top-right of this partition → next MB
        c_avail = cache.is_available(mb_x + 1, mb_y - 1, 0, 3)
        mvC = cache.get_mv(mb_x + 1, mb_y - 1, 0, 3)
        if not c_avail:
            # Try top of sub-MB 0
            c_avail = cache.is_available(mb_x, mb_y - 1, 0, 3)
            mvC = cache.get_mv(mb_x, mb_y - 1, 0, 3)

    elif sub_mb_idx == 2:
        # Bottom-left 8x8: A from left MB, B from sub-MB 0, C from sub-MB 1
        a_avail = cache.is_available(mb_x - 1, mb_y, 3, 2)
        mvA = cache.get_mv(mb_x - 1, mb_y, 3, 2)

        b_avail = cache.is_available(mb_x, mb_y, 0, 1)
        mvB = cache.get_mv(mb_x, mb_y, 0, 1)

        c_avail = cache.is_available(mb_x, mb_y, 2, 1)
        mvC = cache.get_mv(mb_x, mb_y, 2, 1)
        if not c_avail:
            c_avail = cache.is_available(mb_x - 1, mb_y, 3, 1)
            mvC = cache.get_mv(mb_x - 1, mb_y, 3, 1)

    else:  # sub_mb_idx == 3
        # Bottom-right 8x8: A from sub-MB 2, B from sub-MB 1, C from right MB or sub-MB 0
        a_avail = cache.is_available(mb_x, mb_y, 1, 2)
        mvA = cache.get_mv(mb_x, mb_y, 1, 2)

        b_avail = cache.is_available(mb_x, mb_y, 2, 1)
        mvB = cache.get_mv(mb_x, mb_y, 2, 1)

        # C unavailable (inside same MB), use D from sub-MB 0
        c_avail = cache.is_available(mb_x, mb_y, 1, 1)
        mvC = cache.get_mv(mb_x, mb_y, 1, 1)

    # Compute median
    avail_count = sum([a_avail, b_avail, c_avail])

    if avail_count == 0:
        return 0, 0

    if avail_count == 1:
        if a_avail:
            return mvA
        if b_avail:
            return mvB
        return mvC

    ax, ay = mvA if a_avail else (0, 0)
    bx, by = mvB if b_avail else (0, 0)
    cx, cy = mvC if c_avail else (0, 0)

    return _median(ax, bx, cx), _median(ay, by, cy)
