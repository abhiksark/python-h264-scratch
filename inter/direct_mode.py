# h264/inter/direct_mode.py
"""Direct mode MV derivation for B-frames.

B_Direct and B_Skip macroblocks derive MVs without explicit transmission.
Two modes are supported based on direct_spatial_mv_pred_flag:
1. Spatial direct (flag=1): Derive from neighbor MVs
2. Temporal direct (flag=0): Scale co-located MV from L1 reference

H.264 Spec Reference: Section 8.4.1.2 - Derivation of motion vector components
"""

from typing import Tuple, Optional
import numpy as np

from inter.mv_prediction import MVCache
from inter.reference import ReferenceFrameBuffer


def derive_direct_spatial(
    mv_cache: MVCache,
    mb_x: int,
    mb_y: int,
    mv_cache_l1: Optional[MVCache] = None,
) -> Tuple[int, int, int, int]:
    """Derive direct mode MVs using spatial prediction.

    Derives L0 and L1 MVs independently from neighbor MVs in each list.

    Args:
        mv_cache: L0 MV cache with neighbor MVs
        mb_x, mb_y: Macroblock position
        mv_cache_l1: L1 MV cache (if None, L1 MV = median of L0 neighbors negated)

    Returns:
        Tuple of (mvx_l0, mvy_l0, mvx_l1, mvy_l1)

    H.264 Spec: Section 8.4.1.2.2
    """
    # Get L0 neighbor MVs
    mv_a_l0 = mv_cache.get_mv(mb_x - 1, mb_y, 3, 0) if mb_x > 0 else (0, 0)
    mv_b_l0 = mv_cache.get_mv(mb_x, mb_y - 1, 0, 3) if mb_y > 0 else (0, 0)

    if mb_y > 0 and mb_x < mv_cache.width_in_mbs - 1:
        mv_c_l0 = mv_cache.get_mv(mb_x + 1, mb_y - 1, 0, 3)
    elif mb_y > 0 and mb_x > 0:
        mv_c_l0 = mv_cache.get_mv(mb_x - 1, mb_y - 1, 3, 3)
    else:
        mv_c_l0 = (0, 0)

    # Median prediction for L0
    mvx_l0 = _median(mv_a_l0[0], mv_b_l0[0], mv_c_l0[0])
    mvy_l0 = _median(mv_a_l0[1], mv_b_l0[1], mv_c_l0[1])

    # Get L1 neighbor MVs from L1 cache
    if mv_cache_l1 is not None:
        mv_a_l1 = mv_cache_l1.get_mv(mb_x - 1, mb_y, 3, 0) if mb_x > 0 else (0, 0)
        mv_b_l1 = mv_cache_l1.get_mv(mb_x, mb_y - 1, 0, 3) if mb_y > 0 else (0, 0)

        if mb_y > 0 and mb_x < mv_cache_l1.width_in_mbs - 1:
            mv_c_l1 = mv_cache_l1.get_mv(mb_x + 1, mb_y - 1, 0, 3)
        elif mb_y > 0 and mb_x > 0:
            mv_c_l1 = mv_cache_l1.get_mv(mb_x - 1, mb_y - 1, 3, 3)
        else:
            mv_c_l1 = (0, 0)

        mvx_l1 = _median(mv_a_l1[0], mv_b_l1[0], mv_c_l1[0])
        mvy_l1 = _median(mv_a_l1[1], mv_b_l1[1], mv_c_l1[1])
    else:
        mvx_l1 = -mvx_l0
        mvy_l1 = -mvy_l0

    return mvx_l0, mvy_l0, mvx_l1, mvy_l1


def derive_direct_temporal(
    ref_buffer: ReferenceFrameBuffer,
    current_poc: int,
    mb_x: int,
    mb_y: int,
) -> Tuple[int, int, int, int]:
    """Derive direct mode MVs using temporal prediction.

    Scales co-located MV from L1 reference based on POC distances.

    Args:
        ref_buffer: Reference frame buffer with L0/L1 lists
        current_poc: POC of current picture
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (mvx_l0, mvy_l0, mvx_l1, mvy_l1)

    H.264 Spec: Section 8.4.1.2.3
    """
    # Get L1 reference frame (first in L1 list)
    l1_list = ref_buffer.get_l1_list()
    if not l1_list:
        return (0, 0, 0, 0)

    l1_frame = l1_list[0]
    l1_poc = l1_frame.poc

    # Get co-located MV from L1 reference
    col_mv = l1_frame.get_colocated_mv(mb_x, mb_y)
    col_mvx, col_mvy = col_mv

    # If co-located MV is zero, return zeros
    if col_mvx == 0 and col_mvy == 0:
        return (0, 0, 0, 0)

    # Get L0 reference (assume ref_idx=0 for simplicity)
    l0_list = ref_buffer.get_l0_list()
    if not l0_list:
        return (0, 0, 0, 0)

    l0_poc = l0_list[0].poc

    # Calculate temporal distances
    tb = current_poc - l0_poc  # Distance to L0
    td = l1_poc - l0_poc  # Distance between references

    if td == 0:
        # Same POC - can't scale
        return (col_mvx, col_mvy, -col_mvx, -col_mvy)

    # Scale factor: tb/td, represented as (tb * 256) / td
    tx = (16384 + abs(td // 2)) // td
    dist_scale_factor = (tb * tx + 32) >> 6
    dist_scale_factor = max(-1024, min(1023, dist_scale_factor))

    # Scale MVs
    mvx_l0 = (dist_scale_factor * col_mvx + 128) >> 8
    mvy_l0 = (dist_scale_factor * col_mvy + 128) >> 8

    # L1 MV is complementary
    mvx_l1 = mvx_l0 - col_mvx
    mvy_l1 = mvy_l0 - col_mvy

    return mvx_l0, mvy_l0, mvx_l1, mvy_l1


def derive_direct_mv(
    mv_cache: MVCache,
    ref_buffer: Optional[ReferenceFrameBuffer],
    current_poc: int,
    mb_x: int,
    mb_y: int,
    use_spatial: bool,
    mv_cache_l1: Optional[MVCache] = None,
) -> Tuple[int, int, int, int]:
    """Derive direct mode MVs based on direct_spatial_mv_pred_flag.

    Args:
        mv_cache: MV cache for spatial prediction (L0)
        ref_buffer: Reference buffer for temporal prediction
        current_poc: POC of current picture
        mb_x, mb_y: Macroblock position
        use_spatial: True for spatial direct, False for temporal
        mv_cache_l1: L1 MV cache for spatial direct mode

    Returns:
        Tuple of (mvx_l0, mvy_l0, mvx_l1, mvy_l1)
    """
    if use_spatial:
        return derive_direct_spatial(mv_cache, mb_x, mb_y, mv_cache_l1)
    else:
        if ref_buffer is None:
            return (0, 0, 0, 0)
        return derive_direct_temporal(ref_buffer, current_poc, mb_x, mb_y)


def derive_b_skip_mv(
    mv_cache: MVCache,
    ref_buffer: Optional[ReferenceFrameBuffer],
    current_poc: int,
    mb_x: int,
    mb_y: int,
    use_spatial: bool,
    mv_cache_l1: Optional[MVCache] = None,
) -> Tuple[int, int, int, int]:
    """Derive MVs for B_Skip macroblock.

    B_Skip is equivalent to B_Direct_16x16 with no residual.

    Args:
        mv_cache: MV cache for spatial prediction (L0)
        ref_buffer: Reference buffer for temporal prediction
        current_poc: POC of current picture
        mb_x, mb_y: Macroblock position
        use_spatial: True for spatial direct, False for temporal
        mv_cache_l1: L1 MV cache for spatial direct mode

    Returns:
        Tuple of (mvx_l0, mvy_l0, mvx_l1, mvy_l1)
    """
    return derive_direct_mv(
        mv_cache, ref_buffer, current_poc, mb_x, mb_y, use_spatial,
        mv_cache_l1
    )


def _median(a: int, b: int, c: int) -> int:
    """Calculate median of three values."""
    if a > b:
        if b > c:
            return b
        elif a > c:
            return c
        else:
            return a
    else:
        if a > c:
            return a
        elif b > c:
            return c
        else:
            return b
