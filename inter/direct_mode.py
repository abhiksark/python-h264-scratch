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

from inter.mv_prediction import MVCache, predict_mv_16x16, predict_mv_8x8
from inter.reference import ReferenceFrameBuffer


def derive_direct_spatial(
    mv_cache: MVCache,
    mb_x: int,
    mb_y: int,
    mv_cache_l1: Optional[MVCache] = None,
    ref_buffer: Optional[ReferenceFrameBuffer] = None,
) -> Tuple[int, int, int, int, bool, bool]:
    """Derive direct mode MVs using spatial prediction.

    Derives L0 and L1 MVs from neighbor MVs, and determines prediction flags
    based on neighbor reference indices per H.264 8.4.1.2.2.

    Args:
        mv_cache: L0 MV cache with neighbor MVs
        mb_x, mb_y: Macroblock position
        mv_cache_l1: L1 MV cache (if None, L1 MV = median of L0 neighbors negated)
        ref_buffer: Reference frame buffer (for colZeroFlag check)

    Returns:
        Tuple of (mvx_l0, mvy_l0, mvx_l1, mvy_l1, pred_flag_l0, pred_flag_l1)

    H.264 Spec: Section 8.4.1.2.2
    """
    # Derive L0 ref_idx: minimum of available neighbor L0 ref indices
    ref_a_l0 = mv_cache.get_ref_idx(mb_x - 1, mb_y, 3, 0) if mb_x > 0 else -1
    ref_b_l0 = mv_cache.get_ref_idx(mb_x, mb_y - 1, 0, 3) if mb_y > 0 else -1

    if mb_y > 0 and mb_x < mv_cache.width_in_mbs - 1:
        ref_c_l0 = mv_cache.get_ref_idx(mb_x + 1, mb_y - 1, 0, 3)
    elif mb_y > 0 and mb_x > 0:
        ref_c_l0 = mv_cache.get_ref_idx(mb_x - 1, mb_y - 1, 3, 3)
    else:
        ref_c_l0 = -1

    l0_refs = [r for r in (ref_a_l0, ref_b_l0, ref_c_l0) if r >= 0]
    ref_idx_l0 = min(l0_refs) if l0_refs else -1

    # Derive L1 ref_idx similarly
    ref_idx_l1 = -1
    if mv_cache_l1 is not None:
        ref_a_l1 = mv_cache_l1.get_ref_idx(mb_x - 1, mb_y, 3, 0) if mb_x > 0 else -1
        ref_b_l1 = mv_cache_l1.get_ref_idx(mb_x, mb_y - 1, 0, 3) if mb_y > 0 else -1

        if mb_y > 0 and mb_x < mv_cache_l1.width_in_mbs - 1:
            ref_c_l1 = mv_cache_l1.get_ref_idx(mb_x + 1, mb_y - 1, 0, 3)
        elif mb_y > 0 and mb_x > 0:
            ref_c_l1 = mv_cache_l1.get_ref_idx(mb_x - 1, mb_y - 1, 3, 3)
        else:
            ref_c_l1 = -1

        l1_refs = [r for r in (ref_a_l1, ref_b_l1, ref_c_l1) if r >= 0]
        ref_idx_l1 = min(l1_refs) if l1_refs else -1

    # H.264 8.4.1.2.2: Derive prediction flags from ref indices
    pred_flag_l0 = ref_idx_l0 >= 0
    pred_flag_l1 = ref_idx_l1 >= 0

    # If both flags are 0 (no valid refs at all), fallback to bipred with ref 0
    if not pred_flag_l0 and not pred_flag_l1:
        pred_flag_l0 = True
        pred_flag_l1 = True
        ref_idx_l0 = 0
        ref_idx_l1 = 0

    # Derive MVs using standard prediction process (8.4.1.3.1)
    # This includes the single-match ref_idx shortcut
    mvx_l0, mvy_l0 = 0, 0
    if pred_flag_l0:
        mvx_l0, mvy_l0 = predict_mv_16x16(
            mv_cache, mb_x, mb_y, target_ref=ref_idx_l0
        )

    mvx_l1, mvy_l1 = 0, 0
    if pred_flag_l1 and mv_cache_l1 is not None:
        mvx_l1, mvy_l1 = predict_mv_16x16(
            mv_cache_l1, mb_x, mb_y, target_ref=ref_idx_l1
        )
    elif pred_flag_l1:
        mvx_l1 = -mvx_l0
        mvy_l1 = -mvy_l0

    # H.264 8.4.1.2.2: colZeroFlag check
    # When refIdxL0==0 or refIdxL1==0, check co-located block in L1[0].
    # colZeroFlag is true when co-located has refIdx==0 AND |MV| <= 1.
    # JM: (ref_idx[LIST_0]==0 && |mv|<=1) — must check BOTH conditions.
    if ref_buffer is not None and (ref_idx_l0 == 0 or ref_idx_l1 == 0):
        l1_list = ref_buffer.get_l1_list()
        if l1_list:
            col_frame = l1_list[0]
            col_mv = col_frame.get_colocated_mv(mb_x, mb_y)
            col_ref = col_frame.get_colocated_ref_idx(mb_x, mb_y)
            col_mvx, col_mvy = col_mv
            col_zero = (
                col_ref == 0
                and abs(col_mvx) <= 1
                and abs(col_mvy) <= 1
            )
            if col_zero:
                if ref_idx_l0 == 0:
                    mvx_l0 = 0
                    mvy_l0 = 0
                if ref_idx_l1 == 0:
                    mvx_l1 = 0
                    mvy_l1 = 0

    return mvx_l0, mvy_l0, mvx_l1, mvy_l1, pred_flag_l0, pred_flag_l1, ref_idx_l0, ref_idx_l1


def derive_direct_spatial_sub8x8(
    mv_cache_l0: MVCache,
    mv_cache_l1: Optional[MVCache],
    mb_x: int,
    mb_y: int,
    sub_mb_idx: int,
    ref_buffer: Optional[ReferenceFrameBuffer] = None,
) -> Tuple[int, int, int, int, bool, bool]:
    """Derive direct mode MVs for a B_Direct_8x8 sub-block using spatial prediction.

    H.264 8.4.1.2.2: refIdx is derived from MB-level neighbors (same for
    all B_Direct_8x8 sub-blocks in the MB), while the MV prediction uses
    sub-block-level neighbors via predict_mv_8x8.

    Args:
        mv_cache_l0: L0 MV cache
        mv_cache_l1: L1 MV cache
        mb_x, mb_y: Macroblock position
        sub_mb_idx: Sub-macroblock index (0=TL, 1=TR, 2=BL, 3=BR)
        ref_buffer: Reference frame buffer (for colZeroFlag check)

    Returns:
        Tuple of (mvx_l0, mvy_l0, mvx_l1, mvy_l1, pred_flag_l0, pred_flag_l1)

    H.264 Spec: Section 8.4.1.2.2
    """
    # Derive refIdx from MB-level neighbors (same as B_Direct_16x16)
    ref_a_l0 = mv_cache_l0.get_ref_idx(mb_x - 1, mb_y, 3, 0) if mb_x > 0 else -1
    ref_b_l0 = mv_cache_l0.get_ref_idx(mb_x, mb_y - 1, 0, 3) if mb_y > 0 else -1

    if mb_y > 0 and mb_x < mv_cache_l0.width_in_mbs - 1:
        ref_c_l0 = mv_cache_l0.get_ref_idx(mb_x + 1, mb_y - 1, 0, 3)
    elif mb_y > 0 and mb_x > 0:
        ref_c_l0 = mv_cache_l0.get_ref_idx(mb_x - 1, mb_y - 1, 3, 3)
    else:
        ref_c_l0 = -1

    l0_refs = [r for r in (ref_a_l0, ref_b_l0, ref_c_l0) if r >= 0]
    ref_idx_l0 = min(l0_refs) if l0_refs else -1

    ref_idx_l1 = -1
    if mv_cache_l1 is not None:
        ref_a_l1 = mv_cache_l1.get_ref_idx(mb_x - 1, mb_y, 3, 0) if mb_x > 0 else -1
        ref_b_l1 = mv_cache_l1.get_ref_idx(mb_x, mb_y - 1, 0, 3) if mb_y > 0 else -1

        if mb_y > 0 and mb_x < mv_cache_l1.width_in_mbs - 1:
            ref_c_l1 = mv_cache_l1.get_ref_idx(mb_x + 1, mb_y - 1, 0, 3)
        elif mb_y > 0 and mb_x > 0:
            ref_c_l1 = mv_cache_l1.get_ref_idx(mb_x - 1, mb_y - 1, 3, 3)
        else:
            ref_c_l1 = -1

        l1_refs = [r for r in (ref_a_l1, ref_b_l1, ref_c_l1) if r >= 0]
        ref_idx_l1 = min(l1_refs) if l1_refs else -1

    # H.264 8.4.1.2.2: Derive prediction flags
    pred_flag_l0 = ref_idx_l0 >= 0
    pred_flag_l1 = ref_idx_l1 >= 0

    if not pred_flag_l0 and not pred_flag_l1:
        pred_flag_l0 = True
        pred_flag_l1 = True
        ref_idx_l0 = 0
        ref_idx_l1 = 0

    # H.264 8.4.1.2.2: B_Direct_8x8 uses MB-level (16x16) MV prediction,
    # same as B_Direct_16x16. Per JM reference: prepare_direct_params()
    # calls GetMVPredictor with block_size=16x16 once for the whole MB.
    mvx_l0, mvy_l0 = 0, 0
    if pred_flag_l0:
        mvx_l0, mvy_l0 = predict_mv_16x16(
            mv_cache_l0, mb_x, mb_y, target_ref=ref_idx_l0
        )

    mvx_l1, mvy_l1 = 0, 0
    if pred_flag_l1 and mv_cache_l1 is not None:
        mvx_l1, mvy_l1 = predict_mv_16x16(
            mv_cache_l1, mb_x, mb_y, target_ref=ref_idx_l1
        )
    elif pred_flag_l1:
        mvx_l1 = -mvx_l0
        mvy_l1 = -mvy_l0

    # H.264 8.4.1.2.2: colZeroFlag check (per-sub-block co-located data)
    # Must check co-located refIdx==0 AND |MV| <= 1 (both conditions).
    if ref_buffer is not None and (ref_idx_l0 == 0 or ref_idx_l1 == 0):
        l1_list = ref_buffer.get_l1_list()
        if l1_list:
            col_frame = l1_list[0]
            col_mv = col_frame.get_colocated_mv(mb_x, mb_y, sub_idx=sub_mb_idx)
            col_ref = col_frame.get_colocated_ref_idx(mb_x, mb_y, sub_idx=sub_mb_idx)
            col_mvx, col_mvy = col_mv
            col_zero = (
                col_ref == 0
                and abs(col_mvx) <= 1
                and abs(col_mvy) <= 1
            )
            if col_zero:
                if ref_idx_l0 == 0:
                    mvx_l0 = 0
                    mvy_l0 = 0
                if ref_idx_l1 == 0:
                    mvx_l1 = 0
                    mvy_l1 = 0

    return mvx_l0, mvy_l0, mvx_l1, mvy_l1, pred_flag_l0, pred_flag_l1, ref_idx_l0, ref_idx_l1


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
) -> Tuple[int, int, int, int, bool, bool, int, int]:
    """Derive direct mode MVs based on direct_spatial_mv_pred_flag.

    Args:
        mv_cache: MV cache for spatial prediction (L0)
        ref_buffer: Reference buffer for temporal prediction
        current_poc: POC of current picture
        mb_x, mb_y: Macroblock position
        use_spatial: True for spatial direct, False for temporal
        mv_cache_l1: L1 MV cache for spatial direct mode

    Returns:
        Tuple of (mvx_l0, mvy_l0, mvx_l1, mvy_l1,
                  pred_flag_l0, pred_flag_l1, ref_idx_l0, ref_idx_l1)
    """
    if use_spatial:
        return derive_direct_spatial(
            mv_cache, mb_x, mb_y, mv_cache_l1, ref_buffer
        )
    else:
        if ref_buffer is None:
            return (0, 0, 0, 0, True, True, 0, 0)
        mvs = derive_direct_temporal(ref_buffer, current_poc, mb_x, mb_y)
        return mvs + (True, True, 0, 0)  # Temporal always bi-predicts with ref 0


def derive_b_skip_mv(
    mv_cache: MVCache,
    ref_buffer: Optional[ReferenceFrameBuffer],
    current_poc: int,
    mb_x: int,
    mb_y: int,
    use_spatial: bool,
    mv_cache_l1: Optional[MVCache] = None,
) -> Tuple[int, int, int, int, bool, bool, int, int]:
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
        Tuple of (mvx_l0, mvy_l0, mvx_l1, mvy_l1,
                  pred_flag_l0, pred_flag_l1, ref_idx_l0, ref_idx_l1)
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
