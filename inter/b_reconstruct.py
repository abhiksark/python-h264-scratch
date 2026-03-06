# h264/inter/b_reconstruct.py
"""B-macroblock reconstruction.

Combines bi-directional prediction (L0, L1, or both) with residual decoding
to reconstruct B-frame macroblocks.

H.264 Spec Reference: Section 8.4 - Inter prediction process
"""

import logging
from typing import Tuple, Optional, List

import numpy as np

from inter.reference import ReferenceFrameBuffer
from inter.motion_comp import (
    get_luma_block_fractional,
    get_chroma_block_fractional,
)
from inter.mv_prediction import MVCache
from inter.bipred import bipred_average, bipred_chroma, implicit_bipred
from inter.direct_mode import derive_direct_mv, derive_direct_spatial_sub8x8, derive_direct_temporal

logger = logging.getLogger(__name__)


def _bipred(
    pred_l0: np.ndarray,
    pred_l1: np.ndarray,
    weighted_bipred_idc: int = 0,
    current_poc: int = 0,
    l0_poc: int = 0,
    l1_poc: int = 0,
) -> np.ndarray:
    """Bi-predict with implicit weighting when enabled."""
    if weighted_bipred_idc == 2:
        return implicit_bipred(pred_l0, pred_l1, current_poc, l0_poc, l1_poc)
    return bipred_average(pred_l0, pred_l1)


def _apply_inter_prediction(
    ref_luma: np.ndarray,
    ref_x: int,
    ref_y: int,
    frac_x: int,
    frac_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Apply inter prediction to get a block from reference frame."""
    return get_luma_block_fractional(
        ref_luma, ref_x, ref_y, frac_x, frac_y, width, height
    )


def _apply_chroma_prediction(
    ref_chroma: np.ndarray,
    ref_x: int,
    ref_y: int,
    frac_x: int,
    frac_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Apply inter prediction for chroma component.

    H.264 Spec: Section 8.4.2.2.2 - Chroma sample interpolation (1/8-pixel bilinear)
    """
    return get_chroma_block_fractional(
        ref_chroma, ref_x, ref_y, frac_x, frac_y, width, height
    )


def _get_partition_chroma(
    ref_buffer: ReferenceFrameBuffer,
    list_id: str,
    ref_idx: int,
    mvx: int,
    mvy: int,
    mb_x: int,
    mb_y: int,
    cx_off: int,
    cy_off: int,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get chroma prediction for a partition sub-block.

    Args:
        list_id: "L0" or "L1"
        ref_idx: Reference index
        mvx, mvy: Motion vector (quarter-pel luma units)
        mb_x, mb_y: Macroblock position
        cx_off, cy_off: Chroma offset within MB (in chroma samples)
        width, height: Chroma block dimensions

    Returns:
        Tuple of (cb, cr) prediction blocks
    """
    if list_id == "L0":
        ref_frame = ref_buffer.get_l0_frame(ref_idx)
    else:
        ref_frame = ref_buffer.get_l1_frame(ref_idx)

    int_cmvx = mvx >> 3
    int_cmvy = mvy >> 3
    frac_cx = mvx & 7
    frac_cy = mvy & 7

    cx = mb_x * 8 + cx_off + int_cmvx
    cy = mb_y * 8 + cy_off + int_cmvy

    cb = _apply_chroma_prediction(ref_frame.cb, cx, cy, frac_cx, frac_cy, width, height)
    cr = _apply_chroma_prediction(ref_frame.cr, cx, cy, frac_cx, frac_cy, width, height)
    return cb, cr


def _get_prediction_l0(
    ref_buffer: ReferenceFrameBuffer,
    ref_idx: int,
    mvx: int,
    mvy: int,
    mb_x: int,
    mb_y: int,
    width: int = 16,
    height: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get L0 prediction block."""
    int_mvx = mvx >> 2
    int_mvy = mvy >> 2
    frac_x = mvx & 3
    frac_y = mvy & 3

    ref_x = mb_x * 16 + int_mvx
    ref_y = mb_y * 16 + int_mvy

    ref_frame = ref_buffer.get_l0_frame(ref_idx)

    luma = _apply_inter_prediction(
        ref_frame.luma, ref_x, ref_y, frac_x, frac_y, width, height
    )

    # Chroma: luma quarter-pel MV → chroma eighth-pel (4:2:0 halves resolution)
    int_cmvx = mvx >> 3
    int_cmvy = mvy >> 3
    frac_cx = mvx & 7
    frac_cy = mvy & 7

    ref_cx = mb_x * 8 + int_cmvx
    ref_cy = mb_y * 8 + int_cmvy

    cb = _apply_chroma_prediction(
        ref_frame.cb, ref_cx, ref_cy, frac_cx, frac_cy, width // 2, height // 2
    )
    cr = _apply_chroma_prediction(
        ref_frame.cr, ref_cx, ref_cy, frac_cx, frac_cy, width // 2, height // 2
    )

    return luma, cb, cr


def _get_prediction_l1(
    ref_buffer: ReferenceFrameBuffer,
    ref_idx: int,
    mvx: int,
    mvy: int,
    mb_x: int,
    mb_y: int,
    width: int = 16,
    height: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get L1 prediction block."""
    int_mvx = mvx >> 2
    int_mvy = mvy >> 2
    frac_x = mvx & 3
    frac_y = mvy & 3

    ref_x = mb_x * 16 + int_mvx
    ref_y = mb_y * 16 + int_mvy

    ref_frame = ref_buffer.get_l1_frame(ref_idx)

    luma = _apply_inter_prediction(
        ref_frame.luma, ref_x, ref_y, frac_x, frac_y, width, height
    )

    # Chroma: luma quarter-pel MV → chroma eighth-pel (4:2:0 halves resolution)
    int_cmvx = mvx >> 3
    int_cmvy = mvy >> 3
    frac_cx = mvx & 7
    frac_cy = mvy & 7

    ref_cx = mb_x * 8 + int_cmvx
    ref_cy = mb_y * 8 + int_cmvy

    cb = _apply_chroma_prediction(
        ref_frame.cb, ref_cx, ref_cy, frac_cx, frac_cy, width // 2, height // 2
    )
    cr = _apply_chroma_prediction(
        ref_frame.cr, ref_cx, ref_cy, frac_cx, frac_cy, width // 2, height // 2
    )

    return luma, cb, cr


def _add_residual(
    pred: np.ndarray,
    residual: Optional[np.ndarray],
) -> np.ndarray:
    """Add residual to prediction."""
    if residual is None:
        return pred
    return np.clip(
        pred.astype(np.int32) + residual, 0, 255
    ).astype(np.uint8)


def _get_sub_prediction(
    ref_buffer: ReferenceFrameBuffer,
    list_id: str,
    ref_idx: int,
    mvx: int,
    mvy: int,
    mb_x: int,
    mb_y: int,
    x_off: int,
    y_off: int,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get prediction for a sub-block within an MB.

    Args:
        ref_buffer: Reference frame buffer
        list_id: "L0" or "L1"
        ref_idx: Reference index
        mvx, mvy: Motion vector (quarter-pel)
        mb_x, mb_y: Macroblock position
        x_off, y_off: Pixel offset within MB
        width, height: Sub-block dimensions

    Returns:
        Tuple of (luma, cb, cr) prediction blocks
    """
    int_mvx = mvx >> 2
    int_mvy = mvy >> 2
    frac_x = mvx & 3
    frac_y = mvy & 3

    ref_x = mb_x * 16 + x_off + int_mvx
    ref_y = mb_y * 16 + y_off + int_mvy

    if list_id == "L0":
        ref_frame = ref_buffer.get_l0_frame(ref_idx)
    else:
        ref_frame = ref_buffer.get_l1_frame(ref_idx)

    luma = _apply_inter_prediction(
        ref_frame.luma, ref_x, ref_y, frac_x, frac_y, width, height
    )

    int_cmvx = mvx >> 3
    int_cmvy = mvy >> 3
    frac_cx = mvx & 7
    frac_cy = mvy & 7
    cx = mb_x * 8 + x_off // 2 + int_cmvx
    cy = mb_y * 8 + y_off // 2 + int_cmvy

    cb = _apply_chroma_prediction(
        ref_frame.cb, cx, cy, frac_cx, frac_cy, width // 2, height // 2
    )
    cr = _apply_chroma_prediction(
        ref_frame.cr, cx, cy, frac_cx, frac_cy, width // 2, height // 2
    )

    return luma, cb, cr


def reconstruct_b_skip(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    mb_x: int,
    mb_y: int,
    use_spatial: bool = True,
    current_poc: int = 0,
    mv_cache_l1: Optional[MVCache] = None,
    weighted_bipred_idc: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B_Skip macroblock (B_Direct_16x16 with no residual)."""
    return reconstruct_b_direct_16x16(
        ref_buffer=ref_buffer,
        mv_cache=mv_cache,
        residual_luma=None,
        residual_cb=None,
        residual_cr=None,
        mb_x=mb_x,
        mb_y=mb_y,
        use_spatial=use_spatial,
        current_poc=current_poc,
        mv_cache_l1=mv_cache_l1,
        weighted_bipred_idc=weighted_bipred_idc,
    )


def reconstruct_b_l0_16x16(
    ref_buffer: ReferenceFrameBuffer,
    ref_idx: int,
    mvx: int,
    mvy: int,
    residual_luma: Optional[np.ndarray],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B_L0_16x16 macroblock.

    Uses only L0 (forward) prediction.

    Args:
        ref_buffer: Reference frame buffer
        ref_idx: L0 reference index
        mvx, mvy: L0 motion vector
        residual_*: Residual blocks
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    pred_luma, pred_cb, pred_cr = _get_prediction_l0(
        ref_buffer, ref_idx, mvx, mvy, mb_x, mb_y
    )

    luma = _add_residual(pred_luma, residual_luma)
    cb = _add_residual(pred_cb, residual_cb)
    cr = _add_residual(pred_cr, residual_cr)

    logger.debug(f"B_L0_16x16 MB({mb_x},{mb_y}): ref={ref_idx}, MV=({mvx},{mvy})")

    return luma, cb, cr


def reconstruct_b_l1_16x16(
    ref_buffer: ReferenceFrameBuffer,
    ref_idx: int,
    mvx: int,
    mvy: int,
    residual_luma: Optional[np.ndarray],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B_L1_16x16 macroblock.

    Uses only L1 (backward) prediction.

    Args:
        ref_buffer: Reference frame buffer
        ref_idx: L1 reference index
        mvx, mvy: L1 motion vector
        residual_*: Residual blocks
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    pred_luma, pred_cb, pred_cr = _get_prediction_l1(
        ref_buffer, ref_idx, mvx, mvy, mb_x, mb_y
    )

    luma = _add_residual(pred_luma, residual_luma)
    cb = _add_residual(pred_cb, residual_cb)
    cr = _add_residual(pred_cr, residual_cr)

    logger.debug(f"B_L1_16x16 MB({mb_x},{mb_y}): ref={ref_idx}, MV=({mvx},{mvy})")

    return luma, cb, cr


def reconstruct_b_bi_16x16(
    ref_buffer: ReferenceFrameBuffer,
    ref_idx_l0: int,
    mvx_l0: int,
    mvy_l0: int,
    ref_idx_l1: int,
    mvx_l1: int,
    mvy_l1: int,
    residual_luma: Optional[np.ndarray],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
    weighted_bipred_idc: int = 0,
    current_poc: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B_Bi_16x16 macroblock.

    Averages L0 and L1 predictions (with implicit weighting if enabled).

    Args:
        ref_buffer: Reference frame buffer
        ref_idx_l0, ref_idx_l1: Reference indices
        mvx_l0, mvy_l0, mvx_l1, mvy_l1: Motion vectors
        residual_*: Residual blocks
        mb_x, mb_y: Macroblock position
        weighted_bipred_idc: 0=none, 1=explicit, 2=implicit
        current_poc: Current picture POC

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    # Get L0 and L1 predictions
    l0_luma, l0_cb, l0_cr = _get_prediction_l0(
        ref_buffer, ref_idx_l0, mvx_l0, mvy_l0, mb_x, mb_y
    )
    l1_luma, l1_cb, l1_cr = _get_prediction_l1(
        ref_buffer, ref_idx_l1, mvx_l1, mvy_l1, mb_x, mb_y
    )

    # Bi-predict
    l0_poc = ref_buffer.get_l0_frame(ref_idx_l0).poc
    l1_poc = ref_buffer.get_l1_frame(ref_idx_l1).poc
    pred_luma = _bipred(l0_luma, l1_luma, weighted_bipred_idc, current_poc, l0_poc, l1_poc)
    pred_cb = _bipred(l0_cb, l1_cb, weighted_bipred_idc, current_poc, l0_poc, l1_poc)
    pred_cr = _bipred(l0_cr, l1_cr, weighted_bipred_idc, current_poc, l0_poc, l1_poc)

    # Add residual
    luma = _add_residual(pred_luma, residual_luma)
    cb = _add_residual(pred_cb, residual_cb)
    cr = _add_residual(pred_cr, residual_cr)

    logger.debug(
        f"B_Bi_16x16 MB({mb_x},{mb_y}): "
        f"L0=(ref{ref_idx_l0},MV({mvx_l0},{mvy_l0})), "
        f"L1=(ref{ref_idx_l1},MV({mvx_l1},{mvy_l1}))"
    )

    return luma, cb, cr


def reconstruct_b_direct_16x16(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    residual_luma: Optional[np.ndarray],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
    use_spatial: bool = True,
    current_poc: int = 0,
    mv_cache_l1: Optional[MVCache] = None,
    weighted_bipred_idc: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B_Direct_16x16 macroblock as 4 B_Direct_8x8 sub-blocks.

    H.264 8.4.1.2.2: B_Direct_16x16 is treated as 4 independent B_Direct_8x8
    partitions, each with its own colZeroFlag check against the co-located
    8x8 block in the L1 reference.

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache for spatial prediction (L0)
        residual_*: Residual blocks
        mb_x, mb_y: Macroblock position
        use_spatial: True for spatial direct, False for temporal
        current_poc: POC for temporal direct
        mv_cache_l1: L1 MV cache for L1 prediction context
        weighted_bipred_idc: 0=none, 1=explicit, 2=implicit

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    pred_luma = np.zeros((16, 16), dtype=np.uint8)
    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)

    # Sub-block layout: 0=TL(0,0), 1=TR(8,0), 2=BL(0,8), 3=BR(8,8)
    sub_offsets = [(0, 0), (8, 0), (0, 8), (8, 8)]

    for sub_idx in range(4):
        x_off, y_off = sub_offsets[sub_idx]

        if use_spatial:
            (mvx_l0, mvy_l0, mvx_l1, mvy_l1, pf0, pf1,
             ri0, ri1) = derive_direct_spatial_sub8x8(
                mv_cache, mv_cache_l1, mb_x, mb_y, sub_idx,
                ref_buffer=ref_buffer,
            )
        else:
            if ref_buffer is not None:
                mvx_l0, mvy_l0, mvx_l1, mvy_l1 = derive_direct_temporal(
                    ref_buffer, current_poc, mb_x, mb_y, sub_idx=sub_idx,
                )
            else:
                mvx_l0, mvy_l0, mvx_l1, mvy_l1 = 0, 0, 0, 0
            pf0, pf1, ri0, ri1 = True, True, 0, 0

        eff_ri0 = max(ri0, 0)
        eff_ri1 = max(ri1, 0)

        # Get 8x8 prediction per sub-block
        if pf0 and pf1:
            l0 = _get_sub_prediction(
                ref_buffer, "L0", eff_ri0, mvx_l0, mvy_l0,
                mb_x, mb_y, x_off, y_off, 8, 8,
            )
            l1 = _get_sub_prediction(
                ref_buffer, "L1", eff_ri1, mvx_l1, mvy_l1,
                mb_x, mb_y, x_off, y_off, 8, 8,
            )
            l0_poc = ref_buffer.get_l0_frame(eff_ri0).poc
            l1_poc = ref_buffer.get_l1_frame(eff_ri1).poc
            sub_luma = _bipred(l0[0], l1[0], weighted_bipred_idc,
                               current_poc, l0_poc, l1_poc)
            sub_cb = _bipred(l0[1], l1[1], weighted_bipred_idc,
                             current_poc, l0_poc, l1_poc)
            sub_cr = _bipred(l0[2], l1[2], weighted_bipred_idc,
                             current_poc, l0_poc, l1_poc)
        elif pf0:
            sub_luma, sub_cb, sub_cr = _get_sub_prediction(
                ref_buffer, "L0", eff_ri0, mvx_l0, mvy_l0,
                mb_x, mb_y, x_off, y_off, 8, 8,
            )
        else:
            sub_luma, sub_cb, sub_cr = _get_sub_prediction(
                ref_buffer, "L1", eff_ri1, mvx_l1, mvy_l1,
                mb_x, mb_y, x_off, y_off, 8, 8,
            )

        # Place sub-block into prediction arrays
        pred_luma[y_off:y_off + 8, x_off:x_off + 8] = sub_luma
        cx_off, cy_off = x_off // 2, y_off // 2
        pred_cb[cy_off:cy_off + 4, cx_off:cx_off + 4] = sub_cb
        pred_cr[cy_off:cy_off + 4, cx_off:cx_off + 4] = sub_cr

        # Update MV caches per sub-block (4 4x4 blocks per 8x8)
        for dy in range(2):
            for dx in range(2):
                bx = (x_off // 4) + dx
                by = (y_off // 4) + dy
                if pf0:
                    mv_cache.set_mv(mb_x, mb_y, bx, by,
                                    mvx_l0, mvy_l0, ref_idx=eff_ri0)
                else:
                    mv_cache.set_mv(mb_x, mb_y, bx, by, 0, 0, ref_idx=-1)
                if mv_cache_l1 is not None:
                    if pf1:
                        mv_cache_l1.set_mv(mb_x, mb_y, bx, by,
                                           mvx_l1, mvy_l1, ref_idx=eff_ri1)
                    else:
                        mv_cache_l1.set_mv(mb_x, mb_y, bx, by,
                                           0, 0, ref_idx=-1)

    # Add residual
    luma = _add_residual(pred_luma, residual_luma)
    cb = _add_residual(pred_cb, residual_cb)
    cr = _add_residual(pred_cr, residual_cr)

    return luma, cb, cr


def reconstruct_b_16x8(
    ref_buffer: ReferenceFrameBuffer,
    pred_modes: List[str],
    ref_idx_l0: List[int],
    ref_idx_l1: List[int],
    mvx_l0: List[int],
    mvy_l0: List[int],
    mvx_l1: List[int],
    mvy_l1: List[int],
    residual_luma: Optional[np.ndarray],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
    weighted_bipred_idc: int = 0,
    current_poc: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B macroblock with 16x8 partitions.

    Args:
        ref_buffer: Reference frame buffer
        pred_modes: Prediction mode per partition ("L0", "L1", "Bi")
        ref_idx_l0, ref_idx_l1: Reference indices per partition
        mvx_l0, mvy_l0, mvx_l1, mvy_l1: MVs per partition
        residual_*: Residual blocks
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    luma = np.zeros((16, 16), dtype=np.uint8)
    cb = np.zeros((8, 8), dtype=np.uint8)
    cr = np.zeros((8, 8), dtype=np.uint8)

    for part in range(2):
        mode = pred_modes[part]
        part_y = part * 8

        if mode == "L0":
            part_luma, _, _ = _get_prediction_l0(
                ref_buffer, ref_idx_l0[part], mvx_l0[part], mvy_l0[part], mb_x, mb_y
            )
        elif mode == "L1":
            part_luma, _, _ = _get_prediction_l1(
                ref_buffer, ref_idx_l1[part], mvx_l1[part], mvy_l1[part], mb_x, mb_y
            )
        else:  # Bi
            l0_luma, _, _ = _get_prediction_l0(
                ref_buffer, ref_idx_l0[part], mvx_l0[part], mvy_l0[part], mb_x, mb_y
            )
            l1_luma, _, _ = _get_prediction_l1(
                ref_buffer, ref_idx_l1[part], mvx_l1[part], mvy_l1[part], mb_x, mb_y
            )
            l0p = ref_buffer.get_l0_frame(ref_idx_l0[part]).poc
            l1p = ref_buffer.get_l1_frame(ref_idx_l1[part]).poc
            part_luma = _bipred(l0_luma, l1_luma, weighted_bipred_idc, current_poc, l0p, l1p)

        luma[part_y:part_y + 8, :] = part_luma[part_y:part_y + 8, :]

    # Chroma: per-partition (top 4 rows / bottom 4 rows)
    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)
    for part in range(2):
        mode = pred_modes[part]
        cy_off = part * 4

        if mode in ("L0", "Bi"):
            l0_cb, l0_cr = _get_partition_chroma(
                ref_buffer, "L0", ref_idx_l0[part],
                mvx_l0[part], mvy_l0[part], mb_x, mb_y, 0, cy_off, 8, 4
            )
        if mode in ("L1", "Bi"):
            l1_cb, l1_cr = _get_partition_chroma(
                ref_buffer, "L1", ref_idx_l1[part],
                mvx_l1[part], mvy_l1[part], mb_x, mb_y, 0, cy_off, 8, 4
            )

        if mode == "L0":
            pred_cb[cy_off:cy_off + 4, :] = l0_cb
            pred_cr[cy_off:cy_off + 4, :] = l0_cr
        elif mode == "L1":
            pred_cb[cy_off:cy_off + 4, :] = l1_cb
            pred_cr[cy_off:cy_off + 4, :] = l1_cr
        else:
            l0p = ref_buffer.get_l0_frame(ref_idx_l0[part]).poc
            l1p = ref_buffer.get_l1_frame(ref_idx_l1[part]).poc
            pred_cb[cy_off:cy_off + 4, :] = _bipred(l0_cb, l1_cb, weighted_bipred_idc, current_poc, l0p, l1p)
            pred_cr[cy_off:cy_off + 4, :] = _bipred(l0_cr, l1_cr, weighted_bipred_idc, current_poc, l0p, l1p)

    cb = pred_cb
    cr = pred_cr

    # Add residual
    luma = _add_residual(luma, residual_luma)
    cb = _add_residual(cb, residual_cb)
    cr = _add_residual(cr, residual_cr)

    return luma, cb, cr


def reconstruct_b_8x16(
    ref_buffer: ReferenceFrameBuffer,
    pred_modes: List[str],
    ref_idx_l0: List[int],
    ref_idx_l1: List[int],
    mvx_l0: List[int],
    mvy_l0: List[int],
    mvx_l1: List[int],
    mvy_l1: List[int],
    residual_luma: Optional[np.ndarray],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
    weighted_bipred_idc: int = 0,
    current_poc: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B macroblock with 8x16 partitions."""
    luma = np.zeros((16, 16), dtype=np.uint8)
    cb = np.zeros((8, 8), dtype=np.uint8)
    cr = np.zeros((8, 8), dtype=np.uint8)

    for part in range(2):
        mode = pred_modes[part]
        part_x = part * 8

        if mode == "L0":
            part_luma, _, _ = _get_prediction_l0(
                ref_buffer, ref_idx_l0[part], mvx_l0[part], mvy_l0[part], mb_x, mb_y
            )
        elif mode == "L1":
            part_luma, _, _ = _get_prediction_l1(
                ref_buffer, ref_idx_l1[part], mvx_l1[part], mvy_l1[part], mb_x, mb_y
            )
        else:
            l0_luma, _, _ = _get_prediction_l0(
                ref_buffer, ref_idx_l0[part], mvx_l0[part], mvy_l0[part], mb_x, mb_y
            )
            l1_luma, _, _ = _get_prediction_l1(
                ref_buffer, ref_idx_l1[part], mvx_l1[part], mvy_l1[part], mb_x, mb_y
            )
            l0p = ref_buffer.get_l0_frame(ref_idx_l0[part]).poc
            l1p = ref_buffer.get_l1_frame(ref_idx_l1[part]).poc
            part_luma = _bipred(l0_luma, l1_luma, weighted_bipred_idc, current_poc, l0p, l1p)

        luma[:, part_x:part_x + 8] = part_luma[:, part_x:part_x + 8]

    # Chroma: per-partition (left 4 cols / right 4 cols)
    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)
    for part in range(2):
        mode = pred_modes[part]
        cx_off = part * 4

        if mode in ("L0", "Bi"):
            l0_cb, l0_cr = _get_partition_chroma(
                ref_buffer, "L0", ref_idx_l0[part],
                mvx_l0[part], mvy_l0[part], mb_x, mb_y, cx_off, 0, 4, 8
            )
        if mode in ("L1", "Bi"):
            l1_cb, l1_cr = _get_partition_chroma(
                ref_buffer, "L1", ref_idx_l1[part],
                mvx_l1[part], mvy_l1[part], mb_x, mb_y, cx_off, 0, 4, 8
            )

        if mode == "L0":
            pred_cb[:, cx_off:cx_off + 4] = l0_cb
            pred_cr[:, cx_off:cx_off + 4] = l0_cr
        elif mode == "L1":
            pred_cb[:, cx_off:cx_off + 4] = l1_cb
            pred_cr[:, cx_off:cx_off + 4] = l1_cr
        else:
            l0p = ref_buffer.get_l0_frame(ref_idx_l0[part]).poc
            l1p = ref_buffer.get_l1_frame(ref_idx_l1[part]).poc
            pred_cb[:, cx_off:cx_off + 4] = _bipred(l0_cb, l1_cb, weighted_bipred_idc, current_poc, l0p, l1p)
            pred_cr[:, cx_off:cx_off + 4] = _bipred(l0_cr, l1_cr, weighted_bipred_idc, current_poc, l0p, l1p)

    cb = pred_cb
    cr = pred_cr

    luma = _add_residual(luma, residual_luma)
    cb = _add_residual(cb, residual_cb)
    cr = _add_residual(cr, residual_cr)

    return luma, cb, cr


def reconstruct_b_8x8(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    sub_mb_types: List[int],
    pred_modes: List[str],
    ref_idx_l0: List[int],
    ref_idx_l1: List[int],
    mvx_l0: List[int],
    mvy_l0: List[int],
    mvx_l1: List[int],
    mvy_l1: List[int],
    residual_luma: Optional[np.ndarray],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
    weighted_bipred_idc: int = 0,
    current_poc: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B_8x8 macroblock with 4 sub-blocks."""
    luma = np.zeros((16, 16), dtype=np.uint8)

    sub_offsets = [(0, 0), (8, 0), (0, 8), (8, 8)]

    for sub_idx in range(4):
        mode = pred_modes[sub_idx]
        sub_x, sub_y = sub_offsets[sub_idx]

        if mode == "L0":
            sub_luma, _, _ = _get_prediction_l0(
                ref_buffer, ref_idx_l0[sub_idx], mvx_l0[sub_idx], mvy_l0[sub_idx],
                mb_x, mb_y, width=16, height=16
            )
        elif mode == "L1":
            sub_luma, _, _ = _get_prediction_l1(
                ref_buffer, ref_idx_l1[sub_idx], mvx_l1[sub_idx], mvy_l1[sub_idx],
                mb_x, mb_y, width=16, height=16
            )
        else:  # Bi or Direct
            l0_luma, _, _ = _get_prediction_l0(
                ref_buffer, ref_idx_l0[sub_idx], mvx_l0[sub_idx], mvy_l0[sub_idx],
                mb_x, mb_y, width=16, height=16
            )
            l1_luma, _, _ = _get_prediction_l1(
                ref_buffer, ref_idx_l1[sub_idx], mvx_l1[sub_idx], mvy_l1[sub_idx],
                mb_x, mb_y, width=16, height=16
            )
            l0p = ref_buffer.get_l0_frame(ref_idx_l0[sub_idx]).poc
            l1p = ref_buffer.get_l1_frame(ref_idx_l1[sub_idx]).poc
            sub_luma = _bipred(l0_luma, l1_luma, weighted_bipred_idc, current_poc, l0p, l1p)

        luma[sub_y:sub_y + 8, sub_x:sub_x + 8] = sub_luma[sub_y:sub_y + 8, sub_x:sub_x + 8]

    # Chroma: per-sub-MB (4 quadrants of 4x4 chroma)
    chroma_offsets = [(0, 0), (4, 0), (0, 4), (4, 4)]
    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)
    for sub_idx in range(4):
        cx_off, cy_off = chroma_offsets[sub_idx]
        mode = pred_modes[sub_idx]

        if mode in ("L0", "Bi", "Direct"):
            l0_cb, l0_cr = _get_partition_chroma(
                ref_buffer, "L0", ref_idx_l0[sub_idx],
                mvx_l0[sub_idx], mvy_l0[sub_idx], mb_x, mb_y,
                cx_off, cy_off, 4, 4
            )
        if mode in ("L1", "Bi", "Direct"):
            l1_cb, l1_cr = _get_partition_chroma(
                ref_buffer, "L1", ref_idx_l1[sub_idx],
                mvx_l1[sub_idx], mvy_l1[sub_idx], mb_x, mb_y,
                cx_off, cy_off, 4, 4
            )

        if mode == "L0":
            pred_cb[cy_off:cy_off + 4, cx_off:cx_off + 4] = l0_cb
            pred_cr[cy_off:cy_off + 4, cx_off:cx_off + 4] = l0_cr
        elif mode == "L1":
            pred_cb[cy_off:cy_off + 4, cx_off:cx_off + 4] = l1_cb
            pred_cr[cy_off:cy_off + 4, cx_off:cx_off + 4] = l1_cr
        else:  # Bi or Direct
            l0p = ref_buffer.get_l0_frame(ref_idx_l0[sub_idx]).poc
            l1p = ref_buffer.get_l1_frame(ref_idx_l1[sub_idx]).poc
            pred_cb[cy_off:cy_off + 4, cx_off:cx_off + 4] = _bipred(l0_cb, l1_cb, weighted_bipred_idc, current_poc, l0p, l1p)
            pred_cr[cy_off:cy_off + 4, cx_off:cx_off + 4] = _bipred(l0_cr, l1_cr, weighted_bipred_idc, current_poc, l0p, l1p)

    cb = pred_cb
    cr = pred_cr

    luma = _add_residual(luma, residual_luma)
    cb = _add_residual(cb, residual_cb)
    cr = _add_residual(cr, residual_cr)

    return luma, cb, cr
