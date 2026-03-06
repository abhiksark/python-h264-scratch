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
from inter.bipred import bipred_average, bipred_chroma
from inter.direct_mode import derive_direct_mv

logger = logging.getLogger(__name__)


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


def reconstruct_b_skip(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    mb_x: int,
    mb_y: int,
    use_spatial: bool = True,
    current_poc: int = 0,
    mv_cache_l1: Optional[MVCache] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B_Skip macroblock.

    B_Skip uses direct mode MVs with no residual.

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache for spatial prediction (L0)
        mb_x, mb_y: Macroblock position
        use_spatial: True for spatial direct, False for temporal
        current_poc: POC of current picture (for temporal direct)
        mv_cache_l1: L1 MV cache for L1 prediction context

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    # Derive MVs using direct mode
    mvx_l0, mvy_l0, mvx_l1, mvy_l1 = derive_direct_mv(
        mv_cache, ref_buffer, current_poc, mb_x, mb_y, use_spatial,
        mv_cache_l1
    )

    # Get L0 and L1 predictions
    l0_luma, l0_cb, l0_cr = _get_prediction_l0(
        ref_buffer, 0, mvx_l0, mvy_l0, mb_x, mb_y
    )
    l1_luma, l1_cb, l1_cr = _get_prediction_l1(
        ref_buffer, 0, mvx_l1, mvy_l1, mb_x, mb_y
    )

    # Bi-predict
    luma = bipred_average(l0_luma, l1_luma)
    cb, cr = bipred_chroma(l0_cb, l1_cb, l0_cr, l1_cr)

    # Update MV caches
    mv_cache.set_mv_16x16(mb_x, mb_y, mvx_l0, mvy_l0)
    if mv_cache_l1 is not None:
        mv_cache_l1.set_mv_16x16(mb_x, mb_y, mvx_l1, mvy_l1)

    logger.debug(f"B_Skip MB({mb_x},{mb_y}): MVL0=({mvx_l0},{mvy_l0}), MVL1=({mvx_l1},{mvy_l1})")

    return luma, cb, cr


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B_Bi_16x16 macroblock.

    Averages L0 and L1 predictions.

    Args:
        ref_buffer: Reference frame buffer
        ref_idx_l0, ref_idx_l1: Reference indices
        mvx_l0, mvy_l0, mvx_l1, mvy_l1: Motion vectors
        residual_*: Residual blocks
        mb_x, mb_y: Macroblock position

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
    pred_luma = bipred_average(l0_luma, l1_luma)
    pred_cb, pred_cr = bipred_chroma(l0_cb, l1_cb, l0_cr, l1_cr)

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct B_Direct_16x16 macroblock.

    MVs are derived from neighbors (spatial) or co-located (temporal).

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache for spatial prediction (L0)
        residual_*: Residual blocks
        mb_x, mb_y: Macroblock position
        use_spatial: True for spatial direct, False for temporal
        current_poc: POC for temporal direct
        mv_cache_l1: L1 MV cache for L1 prediction context

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    # Derive MVs
    mvx_l0, mvy_l0, mvx_l1, mvy_l1 = derive_direct_mv(
        mv_cache, ref_buffer, current_poc, mb_x, mb_y, use_spatial,
        mv_cache_l1
    )

    # Get predictions
    l0_luma, l0_cb, l0_cr = _get_prediction_l0(
        ref_buffer, 0, mvx_l0, mvy_l0, mb_x, mb_y
    )
    l1_luma, l1_cb, l1_cr = _get_prediction_l1(
        ref_buffer, 0, mvx_l1, mvy_l1, mb_x, mb_y
    )

    # Bi-predict
    pred_luma = bipred_average(l0_luma, l1_luma)
    pred_cb, pred_cr = bipred_chroma(l0_cb, l1_cb, l0_cr, l1_cr)

    # Add residual
    luma = _add_residual(pred_luma, residual_luma)
    cb = _add_residual(pred_cb, residual_cb)
    cr = _add_residual(pred_cr, residual_cr)

    # Update MV caches
    mv_cache.set_mv_16x16(mb_x, mb_y, mvx_l0, mvy_l0)
    if mv_cache_l1 is not None:
        mv_cache_l1.set_mv_16x16(mb_x, mb_y, mvx_l1, mvy_l1)

    logger.debug(f"B_Direct MB({mb_x},{mb_y}): MVL0=({mvx_l0},{mvy_l0}), MVL1=({mvx_l1},{mvy_l1})")

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
            part_luma = bipred_average(l0_luma, l1_luma)

        luma[part_y:part_y + 8, :] = part_luma[part_y:part_y + 8, :]

    # Simplified chroma: use first partition mode
    if pred_modes[0] in ("L0", "Bi"):
        _, pred_cb, pred_cr = _get_prediction_l0(
            ref_buffer, ref_idx_l0[0], mvx_l0[0], mvy_l0[0], mb_x, mb_y
        )
    else:
        _, pred_cb, pred_cr = _get_prediction_l1(
            ref_buffer, ref_idx_l1[0], mvx_l1[0], mvy_l1[0], mb_x, mb_y
        )

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
            part_luma = bipred_average(l0_luma, l1_luma)

        luma[:, part_x:part_x + 8] = part_luma[:, part_x:part_x + 8]

    # Simplified chroma
    if pred_modes[0] in ("L0", "Bi"):
        _, pred_cb, pred_cr = _get_prediction_l0(
            ref_buffer, ref_idx_l0[0], mvx_l0[0], mvy_l0[0], mb_x, mb_y
        )
    else:
        _, pred_cb, pred_cr = _get_prediction_l1(
            ref_buffer, ref_idx_l1[0], mvx_l1[0], mvy_l1[0], mb_x, mb_y
        )

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
            sub_luma = bipred_average(l0_luma, l1_luma)

        luma[sub_y:sub_y + 8, sub_x:sub_x + 8] = sub_luma[sub_y:sub_y + 8, sub_x:sub_x + 8]

    # Simplified chroma
    _, cb, cr = _get_prediction_l0(
        ref_buffer, ref_idx_l0[0], mvx_l0[0], mvy_l0[0], mb_x, mb_y
    )

    luma = _add_residual(luma, residual_luma)
    cb = _add_residual(cb, residual_cb)
    cr = _add_residual(cr, residual_cr)

    return luma, cb, cr
