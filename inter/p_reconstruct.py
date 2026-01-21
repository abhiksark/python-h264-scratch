# h264/inter/p_reconstruct.py
"""P-macroblock reconstruction.

Combines motion compensation with residual decoding to reconstruct
P-frame macroblocks.

H.264 Spec Reference: Section 8.4 - Inter prediction process
"""

import logging
from typing import Tuple, Optional

import numpy as np

from inter.reference import ReferenceFrameBuffer
from inter.motion_comp import get_luma_block_fractional, get_block_integer
from inter.mv_prediction import MVCache, predict_mv_16x16

logger = logging.getLogger(__name__)


def apply_inter_prediction(
    ref_luma: np.ndarray,
    ref_x: int,
    ref_y: int,
    dx: int,
    dy: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Apply inter prediction to get a block from reference frame.

    Args:
        ref_luma: Reference frame luma component
        ref_x: Integer X position in reference
        ref_y: Integer Y position in reference
        dx: Fractional X offset (0-3 quarter-pixels)
        dy: Fractional Y offset (0-3 quarter-pixels)
        width: Block width
        height: Block height

    Returns:
        Predicted block (height x width), uint8
    """
    return get_luma_block_fractional(
        ref_luma, ref_x, ref_y, dx, dy, width, height
    )


def apply_chroma_prediction(
    ref_chroma: np.ndarray,
    ref_x: int,
    ref_y: int,
    dx: int,
    dy: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Apply inter prediction for chroma component.

    Chroma uses bilinear interpolation at 1/8-pixel precision.
    Input dx, dy are in luma quarter-pixel units.

    Args:
        ref_chroma: Reference frame chroma component
        ref_x: Integer X position (in chroma coordinates)
        ref_y: Integer Y position (in chroma coordinates)
        dx: Fractional X from luma MV (0-3)
        dy: Fractional Y from luma MV (0-3)
        width: Block width
        height: Block height

    Returns:
        Predicted chroma block
    """
    # For simplicity, use integer position
    # Full implementation would do bilinear at 1/8 pixel
    return get_block_integer(ref_chroma, ref_x, ref_y, width, height)


def reconstruct_p_skip(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct P_Skip macroblock.

    P_Skip uses:
    - ref_idx = 0 (most recent reference)
    - MV = predicted MV (no MVD transmitted)
    - No residual data

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache for prediction
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr), each reconstructed block
    """
    # Get predicted MV
    mvp_x, mvp_y = predict_mv_16x16(mv_cache, mb_x, mb_y)

    # MV is in quarter-pixel units
    # Convert to integer + fractional parts
    int_mvx = mvp_x >> 2
    int_mvy = mvp_y >> 2
    frac_x = mvp_x & 3
    frac_y = mvp_y & 3

    # Reference position in luma
    ref_x = mb_x * 16 + int_mvx
    ref_y = mb_y * 16 + int_mvy

    # Get reference frame (ref_idx = 0)
    ref_frame = ref_buffer.get_frame(0)

    # Luma prediction (16x16)
    luma = apply_inter_prediction(
        ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 16, 16
    )

    # Chroma prediction (8x8 each)
    # Chroma MV is half of luma MV
    chroma_mvx = mvp_x >> 1
    chroma_mvy = mvp_y >> 1
    int_cmvx = chroma_mvx >> 2
    int_cmvy = chroma_mvy >> 2
    frac_cx = chroma_mvx & 3
    frac_cy = chroma_mvy & 3

    ref_cx = mb_x * 8 + int_cmvx
    ref_cy = mb_y * 8 + int_cmvy

    cb = apply_chroma_prediction(
        ref_frame.cb, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )
    cr = apply_chroma_prediction(
        ref_frame.cr, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )

    # Update MV cache with the predicted MV
    mv_cache.set_mv_16x16(mb_x, mb_y, mvp_x, mvp_y)

    logger.debug(f"P_Skip MB({mb_x},{mb_y}): MV=({mvp_x},{mvp_y})")

    return luma, cb, cr


def reconstruct_p_16x16(
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
    """Reconstruct P_L0_16x16 macroblock.

    Args:
        ref_buffer: Reference frame buffer
        ref_idx: Reference frame index
        mvx, mvy: Motion vector in quarter-pixel units
        residual_luma: 16x16 residual block or None
        residual_cb: 8x8 Cb residual or None
        residual_cr: 8x8 Cr residual or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    # MV to integer + fractional
    int_mvx = mvx >> 2
    int_mvy = mvy >> 2
    frac_x = mvx & 3
    frac_y = mvy & 3

    # Reference position
    ref_x = mb_x * 16 + int_mvx
    ref_y = mb_y * 16 + int_mvy

    # Get reference frame
    ref_frame = ref_buffer.get_frame(ref_idx)

    # Luma prediction
    pred_luma = apply_inter_prediction(
        ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 16, 16
    )

    # Add residual
    if residual_luma is not None:
        luma = np.clip(
            pred_luma.astype(np.int32) + residual_luma, 0, 255
        ).astype(np.uint8)
    else:
        luma = pred_luma

    # Chroma
    chroma_mvx = mvx >> 1
    chroma_mvy = mvy >> 1
    int_cmvx = chroma_mvx >> 2
    int_cmvy = chroma_mvy >> 2
    frac_cx = chroma_mvx & 3
    frac_cy = chroma_mvy & 3

    ref_cx = mb_x * 8 + int_cmvx
    ref_cy = mb_y * 8 + int_cmvy

    pred_cb = apply_chroma_prediction(
        ref_frame.cb, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )
    pred_cr = apply_chroma_prediction(
        ref_frame.cr, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )

    if residual_cb is not None:
        cb = np.clip(
            pred_cb.astype(np.int32) + residual_cb, 0, 255
        ).astype(np.uint8)
    else:
        cb = pred_cb

    if residual_cr is not None:
        cr = np.clip(
            pred_cr.astype(np.int32) + residual_cr, 0, 255
        ).astype(np.uint8)
    else:
        cr = pred_cr

    logger.debug(f"P_16x16 MB({mb_x},{mb_y}): ref={ref_idx}, MV=({mvx},{mvy})")

    return luma, cb, cr


def reconstruct_p_partition(
    ref_buffer: ReferenceFrameBuffer,
    ref_idx: int,
    mvx: int,
    mvy: int,
    mb_x: int,
    mb_y: int,
    part_x: int,
    part_y: int,
    width: int,
    height: int,
    residual_luma: Optional[np.ndarray],
) -> np.ndarray:
    """Reconstruct a single partition within a P-macroblock.

    Args:
        ref_buffer: Reference frame buffer
        ref_idx: Reference frame index
        mvx, mvy: Motion vector in quarter-pixel units
        mb_x, mb_y: Macroblock position in frame
        part_x, part_y: Partition offset within macroblock
        width, height: Partition size
        residual_luma: Residual block or None

    Returns:
        Reconstructed partition (height x width), uint8
    """
    int_mvx = mvx >> 2
    int_mvy = mvy >> 2
    frac_x = mvx & 3
    frac_y = mvy & 3

    # Absolute reference position
    ref_x = mb_x * 16 + part_x + int_mvx
    ref_y = mb_y * 16 + part_y + int_mvy

    ref_frame = ref_buffer.get_frame(ref_idx)

    pred = apply_inter_prediction(
        ref_frame.luma, ref_x, ref_y, frac_x, frac_y, width, height
    )

    if residual_luma is not None:
        result = np.clip(
            pred.astype(np.int32) + residual_luma, 0, 255
        ).astype(np.uint8)
    else:
        result = pred

    return result
