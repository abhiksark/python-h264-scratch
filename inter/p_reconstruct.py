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
from inter.motion_comp import (
    get_luma_block_fractional,
    get_block_integer,
    get_chroma_block_fractional,
)
from inter.mv_prediction import MVCache, predict_mv_skip

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
    Input dx, dy are in chroma eighth-pixel units (0-7).

    Args:
        ref_chroma: Reference frame chroma component
        ref_x: Integer X position (in chroma coordinates)
        ref_y: Integer Y position (in chroma coordinates)
        dx: Fractional X offset (0-7, in eighth-pixels)
        dy: Fractional Y offset (0-7, in eighth-pixels)
        width: Block width
        height: Block height

    Returns:
        Predicted chroma block

    H.264 Spec: Section 8.4.2.2.2
    """
    return get_chroma_block_fractional(
        ref_chroma, ref_x, ref_y, dx, dy, width, height
    )


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
    # Get predicted MV (H.264 8.4.1.2 — P_Skip special case)
    mvp_x, mvp_y = predict_mv_skip(mv_cache, mb_x, mb_y)

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
    # For 4:2:0, quarter-luma-pixel = eighth-chroma-pixel, so MV value is unchanged
    chroma_mvx = mvp_x
    chroma_mvy = mvp_y
    int_cmvx = chroma_mvx >> 3
    int_cmvy = chroma_mvy >> 3
    frac_cx = chroma_mvx & 7
    frac_cy = chroma_mvy & 7

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

    # Chroma: quarter-luma-pixel = eighth-chroma-pixel, MV unchanged
    chroma_mvx = mvx
    chroma_mvy = mvy
    int_cmvx = chroma_mvx >> 3
    int_cmvy = chroma_mvy >> 3
    frac_cx = chroma_mvx & 7
    frac_cy = chroma_mvy & 7

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


def reconstruct_p_16x8(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    ref_idx: list,
    mvx: list,
    mvy: list,
    residual_luma: Optional[list],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct P_L0_L0_16x8 macroblock (two 16x8 partitions).

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache for prediction and storage
        ref_idx: List of 2 reference frame indices [top, bottom]
        mvx, mvy: Lists of 2 motion vectors in quarter-pixel units
        residual_luma: List of 2 residual blocks (8x16 each) or None
        residual_cb, residual_cr: 8x8 chroma residuals or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    luma = np.zeros((16, 16), dtype=np.uint8)

    # Partition 0: top 16x8 (rows 0-7)
    part0 = reconstruct_p_partition(
        ref_buffer=ref_buffer,
        ref_idx=ref_idx[0],
        mvx=mvx[0],
        mvy=mvy[0],
        mb_x=mb_x,
        mb_y=mb_y,
        part_x=0,
        part_y=0,
        width=16,
        height=8,
        residual_luma=residual_luma[0] if residual_luma else None,
    )
    luma[0:8, :] = part0

    # Partition 1: bottom 16x8 (rows 8-15)
    part1 = reconstruct_p_partition(
        ref_buffer=ref_buffer,
        ref_idx=ref_idx[1],
        mvx=mvx[1],
        mvy=mvy[1],
        mb_x=mb_x,
        mb_y=mb_y,
        part_x=0,
        part_y=8,
        width=16,
        height=8,
        residual_luma=residual_luma[1] if residual_luma else None,
    )
    luma[8:16, :] = part1

    # Chroma reconstruction: per-partition MVs (top 4 rows / bottom 4 rows)
    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)
    for part in range(2):
        ref_frame = ref_buffer.get_frame(ref_idx[part])
        cmvx = mvx[part]
        cmvy = mvy[part]
        cx = mb_x * 8 + (cmvx >> 3)
        cy = mb_y * 8 + part * 4 + (cmvy >> 3)
        fx, fy = cmvx & 7, cmvy & 7
        pred_cb[part*4:(part+1)*4, :] = apply_chroma_prediction(
            ref_frame.cb, cx, cy, fx, fy, 8, 4
        )
        pred_cr[part*4:(part+1)*4, :] = apply_chroma_prediction(
            ref_frame.cr, cx, cy, fx, fy, 8, 4
        )

    if residual_cb is not None:
        cb = np.clip(pred_cb.astype(np.int32) + residual_cb, 0, 255).astype(np.uint8)
    else:
        cb = pred_cb
    if residual_cr is not None:
        cr = np.clip(pred_cr.astype(np.int32) + residual_cr, 0, 255).astype(np.uint8)
    else:
        cr = pred_cr

    logger.debug(f"P_16x8 MB({mb_x},{mb_y}): MVs=[({mvx[0]},{mvy[0]}),({mvx[1]},{mvy[1]})]")

    return luma, cb, cr


def reconstruct_p_8x16(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    ref_idx: list,
    mvx: list,
    mvy: list,
    residual_luma: Optional[list],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct P_L0_L0_8x16 macroblock (two 8x16 partitions).

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache for prediction and storage
        ref_idx: List of 2 reference frame indices [left, right]
        mvx, mvy: Lists of 2 motion vectors in quarter-pixel units
        residual_luma: List of 2 residual blocks (16x8 each) or None
        residual_cb, residual_cr: 8x8 chroma residuals or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    luma = np.zeros((16, 16), dtype=np.uint8)

    # Partition 0: left 8x16 (cols 0-7)
    part0 = reconstruct_p_partition(
        ref_buffer=ref_buffer,
        ref_idx=ref_idx[0],
        mvx=mvx[0],
        mvy=mvy[0],
        mb_x=mb_x,
        mb_y=mb_y,
        part_x=0,
        part_y=0,
        width=8,
        height=16,
        residual_luma=residual_luma[0] if residual_luma else None,
    )
    luma[:, 0:8] = part0

    # Partition 1: right 8x16 (cols 8-15)
    part1 = reconstruct_p_partition(
        ref_buffer=ref_buffer,
        ref_idx=ref_idx[1],
        mvx=mvx[1],
        mvy=mvy[1],
        mb_x=mb_x,
        mb_y=mb_y,
        part_x=8,
        part_y=0,
        width=8,
        height=16,
        residual_luma=residual_luma[1] if residual_luma else None,
    )
    luma[:, 8:16] = part1

    # Chroma reconstruction: per-partition MVs (left 4 cols / right 4 cols)
    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)
    for part in range(2):
        ref_frame = ref_buffer.get_frame(ref_idx[part])
        cmvx = mvx[part]
        cmvy = mvy[part]
        cx = mb_x * 8 + part * 4 + (cmvx >> 3)
        cy = mb_y * 8 + (cmvy >> 3)
        fx, fy = cmvx & 7, cmvy & 7
        pred_cb[:, part*4:(part+1)*4] = apply_chroma_prediction(
            ref_frame.cb, cx, cy, fx, fy, 4, 8
        )
        pred_cr[:, part*4:(part+1)*4] = apply_chroma_prediction(
            ref_frame.cr, cx, cy, fx, fy, 4, 8
        )

    if residual_cb is not None:
        cb = np.clip(pred_cb.astype(np.int32) + residual_cb, 0, 255).astype(np.uint8)
    else:
        cb = pred_cb
    if residual_cr is not None:
        cr = np.clip(pred_cr.astype(np.int32) + residual_cr, 0, 255).astype(np.uint8)
    else:
        cr = pred_cr

    logger.debug(f"P_8x16 MB({mb_x},{mb_y}): MVs=[({mvx[0]},{mvy[0]}),({mvx[1]},{mvy[1]})]")

    return luma, cb, cr


def reconstruct_p_8x8(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    ref_idx: list,
    mvx: list,
    mvy: list,
    sub_mb_types: list,
    residual_luma: Optional[list],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct P_8x8 macroblock (four 8x8 sub-macroblocks).

    Sub-MB layout:
        0 | 1
        -----
        2 | 3

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache for prediction and storage
        ref_idx: List of 4 reference frame indices
        mvx, mvy: Lists of 4 motion vectors in quarter-pixel units
        sub_mb_types: List of 4 sub-MB types (0=8x8, 1=8x4, 2=4x8, 3=4x4)
        residual_luma: List of 4 residual blocks (8x8 each) or None
        residual_cb, residual_cr: 8x8 chroma residuals or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    luma = np.zeros((16, 16), dtype=np.uint8)

    # Sub-MB positions within the macroblock
    sub_mb_offsets = [(0, 0), (8, 0), (0, 8), (8, 8)]

    for sub_idx in range(4):
        part_x, part_y = sub_mb_offsets[sub_idx]

        # For now, only handle sub_mb_type 0 (8x8)
        part = reconstruct_p_partition(
            ref_buffer=ref_buffer,
            ref_idx=ref_idx[sub_idx],
            mvx=mvx[sub_idx],
            mvy=mvy[sub_idx],
            mb_x=mb_x,
            mb_y=mb_y,
            part_x=part_x,
            part_y=part_y,
            width=8,
            height=8,
            residual_luma=residual_luma[sub_idx] if residual_luma else None,
        )
        luma[part_y:part_y+8, part_x:part_x+8] = part

    # Chroma reconstruction: per-sub-MB MVs (each sub-MB → 4x4 chroma quadrant)
    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)
    # Sub-MB chroma offsets: (cx_off, cy_off) within 8x8 chroma
    chroma_offsets = [(0, 0), (4, 0), (0, 4), (4, 4)]
    for sub_idx in range(4):
        ref_frame = ref_buffer.get_frame(ref_idx[sub_idx])
        cx_off, cy_off = chroma_offsets[sub_idx]
        cmvx = mvx[sub_idx]
        cmvy = mvy[sub_idx]
        cx = mb_x * 8 + cx_off + (cmvx >> 3)
        cy = mb_y * 8 + cy_off + (cmvy >> 3)
        fx, fy = cmvx & 7, cmvy & 7
        pred_cb[cy_off:cy_off+4, cx_off:cx_off+4] = apply_chroma_prediction(
            ref_frame.cb, cx, cy, fx, fy, 4, 4
        )
        pred_cr[cy_off:cy_off+4, cx_off:cx_off+4] = apply_chroma_prediction(
            ref_frame.cr, cx, cy, fx, fy, 4, 4
        )

    if residual_cb is not None:
        cb = np.clip(pred_cb.astype(np.int32) + residual_cb, 0, 255).astype(np.uint8)
    else:
        cb = pred_cb
    if residual_cr is not None:
        cr = np.clip(pred_cr.astype(np.int32) + residual_cr, 0, 255).astype(np.uint8)
    else:
        cr = pred_cr

    logger.debug(f"P_8x8 MB({mb_x},{mb_y}): MVs={list(zip(mvx, mvy))}")

    return luma, cb, cr


def reconstruct_p_8x8_sub(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    ref_idx: int,
    sub_mb_type: int,
    mvs: list,
    mb_x: int,
    mb_y: int,
    sub_idx: int,
    residual: Optional[np.ndarray],
) -> np.ndarray:
    """Reconstruct a single 8x8 sub-macroblock with sub-partitions.

    Sub-MB types:
        0: 8x8 (1 partition, 1 MV)
        1: 8x4 (2 partitions, 2 MVs - top/bottom)
        2: 4x8 (2 partitions, 2 MVs - left/right)
        3: 4x4 (4 partitions, 4 MVs)

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache
        ref_idx: Reference frame index
        sub_mb_type: Sub-MB partition type (0-3)
        mvs: List of MVs for this sub-MB (1, 2, or 4 depending on type)
        mb_x, mb_y: Macroblock position
        sub_idx: Sub-MB index within macroblock (0-3)
        residual: 8x8 residual or None

    Returns:
        8x8 reconstructed block
    """
    # Sub-MB position within macroblock
    sub_offsets = [(0, 0), (8, 0), (0, 8), (8, 8)]
    sub_x, sub_y = sub_offsets[sub_idx]

    result = np.zeros((8, 8), dtype=np.uint8)
    ref_frame = ref_buffer.get_frame(ref_idx)

    if sub_mb_type == 0:
        # 8x8: single partition
        mvx, mvy = mvs[0]
        int_mvx, int_mvy = mvx >> 2, mvy >> 2
        frac_x, frac_y = mvx & 3, mvy & 3
        ref_x = mb_x * 16 + sub_x + int_mvx
        ref_y = mb_y * 16 + sub_y + int_mvy

        result = apply_inter_prediction(
            ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 8, 8
        )

    elif sub_mb_type == 1:
        # 8x4: two horizontal partitions
        for part in range(2):
            mvx, mvy = mvs[part]
            int_mvx, int_mvy = mvx >> 2, mvy >> 2
            frac_x, frac_y = mvx & 3, mvy & 3
            ref_x = mb_x * 16 + sub_x + int_mvx
            ref_y = mb_y * 16 + sub_y + part * 4 + int_mvy

            part_result = apply_inter_prediction(
                ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 8, 4
            )
            result[part * 4:(part + 1) * 4, :] = part_result

    elif sub_mb_type == 2:
        # 4x8: two vertical partitions
        for part in range(2):
            mvx, mvy = mvs[part]
            int_mvx, int_mvy = mvx >> 2, mvy >> 2
            frac_x, frac_y = mvx & 3, mvy & 3
            ref_x = mb_x * 16 + sub_x + part * 4 + int_mvx
            ref_y = mb_y * 16 + sub_y + int_mvy

            part_result = apply_inter_prediction(
                ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 4, 8
            )
            result[:, part * 4:(part + 1) * 4] = part_result

    elif sub_mb_type == 3:
        # 4x4: four partitions
        part_offsets = [(0, 0), (4, 0), (0, 4), (4, 4)]
        for part, (px, py) in enumerate(part_offsets):
            mvx, mvy = mvs[part]
            int_mvx, int_mvy = mvx >> 2, mvy >> 2
            frac_x, frac_y = mvx & 3, mvy & 3
            ref_x = mb_x * 16 + sub_x + px + int_mvx
            ref_y = mb_y * 16 + sub_y + py + int_mvy

            part_result = apply_inter_prediction(
                ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 4, 4
            )
            result[py:py + 4, px:px + 4] = part_result

    # Add residual if present
    if residual is not None:
        result = np.clip(
            result.astype(np.int32) + residual, 0, 255
        ).astype(np.uint8)

    return result


def update_mv_cache_sub_mb(
    mv_cache: MVCache,
    mb_x: int,
    mb_y: int,
    sub_idx: int,
    sub_mb_type: int,
    mvs: list,
) -> None:
    """Update MV cache for a sub-macroblock with sub-partitions.

    Args:
        mv_cache: MV cache to update
        mb_x, mb_y: Macroblock position
        sub_idx: Sub-MB index (0-3)
        sub_mb_type: Sub-MB partition type (0-3)
        mvs: List of MVs for this sub-MB
    """
    # Sub-MB base position in 4x4 blocks within macroblock
    sub_bx = (sub_idx % 2) * 2
    sub_by = (sub_idx // 2) * 2

    if sub_mb_type == 0:
        # 8x8: all 4 blocks get same MV
        mvx, mvy = mvs[0]
        for dy in range(2):
            for dx in range(2):
                mv_cache.set_mv(mb_x, mb_y, sub_bx + dx, sub_by + dy, mvx, mvy)

    elif sub_mb_type == 1:
        # 8x4: top row gets mvs[0], bottom row gets mvs[1]
        mvx0, mvy0 = mvs[0]
        mvx1, mvy1 = mvs[1]
        for dx in range(2):
            mv_cache.set_mv(mb_x, mb_y, sub_bx + dx, sub_by, mvx0, mvy0)
            mv_cache.set_mv(mb_x, mb_y, sub_bx + dx, sub_by + 1, mvx1, mvy1)

    elif sub_mb_type == 2:
        # 4x8: left col gets mvs[0], right col gets mvs[1]
        mvx0, mvy0 = mvs[0]
        mvx1, mvy1 = mvs[1]
        for dy in range(2):
            mv_cache.set_mv(mb_x, mb_y, sub_bx, sub_by + dy, mvx0, mvy0)
            mv_cache.set_mv(mb_x, mb_y, sub_bx + 1, sub_by + dy, mvx1, mvy1)

    elif sub_mb_type == 3:
        # 4x4: each block gets its own MV
        block_offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for part, (dx, dy) in enumerate(block_offsets):
            mvx, mvy = mvs[part]
            mv_cache.set_mv(mb_x, mb_y, sub_bx + dx, sub_by + dy, mvx, mvy)


def _reconstruct_chroma_16x16(
    ref_buffer: ReferenceFrameBuffer,
    ref_idx: int,
    mvx: int,
    mvy: int,
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct chroma components for a P-macroblock.

    Args:
        ref_buffer: Reference frame buffer
        ref_idx: Reference frame index
        mvx, mvy: Motion vector (uses first partition's MV for simplicity)
        residual_cb, residual_cr: 8x8 chroma residuals or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (cb, cr) reconstructed blocks
    """
    ref_frame = ref_buffer.get_frame(ref_idx)

    # For 4:2:0, quarter-luma-pixel = eighth-chroma-pixel, MV unchanged
    chroma_mvx = mvx
    chroma_mvy = mvy
    int_cmvx = chroma_mvx >> 3
    int_cmvy = chroma_mvy >> 3
    frac_cx = chroma_mvx & 7
    frac_cy = chroma_mvy & 7

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

    return cb, cr


def reconstruct_p_16x16_weighted(
    ref_buffer: ReferenceFrameBuffer,
    ref_idx: int,
    mvx: int,
    mvy: int,
    weight_luma: int,
    offset_luma: int,
    log2_denom_luma: int,
    weight_cb: int,
    offset_cb: int,
    weight_cr: int,
    offset_cr: int,
    log2_denom_chroma: int,
    residual_luma: Optional[np.ndarray],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct P_L0_16x16 macroblock with weighted prediction.

    Args:
        ref_buffer: Reference frame buffer
        ref_idx: Reference frame index
        mvx, mvy: Motion vector in quarter-pixel units
        weight_luma: Luma weight
        offset_luma: Luma offset
        log2_denom_luma: Luma weight denominator
        weight_cb: Cb weight
        offset_cb: Cb offset
        weight_cr: Cr weight
        offset_cr: Cr offset
        log2_denom_chroma: Chroma weight denominator
        residual_luma: 16x16 residual block or None
        residual_cb: 8x8 Cb residual or None
        residual_cr: 8x8 Cr residual or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    from inter.weighted_pred import apply_weighted_prediction, apply_weighted_prediction_chroma

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

    # Luma prediction with motion compensation
    pred_luma = apply_inter_prediction(
        ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 16, 16
    )

    # Apply weighted prediction to luma
    weighted_luma = apply_weighted_prediction(
        pred_luma, weight_luma, offset_luma, log2_denom_luma
    )

    # Add residual
    if residual_luma is not None:
        luma = np.clip(
            weighted_luma.astype(np.int32) + residual_luma, 0, 255
        ).astype(np.uint8)
    else:
        luma = weighted_luma

    # Chroma prediction: quarter-luma-pixel = eighth-chroma-pixel, MV unchanged
    chroma_mvx = mvx
    chroma_mvy = mvy
    int_cmvx = chroma_mvx >> 3
    int_cmvy = chroma_mvy >> 3
    frac_cx = chroma_mvx & 7
    frac_cy = chroma_mvy & 7

    ref_cx = mb_x * 8 + int_cmvx
    ref_cy = mb_y * 8 + int_cmvy

    pred_cb = apply_chroma_prediction(
        ref_frame.cb, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )
    pred_cr = apply_chroma_prediction(
        ref_frame.cr, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )

    # Apply weighted prediction to chroma
    weighted_cb, weighted_cr = apply_weighted_prediction_chroma(
        pred_cb, pred_cr,
        weight_cb, offset_cb,
        weight_cr, offset_cr,
        log2_denom_chroma,
    )

    # Add residual
    if residual_cb is not None:
        cb = np.clip(
            weighted_cb.astype(np.int32) + residual_cb, 0, 255
        ).astype(np.uint8)
    else:
        cb = weighted_cb

    if residual_cr is not None:
        cr = np.clip(
            weighted_cr.astype(np.int32) + residual_cr, 0, 255
        ).astype(np.uint8)
    else:
        cr = weighted_cr

    logger.debug(
        f"P_16x16_weighted MB({mb_x},{mb_y}): "
        f"ref={ref_idx}, MV=({mvx},{mvy}), w={weight_luma}"
    )

    return luma, cb, cr


def reconstruct_p_skip_weighted(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    mb_x: int,
    mb_y: int,
    weight_luma: int,
    offset_luma: int,
    log2_denom_luma: int,
    weight_cb: int,
    offset_cb: int,
    weight_cr: int,
    offset_cr: int,
    log2_denom_chroma: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct P_Skip macroblock with weighted prediction.

    P_Skip uses predicted MV and ref_idx=0, but weighted prediction
    still applies when weighted_pred_flag=1 in PPS.

    H.264 Spec: Section 8.4.2.3.1

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache for prediction
        mb_x, mb_y: Macroblock position
        weight_luma: Luma weight
        offset_luma: Luma offset
        log2_denom_luma: Luma weight denominator
        weight_cb, offset_cb: Cb weight and offset
        weight_cr, offset_cr: Cr weight and offset
        log2_denom_chroma: Chroma weight denominator

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    from inter.weighted_pred import apply_weighted_prediction, apply_weighted_prediction_chroma

    # Get predicted MV (H.264 8.4.1.2 — P_Skip special case)
    mvp_x, mvp_y = predict_mv_skip(mv_cache, mb_x, mb_y)

    # MV to integer + fractional
    int_mvx = mvp_x >> 2
    int_mvy = mvp_y >> 2
    frac_x = mvp_x & 3
    frac_y = mvp_y & 3

    ref_x = mb_x * 16 + int_mvx
    ref_y = mb_y * 16 + int_mvy

    # P_Skip always uses ref_idx=0
    ref_frame = ref_buffer.get_frame(0)

    # Luma prediction
    pred_luma = apply_inter_prediction(
        ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 16, 16
    )

    # Apply weighted prediction to luma
    luma = apply_weighted_prediction(
        pred_luma, weight_luma, offset_luma, log2_denom_luma
    )

    # Chroma prediction: quarter-luma-pixel = eighth-chroma-pixel, MV unchanged
    chroma_mvx = mvp_x
    chroma_mvy = mvp_y
    int_cmvx = chroma_mvx >> 3
    int_cmvy = chroma_mvy >> 3
    frac_cx = chroma_mvx & 7
    frac_cy = chroma_mvy & 7

    ref_cx = mb_x * 8 + int_cmvx
    ref_cy = mb_y * 8 + int_cmvy

    pred_cb = apply_chroma_prediction(
        ref_frame.cb, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )
    pred_cr = apply_chroma_prediction(
        ref_frame.cr, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )

    # Apply weighted prediction to chroma
    cb, cr = apply_weighted_prediction_chroma(
        pred_cb, pred_cr,
        weight_cb, offset_cb,
        weight_cr, offset_cr,
        log2_denom_chroma,
    )

    # Update MV cache
    mv_cache.set_mv_16x16(mb_x, mb_y, mvp_x, mvp_y)

    logger.debug(
        f"P_Skip_weighted MB({mb_x},{mb_y}): MV=({mvp_x},{mvp_y}), w={weight_luma}"
    )

    return luma, cb, cr


def reconstruct_p_16x8_weighted(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    ref_idx: list,
    mvx: list,
    mvy: list,
    weights_luma: list,
    offsets_luma: list,
    log2_denom_luma: int,
    weights_cb: list,
    offsets_cb: list,
    weights_cr: list,
    offsets_cr: list,
    log2_denom_chroma: int,
    residual_luma: Optional[list],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct P_L0_L0_16x8 macroblock with weighted prediction.

    Each 16x8 partition can use different reference frames and weights.

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache
        ref_idx: List of 2 reference indices [top, bottom]
        mvx, mvy: Lists of 2 motion vectors
        weights_luma: List of 2 luma weights
        offsets_luma: List of 2 luma offsets
        log2_denom_luma: Luma weight denominator
        weights_cb, offsets_cb: Cb weights/offsets for each partition
        weights_cr, offsets_cr: Cr weights/offsets for each partition
        log2_denom_chroma: Chroma weight denominator
        residual_luma: List of 2 residual blocks or None
        residual_cb, residual_cr: 8x8 chroma residuals or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    from inter.weighted_pred import apply_weighted_prediction

    luma = np.zeros((16, 16), dtype=np.uint8)

    # Process each 16x8 partition
    for part in range(2):
        part_y = part * 8

        ref_frame = ref_buffer.get_frame(ref_idx[part])

        # MV to integer + fractional
        int_mvx = mvx[part] >> 2
        int_mvy = mvy[part] >> 2
        frac_x = mvx[part] & 3
        frac_y = mvy[part] & 3

        ref_x = mb_x * 16 + int_mvx
        ref_y = mb_y * 16 + part_y + int_mvy

        # Prediction
        pred = apply_inter_prediction(
            ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 16, 8
        )

        # Apply weighted prediction
        weighted = apply_weighted_prediction(
            pred, weights_luma[part], offsets_luma[part], log2_denom_luma
        )

        # Add residual
        if residual_luma is not None and residual_luma[part] is not None:
            weighted = np.clip(
                weighted.astype(np.int32) + residual_luma[part], 0, 255
            ).astype(np.uint8)

        luma[part_y:part_y + 8, :] = weighted

    # Chroma reconstruction: per-partition MVs (top 4 rows / bottom 4 rows)
    from inter.weighted_pred import apply_weighted_prediction_chroma

    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)
    for part in range(2):
        ref_frame = ref_buffer.get_frame(ref_idx[part])
        cmvx = mvx[part]
        cmvy = mvy[part]
        cx = mb_x * 8 + (cmvx >> 3)
        cy = mb_y * 8 + part * 4 + (cmvy >> 3)
        fx, fy = cmvx & 7, cmvy & 7
        part_cb = apply_chroma_prediction(
            ref_frame.cb, cx, cy, fx, fy, 8, 4
        )
        part_cr = apply_chroma_prediction(
            ref_frame.cr, cx, cy, fx, fy, 8, 4
        )
        part_cb, part_cr = apply_weighted_prediction_chroma(
            part_cb, part_cr,
            weights_cb[part], offsets_cb[part],
            weights_cr[part], offsets_cr[part],
            log2_denom_chroma,
        )
        pred_cb[part*4:(part+1)*4, :] = part_cb
        pred_cr[part*4:(part+1)*4, :] = part_cr

    if residual_cb is not None:
        cb = np.clip(pred_cb.astype(np.int32) + residual_cb, 0, 255).astype(np.uint8)
    else:
        cb = pred_cb
    if residual_cr is not None:
        cr = np.clip(pred_cr.astype(np.int32) + residual_cr, 0, 255).astype(np.uint8)
    else:
        cr = pred_cr

    logger.debug(f"P_16x8_weighted MB({mb_x},{mb_y})")

    return luma, cb, cr


def reconstruct_p_8x16_weighted(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    ref_idx: list,
    mvx: list,
    mvy: list,
    weights_luma: list,
    offsets_luma: list,
    log2_denom_luma: int,
    weights_cb: list,
    offsets_cb: list,
    weights_cr: list,
    offsets_cr: list,
    log2_denom_chroma: int,
    residual_luma: Optional[list],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct P_L0_L0_8x16 macroblock with weighted prediction.

    Each 8x16 partition can use different reference frames and weights.

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache
        ref_idx: List of 2 reference indices [left, right]
        mvx, mvy: Lists of 2 motion vectors
        weights_luma: List of 2 luma weights
        offsets_luma: List of 2 luma offsets
        log2_denom_luma: Luma weight denominator
        weights_cb, offsets_cb: Cb weights/offsets for each partition
        weights_cr, offsets_cr: Cr weights/offsets for each partition
        log2_denom_chroma: Chroma weight denominator
        residual_luma: List of 2 residual blocks or None
        residual_cb, residual_cr: 8x8 chroma residuals or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    from inter.weighted_pred import apply_weighted_prediction

    luma = np.zeros((16, 16), dtype=np.uint8)

    # Process each 8x16 partition
    for part in range(2):
        part_x = part * 8

        ref_frame = ref_buffer.get_frame(ref_idx[part])

        # MV to integer + fractional
        int_mvx = mvx[part] >> 2
        int_mvy = mvy[part] >> 2
        frac_x = mvx[part] & 3
        frac_y = mvy[part] & 3

        ref_x = mb_x * 16 + part_x + int_mvx
        ref_y = mb_y * 16 + int_mvy

        # Prediction
        pred = apply_inter_prediction(
            ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 8, 16
        )

        # Apply weighted prediction
        weighted = apply_weighted_prediction(
            pred, weights_luma[part], offsets_luma[part], log2_denom_luma
        )

        # Add residual
        if residual_luma is not None and residual_luma[part] is not None:
            weighted = np.clip(
                weighted.astype(np.int32) + residual_luma[part], 0, 255
            ).astype(np.uint8)

        luma[:, part_x:part_x + 8] = weighted

    # Chroma reconstruction: per-partition MVs (left 4 cols / right 4 cols)
    from inter.weighted_pred import apply_weighted_prediction_chroma

    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)
    for part in range(2):
        ref_frame = ref_buffer.get_frame(ref_idx[part])
        cmvx = mvx[part]
        cmvy = mvy[part]
        cx = mb_x * 8 + part * 4 + (cmvx >> 3)
        cy = mb_y * 8 + (cmvy >> 3)
        fx, fy = cmvx & 7, cmvy & 7
        part_cb = apply_chroma_prediction(
            ref_frame.cb, cx, cy, fx, fy, 4, 8
        )
        part_cr = apply_chroma_prediction(
            ref_frame.cr, cx, cy, fx, fy, 4, 8
        )
        # Apply weighted prediction per partition
        part_cb, part_cr = apply_weighted_prediction_chroma(
            part_cb, part_cr,
            weights_cb[part], offsets_cb[part],
            weights_cr[part], offsets_cr[part],
            log2_denom_chroma,
        )
        pred_cb[:, part*4:(part+1)*4] = part_cb
        pred_cr[:, part*4:(part+1)*4] = part_cr

    if residual_cb is not None:
        cb = np.clip(pred_cb.astype(np.int32) + residual_cb, 0, 255).astype(np.uint8)
    else:
        cb = pred_cb
    if residual_cr is not None:
        cr = np.clip(pred_cr.astype(np.int32) + residual_cr, 0, 255).astype(np.uint8)
    else:
        cr = pred_cr

    logger.debug(f"P_8x16_weighted MB({mb_x},{mb_y})")

    return luma, cb, cr


def reconstruct_p_8x8_weighted(
    ref_buffer: ReferenceFrameBuffer,
    mv_cache: MVCache,
    ref_idx: list,
    mvx: list,
    mvy: list,
    sub_mb_types: list,
    weights_luma: list,
    offsets_luma: list,
    log2_denom_luma: int,
    weights_cb: list,
    offsets_cb: list,
    weights_cr: list,
    offsets_cr: list,
    log2_denom_chroma: int,
    residual_luma: Optional[list],
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct P_8x8 macroblock with weighted prediction.

    Each 8x8 sub-macroblock can use different reference frames and weights.

    Args:
        ref_buffer: Reference frame buffer
        mv_cache: MV cache
        ref_idx: List of 4 reference indices
        mvx, mvy: Lists of 4 motion vectors
        sub_mb_types: List of 4 sub-MB types
        weights_luma: List of 4 luma weights
        offsets_luma: List of 4 luma offsets
        log2_denom_luma: Luma weight denominator
        weights_cb, offsets_cb: Cb weights/offsets for each sub-MB
        weights_cr, offsets_cr: Cr weights/offsets for each sub-MB
        log2_denom_chroma: Chroma weight denominator
        residual_luma: List of 4 residual blocks or None
        residual_cb, residual_cr: 8x8 chroma residuals or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (luma, cb, cr) reconstructed blocks
    """
    from inter.weighted_pred import apply_weighted_prediction

    luma = np.zeros((16, 16), dtype=np.uint8)

    # Sub-MB offsets
    sub_offsets = [(0, 0), (8, 0), (0, 8), (8, 8)]

    for sub_idx in range(4):
        part_x, part_y = sub_offsets[sub_idx]

        ref_frame = ref_buffer.get_frame(ref_idx[sub_idx])

        # MV to integer + fractional
        int_mvx = mvx[sub_idx] >> 2
        int_mvy = mvy[sub_idx] >> 2
        frac_x = mvx[sub_idx] & 3
        frac_y = mvy[sub_idx] & 3

        ref_x = mb_x * 16 + part_x + int_mvx
        ref_y = mb_y * 16 + part_y + int_mvy

        # Prediction (simplified: assume 8x8 sub-MB type)
        pred = apply_inter_prediction(
            ref_frame.luma, ref_x, ref_y, frac_x, frac_y, 8, 8
        )

        # Apply weighted prediction
        weighted = apply_weighted_prediction(
            pred, weights_luma[sub_idx], offsets_luma[sub_idx], log2_denom_luma
        )

        # Add residual
        if residual_luma is not None and residual_luma[sub_idx] is not None:
            weighted = np.clip(
                weighted.astype(np.int32) + residual_luma[sub_idx], 0, 255
            ).astype(np.uint8)

        luma[part_y:part_y + 8, part_x:part_x + 8] = weighted

    # Chroma reconstruction: per-sub-MB MVs (4 quadrants of 4x4 chroma)
    from inter.weighted_pred import apply_weighted_prediction_chroma

    pred_cb = np.zeros((8, 8), dtype=np.uint8)
    pred_cr = np.zeros((8, 8), dtype=np.uint8)
    chroma_offsets = [(0, 0), (4, 0), (0, 4), (4, 4)]
    for sub_idx in range(4):
        coff_x, coff_y = chroma_offsets[sub_idx]
        ref_frame = ref_buffer.get_frame(ref_idx[sub_idx])
        cmvx = mvx[sub_idx]
        cmvy = mvy[sub_idx]
        cx = mb_x * 8 + coff_x + (cmvx >> 3)
        cy = mb_y * 8 + coff_y + (cmvy >> 3)
        fx, fy = cmvx & 7, cmvy & 7
        part_cb = apply_chroma_prediction(
            ref_frame.cb, cx, cy, fx, fy, 4, 4
        )
        part_cr = apply_chroma_prediction(
            ref_frame.cr, cx, cy, fx, fy, 4, 4
        )
        part_cb, part_cr = apply_weighted_prediction_chroma(
            part_cb, part_cr,
            weights_cb[sub_idx], offsets_cb[sub_idx],
            weights_cr[sub_idx], offsets_cr[sub_idx],
            log2_denom_chroma,
        )
        pred_cb[coff_y:coff_y+4, coff_x:coff_x+4] = part_cb
        pred_cr[coff_y:coff_y+4, coff_x:coff_x+4] = part_cr

    if residual_cb is not None:
        cb = np.clip(pred_cb.astype(np.int32) + residual_cb, 0, 255).astype(np.uint8)
    else:
        cb = pred_cb
    if residual_cr is not None:
        cr = np.clip(pred_cr.astype(np.int32) + residual_cr, 0, 255).astype(np.uint8)
    else:
        cr = pred_cr

    logger.debug(f"P_8x8_weighted MB({mb_x},{mb_y})")

    return luma, cb, cr


def _reconstruct_chroma_weighted(
    ref_buffer: ReferenceFrameBuffer,
    ref_idx: int,
    mvx: int,
    mvy: int,
    weight_cb: int,
    offset_cb: int,
    weight_cr: int,
    offset_cr: int,
    log2_denom: int,
    residual_cb: Optional[np.ndarray],
    residual_cr: Optional[np.ndarray],
    mb_x: int,
    mb_y: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct chroma with weighted prediction.

    Helper function for weighted P-MB reconstruction.

    Args:
        ref_buffer: Reference frame buffer
        ref_idx: Reference frame index
        mvx, mvy: Motion vector
        weight_cb, offset_cb: Cb weight and offset
        weight_cr, offset_cr: Cr weight and offset
        log2_denom: Chroma weight denominator
        residual_cb, residual_cr: Chroma residuals or None
        mb_x, mb_y: Macroblock position

    Returns:
        Tuple of (cb, cr) reconstructed blocks
    """
    from inter.weighted_pred import apply_weighted_prediction_chroma

    ref_frame = ref_buffer.get_frame(ref_idx)

    # For 4:2:0, quarter-luma-pixel = eighth-chroma-pixel, MV unchanged
    chroma_mvx = mvx
    chroma_mvy = mvy
    int_cmvx = chroma_mvx >> 3
    int_cmvy = chroma_mvy >> 3
    frac_cx = chroma_mvx & 7
    frac_cy = chroma_mvy & 7

    ref_cx = mb_x * 8 + int_cmvx
    ref_cy = mb_y * 8 + int_cmvy

    pred_cb = apply_chroma_prediction(
        ref_frame.cb, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )
    pred_cr = apply_chroma_prediction(
        ref_frame.cr, ref_cx, ref_cy, frac_cx, frac_cy, 8, 8
    )

    # Apply weighted prediction
    weighted_cb, weighted_cr = apply_weighted_prediction_chroma(
        pred_cb, pred_cr,
        weight_cb, offset_cb,
        weight_cr, offset_cr,
        log2_denom,
    )

    # Add residual
    if residual_cb is not None:
        cb = np.clip(
            weighted_cb.astype(np.int32) + residual_cb, 0, 255
        ).astype(np.uint8)
    else:
        cb = weighted_cb

    if residual_cr is not None:
        cr = np.clip(
            weighted_cr.astype(np.int32) + residual_cr, 0, 255
        ).astype(np.uint8)
    else:
        cr = weighted_cr

    return cb, cr
