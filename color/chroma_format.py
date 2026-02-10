from __future__ import annotations

from typing import Tuple

import numpy as np

from color.yuv_to_rgb import COLOR_MATRICES, ColorMatrix


def get_chroma_array_type(chroma_format_idc: int, separate_colour_plane_flag: bool = False) -> int:
    if separate_colour_plane_flag:
        return 0
    return 0 if chroma_format_idc == 0 else chroma_format_idc


def get_subsampling_factors(chroma_format_idc: int) -> Tuple[int, int]:
    if chroma_format_idc == 0:
        return 0, 0
    if chroma_format_idc == 1:
        return 2, 2
    if chroma_format_idc == 2:
        return 2, 1
    if chroma_format_idc == 3:
        return 1, 1
    raise ValueError(f"Invalid chroma_format_idc: {chroma_format_idc}")


def get_chroma_dimensions(
    luma_width: int,
    luma_height: int,
    chroma_format_idc: int,
    separate_colour_plane_flag: bool = False,
) -> Tuple[int, int, int, int]:
    chroma_array_type = get_chroma_array_type(chroma_format_idc, separate_colour_plane_flag)
    if chroma_array_type == 0:
        return 0, 0, 0, 0

    sub_w, sub_h = get_subsampling_factors(chroma_array_type)
    cb_w = (luma_width + sub_w - 1) // sub_w
    cb_h = (luma_height + sub_h - 1) // sub_h
    return cb_w, cb_h, cb_w, cb_h


def monochrome_to_rgb(luma: np.ndarray) -> np.ndarray:
    if luma.ndim != 2:
        raise ValueError("Expected 2D luma array")
    return np.stack([luma, luma, luma], axis=-1)


def upsample_chroma_422(chroma: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if chroma.ndim != 2:
        raise ValueError("Expected 2D chroma array")
    up = np.repeat(chroma, 2, axis=1)
    return up[:target_height, :target_width].astype(chroma.dtype)


def upsample_chroma_444(chroma: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    if chroma.ndim != 2:
        raise ValueError("Expected 2D chroma array")
    return chroma[:target_height, :target_width].astype(chroma.dtype)


def _convert_444_planes_to_rgb(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    color_matrix: ColorMatrix = ColorMatrix.BT601,
) -> np.ndarray:
    y_f = y.astype(np.float32)
    cb_f = cb.astype(np.float32) - 128.0
    cr_f = cr.astype(np.float32) - 128.0

    matrix = COLOR_MATRICES[color_matrix]

    r = y_f + cr_f * matrix["cr_to_r"]
    g = y_f + cb_f * matrix["cb_to_g"] + cr_f * matrix["cr_to_g"]
    b = y_f + cb_f * matrix["cb_to_b"]

    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    return np.stack([r, g, b], axis=-1)


def ycbcr_422_to_rgb(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    color_matrix: ColorMatrix = ColorMatrix.BT601,
) -> np.ndarray:
    height, width = y.shape
    cb_up = upsample_chroma_422(cb, target_width=width, target_height=height)
    cr_up = upsample_chroma_422(cr, target_width=width, target_height=height)
    return _convert_444_planes_to_rgb(y, cb_up, cr_up, color_matrix=color_matrix)


def ycbcr_444_to_rgb(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    color_matrix: ColorMatrix = ColorMatrix.BT601,
) -> np.ndarray:
    return _convert_444_planes_to_rgb(y, cb, cr, color_matrix=color_matrix)


def get_plane_dimensions_separate(
    luma_width: int,
    luma_height: int,
    plane_id: int,
) -> Tuple[int, int]:
    if plane_id not in (0, 1, 2):
        raise ValueError("plane_id must be 0, 1, or 2")
    return luma_width, luma_height
