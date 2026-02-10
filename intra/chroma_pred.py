from __future__ import annotations

from typing import List, Tuple

import numpy as np


def get_supported_chroma_modes(chroma_format_idc: int) -> List[int]:
    if chroma_format_idc == 0:
        return []
    if chroma_format_idc in (1, 2, 3):
        return [0, 1, 2, 3]
    raise ValueError(f"Invalid chroma_format_idc: {chroma_format_idc}")


def get_chroma_pred_block_size(chroma_format_idc: int) -> Tuple[int, int]:
    if chroma_format_idc == 0:
        return 0, 0
    if chroma_format_idc == 1:
        return 8, 8
    if chroma_format_idc == 2:
        return 8, 16
    if chroma_format_idc == 3:
        return 16, 16
    raise ValueError(f"Invalid chroma_format_idc: {chroma_format_idc}")


def predict_chroma_dc_422(
    left: np.ndarray,
    top: np.ndarray,
    left_available: bool,
    top_available: bool,
) -> np.ndarray:
    height, width = 16, 8

    if left_available and top_available:
        left_sum = int(np.sum(left, dtype=np.int32))
        top_sum = int(np.sum(top, dtype=np.int32))
        denom = int(left.size + top.size)
        dc = (left_sum + top_sum) // denom
    elif left_available:
        dc = int(np.mean(left, dtype=np.float32))
    elif top_available:
        dc = int(np.mean(top, dtype=np.float32))
    else:
        dc = 128

    return np.full((height, width), dc, dtype=np.uint8)


def predict_chroma_plane_444(
    left: np.ndarray,
    top: np.ndarray,
    top_left: np.uint8,
) -> np.ndarray:
    height, width = 16, 16

    dc = int((int(np.mean(left, dtype=np.float32)) + int(np.mean(top, dtype=np.float32)) + int(top_left)) // 3)
    return np.full((height, width), dc, dtype=np.uint8)
