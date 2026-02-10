from __future__ import annotations

from typing import Tuple


def get_chroma_dc_dimensions(chroma_format_idc: int) -> Tuple[int, int]:
    if chroma_format_idc == 0:
        return 0, 0
    if chroma_format_idc == 1:
        return 2, 2
    if chroma_format_idc == 2:
        return 2, 4
    if chroma_format_idc == 3:
        return 4, 4
    raise ValueError(f"Invalid chroma_format_idc: {chroma_format_idc}")
