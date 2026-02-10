from __future__ import annotations

from typing import List


def decode_cbp_monochrome(cbp_value: int) -> int:
    return cbp_value & 0x0F


def get_cbp_table_for_chroma_format(chroma_format_idc: int) -> List[int]:
    if chroma_format_idc not in (0, 1, 2, 3):
        raise ValueError(f"Invalid chroma_format_idc: {chroma_format_idc}")
    return list(range(48))
