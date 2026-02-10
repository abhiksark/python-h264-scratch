from __future__ import annotations


def supports_8x8_chroma(chroma_format_idc: int, transform_8x8_mode_flag: bool) -> bool:
    return bool(transform_8x8_mode_flag and chroma_format_idc == 3)
