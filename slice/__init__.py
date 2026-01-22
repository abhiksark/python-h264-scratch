# h264/slice/__init__.py
"""Slice-level parsing for H.264.

Handles slice header parsing which contains per-slice parameters
needed for macroblock decoding.

H.264 Spec Reference:
- Section 7.3.3: Slice header syntax
- Section 7.4.3: Slice header semantics
"""

from .slice_header import (
    SliceType,
    SliceHeader,
    RefPicListModification,
    DecRefPicMarking,
    parse_slice_header,
    parse_slice_header_weighted,
)

__all__ = [
    "SliceType",
    "SliceHeader",
    "RefPicListModification",
    "DecRefPicMarking",
    "parse_slice_header",
    "parse_slice_header_weighted",
]
