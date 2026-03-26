# h264/decoder/__init__.py
"""H.264 decoder — main entry point.

H.264 Spec Reference:
- Section 7: Syntax and semantics
- Section 8: Decoding process
"""

from .decoder import (
    H264Decoder,
    DecodedFrame,
    DecoderState,
    decode_h264_file,
    decode_h264_bytes,
)

from .poc import (
    POCCalculator,
    calculate_poc,
)

__all__ = [
    "H264Decoder",
    "DecodedFrame",
    "DecoderState",
    "decode_h264_file",
    "decode_h264_bytes",
    "POCCalculator",
    "calculate_poc",
]
