# h264/decoder/__init__.py
"""H.264 Baseline profile decoder.

Main decoder module that orchestrates all decoding stages.

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

__all__ = [
    "H264Decoder",
    "DecodedFrame",
    "DecoderState",
    "decode_h264_file",
    "decode_h264_bytes",
]
