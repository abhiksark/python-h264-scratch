# h264/bitstream/__init__.py
"""Bitstream parsing module for H.264 Annex B format.

Handles NAL unit extraction and bit-level reading.

Two BitReader implementations available:
- NumpyBitReader: Pure NumPy, no external dependencies (default)
- BitReader: Uses bitstring library (legacy, requires pip install bitstring)
"""

from .nal_parser import (
    NALUnitType,
    NALUnit,
    START_CODE_3,
    START_CODE_4,
    find_start_codes,
    remove_emulation_prevention_bytes,
    parse_nal_header,
    parse_nal_unit,
    extract_nal_units,
    iter_nal_units,
    filter_nal_units,
    get_sps_pps,
    create_test_bitstream,
)

from .bit_reader import (
    BitReader,
    BitWriter,
    BitstringBitReader,
    BitstringBitWriter,
    BITSTRING_AVAILABLE,
    USE_NUMPY_READER,
    create_bit_reader,
)

from .numpy_bit_reader import (
    NumpyBitReader,
    NumpyBitWriter,
)

__all__ = [
    # NAL parsing
    "NALUnitType",
    "NALUnit",
    "START_CODE_3",
    "START_CODE_4",
    "find_start_codes",
    "remove_emulation_prevention_bytes",
    "parse_nal_header",
    "parse_nal_unit",
    "extract_nal_units",
    "iter_nal_units",
    "filter_nal_units",
    "get_sps_pps",
    "create_test_bitstream",
    # Bit reading - NumPy (recommended)
    "NumpyBitReader",
    "NumpyBitWriter",
    # Bit reading - bitstring (legacy)
    "BitReader",
    "BitWriter",
    "BitstringBitReader",
    "BitstringBitWriter",
    # Configuration
    "BITSTRING_AVAILABLE",
    "USE_NUMPY_READER",
    "create_bit_reader",
]
