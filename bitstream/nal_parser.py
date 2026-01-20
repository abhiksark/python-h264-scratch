# h264/bitstream/nal_parser.py
"""NAL (Network Abstraction Layer) unit parsing for H.264 Annex B bitstreams.

H.264 bitstreams consist of NAL units, each containing a specific type of data
(SPS, PPS, slice data, etc.). In Annex B format, NAL units are separated by
start codes.

H.264 Spec Reference:
- Section 7.3.1: NAL unit syntax
- Section 7.4.1: NAL unit semantics
- Annex B: Byte stream format

NAL Unit Types (Table 7-1):
- 1: Non-IDR slice
- 5: IDR slice (I-frame)
- 6: SEI (Supplemental Enhancement Information)
- 7: SPS (Sequence Parameter Set)
- 8: PPS (Picture Parameter Set)
- 9: Access Unit Delimiter
"""

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Iterator, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NALUnitType(IntEnum):
    """NAL unit types from H.264 Table 7-1."""
    UNSPECIFIED = 0
    SLICE_NON_IDR = 1      # Coded slice of non-IDR picture
    SLICE_DATA_A = 2       # Coded slice data partition A
    SLICE_DATA_B = 3       # Coded slice data partition B
    SLICE_DATA_C = 4       # Coded slice data partition C
    SLICE_IDR = 5          # Coded slice of IDR picture
    SEI = 6                # Supplemental enhancement information
    SPS = 7                # Sequence parameter set
    PPS = 8                # Picture parameter set
    AUD = 9                # Access unit delimiter
    END_SEQ = 10           # End of sequence
    END_STREAM = 11        # End of stream
    FILLER = 12            # Filler data


@dataclass
class NALUnit:
    """Represents a parsed NAL unit.

    Attributes:
        nal_ref_idc: Reference indicator (0-3). Higher = more important for reference.
        nal_unit_type: Type of NAL unit (see NALUnitType enum).
        rbsp: Raw Byte Sequence Payload (data after removing emulation prevention).
        start_position: Byte position in original bitstream where NAL starts.
        size: Total size of NAL unit in bytes (including header).
    """
    nal_ref_idc: int
    nal_unit_type: int
    rbsp: bytes
    start_position: int = 0
    size: int = 0

    @property
    def type_name(self) -> str:
        """Human-readable NAL unit type name."""
        try:
            return NALUnitType(self.nal_unit_type).name
        except ValueError:
            return f"UNKNOWN({self.nal_unit_type})"

    @property
    def is_vcl(self) -> bool:
        """Whether this is a VCL (Video Coding Layer) NAL unit."""
        return 1 <= self.nal_unit_type <= 5

    @property
    def is_idr(self) -> bool:
        """Whether this is an IDR (Instantaneous Decoder Refresh) slice."""
        return self.nal_unit_type == NALUnitType.SLICE_IDR

    def __repr__(self) -> str:
        return (
            f"NALUnit(type={self.type_name}, ref_idc={self.nal_ref_idc}, "
            f"size={len(self.rbsp)} bytes)"
        )


# Start code patterns
START_CODE_3 = bytes([0x00, 0x00, 0x01])        # 3-byte start code
START_CODE_4 = bytes([0x00, 0x00, 0x00, 0x01])  # 4-byte start code


def find_start_codes(data: bytes) -> List[int]:
    """Find all start code positions in the bitstream.

    Searches for both 3-byte (0x000001) and 4-byte (0x00000001) start codes.

    Args:
        data: Raw bitstream bytes

    Returns:
        List of byte positions where start codes begin

    Note:
        Returns positions of the start code itself, not the NAL unit data.
    """
    positions = []
    i = 0
    data_len = len(data)

    while i < data_len - 2:
        # Check for 3-byte start code
        if data[i] == 0 and data[i + 1] == 0:
            if data[i + 2] == 1:
                # Found 0x000001
                # Check if preceded by 0x00 (making it 0x00000001)
                if i > 0 and data[i - 1] == 0:
                    # 4-byte start code, position is i-1
                    if not positions or positions[-1] != i - 1:
                        positions.append(i - 1)
                else:
                    positions.append(i)
                i += 3
                continue
            elif i < data_len - 3 and data[i + 2] == 0 and data[i + 3] == 1:
                # Found 0x00000001
                positions.append(i)
                i += 4
                continue
        i += 1

    logger.debug(f"Found {len(positions)} start codes at positions: {positions[:10]}...")
    return positions


def remove_emulation_prevention_bytes(data: bytes) -> bytes:
    """Remove emulation prevention bytes from NAL unit data.

    In H.264, the byte sequence 0x000003 is used to prevent start code
    emulation within NAL unit data. The 0x03 byte must be removed to
    get the actual RBSP (Raw Byte Sequence Payload).

    Args:
        data: NAL unit data (after start code, including header)

    Returns:
        Data with emulation prevention bytes removed

    H.264 Spec: Section 7.4.1

    Sequences that are escaped:
    - 0x000000 -> 0x00000300
    - 0x000001 -> 0x00000301
    - 0x000002 -> 0x00000302
    - 0x000003 -> 0x00000303
    """
    result = bytearray()
    i = 0
    data_len = len(data)

    while i < data_len:
        # Check for emulation prevention pattern
        if (i < data_len - 2 and
            data[i] == 0 and data[i + 1] == 0 and data[i + 2] == 3):
            # Found 0x000003, copy 0x0000 and skip the 0x03
            result.append(0)
            result.append(0)
            i += 3

            # The next byte should be 0x00, 0x01, 0x02, or 0x03
            if i < data_len:
                result.append(data[i])
                i += 1
        else:
            result.append(data[i])
            i += 1

    if len(result) != len(data):
        logger.debug(
            f"Removed {len(data) - len(result)} emulation prevention bytes"
        )

    return bytes(result)


def parse_nal_header(header_byte: int) -> Tuple[int, int, int]:
    """Parse NAL unit header byte.

    Args:
        header_byte: First byte of NAL unit (after start code)

    Returns:
        Tuple of (forbidden_zero_bit, nal_ref_idc, nal_unit_type)

    H.264 Spec: Section 7.3.1

    Header format (1 byte):
    - Bit 7: forbidden_zero_bit (should be 0)
    - Bits 6-5: nal_ref_idc (0-3)
    - Bits 4-0: nal_unit_type (0-31)
    """
    forbidden_zero_bit = (header_byte >> 7) & 0x01
    nal_ref_idc = (header_byte >> 5) & 0x03
    nal_unit_type = header_byte & 0x1F

    if forbidden_zero_bit != 0:
        logger.warning(f"NAL forbidden_zero_bit is not zero: {forbidden_zero_bit}")

    return forbidden_zero_bit, nal_ref_idc, nal_unit_type


def parse_nal_unit(data: bytes, start_pos: int = 0) -> NALUnit:
    """Parse a single NAL unit from raw bytes.

    Args:
        data: NAL unit bytes (without start code)
        start_pos: Position in original bitstream (for tracking)

    Returns:
        Parsed NALUnit object
    """
    if len(data) < 1:
        raise ValueError("NAL unit data too short")

    # Parse header
    forbidden, ref_idc, unit_type = parse_nal_header(data[0])

    # Remove emulation prevention and extract RBSP
    rbsp = remove_emulation_prevention_bytes(data[1:])  # Skip header byte

    nal = NALUnit(
        nal_ref_idc=ref_idc,
        nal_unit_type=unit_type,
        rbsp=rbsp,
        start_position=start_pos,
        size=len(data)
    )

    logger.debug(f"Parsed {nal}")
    return nal


def extract_nal_units(bitstream: bytes) -> List[NALUnit]:
    """Extract all NAL units from an Annex B bitstream.

    Args:
        bitstream: Complete Annex B formatted bitstream

    Returns:
        List of parsed NALUnit objects

    Example:
        >>> with open("video.264", "rb") as f:
        ...     data = f.read()
        >>> nals = extract_nal_units(data)
        >>> for nal in nals:
        ...     print(nal)
    """
    positions = find_start_codes(bitstream)

    if not positions:
        logger.warning("No start codes found in bitstream")
        return []

    nal_units = []

    for i, start in enumerate(positions):
        # Determine start code length
        if (start + 3 < len(bitstream) and
            bitstream[start:start + 4] == START_CODE_4):
            data_start = start + 4
        else:
            data_start = start + 3

        # Determine end of NAL unit
        if i + 1 < len(positions):
            data_end = positions[i + 1]
        else:
            data_end = len(bitstream)

        # Extract NAL unit data
        nal_data = bitstream[data_start:data_end]

        # Remove trailing zeros (some streams have them)
        while nal_data and nal_data[-1] == 0:
            nal_data = nal_data[:-1]

        if nal_data:
            try:
                nal = parse_nal_unit(nal_data, start_pos=start)
                nal_units.append(nal)
            except Exception as e:
                logger.error(f"Failed to parse NAL unit at {start}: {e}")

    logger.info(f"Extracted {len(nal_units)} NAL units from bitstream")
    return nal_units


def iter_nal_units(bitstream: bytes) -> Iterator[NALUnit]:
    """Iterate over NAL units in a bitstream (memory-efficient).

    Args:
        bitstream: Complete Annex B formatted bitstream

    Yields:
        NALUnit objects one at a time
    """
    positions = find_start_codes(bitstream)

    for i, start in enumerate(positions):
        # Determine start code length
        if (start + 3 < len(bitstream) and
            bitstream[start:start + 4] == START_CODE_4):
            data_start = start + 4
        else:
            data_start = start + 3

        # Determine end
        if i + 1 < len(positions):
            data_end = positions[i + 1]
        else:
            data_end = len(bitstream)

        nal_data = bitstream[data_start:data_end]

        # Remove trailing zeros
        while nal_data and nal_data[-1] == 0:
            nal_data = nal_data[:-1]

        if nal_data:
            yield parse_nal_unit(nal_data, start_pos=start)


def filter_nal_units(
    nal_units: List[NALUnit],
    unit_types: Optional[List[int]] = None
) -> List[NALUnit]:
    """Filter NAL units by type.

    Args:
        nal_units: List of NAL units
        unit_types: List of NAL unit types to keep (None = keep all)

    Returns:
        Filtered list of NAL units
    """
    if unit_types is None:
        return nal_units

    return [nal for nal in nal_units if nal.nal_unit_type in unit_types]


def get_sps_pps(nal_units: List[NALUnit]) -> Tuple[List[NALUnit], List[NALUnit]]:
    """Extract SPS and PPS NAL units.

    Args:
        nal_units: List of all NAL units

    Returns:
        Tuple of (SPS list, PPS list)
    """
    sps_list = filter_nal_units(nal_units, [NALUnitType.SPS])
    pps_list = filter_nal_units(nal_units, [NALUnitType.PPS])

    logger.debug(f"Found {len(sps_list)} SPS, {len(pps_list)} PPS")
    return sps_list, pps_list


def create_test_bitstream(nal_data_list: List[bytes]) -> bytes:
    """Create an Annex B bitstream from raw NAL unit data.

    Useful for testing - creates a valid bitstream with start codes.

    Args:
        nal_data_list: List of NAL unit bytes (including headers)

    Returns:
        Annex B formatted bitstream
    """
    result = bytearray()

    for nal_data in nal_data_list:
        # Add 4-byte start code
        result.extend(START_CODE_4)
        result.extend(nal_data)

    return bytes(result)
