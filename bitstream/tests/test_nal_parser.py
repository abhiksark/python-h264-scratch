# h264/bitstream/tests/test_nal_parser.py
"""Tests for NAL unit parsing."""

import pytest
import numpy as np

from bitstream import (
    NALUnitType,
    NALUnit,
    START_CODE_3,
    START_CODE_4,
    find_start_codes,
    remove_emulation_prevention_bytes,
    parse_nal_header,
    parse_nal_unit,
    extract_nal_units,
    create_test_bitstream,
    get_sps_pps,
)


class TestStartCodes:
    """Tests for start code constants."""

    def test_start_code_3_bytes(self):
        """3-byte start code should be 0x000001."""
        assert START_CODE_3 == b'\x00\x00\x01'
        assert len(START_CODE_3) == 3

    def test_start_code_4_bytes(self):
        """4-byte start code should be 0x00000001."""
        assert START_CODE_4 == b'\x00\x00\x00\x01'
        assert len(START_CODE_4) == 4


class TestFindStartCodes:
    """Tests for start code detection."""

    def test_find_single_3byte_start_code(self):
        """Find single 3-byte start code."""
        data = b'\x00\x00\x01\x67\x42'
        positions = find_start_codes(data)
        assert positions == [0]

    def test_find_single_4byte_start_code(self):
        """Find single 4-byte start code."""
        data = b'\x00\x00\x00\x01\x67\x42'
        positions = find_start_codes(data)
        assert positions == [0]

    def test_find_multiple_start_codes(self):
        """Find multiple start codes."""
        data = b'\x00\x00\x01\x67\x00\x00\x01\x68'
        positions = find_start_codes(data)
        assert len(positions) == 2
        assert 0 in positions
        assert 4 in positions

    def test_find_mixed_start_codes(self):
        """Find mix of 3-byte and 4-byte start codes."""
        data = b'\x00\x00\x00\x01\x67\x00\x00\x01\x68'
        positions = find_start_codes(data)
        assert len(positions) == 2

    def test_no_start_codes(self):
        """Empty list when no start codes."""
        data = b'\x67\x42\x00\x1f\x96'
        positions = find_start_codes(data)
        assert positions == []

    def test_empty_data(self):
        """Empty data returns empty list."""
        positions = find_start_codes(b'')
        assert positions == []

    def test_start_code_at_end(self):
        """Handle start code at end of data."""
        data = b'\x67\x42\x00\x00\x01'
        positions = find_start_codes(data)
        assert 2 in positions


class TestEmulationPrevention:
    """Tests for emulation prevention byte removal."""

    def test_no_emulation_bytes(self):
        """Data without emulation bytes unchanged."""
        data = b'\x67\x42\x00\x1f\x96'
        result = remove_emulation_prevention_bytes(data)
        assert result == data

    def test_remove_single_emulation_byte(self):
        """Remove single emulation prevention byte."""
        # 0x00 0x00 0x03 0x00 -> 0x00 0x00 0x00
        data = b'\x67\x00\x00\x03\x00\x42'
        result = remove_emulation_prevention_bytes(data)
        assert result == b'\x67\x00\x00\x00\x42'

    def test_remove_multiple_emulation_bytes(self):
        """Remove multiple emulation prevention bytes."""
        data = b'\x00\x00\x03\x01\x00\x00\x03\x02'
        result = remove_emulation_prevention_bytes(data)
        assert result == b'\x00\x00\x01\x00\x00\x02'

    def test_preserve_non_emulation_sequences(self):
        """Don't remove bytes that aren't emulation prevention."""
        # 0x00 0x00 0x04 should be preserved (not 0x03)
        data = b'\x00\x00\x04\x00'
        result = remove_emulation_prevention_bytes(data)
        assert result == data

    def test_empty_data(self):
        """Empty data returns empty."""
        result = remove_emulation_prevention_bytes(b'')
        assert result == b''


class TestParseNALHeader:
    """Tests for NAL header parsing."""

    def test_parse_sps_header(self):
        """Parse SPS NAL header (type 7)."""
        # 0x67 = 0b01100111 = forbidden=0, ref_idc=3, type=7
        forbidden, ref_idc, nal_type = parse_nal_header(0x67)
        assert forbidden == 0
        assert ref_idc == 3
        assert nal_type == 7

    def test_parse_pps_header(self):
        """Parse PPS NAL header (type 8)."""
        # 0x68 = 0b01101000 = forbidden=0, ref_idc=3, type=8
        forbidden, ref_idc, nal_type = parse_nal_header(0x68)
        assert forbidden == 0
        assert ref_idc == 3
        assert nal_type == 8

    def test_parse_idr_header(self):
        """Parse IDR slice NAL header (type 5)."""
        # 0x65 = 0b01100101 = forbidden=0, ref_idc=3, type=5
        forbidden, ref_idc, nal_type = parse_nal_header(0x65)
        assert forbidden == 0
        assert ref_idc == 3
        assert nal_type == 5

    def test_parse_non_idr_header(self):
        """Parse non-IDR slice NAL header (type 1)."""
        # 0x41 = 0b01000001 = forbidden=0, ref_idc=2, type=1
        forbidden, ref_idc, nal_type = parse_nal_header(0x41)
        assert forbidden == 0
        assert ref_idc == 2
        assert nal_type == 1

    def test_forbidden_bit_set(self):
        """Detect forbidden bit set."""
        # 0x87 = 0b10000111 = forbidden=1
        forbidden, ref_idc, nal_type = parse_nal_header(0x87)
        assert forbidden == 1


class TestNALUnit:
    """Tests for NALUnit dataclass."""

    def test_nal_unit_properties(self):
        """Test NALUnit convenience properties."""
        nal = NALUnit(
            nal_ref_idc=3,
            nal_unit_type=NALUnitType.SPS,
            rbsp=b'\x42\x00\x1f'
        )
        assert nal.type_name == "SPS"
        assert not nal.is_vcl
        assert not nal.is_idr

    def test_nal_unit_idr(self):
        """Test IDR detection."""
        nal = NALUnit(
            nal_ref_idc=3,
            nal_unit_type=NALUnitType.SLICE_IDR,
            rbsp=b'\x00'
        )
        assert nal.is_vcl
        assert nal.is_idr

    def test_nal_unit_non_idr_slice(self):
        """Test non-IDR slice detection."""
        nal = NALUnit(
            nal_ref_idc=2,
            nal_unit_type=NALUnitType.SLICE_NON_IDR,
            rbsp=b'\x00'
        )
        assert nal.is_vcl
        assert not nal.is_idr


class TestParseNALUnit:
    """Tests for single NAL unit parsing."""

    def test_parse_simple_nal(self):
        """Parse simple NAL unit."""
        # SPS header + some data
        data = b'\x67\x42\x00\x1f\x96'
        nal = parse_nal_unit(data)

        assert nal.nal_unit_type == 7  # SPS
        assert nal.nal_ref_idc == 3
        assert len(nal.rbsp) == 4  # Data after header

    def test_parse_nal_with_emulation(self):
        """Parse NAL unit with emulation prevention."""
        # Header + data with emulation byte
        data = b'\x67\x00\x00\x03\x00\x42'
        nal = parse_nal_unit(data)

        # RBSP should have emulation byte removed
        assert nal.rbsp == b'\x00\x00\x00\x42'

    def test_parse_empty_raises(self):
        """Empty NAL data should raise."""
        with pytest.raises(ValueError):
            parse_nal_unit(b'')


class TestExtractNALUnits:
    """Tests for extracting NAL units from bitstream."""

    def test_extract_single_nal(self):
        """Extract single NAL unit from bitstream."""
        bitstream = b'\x00\x00\x00\x01\x67\x42\x00\x1f'
        nals = extract_nal_units(bitstream)

        assert len(nals) == 1
        assert nals[0].nal_unit_type == NALUnitType.SPS

    def test_extract_multiple_nals(self):
        """Extract multiple NAL units."""
        bitstream = (
            b'\x00\x00\x00\x01\x67\x42\x00\x1f'  # SPS
            b'\x00\x00\x00\x01\x68\x00\x00'       # PPS
            b'\x00\x00\x01\x65\x00'               # IDR slice (3-byte start)
        )
        nals = extract_nal_units(bitstream)

        assert len(nals) == 3
        assert nals[0].nal_unit_type == NALUnitType.SPS
        assert nals[1].nal_unit_type == NALUnitType.PPS
        assert nals[2].nal_unit_type == NALUnitType.SLICE_IDR

    def test_extract_empty_bitstream(self):
        """Empty bitstream returns empty list."""
        nals = extract_nal_units(b'')
        assert nals == []

    def test_extract_no_start_codes(self):
        """Data without start codes returns empty list."""
        nals = extract_nal_units(b'\x67\x42\x00\x1f')
        assert nals == []


class TestCreateTestBitstream:
    """Tests for test bitstream creation."""

    def test_create_single_nal(self):
        """Create bitstream with single NAL."""
        nal_data = b'\x67\x42\x00\x1f'
        bitstream = create_test_bitstream([nal_data])

        # Should have 4-byte start code + data
        assert bitstream[:4] == START_CODE_4
        assert bitstream[4:] == nal_data

    def test_create_multiple_nals(self):
        """Create bitstream with multiple NALs."""
        nals = [b'\x67\x42', b'\x68\x00']
        bitstream = create_test_bitstream(nals)

        # Extract and verify
        extracted = extract_nal_units(bitstream)
        assert len(extracted) == 2

    def test_roundtrip(self):
        """Create and extract should roundtrip."""
        original_data = [b'\x67\x42\x00\x1f', b'\x68\x00\x00']
        bitstream = create_test_bitstream(original_data)
        extracted = extract_nal_units(bitstream)

        assert len(extracted) == 2
        # Note: RBSP won't exactly match due to header removal
        assert extracted[0].nal_unit_type == 7
        assert extracted[1].nal_unit_type == 8


class TestGetSPSPPS:
    """Tests for SPS/PPS extraction."""

    def test_get_sps_pps(self):
        """Extract SPS and PPS from NAL list."""
        nals = [
            NALUnit(3, NALUnitType.SPS, b'\x42'),
            NALUnit(3, NALUnitType.PPS, b'\x00'),
            NALUnit(3, NALUnitType.SLICE_IDR, b'\x00'),
        ]

        sps_list, pps_list = get_sps_pps(nals)

        assert len(sps_list) == 1
        assert len(pps_list) == 1
        assert sps_list[0].nal_unit_type == NALUnitType.SPS
        assert pps_list[0].nal_unit_type == NALUnitType.PPS

    def test_get_sps_pps_none(self):
        """Handle missing SPS/PPS."""
        nals = [NALUnit(3, NALUnitType.SLICE_IDR, b'\x00')]

        sps_list, pps_list = get_sps_pps(nals)

        assert len(sps_list) == 0
        assert len(pps_list) == 0
