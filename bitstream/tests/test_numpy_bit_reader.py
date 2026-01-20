# h264/bitstream/tests/test_numpy_bit_reader.py
"""Tests for pure NumPy bit reader implementation."""

import pytest
import numpy as np

from bitstream.numpy_bit_reader import NumpyBitReader, NumpyBitWriter


class TestNumpyBitReaderBasic:
    """Basic read operations."""

    def test_initialization(self):
        """Reader initializes correctly."""
        reader = NumpyBitReader(b'\x00\xff')
        assert reader.position == 0
        assert reader.bits_remaining == 16
        assert reader.bytes_remaining == 2

    def test_read_single_bit(self):
        """Read individual bits."""
        # 0b10101010 = 0xAA
        reader = NumpyBitReader(b'\xaa')
        assert reader.read_bit() == 1
        assert reader.read_bit() == 0
        assert reader.read_bit() == 1
        assert reader.read_bit() == 0
        assert reader.position == 4

    def test_read_bits_within_byte(self):
        """Read multiple bits within single byte."""
        # 0b11110000 = 0xF0
        reader = NumpyBitReader(b'\xf0')
        assert reader.read_bits(4) == 0b1111
        assert reader.read_bits(4) == 0b0000

    def test_read_bits_across_bytes(self):
        """Read bits spanning byte boundary."""
        # 0b00001111 0b11110000 = 0x0F 0xF0
        reader = NumpyBitReader(b'\x0f\xf0')
        reader.read_bits(4)  # Skip first 4 bits (0000)
        # Now read 8 bits across boundary: 1111 1111
        assert reader.read_bits(8) == 0b11111111

    def test_read_flag(self):
        """Read boolean flags."""
        reader = NumpyBitReader(b'\x80')  # 0b10000000
        assert reader.read_flag() is True
        assert reader.read_flag() is False

    def test_read_byte(self):
        """Read full bytes."""
        reader = NumpyBitReader(b'\xab\xcd')
        assert reader.read_byte() == 0xAB
        assert reader.read_byte() == 0xCD

    def test_read_bytes_aligned(self):
        """Read multiple bytes when aligned."""
        reader = NumpyBitReader(b'\x01\x02\x03\x04')
        result = reader.read_bytes(2)
        assert result == b'\x01\x02'
        assert reader.position == 16

    def test_read_bytes_unaligned(self):
        """Read bytes when not byte-aligned."""
        reader = NumpyBitReader(b'\xff\x00\xff')
        reader.read_bit()  # Now at position 1
        result = reader.read_bytes(2)
        # Should read 16 bits starting at position 1
        # Original: 11111111 00000000 11111111
        # Read:     1111111_0 0000000_1 -> 0xFE 0x01
        assert result == b'\xfe\x01'


class TestNumpyBitReaderPosition:
    """Position tracking and manipulation."""

    def test_position_tracking(self):
        """Position updates correctly."""
        reader = NumpyBitReader(b'\xff\xff')
        assert reader.position == 0
        reader.read_bits(5)
        assert reader.position == 5
        reader.read_bits(3)
        assert reader.position == 8
        reader.read_bits(1)
        assert reader.position == 9

    def test_set_position(self):
        """Position can be set directly."""
        reader = NumpyBitReader(b'\xaa\xbb')
        reader.position = 8
        assert reader.read_byte() == 0xBB

    def test_bits_remaining(self):
        """Remaining bits counted correctly."""
        reader = NumpyBitReader(b'\x00\x00')
        assert reader.bits_remaining == 16
        reader.read_bits(10)
        assert reader.bits_remaining == 6

    def test_byte_align(self):
        """Byte alignment works."""
        reader = NumpyBitReader(b'\xff\xaa')
        reader.read_bits(3)
        reader.byte_align()
        assert reader.position == 8
        assert reader.read_byte() == 0xAA

    def test_byte_align_already_aligned(self):
        """Byte align when already aligned does nothing."""
        reader = NumpyBitReader(b'\xff\xaa')
        reader.byte_align()
        assert reader.position == 0

    def test_skip_bits(self):
        """Skip bits advances position."""
        reader = NumpyBitReader(b'\xff\xaa')
        reader.skip_bits(10)
        assert reader.position == 10

    def test_peek_bits(self):
        """Peek doesn't advance position."""
        reader = NumpyBitReader(b'\xf0')
        value = reader.peek_bits(4)
        assert value == 0b1111
        assert reader.position == 0
        # Can read same bits again
        assert reader.read_bits(4) == 0b1111


class TestNumpyExpGolomb:
    """Exponential-Golomb decoding tests."""

    def test_read_ue_zero(self):
        """ue(v) decode 0 (single 1 bit)."""
        # ue(0) = 1
        data = b'\x80'  # 0b10000000
        reader = NumpyBitReader(data)
        assert reader.read_ue() == 0

    def test_read_ue_one(self):
        """ue(v) decode 1."""
        # ue(1) = 010
        data = b'\x40'  # 0b01000000
        reader = NumpyBitReader(data)
        assert reader.read_ue() == 1

    def test_read_ue_two(self):
        """ue(v) decode 2."""
        # ue(2) = 011
        data = b'\x60'  # 0b01100000
        reader = NumpyBitReader(data)
        assert reader.read_ue() == 2

    def test_read_ue_three(self):
        """ue(v) decode 3."""
        # ue(3) = 00100
        data = b'\x20'  # 0b00100000
        reader = NumpyBitReader(data)
        assert reader.read_ue() == 3

    def test_read_ue_seven(self):
        """ue(v) decode 7."""
        # ue(7) = 0001000
        # code_num = 8 = 0b1000, needs 3 leading zeros
        # Pattern: 0001000 = 0x10
        data = b'\x10'
        reader = NumpyBitReader(data)
        assert reader.read_ue() == 7

    def test_read_ue_large(self):
        """ue(v) decode larger value."""
        # ue(14) = 0001111
        # code_num = 15 = 0b1111, needs 3 leading zeros
        data = b'\x1e'  # 0b00011110
        reader = NumpyBitReader(data)
        assert reader.read_ue() == 14

    def test_read_se_zero(self):
        """se(v) decode 0."""
        # se(0) maps from ue(0) = 1
        data = b'\x80'
        reader = NumpyBitReader(data)
        assert reader.read_se() == 0

    def test_read_se_positive(self):
        """se(v) decode positive values."""
        # se(1) maps from ue(1) = 010
        data = b'\x40'
        reader = NumpyBitReader(data)
        assert reader.read_se() == 1

        # se(2) maps from ue(3) = 00100
        data = b'\x20'
        reader = NumpyBitReader(data)
        assert reader.read_se() == 2

    def test_read_se_negative(self):
        """se(v) decode negative values."""
        # se(-1) maps from ue(2) = 011
        data = b'\x60'
        reader = NumpyBitReader(data)
        assert reader.read_se() == -1

        # se(-2) maps from ue(4) = 00101
        data = b'\x28'  # 0b00101000
        reader = NumpyBitReader(data)
        assert reader.read_se() == -2

    def test_read_te_max_one(self):
        """te(v) with max=1 reads single inverted bit."""
        reader = NumpyBitReader(b'\x00')  # First bit is 0
        assert reader.read_te(1) == 1  # 1 - 0 = 1

        reader = NumpyBitReader(b'\x80')  # First bit is 1
        assert reader.read_te(1) == 0  # 1 - 1 = 0

    def test_read_te_max_greater(self):
        """te(v) with max>1 uses ue(v)."""
        # Same as ue(3)
        data = b'\x20'
        reader = NumpyBitReader(data)
        assert reader.read_te(5) == 3


class TestNumpyBitReaderEdgeCases:
    """Edge cases and error handling."""

    def test_read_zero_bits(self):
        """Reading 0 bits returns 0."""
        reader = NumpyBitReader(b'\xff')
        assert reader.read_bits(0) == 0
        assert reader.position == 0

    def test_read_32_bits(self):
        """Can read up to 32 bits at once."""
        data = b'\xff\xff\xff\xff'
        reader = NumpyBitReader(data)
        assert reader.read_bits(32) == 0xFFFFFFFF

    def test_read_past_end_raises(self):
        """Reading past end raises error."""
        reader = NumpyBitReader(b'\xff')
        reader.read_bits(8)
        with pytest.raises(ValueError, match="Cannot read"):
            reader.read_bits(1)

    def test_read_too_many_bits_raises(self):
        """Reading >32 bits raises error."""
        reader = NumpyBitReader(b'\xff\xff\xff\xff\xff')
        with pytest.raises(ValueError, match="Cannot read more than 32"):
            reader.read_bits(33)

    def test_empty_data(self):
        """Empty data has no bits."""
        reader = NumpyBitReader(b'')
        assert reader.bits_remaining == 0
        with pytest.raises(ValueError):
            reader.read_bit()

    def test_more_rbsp_data_empty(self):
        """more_rbsp_data returns False when empty."""
        reader = NumpyBitReader(b'')
        assert reader.more_rbsp_data() is False

    def test_more_rbsp_data_trailing(self):
        """more_rbsp_data detects trailing bits at end of byte."""
        # When we're at bit position with <8 bits remaining,
        # check if it's the trailing bit pattern
        reader = NumpyBitReader(b'\x80')
        # Read 1 bit, now 7 bits remain: 0000000
        # But trailing pattern for 7 bits would be 1000000
        reader.read_bit()  # Read the leading 1
        # Now 7 bits remain: 0000000, which doesn't match trailing pattern
        # Actually this test needs rethinking - let's test proper trailing

        # Better test: read until we have 3 bits left that are trailing
        reader = NumpyBitReader(b'\xa0')  # 0b10100000
        reader.read_bits(5)  # Now 3 bits left: 000
        # Trailing for 3 bits is 100, we have 000, so more_rbsp_data=True
        # Let's use a byte that ends with proper trailing

        reader = NumpyBitReader(b'\xfc')  # 0b11111100
        reader.read_bits(5)  # Now 3 bits left: 100 (trailing pattern!)
        assert reader.more_rbsp_data() is False

    def test_more_rbsp_data_with_content(self):
        """more_rbsp_data returns True with real data."""
        reader = NumpyBitReader(b'\xff\xff')
        assert reader.more_rbsp_data() is True


class TestNumpyBitWriter:
    """Tests for NumpyBitWriter."""

    def test_write_bits(self):
        """Write arbitrary bits."""
        writer = NumpyBitWriter()
        writer.write_bits(0b1010, 4)
        writer.write_bits(0b0101, 4)
        assert writer.to_bytes() == b'\xa5'

    def test_write_bit(self):
        """Write individual bits."""
        writer = NumpyBitWriter()
        writer.write_bit(1)
        writer.write_bit(0)
        writer.write_bit(1)
        writer.write_bit(0)
        writer.write_bit(1)
        writer.write_bit(0)
        writer.write_bit(1)
        writer.write_bit(0)
        assert writer.to_bytes() == b'\xaa'

    def test_write_flag(self):
        """Write boolean flags."""
        writer = NumpyBitWriter()
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_bits(0, 4)  # Pad
        assert writer.to_bytes() == b'\xa0'

    def test_write_ue_zero(self):
        """Write ue(0)."""
        writer = NumpyBitWriter()
        writer.write_ue(0)
        writer.write_bits(0, 7)  # Pad
        assert writer.to_bytes() == b'\x80'

    def test_write_ue_values(self):
        """Write various ue values."""
        # Test roundtrip
        for value in [0, 1, 2, 3, 7, 14, 100]:
            writer = NumpyBitWriter()
            writer.write_ue(value)
            data = writer.to_bytes()
            reader = NumpyBitReader(data)
            assert reader.read_ue() == value

    def test_write_se_values(self):
        """Write various se values."""
        # Test roundtrip
        for value in [0, 1, -1, 2, -2, 10, -10]:
            writer = NumpyBitWriter()
            writer.write_se(value)
            data = writer.to_bytes()
            reader = NumpyBitReader(data)
            assert reader.read_se() == value

    def test_padding(self):
        """Bytes are padded to boundary."""
        writer = NumpyBitWriter()
        writer.write_bits(0b111, 3)
        result = writer.to_bytes()
        assert len(result) == 1
        assert result == b'\xe0'  # 0b11100000


class TestNumpyReaderWriterCompatibility:
    """Test that reader and writer are compatible."""

    def test_complex_bitstream(self):
        """Complex bitstream roundtrip."""
        writer = NumpyBitWriter()

        # Write mixed content
        writer.write_flag(True)
        writer.write_bits(0b101, 3)
        writer.write_ue(7)
        writer.write_se(-3)
        writer.write_bits(0xff, 8)

        data = writer.to_bytes()
        reader = NumpyBitReader(data)

        assert reader.read_flag() is True
        assert reader.read_bits(3) == 0b101
        assert reader.read_ue() == 7
        assert reader.read_se() == -3
        assert reader.read_bits(8) == 0xff

    def test_multiple_ue_values(self):
        """Multiple ue values in sequence."""
        writer = NumpyBitWriter()
        values = [0, 1, 2, 3, 4, 5, 6, 7]
        for v in values:
            writer.write_ue(v)

        data = writer.to_bytes()
        reader = NumpyBitReader(data)

        for expected in values:
            assert reader.read_ue() == expected
