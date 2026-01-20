# h264/bitstream/tests/test_bit_reader.py
"""Tests for bit-level reader."""

import pytest

from bitstream import BitReader, BitWriter, BITSTRING_AVAILABLE

# Skip all tests if bitstring not available
pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestBitReader:
    """Tests for BitReader class."""

    def test_read_bits(self):
        """Read fixed number of bits."""
        data = b'\xAB'  # 0b10101011
        reader = BitReader(data)

        assert reader.read_bits(4) == 0b1010
        assert reader.read_bits(4) == 0b1011

    def test_read_single_bit(self):
        """Read bits one at a time."""
        data = b'\x80'  # 0b10000000
        reader = BitReader(data)

        assert reader.read_bit() == 1
        assert reader.read_bit() == 0
        assert reader.read_bit() == 0

    def test_read_flag(self):
        """Read bit as boolean."""
        data = b'\x80'  # 0b10000000
        reader = BitReader(data)

        assert reader.read_flag() is True
        assert reader.read_flag() is False

    def test_read_byte(self):
        """Read full byte."""
        data = b'\x42\x67'
        reader = BitReader(data)

        assert reader.read_byte() == 0x42
        assert reader.read_byte() == 0x67

    def test_position_tracking(self):
        """Track bit position correctly."""
        data = b'\xFF\xFF'
        reader = BitReader(data)

        assert reader.position == 0
        reader.read_bits(5)
        assert reader.position == 5
        reader.read_bits(3)
        assert reader.position == 8

    def test_bits_remaining(self):
        """Calculate remaining bits."""
        data = b'\x00\x00'  # 16 bits
        reader = BitReader(data)

        assert reader.bits_remaining == 16
        reader.read_bits(5)
        assert reader.bits_remaining == 11

    def test_peek_bits(self):
        """Peek without advancing position."""
        data = b'\xAB'  # 0b10101011
        reader = BitReader(data)

        # Peek should not change position
        value = reader.peek_bits(4)
        assert value == 0b1010
        assert reader.position == 0

        # Read should give same value
        assert reader.read_bits(4) == 0b1010
        assert reader.position == 4

    def test_byte_align(self):
        """Align to byte boundary."""
        data = b'\x00\x00'
        reader = BitReader(data)

        reader.read_bits(3)
        assert reader.position == 3

        reader.byte_align()
        assert reader.position == 8


class TestExpGolomb:
    """Tests for Exp-Golomb decoding."""

    def test_read_ue_zero(self):
        """ue(v) decode: 1 -> 0."""
        # 0b10000000 = 0x80, first bit is 1, so value is 0
        data = b'\x80'
        reader = BitReader(data)

        assert reader.read_ue() == 0

    def test_read_ue_one(self):
        """ue(v) decode: 010 -> 1."""
        # 0b01000000 = 0x40
        data = b'\x40'
        reader = BitReader(data)

        assert reader.read_ue() == 1

    def test_read_ue_two(self):
        """ue(v) decode: 011 -> 2."""
        # 0b01100000 = 0x60
        data = b'\x60'
        reader = BitReader(data)

        assert reader.read_ue() == 2

    def test_read_ue_three(self):
        """ue(v) decode: 00100 -> 3."""
        # 0b00100000 = 0x20
        data = b'\x20'
        reader = BitReader(data)

        assert reader.read_ue() == 3

    def test_read_ue_seven(self):
        """ue(v) decode: 00111 -> 6, then 0b1 -> 0."""
        # 0b00111100 = 0x3C -> first ue gives 6
        data = b'\x3C'
        reader = BitReader(data)

        assert reader.read_ue() == 6

    def test_read_ue_large(self):
        """ue(v) decode larger values."""
        # ue(7): code_num=8=0b1000, needs 3 leading zeros
        # Pattern: 0001000 = 0b00010000 = 0x10
        data = b'\x10'
        reader = BitReader(data)

        assert reader.read_ue() == 7

    def test_read_se_zero(self):
        """se(v) decode: 0."""
        data = b'\x80'  # ue = 0 -> se = 0
        reader = BitReader(data)

        assert reader.read_se() == 0

    def test_read_se_positive(self):
        """se(v) decode positive values."""
        # ue=1 -> se=1, ue=3 -> se=2
        data = b'\x40'  # ue = 1
        reader = BitReader(data)

        assert reader.read_se() == 1

    def test_read_se_negative(self):
        """se(v) decode negative values."""
        # ue=2 -> se=-1, ue=4 -> se=-2
        data = b'\x60'  # ue = 2 (011)
        reader = BitReader(data)

        assert reader.read_se() == -1

    def test_read_multiple_ue(self):
        """Read multiple ue values in sequence."""
        # ue=0: 1
        # ue=1: 010
        # ue=2: 011
        # Combined: 1 010 011 0 = 0b10100110 = 0xA6
        data = b'\xA6'
        reader = BitReader(data)

        assert reader.read_ue() == 0  # 1
        assert reader.read_ue() == 1  # 010
        assert reader.read_ue() == 2  # 011


class TestTruncatedExpGolomb:
    """Tests for truncated Exp-Golomb."""

    def test_read_te_max_one(self):
        """te(v) with max_value=1 reads single bit."""
        data = b'\x00'  # First bit is 0
        reader = BitReader(data)

        # max_value=1: reads 1 bit, returns 1-bit
        assert reader.read_te(1) == 1  # 0 -> 1-0 = 1

    def test_read_te_max_larger(self):
        """te(v) with max_value>1 uses ue(v)."""
        data = b'\x80'  # ue = 0
        reader = BitReader(data)

        assert reader.read_te(5) == 0


class TestBitWriter:
    """Tests for BitWriter class."""

    def test_write_bits(self):
        """Write fixed number of bits."""
        writer = BitWriter()
        writer.write_bits(0b1010, 4)
        writer.write_bits(0b1011, 4)

        result = writer.to_bytes()
        assert result == b'\xAB'

    def test_write_flag(self):
        """Write boolean flags."""
        writer = BitWriter()
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_bits(0, 4)  # Pad to byte

        result = writer.to_bytes()
        assert result == b'\x80'

    def test_write_ue_zero(self):
        """Write ue(0)."""
        writer = BitWriter()
        writer.write_ue(0)
        writer.write_bits(0, 7)  # Pad

        result = writer.to_bytes()
        # ue(0) = 1
        assert (result[0] & 0x80) == 0x80

    def test_write_ue_one(self):
        """Write ue(1)."""
        writer = BitWriter()
        writer.write_ue(1)
        writer.write_bits(0, 5)  # Pad

        result = writer.to_bytes()
        # ue(1) = 010
        assert (result[0] & 0xE0) == 0x40

    def test_write_se_positive(self):
        """Write positive se value."""
        writer = BitWriter()
        writer.write_se(1)  # Should be ue(1) = 010
        writer.write_bits(0, 5)

        result = writer.to_bytes()
        assert (result[0] & 0xE0) == 0x40

    def test_write_se_negative(self):
        """Write negative se value."""
        writer = BitWriter()
        writer.write_se(-1)  # Should be ue(2) = 011
        writer.write_bits(0, 5)

        result = writer.to_bytes()
        assert (result[0] & 0xE0) == 0x60


class TestRoundtrip:
    """Test write then read roundtrips."""

    def test_ue_roundtrip(self):
        """Write and read ue values."""
        for value in [0, 1, 2, 3, 7, 15, 31, 100]:
            writer = BitWriter()
            writer.write_ue(value)
            data = writer.to_bytes()

            reader = BitReader(data)
            assert reader.read_ue() == value

    def test_se_roundtrip(self):
        """Write and read se values."""
        for value in [0, 1, -1, 2, -2, 10, -10, 50, -50]:
            writer = BitWriter()
            writer.write_se(value)
            data = writer.to_bytes()

            reader = BitReader(data)
            assert reader.read_se() == value

    def test_mixed_roundtrip(self):
        """Write and read mixed values."""
        writer = BitWriter()
        writer.write_ue(5)
        writer.write_se(-3)
        writer.write_flag(True)
        writer.write_bits(0x1F, 5)
        data = writer.to_bytes()

        reader = BitReader(data)
        assert reader.read_ue() == 5
        assert reader.read_se() == -3
        assert reader.read_flag() is True
        assert reader.read_bits(5) == 0x1F
