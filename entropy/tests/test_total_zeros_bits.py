# h264/entropy/tests/test_total_zeros_bits.py
"""Test total_zeros VLC decoding consumes correct number of bits.

H.264 Spec Reference: Section 9.2.3, Table 9-7, Table 9-8
"""
import pytest
from bitstream import BitReader
from entropy.cavlc import decode_total_zeros


# =============================================================================
# Tests for 4x4 blocks (max_coeffs=16)
# =============================================================================

def test_total_zeros_TC1_code_1():
    """total_zeros with TC=1, code '1' -> TZ=0, 1 bit.

    From H.264 Table 9-7: TC=1, TZ=0 -> (0b1, 1)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=1, max_coeffs=16)

    assert tz == 0
    assert reader.position == 1


def test_total_zeros_TC1_code_011():
    """total_zeros with TC=1, code '011' -> TZ=1, 3 bits.

    From H.264 Table 9-7: TC=1, TZ=1 -> (0b011, 3)
    """
    data = bytes([0b01100000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=1, max_coeffs=16)

    assert tz == 1
    assert reader.position == 3


def test_total_zeros_TC1_code_000000001():
    """total_zeros with TC=1, code '000000001' -> TZ=15, 9 bits.

    From H.264 Table 9-7: TC=1, TZ=15 -> (0b000000001, 9)
    """
    data = bytes([0b00000000, 0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=1, max_coeffs=16)

    assert tz == 15
    assert reader.position == 9


def test_total_zeros_TC2_code_111():
    """total_zeros with TC=2, code '111' -> TZ=0, 3 bits.

    From H.264 Table 9-8: TC=2, TZ=0 -> (0b111, 3)
    """
    data = bytes([0b11100000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=2, max_coeffs=16)

    assert tz == 0
    assert reader.position == 3


def test_total_zeros_TC2_code_001():
    """total_zeros with TC=2, code '001' -> TZ=6, 3 bits.

    From H.264 Table 9-8: TC=2, TZ=6 -> (0b001, 3)
    """
    data = bytes([0b00100000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=2, max_coeffs=16)

    assert tz == 6
    assert reader.position == 3


def test_total_zeros_TC2_code_00000000001():
    """total_zeros with TC=2, code '00000000001' -> TZ=14, 11 bits.

    From H.264 Table 9-7/9-8: TC=2, TZ=14 -> (0b00000000001, 11)
    """
    data = bytes([0b00000000, 0b00100000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=2, max_coeffs=16)

    assert tz == 14
    assert reader.position == 11


def test_total_zeros_TC3_code_0101():
    """total_zeros with TC=3, code '0101' -> TZ=0, 4 bits.

    From H.264 Table 9-8: TC=3, TZ=0 -> (0b0101, 4)
    """
    data = bytes([0b01010000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=3, max_coeffs=16)

    assert tz == 0
    assert reader.position == 4


def test_total_zeros_TC3_code_100():
    """total_zeros with TC=3, code '100' -> TZ=6, 3 bits.

    From H.264 Table 9-8: TC=3, TZ=6 -> (0b100, 3)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=3, max_coeffs=16)

    assert tz == 6
    assert reader.position == 3


def test_total_zeros_TC4_code_00001():
    """total_zeros with TC=4, code '00001' -> TZ=11, 5 bits.

    From H.264 Table 9-7: TC=4, TZ=11 -> (0b00001, 5)
    """
    data = bytes([0b00001000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=4, max_coeffs=16)

    assert tz == 11
    assert reader.position == 5


def test_total_zeros_TC4_code_00011():
    """total_zeros with TC=4, code '00011' -> TZ=0, 5 bits.

    From H.264 Table 9-7: TC=4, TZ=0 -> (0b00011, 5)
    """
    data = bytes([0b00011000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=4, max_coeffs=16)

    assert tz == 0
    assert reader.position == 5


def test_total_zeros_TC5_code_00000():
    """total_zeros with TC=5, code '00000' -> TZ=10, 5 bits.

    From H.264 Table 9-7: TC=5, TZ=10 -> (0b00000, 5)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=5, max_coeffs=16)

    assert tz == 10
    assert reader.position == 5


def test_total_zeros_TC6_code_000001():
    """total_zeros with TC=6, code '000001' -> TZ=0, 6 bits.

    From H.264 Table 9-7: TC=6, TZ=0 -> (0b000001, 6)
    """
    data = bytes([0b00000100])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=6, max_coeffs=16)

    assert tz == 0
    assert reader.position == 6


def test_total_zeros_TC7_code_0001():
    """total_zeros with TC=7, code '0001' -> TZ=7, 4 bits.

    From H.264 Table 9-7: TC=7, TZ=7 -> (0b0001, 4)
    """
    data = bytes([0b00010000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=7, max_coeffs=16)

    assert tz == 7
    assert reader.position == 4


def test_total_zeros_TC7_code_001():
    """total_zeros with TC=7, code '001' -> TZ=8, 3 bits.

    From H.264 Table 9-7: TC=7, TZ=8 -> (0b001, 3)
    """
    data = bytes([0b00100000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=7, max_coeffs=16)

    assert tz == 8
    assert reader.position == 3


def test_total_zeros_TC8_code_000001():
    """total_zeros with TC=8, code '000001' -> TZ=0, 6 bits.

    From H.264 Table 9-7: TC=8, TZ=0 -> (0b000001, 6)
    """
    data = bytes([0b00000100])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=8, max_coeffs=16)

    assert tz == 0
    assert reader.position == 6


def test_total_zeros_TC9_code_01():
    """total_zeros with TC=9, code '01' -> TZ=6, 2 bits.

    From H.264 Table 9-8: TC=9, TZ=6 -> (0b01, 2)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=9, max_coeffs=16)

    assert tz == 6
    assert reader.position == 2


def test_total_zeros_TC10_code_00001():
    """total_zeros with TC=10, code '00001' -> TZ=0, 5 bits.

    From H.264 Table 9-8: TC=10, TZ=0 -> (0b00001, 5)
    """
    data = bytes([0b00001000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=10, max_coeffs=16)

    assert tz == 0
    assert reader.position == 5


def test_total_zeros_TC10_code_01():
    """total_zeros with TC=10, code '01' -> TZ=5, 2 bits.

    From H.264 Table 9-8: TC=10, TZ=5 -> (0b01, 2)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=10, max_coeffs=16)

    assert tz == 5
    assert reader.position == 2


def test_total_zeros_TC15_code_0():
    """total_zeros with TC=15, code '0' -> TZ=0, 1 bit.

    From H.264 Table 9-8: TC=15, TZ=0 -> (0b0, 1)
    Edge case: Only 1 possible zero value
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=15, max_coeffs=16)

    assert tz == 0
    assert reader.position == 1


def test_total_zeros_TC15_code_1():
    """total_zeros with TC=15, code '1' -> TZ=1, 1 bit.

    From H.264 Table 9-8: TC=15, TZ=1 -> (0b1, 1)
    Edge case: Only 1 possible zero value
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=15, max_coeffs=16)

    assert tz == 1
    assert reader.position == 1


def test_total_zeros_TC16_no_decode():
    """total_zeros with TC=16 -> TZ=0, 0 bits.

    Edge case: No zeros possible, no bits consumed
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=16, max_coeffs=16)

    assert tz == 0
    assert reader.position == 0  # No bits consumed


# =============================================================================
# Tests for Chroma DC blocks (max_coeffs=4)
# =============================================================================

def test_total_zeros_chroma_TC1_code_1():
    """total_zeros chroma DC with TC=1, code '1' -> TZ=0, 1 bit.

    From H.264 Table 9-9: TC=1, TZ=0 -> (0b1, 1)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=1, max_coeffs=4)

    assert tz == 0
    assert reader.position == 1


def test_total_zeros_chroma_TC1_code_01():
    """total_zeros chroma DC with TC=1, code '01' -> TZ=1, 2 bits.

    From H.264 Table 9-9: TC=1, TZ=1 -> (0b01, 2)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=1, max_coeffs=4)

    assert tz == 1
    assert reader.position == 2


def test_total_zeros_chroma_TC1_code_000():
    """total_zeros chroma DC with TC=1, code '000' -> TZ=3, 3 bits.

    From H.264 Table 9-9: TC=1, TZ=3 -> (0b000, 3)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=1, max_coeffs=4)

    assert tz == 3
    assert reader.position == 3


def test_total_zeros_chroma_TC2_code_1():
    """total_zeros chroma DC with TC=2, code '1' -> TZ=0, 1 bit.

    From H.264 Table 9-9: TC=2, TZ=0 -> (0b1, 1)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=2, max_coeffs=4)

    assert tz == 0
    assert reader.position == 1


def test_total_zeros_chroma_TC2_code_00():
    """total_zeros chroma DC with TC=2, code '00' -> TZ=2, 2 bits.

    From H.264 Table 9-9: TC=2, TZ=2 -> (0b00, 2)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=2, max_coeffs=4)

    assert tz == 2
    assert reader.position == 2


def test_total_zeros_chroma_TC3_code_1():
    """total_zeros chroma DC with TC=3, code '1' -> TZ=0, 1 bit.

    From H.264 Table 9-9: TC=3, TZ=0 -> (0b1, 1)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=3, max_coeffs=4)

    assert tz == 0
    assert reader.position == 1


def test_total_zeros_chroma_TC3_code_0():
    """total_zeros chroma DC with TC=3, code '0' -> TZ=1, 1 bit.

    From H.264 Table 9-9: TC=3, TZ=1 -> (0b0, 1)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=3, max_coeffs=4)

    assert tz == 1
    assert reader.position == 1


def test_total_zeros_chroma_TC4_no_decode():
    """total_zeros chroma DC with TC=4 -> TZ=0, 0 bits.

    Edge case: No zeros possible, no bits consumed
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=4, max_coeffs=4)

    assert tz == 0
    assert reader.position == 0  # No bits consumed
