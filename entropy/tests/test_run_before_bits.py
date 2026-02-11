# h264/entropy/tests/test_run_before_bits.py
"""Test run_before VLC decoding consumes correct number of bits.

H.264 Spec Reference: Section 9.2.4, Table 9-10
"""
import pytest
from bitstream import BitReader
from entropy.cavlc import decode_run_before


# =============================================================================
# Tests for zeros_left=1 (Table 9-10, column 1)
# =============================================================================

def test_run_before_zeros_left_1_code_1():
    """run_before with zeros_left=1, code '1' -> run=0, 1 bit.

    From H.264 Table 9-10: zeros_left=1, run=0 -> (0b1, 1)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=1)

    assert run == 0
    assert reader.position == 1


def test_run_before_zeros_left_1_code_0():
    """run_before with zeros_left=1, code '0' -> run=1, 1 bit.

    From H.264 Table 9-10: zeros_left=1, run=1 -> (0b0, 1)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=1)

    assert run == 1
    assert reader.position == 1


# =============================================================================
# Tests for zeros_left=2 (Table 9-10, column 2)
# =============================================================================

def test_run_before_zeros_left_2_code_1():
    """run_before with zeros_left=2, code '1' -> run=0, 1 bit.

    From H.264 Table 9-10: zeros_left=2, run=0 -> (0b1, 1)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=2)

    assert run == 0
    assert reader.position == 1


def test_run_before_zeros_left_2_code_01():
    """run_before with zeros_left=2, code '01' -> run=1, 2 bits.

    From H.264 Table 9-10: zeros_left=2, run=1 -> (0b01, 2)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=2)

    assert run == 1
    assert reader.position == 2


def test_run_before_zeros_left_2_code_00():
    """run_before with zeros_left=2, code '00' -> run=2, 2 bits.

    From H.264 Table 9-10: zeros_left=2, run=2 -> (0b00, 2)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=2)

    assert run == 2
    assert reader.position == 2


# =============================================================================
# Tests for zeros_left=3 (Table 9-10, column 3)
# =============================================================================

def test_run_before_zeros_left_3_code_11():
    """run_before with zeros_left=3, code '11' -> run=0, 2 bits.

    From H.264 Table 9-10: zeros_left=3, run=0 -> (0b11, 2)
    """
    data = bytes([0b11000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=3)

    assert run == 0
    assert reader.position == 2


def test_run_before_zeros_left_3_code_10():
    """run_before with zeros_left=3, code '10' -> run=1, 2 bits.

    From H.264 Table 9-10: zeros_left=3, run=1 -> (0b10, 2)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=3)

    assert run == 1
    assert reader.position == 2


def test_run_before_zeros_left_3_code_01():
    """run_before with zeros_left=3, code '01' -> run=2, 2 bits.

    From H.264 Table 9-10: zeros_left=3, run=2 -> (0b01, 2)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=3)

    assert run == 2
    assert reader.position == 2


def test_run_before_zeros_left_3_code_00():
    """run_before with zeros_left=3, code '00' -> run=3, 2 bits.

    From H.264 Table 9-10: zeros_left=3, run=3 -> (0b00, 2)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=3)

    assert run == 3
    assert reader.position == 2


# =============================================================================
# Tests for zeros_left=4 (Table 9-10, column 4)
# =============================================================================

def test_run_before_zeros_left_4_code_11():
    """run_before with zeros_left=4, code '11' -> run=0, 2 bits.

    From H.264 Table 9-10: zeros_left=4, run=0 -> (0b11, 2)
    """
    data = bytes([0b11000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=4)

    assert run == 0
    assert reader.position == 2


def test_run_before_zeros_left_4_code_10():
    """run_before with zeros_left=4, code '10' -> run=1, 2 bits.

    From H.264 Table 9-10: zeros_left=4, run=1 -> (0b10, 2)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=4)

    assert run == 1
    assert reader.position == 2


def test_run_before_zeros_left_4_code_01():
    """run_before with zeros_left=4, code '01' -> run=2, 2 bits.

    From H.264 Table 9-10: zeros_left=4, run=2 -> (0b01, 2)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=4)

    assert run == 2
    assert reader.position == 2


def test_run_before_zeros_left_4_code_001():
    """run_before with zeros_left=4, code '001' -> run=3, 3 bits.

    From H.264 Table 9-10: zeros_left=4, run=3 -> (0b001, 3)
    """
    data = bytes([0b00100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=4)

    assert run == 3
    assert reader.position == 3


def test_run_before_zeros_left_4_code_000():
    """run_before with zeros_left=4, code '000' -> run=4, 3 bits.

    From H.264 Table 9-10: zeros_left=4, run=4 -> (0b000, 3)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=4)

    assert run == 4
    assert reader.position == 3


# =============================================================================
# Tests for zeros_left=5 (Table 9-10, column 5)
# =============================================================================

def test_run_before_zeros_left_5_code_11():
    """run_before with zeros_left=5, code '11' -> run=0, 2 bits.

    From H.264 Table 9-10: zeros_left=5, run=0 -> (0b11, 2)
    """
    data = bytes([0b11000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=5)

    assert run == 0
    assert reader.position == 2


def test_run_before_zeros_left_5_code_10():
    """run_before with zeros_left=5, code '10' -> run=1, 2 bits.

    From H.264 Table 9-10: zeros_left=5, run=1 -> (0b10, 2)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=5)

    assert run == 1
    assert reader.position == 2


def test_run_before_zeros_left_5_code_011():
    """run_before with zeros_left=5, code '011' -> run=2, 3 bits.

    From H.264 Table 9-10: zeros_left=5, run=2 -> (0b011, 3)
    """
    data = bytes([0b01100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=5)

    assert run == 2
    assert reader.position == 3


def test_run_before_zeros_left_5_code_010():
    """run_before with zeros_left=5, code '010' -> run=3, 3 bits.

    From H.264 Table 9-10: zeros_left=5, run=3 -> (0b010, 3)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=5)

    assert run == 3
    assert reader.position == 3


def test_run_before_zeros_left_5_code_001():
    """run_before with zeros_left=5, code '001' -> run=4, 3 bits.

    From H.264 Table 9-10: zeros_left=5, run=4 -> (0b001, 3)
    """
    data = bytes([0b00100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=5)

    assert run == 4
    assert reader.position == 3


def test_run_before_zeros_left_5_code_000():
    """run_before with zeros_left=5, code '000' -> run=5, 3 bits.

    From H.264 Table 9-10: zeros_left=5, run=5 -> (0b000, 3)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=5)

    assert run == 5
    assert reader.position == 3


# =============================================================================
# Tests for zeros_left=6 (Table 9-10, column 6)
# =============================================================================

def test_run_before_zeros_left_6_code_11():
    """run_before with zeros_left=6, code '11' -> run=0, 2 bits.

    From H.264 Table 9-10: zeros_left=6, run=0 -> (0b11, 2)
    """
    data = bytes([0b11000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=6)

    assert run == 0
    assert reader.position == 2


def test_run_before_zeros_left_6_code_000():
    """run_before with zeros_left=6, code '000' -> run=1, 3 bits.

    From H.264 Table 9-10: zeros_left=6, run=1 -> (0b000, 3)
    """
    data = bytes([0b00000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=6)

    assert run == 1
    assert reader.position == 3


def test_run_before_zeros_left_6_code_001():
    """run_before with zeros_left=6, code '001' -> run=2, 3 bits.

    From H.264 Table 9-10: zeros_left=6, run=2 -> (0b001, 3)
    """
    data = bytes([0b00100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=6)

    assert run == 2
    assert reader.position == 3


def test_run_before_zeros_left_6_code_011():
    """run_before with zeros_left=6, code '011' -> run=3, 3 bits.

    From H.264 Table 9-10: zeros_left=6, run=3 -> (0b011, 3)
    """
    data = bytes([0b01100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=6)

    assert run == 3
    assert reader.position == 3


def test_run_before_zeros_left_6_code_010():
    """run_before with zeros_left=6, code '010' -> run=4, 3 bits.

    From H.264 Table 9-10: zeros_left=6, run=4 -> (0b010, 3)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=6)

    assert run == 4
    assert reader.position == 3


def test_run_before_zeros_left_6_code_101():
    """run_before with zeros_left=6, code '101' -> run=5, 3 bits.

    From H.264 Table 9-10: zeros_left=6, run=5 -> (0b101, 3)
    """
    data = bytes([0b10100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=6)

    assert run == 5
    assert reader.position == 3


def test_run_before_zeros_left_6_code_100():
    """run_before with zeros_left=6, code '100' -> run=6, 3 bits.

    From H.264 Table 9-10: zeros_left=6, run=6 -> (0b100, 3)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=6)

    assert run == 6
    assert reader.position == 3


# =============================================================================
# Tests for zeros_left>=7 (Table 9-10, column 7+)
# =============================================================================

def test_run_before_zeros_left_7_code_111():
    """run_before with zeros_left=7, code '111' -> run=0, 3 bits.

    From H.264 Table 9-10: zeros_left>=7, run=0 -> (0b111, 3)
    """
    data = bytes([0b11100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 0
    assert reader.position == 3


def test_run_before_zeros_left_7_code_110():
    """run_before with zeros_left=7, code '110' -> run=1, 3 bits.

    From H.264 Table 9-10: zeros_left>=7, run=1 -> (0b110, 3)
    """
    data = bytes([0b11000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 1
    assert reader.position == 3


def test_run_before_zeros_left_7_code_101():
    """run_before with zeros_left=7, code '101' -> run=2, 3 bits.

    From H.264 Table 9-10: zeros_left>=7, run=2 -> (0b101, 3)
    """
    data = bytes([0b10100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 2
    assert reader.position == 3


def test_run_before_zeros_left_7_code_100():
    """run_before with zeros_left=7, code '100' -> run=3, 3 bits.

    From H.264 Table 9-10: zeros_left>=7, run=3 -> (0b100, 3)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 3
    assert reader.position == 3


def test_run_before_zeros_left_7_code_011():
    """run_before with zeros_left=7, code '011' -> run=4, 3 bits.

    From H.264 Table 9-10: zeros_left>=7, run=4 -> (0b011, 3)
    """
    data = bytes([0b01100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 4
    assert reader.position == 3


def test_run_before_zeros_left_7_code_010():
    """run_before with zeros_left=7, code '010' -> run=5, 3 bits.

    From H.264 Table 9-10: zeros_left>=7, run=5 -> (0b010, 3)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 5
    assert reader.position == 3


def test_run_before_zeros_left_7_code_001():
    """run_before with zeros_left=7, code '001' -> run=6, 3 bits.

    From H.264 Table 9-10: zeros_left>=7, run=6 -> (0b001, 3)
    """
    data = bytes([0b00100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 6
    assert reader.position == 3


def test_run_before_zeros_left_7_code_0001():
    """run_before with zeros_left=7, code '0001' -> run=7, 4 bits.

    From H.264 Table 9-10: zeros_left>=7, run=7 -> (0b0001, 4)
    """
    data = bytes([0b00010000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 7
    assert reader.position == 4


def test_run_before_zeros_left_7_code_00001():
    """run_before with zeros_left=7, code '00001' -> run=8, 5 bits.

    From H.264 Table 9-10: zeros_left>=7, run=8 -> (0b00001, 5)
    """
    data = bytes([0b00001000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 8
    assert reader.position == 5


def test_run_before_zeros_left_7_code_000001():
    """run_before with zeros_left=7, code '000001' -> run=9, 6 bits.

    From H.264 Table 9-10: zeros_left>=7, run=9 -> (0b000001, 6)
    """
    data = bytes([0b00000100])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 9
    assert reader.position == 6


def test_run_before_zeros_left_7_code_0000001():
    """run_before with zeros_left=7, code '0000001' -> run=10, 7 bits.

    From H.264 Table 9-10: zeros_left>=7, run=10 -> (0b0000001, 7)
    """
    data = bytes([0b00000010])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 10
    assert reader.position == 7


def test_run_before_zeros_left_7_code_00000001():
    """run_before with zeros_left=7, code '00000001' -> run=11, 8 bits.

    From H.264 Table 9-10: zeros_left>=7, run=11 -> (0b00000001, 8)
    """
    data = bytes([0b00000001])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 11
    assert reader.position == 8


def test_run_before_zeros_left_7_code_000000001():
    """run_before with zeros_left=7, code '000000001' -> run=12, 9 bits.

    From H.264 Table 9-10: zeros_left>=7, run=12 -> (0b000000001, 9)
    """
    data = bytes([0b00000000, 0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 12
    assert reader.position == 9


def test_run_before_zeros_left_7_code_0000000001():
    """run_before with zeros_left=7, code '0000000001' -> run=13, 10 bits.

    From H.264 Table 9-10: zeros_left>=7, run=13 -> (0b0000000001, 10)
    """
    data = bytes([0b00000000, 0b01000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 13
    assert reader.position == 10


def test_run_before_zeros_left_7_code_00000000001():
    """run_before with zeros_left=7, code '00000000001' -> run=14, 11 bits.

    From H.264 Table 9-10: zeros_left>=7, run=14 -> (0b00000000001, 11)
    """
    data = bytes([0b00000000, 0b00100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=7)

    assert run == 14
    assert reader.position == 11


def test_run_before_zeros_left_10_code_111():
    """run_before with zeros_left=10, code '111' -> run=0, 3 bits.

    zeros_left >= 7 all use the same table (capped at 7)
    From H.264 Table 9-10: zeros_left>=7, run=0 -> (0b111, 3)
    """
    data = bytes([0b11100000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=10)

    assert run == 0
    assert reader.position == 3


def test_run_before_zeros_left_14_code_00001():
    """run_before with zeros_left=14, code '00001' -> run=8, 5 bits.

    zeros_left >= 7 all use the same table (capped at 7)
    From H.264 Table 9-10: zeros_left>=7, run=8 -> (0b00001, 5)
    """
    data = bytes([0b00001000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=14)

    assert run == 8
    assert reader.position == 5


# =============================================================================
# Edge cases
# =============================================================================

def test_run_before_zeros_left_0():
    """run_before with zeros_left=0 -> run=0, 0 bits.

    Edge case: No zeros left, no bits consumed
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=0)

    assert run == 0
    assert reader.position == 0  # No bits consumed
