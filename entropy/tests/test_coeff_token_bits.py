# h264/entropy/tests/test_coeff_token_bits.py
"""Test coeff_token VLC decoding consumes correct number of bits.

H.264 Spec Reference: Section 9.2.1, Table 9-5
"""
import pytest
from bitstream import BitReader
from entropy.cavlc import decode_coeff_token


def test_coeff_token_nC0_code_1():
    """coeff_token with nC=0, code '1' -> TC=0, T1=0, 1 bit."""
    data = bytes([0b10000000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=0)

    assert tc == 0
    assert t1 == 0
    assert reader.position == 1  # Consumed exactly 1 bit


def test_coeff_token_nC0_code_000101():
    """coeff_token with nC=0, code '000101' -> TC=1, T1=0, 6 bits.

    From H.264 Table 9-5(a): (1,0) = (0b000101, 6)
    """
    data = bytes([0b00010100])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=0)

    assert tc == 1
    assert t1 == 0
    assert reader.position == 6  # Consumed exactly 6 bits


def test_coeff_token_nC0_code_01():
    """coeff_token with nC=0, code '01' -> TC=1, T1=1, 2 bits.

    From H.264 Table 9-5(a): (1,1) = (0b01, 2)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=0)

    assert tc == 1
    assert t1 == 1
    assert reader.position == 2


def test_coeff_token_nC2_code_11():
    """coeff_token with nC=2, code '11' -> TC=0, T1=0, 2 bits.

    From H.264 Table 9-5(b): (0,0) = (0b11, 2)
    """
    data = bytes([0b11000000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=2)

    assert tc == 0
    assert t1 == 0
    assert reader.position == 2


def test_coeff_token_nC2_code_0100():
    """coeff_token with nC=2, code '0100' -> TC=4, T1=3, 4 bits.

    From H.264 Table 9-5(b): (4,3) = (0b0100, 4)
    """
    data = bytes([0b01000000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=2)

    assert tc == 4
    assert t1 == 3
    assert reader.position == 4


def test_coeff_token_nC4_code_1111():
    """coeff_token with nC=4, code '1111' -> TC=0, T1=0, 4 bits.

    From H.264 Table 9-5(c): (0,0) = (0b1111, 4)
    """
    data = bytes([0b11110000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=4)

    assert tc == 0
    assert t1 == 0
    assert reader.position == 4


def test_coeff_token_nC4_code_1110():
    """coeff_token with nC=4, code '1110' -> TC=1, T1=1, 4 bits.

    From H.264 Table 9-5(c): (1,1) = (0b1110, 4)
    """
    data = bytes([0b11100000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=4)

    assert tc == 1
    assert t1 == 1
    assert reader.position == 4


def test_coeff_token_nC8_fixed_6bits_tc0():
    """coeff_token with nC>=8 uses fixed 6-bit code: code=3 -> TC=0, T1=0."""
    # Fixed code 000011 for TC=0, T1=0
    data = bytes([0b00001100])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=8)

    assert tc == 0
    assert t1 == 0
    assert reader.position == 6


def test_coeff_token_nC8_fixed_6bits_tc1_t1_0():
    """coeff_token with nC>=8: code=0 (000000) -> TC=1, T1=0.

    From Table 9-5(d): code = (T1 << 4) | (TC - 1)
    For TC=1, T1=0: code = (0 << 4) | (1 - 1) = 0
    """
    # Fixed code 000000 for TC=1, T1=0
    data = bytes([0b00000000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=8)

    assert tc == 1
    assert t1 == 0
    assert reader.position == 6


def test_coeff_token_nC8_fixed_6bits_tc4_t1_1():
    """coeff_token with nC>=8: code=13 (001101) -> TC=4, T1=1.

    Per JM reference: T1 = code & 3, TC = (code >> 2) + 1
    TC=4 -> code>>2=3, T1=1 -> code&3=1, code = (3<<2)|1 = 13
    """
    # Fixed code 001101 = 13: TC=4, T1=1
    data = bytes([0b00110100])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=8)

    assert tc == 4
    assert t1 == 1
    assert reader.position == 6


def test_coeff_token_chroma_dc_code_1():
    """coeff_token for chroma DC (nC=-1), code '1' -> TC=1, T1=1, 1 bit.

    From H.264 Table 9-5(e): (1,1) = (0b1, 1)
    """
    data = bytes([0b10000000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=-1)

    assert tc == 1
    assert t1 == 1
    assert reader.position == 1


def test_coeff_token_chroma_dc_code_00000010():
    """coeff_token for chroma DC (nC=-1), code '00000010' -> TC=4, T1=2, 8 bits.

    From H.264 Table 9-5(e) / JM reference: (4,2) = (0b00000010, 8)
    """
    data = bytes([0b00000010])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=-1)

    assert tc == 4
    assert t1 == 2
    assert reader.position == 8


def test_coeff_token_nC1_code_00011():
    """Test nC=1 with code '00011' -> TC=3, T1=3, 5 bits.

    From H.264 Table 9-5(a): (3,3) = (0b00011, 5)
    """
    data = bytes([0b00011000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=1)

    assert tc == 3
    assert t1 == 3
    assert reader.position == 5
