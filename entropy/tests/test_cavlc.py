# h264/entropy/tests/test_cavlc.py
"""Tests for CAVLC entropy decoding."""

import pytest
import numpy as np

from bitstream import BitWriter, BitReader, BITSTRING_AVAILABLE
from entropy.cavlc import (
    CAVLCBlock,
    decode_coeff_token,
    decode_trailing_ones_signs,
    decode_levels,
    decode_total_zeros,
    decode_run_before,
    decode_residual_block,
    decode_residual_4x4,
    decode_chroma_dc,
    calculate_nC,
)
from entropy.tables import (
    COEFF_TOKEN_0,
    COEFF_TOKEN_2,
    COEFF_TOKEN_4,
    COEFF_TOKEN_CHROMA_DC,
    TOTAL_ZEROS_4x4,
    RUN_BEFORE,
    ZIGZAG_4x4,
)

# Skip all tests if bitstring not available
pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestCAVLCBlock:
    """Tests for CAVLCBlock dataclass."""

    def test_empty_block(self):
        """Empty block detection."""
        block = CAVLCBlock(
            total_coeff=0,
            trailing_ones=0,
            coefficients=np.zeros(16, dtype=np.int32)
        )
        assert block.is_empty is True

    def test_non_empty_block(self):
        """Non-empty block detection."""
        coeffs = np.zeros(16, dtype=np.int32)
        coeffs[0] = 5
        block = CAVLCBlock(
            total_coeff=1,
            trailing_ones=0,
            coefficients=coeffs
        )
        assert block.is_empty is False


class TestCalculateNC:
    """Tests for nC calculation."""

    def test_both_available(self):
        """Both neighbors available."""
        assert calculate_nC(2, 4) == 3  # (2+4+1)>>1 = 3
        assert calculate_nC(0, 0) == 0
        assert calculate_nC(5, 3) == 4  # (5+3+1)>>1 = 4

    def test_only_left(self):
        """Only left neighbor available."""
        assert calculate_nC(3, None) == 3
        assert calculate_nC(0, None) == 0

    def test_only_top(self):
        """Only top neighbor available."""
        assert calculate_nC(None, 5) == 5
        assert calculate_nC(None, 0) == 0

    def test_neither_available(self):
        """Neither neighbor available."""
        assert calculate_nC(None, None) == 0


class TestDecodeCoeffToken:
    """Tests for coeff_token decoding."""

    def test_decode_zero_coeffs_nc0(self):
        """Decode TotalCoeff=0 with nC<2."""
        # (0,0) -> 0b1, 1 bit
        writer = BitWriter()
        writer.write_bits(0b1, 1)
        writer.write_bits(0, 7)  # Padding
        data = writer.to_bytes()

        reader = BitReader(data)
        tc, t1 = decode_coeff_token(reader, nC=0)

        assert tc == 0
        assert t1 == 0

    def test_decode_one_trailing_one_nc0(self):
        """Decode TotalCoeff=1, TrailingOnes=1 with nC<2."""
        # (1,1) -> 0b01, 2 bits
        writer = BitWriter()
        writer.write_bits(0b01, 2)
        writer.write_bits(0, 6)
        data = writer.to_bytes()

        reader = BitReader(data)
        tc, t1 = decode_coeff_token(reader, nC=0)

        assert tc == 1
        assert t1 == 1

    def test_decode_nc2_range(self):
        """Decode with nC in [2,4) range."""
        # (0,0) -> 0b11, 2 bits for nC=2
        writer = BitWriter()
        writer.write_bits(0b11, 2)
        writer.write_bits(0, 6)
        data = writer.to_bytes()

        reader = BitReader(data)
        tc, t1 = decode_coeff_token(reader, nC=2)

        assert tc == 0
        assert t1 == 0

    def test_decode_nc4_range(self):
        """Decode with nC in [4,8) range."""
        # (0,0) -> 0b1111, 4 bits for nC=4
        writer = BitWriter()
        writer.write_bits(0b1111, 4)
        writer.write_bits(0, 4)
        data = writer.to_bytes()

        reader = BitReader(data)
        tc, t1 = decode_coeff_token(reader, nC=4)

        assert tc == 0
        assert t1 == 0

    def test_decode_nc8_fixed_zero(self):
        """Decode TotalCoeff=0 with nC>=8 (fixed length)."""
        # TotalCoeff=0 -> code = 3 (0b000011)
        writer = BitWriter()
        writer.write_bits(0b000011, 6)
        writer.write_bits(0, 2)
        data = writer.to_bytes()

        reader = BitReader(data)
        tc, t1 = decode_coeff_token(reader, nC=8)

        assert tc == 0
        assert t1 == 0

    def test_decode_nc8_fixed_nonzero(self):
        """Decode with nC>=8 (fixed length), non-zero coeffs."""
        # TotalCoeff=5, TrailingOnes=2
        # code = (T1 << 4) | (TC - 1) = (2 << 4) | 4 = 0b100100
        writer = BitWriter()
        writer.write_bits(0b100100, 6)
        writer.write_bits(0, 2)
        data = writer.to_bytes()

        reader = BitReader(data)
        tc, t1 = decode_coeff_token(reader, nC=8)

        assert tc == 5
        assert t1 == 2

    def test_decode_chroma_dc(self):
        """Decode chroma DC coeff_token."""
        # (0,0) -> 0b01, 2 bits for chroma DC
        writer = BitWriter()
        writer.write_bits(0b01, 2)
        writer.write_bits(0, 6)
        data = writer.to_bytes()

        reader = BitReader(data)
        tc, t1 = decode_coeff_token(reader, nC=-1)

        assert tc == 0
        assert t1 == 0


class TestDecodeTrailingOnesSigns:
    """Tests for trailing ones sign decoding."""

    def test_zero_trailing_ones(self):
        """No trailing ones."""
        writer = BitWriter()
        writer.write_bits(0, 8)
        reader = BitReader(writer.to_bytes())

        signs = decode_trailing_ones_signs(reader, 0)
        assert signs == []

    def test_one_trailing_one_positive(self):
        """One positive trailing one."""
        writer = BitWriter()
        writer.write_bits(0, 1)  # 0 = positive
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        signs = decode_trailing_ones_signs(reader, 1)
        assert signs == [1]

    def test_one_trailing_one_negative(self):
        """One negative trailing one."""
        writer = BitWriter()
        writer.write_bits(1, 1)  # 1 = negative
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        signs = decode_trailing_ones_signs(reader, 1)
        assert signs == [-1]

    def test_three_trailing_ones(self):
        """Three trailing ones with mixed signs."""
        writer = BitWriter()
        # Signs read in reverse, so write: sign3, sign2, sign1
        # Want result [+1, -1, +1], read reverse: +1, -1, +1
        # Read bits: 0, 1, 0 -> result after reverse: [+1, -1, +1]
        writer.write_bits(0b010, 3)  # 0=+1, 1=-1, 0=+1
        writer.write_bits(0, 5)
        reader = BitReader(writer.to_bytes())

        signs = decode_trailing_ones_signs(reader, 3)
        assert signs == [1, -1, 1]


class TestDecodeTotalZeros:
    """Tests for total_zeros decoding."""

    def test_total_zeros_tc1(self):
        """Decode total_zeros with TotalCoeff=1."""
        # For TC=1, total_zeros=0 -> 0b1, 1 bit
        writer = BitWriter()
        writer.write_bits(0b1, 1)
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        tz = decode_total_zeros(reader, total_coeff=1)
        assert tz == 0

    def test_total_zeros_tc2(self):
        """Decode total_zeros with TotalCoeff=2."""
        # For TC=2, total_zeros=0 -> 0b111, 3 bits
        writer = BitWriter()
        writer.write_bits(0b111, 3)
        writer.write_bits(0, 5)
        reader = BitReader(writer.to_bytes())

        tz = decode_total_zeros(reader, total_coeff=2)
        assert tz == 0

    def test_total_zeros_max_coeffs(self):
        """Max coefficients means no zeros."""
        writer = BitWriter()
        reader = BitReader(writer.to_bytes())

        tz = decode_total_zeros(reader, total_coeff=16)
        assert tz == 0


class TestDecodeRunBefore:
    """Tests for run_before decoding."""

    def test_run_before_zeros_left_0(self):
        """No zeros left means run=0."""
        writer = BitWriter()
        reader = BitReader(writer.to_bytes())

        run = decode_run_before(reader, zeros_left=0)
        assert run == 0

    def test_run_before_zeros_left_1(self):
        """Decode run_before with zerosLeft=1."""
        # zerosLeft=1: run=0 -> 0b1, run=1 -> 0b0
        writer = BitWriter()
        writer.write_bits(0b1, 1)
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        run = decode_run_before(reader, zeros_left=1)
        assert run == 0

    def test_run_before_zeros_left_3(self):
        """Decode run_before with zerosLeft=3."""
        # zerosLeft=3: run=0 -> 0b11
        writer = BitWriter()
        writer.write_bits(0b11, 2)
        writer.write_bits(0, 6)
        reader = BitReader(writer.to_bytes())

        run = decode_run_before(reader, zeros_left=3)
        assert run == 0


class TestDecodeResidualBlock:
    """Tests for complete residual block decoding."""

    def test_decode_empty_block(self):
        """Decode block with no coefficients."""
        writer = BitWriter()
        # TC=0, T1=0 with nC=0 -> 0b1
        writer.write_bits(0b1, 1)
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        block = decode_residual_block(reader, nC=0)

        assert block.total_coeff == 0
        assert block.trailing_ones == 0
        assert block.is_empty is True
        assert np.all(block.coefficients == 0)

    def test_decode_single_trailing_one(self):
        """Decode block with single +1 coefficient."""
        writer = BitWriter()
        # TC=1, T1=1 with nC=0 -> 0b01
        code, bits = COEFF_TOKEN_0[(1, 1)]
        writer.write_bits(code, bits)
        # T1 sign: 0 = positive
        writer.write_bits(0, 1)
        # total_zeros for TC=1, TZ=0 -> 0b1
        writer.write_bits(0b1, 1)
        # Pad
        writer.write_bits(0, 8 - (bits + 2) % 8)
        reader = BitReader(writer.to_bytes())

        block = decode_residual_block(reader, nC=0)

        assert block.total_coeff == 1
        assert block.trailing_ones == 1
        assert block.coefficients[0] == 1

    def test_decode_block_with_level(self):
        """Decode block with level > 1."""
        writer = BitWriter()
        # TC=1, T1=0 with nC=0 -> 0b000101 (6 bits)
        code, bits = COEFF_TOKEN_0[(1, 0)]
        writer.write_bits(code, bits)
        # Level: prefix=0 (0b1), suffix_len=0, first level so add 1
        # level_code = 0, level = 1, then +1 -> 2
        writer.write_bits(0b1, 1)  # prefix = 0
        # total_zeros TC=1, TZ=0 -> 0b1
        writer.write_bits(0b1, 1)
        writer.write_bits(0, 8)
        reader = BitReader(writer.to_bytes())

        block = decode_residual_block(reader, nC=0)

        assert block.total_coeff == 1
        assert block.trailing_ones == 0
        assert block.coefficients[0] == 2  # First level gets +1


class TestDecodeResidual4x4:
    """Tests for 4x4 residual decoding."""

    def test_decode_empty_4x4(self):
        """Decode empty 4x4 block."""
        writer = BitWriter()
        writer.write_bits(0b1, 1)  # TC=0 for nC=0
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        coeffs = decode_residual_4x4(reader, nC=0)

        assert coeffs.shape == (4, 4)
        assert np.all(coeffs == 0)


class TestDecodeChromaDC:
    """Tests for chroma DC decoding."""

    def test_decode_empty_chroma_dc(self):
        """Decode empty chroma DC block."""
        writer = BitWriter()
        # TC=0 for chroma DC -> 0b01
        writer.write_bits(0b01, 2)
        writer.write_bits(0, 6)
        reader = BitReader(writer.to_bytes())

        coeffs = decode_chroma_dc(reader)

        assert coeffs.shape == (2, 2)
        assert np.all(coeffs == 0)


class TestZigzagTables:
    """Tests for zigzag scan tables."""

    def test_zigzag_4x4_coverage(self):
        """All positions covered in zigzag scan."""
        assert len(ZIGZAG_4x4) == 16
        assert set(ZIGZAG_4x4) == set(range(16))

    def test_zigzag_4x4_starts_at_dc(self):
        """First position is DC (0,0)."""
        assert ZIGZAG_4x4[0] == 0

    def test_zigzag_4x4_ends_at_last(self):
        """Last position is (3,3)."""
        assert ZIGZAG_4x4[-1] == 15


class TestVLCTableConsistency:
    """Tests for VLC table integrity."""

    def test_coeff_token_0_coverage(self):
        """Table 0 covers all (TC, T1) combinations."""
        for tc in range(17):
            for t1 in range(min(tc, 3) + 1):
                assert (tc, t1) in COEFF_TOKEN_0, f"Missing ({tc}, {t1})"

    def test_coeff_token_unique_codes(self):
        """No duplicate codes in table."""
        codes = [(code, bits) for (code, bits) in COEFF_TOKEN_0.values()]
        assert len(codes) == len(set(codes)), "Duplicate codes in table"

    def test_total_zeros_coverage(self):
        """total_zeros tables cover all TotalCoeff values."""
        for tc in range(1, 16):
            assert tc in TOTAL_ZEROS_4x4, f"Missing TC={tc}"

    def test_run_before_coverage(self):
        """run_before tables cover zerosLeft 1-7."""
        for zl in range(1, 8):
            assert zl in RUN_BEFORE, f"Missing zerosLeft={zl}"


class TestZigzag8x8:
    """Tests for 8x8 zigzag scan table."""

    def test_zigzag_8x8_exists(self):
        """ZIGZAG_8x8 table should exist."""
        from entropy.tables import ZIGZAG_8x8
        assert ZIGZAG_8x8 is not None

    def test_zigzag_8x8_length(self):
        """ZIGZAG_8x8 should have 64 elements."""
        from entropy.tables import ZIGZAG_8x8
        assert len(ZIGZAG_8x8) == 64

    def test_zigzag_8x8_coverage(self):
        """All 64 positions covered in 8x8 zigzag scan."""
        from entropy.tables import ZIGZAG_8x8
        assert set(ZIGZAG_8x8) == set(range(64))

    def test_zigzag_8x8_starts_at_dc(self):
        """First position is DC (0,0) = position 0."""
        from entropy.tables import ZIGZAG_8x8
        assert ZIGZAG_8x8[0] == 0

    def test_zigzag_8x8_ends_at_last(self):
        """Last position is (7,7) = position 63."""
        from entropy.tables import ZIGZAG_8x8
        assert ZIGZAG_8x8[-1] == 63

    def test_zigzag_8x8_diagonal_pattern(self):
        """Verify diagonal scan pattern at start."""
        from entropy.tables import ZIGZAG_8x8
        # First diagonal: (0,0)=0
        # Second diagonal: (0,1)=1, (1,0)=8
        # Third diagonal: (2,0)=16, (1,1)=9, (0,2)=2
        expected_start = [0, 1, 8, 16, 9, 2, 3, 10]
        assert list(ZIGZAG_8x8[:8]) == expected_start

    def test_zigzag_8x8_inv_exists(self):
        """ZIGZAG_8x8_INV inverse table should exist."""
        from entropy.tables import ZIGZAG_8x8_INV
        assert ZIGZAG_8x8_INV is not None

    def test_zigzag_8x8_inv_roundtrip(self):
        """Inverse zigzag should correctly invert the scan."""
        from entropy.tables import ZIGZAG_8x8, ZIGZAG_8x8_INV
        # zigzag[i] = raster_pos means inv[raster_pos] = i
        for scan_idx, raster_pos in enumerate(ZIGZAG_8x8):
            assert ZIGZAG_8x8_INV[raster_pos] == scan_idx


class TestDecodeResidual8x8:
    """Tests for 8x8 residual decoding."""

    def test_decode_residual_8x8_exists(self):
        """decode_residual_8x8 function should exist."""
        from entropy.cavlc import decode_residual_8x8
        assert callable(decode_residual_8x8)

    def test_decode_empty_8x8(self):
        """Decode empty 8x8 block."""
        from entropy.cavlc import decode_residual_8x8

        writer = BitWriter()
        writer.write_bits(0b1, 1)  # TC=0 for nC=0
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        coeffs = decode_residual_8x8(reader, nC=0)

        assert coeffs.shape == (8, 8)
        assert np.all(coeffs == 0)

    def test_decode_8x8_returns_correct_shape(self):
        """8x8 residual returns 8x8 array."""
        from entropy.cavlc import decode_residual_8x8

        writer = BitWriter()
        writer.write_bits(0b1, 1)  # TC=0
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        coeffs = decode_residual_8x8(reader, nC=0)

        assert coeffs.shape == (8, 8)
        assert coeffs.dtype == np.int32
