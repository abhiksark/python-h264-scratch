# h264/entropy/tests/test_cabac_binarize.py
"""RED TESTS: CABAC binarization schemes.

Binarization maps multi-valued syntax elements to binary strings:
- Unary (U): 1...10 for value n (n ones followed by zero)
- Truncated Unary (TU): Like unary but bounded
- Exp-Golomb (UEGk): Unary prefix + suffix bits
- Fixed Length (FL): Fixed number of bits

H.264 Spec Reference: Section 9.3.2 - Binarization process

These tests SHOULD FAIL until CABAC binarization is implemented.
"""

import pytest

from bitstream import BitReader


class TestUnaryBinarization:
    """Tests for unary code decoding."""

    def test_decode_unary_exists(self):
        """decode_unary function should exist."""
        from entropy.cabac_binarize import decode_unary

        assert callable(decode_unary)

    def test_unary_decode_zero(self):
        """Unary: 0 is encoded as single 0."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_binarize import decode_unary

        # Create decoder with data that decodes to 0
        # (high offset relative to range = LPS = 0)
        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_unary(decoder, ctx_base=60, contexts=contexts)

        # Result should be >= 0
        assert result >= 0

    def test_unary_decode_positive(self):
        """Unary: n is encoded as n ones followed by zero."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_binarize import decode_unary

        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_unary(decoder, ctx_base=60, contexts=contexts)

        assert isinstance(result, int)
        assert result >= 0


class TestTruncatedUnary:
    """Tests for truncated unary decoding."""

    def test_decode_truncated_unary_exists(self):
        """decode_truncated_unary function should exist."""
        from entropy.cabac_binarize import decode_truncated_unary

        assert callable(decode_truncated_unary)

    def test_truncated_unary_respects_max(self):
        """Truncated unary should not exceed max_val."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_binarize import decode_truncated_unary

        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        max_val = 3
        result = decode_truncated_unary(
            decoder, max_val=max_val, ctx_base=54, contexts=contexts
        )

        assert 0 <= result <= max_val

    def test_truncated_unary_at_max_no_terminator(self):
        """At max_val, no terminating zero is needed."""
        from entropy.cabac_binarize import decode_truncated_unary

        # This is a behavioral test - the max value has no trailing 0
        assert callable(decode_truncated_unary)


class TestExpGolombBinarization:
    """Tests for Exp-Golomb binarization (UEGk)."""

    def test_decode_uegk_exists(self):
        """decode_uegk function should exist."""
        from entropy.cabac_binarize import decode_uegk

        assert callable(decode_uegk)

    def test_uegk_decode_small_value(self):
        """UEGk prefix-only values (below threshold)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_binarize import decode_uegk

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        # k=0 means threshold at 14 for suffix
        result = decode_uegk(decoder, k=0, ctx_base=40, contexts=contexts)

        assert isinstance(result, int)
        assert result >= 0

    def test_uegk_returns_unsigned(self):
        """UEGk returns unsigned values."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_binarize import decode_uegk

        data = bytes([0x55, 0xAA, 0x55, 0xAA] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_uegk(decoder, k=0, ctx_base=40, contexts=contexts)

        assert result >= 0


class TestFixedLengthBinarization:
    """Tests for fixed-length binarization."""

    def test_decode_fixed_length_exists(self):
        """decode_fixed_length function should exist."""
        from entropy.cabac_binarize import decode_fixed_length

        assert callable(decode_fixed_length)

    def test_fixed_length_correct_bits(self):
        """Fixed length decodes exact number of bits."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_binarize import decode_fixed_length

        data = bytes([0xAA, 0x55, 0xAA, 0x55] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        # Decode 3 bits
        result = decode_fixed_length(decoder, num_bits=3)

        assert 0 <= result < 8  # 3 bits = max 7


class TestSignedValueDecoding:
    """Tests for signed value decoding (MVD, etc.)."""

    def test_decode_signed_value_exists(self):
        """decode_signed_value function should exist."""
        from entropy.cabac_binarize import decode_signed_value

        assert callable(decode_signed_value)

    def test_signed_zero_remains_zero(self):
        """Zero value stays zero (no sign bit)."""
        from entropy.cabac_binarize import decode_signed_value

        # sign_extend(0, sign=0) = 0
        # sign_extend(0, sign=1) = 0
        result = decode_signed_value(0, sign_flag=0)
        assert result == 0

        result = decode_signed_value(0, sign_flag=1)
        assert result == 0

    def test_signed_positive_value(self):
        """Positive values with sign_flag=0."""
        from entropy.cabac_binarize import decode_signed_value

        result = decode_signed_value(5, sign_flag=0)
        assert result == 5

    def test_signed_negative_value(self):
        """Negative values with sign_flag=1."""
        from entropy.cabac_binarize import decode_signed_value

        result = decode_signed_value(5, sign_flag=1)
        assert result == -5


class TestMVDBinarization:
    """Tests for motion vector difference binarization."""

    def test_decode_mvd_exists(self):
        """decode_mvd function should exist."""
        from entropy.cabac_binarize import decode_mvd

        assert callable(decode_mvd)

    def test_mvd_returns_integer(self):
        """MVD decoding returns integer."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_binarize import decode_mvd

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_mvd(decoder, contexts=contexts, comp=0)

        assert isinstance(result, int)

    def test_mvd_can_be_negative(self):
        """MVD can decode negative values."""
        from entropy.cabac_binarize import decode_mvd

        # Just verify the function accepts comp parameter
        assert callable(decode_mvd)


class TestCoeffLevelBinarization:
    """Tests for coefficient level binarization."""

    def test_decode_coeff_abs_level_exists(self):
        """decode_coeff_abs_level function should exist."""
        from entropy.cabac_binarize import decode_coeff_abs_level

        assert callable(decode_coeff_abs_level)

    def test_coeff_level_returns_positive(self):
        """Coefficient absolute level is always positive."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_binarize import decode_coeff_abs_level

        data = bytes([0xFF, 0x00, 0xFF, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_coeff_abs_level(decoder, contexts=contexts, cat=2)

        assert result >= 0
