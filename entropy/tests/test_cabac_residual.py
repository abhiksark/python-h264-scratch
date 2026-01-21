# h264/entropy/tests/test_cabac_residual.py
"""RED TESTS: CABAC residual block decoding.

Decode transform coefficient blocks using CABAC.
Uses significant_coeff_flag, last_significant_coeff_flag, coeff_abs_level.

H.264 Spec Reference: Section 9.3.3.1.3 - Decoding process for significance map

These tests SHOULD FAIL until CABAC residual decoding is implemented.
"""

import pytest
import numpy as np

from bitstream import BitReader


class TestResidualBlockDecoder:
    """Tests for residual block decoding function."""

    def test_decode_residual_block_cabac_exists(self):
        """decode_residual_block_cabac function should exist."""
        from entropy.cabac_residual import decode_residual_block_cabac

        assert callable(decode_residual_block_cabac)

    def test_residual_block_returns_array(self):
        """Residual block should return numpy array."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_cabac

        data = bytes([0xFF, 0x00, 0xFF, 0x00] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_cabac(
            decoder, contexts, max_coeff=16, block_cat=2
        )

        assert isinstance(result, np.ndarray)

    def test_residual_block_correct_length(self):
        """Residual block should have correct number of coefficients."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_cabac(
            decoder, contexts, max_coeff=16, block_cat=2
        )

        assert len(result) == 16


class TestSignificanceMap:
    """Tests for significance map decoding."""

    def test_decode_significant_coeff_flag_exists(self):
        """decode_significant_coeff_flag function should exist."""
        from entropy.cabac_residual import decode_significant_coeff_flag

        assert callable(decode_significant_coeff_flag)

    def test_decode_last_significant_coeff_flag_exists(self):
        """decode_last_significant_coeff_flag function should exist."""
        from entropy.cabac_residual import decode_last_significant_coeff_flag

        assert callable(decode_last_significant_coeff_flag)

    def test_sig_coeff_flag_binary(self):
        """significant_coeff_flag should return 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_significant_coeff_flag

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_significant_coeff_flag(
            decoder, contexts, block_cat=2, scan_idx=0
        )

        assert result in (0, 1)


class TestCoeffLevelDecoding:
    """Tests for coefficient level decoding."""

    def test_decode_coeff_abs_level_minus1_exists(self):
        """decode_coeff_abs_level_minus1 function should exist."""
        from entropy.cabac_residual import decode_coeff_abs_level_minus1

        assert callable(decode_coeff_abs_level_minus1)

    def test_coeff_level_non_negative(self):
        """coeff_abs_level_minus1 should be non-negative."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_coeff_abs_level_minus1

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_coeff_abs_level_minus1(
            decoder, contexts, block_cat=2,
            num_decode_abs_level_eq1=0, num_decode_abs_level_gt1=0
        )

        assert result >= 0

    def test_decode_coeff_sign_flag_exists(self):
        """decode_coeff_sign_flag function should exist."""
        from entropy.cabac_residual import decode_coeff_sign_flag

        assert callable(decode_coeff_sign_flag)


class TestBlockCategories:
    """Tests for different block categories."""

    def test_luma_dc_category(self):
        """Category 0 = Luma DC for I_16x16."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        result = decode_residual_block_cabac(
            decoder, contexts, max_coeff=16, block_cat=0
        )

        assert len(result) == 16

    def test_luma_ac_category(self):
        """Category 1 = Luma AC for I_16x16 (15 coeffs, skip DC)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        result = decode_residual_block_cabac(
            decoder, contexts, max_coeff=15, block_cat=1
        )

        assert len(result) == 15

    def test_luma_4x4_category(self):
        """Category 2 = Luma 4x4 (16 coeffs)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_cabac(
            decoder, contexts, max_coeff=16, block_cat=2
        )

        assert len(result) == 16

    def test_chroma_dc_category(self):
        """Category 3 = Chroma DC (4 coeffs for 4:2:0)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_cabac(
            decoder, contexts, max_coeff=4, block_cat=3
        )

        assert len(result) == 4

    def test_chroma_ac_category(self):
        """Category 4 = Chroma AC (15 coeffs)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_cabac(
            decoder, contexts, max_coeff=15, block_cat=4
        )

        assert len(result) == 15


class TestScanOrder:
    """Tests for coefficient scan order."""

    def test_get_scan_order_exists(self):
        """get_scan_order function should exist."""
        from entropy.cabac_residual import get_scan_order

        assert callable(get_scan_order)

    def test_zigzag_scan_4x4(self):
        """4x4 zigzag scan should have 16 positions."""
        from entropy.cabac_residual import get_scan_order

        scan = get_scan_order(block_cat=2)

        assert len(scan) == 16

    def test_scan_covers_all_positions(self):
        """Scan order should cover all positions exactly once."""
        from entropy.cabac_residual import get_scan_order

        scan = get_scan_order(block_cat=2)

        # Should be 16 unique positions
        assert len(set(scan)) == 16
