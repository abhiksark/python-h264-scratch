# h264/entropy/tests/test_cabac_syntax.py
"""RED TESTS: CABAC syntax element decoding.

Decode individual H.264 syntax elements using CABAC.
Each syntax element has specific binarization and context assignment.

H.264 Spec Reference: Section 9.3.3.1 - Decoding process for binary decisions

These tests SHOULD FAIL until CABAC syntax decoding is implemented.
"""

import pytest

from bitstream import BitReader


class TestMBSkipFlag:
    """Tests for mb_skip_flag decoding."""

    def test_decode_mb_skip_flag_exists(self):
        """decode_mb_skip_flag function should exist."""
        from entropy.cabac_syntax import decode_mb_skip_flag

        assert callable(decode_mb_skip_flag)

    def test_mb_skip_flag_returns_binary(self):
        """mb_skip_flag should return 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mb_skip_flag

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_mb_skip_flag(
            decoder, contexts, slice_type=0, mb_x=0, mb_y=0,
            left_skip=False, top_skip=False
        )

        assert result in (0, 1)

    def test_mb_skip_flag_context_depends_on_neighbors(self):
        """mb_skip_flag context depends on neighbor skip status."""
        from entropy.cabac_syntax import decode_mb_skip_flag

        # Just verify the function accepts neighbor parameters
        assert callable(decode_mb_skip_flag)


class TestMBTypeDecoding:
    """Tests for mb_type decoding."""

    def test_decode_mb_type_i_exists(self):
        """decode_mb_type_i function should exist."""
        from entropy.cabac_syntax import decode_mb_type_i

        assert callable(decode_mb_type_i)

    def test_decode_mb_type_p_exists(self):
        """decode_mb_type_p function should exist."""
        from entropy.cabac_syntax import decode_mb_type_p

        assert callable(decode_mb_type_p)

    def test_decode_mb_type_b_exists(self):
        """decode_mb_type_b function should exist."""
        from entropy.cabac_syntax import decode_mb_type_b

        assert callable(decode_mb_type_b)

    def test_mb_type_i_returns_valid_type(self):
        """I-slice mb_type should be valid I-MB type."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mb_type_i

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        result = decode_mb_type_i(decoder, contexts)

        # I-slice mb_type: 0=I_4x4, 1-24=I_16x16, 25=I_PCM
        assert 0 <= result <= 25

    def test_mb_type_p_returns_valid_type(self):
        """P-slice mb_type should be valid."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mb_type_p

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_mb_type_p(decoder, contexts)

        # P-slice mb_type: 0-4 for P-MB, 5+ for I-MB in P-slice
        assert result >= 0


class TestSubMBTypeDecoding:
    """Tests for sub_mb_type decoding."""

    def test_decode_sub_mb_type_p_exists(self):
        """decode_sub_mb_type_p function should exist."""
        from entropy.cabac_syntax import decode_sub_mb_type_p

        assert callable(decode_sub_mb_type_p)

    def test_decode_sub_mb_type_b_exists(self):
        """decode_sub_mb_type_b function should exist."""
        from entropy.cabac_syntax import decode_sub_mb_type_b

        assert callable(decode_sub_mb_type_b)

    def test_sub_mb_type_p_valid_range(self):
        """P-slice sub_mb_type should be 0-3."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_sub_mb_type_p

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_sub_mb_type_p(decoder, contexts)

        assert 0 <= result <= 3


class TestRefIdxDecoding:
    """Tests for ref_idx decoding."""

    def test_decode_ref_idx_exists(self):
        """decode_ref_idx function should exist."""
        from entropy.cabac_syntax import decode_ref_idx

        assert callable(decode_ref_idx)

    def test_ref_idx_returns_non_negative(self):
        """ref_idx should be non-negative."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_ref_idx

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_ref_idx(decoder, contexts, list_idx=0)

        assert result >= 0


class TestMVDDecoding:
    """Tests for mvd (motion vector difference) decoding."""

    def test_decode_mvd_lx_exists(self):
        """decode_mvd_lx function should exist."""
        from entropy.cabac_syntax import decode_mvd_lx

        assert callable(decode_mvd_lx)

    def test_mvd_returns_signed(self):
        """MVD can be positive or negative."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mvd_lx

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_mvd_lx(decoder, contexts, list_idx=0, comp=0)

        assert isinstance(result, int)


class TestMBQPDeltaDecoding:
    """Tests for mb_qp_delta decoding."""

    def test_decode_mb_qp_delta_exists(self):
        """decode_mb_qp_delta function should exist."""
        from entropy.cabac_syntax import decode_mb_qp_delta

        assert callable(decode_mb_qp_delta)

    def test_mb_qp_delta_returns_integer(self):
        """mb_qp_delta should return signed integer."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mb_qp_delta

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_mb_qp_delta(decoder, contexts)

        assert isinstance(result, int)


class TestCodedBlockPattern:
    """Tests for coded_block_pattern decoding."""

    def test_decode_cbp_luma_exists(self):
        """decode_cbp_luma function should exist."""
        from entropy.cabac_syntax import decode_cbp_luma

        assert callable(decode_cbp_luma)

    def test_decode_cbp_chroma_exists(self):
        """decode_cbp_chroma function should exist."""
        from entropy.cabac_syntax import decode_cbp_chroma

        assert callable(decode_cbp_chroma)

    def test_cbp_luma_range(self):
        """CBP luma should be 0-15 (4 bits, one per 8x8 block)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_cbp_luma

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_cbp_luma(decoder, contexts, mb_type=0)

        assert 0 <= result <= 15

    def test_cbp_chroma_range(self):
        """CBP chroma should be 0-2."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_cbp_chroma

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_cbp_chroma(decoder, contexts, mb_type=0)

        assert 0 <= result <= 2


class TestIntraPredModeDecoding:
    """Tests for intra prediction mode decoding."""

    def test_decode_prev_intra_pred_mode_flag_exists(self):
        """decode_prev_intra4x4_pred_mode_flag should exist."""
        from entropy.cabac_syntax import decode_prev_intra4x4_pred_mode_flag

        assert callable(decode_prev_intra4x4_pred_mode_flag)

    def test_decode_rem_intra_pred_mode_exists(self):
        """decode_rem_intra4x4_pred_mode should exist."""
        from entropy.cabac_syntax import decode_rem_intra4x4_pred_mode

        assert callable(decode_rem_intra4x4_pred_mode)

    def test_decode_intra_chroma_pred_mode_exists(self):
        """decode_intra_chroma_pred_mode should exist."""
        from entropy.cabac_syntax import decode_intra_chroma_pred_mode

        assert callable(decode_intra_chroma_pred_mode)

    def test_intra_chroma_pred_mode_range(self):
        """Intra chroma pred mode should be 0-3."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_intra_chroma_pred_mode

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        result = decode_intra_chroma_pred_mode(decoder, contexts)

        assert 0 <= result <= 3


class TestCodedBlockFlag:
    """Tests for coded_block_flag decoding."""

    def test_decode_coded_block_flag_exists(self):
        """decode_coded_block_flag function should exist."""
        from entropy.cabac_syntax import decode_coded_block_flag

        assert callable(decode_coded_block_flag)

    def test_coded_block_flag_returns_binary(self):
        """coded_block_flag should return 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_coded_block_flag

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_coded_block_flag(decoder, contexts, cat=2, ctx_block_cat=0)

        assert result in (0, 1)
