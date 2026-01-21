# h264/entropy/tests/test_cabac_8x8.py
"""RED TESTS: CABAC 8x8 residual block decoding for High profile.

CABAC decoding of 8x8 transform coefficients requires:
1. Different context model indices (Table 9-43)
2. Different zigzag scan pattern (64 positions vs 16)
3. Different coded_block_flag context selection
4. Extended significant_coeff_flag contexts
5. transform_8x8_flag parsing

H.264 Spec Reference:
- Section 9.3.3 (CABAC arithmetic decoding)
- Section 9.3.3.1.3 (Significance map decoding)
- Table 9-43 (ctxIdxInc for 8x8 blocks)
- Table 8-15 (8x8 zigzag scan)

These tests SHOULD FAIL until CABAC 8x8 residual decoding is implemented.
"""

import pytest
import numpy as np

from bitstream import BitReader


# =============================================================================
# Test: decode_residual_block_8x8 function existence and basic behavior
# =============================================================================

class TestResidualBlock8x8Decoder:
    """Tests for 8x8 residual block decoding function."""

    @pytest.mark.xfail(reason="decode_residual_block_8x8 not yet implemented")
    def test_decode_residual_block_8x8_exists(self):
        """decode_residual_block_8x8 function should exist for High profile."""
        from entropy.cabac_residual import decode_residual_block_8x8

        assert callable(decode_residual_block_8x8)

    @pytest.mark.xfail(reason="decode_residual_block_8x8 not yet implemented")
    def test_residual_block_8x8_returns_array(self):
        """8x8 residual block should return numpy array."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        data = bytes([0xFF, 0x00, 0xFF, 0x00] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_8x8(
            decoder, contexts, block_cat=5  # Luma 8x8
        )

        assert isinstance(result, np.ndarray)

    @pytest.mark.xfail(reason="decode_residual_block_8x8 not yet implemented")
    def test_residual_block_8x8_correct_length(self):
        """8x8 residual block should have 64 coefficients."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        data = bytes([0x80, 0x40, 0x20, 0x10] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_8x8(
            decoder, contexts, block_cat=5  # Luma 8x8
        )

        assert len(result) == 64

    @pytest.mark.xfail(reason="decode_residual_block_8x8 not yet implemented")
    def test_residual_block_8x8_coefficients_dtype(self):
        """8x8 coefficients should be int32 for computation."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        data = bytes([0xAA, 0x55, 0xAA, 0x55] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_8x8(
            decoder, contexts, block_cat=5
        )

        assert result.dtype == np.int32


# =============================================================================
# Test: 8x8 block categories (H.264 Table 9-42)
# =============================================================================

class TestBlockCategories8x8:
    """Tests for 8x8 block category definitions.

    Block categories 5-9 are for 8x8 transform (High profile):
    - 5: Luma 8x8
    - 6: Luma 8x8 (alternative)
    - 7: Cb 8x8 (4:4:4)
    - 8: Cr 8x8 (4:4:4)
    - 9: Cb/Cr 8x8 (4:2:2)
    """

    @pytest.mark.xfail(reason="8x8 block categories not yet defined")
    def test_block_cat_8x8_offsets_exist(self):
        """Context offsets for 8x8 block categories should exist."""
        from entropy.cabac_residual import BLOCK_CAT_CTX_OFFSETS_8X8

        assert 5 in BLOCK_CAT_CTX_OFFSETS_8X8  # Luma 8x8

    @pytest.mark.xfail(reason="8x8 block categories not yet defined")
    def test_luma_8x8_category_value(self):
        """Block category 5 is Luma 8x8."""
        from entropy.cabac_residual import BLOCK_CAT_8X8_LUMA

        assert BLOCK_CAT_8X8_LUMA == 5

    @pytest.mark.xfail(reason="8x8 block categories not yet defined")
    def test_block_cat_8x8_ctx_offsets_format(self):
        """8x8 context offsets should be (sig, last, abs) tuples."""
        from entropy.cabac_residual import BLOCK_CAT_CTX_OFFSETS_8X8

        offsets = BLOCK_CAT_CTX_OFFSETS_8X8.get(5)  # Luma 8x8
        assert offsets is not None
        assert len(offsets) == 3
        sig_offset, last_offset, abs_offset = offsets
        assert isinstance(sig_offset, int)
        assert isinstance(last_offset, int)
        assert isinstance(abs_offset, int)


# =============================================================================
# Test: coded_block_flag context selection for 8x8 blocks
# =============================================================================

class TestCodedBlockFlag8x8:
    """Tests for coded_block_flag context selection with 8x8 blocks.

    H.264 Section 9.3.3.1.1.9 - ctxIdxInc for coded_block_flag
    The context depends on neighboring block availability and CBF values.
    For 8x8 blocks, the neighbor selection differs from 4x4.
    """

    @pytest.mark.xfail(reason="8x8 coded_block_flag context not implemented")
    def test_coded_block_flag_8x8_ctx_idx_base(self):
        """coded_block_flag for 8x8 uses different context base."""
        from entropy.cabac_context import CTX_CODED_BLOCK_FLAG_8X8_START

        # 8x8 coded_block_flag contexts are separate from 4x4
        assert CTX_CODED_BLOCK_FLAG_8X8_START >= 85  # After 4x4 CBF contexts

    @pytest.mark.xfail(reason="8x8 coded_block_flag context not implemented")
    def test_decode_coded_block_flag_8x8_exists(self):
        """decode_coded_block_flag_8x8 function should exist."""
        from entropy.cabac_residual import decode_coded_block_flag_8x8

        assert callable(decode_coded_block_flag_8x8)

    @pytest.mark.xfail(reason="8x8 coded_block_flag context not implemented")
    def test_coded_block_flag_8x8_returns_binary(self):
        """coded_block_flag_8x8 should return 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_coded_block_flag_8x8

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        neighbor_info = {
            'left_cbf': 0,
            'top_cbf': 0,
            'left_available': True,
            'top_available': True,
        }

        result = decode_coded_block_flag_8x8(
            decoder, contexts, block_cat=5, neighbor_info=neighbor_info
        )

        assert result in (0, 1)

    @pytest.mark.xfail(reason="8x8 coded_block_flag context not implemented")
    def test_coded_block_flag_8x8_ctx_depends_on_neighbors(self):
        """Context index depends on left and top neighbor CBF values.

        ctxIdxInc = condTermFlagA + condTermFlagB
        where condTermFlag is 0 if neighbor available and CBF=0, else 1
        """
        from entropy.cabac_residual import get_coded_block_flag_8x8_ctx_inc

        # Both neighbors available, both CBF=0 -> ctxIdxInc = 0
        ctx_inc = get_coded_block_flag_8x8_ctx_inc(
            left_available=True, left_cbf=0,
            top_available=True, top_cbf=0
        )
        assert ctx_inc == 0

        # Both neighbors available, both CBF=1 -> ctxIdxInc = 2
        ctx_inc = get_coded_block_flag_8x8_ctx_inc(
            left_available=True, left_cbf=1,
            top_available=True, top_cbf=1
        )
        assert ctx_inc == 2

        # Left not available -> condTermFlagA=1
        ctx_inc = get_coded_block_flag_8x8_ctx_inc(
            left_available=False, left_cbf=0,
            top_available=True, top_cbf=0
        )
        assert ctx_inc == 1


# =============================================================================
# Test: significant_coeff_flag contexts for 8x8 scan patterns
# =============================================================================

class TestSignificantCoeffFlag8x8:
    """Tests for significant_coeff_flag context selection in 8x8 blocks.

    H.264 Table 9-43 specifies ctxIdxInc for significant_coeff_flag
    in 8x8 blocks. The context depends on scan position with a
    different mapping than 4x4 blocks.
    """

    @pytest.mark.xfail(reason="8x8 sig_coeff_flag contexts not implemented")
    def test_sig_coeff_flag_8x8_ctx_start(self):
        """significant_coeff_flag for 8x8 has dedicated context range."""
        from entropy.cabac_context import CTX_SIG_COEFF_FLAG_8X8_START

        # 8x8 sig_coeff contexts start after 4x4 contexts
        assert CTX_SIG_COEFF_FLAG_8X8_START >= 166

    @pytest.mark.xfail(reason="8x8 sig_coeff_flag contexts not implemented")
    def test_decode_significant_coeff_flag_8x8_exists(self):
        """decode_significant_coeff_flag_8x8 function should exist."""
        from entropy.cabac_residual import decode_significant_coeff_flag_8x8

        assert callable(decode_significant_coeff_flag_8x8)

    @pytest.mark.xfail(reason="8x8 sig_coeff_flag contexts not implemented")
    def test_sig_coeff_flag_8x8_ctx_mapping_table(self):
        """ctxIdxInc mapping table for 8x8 should exist (Table 9-43)."""
        from entropy.cabac_residual import SIG_COEFF_FLAG_CTX_INC_8X8

        # Table 9-43 has 63 entries (scan positions 0-62, pos 63 is implicit)
        assert len(SIG_COEFF_FLAG_CTX_INC_8X8) >= 63

    @pytest.mark.xfail(reason="8x8 sig_coeff_flag contexts not implemented")
    def test_sig_coeff_flag_8x8_first_position(self):
        """Position 0 in 8x8 scan uses specific context."""
        from entropy.cabac_residual import get_sig_coeff_flag_8x8_ctx_inc

        ctx_inc = get_sig_coeff_flag_8x8_ctx_inc(scan_idx=0)
        assert isinstance(ctx_inc, int)
        assert 0 <= ctx_inc < 63  # Valid context increment

    @pytest.mark.xfail(reason="8x8 sig_coeff_flag contexts not implemented")
    def test_sig_coeff_flag_8x8_last_position(self):
        """Position 62 (second-to-last) in 8x8 scan."""
        from entropy.cabac_residual import get_sig_coeff_flag_8x8_ctx_inc

        ctx_inc = get_sig_coeff_flag_8x8_ctx_inc(scan_idx=62)
        assert isinstance(ctx_inc, int)

    @pytest.mark.xfail(reason="8x8 sig_coeff_flag contexts not implemented")
    def test_sig_coeff_flag_8x8_diagonal_positions(self):
        """Context varies along diagonals in 8x8 scan pattern."""
        from entropy.cabac_residual import SIG_COEFF_FLAG_CTX_INC_8X8

        # Sample diagonal positions and verify different contexts
        # Positions 0, 2, 5, 9, 14, 20, ... are diagonal starts
        diag_starts = [0, 2, 5, 9, 14, 20, 27, 35]
        ctx_values = [SIG_COEFF_FLAG_CTX_INC_8X8[p] for p in diag_starts]

        # Not all diagonal starts should have same context
        assert len(set(ctx_values)) > 1


# =============================================================================
# Test: last_significant_coeff_flag for 8x8
# =============================================================================

class TestLastSignificantCoeffFlag8x8:
    """Tests for last_significant_coeff_flag context in 8x8 blocks.

    Similar to significant_coeff_flag, but indicates if current
    position is the last non-zero coefficient.
    """

    @pytest.mark.xfail(reason="8x8 last_sig_coeff_flag not implemented")
    def test_last_sig_coeff_flag_8x8_ctx_start(self):
        """last_significant_coeff_flag for 8x8 has dedicated context."""
        from entropy.cabac_context import CTX_LAST_SIG_COEFF_8X8_START

        assert CTX_LAST_SIG_COEFF_8X8_START >= 227

    @pytest.mark.xfail(reason="8x8 last_sig_coeff_flag not implemented")
    def test_decode_last_significant_coeff_flag_8x8_exists(self):
        """decode_last_significant_coeff_flag_8x8 function should exist."""
        from entropy.cabac_residual import decode_last_significant_coeff_flag_8x8

        assert callable(decode_last_significant_coeff_flag_8x8)

    @pytest.mark.xfail(reason="8x8 last_sig_coeff_flag not implemented")
    def test_last_sig_coeff_flag_8x8_ctx_mapping_table(self):
        """ctxIdxInc mapping table for last_sig in 8x8 exists."""
        from entropy.cabac_residual import LAST_SIG_COEFF_FLAG_CTX_INC_8X8

        assert len(LAST_SIG_COEFF_FLAG_CTX_INC_8X8) >= 63

    @pytest.mark.xfail(reason="8x8 last_sig_coeff_flag not implemented")
    def test_last_sig_coeff_flag_8x8_returns_binary(self):
        """last_significant_coeff_flag_8x8 returns 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_last_significant_coeff_flag_8x8

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_last_significant_coeff_flag_8x8(
            decoder, contexts, block_cat=5, scan_idx=10
        )

        assert result in (0, 1)

    @pytest.mark.xfail(reason="8x8 last_sig_coeff_flag not implemented")
    def test_last_sig_coeff_flag_8x8_not_decoded_at_position_63(self):
        """Position 63 (last) doesn't need last_sig_coeff_flag.

        If we reach position 63 and sig_coeff_flag=1, it's implicitly last.
        """
        from entropy.cabac_residual import LAST_SIG_COEFF_FLAG_CTX_INC_8X8

        # Table should only have 63 entries (0-62), not 64
        assert len(LAST_SIG_COEFF_FLAG_CTX_INC_8X8) == 63


# =============================================================================
# Test: coeff_abs_level_minus1 decoding for 8x8
# =============================================================================

class TestCoeffAbsLevel8x8:
    """Tests for coefficient level decoding in 8x8 blocks.

    coeff_abs_level_minus1 encoding is similar to 4x4 but uses
    8x8-specific context offsets.
    """

    @pytest.mark.xfail(reason="8x8 coeff_abs_level not implemented")
    def test_coeff_abs_level_8x8_ctx_offset(self):
        """8x8 blocks use specific context offset for coeff_abs_level."""
        from entropy.cabac_residual import BLOCK_CAT_CTX_OFFSETS_8X8

        offsets = BLOCK_CAT_CTX_OFFSETS_8X8[5]  # Luma 8x8
        _, _, abs_offset = offsets

        # 8x8 has different offset than 4x4
        assert abs_offset >= 0

    @pytest.mark.xfail(reason="8x8 coeff_abs_level not implemented")
    def test_decode_coeff_abs_level_minus1_8x8_exists(self):
        """decode_coeff_abs_level_minus1_8x8 function should exist."""
        from entropy.cabac_residual import decode_coeff_abs_level_minus1_8x8

        assert callable(decode_coeff_abs_level_minus1_8x8)

    @pytest.mark.xfail(reason="8x8 coeff_abs_level not implemented")
    def test_coeff_abs_level_8x8_non_negative(self):
        """coeff_abs_level_minus1 for 8x8 should be non-negative."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_coeff_abs_level_minus1_8x8

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_coeff_abs_level_minus1_8x8(
            decoder, contexts, block_cat=5,
            num_decode_abs_level_eq1=0, num_decode_abs_level_gt1=0
        )

        assert result >= 0

    @pytest.mark.xfail(reason="8x8 coeff_abs_level not implemented")
    def test_coeff_abs_level_8x8_context_adaptation(self):
        """Context adapts based on previously decoded level magnitudes."""
        from entropy.cabac_residual import get_coeff_abs_level_8x8_ctx_inc

        # First coefficient: no prior levels
        ctx_inc = get_coeff_abs_level_8x8_ctx_inc(
            num_eq1=0, num_gt1=0, bin_idx=0
        )
        assert ctx_inc == 1  # Initial context

        # After decoding one level=1
        ctx_inc = get_coeff_abs_level_8x8_ctx_inc(
            num_eq1=1, num_gt1=0, bin_idx=0
        )
        assert ctx_inc == 2

        # After decoding one level>1
        ctx_inc = get_coeff_abs_level_8x8_ctx_inc(
            num_eq1=0, num_gt1=1, bin_idx=0
        )
        assert ctx_inc == 0

    @pytest.mark.xfail(reason="8x8 coeff_abs_level not implemented")
    def test_coeff_abs_level_8x8_large_values(self):
        """8x8 blocks can have larger coefficient magnitudes."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_coeff_abs_level_minus1_8x8

        # Bitstream designed to produce large level value
        # (using escape coding for values >= 14)
        data = bytes([0xFF] * 100)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_coeff_abs_level_minus1_8x8(
            decoder, contexts, block_cat=5,
            num_decode_abs_level_eq1=0, num_decode_abs_level_gt1=0
        )

        # Should be able to decode large values
        assert result >= 0


# =============================================================================
# Test: 8x8 zigzag scan order (different from 4x4)
# =============================================================================

class TestZigzagScan8x8:
    """Tests for 8x8 zigzag scan order.

    H.264 Table 8-15 defines the diagonal scan for 8x8 blocks.
    This differs from the 4x4 scan pattern.
    """

    @pytest.mark.xfail(reason="get_scan_order_8x8 not implemented")
    def test_get_scan_order_8x8_exists(self):
        """get_scan_order_8x8 function should exist."""
        from entropy.cabac_residual import get_scan_order_8x8

        assert callable(get_scan_order_8x8)

    @pytest.mark.xfail(reason="get_scan_order_8x8 not implemented")
    def test_scan_8x8_has_64_positions(self):
        """8x8 scan order should have 64 positions."""
        from entropy.cabac_residual import get_scan_order_8x8

        scan = get_scan_order_8x8()
        assert len(scan) == 64

    @pytest.mark.xfail(reason="get_scan_order_8x8 not implemented")
    def test_scan_8x8_covers_all_positions(self):
        """8x8 scan should cover all positions exactly once."""
        from entropy.cabac_residual import get_scan_order_8x8

        scan = get_scan_order_8x8()
        assert set(scan) == set(range(64))
        assert len(scan) == len(set(scan))

    @pytest.mark.xfail(reason="get_scan_order_8x8 not implemented")
    def test_scan_8x8_starts_at_dc(self):
        """8x8 scan starts at DC position (0,0)."""
        from entropy.cabac_residual import get_scan_order_8x8

        scan = get_scan_order_8x8()
        assert scan[0] == 0  # (0,0) position

    @pytest.mark.xfail(reason="get_scan_order_8x8 not implemented")
    def test_scan_8x8_ends_at_corner(self):
        """8x8 scan ends at position (7,7)."""
        from entropy.cabac_residual import get_scan_order_8x8

        scan = get_scan_order_8x8()
        assert scan[-1] == 63  # (7,7) position = 7*8+7 = 63

    @pytest.mark.xfail(reason="get_scan_order_8x8 not implemented")
    def test_scan_8x8_diagonal_pattern(self):
        """8x8 scan follows diagonal pattern per H.264 Table 8-15."""
        from entropy.cabac_residual import get_scan_order_8x8

        scan = get_scan_order_8x8()

        # First 8 positions (first two diagonals):
        # (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2)
        # = 0, 1, 8, 16, 9, 2, 3, 10
        expected_first_8 = [0, 1, 8, 16, 9, 2, 3, 10]
        assert list(scan[:8]) == expected_first_8

    @pytest.mark.xfail(reason="ZIGZAG_8X8_CABAC not in cabac_residual")
    def test_zigzag_8x8_constant_exists(self):
        """ZIGZAG_8X8_CABAC constant should exist in cabac_residual."""
        from entropy.cabac_residual import ZIGZAG_8X8_CABAC

        assert len(ZIGZAG_8X8_CABAC) == 64


# =============================================================================
# Test: Context model initialization for 8x8 blocks
# =============================================================================

class TestContextModel8x8Initialization:
    """Tests for 8x8-specific context model initialization.

    8x8 transform adds additional contexts that need proper
    initialization based on (m,n) parameters from spec tables.
    """

    @pytest.mark.xfail(reason="8x8 context ranges not defined")
    def test_num_contexts_includes_8x8(self):
        """Total context count should include 8x8 contexts.

        Base CABAC has 460 contexts. High profile 8x8 adds more.
        """
        from entropy.cabac_context import NUM_CONTEXTS_8X8

        # 8x8 requires additional contexts
        assert NUM_CONTEXTS_8X8 >= 460

    @pytest.mark.xfail(reason="8x8 context init not implemented")
    def test_init_context_models_8x8_exists(self):
        """init_context_models_8x8 function should exist."""
        from entropy.cabac_context import init_context_models_8x8

        assert callable(init_context_models_8x8)

    @pytest.mark.xfail(reason="8x8 context init not implemented")
    def test_init_context_models_8x8_returns_extended_list(self):
        """8x8 initialization returns extended context list."""
        from entropy.cabac_context import init_context_models_8x8

        contexts = init_context_models_8x8(slice_type=0, slice_qp=26)

        # Should have all base + 8x8 contexts
        assert len(contexts) >= 460

    @pytest.mark.xfail(reason="8x8 context init not implemented")
    def test_sig_coeff_flag_8x8_contexts_initialized(self):
        """significant_coeff_flag 8x8 contexts properly initialized."""
        from entropy.cabac_context import (
            init_context_models_8x8,
            CTX_SIG_COEFF_FLAG_8X8_START,
        )

        contexts = init_context_models_8x8(slice_type=0, slice_qp=26)

        # Check some 8x8 sig_coeff contexts
        ctx_start = CTX_SIG_COEFF_FLAG_8X8_START
        for i in range(10):
            ctx = contexts[ctx_start + i]
            assert 0 <= ctx.pStateIdx <= 63
            assert ctx.valMPS in (0, 1)

    @pytest.mark.xfail(reason="8x8 context init params not defined")
    def test_8x8_init_params_from_spec_tables(self):
        """8x8 context init uses m,n from H.264 tables."""
        from entropy.cabac_context import (
            INIT_PARAMS_SIG_COEFF_8X8_I,
            INIT_PARAMS_SIG_COEFF_8X8_P,
            INIT_PARAMS_SIG_COEFF_8X8_B,
        )

        # I-slice, P-slice, B-slice each have different init params
        assert len(INIT_PARAMS_SIG_COEFF_8X8_I) > 0
        assert len(INIT_PARAMS_SIG_COEFF_8X8_P) > 0
        assert len(INIT_PARAMS_SIG_COEFF_8X8_B) > 0


# =============================================================================
# Test: transform_8x8_flag parsing
# =============================================================================

class TestTransform8x8FlagParsing:
    """Tests for transform_8x8_flag CABAC decoding.

    transform_8x8_flag indicates whether 8x8 transform is used
    for current macroblock (when allowed by PPS).
    """

    @pytest.mark.xfail(reason="transform_8x8_flag parsing not implemented")
    def test_decode_transform_8x8_flag_exists(self):
        """decode_transform_8x8_flag function should exist."""
        from entropy.cabac_residual import decode_transform_8x8_flag

        assert callable(decode_transform_8x8_flag)

    @pytest.mark.xfail(reason="transform_8x8_flag parsing not implemented")
    def test_transform_8x8_flag_ctx_idx(self):
        """transform_8x8_flag uses specific context index."""
        from entropy.cabac_context import CTX_TRANSFORM_8X8_FLAG_START

        # Context index for transform_8x8_flag
        assert CTX_TRANSFORM_8X8_FLAG_START >= 0

    @pytest.mark.xfail(reason="transform_8x8_flag parsing not implemented")
    def test_transform_8x8_flag_returns_binary(self):
        """transform_8x8_flag returns 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_transform_8x8_flag

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        neighbor_info = {
            'left_transform_8x8_flag': 0,
            'top_transform_8x8_flag': 0,
            'left_available': True,
            'top_available': True,
        }

        result = decode_transform_8x8_flag(decoder, contexts, neighbor_info)

        assert result in (0, 1)

    @pytest.mark.xfail(reason="transform_8x8_flag parsing not implemented")
    def test_transform_8x8_flag_ctx_depends_on_neighbors(self):
        """transform_8x8_flag context depends on neighbors."""
        from entropy.cabac_residual import get_transform_8x8_flag_ctx_inc

        # Both neighbors use 4x4 -> ctxIdxInc = 0
        ctx_inc = get_transform_8x8_flag_ctx_inc(
            left_available=True, left_8x8=0,
            top_available=True, top_8x8=0
        )
        assert ctx_inc == 0

        # Both neighbors use 8x8 -> ctxIdxInc = 2
        ctx_inc = get_transform_8x8_flag_ctx_inc(
            left_available=True, left_8x8=1,
            top_available=True, top_8x8=1
        )
        assert ctx_inc == 2

        # Left unavailable, top uses 8x8
        ctx_inc = get_transform_8x8_flag_ctx_inc(
            left_available=False, left_8x8=0,
            top_available=True, top_8x8=1
        )
        # Default when unavailable is 0
        assert ctx_inc == 1


# =============================================================================
# Test: Complete 8x8 block decoding flow
# =============================================================================

class TestComplete8x8DecodingFlow:
    """Integration tests for complete 8x8 block decoding."""

    @pytest.mark.xfail(reason="Complete 8x8 decoding not implemented")
    def test_decode_single_nonzero_coefficient(self):
        """Decode 8x8 block with single DC coefficient."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        # Bitstream encoding: DC=5, all other coeffs=0
        data = bytes([0x00, 0x80, 0x00, 0x40] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_8x8(decoder, contexts, block_cat=5)

        assert len(result) == 64
        # At least verifies we get a valid array

    @pytest.mark.xfail(reason="Complete 8x8 decoding not implemented")
    def test_decode_sparse_8x8_block(self):
        """Decode 8x8 block with sparse coefficients."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        data = bytes([0x55, 0xAA, 0x55, 0xAA] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_8x8(decoder, contexts, block_cat=5)

        assert len(result) == 64
        assert result.dtype == np.int32

    @pytest.mark.xfail(reason="Complete 8x8 decoding not implemented")
    def test_decode_8x8_with_negative_coefficients(self):
        """8x8 coefficients can be negative."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        data = bytes([0xAA, 0x55, 0xAA, 0x55] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_8x8(decoder, contexts, block_cat=5)

        # Coefficients can be positive or negative
        # Just check we got valid output
        assert len(result) == 64

    @pytest.mark.xfail(reason="Complete 8x8 decoding not implemented")
    def test_decode_8x8_preserves_scan_order(self):
        """Decoded coefficients placed according to 8x8 scan order."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8, ZIGZAG_8X8_CABAC

        data = bytes([0x80, 0x00, 0x80, 0x00] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_8x8(decoder, contexts, block_cat=5)

        # Verify result is in scan order
        assert len(result) == len(ZIGZAG_8X8_CABAC)


# =============================================================================
# Test: 8x8 field vs frame scan
# =============================================================================

class TestFieldVsFrameScan8x8:
    """Tests for field vs frame scanning in 8x8 blocks.

    Interlaced video uses different scan patterns.
    """

    @pytest.mark.xfail(reason="8x8 field scan not implemented")
    def test_get_scan_order_8x8_frame(self):
        """Frame mode uses diagonal scan for 8x8."""
        from entropy.cabac_residual import get_scan_order_8x8

        scan = get_scan_order_8x8(field_scan=False)
        assert len(scan) == 64
        # Frame scan starts with diagonal pattern
        assert scan[0] == 0
        assert scan[1] == 1
        assert scan[2] == 8

    @pytest.mark.xfail(reason="8x8 field scan not implemented")
    def test_get_scan_order_8x8_field(self):
        """Field mode uses different scan pattern."""
        from entropy.cabac_residual import get_scan_order_8x8

        scan = get_scan_order_8x8(field_scan=True)
        assert len(scan) == 64
        # Field scan has different pattern (Table 8-14)

    @pytest.mark.xfail(reason="8x8 field scan not implemented")
    def test_field_scan_8x8_exists(self):
        """Field scan table for 8x8 should exist."""
        from entropy.cabac_residual import ZIGZAG_8X8_FIELD

        assert len(ZIGZAG_8X8_FIELD) == 64


# =============================================================================
# Test: 8x8 chroma blocks (4:4:4 chroma format)
# =============================================================================

class TestChroma8x8Blocks:
    """Tests for 8x8 chroma blocks in 4:4:4 chroma format.

    4:4:4 chroma format allows 8x8 chroma transforms.
    """

    @pytest.mark.xfail(reason="8x8 chroma not implemented")
    def test_cb_8x8_block_category(self):
        """Block category 7 is Cb 8x8 for 4:4:4."""
        from entropy.cabac_residual import BLOCK_CAT_8X8_CB

        assert BLOCK_CAT_8X8_CB == 7

    @pytest.mark.xfail(reason="8x8 chroma not implemented")
    def test_cr_8x8_block_category(self):
        """Block category 8 is Cr 8x8 for 4:4:4."""
        from entropy.cabac_residual import BLOCK_CAT_8X8_CR

        assert BLOCK_CAT_8X8_CR == 8

    @pytest.mark.xfail(reason="8x8 chroma not implemented")
    def test_decode_cb_8x8_block(self):
        """Decode Cb 8x8 block."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        data = bytes([0x80, 0x40, 0x20, 0x10] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_8x8(decoder, contexts, block_cat=7)

        assert len(result) == 64

    @pytest.mark.xfail(reason="8x8 chroma not implemented")
    def test_decode_cr_8x8_block(self):
        """Decode Cr 8x8 block."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        data = bytes([0x80, 0x40, 0x20, 0x10] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_residual_block_8x8(decoder, contexts, block_cat=8)

        assert len(result) == 64


# =============================================================================
# Test: Table 9-43 ctxIdxInc values
# =============================================================================

class TestTable9_43CtxIdxInc:
    """Tests for H.264 Table 9-43 context index increments.

    Table 9-43 specifies ctxIdxInc for significant_coeff_flag
    and last_significant_coeff_flag in 8x8 blocks.
    """

    @pytest.mark.xfail(reason="Table 9-43 values not implemented")
    def test_table_9_43_sig_coeff_values(self):
        """Verify ctxIdxInc values from Table 9-43 for sig_coeff."""
        from entropy.cabac_residual import SIG_COEFF_FLAG_CTX_INC_8X8

        # Sample values from H.264 Table 9-43 (frame scan)
        # Position 0: ctxIdxInc = 0
        # Position 1: ctxIdxInc = 1
        # Position 2: ctxIdxInc = 2
        # ... (varies by diagonal)
        expected_first_10 = [0, 1, 2, 3, 4, 5, 5, 4, 4, 3]

        for i, expected in enumerate(expected_first_10):
            assert SIG_COEFF_FLAG_CTX_INC_8X8[i] == expected, \
                f"Position {i}: expected {expected}, got {SIG_COEFF_FLAG_CTX_INC_8X8[i]}"

    @pytest.mark.xfail(reason="Table 9-43 values not implemented")
    def test_table_9_43_last_sig_values(self):
        """Verify ctxIdxInc values from Table 9-43 for last_sig."""
        from entropy.cabac_residual import LAST_SIG_COEFF_FLAG_CTX_INC_8X8

        # Sample values from H.264 Table 9-43 (frame scan)
        # Different from sig_coeff pattern
        expected_first_10 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        for i, expected in enumerate(expected_first_10):
            assert LAST_SIG_COEFF_FLAG_CTX_INC_8X8[i] == expected, \
                f"Position {i}: expected {expected}, got {LAST_SIG_COEFF_FLAG_CTX_INC_8X8[i]}"

    @pytest.mark.xfail(reason="Table 9-43 values not implemented")
    def test_ctx_idx_inc_range(self):
        """ctxIdxInc values should be within valid range."""
        from entropy.cabac_residual import (
            SIG_COEFF_FLAG_CTX_INC_8X8,
            LAST_SIG_COEFF_FLAG_CTX_INC_8X8,
        )

        for val in SIG_COEFF_FLAG_CTX_INC_8X8:
            assert 0 <= val < 64, f"sig_coeff ctxIdxInc out of range: {val}"

        for val in LAST_SIG_COEFF_FLAG_CTX_INC_8X8:
            assert 0 <= val < 64, f"last_sig ctxIdxInc out of range: {val}"


# =============================================================================
# Test: Integration with existing 4x4 infrastructure
# =============================================================================

class TestIntegrationWith4x4:
    """Tests for integration between 4x4 and 8x8 decoding."""

    @pytest.mark.xfail(reason="8x8 context offsets not defined in BLOCK_CAT_CTX_OFFSETS")
    def test_decode_residual_block_handles_8x8_category(self):
        """Residual decoder should recognize block category 5 as 8x8."""
        from entropy.cabac_residual import BLOCK_CAT_CTX_OFFSETS

        # Block category 5 (Luma 8x8) should have dedicated context offsets
        # Currently BLOCK_CAT_CTX_OFFSETS only has categories 0-4
        assert 5 in BLOCK_CAT_CTX_OFFSETS, "Block category 5 (Luma 8x8) not defined"

        # Verify it has proper 8x8 offsets
        offsets = BLOCK_CAT_CTX_OFFSETS[5]
        sig_offset, last_offset, abs_offset = offsets

        # 8x8 should have distinct offsets from 4x4 categories
        offsets_4x4 = BLOCK_CAT_CTX_OFFSETS[2]
        assert offsets != offsets_4x4, "8x8 should have different offsets than 4x4"

    @pytest.mark.xfail(reason="Block category detection not implemented")
    def test_is_8x8_block_category(self):
        """Function to check if block category is 8x8."""
        from entropy.cabac_residual import is_8x8_block_category

        assert not is_8x8_block_category(0)  # Luma DC
        assert not is_8x8_block_category(1)  # Luma AC
        assert not is_8x8_block_category(2)  # Luma 4x4
        assert not is_8x8_block_category(3)  # Chroma DC
        assert not is_8x8_block_category(4)  # Chroma AC
        assert is_8x8_block_category(5)      # Luma 8x8
        assert is_8x8_block_category(7)      # Cb 8x8
        assert is_8x8_block_category(8)      # Cr 8x8

    @pytest.mark.xfail(reason="get_max_coeff not implemented")
    def test_get_max_coeff_for_block_category(self):
        """Get maximum coefficients for block category."""
        from entropy.cabac_residual import get_max_coeff_for_category

        assert get_max_coeff_for_category(0) == 16   # Luma DC
        assert get_max_coeff_for_category(1) == 15   # Luma AC
        assert get_max_coeff_for_category(2) == 16   # Luma 4x4
        assert get_max_coeff_for_category(3) == 4    # Chroma DC (4:2:0)
        assert get_max_coeff_for_category(4) == 15   # Chroma AC
        assert get_max_coeff_for_category(5) == 64   # Luma 8x8


# =============================================================================
# Test: High Profile 8x8 significance map advanced features
# =============================================================================

class TestSignificanceMap8x8Advanced:
    """Advanced tests for 8x8 significance map decoding.

    The significance map for 8x8 blocks differs significantly from 4x4:
    - 63 significance decisions (positions 0-62, 63 is implicit)
    - Different context model selection per position
    - Different scan orders for frame vs field
    - Requires High profile support

    H.264 Spec Reference:
    - Section 9.3.3.1.3 - Significance map decoding
    - Table 9-43 - ctxIdxInc for 8x8 blocks
    """

    @pytest.mark.xfail(reason="8x8 significance map not implemented")
    def test_sig_map_8x8_decoding_flow(self):
        """Complete significance map decoding for 8x8 block."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_significance_map_8x8

        data = bytes([0x80, 0x40, 0x20, 0x10] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        sig_map = decode_significance_map_8x8(decoder, contexts, block_cat=5)

        # Should return 64 flags (or list of significant positions)
        assert len(sig_map) == 64

    @pytest.mark.xfail(reason="8x8 significance map not implemented")
    def test_sig_map_8x8_all_zero(self):
        """Significance map with all zeros (empty block)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_significance_map_8x8

        # Bitstream that should decode to all zeros
        data = bytes([0x00] * 100)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        sig_map = decode_significance_map_8x8(decoder, contexts, block_cat=5)

        # Check we got a valid map
        assert all(s in (0, 1) for s in sig_map)

    @pytest.mark.xfail(reason="8x8 significance map not implemented")
    def test_sig_map_8x8_dc_only(self):
        """Significance map with only DC coefficient."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_significance_map_8x8

        data = bytes([0xFF, 0x80] + [0x00] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        sig_map = decode_significance_map_8x8(decoder, contexts, block_cat=5)

        # First position (DC) may be significant
        assert isinstance(sig_map[0], (int, bool))

    @pytest.mark.xfail(reason="8x8 significance map not implemented")
    def test_sig_map_8x8_last_sig_terminates_scan(self):
        """last_significant_coeff_flag=1 terminates significance scan."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_significance_map_8x8

        data = bytes([0xFF, 0xFF, 0x80, 0x00] * 30)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        sig_map = decode_significance_map_8x8(decoder, contexts, block_cat=5)

        # Should terminate at some position <= 63
        # Remaining positions are implicitly 0
        assert len(sig_map) == 64

    @pytest.mark.xfail(reason="8x8 significance map not implemented")
    def test_sig_map_8x8_position_63_implicit_last(self):
        """Position 63 is implicitly significant if scan reaches it."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_significance_map_8x8

        # Construct bitstream that reaches position 63
        data = bytes([0xFF] * 100)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        sig_map = decode_significance_map_8x8(decoder, contexts, block_cat=5)

        # Position 63 doesn't need last_sig_flag if reached
        assert len(sig_map) == 64

    @pytest.mark.xfail(reason="8x8 context selection not implemented")
    def test_sig_flag_8x8_context_varies_by_position(self):
        """significant_coeff_flag context varies by scan position."""
        from entropy.cabac_residual import get_sig_coeff_8x8_ctx_idx

        # Different positions should give different contexts
        ctx_0 = get_sig_coeff_8x8_ctx_idx(scan_idx=0, block_cat=5)
        ctx_10 = get_sig_coeff_8x8_ctx_idx(scan_idx=10, block_cat=5)
        ctx_30 = get_sig_coeff_8x8_ctx_idx(scan_idx=30, block_cat=5)

        # Not all should be the same (though some may be)
        contexts = [ctx_0, ctx_10, ctx_30]
        assert len(set(contexts)) >= 1  # At least valid contexts

    @pytest.mark.xfail(reason="8x8 context selection not implemented")
    def test_last_sig_flag_8x8_context_varies_by_position(self):
        """last_significant_coeff_flag context varies by scan position."""
        from entropy.cabac_residual import get_last_sig_coeff_8x8_ctx_idx

        ctx_0 = get_last_sig_coeff_8x8_ctx_idx(scan_idx=0, block_cat=5)
        ctx_20 = get_last_sig_coeff_8x8_ctx_idx(scan_idx=20, block_cat=5)
        ctx_50 = get_last_sig_coeff_8x8_ctx_idx(scan_idx=50, block_cat=5)

        # Should return valid context indices
        assert all(isinstance(c, int) for c in [ctx_0, ctx_20, ctx_50])


class TestCoeffLevelDecoding8x8Advanced:
    """Advanced tests for 8x8 coefficient level decoding.

    After the significance map, non-zero coefficient levels are
    decoded in reverse scan order using coeff_abs_level_minus1.

    H.264 Spec Reference: Section 9.3.3.1.3
    """

    @pytest.mark.xfail(reason="8x8 level decoding not implemented")
    def test_coeff_level_8x8_reverse_scan_order(self):
        """Coefficient levels decoded in reverse scan order."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        data = bytes([0x80, 0x40, 0x20, 0x10] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        coeffs = decode_residual_block_8x8(decoder, contexts, block_cat=5)

        # Coefficients should be in scan order (not reverse)
        # But levels are decoded in reverse internally
        assert len(coeffs) == 64

    @pytest.mark.xfail(reason="8x8 level decoding not implemented")
    def test_coeff_level_8x8_context_adaptation(self):
        """Level context adapts based on previously decoded levels."""
        from entropy.cabac_residual import get_coeff_abs_level_8x8_ctx_idx

        # First level: numDecodedAbsLevel{Eq1,Gt1} = 0
        ctx_first = get_coeff_abs_level_8x8_ctx_idx(
            block_cat=5, num_eq1=0, num_gt1=0, bin_idx=0
        )

        # After one level=1: numDecodedAbsLevelEq1 = 1
        ctx_after_eq1 = get_coeff_abs_level_8x8_ctx_idx(
            block_cat=5, num_eq1=1, num_gt1=0, bin_idx=0
        )

        # After one level>1: numDecodedAbsLevelGt1 = 1
        ctx_after_gt1 = get_coeff_abs_level_8x8_ctx_idx(
            block_cat=5, num_eq1=0, num_gt1=1, bin_idx=0
        )

        # Contexts should differ based on history
        assert isinstance(ctx_first, int)
        assert isinstance(ctx_after_eq1, int)
        assert isinstance(ctx_after_gt1, int)

    @pytest.mark.xfail(reason="8x8 level decoding not implemented")
    def test_coeff_level_8x8_prefix_suffix_split(self):
        """Large levels use prefix (TU) + suffix (bypass EG)."""
        from entropy.cabac_residual import decode_coeff_abs_level_8x8_full

        # Level encoding: prefix=14 triggers suffix coding
        # prefix < 14: value = prefix
        # prefix = 14: value = 14 + suffix (Exp-Golomb in bypass)
        assert callable(decode_coeff_abs_level_8x8_full)

    @pytest.mark.xfail(reason="8x8 level decoding not implemented")
    def test_coeff_level_8x8_sign_bypass(self):
        """Coefficient signs decoded in bypass mode."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_residual import decode_residual_block_8x8

        data = bytes([0xAA, 0x55, 0xAA, 0x55] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        coeffs = decode_residual_block_8x8(decoder, contexts, block_cat=5)

        # Coefficients can be positive or negative
        # (signs are decoded via bypass)
        assert all(isinstance(c, (int, np.integer)) for c in coeffs)

    @pytest.mark.xfail(reason="8x8 level decoding not implemented")
    def test_coeff_level_8x8_max_bins_prefix(self):
        """Prefix part uses maximum 14 bins (TU14)."""
        from entropy.cabac_binarize import COEFF_ABS_LEVEL_PREFIX_MAX_BINS

        # H.264 uses truncated unary with max 14 bins
        assert COEFF_ABS_LEVEL_PREFIX_MAX_BINS == 14


class Test8x8HighProfileIntegration:
    """Integration tests for 8x8 transform in High profile decoding."""

    @pytest.mark.xfail(reason="High profile 8x8 not implemented")
    def test_pps_transform_8x8_mode_flag(self):
        """PPS transform_8x8_mode_flag enables 8x8 transform."""
        from entropy.cabac_macroblock import is_8x8_transform_allowed

        pps = {'transform_8x8_mode_flag': 1, 'profile_idc': 100}
        assert is_8x8_transform_allowed(pps)

        pps_disabled = {'transform_8x8_mode_flag': 0, 'profile_idc': 100}
        assert not is_8x8_transform_allowed(pps_disabled)

    @pytest.mark.xfail(reason="High profile 8x8 not implemented")
    def test_8x8_only_for_specific_mb_types(self):
        """8x8 transform only for I_NxN and inter without 8x8 sub-partitions."""
        from entropy.cabac_macroblock import can_use_8x8_transform

        # I_4x4 can use 8x8 (becomes I_8x8)
        assert can_use_8x8_transform(mb_type=0, slice_type=2, sub_mb_types=None)

        # I_16x16 cannot use 8x8
        assert not can_use_8x8_transform(mb_type=1, slice_type=2, sub_mb_types=None)

        # P_L0_16x16 can use 8x8
        assert can_use_8x8_transform(mb_type=0, slice_type=0, sub_mb_types=None)

        # P_8x8 with 8x8 sub-partitions can use 8x8
        assert can_use_8x8_transform(
            mb_type=3, slice_type=0, sub_mb_types=[0, 0, 0, 0]
        )  # All P_L0_8x8

        # P_8x8 with 4x4 sub-partitions cannot
        assert not can_use_8x8_transform(
            mb_type=3, slice_type=0, sub_mb_types=[3, 3, 3, 3]
        )  # All P_L0_4x4

    @pytest.mark.xfail(reason="High profile 8x8 not implemented")
    def test_8x8_intra_prediction_mode_decoding(self):
        """I_8x8 uses different intra prediction mode decoding."""
        from entropy.cabac_macroblock import decode_intra_8x8_pred_modes_cabac

        # I_8x8 has 4 prediction modes (one per 8x8 block)
        # vs I_4x4 which has 16 prediction modes
        assert callable(decode_intra_8x8_pred_modes_cabac)

    @pytest.mark.xfail(reason="High profile 8x8 not implemented")
    def test_8x8_residual_placed_correctly(self):
        """8x8 residual placed in correct position in macroblock."""
        from entropy.cabac_macroblock import place_8x8_residual_in_mb

        # 64 coefficients placed in 8x8 region
        coeffs = np.zeros(64, dtype=np.int32)
        coeffs[0] = 100  # DC

        mb_residual = place_8x8_residual_in_mb(
            coeffs, block_idx=0  # First 8x8 block (top-left)
        )

        assert mb_residual.shape == (16, 16)
        # DC should be in top-left 8x8
        assert mb_residual[0, 0] == 100

    @pytest.mark.xfail(reason="High profile 8x8 not implemented")
    def test_cbp_interpretation_for_8x8(self):
        """CBP interpretation differs slightly for 8x8."""
        from entropy.cabac_macroblock import interpret_cbp_for_8x8

        # CBP luma still 4 bits (one per 8x8)
        # When using 8x8 transform, each bit maps to one 8x8 block
        cbp_info = interpret_cbp_for_8x8(cbp_luma=0b1010, cbp_chroma=2)

        assert cbp_info['block_0_coded'] is False
        assert cbp_info['block_1_coded'] is True
        assert cbp_info['block_2_coded'] is False
        assert cbp_info['block_3_coded'] is True


class TestScalingLists8x8:
    """Tests for 8x8 scaling list support in CABAC residual decoding.

    High profile supports 8x8 scaling lists for dequantization.
    """

    @pytest.mark.xfail(reason="8x8 scaling lists not implemented")
    def test_8x8_scaling_list_size(self):
        """8x8 scaling lists have 64 elements."""
        from parameters.scaling_lists import get_default_8x8_scaling_list

        # Intra 8x8
        list_intra = get_default_8x8_scaling_list(is_intra=True)
        assert len(list_intra) == 64

        # Inter 8x8
        list_inter = get_default_8x8_scaling_list(is_intra=False)
        assert len(list_inter) == 64

    @pytest.mark.xfail(reason="8x8 scaling lists not implemented")
    def test_8x8_scaling_list_scan_order(self):
        """8x8 scaling list uses specific scan order."""
        from parameters.scaling_lists import SCALING_LIST_8X8_SCAN

        assert len(SCALING_LIST_8X8_SCAN) == 64

    @pytest.mark.xfail(reason="8x8 scaling lists not implemented")
    def test_decode_8x8_scaling_list_from_pps(self):
        """PPS can contain custom 8x8 scaling lists."""
        from parameters.scaling_lists import decode_scaling_list_8x8

        # Delta values (like CAVLC decoding)
        data = bytes([0x00] * 64)  # All deltas = 0
        reader = BitReader(data)

        scaling_list = decode_scaling_list_8x8(reader, use_default=False)

        assert len(scaling_list) == 64
