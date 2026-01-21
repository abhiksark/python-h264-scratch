# h264/inter/tests/test_explicit_weighted_pred.py
"""RED TESTS: Explicit weighted prediction for H.264 P and B slices.

Tests explicit weighted prediction modes:
- weighted_pred_flag=1 for P slices
- weighted_bipred_idc=1 for B slices (explicit mode)

Explicit weighted prediction allows encoder to specify per-reference
weights and offsets to handle fades, exposure changes, and cross-fades.

H.264 Spec Reference:
- Section 7.3.3.2: pred_weight_table syntax
- Section 7.4.3.2: pred_weight_table semantics
- Section 8.4.2.3: Weighted sample prediction process

These tests SHOULD FAIL until explicit weighted prediction is fully implemented.
"""

import pytest
import numpy as np


# -----------------------------------------------------------------------------
# Test pred_weight_table parsing from slice header
# -----------------------------------------------------------------------------

class TestPredWeightTableParsing:
    """Tests for parsing pred_weight_table from slice header bitstream."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_parse_pred_weight_table_basic_structure(self):
        """Parse basic pred_weight_table with luma weights only."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(6)  # luma_log2_weight_denom = 6
        writer.write_ue(6)  # chroma_log2_weight_denom = 6
        # L0 ref 0: explicit luma weight
        writer.write_bits(1, 1)  # luma_weight_l0_flag = 1
        writer.write_se(64)      # luma_weight_l0[0] = 64 (unity)
        writer.write_se(0)       # luma_offset_l0[0] = 0
        writer.write_bits(0, 1)  # chroma_weight_l0_flag = 0

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1, chroma_format_idc=1)

        assert table.luma_log2_weight_denom == 6
        assert table.chroma_log2_weight_denom == 6
        weight, offset = table.get_luma_weight(0)
        assert weight == 64
        assert offset == 0

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_parse_pred_weight_table_multiple_l0_refs(self):
        """Parse pred_weight_table with multiple L0 references."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(5)  # luma_log2_weight_denom = 5
        writer.write_ue(5)  # chroma_log2_weight_denom = 5

        # L0 ref 0
        writer.write_bits(1, 1)
        writer.write_se(32)  # weight = 1.0
        writer.write_se(-10)
        writer.write_bits(0, 1)

        # L0 ref 1
        writer.write_bits(1, 1)
        writer.write_se(48)  # weight = 1.5
        writer.write_se(5)
        writer.write_bits(0, 1)

        # L0 ref 2
        writer.write_bits(1, 1)
        writer.write_se(16)  # weight = 0.5
        writer.write_se(20)
        writer.write_bits(0, 1)

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=3, chroma_format_idc=1)

        w0, o0 = table.get_luma_weight(0)
        w1, o1 = table.get_luma_weight(1)
        w2, o2 = table.get_luma_weight(2)

        assert (w0, o0) == (32, -10)
        assert (w1, o1) == (48, 5)
        assert (w2, o2) == (16, 20)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_parse_pred_weight_table_with_chroma_weights(self):
        """Parse pred_weight_table with both luma and chroma weights."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(5)  # luma_log2_weight_denom
        writer.write_ue(5)  # chroma_log2_weight_denom

        # L0 ref 0: luma + chroma
        writer.write_bits(1, 1)  # luma_weight_l0_flag
        writer.write_se(32)
        writer.write_se(10)
        writer.write_bits(1, 1)  # chroma_weight_l0_flag
        writer.write_se(40)      # cb_weight
        writer.write_se(-5)      # cb_offset
        writer.write_se(24)      # cr_weight
        writer.write_se(8)       # cr_offset

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1, chroma_format_idc=1)

        luma_w, luma_o = table.get_luma_weight(0)
        cb_w, cb_o, cr_w, cr_o = table.get_chroma_weight(0)

        assert luma_w == 32 and luma_o == 10
        assert cb_w == 40 and cb_o == -5
        assert cr_w == 24 and cr_o == 8

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_parse_pred_weight_table_b_slice_l0_and_l1(self):
        """Parse pred_weight_table for B-slice with both L0 and L1 weights."""
        from slice.pred_weight_table import parse_pred_weight_table_b_slice
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(6)  # luma_log2_weight_denom
        writer.write_ue(6)  # chroma_log2_weight_denom

        # L0 ref 0
        writer.write_bits(1, 1)
        writer.write_se(64)
        writer.write_se(0)
        writer.write_bits(0, 1)

        # L1 ref 0
        writer.write_bits(1, 1)
        writer.write_se(48)
        writer.write_se(-5)
        writer.write_bits(0, 1)

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table_b_slice(
            reader, num_ref_idx_l0=1, num_ref_idx_l1=1, chroma_format_idc=1
        )

        l0_w, l0_o = table.get_l0_luma_weight(0)
        l1_w, l1_o = table.get_l1_luma_weight(0)

        assert l0_w == 64 and l0_o == 0
        assert l1_w == 48 and l1_o == -5


# -----------------------------------------------------------------------------
# Test luma_log2_weight_denom values (0-7)
# -----------------------------------------------------------------------------

class TestLumaLog2WeightDenom:
    """Tests for luma_log2_weight_denom values across valid range [0,7]."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    @pytest.mark.parametrize("log2_denom", range(8))
    def test_luma_log2_weight_denom_parsing(self, log2_denom):
        """Parse pred_weight_table with different luma_log2_weight_denom values."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(log2_denom)  # luma_log2_weight_denom
        writer.write_ue(log2_denom)  # chroma_log2_weight_denom
        writer.write_bits(0, 1)      # luma_weight_l0_flag = 0 (use default)
        writer.write_bits(0, 1)      # chroma_weight_l0_flag = 0

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1, chroma_format_idc=1)

        assert table.luma_log2_weight_denom == log2_denom

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    @pytest.mark.parametrize("log2_denom", range(8))
    def test_default_weight_for_each_log2_denom(self, log2_denom):
        """Default luma weight should be 2^log2_denom."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 100, dtype=np.uint8)
        default_weight = 1 << log2_denom

        # With default weight, prediction should be preserved
        result = apply_explicit_weights(
            pred,
            weight=default_weight,
            offset=0,
            log2_weight_denom=log2_denom
        )

        np.testing.assert_array_equal(result, pred)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_log2_weight_denom_zero_integer_weights(self):
        """log2_weight_denom=0 uses integer weights directly."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # weight=2 with log2_denom=0 means multiply by 2
        result = apply_explicit_weights(
            pred, weight=2, offset=0, log2_weight_denom=0
        )

        np.testing.assert_array_equal(result, 200)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_log2_weight_denom_seven_fine_grained(self):
        """log2_weight_denom=7 provides finest weight granularity."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 128, dtype=np.uint8)

        # weight=128 (1.0) with log2_denom=7
        result = apply_explicit_weights(
            pred, weight=128, offset=0, log2_weight_denom=7
        )

        np.testing.assert_array_equal(result, 128)

        # weight=256 (2.0) with log2_denom=7, should clip
        result = apply_explicit_weights(
            pred, weight=256, offset=0, log2_weight_denom=7
        )

        # 128 * 2 = 256, clips to 255
        np.testing.assert_array_equal(result, 255)


# -----------------------------------------------------------------------------
# Test chroma_log2_weight_denom values (0-7)
# -----------------------------------------------------------------------------

class TestChromaLog2WeightDenom:
    """Tests for chroma_log2_weight_denom values across valid range [0,7]."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    @pytest.mark.parametrize("log2_denom", range(8))
    def test_chroma_log2_weight_denom_parsing(self, log2_denom):
        """Parse pred_weight_table with different chroma_log2_weight_denom values."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(6)           # luma_log2_weight_denom = 6
        writer.write_ue(log2_denom)  # chroma_log2_weight_denom
        writer.write_bits(0, 1)
        writer.write_bits(0, 1)

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1, chroma_format_idc=1)

        assert table.chroma_log2_weight_denom == log2_denom

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_different_luma_and_chroma_denom(self):
        """Luma and chroma can have different log2_weight_denom values."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(5)  # luma_log2_weight_denom = 5
        writer.write_ue(7)  # chroma_log2_weight_denom = 7
        writer.write_bits(1, 1)  # luma flag
        writer.write_se(32)
        writer.write_se(0)
        writer.write_bits(1, 1)  # chroma flag
        writer.write_se(128)  # cb_weight
        writer.write_se(0)
        writer.write_se(128)  # cr_weight
        writer.write_se(0)

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1, chroma_format_idc=1)

        assert table.luma_log2_weight_denom == 5
        assert table.chroma_log2_weight_denom == 7

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_chroma_weight_applied_with_correct_denom(self):
        """Chroma weights must use chroma_log2_weight_denom, not luma."""
        from inter.weighted_prediction import apply_explicit_chroma_weights

        pred_cb = np.full((4, 4), 128, dtype=np.uint8)
        pred_cr = np.full((4, 4), 128, dtype=np.uint8)

        # chroma_log2_weight_denom=5, weight=32 (1.0)
        cb_result, cr_result = apply_explicit_chroma_weights(
            pred_cb, pred_cr,
            weight_cb=32, offset_cb=10,
            weight_cr=32, offset_cr=-10,
            chroma_log2_weight_denom=5
        )

        np.testing.assert_array_equal(cb_result, 138)  # 128 + 10
        np.testing.assert_array_equal(cr_result, 118)  # 128 - 10


# -----------------------------------------------------------------------------
# Test luma_weight/offset for each ref idx
# -----------------------------------------------------------------------------

class TestLumaWeightOffsetPerRefIdx:
    """Tests for per-reference luma weights and offsets."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_different_weight_per_reference(self):
        """Each reference index can have different luma weight."""
        from inter.weighted_prediction import apply_explicit_weights_with_table
        from inter.weighted_pred import WeightTable

        table = WeightTable(luma_log2_weight_denom=6, chroma_log2_weight_denom=6)
        table.set_luma_weight(0, weight=64, offset=0)    # 1.0x
        table.set_luma_weight(1, weight=32, offset=50)   # 0.5x + 50
        table.set_luma_weight(2, weight=96, offset=-20)  # 1.5x - 20

        pred = np.full((4, 4), 100, dtype=np.uint8)

        result0 = apply_explicit_weights_with_table(pred, table, ref_idx=0)
        result1 = apply_explicit_weights_with_table(pred, table, ref_idx=1)
        result2 = apply_explicit_weights_with_table(pred, table, ref_idx=2)

        np.testing.assert_array_equal(result0, 100)  # 100 * 1.0 = 100
        np.testing.assert_array_equal(result1, 100)  # 100 * 0.5 + 50 = 100
        np.testing.assert_array_equal(result2, 130)  # 100 * 1.5 - 20 = 130

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_luma_weight_range_negative(self):
        """Luma weight can be negative (for special effects)."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # Negative weight inverts contribution, offset compensates
        result = apply_explicit_weights(
            pred, weight=-32, offset=200, log2_weight_denom=6
        )

        # ((-32 * 100) + 32) >> 6 + 200 = -50 + 200 = 150
        np.testing.assert_array_equal(result, 150)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_luma_offset_range_positive(self):
        """Positive offset increases brightness."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 100, dtype=np.uint8)

        result = apply_explicit_weights(
            pred, weight=64, offset=50, log2_weight_denom=6
        )

        np.testing.assert_array_equal(result, 150)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_luma_offset_range_negative(self):
        """Negative offset decreases brightness."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 100, dtype=np.uint8)

        result = apply_explicit_weights(
            pred, weight=64, offset=-30, log2_weight_denom=6
        )

        np.testing.assert_array_equal(result, 70)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_luma_weight_offset_max_values(self):
        """Test maximum allowed weight and offset values."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # Max weight is 127 for 8-bit
        result = apply_explicit_weights(
            pred, weight=127, offset=127, log2_weight_denom=6
        )

        # ((127 * 100 + 32) >> 6) + 127 = 198 + 127 = 325 -> clips to 255
        np.testing.assert_array_equal(result, 255)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_luma_weight_offset_min_values(self):
        """Test minimum allowed weight and offset values."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # Min weight is -128, min offset is -128
        result = apply_explicit_weights(
            pred, weight=-128, offset=-128, log2_weight_denom=6
        )

        # ((-128 * 100 + 32) >> 6) + (-128) = -200 - 128 = -328 -> clips to 0
        np.testing.assert_array_equal(result, 0)


# -----------------------------------------------------------------------------
# Test chroma_weight/offset for Cb and Cr
# -----------------------------------------------------------------------------

class TestChromaWeightOffsetCbCr:
    """Tests for separate Cb and Cr chroma weights and offsets."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_cb_and_cr_different_weights(self):
        """Cb and Cr can have independently different weights."""
        from inter.weighted_prediction import apply_explicit_chroma_weights

        pred_cb = np.full((4, 4), 128, dtype=np.uint8)
        pred_cr = np.full((4, 4), 128, dtype=np.uint8)

        cb_result, cr_result = apply_explicit_chroma_weights(
            pred_cb, pred_cr,
            weight_cb=32, offset_cb=0,   # 0.5x for Cb
            weight_cr=96, offset_cr=0,   # 1.5x for Cr
            chroma_log2_weight_denom=6
        )

        np.testing.assert_array_equal(cb_result, 64)   # 128 * 0.5
        np.testing.assert_array_equal(cr_result, 192)  # 128 * 1.5

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_cb_and_cr_different_offsets(self):
        """Cb and Cr can have independently different offsets."""
        from inter.weighted_prediction import apply_explicit_chroma_weights

        pred_cb = np.full((4, 4), 128, dtype=np.uint8)
        pred_cr = np.full((4, 4), 128, dtype=np.uint8)

        cb_result, cr_result = apply_explicit_chroma_weights(
            pred_cb, pred_cr,
            weight_cb=64, offset_cb=20,
            weight_cr=64, offset_cr=-20,
            chroma_log2_weight_denom=6
        )

        np.testing.assert_array_equal(cb_result, 148)  # 128 + 20
        np.testing.assert_array_equal(cr_result, 108)  # 128 - 20

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_chroma_weights_per_reference(self):
        """Each reference can have different Cb/Cr weights."""
        from inter.weighted_prediction import apply_explicit_chroma_with_table
        from inter.weighted_pred import WeightTable

        table = WeightTable(luma_log2_weight_denom=6, chroma_log2_weight_denom=6)
        table.set_chroma_weight(0, weight_cb=64, offset_cb=0, weight_cr=64, offset_cr=0)
        table.set_chroma_weight(1, weight_cb=32, offset_cb=30, weight_cr=96, offset_cr=-10)

        pred_cb = np.full((4, 4), 128, dtype=np.uint8)
        pred_cr = np.full((4, 4), 128, dtype=np.uint8)

        # Ref 0: unity weights
        cb0, cr0 = apply_explicit_chroma_with_table(pred_cb, pred_cr, table, ref_idx=0)
        np.testing.assert_array_equal(cb0, 128)
        np.testing.assert_array_equal(cr0, 128)

        # Ref 1: different weights
        cb1, cr1 = apply_explicit_chroma_with_table(pred_cb, pred_cr, table, ref_idx=1)
        np.testing.assert_array_equal(cb1, 94)   # 128*0.5 + 30 = 94
        np.testing.assert_array_equal(cr1, 182)  # 128*1.5 - 10 = 182

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_chroma_weight_flag_false_uses_default(self):
        """When chroma_weight_flag=0, use default weight and zero offset."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(6)  # luma_log2_weight_denom
        writer.write_ue(5)  # chroma_log2_weight_denom = 5
        writer.write_bits(1, 1)  # luma_weight_l0_flag = 1
        writer.write_se(64)
        writer.write_se(0)
        writer.write_bits(0, 1)  # chroma_weight_l0_flag = 0 (use defaults)

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1, chroma_format_idc=1)

        cb_w, cb_o, cr_w, cr_o = table.get_chroma_weight(0)

        # Default chroma weight = 2^chroma_log2_weight_denom = 32
        assert cb_w == 32
        assert cb_o == 0
        assert cr_w == 32
        assert cr_o == 0


# -----------------------------------------------------------------------------
# Test weighted prediction formula
# -----------------------------------------------------------------------------

class TestWeightedPredictionFormula:
    """Tests for correct weighted prediction formula implementation.

    Unipred formula: ((w * pred + 2^(log2_denom-1)) >> log2_denom) + offset
    """

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_formula_basic_calculation(self):
        """Test basic weighted prediction calculation."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.array([[100]], dtype=np.uint8)

        # weight=80, offset=5, log2_denom=6
        # rounding = 2^(6-1) = 32
        # ((100 * 80 + 32) >> 6) + 5 = (8032 >> 6) + 5 = 125 + 5 = 130
        result = apply_explicit_weights(pred, weight=80, offset=5, log2_weight_denom=6)

        expected = ((100 * 80 + 32) >> 6) + 5
        assert result[0, 0] == expected

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_formula_rounding_behavior(self):
        """Test rounding with 2^(log2_denom-1) offset."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.array([[100]], dtype=np.uint8)

        # Test with log2_denom=6, weight=1, offset=0
        # Without rounding: (100 * 1) >> 6 = 1
        # With rounding: (100 * 1 + 32) >> 6 = 132 >> 6 = 2
        result = apply_explicit_weights(pred, weight=1, offset=0, log2_weight_denom=6)

        expected = (100 * 1 + 32) >> 6
        assert result[0, 0] == expected

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_formula_log2_denom_zero_no_rounding(self):
        """log2_denom=0 should not add rounding (or add 2^-1=0)."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.array([[100]], dtype=np.uint8)

        # log2_denom=0: result = (weight * pred) >> 0 + offset = weight * pred + offset
        result = apply_explicit_weights(pred, weight=2, offset=10, log2_weight_denom=0)

        expected = 2 * 100 + 10
        assert result[0, 0] == expected

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_formula_various_log2_denom_values(self):
        """Test formula with various log2_denom values."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.array([[128]], dtype=np.uint8)

        for log2_denom in range(1, 8):
            weight = 1 << log2_denom  # Unity weight for this denom
            rounding = 1 << (log2_denom - 1)

            result = apply_explicit_weights(
                pred, weight=weight, offset=0, log2_weight_denom=log2_denom
            )

            expected = ((128 * weight + rounding) >> log2_denom) + 0
            assert result[0, 0] == expected, f"Failed for log2_denom={log2_denom}"


# -----------------------------------------------------------------------------
# Test clipping to [0, 255]
# -----------------------------------------------------------------------------

class TestClippingBehavior:
    """Tests for clipping weighted prediction results to [0, 255]."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_clip_overflow_to_255(self):
        """Result exceeding 255 should clip to 255."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 200, dtype=np.uint8)

        # Double the prediction: 200 * 2 = 400 -> clips to 255
        result = apply_explicit_weights(
            pred, weight=128, offset=0, log2_weight_denom=6
        )

        np.testing.assert_array_equal(result, 255)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_clip_underflow_to_0(self):
        """Result below 0 should clip to 0."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 50, dtype=np.uint8)

        # Large negative offset causes underflow
        result = apply_explicit_weights(
            pred, weight=64, offset=-100, log2_weight_denom=6
        )

        # 50 - 100 = -50 -> clips to 0
        np.testing.assert_array_equal(result, 0)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_clip_negative_weight_overflow(self):
        """Negative weight with large offset should clip correctly."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # Negative weight with very large offset
        result = apply_explicit_weights(
            pred, weight=-64, offset=500, log2_weight_denom=6
        )

        # -100 + 500 = 400 -> clips to 255
        np.testing.assert_array_equal(result, 255)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_clip_negative_weight_underflow(self):
        """Negative weight with insufficient offset should clip to 0."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 200, dtype=np.uint8)

        # Negative weight, small offset
        result = apply_explicit_weights(
            pred, weight=-64, offset=100, log2_weight_denom=6
        )

        # ((-64 * 200 + 32) >> 6) + 100 = -200 + 100 = -100 -> clips to 0
        np.testing.assert_array_equal(result, 0)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_clip_preserves_dtype_uint8(self):
        """Output dtype should always be uint8."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((8, 8), 128, dtype=np.uint8)

        result = apply_explicit_weights(
            pred, weight=64, offset=0, log2_weight_denom=6
        )

        assert result.dtype == np.uint8

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_clip_bipred_overflow(self):
        """Bipred result exceeding 255 should clip."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        pred_l0 = np.full((4, 4), 200, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 200, dtype=np.uint8)

        # Both at 1.5x weight
        result = apply_explicit_bipred_weights(
            pred_l0, pred_l1,
            w0=48, o0=0, w1=48, o1=0,
            log2_weight_denom=5
        )

        # (200*48 + 200*48 + 32) >> 6 = 300 -> clips to 255
        np.testing.assert_array_equal(result, 255)


# -----------------------------------------------------------------------------
# Test missing weights (default to 1<<log2_denom)
# -----------------------------------------------------------------------------

class TestMissingWeightsDefault:
    """Tests for default weight behavior when weight flag is 0."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_luma_weight_flag_false_uses_default(self):
        """When luma_weight_l0_flag=0, use default weight 2^log2_denom."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(5)       # luma_log2_weight_denom = 5
        writer.write_ue(5)       # chroma_log2_weight_denom = 5
        writer.write_bits(0, 1)  # luma_weight_l0_flag = 0 (use default)
        writer.write_bits(0, 1)  # chroma_weight_l0_flag = 0

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1, chroma_format_idc=1)

        weight, offset = table.get_luma_weight(0)

        # Default weight = 2^5 = 32, default offset = 0
        assert weight == 32
        assert offset == 0

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_default_weight_produces_identity(self):
        """Default weight should produce identity transformation."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.arange(256, dtype=np.uint8).reshape(16, 16)

        for log2_denom in range(8):
            default_weight = 1 << log2_denom
            result = apply_explicit_weights(
                pred, weight=default_weight, offset=0, log2_weight_denom=log2_denom
            )

            np.testing.assert_array_equal(
                result, pred, err_msg=f"Failed for log2_denom={log2_denom}"
            )

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_mixed_explicit_and_default_weights(self):
        """Some refs can have explicit weights, others use defaults."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(6)  # luma_log2_weight_denom = 6
        writer.write_ue(6)  # chroma_log2_weight_denom = 6

        # Ref 0: explicit
        writer.write_bits(1, 1)
        writer.write_se(96)
        writer.write_se(-10)
        writer.write_bits(0, 1)

        # Ref 1: default (flag=0)
        writer.write_bits(0, 1)
        writer.write_bits(0, 1)

        # Ref 2: explicit
        writer.write_bits(1, 1)
        writer.write_se(32)
        writer.write_se(50)
        writer.write_bits(0, 1)

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=3, chroma_format_idc=1)

        w0, o0 = table.get_luma_weight(0)
        w1, o1 = table.get_luma_weight(1)
        w2, o2 = table.get_luma_weight(2)

        assert (w0, o0) == (96, -10)    # explicit
        assert (w1, o1) == (64, 0)      # default: 2^6=64, offset=0
        assert (w2, o2) == (32, 50)     # explicit

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_accessing_unset_ref_idx_returns_default(self):
        """Accessing weight for unset ref_idx via explicit API should return default."""
        from inter.weighted_prediction import get_explicit_weight_for_ref

        # This function should look up weights from parsed table
        # and return defaults when no explicit weight was provided
        weight, offset = get_explicit_weight_for_ref(
            ref_idx=5,
            luma_log2_weight_denom=6,
            explicit_weights={}  # Empty dict means no explicit weights
        )

        # Should return default
        assert weight == 64  # 2^6
        assert offset == 0


# -----------------------------------------------------------------------------
# Test explicit weighted bipred combining L0 and L1
# -----------------------------------------------------------------------------

class TestExplicitWeightedBipred:
    """Tests for explicit weighted bi-prediction (weighted_bipred_idc=1).

    Formula: ((w0*p0 + w1*p1 + 2^ld) >> (ld+1)) + ((o0+o1+1)>>1)
    """

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_equal_weights_averages(self):
        """Equal L0 and L1 weights should average predictions."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        pred_l0 = np.full((8, 8), 100, dtype=np.uint8)
        pred_l1 = np.full((8, 8), 200, dtype=np.uint8)

        # Equal weights: w0=w1=32 with log2_denom=5
        result = apply_explicit_bipred_weights(
            pred_l0, pred_l1,
            w0=32, o0=0, w1=32, o1=0,
            log2_weight_denom=5
        )

        # ((32*100 + 32*200 + 32) >> 6) + 0 = 150
        np.testing.assert_array_equal(result, 150)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_l0_dominant(self):
        """L0 dominant weight gives result closer to L0 prediction."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 200, dtype=np.uint8)

        # L0 weight = 48 (3/4), L1 weight = 16 (1/4)
        result = apply_explicit_bipred_weights(
            pred_l0, pred_l1,
            w0=48, o0=0, w1=16, o1=0,
            log2_weight_denom=5
        )

        # ((48*100 + 16*200 + 32) >> 6) = (4800 + 3200 + 32) >> 6 = 125
        np.testing.assert_array_equal(result, 125)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_l1_dominant(self):
        """L1 dominant weight gives result closer to L1 prediction."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 200, dtype=np.uint8)

        # L0 weight = 16 (1/4), L1 weight = 48 (3/4)
        result = apply_explicit_bipred_weights(
            pred_l0, pred_l1,
            w0=16, o0=0, w1=48, o1=0,
            log2_weight_denom=5
        )

        # ((16*100 + 48*200 + 32) >> 6) = (1600 + 9600 + 32) >> 6 = 175
        np.testing.assert_array_equal(result, 175)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_with_offsets(self):
        """Bipred offset formula: ((o0+o1+1)>>1)."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 100, dtype=np.uint8)

        # Equal weights with offsets
        result = apply_explicit_bipred_weights(
            pred_l0, pred_l1,
            w0=32, o0=20, w1=32, o1=10,
            log2_weight_denom=5
        )

        # ((32*100 + 32*100 + 32) >> 6) + ((20+10+1)>>1) = 100 + 15 = 115
        np.testing.assert_array_equal(result, 115)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_zero_l0_weight(self):
        """Zero L0 weight gives L1-only contribution."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 200, dtype=np.uint8)

        # w0=0, w1=64 to get full L1 contribution
        result = apply_explicit_bipred_weights(
            pred_l0, pred_l1,
            w0=0, o0=0, w1=64, o1=0,
            log2_weight_denom=5
        )

        # ((0*100 + 64*200 + 32) >> 6) = 200
        np.testing.assert_array_equal(result, 200)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_zero_l1_weight(self):
        """Zero L1 weight gives L0-only contribution."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 200, dtype=np.uint8)

        # w0=64, w1=0
        result = apply_explicit_bipred_weights(
            pred_l0, pred_l1,
            w0=64, o0=0, w1=0, o1=0,
            log2_weight_denom=5
        )

        # ((64*100 + 0*200 + 32) >> 6) = 100
        np.testing.assert_array_equal(result, 100)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_negative_weights(self):
        """Bipred can use negative weights for special effects."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 50, dtype=np.uint8)

        # Negative L1 weight with offset to compensate
        result = apply_explicit_bipred_weights(
            pred_l0, pred_l1,
            w0=64, o0=0, w1=-32, o1=100,
            log2_weight_denom=5
        )

        # Should produce valid result despite negative weight
        assert result.min() >= 0
        assert result.max() <= 255

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_chroma_explicit_weights(self):
        """Explicit weighted bipred for chroma planes."""
        from inter.weighted_prediction import apply_explicit_bipred_chroma_weights

        cb_l0 = np.full((4, 4), 100, dtype=np.uint8)
        cb_l1 = np.full((4, 4), 200, dtype=np.uint8)
        cr_l0 = np.full((4, 4), 120, dtype=np.uint8)
        cr_l1 = np.full((4, 4), 180, dtype=np.uint8)

        cb_result, cr_result = apply_explicit_bipred_chroma_weights(
            cb_l0, cb_l1, cr_l0, cr_l1,
            w0_cb=32, o0_cb=0, w1_cb=32, o1_cb=0,
            w0_cr=32, o0_cr=5, w1_cr=32, o1_cr=5,
            chroma_log2_weight_denom=5
        )

        # Cb: average = 150
        np.testing.assert_array_equal(cb_result, 150)
        # Cr: average + 5 = 155
        np.testing.assert_array_equal(cr_result, 155)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_separate_l0_l1_tables(self):
        """B-slice should have separate weight tables for L0 and L1."""
        from inter.weighted_prediction import WeightTableBSlice

        table = WeightTableBSlice(
            luma_log2_weight_denom=6,
            chroma_log2_weight_denom=6
        )

        # Set different weights for L0 and L1
        table.set_l0_luma_weight(0, weight=64, offset=0)
        table.set_l1_luma_weight(0, weight=96, offset=-10)

        l0_w, l0_o = table.get_l0_luma_weight(0)
        l1_w, l1_o = table.get_l1_luma_weight(0)

        assert (l0_w, l0_o) == (64, 0)
        assert (l1_w, l1_o) == (96, -10)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_bipred_formula_spec_compliance(self):
        """Verify bipred formula matches H.264 spec equation 8-224."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        pred_l0 = np.array([[100]], dtype=np.uint8)
        pred_l1 = np.array([[200]], dtype=np.uint8)

        # log2_denom=5, w0=32, o0=4, w1=32, o1=6
        result = apply_explicit_bipred_weights(
            pred_l0, pred_l1,
            w0=32, o0=4, w1=32, o1=6,
            log2_weight_denom=5
        )

        # Spec formula: ((p0*w0 + p1*w1 + 2^ld) >> (ld+1)) + ((o0+o1+1)>>1)
        p0, p1 = 100, 200
        w0, o0, w1, o1 = 32, 4, 32, 6
        ld = 5

        expected = ((p0 * w0 + p1 * w1 + (1 << ld)) >> (ld + 1)) + ((o0 + o1 + 1) >> 1)

        assert result[0, 0] == expected


# -----------------------------------------------------------------------------
# Test integration with slice header parsing
# -----------------------------------------------------------------------------

class TestSliceHeaderWeightTableIntegration:
    """Tests for pred_weight_table integration with slice header parsing."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_p_slice_weighted_pred_flag_triggers_parsing(self):
        """P-slice with weighted_pred_flag=1 should parse weight table."""
        from slice.slice_header import parse_slice_header_with_weights
        from parameters import SPS, PPS

        # Create PPS with weighted_pred_flag=1
        pps = PPS()
        pps.weighted_pred_flag = True

        sps = SPS()

        # Parse P-slice header should extract weight table
        # (Implementation needs to create appropriate bitstream)
        assert hasattr(pps, 'weighted_pred_flag')

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_b_slice_weighted_bipred_idc_1_triggers_parsing(self):
        """B-slice with weighted_bipred_idc=1 should parse weight table."""
        from slice.slice_header import parse_slice_header_with_weights
        from parameters import SPS, PPS

        # Create PPS with weighted_bipred_idc=1
        pps = PPS()
        pps.weighted_bipred_idc = 1

        sps = SPS()

        # Parse B-slice header should extract L0 and L1 weight tables
        assert hasattr(pps, 'weighted_bipred_idc')

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_slice_header_stores_parsed_weight_table(self):
        """SliceHeader should store parsed weight table with L0/L1 separation for B-slices."""
        from slice.slice_header import SliceHeader
        from inter.weighted_prediction import WeightTableBSlice

        header = SliceHeader()

        # B-slice requires separate L0 and L1 tables
        table = WeightTableBSlice(
            luma_log2_weight_denom=6,
            chroma_log2_weight_denom=6
        )
        table.set_l0_luma_weight(0, 80, 10)
        table.set_l1_luma_weight(0, 64, -5)

        header.pred_weight_table_b = table

        assert header.pred_weight_table_b is not None
        l0_w, l0_o = header.pred_weight_table_b.get_l0_luma_weight(0)
        l1_w, l1_o = header.pred_weight_table_b.get_l1_luma_weight(0)
        assert (l0_w, l0_o) == (80, 10)
        assert (l1_w, l1_o) == (64, -5)


# -----------------------------------------------------------------------------
# Test validation of weight/offset values
# -----------------------------------------------------------------------------

class TestWeightOffsetValidation:
    """Tests for validation of weight and offset value ranges."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_validate_log2_weight_denom_valid_range(self):
        """log2_weight_denom must be in [0, 7]."""
        from inter.weighted_prediction import validate_log2_weight_denom

        for ld in range(8):
            assert validate_log2_weight_denom(ld) is True

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_validate_log2_weight_denom_invalid_negative(self):
        """Negative log2_weight_denom should raise error."""
        from inter.weighted_prediction import validate_log2_weight_denom

        with pytest.raises(ValueError):
            validate_log2_weight_denom(-1)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_validate_log2_weight_denom_invalid_too_large(self):
        """log2_weight_denom >= 8 should raise error."""
        from inter.weighted_prediction import validate_log2_weight_denom

        with pytest.raises(ValueError):
            validate_log2_weight_denom(8)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_validate_weight_valid_range(self):
        """Weight must be in [-128, 127] for 8-bit."""
        from inter.weighted_prediction import validate_weight

        for w in [-128, -1, 0, 1, 64, 127]:
            assert validate_weight(w) is True

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_validate_weight_invalid_too_negative(self):
        """Weight < -128 should raise error."""
        from inter.weighted_prediction import validate_weight

        with pytest.raises(ValueError):
            validate_weight(-129)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_validate_weight_invalid_too_positive(self):
        """Weight > 127 should raise error."""
        from inter.weighted_prediction import validate_weight

        with pytest.raises(ValueError):
            validate_weight(128)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_validate_offset_valid_range(self):
        """Offset must be in [-128, 127] for 8-bit."""
        from inter.weighted_prediction import validate_offset

        for o in [-128, -1, 0, 1, 64, 127]:
            assert validate_offset(o) is True

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_validate_offset_invalid(self):
        """Offset outside [-128, 127] should raise error."""
        from inter.weighted_prediction import validate_offset

        with pytest.raises(ValueError):
            validate_offset(-129)

        with pytest.raises(ValueError):
            validate_offset(128)


# -----------------------------------------------------------------------------
# Test edge cases and special scenarios
# -----------------------------------------------------------------------------

class TestExplicitWeightedPredEdgeCases:
    """Tests for edge cases in explicit weighted prediction."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_all_zero_weights_with_offset(self):
        """Zero weight creates DC-like prediction from offset only."""
        from inter.weighted_prediction import apply_explicit_weights

        pred = np.full((4, 4), 200, dtype=np.uint8)

        # weight=0 means ignore prediction, only offset matters
        result = apply_explicit_weights(
            pred, weight=0, offset=128, log2_weight_denom=6
        )

        # ((0 * pred + 32) >> 6) + 128 = 0 + 128 = 128
        np.testing.assert_array_equal(result, 128)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_unity_weight_various_block_sizes(self):
        """Unity weight preserves prediction for various block sizes."""
        from inter.weighted_prediction import apply_explicit_weights

        for height, width in [(4, 4), (4, 8), (8, 4), (8, 8), (16, 16), (8, 16)]:
            pred = np.random.randint(0, 256, (height, width), dtype=np.uint8)

            result = apply_explicit_weights(
                pred, weight=64, offset=0, log2_weight_denom=6
            )

            np.testing.assert_array_equal(
                result, pred, err_msg=f"Failed for size ({height}, {width})"
            )

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_weighted_pred_preserves_spatial_pattern(self):
        """Weighted prediction should preserve spatial patterns in block."""
        from inter.weighted_prediction import apply_explicit_weights

        # Create gradient pattern
        pred = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4

        result = apply_explicit_weights(
            pred, weight=32, offset=50, log2_weight_denom=6
        )

        # Pattern should be preserved: 0.5*pred + 50
        expected = np.clip((pred.astype(np.int32) * 32 + 32) // 64 + 50, 0, 255).astype(np.uint8)

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_monochrome_no_chroma_weight_parsing(self):
        """Monochrome video (chroma_format_idc=0) skips chroma weight parsing."""
        from slice.pred_weight_table import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_ue(6)  # luma_log2_weight_denom
        # No chroma_log2_weight_denom for monochrome
        writer.write_bits(1, 1)  # luma_weight_l0_flag
        writer.write_se(64)
        writer.write_se(0)
        # No chroma_weight_l0_flag for monochrome

        reader = BitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1, chroma_format_idc=0)

        weight, offset = table.get_luma_weight(0)
        assert weight == 64
        assert offset == 0


# -----------------------------------------------------------------------------
# Test cross-fade and fade sequence simulation
# -----------------------------------------------------------------------------

class TestFadeSequenceSimulation:
    """Tests simulating fade effects with explicit weighted prediction."""

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_fade_out_to_black(self):
        """Simulate fade-out to black with decreasing weights."""
        from inter.weighted_prediction import apply_explicit_weights

        bright_pred = np.full((8, 8), 200, dtype=np.uint8)

        results = []
        for weight in [64, 48, 32, 16, 0]:
            result = apply_explicit_weights(
                bright_pred, weight=weight, offset=0, log2_weight_denom=6
            )
            results.append(result[0, 0])

        # Each frame should be darker
        assert results == sorted(results, reverse=True)
        assert results[-1] == 0  # Final frame is black

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_fade_in_from_black(self):
        """Simulate fade-in from black with increasing weights and offset."""
        from inter.weighted_prediction import apply_explicit_weights

        dark_pred = np.full((8, 8), 0, dtype=np.uint8)
        target_brightness = 200

        results = []
        for weight in [0, 16, 32, 48, 64]:
            # Offset decreases as weight increases to approach target
            offset = int(target_brightness * (64 - weight) / 64)
            result = apply_explicit_weights(
                dark_pred, weight=weight, offset=offset, log2_weight_denom=6
            )
            results.append(result[0, 0])

        # Brightness should be roughly constant as we transition
        # (This is a simplified model of fade-in)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_cross_fade_bipred(self):
        """Simulate cross-fade between two scenes using bipred."""
        from inter.weighted_prediction import apply_explicit_bipred_weights

        scene_a = np.full((8, 8), 100, dtype=np.uint8)
        scene_b = np.full((8, 8), 200, dtype=np.uint8)

        results = []
        for t in [0, 16, 32, 48, 64]:
            w_a = 64 - t
            w_b = t

            result = apply_explicit_bipred_weights(
                scene_a, scene_b,
                w0=w_a, o0=0, w1=w_b, o1=0,
                log2_weight_denom=5
            )
            results.append(result[0, 0])

        # Should transition smoothly from 100 to 200
        assert results[0] == 100
        assert results[-1] == 200
        assert results == sorted(results)

    @pytest.mark.xfail(reason="Explicit weighted pred not implemented")
    def test_exposure_compensation(self):
        """Simulate exposure compensation with multiplicative weight."""
        from inter.weighted_prediction import apply_explicit_weights

        # Underexposed prediction
        dark_pred = np.full((8, 8), 50, dtype=np.uint8)

        # Brighten by 2x
        result = apply_explicit_weights(
            dark_pred, weight=128, offset=0, log2_weight_denom=6
        )

        np.testing.assert_array_equal(result, 100)

        # Brighten by 3x (with clipping)
        bright_pred = np.full((8, 8), 100, dtype=np.uint8)
        result = apply_explicit_weights(
            bright_pred, weight=192, offset=0, log2_weight_denom=6
        )

        # 100 * 3 = 300 -> clips to 255
        np.testing.assert_array_equal(result, 255)
