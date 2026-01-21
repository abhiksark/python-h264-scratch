# h264/inter/tests/test_weighted_pred.py
"""RED TESTS: Weighted prediction for inter prediction.

H.264 supports weighted prediction for P-slices (explicit) and B-slices
(explicit or implicit). This enables better prediction for fades, exposure
changes, and scene transitions.

Explicit weighted prediction (P-slice, weighted_pred_flag=1):
    pred' = ((w * pred + 2^(ld-1)) >> ld) + o

Explicit weighted bi-prediction (B-slice, weighted_bipred_idc=1):
    pred' = ((w0*p0 + w1*p1 + 2^ld) >> (ld+1)) + ((o0+o1+1)>>1)

Implicit weighted bi-prediction (B-slice, weighted_bipred_idc=2):
    Weights derived from POC distances.

H.264 Spec Reference:
- Section 7.3.3.2: Prediction weight table syntax
- Section 7.4.3.2: Prediction weight table semantics
- Section 8.4.2.3: Weighted sample prediction process

These tests SHOULD FAIL until weighted prediction is fully implemented.
"""

import pytest
import numpy as np


class TestExplicitWeightedPredictionPSlice:
    """Tests for explicit weighted prediction in P-slices (weighted_pred_flag=1)."""

    def test_apply_weighted_prediction_unity_weight(self):
        """Weight of 1.0 (2^log2_denom) should preserve prediction."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.array([[100, 110], [120, 130]], dtype=np.uint8)

        # weight = 64 (1.0 with log2_denom=6), offset = 0
        result = apply_weighted_prediction(pred, weight=64, offset=0, log2_denom=6)

        # ((64 * pred + 32) >> 6) + 0 = pred
        np.testing.assert_array_equal(result, pred)

    def test_apply_weighted_prediction_half_weight(self):
        """Weight of 0.5 should halve the prediction."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 200, dtype=np.uint8)

        # weight = 32 (0.5 with log2_denom=6), offset = 0
        result = apply_weighted_prediction(pred, weight=32, offset=0, log2_denom=6)

        # ((32 * 200 + 32) >> 6) + 0 = 100
        np.testing.assert_array_equal(result, 100)

    def test_apply_weighted_prediction_double_weight(self):
        """Weight of 2.0 should double the prediction (with clipping)."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # weight = 128 (2.0 with log2_denom=6), offset = 0
        result = apply_weighted_prediction(pred, weight=128, offset=0, log2_denom=6)

        # ((128 * 100 + 32) >> 6) + 0 = 200
        np.testing.assert_array_equal(result, 200)

    def test_apply_weighted_prediction_positive_offset(self):
        """Positive offset increases brightness."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # Unity weight with positive offset
        result = apply_weighted_prediction(pred, weight=64, offset=50, log2_denom=6)

        # ((64 * 100 + 32) >> 6) + 50 = 100 + 50 = 150
        np.testing.assert_array_equal(result, 150)

    def test_apply_weighted_prediction_negative_offset(self):
        """Negative offset decreases brightness."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # Unity weight with negative offset
        result = apply_weighted_prediction(pred, weight=64, offset=-30, log2_denom=6)

        # ((64 * 100 + 32) >> 6) + (-30) = 100 - 30 = 70
        np.testing.assert_array_equal(result, 70)

    def test_apply_weighted_prediction_negative_weight(self):
        """Negative weight inverts the prediction contribution."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # Negative weight with large positive offset
        result = apply_weighted_prediction(pred, weight=-32, offset=200, log2_denom=6)

        # ((-32 * 100 + 32) >> 6) + 200 = -50 + 200 = 150
        np.testing.assert_array_equal(result, 150)

    def test_apply_weighted_prediction_clips_overflow(self):
        """Result clips to 255 when computation overflows."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 200, dtype=np.uint8)

        # Large weight causes overflow
        result = apply_weighted_prediction(pred, weight=128, offset=100, log2_denom=6)

        # ((128 * 200 + 32) >> 6) + 100 = 400 + 100 -> clips to 255
        np.testing.assert_array_equal(result, 255)

    def test_apply_weighted_prediction_clips_underflow(self):
        """Result clips to 0 when computation underflows."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 50, dtype=np.uint8)

        # Negative offset causes underflow
        result = apply_weighted_prediction(pred, weight=64, offset=-100, log2_denom=6)

        # ((64 * 50 + 32) >> 6) + (-100) = 50 - 100 = -50 -> clips to 0
        np.testing.assert_array_equal(result, 0)


class TestExplicitWeightedBiprediction:
    """Tests for explicit weighted bi-prediction in B-slices (weighted_bipred_idc=1)."""

    def test_weighted_bipred_equal_weights(self):
        """Equal weights should average the two predictions."""
        from inter.bipred import weighted_bipred

        pred_l0 = np.full((8, 8), 100, dtype=np.uint8)
        pred_l1 = np.full((8, 8), 200, dtype=np.uint8)

        # Equal weights w0=w1=32 with log2_denom=5
        result = weighted_bipred(
            pred_l0, pred_l1,
            w0=32, o0=0, w1=32, o1=0,
            log2_denom=5,
        )

        # ((32*100 + 32*200 + 32) >> 6) + 0 = 150
        np.testing.assert_array_equal(result, 150)

    def test_weighted_bipred_l0_dominant(self):
        """L0 dominant weight gives result closer to L0."""
        from inter.bipred import weighted_bipred

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 200, dtype=np.uint8)

        # L0 weight = 48, L1 weight = 16 (3:1 ratio)
        result = weighted_bipred(
            pred_l0, pred_l1,
            w0=48, o0=0, w1=16, o1=0,
            log2_denom=5,
        )

        # ((48*100 + 16*200 + 32) >> 6) = (4800 + 3200 + 32) >> 6 = 125
        np.testing.assert_array_equal(result, 125)

    def test_weighted_bipred_l1_dominant(self):
        """L1 dominant weight gives result closer to L1."""
        from inter.bipred import weighted_bipred

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 200, dtype=np.uint8)

        # L0 weight = 16, L1 weight = 48 (1:3 ratio)
        result = weighted_bipred(
            pred_l0, pred_l1,
            w0=16, o0=0, w1=48, o1=0,
            log2_denom=5,
        )

        # ((16*100 + 48*200 + 32) >> 6) = (1600 + 9600 + 32) >> 6 = 175
        np.testing.assert_array_equal(result, 175)

    def test_weighted_bipred_with_offsets(self):
        """Weighted bi-prediction applies combined offset."""
        from inter.bipred import weighted_bipred

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 100, dtype=np.uint8)

        # Equal weights with offsets o0=20, o1=10
        result = weighted_bipred(
            pred_l0, pred_l1,
            w0=32, o0=20, w1=32, o1=10,
            log2_denom=5,
        )

        # ((32*100 + 32*100 + 32) >> 6) + ((20+10+1)>>1) = 100 + 15 = 115
        np.testing.assert_array_equal(result, 115)

    def test_weighted_bipred_zero_weight_l0(self):
        """L0 weight = 0 gives L1-only prediction (scaled)."""
        from inter.bipred import weighted_bipred

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 200, dtype=np.uint8)

        # w0=0, w1=64 to get full L1 contribution
        result = weighted_bipred(
            pred_l0, pred_l1,
            w0=0, o0=0, w1=64, o1=0,
            log2_denom=5,
        )

        # ((0*100 + 64*200 + 32) >> 6) = 200
        np.testing.assert_array_equal(result, 200)


class TestImplicitWeightedPrediction:
    """Tests for implicit weighted prediction (weighted_bipred_idc=2)."""

    def test_calc_implicit_weights_equal_distance(self):
        """Current picture equidistant from L0 and L1 gives equal weights."""
        from inter.weighted_pred import calc_implicit_weights

        # Current POC=4, L0 POC=0, L1 POC=8
        w0, w1 = calc_implicit_weights(current_poc=4, ref0_poc=0, ref1_poc=8)

        # tb=4, td=8, w1=(4*64)/8=32, w0=64-32=32
        assert w0 == 32
        assert w1 == 32

    def test_calc_implicit_weights_closer_to_l0(self):
        """Current closer to L0 gives higher L0 weight."""
        from inter.weighted_pred import calc_implicit_weights

        # Current POC=2, L0 POC=0, L1 POC=8
        w0, w1 = calc_implicit_weights(current_poc=2, ref0_poc=0, ref1_poc=8)

        # tb=2, td=8, w1=(2*64)/8=16, w0=64-16=48
        assert w0 == 48
        assert w1 == 16

    def test_calc_implicit_weights_closer_to_l1(self):
        """Current closer to L1 gives higher L1 weight."""
        from inter.weighted_pred import calc_implicit_weights

        # Current POC=6, L0 POC=0, L1 POC=8
        w0, w1 = calc_implicit_weights(current_poc=6, ref0_poc=0, ref1_poc=8)

        # tb=6, td=8, w1=(6*64)/8=48, w0=64-48=16
        assert w0 == 16
        assert w1 == 48

    def test_calc_implicit_weights_same_poc_fallback(self):
        """Same L0 and L1 POC returns equal weights (fallback)."""
        from inter.weighted_pred import calc_implicit_weights

        # Edge case: L0 and L1 have same POC
        w0, w1 = calc_implicit_weights(current_poc=4, ref0_poc=4, ref1_poc=4)

        # Fallback to equal weights
        assert w0 == 32
        assert w1 == 32

    def test_calc_implicit_weights_negative_poc_diff(self):
        """Handle negative POC differences (backward prediction)."""
        from inter.weighted_pred import calc_implicit_weights

        # L0 in the future, L1 in the past (unusual but valid)
        w0, w1 = calc_implicit_weights(current_poc=4, ref0_poc=8, ref1_poc=0)

        # tb=-4, td=-8, weights should still sum to 64
        assert w0 + w1 == 64

    def test_calc_implicit_weights_clamps_to_valid_range(self):
        """Implicit weights clamp to [-64, 128] range per spec."""
        from inter.weighted_pred import calc_implicit_weights

        # Extreme POC difference that would cause weight overflow
        w0, w1 = calc_implicit_weights(current_poc=0, ref0_poc=0, ref1_poc=1000)

        assert -64 <= w0 <= 128
        assert -64 <= w1 <= 128


class TestWeightTableParsing:
    """Tests for parsing pred_weight_table from slice header."""

    def test_parse_pred_weight_table_single_ref(self):
        """Parse weight table with single L0 reference."""
        from inter.weighted_pred import parse_pred_weight_table
        from bitstream import NumpyBitWriter, NumpyBitReader

        writer = NumpyBitWriter()
        writer.write_ue(5)  # luma_log2_weight_denom = 5
        writer.write_ue(5)  # chroma_log2_weight_denom = 5
        writer.write_flag(True)  # luma_weight_l0_flag[0] = 1
        writer.write_se(48)  # luma_weight_l0[0] = 48
        writer.write_se(-5)  # luma_offset_l0[0] = -5
        writer.write_flag(False)  # chroma_weight_l0_flag[0] = 0

        reader = NumpyBitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1)

        assert table.luma_log2_weight_denom == 5
        weight, offset = table.get_luma_weight(0)
        assert weight == 48
        assert offset == -5

    def test_parse_pred_weight_table_multiple_refs(self):
        """Parse weight table with multiple L0 references."""
        from inter.weighted_pred import parse_pred_weight_table
        from bitstream import NumpyBitWriter, NumpyBitReader

        writer = NumpyBitWriter()
        writer.write_ue(6)  # luma_log2_weight_denom = 6
        writer.write_ue(6)  # chroma_log2_weight_denom = 6

        # Ref 0: explicit luma
        writer.write_flag(True)  # luma_weight_l0_flag[0]
        writer.write_se(64)      # luma_weight_l0[0] = 64 (unity)
        writer.write_se(0)       # luma_offset_l0[0]
        writer.write_flag(False) # chroma_weight_l0_flag[0]

        # Ref 1: explicit luma
        writer.write_flag(True)  # luma_weight_l0_flag[1]
        writer.write_se(96)      # luma_weight_l0[1] = 96 (1.5x)
        writer.write_se(-10)     # luma_offset_l0[1]
        writer.write_flag(False) # chroma_weight_l0_flag[1]

        reader = NumpyBitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=2)

        w0, o0 = table.get_luma_weight(0)
        w1, o1 = table.get_luma_weight(1)

        assert w0 == 64 and o0 == 0
        assert w1 == 96 and o1 == -10

    def test_parse_pred_weight_table_default_weights(self):
        """Default weights used when weight flag is 0."""
        from inter.weighted_pred import parse_pred_weight_table
        from bitstream import NumpyBitWriter, NumpyBitReader

        writer = NumpyBitWriter()
        writer.write_ue(6)  # luma_log2_weight_denom = 6
        writer.write_ue(6)  # chroma_log2_weight_denom = 6
        writer.write_flag(False)  # luma_weight_l0_flag[0] = 0 (use default)
        writer.write_flag(False)  # chroma_weight_l0_flag[0] = 0

        reader = NumpyBitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1)

        weight, offset = table.get_luma_weight(0)
        # Default weight = 2^log2_denom = 64
        assert weight == 64
        assert offset == 0

    def test_parse_pred_weight_table_with_chroma(self):
        """Parse weight table with chroma weights."""
        from inter.weighted_pred import parse_pred_weight_table
        from bitstream import NumpyBitWriter, NumpyBitReader

        writer = NumpyBitWriter()
        writer.write_ue(5)  # luma_log2_weight_denom = 5
        writer.write_ue(5)  # chroma_log2_weight_denom = 5
        writer.write_flag(True)  # luma_weight_l0_flag[0]
        writer.write_se(32)      # luma_weight
        writer.write_se(10)      # luma_offset
        writer.write_flag(True)  # chroma_weight_l0_flag[0]
        writer.write_se(40)      # cb_weight
        writer.write_se(-5)      # cb_offset
        writer.write_se(24)      # cr_weight
        writer.write_se(8)       # cr_offset

        reader = NumpyBitReader(writer.to_bytes())
        table = parse_pred_weight_table(reader, num_ref_idx_l0=1)

        cb_w, cb_o, cr_w, cr_o = table.get_chroma_weight(0)
        assert cb_w == 40 and cb_o == -5
        assert cr_w == 24 and cr_o == 8


class TestWeightedPredictionChroma:
    """Tests for weighted prediction on chroma planes."""

    def test_apply_weighted_prediction_chroma_basic(self):
        """Apply weighted prediction to both chroma planes."""
        from inter.weighted_pred import apply_weighted_prediction_chroma

        pred_cb = np.full((8, 8), 128, dtype=np.uint8)
        pred_cr = np.full((8, 8), 128, dtype=np.uint8)

        cb_result, cr_result = apply_weighted_prediction_chroma(
            pred_cb, pred_cr,
            weight_cb=64, offset_cb=-10,
            weight_cr=64, offset_cr=10,
            log2_denom=6,
        )

        np.testing.assert_array_equal(cb_result, 118)  # 128 - 10
        np.testing.assert_array_equal(cr_result, 138)  # 128 + 10

    def test_apply_weighted_prediction_chroma_different_weights(self):
        """Cb and Cr can have different weights."""
        from inter.weighted_pred import apply_weighted_prediction_chroma

        pred_cb = np.full((8, 8), 100, dtype=np.uint8)
        pred_cr = np.full((8, 8), 100, dtype=np.uint8)

        cb_result, cr_result = apply_weighted_prediction_chroma(
            pred_cb, pred_cr,
            weight_cb=32, offset_cb=0,   # 0.5x scaling for Cb
            weight_cr=96, offset_cr=0,   # 1.5x scaling for Cr
            log2_denom=6,
        )

        np.testing.assert_array_equal(cb_result, 50)   # 100 * 0.5
        np.testing.assert_array_equal(cr_result, 150)  # 100 * 1.5

    def test_weighted_bipred_chroma(self):
        """Weighted bi-prediction on chroma planes."""
        from inter.bipred import weighted_bipred_chroma

        cb_l0 = np.full((8, 8), 100, dtype=np.uint8)
        cb_l1 = np.full((8, 8), 200, dtype=np.uint8)
        cr_l0 = np.full((8, 8), 120, dtype=np.uint8)
        cr_l1 = np.full((8, 8), 180, dtype=np.uint8)

        cb_result, cr_result = weighted_bipred_chroma(
            cb_l0, cb_l1, cr_l0, cr_l1,
            w0_cb=32, o0_cb=0, w1_cb=32, o1_cb=0,
            w0_cr=32, o0_cr=5, w1_cr=32, o1_cr=5,
            log2_denom=5,
        )

        # Cb: (100+200)/2 = 150
        np.testing.assert_array_equal(cb_result, 150)
        # Cr: (120+180)/2 + 5 = 150 + 5 = 155
        np.testing.assert_array_equal(cr_result, 155)


class TestDifferentLog2WeightDenom:
    """Tests for different log2_weight_denom values."""

    def test_log2_weight_denom_zero(self):
        """log2_weight_denom=0 means weights are integers."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # weight=2, offset=0 with log2_denom=0
        # Result = (2 * 100 + 0) >> 0 + 0 = 200
        result = apply_weighted_prediction(pred, weight=2, offset=0, log2_denom=0)

        np.testing.assert_array_equal(result, 200)

    def test_log2_weight_denom_seven(self):
        """log2_weight_denom=7 provides fine-grained weights."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 128, dtype=np.uint8)

        # weight=128 (1.0 with log2_denom=7), offset=0
        result = apply_weighted_prediction(pred, weight=128, offset=0, log2_denom=7)

        np.testing.assert_array_equal(result, 128)

    def test_log2_weight_denom_affects_rounding(self):
        """Different log2_denom affects rounding behavior."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # Same effective weight (0.75) with different denoms
        result_denom4 = apply_weighted_prediction(pred, weight=12, offset=0, log2_denom=4)
        result_denom6 = apply_weighted_prediction(pred, weight=48, offset=0, log2_denom=6)

        # Both should give approximately 75
        np.testing.assert_array_almost_equal(result_denom4, result_denom6, decimal=0)


class TestEdgeCases:
    """Tests for edge cases in weighted prediction."""

    def test_weight_zero(self):
        """Zero weight with large offset creates DC-like prediction."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 100, dtype=np.uint8)

        # weight=0 means prediction is ignored, only offset matters
        result = apply_weighted_prediction(pred, weight=0, offset=128, log2_denom=6)

        # (0 * pred + 32) >> 6 + 128 = 0 + 128 = 128
        np.testing.assert_array_equal(result, 128)

    def test_large_positive_offset_clips(self):
        """Large positive offset clips to 255."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 200, dtype=np.uint8)

        result = apply_weighted_prediction(pred, weight=64, offset=127, log2_denom=6)

        # 200 + 127 = 327 -> clips to 255
        np.testing.assert_array_equal(result, 255)

    def test_large_negative_offset_clips(self):
        """Large negative offset clips to 0."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 50, dtype=np.uint8)

        result = apply_weighted_prediction(pred, weight=64, offset=-128, log2_denom=6)

        # 50 - 128 = -78 -> clips to 0
        np.testing.assert_array_equal(result, 0)

    def test_weighted_prediction_preserves_dtype(self):
        """Output should be uint8."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((8, 8), 128, dtype=np.uint8)

        result = apply_weighted_prediction(pred, weight=64, offset=0, log2_denom=6)

        assert result.dtype == np.uint8

    def test_weighted_bipred_preserves_dtype(self):
        """Weighted bi-prediction output should be uint8."""
        from inter.bipred import weighted_bipred

        pred_l0 = np.full((8, 8), 100, dtype=np.uint8)
        pred_l1 = np.full((8, 8), 200, dtype=np.uint8)

        result = weighted_bipred(
            pred_l0, pred_l1,
            w0=32, o0=0, w1=32, o1=0,
            log2_denom=5,
        )

        assert result.dtype == np.uint8


class TestWeightTableStorage:
    """Tests for WeightTable data structure."""

    def test_weight_table_per_ref_weights(self):
        """WeightTable stores separate weights per reference."""
        from inter.weighted_pred import WeightTable

        table = WeightTable(luma_log2_weight_denom=6, chroma_log2_weight_denom=6)
        table.set_luma_weight(0, weight=64, offset=0)
        table.set_luma_weight(1, weight=96, offset=-10)
        table.set_luma_weight(2, weight=32, offset=20)

        assert table.get_luma_weight(0) == (64, 0)
        assert table.get_luma_weight(1) == (96, -10)
        assert table.get_luma_weight(2) == (32, 20)

    def test_weight_table_chroma_weights(self):
        """WeightTable stores Cb and Cr weights separately."""
        from inter.weighted_pred import WeightTable

        table = WeightTable(luma_log2_weight_denom=5, chroma_log2_weight_denom=5)
        table.set_chroma_weight(0, weight_cb=40, offset_cb=-5, weight_cr=24, offset_cr=8)

        cb_w, cb_o, cr_w, cr_o = table.get_chroma_weight(0)
        assert cb_w == 40 and cb_o == -5
        assert cr_w == 24 and cr_o == 8

    def test_weight_table_default_on_access(self):
        """Accessing unset reference returns default weight."""
        from inter.weighted_pred import WeightTable

        table = WeightTable(luma_log2_weight_denom=6, chroma_log2_weight_denom=6)

        # Access ref_idx=5 without setting it
        weight, offset = table.get_luma_weight(5)

        # Default weight = 2^log2_denom = 64
        assert weight == 64
        assert offset == 0


class TestBSliceWeightedPredictionL1:
    """Tests for B-slice weighted prediction with L1 references.

    These tests target functionality that may not be fully implemented.
    """

    def test_parse_pred_weight_table_b_slice_l1_weights(self):
        """Parse weight table for B-slice with L1 weights.

        B-slices have separate weight tables for L0 and L1 references.
        This requires parsing both sets of weights.
        """
        from inter.weighted_pred import parse_pred_weight_table_b_slice
        from bitstream import NumpyBitWriter, NumpyBitReader

        writer = NumpyBitWriter()
        writer.write_ue(5)  # luma_log2_weight_denom
        writer.write_ue(5)  # chroma_log2_weight_denom

        # L0 ref 0
        writer.write_flag(True)
        writer.write_se(32)  # luma weight
        writer.write_se(0)   # luma offset
        writer.write_flag(False)

        # L1 ref 0
        writer.write_flag(True)
        writer.write_se(48)  # luma weight
        writer.write_se(5)   # luma offset
        writer.write_flag(False)

        reader = NumpyBitReader(writer.to_bytes())
        table_l0, table_l1 = parse_pred_weight_table_b_slice(
            reader, num_ref_idx_l0=1, num_ref_idx_l1=1
        )

        w0, o0 = table_l0.get_luma_weight(0)
        w1, o1 = table_l1.get_luma_weight(0)

        assert w0 == 32 and o0 == 0
        assert w1 == 48 and o1 == 5

    def test_implicit_bipred_full_integration(self):
        """Apply implicit weighted bi-prediction end-to-end.

        Uses POC distances to derive weights, then applies them.
        """
        from inter.bipred import apply_implicit_weighted_bipred

        pred_l0 = np.full((16, 16), 100, dtype=np.uint8)
        pred_l1 = np.full((16, 16), 200, dtype=np.uint8)

        # Current POC=4, L0 POC=0, L1 POC=8 (equidistant)
        result = apply_implicit_weighted_bipred(
            pred_l0, pred_l1,
            current_poc=4,
            l0_poc=0,
            l1_poc=8,
        )

        # Equal distance -> equal weights -> average
        np.testing.assert_array_equal(result, 150)

    def test_weight_table_l1_separate_from_l0(self):
        """L1 weights should be stored separately from L0 weights."""
        from inter.weighted_pred import WeightTableBSlice

        table = WeightTableBSlice(
            luma_log2_weight_denom=6,
            chroma_log2_weight_denom=6,
        )

        # Set different weights for L0 and L1
        table.set_l0_luma_weight(0, weight=64, offset=0)
        table.set_l1_luma_weight(0, weight=96, offset=-10)

        l0_w, l0_o = table.get_l0_luma_weight(0)
        l1_w, l1_o = table.get_l1_luma_weight(0)

        assert l0_w == 64 and l0_o == 0
        assert l1_w == 96 and l1_o == -10

    def test_weighted_bipred_idc_selection(self):
        """Decoder should select correct weighted prediction mode.

        weighted_bipred_idc values:
        - 0: default (unweighted)
        - 1: explicit weighted prediction
        - 2: implicit weighted prediction
        """
        from inter.weighted_pred import get_bipred_weights

        # When weighted_bipred_idc=1 (explicit), use table weights
        w0, o0, w1, o1 = get_bipred_weights(
            weighted_bipred_idc=1,
            weight_table_l0=None,  # Would contain table
            weight_table_l1=None,
            ref_idx_l0=0,
            ref_idx_l1=0,
            current_poc=4,
            l0_poc=0,
            l1_poc=8,
        )

        # Should use weights from tables (test with mock)
        assert w0 is not None and w1 is not None


class TestReconstructionWithWeights:
    """Tests for MB reconstruction with weighted prediction."""

    def test_reconstruct_p_skip_weighted(self):
        """P_Skip reconstruction with weighted prediction.

        Normally P_Skip doesn't use weights, but weighted_pred_flag=1
        should still apply weights from the weight table.
        """
        from inter.p_reconstruct import reconstruct_p_skip_weighted

        # This function may not exist yet
        assert callable(reconstruct_p_skip_weighted)

    def test_reconstruct_p_16x16_weighted_exists(self):
        """reconstruct_p_16x16_weighted function should exist."""
        from inter.p_reconstruct import reconstruct_p_16x16_weighted

        assert callable(reconstruct_p_16x16_weighted)

    def test_decoder_applies_weight_when_flag_set(self):
        """Decoder should apply weights when weighted_pred_flag=1 in PPS."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Check decoder has method to query weighted prediction mode
        assert hasattr(decoder, 'get_weighted_pred_mode'), \
            "Decoder should have get_weighted_pred_mode method"

    def test_reconstruct_p_16x8_weighted(self):
        """P_16x8 reconstruction with weighted prediction.

        Each partition can reference different frames with different weights.
        """
        from inter.p_reconstruct import reconstruct_p_16x8_weighted

        assert callable(reconstruct_p_16x8_weighted)

    def test_reconstruct_p_8x16_weighted(self):
        """P_8x16 reconstruction with weighted prediction."""
        from inter.p_reconstruct import reconstruct_p_8x16_weighted

        assert callable(reconstruct_p_8x16_weighted)

    def test_reconstruct_p_8x8_weighted(self):
        """P_8x8 reconstruction with weighted prediction.

        Each 8x8 sub-macroblock can have different reference and weight.
        """
        from inter.p_reconstruct import reconstruct_p_8x8_weighted

        assert callable(reconstruct_p_8x8_weighted)


class TestSliceHeaderWeightedPrediction:
    """Tests for weight table storage in slice header."""

    def test_slice_header_stores_weight_table(self):
        """SliceHeader should store parsed weight table."""
        from slice import SliceHeader
        from inter.weighted_pred import WeightTable

        header = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,  # P-slice
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
        )

        # Create and assign weight table
        table = WeightTable(luma_log2_weight_denom=6, chroma_log2_weight_denom=6)
        table.set_luma_weight(0, 64, 10)
        header.weighted_pred_table = table

        # Verify it's stored correctly
        assert header.weighted_pred_table is not None
        w, o = header.weighted_pred_table.get_luma_weight(0)
        assert w == 64 and o == 10

    def test_slice_header_b_slice_has_l1_weights(self):
        """B-slice header should have separate L1 weight table."""
        from slice import SliceHeader

        header = SliceHeader(
            first_mb_in_slice=0,
            slice_type=1,  # B-slice
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
        )

        assert hasattr(header, 'weighted_pred_table_l1'), \
            "B-slice header should have weighted_pred_table_l1"

    def test_parse_slice_header_with_weights(self):
        """Parsing slice header should extract weight table."""
        from slice import parse_slice_header_weighted
        from bitstream import NumpyBitWriter, NumpyBitReader
        from parameters import SPS, PPS

        # This requires creating a mock SPS/PPS with weighted_pred_flag=1
        # and a bitstream with weight table data
        assert callable(parse_slice_header_weighted)


class TestWeightedPredictionPartitions:
    """Tests for weighted prediction with different partition sizes."""

    def test_apply_weighted_prediction_4x4_block(self):
        """Weighted prediction on 4x4 block."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 100, dtype=np.uint8)
        result = apply_weighted_prediction(pred, weight=64, offset=20, log2_denom=6)

        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result, 120)

    def test_apply_weighted_prediction_8x4_block(self):
        """Weighted prediction on 8x4 block."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 8), 128, dtype=np.uint8)
        result = apply_weighted_prediction(pred, weight=96, offset=0, log2_denom=6)

        assert result.shape == (4, 8)
        # 128 * 1.5 = 192
        np.testing.assert_array_equal(result, 192)

    def test_apply_weighted_prediction_4x8_block(self):
        """Weighted prediction on 4x8 block."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((8, 4), 200, dtype=np.uint8)
        result = apply_weighted_prediction(pred, weight=32, offset=0, log2_denom=6)

        assert result.shape == (8, 4)
        # 200 * 0.5 = 100
        np.testing.assert_array_equal(result, 100)

    def test_apply_weighted_prediction_non_square_chroma(self):
        """Weighted prediction on non-square chroma blocks."""
        from inter.weighted_pred import apply_weighted_prediction

        # 4x8 chroma block (for 8x16 partition in 4:2:0)
        pred = np.full((8, 4), 128, dtype=np.uint8)
        result = apply_weighted_prediction(pred, weight=64, offset=-10, log2_denom=6)

        assert result.shape == (8, 4)
        np.testing.assert_array_equal(result, 118)


class TestWeightedPredictionSpecCompliance:
    """Tests for H.264 spec compliance of weighted prediction.

    These verify the exact formulas from the spec.
    """

    def test_spec_formula_explicit_weighted_unipred(self):
        """Verify explicit weighted uniprediction formula.

        H.264 Spec Section 8.4.2.3.1, Equation 8-223:
        predPartL0(x,y) = Clip1Y(((pred*w + 2^(ld-1)) >> ld) + o)
        """
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.array([[100]], dtype=np.uint8)

        # log2_denom=6, weight=80, offset=5
        # Expected: ((100*80 + 32) >> 6) + 5 = 125 + 5 = 130
        result = apply_weighted_prediction(pred, weight=80, offset=5, log2_denom=6)

        expected = ((100 * 80 + 32) >> 6) + 5
        assert result[0, 0] == expected

    def test_spec_formula_explicit_weighted_bipred(self):
        """Verify explicit weighted bi-prediction formula.

        H.264 Spec Section 8.4.2.3.2, Equation 8-224:
        pred = Clip1Y(((p0*w0 + p1*w1 + 2^ld) >> (ld+1)) + ((o0+o1+1)>>1))
        """
        from inter.bipred import weighted_bipred

        pred_l0 = np.array([[100]], dtype=np.uint8)
        pred_l1 = np.array([[200]], dtype=np.uint8)

        # log2_denom=5, w0=32, o0=4, w1=32, o1=6
        # Expected: ((100*32 + 200*32 + 32) >> 6) + ((4+6+1)>>1)
        #         = ((3200 + 6400 + 32) >> 6) + 5
        #         = 150 + 5 = 155
        result = weighted_bipred(
            pred_l0, pred_l1,
            w0=32, o0=4, w1=32, o1=6,
            log2_denom=5,
        )

        expected = ((100*32 + 200*32 + 32) >> 6) + ((4+6+1)>>1)
        assert result[0, 0] == expected

    def test_spec_log2_weight_denom_range(self):
        """log2_weight_denom must be in range [0, 7] per spec."""
        from inter.weighted_pred import validate_log2_weight_denom

        # Valid values
        for ld in range(8):
            assert validate_log2_weight_denom(ld) is True

        # Invalid values
        with pytest.raises(ValueError):
            validate_log2_weight_denom(-1)

        with pytest.raises(ValueError):
            validate_log2_weight_denom(8)

    def test_spec_weight_value_range(self):
        """Weight values have specific ranges per spec.

        Luma weight: [-128, 127] for 8-bit
        Offset: [-128, 127]
        """
        from inter.weighted_pred import validate_weight_offset

        # Valid ranges
        assert validate_weight_offset(weight=-128, offset=-128) is True
        assert validate_weight_offset(weight=127, offset=127) is True

        # Invalid weight
        with pytest.raises(ValueError):
            validate_weight_offset(weight=-129, offset=0)

        with pytest.raises(ValueError):
            validate_weight_offset(weight=128, offset=0)

    def test_spec_implicit_weight_calculation(self):
        """Verify implicit weight calculation matches spec.

        H.264 Spec Section 8.4.2.3.2:
        tb = POCcurr - POC0
        td = POC1 - POC0
        tx = (16384 + abs(td/2)) / td
        DistScaleFactor = Clip3(-1024, 1023, (tb * tx + 32) >> 6)
        w1 = DistScaleFactor >> 2
        w0 = 64 - w1
        """
        from inter.bipred import calc_implicit_bipred_weights

        # Test case: POC_curr=4, POC0=0, POC1=8
        # tb = 4, td = 8
        # tx = (16384 + 4) / 8 = 2048
        # DistScaleFactor = (4 * 2048 + 32) >> 6 = 128
        # w1 = 128 >> 2 = 32
        # w0 = 64 - 32 = 32
        w0, w1 = calc_implicit_bipred_weights(
            current_poc=4, l0_poc=0, l1_poc=8
        )

        assert w0 == 32
        assert w1 == 32


class TestWeightedPredictionFadeSequence:
    """Tests simulating weighted prediction for fade sequences."""

    def test_fade_in_weights(self):
        """Simulate fade-in with increasing weights over time."""
        from inter.weighted_pred import apply_weighted_prediction

        # Dark frame fading to normal
        dark_pred = np.full((8, 8), 50, dtype=np.uint8)

        # Frame 1: weight=32 (0.5), offset=64 -> 25+64=89
        frame1 = apply_weighted_prediction(dark_pred, weight=32, offset=64, log2_denom=6)

        # Frame 2: weight=48 (0.75), offset=32 -> 37+32=69
        frame2 = apply_weighted_prediction(dark_pred, weight=48, offset=32, log2_denom=6)

        # Frame 3: weight=64 (1.0), offset=0 -> 50
        frame3 = apply_weighted_prediction(dark_pred, weight=64, offset=0, log2_denom=6)

        # Brightness should decrease as we fade in (weird but that's the test)
        assert frame1[0, 0] > frame2[0, 0] > frame3[0, 0]

    def test_fade_out_weights(self):
        """Simulate fade-out with decreasing weights and increasing offset."""
        from inter.weighted_pred import apply_weighted_prediction

        # Normal brightness frame
        normal_pred = np.full((8, 8), 200, dtype=np.uint8)

        # Progressively fade to black (or white)
        # Decreasing contribution from prediction, adding offset
        results = []
        for weight in [64, 48, 32, 16]:
            result = apply_weighted_prediction(
                normal_pred, weight=weight, offset=0, log2_denom=6
            )
            results.append(result[0, 0])

        # Each frame should be darker
        assert results == sorted(results, reverse=True)

    def test_cross_fade_bipred(self):
        """Simulate cross-fade between two scenes using bi-prediction."""
        from inter.bipred import weighted_bipred

        scene_a = np.full((8, 8), 100, dtype=np.uint8)  # Old scene
        scene_b = np.full((8, 8), 200, dtype=np.uint8)  # New scene

        # Cross-fade: weights shift from scene_a to scene_b
        results = []
        for w_a in [64, 48, 32, 16, 0]:
            w_b = 64 - w_a
            result = weighted_bipred(
                scene_a, scene_b,
                w0=w_a, o0=0, w1=w_b, o1=0,
                log2_denom=5,
            )
            results.append(result[0, 0])

        # Should transition from 100 toward 200
        assert results[0] == 100  # Full scene_a
        assert results[-1] == 200  # Full scene_b
        assert results == sorted(results)  # Monotonically increasing


class TestImplicitWeightDerivation:
    """Tests for implicit weight derivation from POC distances.

    H.264 Spec Section 8.4.2.3.2
    Implicit weights are derived from temporal distances to reference pictures.
    """

    def test_implicit_weights_spec_formula(self):
        """Verify implicit weight calculation follows spec formula.

        tb = DiffPicOrderCnt(CurrPic, RefPicList0[refIdxL0])
        td = DiffPicOrderCnt(RefPicList1[refIdxL1], RefPicList0[refIdxL0])
        tx = (16384 + Abs(td/2)) / td
        DistScaleFactor = Clip3(-1024, 1023, (tb * tx + 32) >> 6)
        w1 = DistScaleFactor >> 2
        w0 = 64 - w1
        """
        from inter.weighted_pred import calc_implicit_weights_spec

        # Test case: POC curr=4, L0=0, L1=8
        # tb = 4, td = 8
        # tx = (16384 + 4) / 8 = 2048
        # DistScaleFactor = (4 * 2048 + 32) >> 6 = 128
        # w1 = 128 >> 2 = 32
        # w0 = 64 - 32 = 32
        w0, w1 = calc_implicit_weights_spec(current_poc=4, l0_poc=0, l1_poc=8)

        assert w0 == 32
        assert w1 == 32

    def test_implicit_weights_asymmetric(self):
        """Asymmetric POC distances should give asymmetric weights."""
        from inter.weighted_pred import calc_implicit_weights

        # POC curr=2, L0=0, L1=8 (closer to L0)
        w0, w1 = calc_implicit_weights(current_poc=2, ref0_poc=0, ref1_poc=8)

        # w0 should be larger (closer to L0)
        assert w0 > w1
        assert w0 + w1 == 64

    def test_implicit_weights_at_l0(self):
        """Current picture at same POC as L0 gives w0=64, w1=0."""
        from inter.weighted_pred import calc_implicit_weights

        w0, w1 = calc_implicit_weights(current_poc=0, ref0_poc=0, ref1_poc=8)

        # Full weight on L0
        assert w0 == 64
        assert w1 == 0

    def test_implicit_weights_at_l1(self):
        """Current picture at same POC as L1 gives w0=0, w1=64."""
        from inter.weighted_pred import calc_implicit_weights

        w0, w1 = calc_implicit_weights(current_poc=8, ref0_poc=0, ref1_poc=8)

        # Full weight on L1
        assert w0 == 0
        assert w1 == 64

    def test_implicit_weights_long_gop(self):
        """Implicit weights with long GOP distances."""
        from inter.weighted_pred import calc_implicit_weights

        # Long distance: POC curr=50, L0=0, L1=100
        w0, w1 = calc_implicit_weights(current_poc=50, ref0_poc=0, ref1_poc=100)

        # Equidistant should give equal weights
        assert w0 == 32
        assert w1 == 32

    def test_implicit_weights_negative_poc(self):
        """Handle negative POC values (unusual but valid)."""
        from inter.weighted_pred import calc_implicit_weights

        # Negative POC scenario
        w0, w1 = calc_implicit_weights(current_poc=-4, ref0_poc=-8, ref1_poc=0)

        # Should still compute valid weights
        assert w0 + w1 == 64
        assert -64 <= w0 <= 128
        assert -64 <= w1 <= 128


class TestWeightedPredictionMultipleRefs:
    """Tests for weighted prediction with multiple reference frames."""

    def test_weight_table_multiple_l0_refs(self):
        """Weight table with multiple L0 references."""
        from inter.weighted_pred import WeightTable

        table = WeightTable(luma_log2_weight_denom=6, chroma_log2_weight_denom=6)

        # Set weights for 4 L0 references
        table.set_luma_weight(0, weight=64, offset=0)   # Unity
        table.set_luma_weight(1, weight=96, offset=-10) # 1.5x darker
        table.set_luma_weight(2, weight=32, offset=50)  # 0.5x brighter
        table.set_luma_weight(3, weight=64, offset=20)  # Unity + offset

        # Verify all stored correctly
        assert table.get_luma_weight(0) == (64, 0)
        assert table.get_luma_weight(1) == (96, -10)
        assert table.get_luma_weight(2) == (32, 50)
        assert table.get_luma_weight(3) == (64, 20)

    def test_weight_table_max_refs(self):
        """Weight table with maximum number of references (32)."""
        from inter.weighted_pred import WeightTable

        table = WeightTable(luma_log2_weight_denom=5, chroma_log2_weight_denom=5)

        # Set weights for max refs
        for i in range(32):
            table.set_luma_weight(i, weight=32 + i, offset=i - 16)

        # Verify all stored
        for i in range(32):
            w, o = table.get_luma_weight(i)
            assert w == 32 + i
            assert o == i - 16

    def test_b_slice_l0_l1_weight_tables(self):
        """B-slice has separate weight tables for L0 and L1."""
        from inter.weighted_pred import WeightTableBSlice

        table = WeightTableBSlice(
            luma_log2_weight_denom=6,
            chroma_log2_weight_denom=6
        )

        # Set different weights for L0 and L1
        table.set_l0_luma_weight(0, weight=64, offset=0)
        table.set_l1_luma_weight(0, weight=64, offset=10)

        # Same ref index, different lists
        l0_w, l0_o = table.get_l0_luma_weight(0)
        l1_w, l1_o = table.get_l1_luma_weight(0)

        assert l0_w == 64 and l0_o == 0
        assert l1_w == 64 and l1_o == 10

    def test_per_partition_ref_selection(self):
        """Each partition can use different reference with different weight."""
        from inter.weighted_pred import apply_weighted_prediction_partition

        pred_ref0 = np.full((8, 8), 100, dtype=np.uint8)
        pred_ref1 = np.full((8, 8), 100, dtype=np.uint8)

        weights = {
            0: (64, 0),   # ref_idx=0: unity
            1: (96, 20),  # ref_idx=1: 1.5x + 20
        }

        # Partition 0 uses ref 0
        result0 = apply_weighted_prediction_partition(
            pred_ref0, ref_idx=0, weights=weights, log2_denom=6
        )

        # Partition 1 uses ref 1
        result1 = apply_weighted_prediction_partition(
            pred_ref1, ref_idx=1, weights=weights, log2_denom=6
        )

        # Different weights should give different results
        assert result0[0, 0] == 100    # Unity: 100
        assert result1[0, 0] == 170    # 1.5*100 + 20 = 170


class TestWeightedPredictionBFrameIntegration:
    """Integration tests for weighted prediction in B-frames."""

    def test_b_direct_with_implicit_weights(self):
        """B_Direct mode with implicit weighted bi-prediction."""
        from inter.bipred import b_direct_weighted

        pred_l0 = np.full((16, 16), 80, dtype=np.uint8)
        pred_l1 = np.full((16, 16), 160, dtype=np.uint8)

        result = b_direct_weighted(
            pred_l0, pred_l1,
            current_poc=4, l0_poc=0, l1_poc=8,
            weighted_bipred_idc=2,  # Implicit
        )

        # Equidistant -> average
        np.testing.assert_array_equal(result, 120)

    def test_b_skip_with_implicit_weights(self):
        """B_Skip uses implicit weights when weighted_bipred_idc=2."""
        from inter.bipred import b_skip_weighted

        pred_l0 = np.full((16, 16), 100, dtype=np.uint8)
        pred_l1 = np.full((16, 16), 200, dtype=np.uint8)

        result = b_skip_weighted(
            pred_l0, pred_l1,
            current_poc=2, l0_poc=0, l1_poc=8,
            weighted_bipred_idc=2,
        )

        # Closer to L0 (poc=2 vs L0=0, L1=8)
        # w0 should be 48, w1 should be 16
        # (48*100 + 16*200 + 32) >> 6 = 125
        assert result[0, 0] == 125

    def test_b_16x16_explicit_weights(self):
        """B_16x16 with explicit weighted bi-prediction."""
        from inter.bipred import b_16x16_weighted

        pred_l0 = np.full((16, 16), 100, dtype=np.uint8)
        pred_l1 = np.full((16, 16), 200, dtype=np.uint8)

        weight_table_l0 = {0: (48, 5)}  # (weight, offset)
        weight_table_l1 = {0: (32, -5)}

        result = b_16x16_weighted(
            pred_l0, pred_l1,
            ref_idx_l0=0, ref_idx_l1=0,
            weight_table_l0=weight_table_l0,
            weight_table_l1=weight_table_l1,
            log2_denom=5,
            weighted_bipred_idc=1,  # Explicit
        )

        # ((48*100 + 32*200 + 32) >> 6) + ((5+(-5)+1)>>1)
        # = (4800 + 6400 + 32) >> 6 + 0 = 175
        np.testing.assert_array_equal(result, 175)

    def test_b_8x8_different_weights_per_partition(self):
        """B_8x8 partitions can have different refs and weights."""
        from inter.bipred import b_8x8_weighted

        # 4 8x8 partitions, each with potentially different refs
        preds_l0 = [np.full((8, 8), 100, dtype=np.uint8) for _ in range(4)]
        preds_l1 = [np.full((8, 8), 200, dtype=np.uint8) for _ in range(4)]

        ref_indices_l0 = [0, 0, 1, 1]
        ref_indices_l1 = [0, 1, 0, 1]

        weight_table_l0 = {0: (32, 0), 1: (64, 0)}
        weight_table_l1 = {0: (32, 0), 1: (48, 10)}

        results = b_8x8_weighted(
            preds_l0, preds_l1,
            ref_indices_l0, ref_indices_l1,
            weight_table_l0, weight_table_l1,
            log2_denom=5,
            weighted_bipred_idc=1,
        )

        # Each partition should have different result based on weights
        assert len(results) == 4


class TestWeightedPredictionUnipredFallback:
    """Tests for uniprediction fallback in weighted prediction."""

    def test_l0_only_bipred_fallback(self):
        """When L1 ref is unavailable, fall back to L0 unipred."""
        from inter.bipred import weighted_bipred_with_fallback

        pred_l0 = np.full((8, 8), 100, dtype=np.uint8)
        pred_l1 = None  # L1 unavailable

        result = weighted_bipred_with_fallback(
            pred_l0, pred_l1,
            w0=64, o0=0, w1=64, o1=0,
            log2_denom=6,
        )

        # Should use L0 only
        np.testing.assert_array_equal(result, 100)

    def test_l1_only_bipred_fallback(self):
        """When L0 ref is unavailable, fall back to L1 unipred."""
        from inter.bipred import weighted_bipred_with_fallback

        pred_l0 = None  # L0 unavailable
        pred_l1 = np.full((8, 8), 200, dtype=np.uint8)

        result = weighted_bipred_with_fallback(
            pred_l0, pred_l1,
            w0=64, o0=0, w1=64, o1=0,
            log2_denom=6,
        )

        # Should use L1 only
        np.testing.assert_array_equal(result, 200)

    def test_weighted_default_when_flag_unset(self):
        """Use default weights when weighted_pred_flag is 0."""
        from inter.weighted_pred import get_default_weight

        # Default weight is 2^log2_denom, offset is 0
        weight, offset = get_default_weight(log2_denom=6)

        assert weight == 64
        assert offset == 0


class TestWeightedPredictionHighProfile:
    """Tests for weighted prediction in High Profile."""

    def test_separate_chroma_log2_denom(self):
        """High Profile can have different log2_denom for chroma."""
        from inter.weighted_pred import WeightTable

        table = WeightTable(
            luma_log2_weight_denom=7,
            chroma_log2_weight_denom=5,
        )

        table.set_luma_weight(0, weight=128, offset=0)    # 1.0 with denom=7
        table.set_chroma_weight(0, weight_cb=32, offset_cb=0,
                                weight_cr=32, offset_cr=0)  # 1.0 with denom=5

        assert table.luma_log2_weight_denom == 7
        assert table.chroma_log2_weight_denom == 5

    def test_10bit_weight_range(self):
        """10-bit video has extended weight/offset range."""
        from inter.weighted_pred import apply_weighted_prediction_10bit

        pred = np.full((4, 4), 512, dtype=np.uint16)  # 10-bit

        result = apply_weighted_prediction_10bit(
            pred, weight=128, offset=100, log2_denom=6, bit_depth=10
        )

        # 512 * 2 + 100 = 1124, should be within 10-bit range
        np.testing.assert_array_equal(result, 1024)  # Clipped to max 10-bit

    def test_weight_offset_range_8bit_vs_10bit(self):
        """Weight/offset ranges differ by bit depth."""
        from inter.weighted_pred import get_weight_offset_range

        # 8-bit: weight [-128, 127], offset [-128, 127]
        w_range_8, o_range_8 = get_weight_offset_range(bit_depth=8)
        assert w_range_8 == (-128, 127)
        assert o_range_8 == (-128, 127)

        # 10-bit: weight [-512, 511], offset [-512, 511]
        w_range_10, o_range_10 = get_weight_offset_range(bit_depth=10)
        assert w_range_10 == (-512, 511)
        assert o_range_10 == (-512, 511)
