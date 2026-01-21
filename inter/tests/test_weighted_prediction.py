# h264/inter/tests/test_weighted_prediction.py
"""RED TESTS: Weighted prediction for P-slices.

H.264 supports weighted prediction where the prediction is scaled:
    pred' = ((w * pred + 2^(ld-1)) >> ld) + o

where w=weight, o=offset, ld=log2_weight_denom

This enables better prediction for fades, exposure changes, etc.

These tests SHOULD FAIL until weighted prediction is implemented.
"""

import pytest
import numpy as np


class TestWeightedPredictionParsing:
    """Tests for parsing weighted prediction tables."""

    def test_parse_pred_weight_table(self):
        """Parse pred_weight_table from slice header."""
        from inter.weighted_pred import parse_pred_weight_table
        from bitstream import BitWriter, BitReader

        # Create weight table: luma_log2_weight_denom=5, one ref with w=32, o=0
        writer = BitWriter()
        writer.write_ue(5)  # luma_log2_weight_denom
        writer.write_ue(5)  # chroma_log2_weight_denom
        writer.write_bits(1, 1)  # luma_weight_l0_flag[0]
        writer.write_se(32)  # luma_weight_l0[0]
        writer.write_se(0)   # luma_offset_l0[0]
        writer.write_bits(0, 1)  # chroma_weight_l0_flag[0]

        reader = BitReader(writer.to_bytes())

        table = parse_pred_weight_table(reader, num_ref_idx_l0=1)

        assert table.luma_log2_weight_denom == 5
        assert table.luma_weight[0] == 32
        assert table.luma_offset[0] == 0

    def test_slice_header_has_weight_table(self):
        """Slice header should contain weight table for weighted pred."""
        from slice import SliceHeader

        # SliceHeader should have weighted_pred_table attribute
        header = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,  # P-slice
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
        )

        assert hasattr(header, 'weighted_pred_table'), \
            "SliceHeader should have weighted_pred_table"


class TestExplicitWeightedPrediction:
    """Tests for explicit weighted prediction (weighted_pred_flag=1)."""

    def test_apply_weighted_prediction_luma(self):
        """Apply weight and offset to luma prediction."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((16, 16), 100, dtype=np.uint8)

        # weight=64 (1.0 in log2_denom=6), offset=10
        result = apply_weighted_prediction(
            pred,
            weight=64,
            offset=10,
            log2_denom=6,
        )

        # result = ((64 * 100 + 32) >> 6) + 10 = 100 + 10 = 110
        np.testing.assert_array_equal(result, 110)

    def test_weighted_prediction_scaling(self):
        """Weighted prediction can scale brightness."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((8, 8), 128, dtype=np.uint8)

        # weight=96 (1.5x in log2_denom=6)
        result = apply_weighted_prediction(
            pred,
            weight=96,
            offset=0,
            log2_denom=6,
        )

        # result = ((96 * 128 + 32) >> 6) = 192
        np.testing.assert_array_equal(result, 192)

    def test_weighted_prediction_clipping(self):
        """Weighted prediction clips to [0, 255]."""
        from inter.weighted_pred import apply_weighted_prediction

        pred = np.full((4, 4), 200, dtype=np.uint8)

        # Large weight would overflow
        result = apply_weighted_prediction(
            pred,
            weight=128,  # 2.0x
            offset=50,
            log2_denom=6,
        )

        # Clipped to 255
        np.testing.assert_array_equal(result, 255)

    def test_weighted_prediction_chroma(self):
        """Apply weighted prediction to chroma planes."""
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


class TestImplicitWeightedPrediction:
    """Tests for implicit weighted prediction (weighted_bipred_idc=2)."""

    def test_calc_implicit_weights(self):
        """Calculate implicit weights from POC distances."""
        from inter.weighted_pred import calc_implicit_weights

        # Current POC=4, ref0 POC=0, ref1 POC=8
        w0, w1 = calc_implicit_weights(
            current_poc=4,
            ref0_poc=0,
            ref1_poc=8,
        )

        # w0 and w1 should be proportional to distances
        # tb = 4-0 = 4, td = 8-0 = 8
        # w1 = (tb * 64 + 32) >> 6 = 32
        # w0 = 64 - w1 = 32
        assert w0 == 32
        assert w1 == 32

    def test_implicit_weights_asymmetric(self):
        """Implicit weights for asymmetric POC distances."""
        from inter.weighted_pred import calc_implicit_weights

        # Current POC=2, ref0 POC=0, ref1 POC=4
        w0, w1 = calc_implicit_weights(
            current_poc=2,
            ref0_poc=0,
            ref1_poc=4,
        )

        # tb=2, td=4, w1=(2*64+32)>>6 = 32, w0=64-32=32
        # Equal weights for equal distances
        assert w0 == 32
        assert w1 == 32


class TestWeightedPredictionIntegration:
    """Tests for weighted prediction in decoder."""

    def test_decoder_supports_weighted_pred_flag(self):
        """Decoder should check weighted_pred_flag in PPS."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        assert hasattr(decoder, '_use_weighted_prediction'), \
            "Decoder should have _use_weighted_prediction method"

    def test_reconstruct_with_weight(self):
        """P-MB reconstruction should apply weights when enabled."""
        from inter.p_reconstruct import reconstruct_p_16x16_weighted

        # This function should exist and apply weights
        assert callable(reconstruct_p_16x16_weighted), \
            "reconstruct_p_16x16_weighted should exist"

    def test_weighted_prediction_per_ref(self):
        """Different references can have different weights."""
        from inter.weighted_pred import WeightTable

        table = WeightTable(
            luma_log2_weight_denom=6,
            chroma_log2_weight_denom=6,
        )
        table.set_luma_weight(ref_idx=0, weight=64, offset=0)
        table.set_luma_weight(ref_idx=1, weight=96, offset=-10)

        w0, o0 = table.get_luma_weight(0)
        w1, o1 = table.get_luma_weight(1)

        assert w0 == 64 and o0 == 0
        assert w1 == 96 and o1 == -10
