# h264/inter/tests/test_bipred.py
"""RED TESTS: Bi-directional prediction for B-frames.

Bi-prediction averages predictions from L0 and L1 reference frames:
    result = (pred_l0 + pred_l1 + 1) >> 1

H.264 Spec Reference: Section 8.4.2.2 - Decoding process for bi-prediction

These tests SHOULD FAIL until bi-prediction is implemented.
"""

import pytest
import numpy as np


class TestBipredAverage:
    """Tests for bi-prediction averaging."""

    def test_bipred_average_exists(self):
        """bipred_average function should exist."""
        from inter.bipred import bipred_average

        assert callable(bipred_average)

    def test_bipred_identical_blocks(self):
        """Averaging identical blocks returns the same block."""
        from inter.bipred import bipred_average

        block = np.full((16, 16), 128, dtype=np.uint8)

        result = bipred_average(block, block)

        np.testing.assert_array_equal(result, 128)

    def test_bipred_different_blocks(self):
        """Averaging different blocks returns their average."""
        from inter.bipred import bipred_average

        pred_l0 = np.full((16, 16), 100, dtype=np.uint8)
        pred_l1 = np.full((16, 16), 200, dtype=np.uint8)

        result = bipred_average(pred_l0, pred_l1)

        # (100 + 200 + 1) >> 1 = 150
        np.testing.assert_array_equal(result, 150)

    def test_bipred_rounding(self):
        """Bi-prediction uses round-up: (a+b+1)>>1."""
        from inter.bipred import bipred_average

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 101, dtype=np.uint8)

        result = bipred_average(pred_l0, pred_l1)

        # (100 + 101 + 1) >> 1 = 101 (rounds up)
        np.testing.assert_array_equal(result, 101)

    def test_bipred_output_clipped(self):
        """Bi-prediction output is clipped to [0, 255]."""
        from inter.bipred import bipred_average

        pred_l0 = np.full((4, 4), 250, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 250, dtype=np.uint8)

        result = bipred_average(pred_l0, pred_l1)

        # (250 + 250 + 1) >> 1 = 250
        assert result.max() <= 255
        np.testing.assert_array_equal(result, 250)

    def test_bipred_preserves_shape(self):
        """Bi-prediction preserves block shape."""
        from inter.bipred import bipred_average

        pred_l0 = np.zeros((8, 8), dtype=np.uint8)
        pred_l1 = np.zeros((8, 8), dtype=np.uint8)

        result = bipred_average(pred_l0, pred_l1)

        assert result.shape == (8, 8)


class TestBipredChroma:
    """Tests for bi-prediction on chroma planes."""

    def test_bipred_chroma_exists(self):
        """bipred_chroma function should exist."""
        from inter.bipred import bipred_chroma

        assert callable(bipred_chroma)

    def test_bipred_chroma_returns_tuple(self):
        """bipred_chroma should return (cb, cr) tuple."""
        from inter.bipred import bipred_chroma

        cb_l0 = np.full((8, 8), 128, dtype=np.uint8)
        cb_l1 = np.full((8, 8), 128, dtype=np.uint8)
        cr_l0 = np.full((8, 8), 128, dtype=np.uint8)
        cr_l1 = np.full((8, 8), 128, dtype=np.uint8)

        result = bipred_chroma(cb_l0, cb_l1, cr_l0, cr_l1)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_bipred_chroma_averaging(self):
        """Chroma bi-prediction averages both planes."""
        from inter.bipred import bipred_chroma

        cb_l0 = np.full((8, 8), 100, dtype=np.uint8)
        cb_l1 = np.full((8, 8), 150, dtype=np.uint8)
        cr_l0 = np.full((8, 8), 120, dtype=np.uint8)
        cr_l1 = np.full((8, 8), 140, dtype=np.uint8)

        cb_result, cr_result = bipred_chroma(cb_l0, cb_l1, cr_l0, cr_l1)

        # (100 + 150 + 1) >> 1 = 125
        np.testing.assert_array_equal(cb_result, 125)
        # (120 + 140 + 1) >> 1 = 130
        np.testing.assert_array_equal(cr_result, 130)


class TestWeightedBipred:
    """Tests for weighted bi-prediction."""

    def test_weighted_bipred_exists(self):
        """weighted_bipred function should exist."""
        from inter.bipred import weighted_bipred

        assert callable(weighted_bipred)

    def test_weighted_bipred_equal_weights(self):
        """Equal weights should give same result as simple average."""
        from inter.bipred import weighted_bipred, bipred_average

        pred_l0 = np.full((16, 16), 100, dtype=np.uint8)
        pred_l1 = np.full((16, 16), 200, dtype=np.uint8)

        # Default weights (w0=32, w1=32 with log2_denom=6)
        result_weighted = weighted_bipred(
            pred_l0, pred_l1,
            w0=32, o0=0, w1=32, o1=0,
            log2_denom=5,
        )

        result_simple = bipred_average(pred_l0, pred_l1)

        np.testing.assert_array_almost_equal(result_weighted, result_simple, decimal=0)

    def test_weighted_bipred_l0_only(self):
        """w1=0 with doubled w0 gives L0-only prediction."""
        from inter.bipred import weighted_bipred

        pred_l0 = np.full((8, 8), 100, dtype=np.uint8)
        pred_l1 = np.full((8, 8), 200, dtype=np.uint8)

        # Bi-pred formula divides by 2^(log2_denom+1), so we need w0=128
        # to get full L0 weight when w1=0
        result = weighted_bipred(
            pred_l0, pred_l1,
            w0=128, o0=0, w1=0, o1=0,
            log2_denom=6,
        )

        # ((128*100 + 0 + 64) >> 7) = (12864 >> 7) = 100
        np.testing.assert_array_equal(result, 100)

    def test_weighted_bipred_with_offset(self):
        """Weighted bi-prediction with non-zero offset."""
        from inter.bipred import weighted_bipred

        pred_l0 = np.full((4, 4), 100, dtype=np.uint8)
        pred_l1 = np.full((4, 4), 100, dtype=np.uint8)

        result = weighted_bipred(
            pred_l0, pred_l1,
            w0=32, o0=10, w1=32, o1=10,
            log2_denom=5,
        )

        # ((32*100 + 32*100 + 16) >> 5) + 10 = 100 + 10 = 110
        # With both offsets: more complex formula
        assert result[0, 0] > 100  # Should have positive offset effect
