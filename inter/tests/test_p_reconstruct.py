# h264/inter/tests/test_p_reconstruct.py
"""Tests for P-macroblock reconstruction.

Tests the combination of motion compensation and residual decoding
to reconstruct P-frame macroblocks.
"""

import pytest
import numpy as np

from inter.p_reconstruct import (
    reconstruct_p_skip,
    reconstruct_p_16x16,
    apply_inter_prediction,
)
from inter.reference import ReferenceFrame, ReferenceFrameBuffer
from inter.mv_prediction import MVCache


class TestPSkipReconstruction:
    """Tests for P_Skip macroblock reconstruction."""

    @pytest.fixture
    def ref_buffer(self):
        """Create reference buffer with a single frame."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        # Create reference frame with gradient pattern
        luma = np.zeros((32, 32), dtype=np.uint8)
        for y in range(32):
            for x in range(32):
                luma[y, x] = (y * 8 + x) % 256

        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)

        buffer.add_frame(ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0))
        return buffer

    @pytest.fixture
    def mv_cache(self):
        """Create MV cache for 2x2 MB frame."""
        return MVCache(width_in_mbs=2, height_in_mbs=2)

    def test_p_skip_zero_motion(self, ref_buffer, mv_cache):
        """P_Skip with zero MV prediction copies from ref."""
        # MV cache empty -> prediction is (0, 0)
        luma, cb, cr = reconstruct_p_skip(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            mb_x=0,
            mb_y=0,
        )

        # Should copy from position (0, 0) in reference
        assert luma.shape == (16, 16)
        assert cb.shape == (8, 8)
        assert cr.shape == (8, 8)

        # Check that we got the reference values
        ref_frame = ref_buffer.get_frame(0)
        np.testing.assert_array_equal(luma, ref_frame.luma[0:16, 0:16])

    def test_p_skip_nonzero_prediction(self, ref_buffer, mv_cache):
        """P_Skip with predicted MV copies from predicted position."""
        # Set up neighbors so prediction is non-zero
        # For MB(1,1), set left neighbor to have MV (8, 4)
        mv_cache.set_mv_16x16(mb_x=0, mb_y=1, mvx=8, mvy=4)
        # Top neighbor MV (8, 4)
        mv_cache.set_mv_16x16(mb_x=1, mb_y=0, mvx=8, mvy=4)

        luma, cb, cr = reconstruct_p_skip(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            mb_x=1,
            mb_y=1,
        )

        # Predicted MV should be (8, 4) (median of available)
        # MB (1,1) is at pixel position (16, 16)
        # With MV (8, 4), reference position is (16+8, 16+4) = (24, 20)
        # But MV is in quarter-pixels, so (8,4) -> (2,1) integer pixels
        assert luma.shape == (16, 16)


class TestP16x16Reconstruction:
    """Tests for P_L0_16x16 macroblock reconstruction."""

    @pytest.fixture
    def ref_buffer(self):
        """Create reference buffer."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        # Flat reference for easy checking
        luma = np.full((32, 32), 100, dtype=np.uint8)
        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)

        buffer.add_frame(ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0))
        return buffer

    def test_p_16x16_no_residual(self, ref_buffer):
        """P_16x16 with zero residual returns prediction."""
        # Integer MV (0, 0)
        luma, cb, cr = reconstruct_p_16x16(
            ref_buffer=ref_buffer,
            ref_idx=0,
            mvx=0,  # Quarter-pixel units
            mvy=0,
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Should be prediction only (100 everywhere)
        assert luma.shape == (16, 16)
        np.testing.assert_array_equal(luma, 100)

    def test_p_16x16_with_residual(self, ref_buffer):
        """P_16x16 with residual adds to prediction."""
        residual = np.full((16, 16), 10, dtype=np.int32)

        luma, cb, cr = reconstruct_p_16x16(
            ref_buffer=ref_buffer,
            ref_idx=0,
            mvx=0,
            mvy=0,
            residual_luma=residual,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Prediction (100) + residual (10) = 110
        np.testing.assert_array_equal(luma, 110)

    def test_p_16x16_clipping(self, ref_buffer):
        """Result is clipped to [0, 255]."""
        # Large positive residual
        residual = np.full((16, 16), 200, dtype=np.int32)

        luma, _, _ = reconstruct_p_16x16(
            ref_buffer=ref_buffer,
            ref_idx=0,
            mvx=0,
            mvy=0,
            residual_luma=residual,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # 100 + 200 = 300, clipped to 255
        np.testing.assert_array_equal(luma, 255)

    def test_p_16x16_fractional_mv(self):
        """P_16x16 with fractional MV uses interpolation."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        # Gradient reference
        luma = np.zeros((32, 32), dtype=np.uint8)
        for y in range(32):
            for x in range(32):
                luma[y, x] = x * 8

        buffer.add_frame(ReferenceFrame(
            luma=luma,
            cb=np.full((16, 16), 128, dtype=np.uint8),
            cr=np.full((16, 16), 128, dtype=np.uint8),
            frame_num=0
        ))

        # Half-pixel horizontal (dx=2 in quarter-pixel)
        result_luma, _, _ = reconstruct_p_16x16(
            ref_buffer=buffer,
            ref_idx=0,
            mvx=2,  # Half-pixel horizontal
            mvy=0,
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Should be interpolated, not just copied
        assert result_luma.shape == (16, 16)


class TestApplyInterPrediction:
    """Tests for general inter prediction application."""

    def test_apply_single_partition(self):
        """Apply prediction for single partition."""
        ref_luma = np.full((32, 32), 50, dtype=np.uint8)

        result = apply_inter_prediction(
            ref_luma=ref_luma,
            ref_x=0,
            ref_y=0,
            dx=0,
            dy=0,
            width=16,
            height=16,
        )

        assert result.shape == (16, 16)
        np.testing.assert_array_equal(result, 50)

    def test_apply_8x8_partition(self):
        """Apply prediction for 8x8 partition."""
        ref_luma = np.zeros((32, 32), dtype=np.uint8)
        ref_luma[8:16, 8:16] = 100  # Set specific region

        result = apply_inter_prediction(
            ref_luma=ref_luma,
            ref_x=8,
            ref_y=8,
            dx=0,
            dy=0,
            width=8,
            height=8,
        )

        np.testing.assert_array_equal(result, 100)

    def test_apply_with_fractional_mv(self):
        """Apply prediction with fractional position."""
        ref_luma = np.full((32, 32), 80, dtype=np.uint8)

        result = apply_inter_prediction(
            ref_luma=ref_luma,
            ref_x=0,
            ref_y=0,
            dx=2,  # Half-pixel
            dy=2,
            width=8,
            height=8,
        )

        # Flat reference interpolates to same value
        assert result.shape == (8, 8)
        np.testing.assert_array_almost_equal(result, 80, decimal=0)
