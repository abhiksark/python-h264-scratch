# h264/inter/tests/test_p_residual.py
"""Tests for P-macroblock residual decoding.

Tests the combination of inter prediction with residual data
for non-skip P-macroblocks.
"""

import pytest
import numpy as np

from inter.p_reconstruct import (
    reconstruct_p_16x16,
    reconstruct_p_partition,
)
from inter.reference import ReferenceFrame, ReferenceFrameBuffer


class TestP16x16WithResidual:
    """Tests for P_L0_16x16 with residual data."""

    @pytest.fixture
    def ref_buffer(self):
        """Create reference buffer with flat gray frame."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        luma = np.full((32, 32), 100, dtype=np.uint8)
        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)
        buffer.add_frame(ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0))
        return buffer

    def test_positive_residual(self, ref_buffer):
        """Positive residual increases pixel values."""
        residual = np.full((16, 16), 50, dtype=np.int32)

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

        # 100 (pred) + 50 (residual) = 150
        np.testing.assert_array_equal(luma, 150)

    def test_negative_residual(self, ref_buffer):
        """Negative residual decreases pixel values."""
        residual = np.full((16, 16), -30, dtype=np.int32)

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

        # 100 - 30 = 70
        np.testing.assert_array_equal(luma, 70)

    def test_residual_clipping_high(self, ref_buffer):
        """Residual result clipped to 255."""
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

    def test_residual_clipping_low(self, ref_buffer):
        """Residual result clipped to 0."""
        residual = np.full((16, 16), -150, dtype=np.int32)

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

        # 100 - 150 = -50, clipped to 0
        np.testing.assert_array_equal(luma, 0)

    def test_spatially_varying_residual(self, ref_buffer):
        """Residual with spatial variation."""
        residual = np.zeros((16, 16), dtype=np.int32)
        residual[0:8, 0:8] = 20    # Top-left quadrant
        residual[0:8, 8:16] = -20  # Top-right quadrant
        residual[8:16, 0:8] = 40   # Bottom-left
        residual[8:16, 8:16] = -40 # Bottom-right

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

        assert luma[0, 0] == 120   # 100 + 20
        assert luma[0, 8] == 80    # 100 - 20
        assert luma[8, 0] == 140   # 100 + 40
        assert luma[8, 8] == 60    # 100 - 40

    def test_chroma_residual(self, ref_buffer):
        """Chroma components with residual."""
        residual_cb = np.full((8, 8), 10, dtype=np.int32)
        residual_cr = np.full((8, 8), -10, dtype=np.int32)

        _, cb, cr = reconstruct_p_16x16(
            ref_buffer=ref_buffer,
            ref_idx=0,
            mvx=0,
            mvy=0,
            residual_luma=None,
            residual_cb=residual_cb,
            residual_cr=residual_cr,
            mb_x=0,
            mb_y=0,
        )

        np.testing.assert_array_equal(cb, 138)  # 128 + 10
        np.testing.assert_array_equal(cr, 118)  # 128 - 10

    def test_residual_with_fractional_mv(self):
        """Residual added to interpolated prediction."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        # Gradient reference
        luma = np.zeros((32, 32), dtype=np.uint8)
        for x in range(32):
            luma[:, x] = x * 8
        buffer.add_frame(ReferenceFrame(
            luma=luma,
            cb=np.full((16, 16), 128, dtype=np.uint8),
            cr=np.full((16, 16), 128, dtype=np.uint8),
            frame_num=0
        ))

        residual = np.full((16, 16), 5, dtype=np.int32)

        result_luma, _, _ = reconstruct_p_16x16(
            ref_buffer=buffer,
            ref_idx=0,
            mvx=2,  # Half-pixel
            mvy=0,
            residual_luma=residual,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Result should be interpolated prediction + 5
        assert result_luma.shape == (16, 16)
        # Just verify it's not the same as without residual
        no_residual, _, _ = reconstruct_p_16x16(
            ref_buffer=buffer, ref_idx=0, mvx=2, mvy=0,
            residual_luma=None, residual_cb=None, residual_cr=None,
            mb_x=0, mb_y=0,
        )
        assert not np.array_equal(result_luma, no_residual)


class TestPartitionResidual:
    """Tests for partition-level residual application."""

    @pytest.fixture
    def ref_buffer(self):
        """Create reference buffer."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        luma = np.full((32, 32), 80, dtype=np.uint8)
        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)
        buffer.add_frame(ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0))
        return buffer

    def test_8x8_partition_residual(self, ref_buffer):
        """8x8 partition with residual."""
        residual = np.full((8, 8), 20, dtype=np.int32)

        result = reconstruct_p_partition(
            ref_buffer=ref_buffer,
            ref_idx=0,
            mvx=0,
            mvy=0,
            mb_x=0,
            mb_y=0,
            part_x=0,
            part_y=0,
            width=8,
            height=8,
            residual_luma=residual,
        )

        assert result.shape == (8, 8)
        np.testing.assert_array_equal(result, 100)  # 80 + 20

    def test_16x8_partition_residual(self, ref_buffer):
        """16x8 partition with residual."""
        residual = np.full((8, 16), 30, dtype=np.int32)

        result = reconstruct_p_partition(
            ref_buffer=ref_buffer,
            ref_idx=0,
            mvx=0,
            mvy=0,
            mb_x=0,
            mb_y=0,
            part_x=0,
            part_y=0,
            width=16,
            height=8,
            residual_luma=residual,
        )

        assert result.shape == (8, 16)
        np.testing.assert_array_equal(result, 110)  # 80 + 30

    def test_8x16_partition_residual(self, ref_buffer):
        """8x16 partition with residual."""
        residual = np.full((16, 8), -10, dtype=np.int32)

        result = reconstruct_p_partition(
            ref_buffer=ref_buffer,
            ref_idx=0,
            mvx=0,
            mvy=0,
            mb_x=0,
            mb_y=0,
            part_x=0,
            part_y=0,
            width=8,
            height=16,
            residual_luma=residual,
        )

        assert result.shape == (16, 8)
        np.testing.assert_array_equal(result, 70)  # 80 - 10
