# h264/inter/tests/test_p_partitions.py
"""Tests for P-macroblock partition types.

Tests reconstruction of P_L0_L0_16x8, P_L0_L0_8x16, and P_8x8 partitions.
"""

import pytest
import numpy as np

from inter.p_reconstruct import (
    reconstruct_p_16x8,
    reconstruct_p_8x16,
    reconstruct_p_8x8,
)
from inter.reference import ReferenceFrame, ReferenceFrameBuffer
from inter.mv_prediction import MVCache


class TestP16x8Reconstruction:
    """Tests for P_L0_L0_16x8 macroblock (two 16x8 partitions)."""

    @pytest.fixture
    def ref_buffer(self):
        """Create reference buffer with pattern."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        luma = np.zeros((32, 32), dtype=np.uint8)
        # Top half = 100, bottom half = 200
        luma[0:16, :] = 100
        luma[16:32, :] = 200
        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)
        buffer.add_frame(ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0))
        return buffer

    @pytest.fixture
    def mv_cache(self):
        return MVCache(width_in_mbs=2, height_in_mbs=2)

    def test_16x8_both_zero_mv(self, ref_buffer, mv_cache):
        """Both partitions with zero MV."""
        luma, cb, cr = reconstruct_p_16x8(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0],
            mvx=[0, 0],
            mvy=[0, 0],
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        assert luma.shape == (16, 16)
        # Top 8 rows from ref position (0,0) = 100
        np.testing.assert_array_equal(luma[0:8, :], 100)
        # Bottom 8 rows from ref position (0,8) = 100 (still in top half of ref)
        np.testing.assert_array_equal(luma[8:16, :], 100)

    def test_16x8_different_mvs(self, ref_buffer, mv_cache):
        """Each partition with different MV."""
        # Top partition: MV (0, 0) -> gets from ref (0,0)
        # Bottom partition: MV (0, 32) -> 8 pixels down in quarter-pel = 8 pixels
        luma, cb, cr = reconstruct_p_16x8(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0],
            mvx=[0, 0],
            mvy=[0, 32],  # 32 quarter-pixels = 8 pixels down
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Top partition from (0,0) = 100
        np.testing.assert_array_equal(luma[0:8, :], 100)
        # Bottom partition from (0, 8+8=16) = 200
        np.testing.assert_array_equal(luma[8:16, :], 200)

    def test_16x8_with_residual(self, ref_buffer, mv_cache):
        """16x8 with residual for each partition."""
        residual_top = np.full((8, 16), 20, dtype=np.int32)
        residual_bottom = np.full((8, 16), -20, dtype=np.int32)

        luma, _, _ = reconstruct_p_16x8(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0],
            mvx=[0, 0],
            mvy=[0, 0],
            residual_luma=[residual_top, residual_bottom],
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        np.testing.assert_array_equal(luma[0:8, :], 120)  # 100 + 20
        np.testing.assert_array_equal(luma[8:16, :], 80)  # 100 - 20

    def test_16x8_updates_mv_cache(self, ref_buffer, mv_cache):
        """MV cache updated for both partitions."""
        reconstruct_p_16x8(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0],
            mvx=[10, 20],
            mvy=[5, 15],
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Top partition blocks (rows 0-1) should have MV (10, 5)
        assert mv_cache.get_mv(0, 0, 0, 0) == (10, 5)
        assert mv_cache.get_mv(0, 0, 0, 1) == (10, 5)
        # Bottom partition blocks (rows 2-3) should have MV (20, 15)
        assert mv_cache.get_mv(0, 0, 0, 2) == (20, 15)
        assert mv_cache.get_mv(0, 0, 0, 3) == (20, 15)


class TestP8x16Reconstruction:
    """Tests for P_L0_L0_8x16 macroblock (two 8x16 partitions)."""

    @pytest.fixture
    def ref_buffer(self):
        """Create reference buffer with left/right pattern."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        luma = np.zeros((32, 32), dtype=np.uint8)
        # Left half = 100, right half = 200
        luma[:, 0:16] = 100
        luma[:, 16:32] = 200
        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)
        buffer.add_frame(ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0))
        return buffer

    @pytest.fixture
    def mv_cache(self):
        return MVCache(width_in_mbs=2, height_in_mbs=2)

    def test_8x16_both_zero_mv(self, ref_buffer, mv_cache):
        """Both partitions with zero MV."""
        luma, cb, cr = reconstruct_p_8x16(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0],
            mvx=[0, 0],
            mvy=[0, 0],
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        assert luma.shape == (16, 16)
        # Left 8 columns from ref = 100
        np.testing.assert_array_equal(luma[:, 0:8], 100)
        # Right 8 columns from ref = 100 (still in left half)
        np.testing.assert_array_equal(luma[:, 8:16], 100)

    def test_8x16_different_mvs(self, ref_buffer, mv_cache):
        """Each partition with different MV."""
        # Left partition: MV (0, 0)
        # Right partition: MV (32, 0) -> 8 pixels right
        luma, cb, cr = reconstruct_p_8x16(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0],
            mvx=[0, 32],  # Right partition shifted 8 pixels right
            mvy=[0, 0],
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Left partition from (0,0) = 100
        np.testing.assert_array_equal(luma[:, 0:8], 100)
        # Right partition from (8+8=16, 0) = 200
        np.testing.assert_array_equal(luma[:, 8:16], 200)

    def test_8x16_with_residual(self, ref_buffer, mv_cache):
        """8x16 with residual for each partition."""
        residual_left = np.full((16, 8), 30, dtype=np.int32)
        residual_right = np.full((16, 8), -30, dtype=np.int32)

        luma, _, _ = reconstruct_p_8x16(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0],
            mvx=[0, 0],
            mvy=[0, 0],
            residual_luma=[residual_left, residual_right],
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        np.testing.assert_array_equal(luma[:, 0:8], 130)   # 100 + 30
        np.testing.assert_array_equal(luma[:, 8:16], 70)   # 100 - 30

    def test_8x16_updates_mv_cache(self, ref_buffer, mv_cache):
        """MV cache updated for both partitions."""
        reconstruct_p_8x16(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0],
            mvx=[10, 20],
            mvy=[5, 15],
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Left partition blocks (cols 0-1) should have MV (10, 5)
        assert mv_cache.get_mv(0, 0, 0, 0) == (10, 5)
        assert mv_cache.get_mv(0, 0, 1, 0) == (10, 5)
        # Right partition blocks (cols 2-3) should have MV (20, 15)
        assert mv_cache.get_mv(0, 0, 2, 0) == (20, 15)
        assert mv_cache.get_mv(0, 0, 3, 0) == (20, 15)


class TestP8x8Reconstruction:
    """Tests for P_8x8 macroblock (four 8x8 sub-macroblocks)."""

    @pytest.fixture
    def ref_buffer(self):
        """Create reference buffer with quadrant pattern."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        luma = np.zeros((32, 32), dtype=np.uint8)
        luma[0:16, 0:16] = 50     # Top-left
        luma[0:16, 16:32] = 100   # Top-right
        luma[16:32, 0:16] = 150   # Bottom-left
        luma[16:32, 16:32] = 200  # Bottom-right
        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)
        buffer.add_frame(ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0))
        return buffer

    @pytest.fixture
    def mv_cache(self):
        return MVCache(width_in_mbs=2, height_in_mbs=2)

    def test_8x8_all_zero_mv(self, ref_buffer, mv_cache):
        """All four partitions with zero MV."""
        luma, cb, cr = reconstruct_p_8x8(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0, 0, 0],
            mvx=[0, 0, 0, 0],
            mvy=[0, 0, 0, 0],
            sub_mb_types=[0, 0, 0, 0],  # All 8x8
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        assert luma.shape == (16, 16)
        # All from ref quadrant (0,0) = 50
        np.testing.assert_array_equal(luma, 50)

    def test_8x8_different_mvs(self, ref_buffer, mv_cache):
        """Each sub-MB with different MV to get different quadrants."""
        # Sub-MB 0 (TL): MV (0, 0) -> ref (0,0) = 50
        # Sub-MB 1 (TR): MV (32, 0) -> ref (16,0) = 100
        # Sub-MB 2 (BL): MV (0, 32) -> ref (0,16) = 150
        # Sub-MB 3 (BR): MV (32, 32) -> ref (16,16) = 200
        luma, cb, cr = reconstruct_p_8x8(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0, 0, 0],
            mvx=[0, 32, 0, 32],
            mvy=[0, 0, 32, 32],
            sub_mb_types=[0, 0, 0, 0],
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        np.testing.assert_array_equal(luma[0:8, 0:8], 50)     # TL
        np.testing.assert_array_equal(luma[0:8, 8:16], 100)   # TR
        np.testing.assert_array_equal(luma[8:16, 0:8], 150)   # BL
        np.testing.assert_array_equal(luma[8:16, 8:16], 200)  # BR

    def test_8x8_with_residuals(self, ref_buffer, mv_cache):
        """P_8x8 with residual for each sub-MB."""
        residuals = [
            np.full((8, 8), 10, dtype=np.int32),
            np.full((8, 8), 20, dtype=np.int32),
            np.full((8, 8), 30, dtype=np.int32),
            np.full((8, 8), 40, dtype=np.int32),
        ]

        luma, _, _ = reconstruct_p_8x8(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0, 0, 0],
            mvx=[0, 0, 0, 0],
            mvy=[0, 0, 0, 0],
            sub_mb_types=[0, 0, 0, 0],
            residual_luma=residuals,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # All from ref (0,0) = 50, plus residuals
        np.testing.assert_array_equal(luma[0:8, 0:8], 60)     # 50 + 10
        np.testing.assert_array_equal(luma[0:8, 8:16], 70)    # 50 + 20
        np.testing.assert_array_equal(luma[8:16, 0:8], 80)    # 50 + 30
        np.testing.assert_array_equal(luma[8:16, 8:16], 90)   # 50 + 40

    def test_8x8_updates_mv_cache(self, ref_buffer, mv_cache):
        """MV cache updated for all four sub-MBs."""
        reconstruct_p_8x8(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=[0, 0, 0, 0],
            mvx=[10, 20, 30, 40],
            mvy=[1, 2, 3, 4],
            sub_mb_types=[0, 0, 0, 0],
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Sub-MB 0 (TL): blocks (0,0), (1,0), (0,1), (1,1)
        assert mv_cache.get_mv(0, 0, 0, 0) == (10, 1)
        assert mv_cache.get_mv(0, 0, 1, 1) == (10, 1)

        # Sub-MB 1 (TR): blocks (2,0), (3,0), (2,1), (3,1)
        assert mv_cache.get_mv(0, 0, 2, 0) == (20, 2)

        # Sub-MB 2 (BL): blocks (0,2), (1,2), (0,3), (1,3)
        assert mv_cache.get_mv(0, 0, 0, 2) == (30, 3)

        # Sub-MB 3 (BR): blocks (2,2), (3,2), (2,3), (3,3)
        assert mv_cache.get_mv(0, 0, 2, 2) == (40, 4)

    def test_8x8_different_ref_idx(self, mv_cache):
        """P_8x8 with different reference frames."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        # Add two reference frames with different values
        luma0 = np.full((32, 32), 50, dtype=np.uint8)
        luma1 = np.full((32, 32), 150, dtype=np.uint8)
        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)

        buffer.add_frame(ReferenceFrame(luma=luma0, cb=cb, cr=cr, frame_num=0))
        buffer.add_frame(ReferenceFrame(luma=luma1, cb=cb.copy(), cr=cr.copy(), frame_num=1))

        # Sub-MBs 0,2 use ref 0, sub-MBs 1,3 use ref 1
        luma, _, _ = reconstruct_p_8x8(
            ref_buffer=buffer,
            mv_cache=mv_cache,
            ref_idx=[1, 0, 1, 0],  # ref_idx 0 = most recent (luma1)
            mvx=[0, 0, 0, 0],
            mvy=[0, 0, 0, 0],
            sub_mb_types=[0, 0, 0, 0],
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # ref_idx 0 = most recent = luma1 = 150
        # ref_idx 1 = older = luma0 = 50
        np.testing.assert_array_equal(luma[0:8, 0:8], 50)     # ref 1
        np.testing.assert_array_equal(luma[0:8, 8:16], 150)   # ref 0
        np.testing.assert_array_equal(luma[8:16, 0:8], 50)    # ref 1
        np.testing.assert_array_equal(luma[8:16, 8:16], 150)  # ref 0
