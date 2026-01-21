# h264/inter/tests/test_sub_mb_partitions.py
"""RED TESTS: Sub-macroblock partitions for P_8x8.

P_8x8 macroblocks can have sub-partitions within each 8x8 block:
- sub_mb_type=0: 8x8 (1 partition) - IMPLEMENTED
- sub_mb_type=1: 8x4 (2 partitions) - NOT IMPLEMENTED
- sub_mb_type=2: 4x8 (2 partitions) - NOT IMPLEMENTED
- sub_mb_type=3: 4x4 (4 partitions) - NOT IMPLEMENTED

These tests SHOULD FAIL until sub-partitions are implemented.
"""

import pytest
import numpy as np

from inter.reference import ReferenceFrame, ReferenceFrameBuffer
from inter.mv_prediction import MVCache


class TestSubMB8x4Reconstruction:
    """Tests for 8x4 sub-macroblock partitions."""

    @pytest.fixture
    def ref_buffer(self):
        """Reference with horizontal gradient."""
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

    def test_8x4_sub_partition_exists(self):
        """reconstruct_p_8x8_sub should handle 8x4 partitions."""
        from inter.p_reconstruct import reconstruct_p_8x8_sub

        # This function should exist and handle sub-partitions
        assert callable(reconstruct_p_8x8_sub), \
            "reconstruct_p_8x8_sub should exist"

    def test_8x4_two_mvs_per_sub_mb(self, ref_buffer, mv_cache):
        """8x4 sub-MB needs 2 MVs (top and bottom 8x4 blocks)."""
        from inter.p_reconstruct import reconstruct_p_8x8_sub

        # Sub-MB 0 with 8x4 partitions: needs 2 MVs
        result = reconstruct_p_8x8_sub(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=0,
            sub_mb_type=1,  # 8x4
            mvs=[(0, 0), (0, 16)],  # Two MVs for top/bottom 8x4
            mb_x=0,
            mb_y=0,
            sub_idx=0,  # First 8x8 block
            residual=None,
        )

        assert result.shape == (8, 8)
        # Top 4 rows from (0,0), bottom 4 rows from (0,4)
        np.testing.assert_array_equal(result[0:4, :], 100)
        np.testing.assert_array_equal(result[4:8, :], 100)

    def test_8x4_different_mvs(self, ref_buffer, mv_cache):
        """8x4 with different MVs for each partition."""
        from inter.p_reconstruct import reconstruct_p_8x8_sub

        # MV (0,0) for top, MV (0,64) for bottom (16 pixels down)
        result = reconstruct_p_8x8_sub(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=0,
            sub_mb_type=1,
            mvs=[(0, 0), (0, 64)],  # 64 qpel = 16 pixels
            mb_x=0,
            mb_y=0,
            sub_idx=0,
            residual=None,
        )

        # Top from ref (0,0)=100, bottom from ref (0,16)=200
        np.testing.assert_array_equal(result[0:4, :], 100)
        np.testing.assert_array_equal(result[4:8, :], 200)


class TestSubMB4x8Reconstruction:
    """Tests for 4x8 sub-macroblock partitions."""

    @pytest.fixture
    def ref_buffer(self):
        """Reference with vertical gradient."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        luma = np.zeros((32, 32), dtype=np.uint8)
        luma[:, 0:16] = 100
        luma[:, 16:32] = 200
        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)
        buffer.add_frame(ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0))
        return buffer

    @pytest.fixture
    def mv_cache(self):
        return MVCache(width_in_mbs=2, height_in_mbs=2)

    def test_4x8_sub_partition(self, ref_buffer, mv_cache):
        """4x8 sub-MB needs 2 MVs (left and right 4x8 blocks)."""
        from inter.p_reconstruct import reconstruct_p_8x8_sub

        result = reconstruct_p_8x8_sub(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=0,
            sub_mb_type=2,  # 4x8
            mvs=[(0, 0), (64, 0)],  # Two MVs for left/right 4x8
            mb_x=0,
            mb_y=0,
            sub_idx=0,
            residual=None,
        )

        assert result.shape == (8, 8)
        # Left 4 cols from (0,0)=100, right 4 cols from (16,0)=200
        np.testing.assert_array_equal(result[:, 0:4], 100)
        np.testing.assert_array_equal(result[:, 4:8], 200)


class TestSubMB4x4Reconstruction:
    """Tests for 4x4 sub-macroblock partitions."""

    @pytest.fixture
    def ref_buffer(self):
        """Reference with quadrant pattern."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        luma = np.zeros((32, 32), dtype=np.uint8)
        luma[0:16, 0:16] = 50
        luma[0:16, 16:32] = 100
        luma[16:32, 0:16] = 150
        luma[16:32, 16:32] = 200
        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)
        buffer.add_frame(ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0))
        return buffer

    @pytest.fixture
    def mv_cache(self):
        return MVCache(width_in_mbs=2, height_in_mbs=2)

    def test_4x4_sub_partition(self, ref_buffer, mv_cache):
        """4x4 sub-MB needs 4 MVs (one per 4x4 block)."""
        from inter.p_reconstruct import reconstruct_p_8x8_sub

        # Four MVs pointing to four different quadrants
        result = reconstruct_p_8x8_sub(
            ref_buffer=ref_buffer,
            mv_cache=mv_cache,
            ref_idx=0,
            sub_mb_type=3,  # 4x4
            mvs=[(0, 0), (64, 0), (0, 64), (64, 64)],
            mb_x=0,
            mb_y=0,
            sub_idx=0,
            residual=None,
        )

        assert result.shape == (8, 8)
        np.testing.assert_array_equal(result[0:4, 0:4], 50)
        np.testing.assert_array_equal(result[0:4, 4:8], 100)
        np.testing.assert_array_equal(result[4:8, 0:4], 150)
        np.testing.assert_array_equal(result[4:8, 4:8], 200)


class TestSubMBMVCacheUpdate:
    """Tests for MV cache updates with sub-partitions."""

    @pytest.fixture
    def mv_cache(self):
        return MVCache(width_in_mbs=2, height_in_mbs=2)

    def test_8x4_mv_cache_granularity(self, mv_cache):
        """8x4 should set MVs at 4x4 block granularity."""
        from inter.p_reconstruct import update_mv_cache_sub_mb

        # 8x4 partition in sub-MB 0 (top-left 8x8)
        # Top 8x4 covers blocks (0,0) and (1,0)
        # Bottom 8x4 covers blocks (0,1) and (1,1)
        update_mv_cache_sub_mb(
            mv_cache,
            mb_x=0, mb_y=0,
            sub_idx=0,
            sub_mb_type=1,  # 8x4
            mvs=[(10, 20), (30, 40)],
        )

        # Top row of sub-MB should have first MV
        assert mv_cache.get_mv(0, 0, 0, 0) == (10, 20)
        assert mv_cache.get_mv(0, 0, 1, 0) == (10, 20)
        # Bottom row of sub-MB should have second MV
        assert mv_cache.get_mv(0, 0, 0, 1) == (30, 40)
        assert mv_cache.get_mv(0, 0, 1, 1) == (30, 40)

    def test_4x4_mv_cache_each_block(self, mv_cache):
        """4x4 should set each 4x4 block independently."""
        from inter.p_reconstruct import update_mv_cache_sub_mb

        update_mv_cache_sub_mb(
            mv_cache,
            mb_x=0, mb_y=0,
            sub_idx=0,
            sub_mb_type=3,  # 4x4
            mvs=[(1, 1), (2, 2), (3, 3), (4, 4)],
        )

        assert mv_cache.get_mv(0, 0, 0, 0) == (1, 1)
        assert mv_cache.get_mv(0, 0, 1, 0) == (2, 2)
        assert mv_cache.get_mv(0, 0, 0, 1) == (3, 3)
        assert mv_cache.get_mv(0, 0, 1, 1) == (4, 4)
