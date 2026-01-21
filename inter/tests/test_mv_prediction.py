# h264/inter/tests/test_mv_prediction.py
"""Tests for motion vector prediction.

Tests the spatial median prediction algorithm for P-frames.
H.264 Spec Reference: Section 8.4.1.3 - Derivation of motion vector predictors
"""

import pytest
import numpy as np

from inter.mv_prediction import (
    MVCache,
    predict_mv_16x16,
    predict_mv_16x8,
    predict_mv_8x16,
    predict_mv_8x8,
)


class TestMVCache:
    """Tests for motion vector cache storage."""

    def test_create_cache(self):
        """Create MV cache for frame."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        assert cache.width_in_mbs == 4
        assert cache.height_in_mbs == 4

    def test_store_and_retrieve_mv(self):
        """Store and retrieve MV for a 4x4 block."""
        cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        cache.set_mv(mb_x=0, mb_y=0, block_x=0, block_y=0, mvx=10, mvy=20)
        mvx, mvy = cache.get_mv(mb_x=0, mb_y=0, block_x=0, block_y=0)

        assert mvx == 10
        assert mvy == 20

    def test_default_mv_is_zero(self):
        """Unset MVs default to (0, 0)."""
        cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        mvx, mvy = cache.get_mv(mb_x=0, mb_y=0, block_x=0, block_y=0)

        assert mvx == 0
        assert mvy == 0

    def test_set_16x16_mv(self):
        """Set MV for entire 16x16 macroblock."""
        cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        cache.set_mv_16x16(mb_x=0, mb_y=0, mvx=5, mvy=-3)

        # All 16 4x4 blocks should have the same MV
        for by in range(4):
            for bx in range(4):
                mvx, mvy = cache.get_mv(mb_x=0, mb_y=0, block_x=bx, block_y=by)
                assert mvx == 5
                assert mvy == -3

    def test_out_of_bounds_returns_unavailable(self):
        """Accessing outside frame returns unavailable marker."""
        cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        # Negative positions
        assert cache.is_available(mb_x=-1, mb_y=0, block_x=0, block_y=0) is False

        # Past frame edge
        assert cache.is_available(mb_x=2, mb_y=0, block_x=0, block_y=0) is False


class TestPredict16x16:
    """Tests for 16x16 partition MV prediction (P_L0_16x16)."""

    def test_all_neighbors_available(self):
        """Median prediction when A, B, C all available."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # Set up neighbors for MB at (1, 1)
        # A = left neighbor (MB 0,1), rightmost column
        cache.set_mv_16x16(mb_x=0, mb_y=1, mvx=10, mvy=5)
        # B = top neighbor (MB 1,0), bottom row
        cache.set_mv_16x16(mb_x=1, mb_y=0, mvx=20, mvy=15)
        # C = top-right neighbor (MB 2,0), bottom-left block
        cache.set_mv_16x16(mb_x=2, mb_y=0, mvx=30, mvy=10)

        mvp_x, mvp_y = predict_mv_16x16(cache, mb_x=1, mb_y=1)

        # Median of (10, 20, 30) = 20, median of (5, 15, 10) = 10
        assert mvp_x == 20
        assert mvp_y == 10

    def test_first_mb_in_frame(self):
        """First MB (0,0) has no neighbors - predict (0,0)."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        mvp_x, mvp_y = predict_mv_16x16(cache, mb_x=0, mb_y=0)

        assert mvp_x == 0
        assert mvp_y == 0

    def test_first_column_no_left(self):
        """First column has no left neighbor."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # Set B (top) neighbor
        cache.set_mv_16x16(mb_x=0, mb_y=0, mvx=10, mvy=5)
        # Set C (top-right) neighbor
        cache.set_mv_16x16(mb_x=1, mb_y=0, mvx=20, mvy=15)

        mvp_x, mvp_y = predict_mv_16x16(cache, mb_x=0, mb_y=1)

        # A unavailable, B and C available
        # Median with A=0: median(0, 10, 20) = 10
        assert mvp_x == 10

    def test_first_row_no_top(self):
        """First row has no top neighbor - use left only."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # Set A (left) neighbor
        cache.set_mv_16x16(mb_x=0, mb_y=0, mvx=10, mvy=5)

        mvp_x, mvp_y = predict_mv_16x16(cache, mb_x=1, mb_y=0)

        # B and C unavailable, use A directly
        assert mvp_x == 10
        assert mvp_y == 5

    def test_top_right_unavailable_use_top_left(self):
        """When C unavailable, use D (top-left) instead."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # MB at (3, 1) - rightmost column, C would be (4, 0) which is OOB
        cache.set_mv_16x16(mb_x=2, mb_y=1, mvx=10, mvy=5)   # A (left)
        cache.set_mv_16x16(mb_x=3, mb_y=0, mvx=20, mvy=15)  # B (top)
        cache.set_mv_16x16(mb_x=2, mb_y=0, mvx=30, mvy=10)  # D (top-left)

        mvp_x, mvp_y = predict_mv_16x16(cache, mb_x=3, mb_y=1)

        # C unavailable, use D instead
        # Median of (10, 20, 30) = 20
        assert mvp_x == 20


class TestPredict16x8:
    """Tests for 16x8 partition MV prediction (P_L0_L0_16x8)."""

    def test_top_partition(self):
        """Top 16x8 partition uses standard median."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        cache.set_mv_16x16(mb_x=0, mb_y=1, mvx=10, mvy=5)   # A
        cache.set_mv_16x16(mb_x=1, mb_y=0, mvx=20, mvy=15)  # B
        cache.set_mv_16x16(mb_x=2, mb_y=0, mvx=30, mvy=10)  # C

        mvp_x, mvp_y = predict_mv_16x8(cache, mb_x=1, mb_y=1, partition=0)

        # Same as 16x16
        assert mvp_x == 20
        assert mvp_y == 10

    def test_bottom_partition_uses_top_as_b(self):
        """Bottom 16x8 uses top partition of same MB as B."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # First decode top partition
        cache.set_mv(mb_x=1, mb_y=1, block_x=0, block_y=0, mvx=100, mvy=50)
        cache.set_mv(mb_x=1, mb_y=1, block_x=3, block_y=0, mvx=100, mvy=50)

        # Left neighbor (bottom half)
        cache.set_mv(mb_x=0, mb_y=1, block_x=3, block_y=2, mvx=10, mvy=5)

        mvp_x, mvp_y = predict_mv_16x8(cache, mb_x=1, mb_y=1, partition=1)

        # B comes from top partition of same MB
        # Should prefer B for bottom 16x8
        assert mvp_x == 100
        assert mvp_y == 50


class TestPredict8x16:
    """Tests for 8x16 partition MV prediction (P_L0_L0_8x16)."""

    def test_left_partition(self):
        """Left 8x16 partition uses A for prediction."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        cache.set_mv_16x16(mb_x=0, mb_y=1, mvx=10, mvy=5)   # A (left MB)
        cache.set_mv_16x16(mb_x=1, mb_y=0, mvx=20, mvy=15)  # B

        mvp_x, mvp_y = predict_mv_8x16(cache, mb_x=1, mb_y=1, partition=0)

        # For left 8x16, prefer A
        assert mvp_x == 10
        assert mvp_y == 5

    def test_right_partition_uses_left_as_a(self):
        """Right 8x16 uses left partition of same MB as A."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # First decode left partition
        cache.set_mv(mb_x=1, mb_y=1, block_x=0, block_y=0, mvx=100, mvy=50)
        cache.set_mv(mb_x=1, mb_y=1, block_x=0, block_y=3, mvx=100, mvy=50)

        # C neighbor (top-right)
        cache.set_mv(mb_x=2, mb_y=0, block_x=0, block_y=3, mvx=30, mvy=10)

        mvp_x, mvp_y = predict_mv_8x16(cache, mb_x=1, mb_y=1, partition=1)

        # For right 8x16, prefer C
        assert mvp_x == 30
        assert mvp_y == 10


class TestPredict8x8:
    """Tests for 8x8 sub-macroblock MV prediction."""

    def test_top_left_8x8(self):
        """Top-left 8x8 uses median from external neighbors.

        For top-left 8x8:
        - A from left MB (block 3,0)
        - B from top MB (block 0,3)
        - C from top MB (block 2,3) - NOT top-right MB
        """
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        cache.set_mv_16x16(mb_x=0, mb_y=1, mvx=10, mvy=5)   # A - left MB
        # Set B and C separately in top MB
        cache.set_mv(mb_x=1, mb_y=0, block_x=0, block_y=3, mvx=20, mvy=15)  # B
        cache.set_mv(mb_x=1, mb_y=0, block_x=2, block_y=3, mvx=30, mvy=10)  # C

        mvp_x, mvp_y = predict_mv_8x8(cache, mb_x=1, mb_y=1, sub_mb_idx=0)

        # Median of (10, 20, 30) = 20, median of (5, 15, 10) = 10
        assert mvp_x == 20
        assert mvp_y == 10

    def test_top_right_8x8(self):
        """Top-right 8x8 (idx=1) uses top-left as A neighbor."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # Set top-left partition MV
        cache.set_mv(mb_x=1, mb_y=1, block_x=0, block_y=0, mvx=100, mvy=50)
        cache.set_mv(mb_x=1, mb_y=1, block_x=1, block_y=0, mvx=100, mvy=50)
        cache.set_mv(mb_x=1, mb_y=1, block_x=0, block_y=1, mvx=100, mvy=50)
        cache.set_mv(mb_x=1, mb_y=1, block_x=1, block_y=1, mvx=100, mvy=50)

        # B from top MB
        cache.set_mv(mb_x=1, mb_y=0, block_x=2, block_y=3, mvx=20, mvy=15)

        # C from top-right MB
        cache.set_mv(mb_x=2, mb_y=0, block_x=0, block_y=3, mvx=30, mvy=10)

        mvp_x, mvp_y = predict_mv_8x8(cache, mb_x=1, mb_y=1, sub_mb_idx=1)

        # Median of (100, 20, 30)
        assert mvp_x == 30

    def test_all_four_8x8_partitions(self):
        """All four 8x8 partitions give valid predictions."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # Set up surrounding MVs
        cache.set_mv_16x16(mb_x=0, mb_y=1, mvx=10, mvy=5)
        cache.set_mv_16x16(mb_x=1, mb_y=0, mvx=20, mvy=15)
        cache.set_mv_16x16(mb_x=2, mb_y=0, mvx=30, mvy=10)
        cache.set_mv_16x16(mb_x=2, mb_y=1, mvx=40, mvy=20)
        cache.set_mv_16x16(mb_x=0, mb_y=0, mvx=5, mvy=2)

        for sub_mb_idx in range(4):
            mvp_x, mvp_y = predict_mv_8x8(cache, mb_x=1, mb_y=1, sub_mb_idx=sub_mb_idx)
            # Just verify it returns something reasonable
            assert isinstance(mvp_x, (int, np.integer))
            assert isinstance(mvp_y, (int, np.integer))


class TestMedianCalculation:
    """Tests for median calculation edge cases."""

    def test_median_all_same(self):
        """Median of identical values."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        cache.set_mv_16x16(mb_x=0, mb_y=1, mvx=10, mvy=10)
        cache.set_mv_16x16(mb_x=1, mb_y=0, mvx=10, mvy=10)
        cache.set_mv_16x16(mb_x=2, mb_y=0, mvx=10, mvy=10)

        mvp_x, mvp_y = predict_mv_16x16(cache, mb_x=1, mb_y=1)

        assert mvp_x == 10
        assert mvp_y == 10

    def test_median_negative_mvs(self):
        """Median works with negative motion vectors."""
        cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        cache.set_mv_16x16(mb_x=0, mb_y=1, mvx=-20, mvy=-10)
        cache.set_mv_16x16(mb_x=1, mb_y=0, mvx=10, mvy=5)
        cache.set_mv_16x16(mb_x=2, mb_y=0, mvx=-5, mvy=-30)

        mvp_x, mvp_y = predict_mv_16x16(cache, mb_x=1, mb_y=1)

        # Median of (-20, 10, -5) = -5
        # Median of (-10, 5, -30) = -10
        assert mvp_x == -5
        assert mvp_y == -10
