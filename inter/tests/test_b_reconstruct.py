# h264/inter/tests/test_b_reconstruct.py
"""RED TESTS: B-macroblock reconstruction.

B-macroblocks combine prediction (L0, L1, or Bi) with residual data.
Unlike P-MBs, B-MBs can use bi-directional prediction.

H.264 Spec Reference: Section 8.4 - Inter prediction process

These tests SHOULD FAIL until B-macroblock reconstruction is implemented.
"""

import pytest
import numpy as np

from inter.reference import ReferenceFrame, ReferenceFrameBuffer


def create_test_ref_buffer():
    """Create a reference buffer with L0 and L1 frames."""
    buffer = ReferenceFrameBuffer(max_frames=4)

    # L0 frame (past, POC=0)
    l0_frame = ReferenceFrame(
        luma=np.full((32, 32), 100, dtype=np.uint8),
        cb=np.full((16, 16), 128, dtype=np.uint8),
        cr=np.full((16, 16), 128, dtype=np.uint8),
        frame_num=0,
        poc=0,
    )
    buffer.add_frame(l0_frame)

    # L1 frame (future, POC=4)
    l1_frame = ReferenceFrame(
        luma=np.full((32, 32), 150, dtype=np.uint8),
        cb=np.full((16, 16), 128, dtype=np.uint8),
        cr=np.full((16, 16), 128, dtype=np.uint8),
        frame_num=1,
        poc=4,
    )
    buffer.add_frame(l1_frame)

    return buffer


class TestBSkipReconstruction:
    """Tests for B_Skip macroblock reconstruction."""

    def test_reconstruct_b_skip_exists(self):
        """reconstruct_b_skip function should exist."""
        from inter.b_reconstruct import reconstruct_b_skip

        assert callable(reconstruct_b_skip)

    def test_b_skip_returns_luma_cb_cr(self):
        """B_Skip should return (luma, cb, cr) tuple."""
        from inter.b_reconstruct import reconstruct_b_skip
        from inter.mv_prediction import MVCache

        buffer = create_test_ref_buffer()
        buffer.build_ref_lists(current_poc=2)

        mv_cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        luma, cb, cr = reconstruct_b_skip(
            ref_buffer=buffer,
            mv_cache=mv_cache,
            mb_x=0,
            mb_y=0,
            use_spatial=True,
        )

        assert luma.shape == (16, 16)
        assert cb.shape == (8, 8)
        assert cr.shape == (8, 8)

    def test_b_skip_no_residual(self):
        """B_Skip has no residual - pure prediction."""
        from inter.b_reconstruct import reconstruct_b_skip
        from inter.mv_prediction import MVCache

        buffer = create_test_ref_buffer()
        buffer.build_ref_lists(current_poc=2)

        mv_cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        luma, cb, cr = reconstruct_b_skip(
            ref_buffer=buffer,
            mv_cache=mv_cache,
            mb_x=0,
            mb_y=0,
            use_spatial=True,
        )

        # With direct mode and zero MVs, should get bi-prediction
        # (100 + 150 + 1) >> 1 = 125
        assert 100 <= luma.mean() <= 150


class TestBL0Reconstruction:
    """Tests for B_L0 (forward only) reconstruction."""

    def test_reconstruct_b_l0_16x16_exists(self):
        """reconstruct_b_l0_16x16 function should exist."""
        from inter.b_reconstruct import reconstruct_b_l0_16x16

        assert callable(reconstruct_b_l0_16x16)

    def test_b_l0_uses_only_l0_prediction(self):
        """B_L0 should use only L0 (forward) prediction."""
        from inter.b_reconstruct import reconstruct_b_l0_16x16

        buffer = create_test_ref_buffer()
        buffer.build_ref_lists(current_poc=2)

        luma, cb, cr = reconstruct_b_l0_16x16(
            ref_buffer=buffer,
            ref_idx=0,
            mvx=0,
            mvy=0,
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Should match L0 reference (100)
        np.testing.assert_array_almost_equal(luma, 100, decimal=0)


class TestBL1Reconstruction:
    """Tests for B_L1 (backward only) reconstruction."""

    def test_reconstruct_b_l1_16x16_exists(self):
        """reconstruct_b_l1_16x16 function should exist."""
        from inter.b_reconstruct import reconstruct_b_l1_16x16

        assert callable(reconstruct_b_l1_16x16)

    def test_b_l1_uses_only_l1_prediction(self):
        """B_L1 should use only L1 (backward) prediction."""
        from inter.b_reconstruct import reconstruct_b_l1_16x16

        buffer = create_test_ref_buffer()
        buffer.build_ref_lists(current_poc=2)

        luma, cb, cr = reconstruct_b_l1_16x16(
            ref_buffer=buffer,
            ref_idx=0,
            mvx=0,
            mvy=0,
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Should match L1 reference (150)
        np.testing.assert_array_almost_equal(luma, 150, decimal=0)


class TestBBiReconstruction:
    """Tests for B_Bi (bi-directional) reconstruction."""

    def test_reconstruct_b_bi_16x16_exists(self):
        """reconstruct_b_bi_16x16 function should exist."""
        from inter.b_reconstruct import reconstruct_b_bi_16x16

        assert callable(reconstruct_b_bi_16x16)

    def test_b_bi_averages_l0_and_l1(self):
        """B_Bi should average L0 and L1 predictions."""
        from inter.b_reconstruct import reconstruct_b_bi_16x16

        buffer = create_test_ref_buffer()
        buffer.build_ref_lists(current_poc=2)

        luma, cb, cr = reconstruct_b_bi_16x16(
            ref_buffer=buffer,
            ref_idx_l0=0,
            mvx_l0=0,
            mvy_l0=0,
            ref_idx_l1=0,
            mvx_l1=0,
            mvy_l1=0,
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # (100 + 150 + 1) >> 1 = 125
        np.testing.assert_array_almost_equal(luma, 125, decimal=0)

    def test_b_bi_with_residual(self):
        """B_Bi with residual data."""
        from inter.b_reconstruct import reconstruct_b_bi_16x16

        buffer = create_test_ref_buffer()
        buffer.build_ref_lists(current_poc=2)

        residual = np.full((16, 16), 10, dtype=np.int32)

        luma, cb, cr = reconstruct_b_bi_16x16(
            ref_buffer=buffer,
            ref_idx_l0=0,
            mvx_l0=0,
            mvy_l0=0,
            ref_idx_l1=0,
            mvx_l1=0,
            mvy_l1=0,
            residual_luma=residual,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # 125 + 10 = 135
        np.testing.assert_array_almost_equal(luma, 135, decimal=0)


class TestBDirectReconstruction:
    """Tests for B_Direct reconstruction."""

    def test_reconstruct_b_direct_16x16_exists(self):
        """reconstruct_b_direct_16x16 function should exist."""
        from inter.b_reconstruct import reconstruct_b_direct_16x16

        assert callable(reconstruct_b_direct_16x16)

    def test_b_direct_derives_mvs(self):
        """B_Direct derives MVs, doesn't take them as input."""
        from inter.b_reconstruct import reconstruct_b_direct_16x16
        from inter.mv_prediction import MVCache

        buffer = create_test_ref_buffer()
        buffer.build_ref_lists(current_poc=2)

        mv_cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        # B_Direct doesn't take explicit MVs
        luma, cb, cr = reconstruct_b_direct_16x16(
            ref_buffer=buffer,
            mv_cache=mv_cache,
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
            use_spatial=True,
        )

        assert luma.shape == (16, 16)


class TestBPartitionReconstruction:
    """Tests for B partition reconstruction (16x8, 8x16, 8x8)."""

    def test_reconstruct_b_16x8_exists(self):
        """reconstruct_b_16x8 function should exist."""
        from inter.b_reconstruct import reconstruct_b_16x8

        assert callable(reconstruct_b_16x8)

    def test_reconstruct_b_8x16_exists(self):
        """reconstruct_b_8x16 function should exist."""
        from inter.b_reconstruct import reconstruct_b_8x16

        assert callable(reconstruct_b_8x16)

    def test_reconstruct_b_8x8_exists(self):
        """reconstruct_b_8x8 function should exist."""
        from inter.b_reconstruct import reconstruct_b_8x8

        assert callable(reconstruct_b_8x8)

    def test_b_16x8_two_partitions(self):
        """B_16x8 has two 16x8 partitions."""
        from inter.b_reconstruct import reconstruct_b_16x8

        buffer = create_test_ref_buffer()
        buffer.build_ref_lists(current_poc=2)

        # Each partition can have different prediction mode
        luma, cb, cr = reconstruct_b_16x8(
            ref_buffer=buffer,
            pred_modes=["L0", "L1"],  # Top uses L0, bottom uses L1
            ref_idx_l0=[0, 0],
            ref_idx_l1=[0, 0],
            mvx_l0=[0, 0],
            mvy_l0=[0, 0],
            mvx_l1=[0, 0],
            mvy_l1=[0, 0],
            residual_luma=None,
            residual_cb=None,
            residual_cr=None,
            mb_x=0,
            mb_y=0,
        )

        # Top half should be ~100 (L0), bottom half ~150 (L1)
        assert luma[:8, :].mean() < luma[8:, :].mean()
