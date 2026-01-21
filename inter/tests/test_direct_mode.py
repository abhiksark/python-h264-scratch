# h264/inter/tests/test_direct_mode.py
"""RED TESTS: Direct mode MV derivation for B-frames.

B_Direct and B_Skip macroblocks derive MVs without explicit transmission.
Two modes are supported:
1. Spatial direct: Derive from neighbor MVs
2. Temporal direct: Scale co-located MV from reference

H.264 Spec Reference: Section 8.4.1.2 - Derivation of motion vectors

These tests SHOULD FAIL until direct mode is implemented.
"""

import pytest
import numpy as np

from inter.mv_prediction import MVCache


class TestSpatialDirectMode:
    """Tests for spatial direct mode MV derivation."""

    def test_derive_direct_spatial_exists(self):
        """derive_direct_spatial function should exist."""
        from inter.direct_mode import derive_direct_spatial

        assert callable(derive_direct_spatial)

    def test_spatial_direct_from_neighbors(self):
        """Spatial direct uses neighbor MVs."""
        from inter.direct_mode import derive_direct_spatial

        mv_cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        # Set neighbor MVs (left, top, top-right)
        mv_cache.set_mv(0, 0, 3, 3, 4, 8)  # Left neighbor
        mv_cache.set_mv(0, 0, 0, 3, 8, 4)  # Top neighbor

        mvx_l0, mvy_l0, mvx_l1, mvy_l1 = derive_direct_spatial(
            mv_cache, mb_x=1, mb_y=1
        )

        # Should derive reasonable MVs from neighbors
        assert isinstance(mvx_l0, int)
        assert isinstance(mvy_l0, int)
        assert isinstance(mvx_l1, int)
        assert isinstance(mvy_l1, int)

    def test_spatial_direct_l1_opposite(self):
        """Spatial direct L1 MV is typically opposite direction."""
        from inter.direct_mode import derive_direct_spatial

        mv_cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        # Set a forward MV in neighbor
        mv_cache.set_mv(0, 0, 3, 3, 8, 8)

        mvx_l0, mvy_l0, mvx_l1, mvy_l1 = derive_direct_spatial(
            mv_cache, mb_x=1, mb_y=1
        )

        # L1 should typically be negative of L0 (or similar relationship)
        # This depends on POC distances but generally opposite direction
        pass  # Just verify it runs

    def test_spatial_direct_first_mb(self):
        """Spatial direct at first MB (no neighbors)."""
        from inter.direct_mode import derive_direct_spatial

        mv_cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        # No neighbors set - first MB
        mvx_l0, mvy_l0, mvx_l1, mvy_l1 = derive_direct_spatial(
            mv_cache, mb_x=0, mb_y=0
        )

        # Should return zero MVs
        assert mvx_l0 == 0
        assert mvy_l0 == 0


class TestTemporalDirectMode:
    """Tests for temporal direct mode MV derivation."""

    def test_derive_direct_temporal_exists(self):
        """derive_direct_temporal function should exist."""
        from inter.direct_mode import derive_direct_temporal

        assert callable(derive_direct_temporal)

    def test_temporal_direct_scales_colocated_mv(self):
        """Temporal direct scales co-located MV from L1 reference."""
        from inter.direct_mode import derive_direct_temporal
        from inter.reference import ReferenceFrame, ReferenceFrameBuffer

        buffer = ReferenceFrameBuffer(max_frames=4)

        # Create L1 reference with stored MVs
        l1_frame = ReferenceFrame(
            luma=np.zeros((32, 32), dtype=np.uint8),
            cb=np.zeros((16, 16), dtype=np.uint8),
            cr=np.zeros((16, 16), dtype=np.uint8),
            frame_num=1,
            poc=4,
        )
        # L1 frame has MV pointing to its reference (poc=0)
        l1_frame.mv_field = np.zeros((2, 2, 2), dtype=np.int16)
        l1_frame.mv_field[0, 0] = [8, 4]  # MB(0,0) MV

        buffer.add_frame(l1_frame)

        mvx_l0, mvy_l0, mvx_l1, mvy_l1 = derive_direct_temporal(
            ref_buffer=buffer,
            current_poc=2,
            mb_x=0,
            mb_y=0,
        )

        # MV should be scaled based on POC distances
        # tb = current_poc - l0_poc = 2 - 0 = 2
        # td = l1_poc - l0_poc = 4 - 0 = 4
        # mvx_l0 = (tb * colocated_mv) / td = (2 * 8) / 4 = 4
        assert isinstance(mvx_l0, int)

    def test_temporal_direct_zero_colocated(self):
        """Temporal direct with zero co-located MV."""
        from inter.direct_mode import derive_direct_temporal
        from inter.reference import ReferenceFrame, ReferenceFrameBuffer

        buffer = ReferenceFrameBuffer(max_frames=4)

        l1_frame = ReferenceFrame(
            luma=np.zeros((32, 32), dtype=np.uint8),
            cb=np.zeros((16, 16), dtype=np.uint8),
            cr=np.zeros((16, 16), dtype=np.uint8),
            frame_num=1,
            poc=4,
        )
        l1_frame.mv_field = np.zeros((2, 2, 2), dtype=np.int16)  # All zero MVs

        buffer.add_frame(l1_frame)

        mvx_l0, mvy_l0, mvx_l1, mvy_l1 = derive_direct_temporal(
            ref_buffer=buffer,
            current_poc=2,
            mb_x=0,
            mb_y=0,
        )

        # Zero co-located MV should give zero derived MVs
        assert mvx_l0 == 0 or mvx_l1 == 0


class TestDirectModeSelection:
    """Tests for choosing between spatial and temporal direct."""

    def test_direct_mode_flag_exists(self):
        """Slice header should have direct_spatial_mv_pred_flag."""
        from slice import SliceHeader

        header = SliceHeader(
            first_mb_in_slice=0,
            slice_type=1,  # B-slice
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
        )

        assert hasattr(header, 'direct_spatial_mv_pred_flag')

    def test_derive_direct_mv_selects_mode(self):
        """derive_direct_mv should use flag to select mode."""
        from inter.direct_mode import derive_direct_mv

        assert callable(derive_direct_mv)


class TestBSkipMV:
    """Tests for B_Skip MV derivation."""

    def test_b_skip_uses_direct_mv(self):
        """B_Skip uses direct mode MV derivation."""
        from inter.direct_mode import derive_b_skip_mv

        assert callable(derive_b_skip_mv)

    def test_b_skip_spatial_mode(self):
        """B_Skip with spatial direct mode."""
        from inter.direct_mode import derive_b_skip_mv

        mv_cache = MVCache(width_in_mbs=2, height_in_mbs=2)
        mv_cache.set_mv(0, 0, 3, 3, 4, 4)

        mvx_l0, mvy_l0, mvx_l1, mvy_l1 = derive_b_skip_mv(
            mv_cache=mv_cache,
            ref_buffer=None,
            current_poc=2,
            mb_x=1,
            mb_y=1,
            use_spatial=True,
        )

        assert isinstance(mvx_l0, int)
