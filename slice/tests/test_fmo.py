# h264/slice/tests/test_fmo.py
"""Tests for Flexible Macroblock Ordering (FMO) support.

H.264 Spec Reference:
- Section 8.2.2: Decoding process for macroblock to slice group map

FMO allows macroblocks to be assigned to different slice groups using
various mapping patterns. This enables error resilience and flexible
picture partitioning.

These are TDD RED tests - they test functionality not yet implemented.
"""

import numpy as np
import pytest

from parameters.pps import PPS
from parameters.sps import SPS


# All tests should fail initially since FMO is not implemented
pytestmark = pytest.mark.xfail(reason="FMO not implemented")


class TestSliceGroupMapType0Interleaved:
    """Tests for slice_group_map_type 0 (interleaved run-length).

    H.264 Spec: Section 8.2.2.1
    Macroblocks are interleaved among slice groups based on run_length_minus1
    for each group. Groups take turns, each group getting run_length MBs.
    """

    def _create_sps(self, width_mbs: int = 4, height_mbs: int = 4) -> SPS:
        """Create SPS for FMO testing."""
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def _create_pps_type0(
        self,
        num_slice_groups: int,
        run_lengths: list,
    ) -> PPS:
        """Create PPS with interleaved slice groups (map_type=0)."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_slice_groups_minus1=num_slice_groups - 1,
            slice_group_map_type=0,
            run_length_minus1=run_lengths,
        )

    def test_two_groups_alternating_rows(self):
        """Two groups alternating every row (4 MBs per row)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type0(
            num_slice_groups=2,
            run_lengths=[3, 3],  # run_length_minus1=[3,3] means 4 MBs each
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # 4x4 = 16 MBs, alternating rows:
        # Row 0 (MB 0-3): Group 0
        # Row 1 (MB 4-7): Group 1
        # Row 2 (MB 8-11): Group 0
        # Row 3 (MB 12-15): Group 1
        assert len(slice_group_map) == 16
        assert slice_group_map[0] == 0
        assert slice_group_map[3] == 0
        assert slice_group_map[4] == 1
        assert slice_group_map[7] == 1
        assert slice_group_map[8] == 0
        assert slice_group_map[12] == 1

    def test_two_groups_single_mb_alternating(self):
        """Two groups alternating every single MB."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=2)
        pps = self._create_pps_type0(
            num_slice_groups=2,
            run_lengths=[0, 0],  # run_length_minus1=[0,0] means 1 MB each
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # 4x2 = 8 MBs, alternating:
        # MB 0: Group 0, MB 1: Group 1, MB 2: Group 0, ...
        assert len(slice_group_map) == 8
        for i in range(8):
            assert slice_group_map[i] == i % 2

    def test_three_groups_interleaved(self):
        """Three groups with different run lengths."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=6, height_mbs=3)
        pps = self._create_pps_type0(
            num_slice_groups=3,
            run_lengths=[1, 2, 2],  # Group 0: 2 MBs, Group 1: 3 MBs, Group 2: 3 MBs
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # 6x3 = 18 MBs
        # First cycle: MB 0-1 (G0), MB 2-4 (G1), MB 5-7 (G2)
        # Second cycle: MB 8-9 (G0), MB 10-12 (G1), MB 13-15 (G2)
        # Remaining: MB 16-17 (G0)
        assert len(slice_group_map) == 18
        assert slice_group_map[0] == 0
        assert slice_group_map[1] == 0
        assert slice_group_map[2] == 1
        assert slice_group_map[4] == 1
        assert slice_group_map[5] == 2

    def test_uneven_distribution_last_group_gets_remainder(self):
        """When total MBs don't divide evenly, distribution wraps."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=3, height_mbs=3)
        pps = self._create_pps_type0(
            num_slice_groups=2,
            run_lengths=[1, 1],  # 2 MBs each per cycle
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # 3x3 = 9 MBs, cycle = 4 MBs
        # Cycle 1: MB 0-1 (G0), MB 2-3 (G1)
        # Cycle 2: MB 4-5 (G0), MB 6-7 (G1)
        # Remaining: MB 8 (G0)
        assert len(slice_group_map) == 9
        assert slice_group_map[8] == 0


class TestSliceGroupMapType1Dispersed:
    """Tests for slice_group_map_type 1 (dispersed).

    H.264 Spec: Section 8.2.2.2
    Macroblocks are assigned in a dispersed pattern based on
    (i % (num_slice_groups_minus1 + 1)) where pattern varies by row.
    """

    def _create_sps(self, width_mbs: int = 4, height_mbs: int = 4) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def _create_pps_type1(self, num_slice_groups: int) -> PPS:
        """Create PPS with dispersed slice groups (map_type=1)."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_slice_groups_minus1=num_slice_groups - 1,
            slice_group_map_type=1,
        )

    def test_two_groups_checkerboard_pattern(self):
        """Two groups create checkerboard-like pattern."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type1(num_slice_groups=2)

        slice_group_map = generate_slice_group_map(sps, pps)

        # Dispersed pattern formula (Section 8.2.2.2):
        # mapUnitToSliceGroupMap[i] = ((i % PicWidthInMbs) +
        #   (((i / PicWidthInMbs) * (num_slice_groups_minus1 + 1)) / 2))
        #   % (num_slice_groups_minus1 + 1)
        assert len(slice_group_map) == 16
        # For 2 groups, 4-wide: creates a pattern where adjacent MBs alternate
        # Row 0: 0,1,0,1 (offset 0)
        # Row 1: 1,0,1,0 (offset 1)
        # etc.
        assert slice_group_map[0] == 0
        assert slice_group_map[1] == 1
        assert slice_group_map[4] == 1  # Row 1, col 0
        assert slice_group_map[5] == 0  # Row 1, col 1

    def test_three_groups_dispersed(self):
        """Three groups dispersed pattern."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=6, height_mbs=4)
        pps = self._create_pps_type1(num_slice_groups=3)

        slice_group_map = generate_slice_group_map(sps, pps)

        # 6x4 = 24 MBs, 3 groups
        # Pattern cycles through 0,1,2 with row offset
        assert len(slice_group_map) == 24
        # Row 0: 0,1,2,0,1,2
        assert slice_group_map[0] == 0
        assert slice_group_map[1] == 1
        assert slice_group_map[2] == 2
        assert slice_group_map[3] == 0

    def test_dispersed_with_odd_width(self):
        """Dispersed pattern with odd picture width."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=5, height_mbs=3)
        pps = self._create_pps_type1(num_slice_groups=2)

        slice_group_map = generate_slice_group_map(sps, pps)

        # 5x3 = 15 MBs
        # Odd width changes the alternation pattern per row
        assert len(slice_group_map) == 15


class TestSliceGroupMapType2Foreground:
    """Tests for slice_group_map_type 2 (foreground with leftover).

    H.264 Spec: Section 8.2.2.3
    Defines rectangular foreground regions with a leftover background group.
    Each foreground group i has a top_left[i] and bottom_right[i] corner.
    """

    def _create_sps(self, width_mbs: int = 8, height_mbs: int = 6) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def _create_pps_type2(
        self,
        num_slice_groups: int,
        top_left: list,
        bottom_right: list,
    ) -> PPS:
        """Create PPS with foreground/background groups (map_type=2)."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_slice_groups_minus1=num_slice_groups - 1,
            slice_group_map_type=2,
            top_left=top_left,
            bottom_right=bottom_right,
        )

    def test_single_foreground_rectangle(self):
        """Single foreground rectangle in center."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=8, height_mbs=6)
        # 8x6 = 48 MBs, foreground region (2,1) to (5,4)
        # top_left = row*width + col = 1*8 + 2 = 10
        # bottom_right = 4*8 + 5 = 37
        pps = self._create_pps_type2(
            num_slice_groups=2,
            top_left=[10],
            bottom_right=[37],
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 48
        # Group 0 is foreground rectangle
        assert slice_group_map[10] == 0  # Top-left of rect
        assert slice_group_map[11] == 0  # Inside rect
        assert slice_group_map[37] == 0  # Bottom-right of rect
        # Group 1 is background (outside rectangle)
        assert slice_group_map[0] == 1  # Outside
        assert slice_group_map[7] == 1  # Outside
        assert slice_group_map[47] == 1  # Outside

    def test_two_overlapping_foreground_rectangles(self):
        """Two foreground rectangles with overlap (earlier group wins)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=8, height_mbs=6)
        # Rectangle 0: (1,1) to (4,3) -> top_left=9, bottom_right=28
        # Rectangle 1: (3,2) to (6,4) -> top_left=19, bottom_right=46
        pps = self._create_pps_type2(
            num_slice_groups=3,
            top_left=[9, 19],
            bottom_right=[28, 46],
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 48
        # Group 0 rectangle
        assert slice_group_map[9] == 0
        # Overlap region (earlier group wins)
        assert slice_group_map[19] == 0  # In both, but G0 takes precedence
        # Group 1 only region
        assert slice_group_map[46] == 1
        # Background (Group 2)
        assert slice_group_map[0] == 2

    def test_foreground_rectangle_at_corner(self):
        """Foreground rectangle at top-left corner."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        # Rectangle: (0,0) to (1,1) -> top_left=0, bottom_right=5
        pps = self._create_pps_type2(
            num_slice_groups=2,
            top_left=[0],
            bottom_right=[5],
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # 4x4 = 16 MBs
        assert len(slice_group_map) == 16
        assert slice_group_map[0] == 0  # (0,0)
        assert slice_group_map[1] == 0  # (0,1)
        assert slice_group_map[4] == 0  # (1,0)
        assert slice_group_map[5] == 0  # (1,1)
        assert slice_group_map[2] == 1  # Outside


class TestSliceGroupMapType3BoxOut:
    """Tests for slice_group_map_type 3 (box-out).

    H.264 Spec: Section 8.2.2.4
    Macroblocks spiral out from center (or in from edges) based on
    slice_group_change_direction_flag and slice_group_change_rate.
    """

    def _create_sps(self, width_mbs: int = 4, height_mbs: int = 4) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def _create_pps_type3(
        self,
        change_direction: bool,
        change_rate_minus1: int,
        slice_group_change_cycle: int = 1,
    ) -> PPS:
        """Create PPS with box-out pattern (map_type=3)."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_slice_groups_minus1=1,  # Always 2 groups for type 3
            slice_group_map_type=3,
            slice_group_change_direction_flag=change_direction,
            slice_group_change_rate_minus1=change_rate_minus1,
            slice_group_change_cycle=slice_group_change_cycle,
        )

    def test_box_out_clockwise_from_center(self):
        """Box-out clockwise from center (direction=0)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type3(
            change_direction=False,  # Clockwise
            change_rate_minus1=3,  # 4 MBs per cycle
            slice_group_change_cycle=2,  # 8 MBs changed
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # 4x4 = 16 MBs, 8 changed from center outward
        # Center MBs (5, 6, 9, 10) and surrounding spiral should be group 0
        assert len(slice_group_map) == 16
        # Exact pattern depends on implementation

    def test_box_out_counterclockwise(self):
        """Box-out counter-clockwise (direction=1)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type3(
            change_direction=True,  # Counter-clockwise
            change_rate_minus1=1,
            slice_group_change_cycle=4,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 16

    def test_box_out_zero_change_cycle(self):
        """Box-out with change_cycle=0 (all MBs in group 1)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type3(
            change_direction=False,
            change_rate_minus1=1,
            slice_group_change_cycle=0,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # All MBs should be in group 1 (no box-out yet)
        assert len(slice_group_map) == 16
        assert all(g == 1 for g in slice_group_map)

    def test_box_out_full_picture(self):
        """Box-out covering entire picture (all group 0)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type3(
            change_direction=False,
            change_rate_minus1=3,  # 4 MBs per cycle
            slice_group_change_cycle=4,  # 16 MBs total
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # All MBs should be group 0
        assert len(slice_group_map) == 16
        assert all(g == 0 for g in slice_group_map)


class TestSliceGroupMapType4RasterScan:
    """Tests for slice_group_map_type 4 (raster scan).

    H.264 Spec: Section 8.2.2.5
    Divides picture into two groups in raster scan order.
    Group 0 contains first N MBs, group 1 contains the rest.
    """

    def _create_sps(self, width_mbs: int = 4, height_mbs: int = 4) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def _create_pps_type4(
        self,
        change_direction: bool,
        change_rate_minus1: int,
        slice_group_change_cycle: int,
    ) -> PPS:
        """Create PPS with raster scan pattern (map_type=4)."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_slice_groups_minus1=1,
            slice_group_map_type=4,
            slice_group_change_direction_flag=change_direction,
            slice_group_change_rate_minus1=change_rate_minus1,
            slice_group_change_cycle=slice_group_change_cycle,
        )

    def test_raster_scan_first_half_group0(self):
        """First half of MBs in group 0 (direction=0)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type4(
            change_direction=False,
            change_rate_minus1=3,  # 4 MBs per cycle
            slice_group_change_cycle=2,  # 8 MBs in group 0
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # 4x4 = 16 MBs, first 8 in group 0
        assert len(slice_group_map) == 16
        for i in range(8):
            assert slice_group_map[i] == 0
        for i in range(8, 16):
            assert slice_group_map[i] == 1

    def test_raster_scan_reversed_direction(self):
        """Last half of MBs in group 0 (direction=1)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type4(
            change_direction=True,  # Reverse
            change_rate_minus1=3,
            slice_group_change_cycle=2,  # 8 MBs changed from end
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # 4x4 = 16 MBs, last 8 in group 0
        assert len(slice_group_map) == 16
        for i in range(8):
            assert slice_group_map[i] == 1
        for i in range(8, 16):
            assert slice_group_map[i] == 0

    def test_raster_scan_single_mb_group0(self):
        """Only first MB in group 0."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type4(
            change_direction=False,
            change_rate_minus1=0,  # 1 MB per cycle
            slice_group_change_cycle=1,  # 1 MB in group 0
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 16
        assert slice_group_map[0] == 0
        for i in range(1, 16):
            assert slice_group_map[i] == 1


class TestSliceGroupMapType5Wipe:
    """Tests for slice_group_map_type 5 (wipe).

    H.264 Spec: Section 8.2.2.6
    Vertical wipe pattern dividing picture column by column.
    """

    def _create_sps(self, width_mbs: int = 6, height_mbs: int = 4) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def _create_pps_type5(
        self,
        change_direction: bool,
        change_rate_minus1: int,
        slice_group_change_cycle: int,
    ) -> PPS:
        """Create PPS with wipe pattern (map_type=5)."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_slice_groups_minus1=1,
            slice_group_map_type=5,
            slice_group_change_direction_flag=change_direction,
            slice_group_change_rate_minus1=change_rate_minus1,
            slice_group_change_cycle=slice_group_change_cycle,
        )

    def test_wipe_left_to_right(self):
        """Wipe from left to right (direction=0)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=6, height_mbs=4)
        pps = self._create_pps_type5(
            change_direction=False,
            change_rate_minus1=3,  # 4 MBs per cycle
            slice_group_change_cycle=3,  # 12 MBs = 3 columns
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # 6x4 = 24 MBs, first 3 columns in group 0
        assert len(slice_group_map) == 24
        # Column 0, 1, 2 should be group 0
        for row in range(4):
            for col in range(3):
                mb_addr = row * 6 + col
                assert slice_group_map[mb_addr] == 0
        # Column 3, 4, 5 should be group 1
        for row in range(4):
            for col in range(3, 6):
                mb_addr = row * 6 + col
                assert slice_group_map[mb_addr] == 1

    def test_wipe_right_to_left(self):
        """Wipe from right to left (direction=1)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=6, height_mbs=4)
        pps = self._create_pps_type5(
            change_direction=True,
            change_rate_minus1=3,
            slice_group_change_cycle=3,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # Last 3 columns should be group 0
        assert len(slice_group_map) == 24
        for row in range(4):
            for col in range(3):
                mb_addr = row * 6 + col
                assert slice_group_map[mb_addr] == 1
            for col in range(3, 6):
                mb_addr = row * 6 + col
                assert slice_group_map[mb_addr] == 0

    def test_wipe_single_column(self):
        """Wipe with only first column in group 0."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = self._create_pps_type5(
            change_direction=False,
            change_rate_minus1=3,  # 4 MBs per cycle (1 column)
            slice_group_change_cycle=1,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # First column (MBs 0, 4, 8, 12) in group 0
        assert slice_group_map[0] == 0
        assert slice_group_map[4] == 0
        assert slice_group_map[8] == 0
        assert slice_group_map[12] == 0
        # Others in group 1
        assert slice_group_map[1] == 1


class TestSliceGroupMapType6Explicit:
    """Tests for slice_group_map_type 6 (explicit).

    H.264 Spec: Section 8.2.2.7
    Each macroblock's slice group is explicitly specified in PPS.
    """

    def _create_sps(self, width_mbs: int = 4, height_mbs: int = 4) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def _create_pps_type6(
        self,
        num_slice_groups: int,
        slice_group_id: list,
    ) -> PPS:
        """Create PPS with explicit slice group map (map_type=6)."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_slice_groups_minus1=num_slice_groups - 1,
            slice_group_map_type=6,
            slice_group_id=slice_group_id,
        )

    def test_explicit_two_groups_custom_pattern(self):
        """Explicit assignment with custom 2-group pattern."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        # Custom pattern: diagonal split
        explicit_map = [
            0, 0, 0, 1,
            0, 0, 1, 1,
            0, 1, 1, 1,
            1, 1, 1, 1,
        ]
        pps = self._create_pps_type6(
            num_slice_groups=2,
            slice_group_id=explicit_map,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 16
        assert slice_group_map == explicit_map

    def test_explicit_four_quadrants(self):
        """Explicit assignment dividing picture into 4 quadrants."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        # 4 quadrants
        explicit_map = [
            0, 0, 1, 1,
            0, 0, 1, 1,
            2, 2, 3, 3,
            2, 2, 3, 3,
        ]
        pps = self._create_pps_type6(
            num_slice_groups=4,
            slice_group_id=explicit_map,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 16
        assert slice_group_map == explicit_map

    def test_explicit_single_group(self):
        """Explicit with all MBs in single group."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=2, height_mbs=2)
        explicit_map = [0, 0, 0, 0]
        pps = self._create_pps_type6(
            num_slice_groups=1,
            slice_group_id=explicit_map,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 4
        assert all(g == 0 for g in slice_group_map)

    def test_explicit_random_assignment(self):
        """Explicit with pseudo-random group assignment."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        explicit_map = [
            2, 0, 1, 2,
            1, 2, 0, 1,
            0, 1, 2, 0,
            2, 0, 1, 2,
        ]
        pps = self._create_pps_type6(
            num_slice_groups=3,
            slice_group_id=explicit_map,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 16
        assert slice_group_map == explicit_map


class TestMbToSliceGroupMapGeneration:
    """Tests for MbToSliceGroupMap array generation.

    H.264 Spec: Section 8.2.2
    The MbToSliceGroupMap array maps each macroblock address to its
    slice group ID.
    """

    def _create_sps(self, width_mbs: int = 4, height_mbs: int = 4) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def test_map_length_equals_total_mbs(self):
        """MbToSliceGroupMap has one entry per macroblock."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=8, height_mbs=6)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=1,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 48  # 8 * 6

    def test_map_values_within_range(self):
        """All map values are valid slice group IDs."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(
            num_slice_groups_minus1=3,  # 4 groups
            slice_group_map_type=1,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert all(0 <= g <= 3 for g in slice_group_map)

    def test_single_slice_group_all_zeros(self):
        """Single slice group results in all-zero map."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(
            num_slice_groups_minus1=0,  # Single group
            slice_group_map_type=0,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert all(g == 0 for g in slice_group_map)

    def test_map_as_numpy_array(self):
        """MbToSliceGroupMap can be returned as numpy array."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=1,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # Should be convertible to numpy for efficient indexing
        map_array = np.array(slice_group_map)
        assert map_array.shape == (16,)
        assert map_array.dtype in (np.int32, np.int64, np.uint8)


class TestNextMbAddressWithFmo:
    """Tests for NextMbAddress calculation with FMO.

    H.264 Spec: Section 8.2.2.8
    NextMbAddress(n) returns the next macroblock address in the same
    slice group, or -1 if no more MBs in that group.
    """

    def _create_sps(self, width_mbs: int = 4, height_mbs: int = 4) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def test_next_mb_addr_no_fmo(self):
        """NextMbAddress without FMO is simply n+1."""
        from slice.fmo import get_next_mb_address

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(num_slice_groups_minus1=0)  # No FMO

        next_addr = get_next_mb_address(
            current_mb_addr=5,
            sps=sps,
            pps=pps,
        )

        assert next_addr == 6

    def test_next_mb_addr_last_mb_returns_none(self):
        """NextMbAddress for last MB returns -1 or None."""
        from slice.fmo import get_next_mb_address

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(num_slice_groups_minus1=0)

        # Last MB is 15 (4*4 - 1)
        next_addr = get_next_mb_address(
            current_mb_addr=15,
            sps=sps,
            pps=pps,
        )

        assert next_addr == -1 or next_addr is None

    def test_next_mb_addr_interleaved_skips_other_group(self):
        """NextMbAddress with interleaved FMO skips MBs in other groups."""
        from slice.fmo import get_next_mb_address

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=0,
            run_length_minus1=[3, 3],  # Alternating rows
        )

        # In group 0 (MBs 0-3, 8-11), next after 3 should be 8
        next_addr = get_next_mb_address(
            current_mb_addr=3,
            sps=sps,
            pps=pps,
            slice_group_id=0,
        )

        assert next_addr == 8

    def test_next_mb_addr_dispersed_pattern(self):
        """NextMbAddress with dispersed FMO follows dispersed pattern."""
        from slice.fmo import get_next_mb_address

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=1,
        )

        # With dispersed 2-group pattern, MBs alternate
        # Group 0: 0, 2, 4, 6, ... or similar pattern
        next_addr = get_next_mb_address(
            current_mb_addr=0,
            sps=sps,
            pps=pps,
            slice_group_id=0,
        )

        # Next should skip MB 1 if it's in group 1
        assert next_addr > 0

    def test_next_mb_addr_at_end_of_slice_group(self):
        """NextMbAddress returns -1 when at end of slice group."""
        from slice.fmo import get_next_mb_address

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=0,
            run_length_minus1=[3, 3],
        )

        # Group 0 has MBs 0-3, 8-11. Last MB in group 0 is 11
        next_addr = get_next_mb_address(
            current_mb_addr=11,
            sps=sps,
            pps=pps,
            slice_group_id=0,
        )

        assert next_addr == -1 or next_addr is None

    def test_next_mb_addr_explicit_map(self):
        """NextMbAddress with explicit map follows explicit assignment."""
        from slice.fmo import get_next_mb_address

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        explicit_map = [
            0, 1, 0, 1,
            1, 0, 1, 0,
            0, 1, 0, 1,
            1, 0, 1, 0,
        ]
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=6,
            slice_group_id=explicit_map,
        )

        # Group 0 MBs: 0, 2, 5, 7, 8, 10, 13, 15
        next_addr = get_next_mb_address(
            current_mb_addr=0,
            sps=sps,
            pps=pps,
            slice_group_id=0,
        )

        assert next_addr == 2  # Skip MB 1 (group 1)


class TestFmoEdgeCases:
    """Edge case tests for FMO implementation."""

    def _create_sps(self, width_mbs: int, height_mbs: int) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def test_single_macroblock_picture(self):
        """FMO with 1x1 macroblock picture."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=1, height_mbs=1)
        pps = PPS(
            num_slice_groups_minus1=0,
            slice_group_map_type=0,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 1
        assert slice_group_map[0] == 0

    def test_single_row_picture(self):
        """FMO with single row (1xN) picture."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=8, height_mbs=1)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=1,  # Dispersed
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 8

    def test_single_column_picture(self):
        """FMO with single column (Nx1) picture."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=1, height_mbs=8)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=1,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 8

    def test_max_slice_groups(self):
        """FMO with maximum number of slice groups (8)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=8, height_mbs=8)
        pps = PPS(
            num_slice_groups_minus1=7,  # 8 groups (max allowed)
            slice_group_map_type=1,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 64
        assert all(0 <= g <= 7 for g in slice_group_map)

    def test_large_picture(self):
        """FMO with large picture (1080p = 120x68 MBs)."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=120, height_mbs=68)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=1,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        assert len(slice_group_map) == 120 * 68

    def test_mbaff_mode_fmo(self):
        """FMO with MBAFF (macroblock-adaptive frame-field) mode."""
        from slice.fmo import generate_slice_group_map

        sps = SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=3,  # 4 pairs
            frame_mbs_only_flag=False,
            mb_adaptive_frame_field_flag=True,
        )
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=1,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # With MBAFF, map_units are MB pairs
        # Total MBs = 2 * pic_height_in_map_units * pic_width_in_mbs = 2*4*4 = 32
        # But slice group map is for map units = 16
        assert len(slice_group_map) == 16


class TestFmoIntegrationWithSliceData:
    """Integration tests for FMO with slice data parsing."""

    def _create_sps(self, width_mbs: int = 4, height_mbs: int = 4) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=width_mbs - 1,
            pic_height_in_map_units_minus1=height_mbs - 1,
            frame_mbs_only_flag=True,
        )

    def test_slice_contains_correct_mbs_with_fmo(self):
        """Slice only processes MBs in its slice group."""
        from slice.fmo import generate_slice_group_map, get_next_mb_address
        from slice.slice_header import SliceHeader, SliceType

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=0,
            run_length_minus1=[3, 3],
        )
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # MBs for slice group 0: 0,1,2,3,8,9,10,11
        group_0_mbs = [i for i, g in enumerate(slice_group_map) if g == 0]
        assert group_0_mbs == [0, 1, 2, 3, 8, 9, 10, 11]

    def test_mb_skip_run_with_fmo(self):
        """mb_skip_run correctly skips MBs in same slice group."""
        from slice.fmo import generate_slice_group_map, get_next_mb_address

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=0,
            run_length_minus1=[3, 3],
        )

        # Skipping 2 MBs in group 0 starting from MB 0
        # Should skip 0, 1 and land on 2
        current = 0
        skip_count = 2
        skipped = []

        for _ in range(skip_count):
            skipped.append(current)
            current = get_next_mb_address(current, sps, pps, slice_group_id=0)

        assert skipped == [0, 1]
        assert current == 2

    def test_first_mb_in_slice_with_fmo(self):
        """first_mb_in_slice correctly identifies starting MB in slice group."""
        from slice.fmo import generate_slice_group_map

        sps = self._create_sps(width_mbs=4, height_mbs=4)
        pps = PPS(
            num_slice_groups_minus1=1,
            slice_group_map_type=0,
            run_length_minus1=[3, 3],
        )

        slice_group_map = generate_slice_group_map(sps, pps)

        # First MB of group 1 is MB 4
        first_mb_group_1 = next(
            i for i, g in enumerate(slice_group_map) if g == 1
        )
        assert first_mb_group_1 == 4
