# h264/slice/tests/test_slice_data.py
"""Tests for slice data parsing.

H.264 Spec Reference:
- Section 7.3.4: Slice data syntax
- Section 7.4.4: Slice data semantics

These are TDD RED tests - they test functionality not yet implemented.
"""

import pytest

from bitstream import NumpyBitWriter
from parameters import SPS, PPS
from slice.slice_header import SliceHeader, SliceType


# Skip all tests - these are RED tests for unimplemented functionality
pytestmark = pytest.mark.skip(reason="RED tests - slice data parsing not implemented")


class TestMbSkipRunPSlice:
    """Tests for mb_skip_run parsing in P-slices.

    H.264 Spec: Section 7.3.4, 7.4.4
    In P-slices using CAVLC, mb_skip_run specifies the number of consecutive
    skipped macroblocks before the current macroblock.
    """

    def _create_default_sps(self) -> SPS:
        """Create default SPS for testing (4x3 macroblocks = 64x48 pixels)."""
        return SPS(
            profile_idc=66,
            level_idc=30,
            log2_max_frame_num_minus4=0,
            pic_order_cnt_type=2,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=3,  # 4 macroblocks wide
            pic_height_in_map_units_minus1=2,  # 3 macroblocks tall
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        """Create default PPS for testing (CAVLC mode)."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,  # CAVLC
            num_ref_idx_l0_default_active_minus1=0,
            deblocking_filter_control_present_flag=False,
        )

    def test_mb_skip_run_zero(self):
        """Parse P-slice with mb_skip_run=0 (no skipped MBs)."""
        # Imports that should exist when implemented
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=0,
            num_ref_idx_l0_active_minus1=0,
        )

        writer = NumpyBitWriter()
        writer.write_ue(0)  # mb_skip_run = 0 (no skip before first MB)
        # ... macroblock data would follow
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        assert slice_data.mb_skip_run[0] == 0
        assert slice_data.current_mb_addr == 0

    def test_mb_skip_run_single_skip(self):
        """Parse P-slice with mb_skip_run=1 (one skipped MB)."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=0,
            num_ref_idx_l0_active_minus1=0,
        )

        writer = NumpyBitWriter()
        writer.write_ue(1)  # mb_skip_run = 1 (skip first MB)
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # First MB was skipped, current should be at MB 1
        assert slice_data.mb_skip_run[0] == 1
        assert slice_data.skipped_mb_addresses == [0]

    def test_mb_skip_run_multiple_skips(self):
        """Parse P-slice with mb_skip_run=5 (five consecutive skipped MBs)."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=0,
            num_ref_idx_l0_active_minus1=0,
        )

        writer = NumpyBitWriter()
        writer.write_ue(5)  # mb_skip_run = 5
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        assert slice_data.mb_skip_run[0] == 5
        assert slice_data.skipped_mb_addresses == [0, 1, 2, 3, 4]
        # Next non-skipped MB should be at address 5
        assert slice_data.current_mb_addr == 5

    def test_mb_skip_run_entire_row(self):
        """Parse P-slice skipping entire first row (4 MBs)."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()  # 4x3 MBs
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=0,
            num_ref_idx_l0_active_minus1=0,
        )

        writer = NumpyBitWriter()
        writer.write_ue(4)  # Skip entire first row
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        assert slice_data.mb_skip_run[0] == 4
        # First row (0-3) should be skipped
        assert len(slice_data.skipped_mb_addresses) == 4
        # Next MB should be first of second row
        assert slice_data.current_mb_addr == 4


class TestMbSkipRunBSlice:
    """Tests for mb_skip_run parsing in B-slices.

    B-slices also support mb_skip_run for skipped macroblocks, but the
    motion vector derivation differs from P-slices.
    """

    def _create_default_sps(self) -> SPS:
        """Create default SPS for testing."""
        return SPS(
            profile_idc=77,  # Main profile for B-slices
            level_idc=30,
            log2_max_frame_num_minus4=0,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,
            max_num_ref_frames=2,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=2,
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        """Create default PPS for B-slice testing."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_ref_idx_l0_default_active_minus1=0,
            num_ref_idx_l1_default_active_minus1=0,
            deblocking_filter_control_present_flag=False,
        )

    def test_b_slice_mb_skip_run_direct_mode(self):
        """Parse B-slice with skipped MBs using direct mode."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.B,
            first_mb_in_slice=0,
            num_ref_idx_l0_active_minus1=0,
            num_ref_idx_l1_active_minus1=0,
            direct_spatial_mv_pred_flag=True,  # Spatial direct mode
        )

        writer = NumpyBitWriter()
        writer.write_ue(3)  # mb_skip_run = 3
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        assert slice_data.mb_skip_run[0] == 3
        # B_Skip uses direct mode for motion vectors
        assert slice_data.skipped_mb_addresses == [0, 1, 2]

    def test_b_slice_mb_skip_run_temporal_direct(self):
        """Parse B-slice with skipped MBs using temporal direct mode."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.B,
            first_mb_in_slice=0,
            direct_spatial_mv_pred_flag=False,  # Temporal direct mode
        )

        writer = NumpyBitWriter()
        writer.write_ue(2)  # mb_skip_run = 2
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        assert slice_data.mb_skip_run[0] == 2
        assert len(slice_data.skipped_mb_addresses) == 2


class TestEndOfSliceDetection:
    """Tests for end_of_slice_flag and slice termination detection.

    H.264 Spec: Section 7.3.4
    """

    def _create_default_sps(self) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=1,  # 2x2 MBs = 32x32 pixels
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            deblocking_filter_control_present_flag=False,
        )

    def test_end_of_slice_after_last_mb(self):
        """Detect end of slice after processing all MBs."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()  # 2x2 = 4 MBs
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
        )

        # Create bitstream with 4 I-MBs
        writer = NumpyBitWriter()
        # ... MB data for all 4 MBs ...
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Should detect end after MB 3 (last MB)
        assert slice_data.end_of_slice_detected is True
        assert slice_data.last_mb_addr == 3

    def test_end_of_slice_partial_slice(self):
        """Detect end of slice in partial slice (not all MBs)."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()  # 2x2 = 4 MBs
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
        )

        # Create bitstream that ends after 2 MBs
        writer = NumpyBitWriter()
        # ... MB data for first 2 MBs only ...
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Should detect end of slice data
        assert slice_data.end_of_slice_detected is True
        # Only 2 MBs processed
        assert len(slice_data.macroblocks) == 2

    def test_more_rbsp_data_detection(self):
        """Correctly identify more_rbsp_data() condition."""
        from slice.slice_data import parse_slice_data, more_rbsp_data

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(slice_type=SliceType.I, first_mb_in_slice=0)

        # RBSP with trailing bits (1 followed by zeros)
        rbsp_with_trailing = bytes([0x80])  # 10000000 - stop bit

        # Should return False (no more data, just trailing bits)
        assert more_rbsp_data(rbsp_with_trailing, bit_position=0) is False

        # RBSP with more data
        rbsp_with_data = bytes([0xFF, 0x80])  # Data + stop bit
        assert more_rbsp_data(rbsp_with_data, bit_position=0) is True


class TestMacroblockAddressCalculation:
    """Tests for macroblock address calculation with skipped MBs.

    H.264 Spec: Section 6.4, 7.4.4
    """

    def _create_default_sps(self) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,  # 4 MBs wide
            pic_height_in_map_units_minus1=2,  # 3 MBs tall = 12 total
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
        )

    def test_mb_addr_after_skip(self):
        """Verify MB address calculation after skip run."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=0,
        )

        writer = NumpyBitWriter()
        writer.write_ue(3)  # Skip MBs 0, 1, 2
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # CurrMbAddr should be 3 after skipping 0, 1, 2
        assert slice_data.current_mb_addr == 3

    def test_mb_addr_with_first_mb_offset(self):
        """MB address calculation when first_mb_in_slice is non-zero."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=4,  # Start at second row
        )

        writer = NumpyBitWriter()
        writer.write_ue(2)  # Skip MBs 4, 5
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Started at 4, skipped 2, now at 6
        assert slice_data.current_mb_addr == 6
        assert slice_data.skipped_mb_addresses == [4, 5]

    def test_mb_addr_wrap_to_next_row(self):
        """MB address wraps correctly to next row."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()  # 4 MBs per row
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=2,  # Start mid-row
        )

        writer = NumpyBitWriter()
        writer.write_ue(3)  # Skip MBs 2, 3, 4 (crosses row boundary)
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Skipped 2, 3 (end of row 0), 4 (start of row 1)
        assert slice_data.skipped_mb_addresses == [2, 3, 4]
        assert slice_data.current_mb_addr == 5


class TestSliceGroupHandling:
    """Tests for Flexible Macroblock Ordering (FMO) slice group handling.

    H.264 Spec: Section 8.2.2
    FMO allows macroblocks to be assigned to different slice groups,
    enabling error resilience and other features.
    """

    def _create_fmo_sps(self) -> SPS:
        """Create SPS for FMO testing."""
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,  # 4 MBs wide
            pic_height_in_map_units_minus1=3,  # 4 MBs tall = 16 total
            frame_mbs_only_flag=True,
        )

    def _create_fmo_pps(self, num_slice_groups: int, map_type: int) -> PPS:
        """Create PPS with FMO enabled."""
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_slice_groups_minus1=num_slice_groups - 1,
            slice_group_map_type=map_type,
        )

    def test_interleaved_slice_groups(self):
        """Parse slice data with interleaved slice groups (map_type=0)."""
        from slice.slice_data import parse_slice_data, SliceData
        from slice.slice_data import build_slice_group_map

        sps = self._create_fmo_sps()
        pps = self._create_fmo_pps(num_slice_groups=2, map_type=0)
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
        )

        # Build slice group map for interleaved groups
        # With run_length_minus1=[3, 3], groups alternate every 4 MBs
        slice_group_map = build_slice_group_map(sps, pps)

        # Group 0: MBs 0-3, 8-11
        # Group 1: MBs 4-7, 12-15
        assert slice_group_map[0] == 0
        assert slice_group_map[4] == 1
        assert slice_group_map[8] == 0
        assert slice_group_map[12] == 1

    def test_dispersed_slice_groups(self):
        """Parse slice data with dispersed slice groups (map_type=1)."""
        from slice.slice_data import build_slice_group_map

        sps = self._create_fmo_sps()
        pps = self._create_fmo_pps(num_slice_groups=2, map_type=1)

        slice_group_map = build_slice_group_map(sps, pps)

        # Dispersed pattern: checkerboard-like
        # Group alternates based on (i % num_groups)
        # Pattern depends on pic_width_in_mbs and row
        assert len(slice_group_map) == 16

    def test_foreground_background_groups(self):
        """Parse slice with foreground/background groups (map_type=2)."""
        from slice.slice_data import build_slice_group_map

        sps = self._create_fmo_sps()
        pps = self._create_fmo_pps(num_slice_groups=2, map_type=2)
        # Would need top_left and bottom_right in PPS

        slice_group_map = build_slice_group_map(sps, pps)

        # Rectangular region for foreground, rest is background
        assert len(slice_group_map) == 16

    def test_next_mb_addr_with_fmo(self):
        """Calculate next MB address with FMO enabled."""
        from slice.slice_data import next_mb_addr_fmo

        sps = self._create_fmo_sps()
        pps = self._create_fmo_pps(num_slice_groups=2, map_type=0)

        # With interleaved groups, next MB in same group might not be adjacent
        # For slice group 0 with run_length=4: 0,1,2,3 -> 8,9,10,11
        next_addr = next_mb_addr_fmo(
            current_addr=3,
            slice_group_id=0,
            sps=sps,
            pps=pps
        )

        # Next MB in group 0 after 3 should be 8
        assert next_addr == 8


class TestConstrainedIntraPrediction:
    """Tests for constrained intra prediction flag effect on parsing.

    H.264 Spec: Section 8.3.1
    When constrained_intra_pred_flag is set, intra prediction can only
    use samples from intra-coded neighbors (for error resilience).
    """

    def _create_default_sps(self) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=2,
            frame_mbs_only_flag=True,
        )

    def test_constrained_intra_pred_enabled(self):
        """Parse slice with constrained_intra_pred_flag set."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            constrained_intra_pred_flag=True,  # Enabled
        )
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=0,
        )

        writer = NumpyBitWriter()
        # ... slice data ...
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Parser should track constrained intra mode
        assert slice_data.constrained_intra_pred is True

    def test_constrained_intra_neighbor_availability(self):
        """Verify neighbor availability with constrained intra pred."""
        from slice.slice_data import get_neighbor_availability

        sps = SPS(
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=2,
            frame_mbs_only_flag=True,
        )
        pps = PPS(constrained_intra_pred_flag=True)

        # MB at address 5 (row 1, col 1) with inter-coded left neighbor
        mb_types = {
            4: 'P_Skip',  # Left neighbor is inter-coded
            1: 'I_4x4',   # Top neighbor is intra-coded
        }

        availability = get_neighbor_availability(
            mb_addr=5,
            mb_types=mb_types,
            sps=sps,
            pps=pps
        )

        # With constrained_intra_pred_flag, inter-coded neighbors are unavailable
        assert availability['left'] is False  # Inter-coded, not available
        assert availability['top'] is True    # Intra-coded, available


class TestSliceDataAlignment:
    """Tests for slice data byte alignment and boundaries.

    H.264 Spec: Section 7.3.4
    """

    def _create_default_sps(self) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=1,
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
        )

    def test_slice_data_starts_after_header(self):
        """Slice data starts at correct bit position after header."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
            header_bit_size=24,  # Header was 24 bits
        )

        writer = NumpyBitWriter()
        # ... slice data starting at bit 24 ...
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Verify parsing started at correct position
        assert slice_data.data_start_bit == 24

    def test_byte_alignment_with_cabac(self):
        """CABAC requires byte alignment before slice data."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=True,  # CABAC
        )
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
            header_bit_size=27,  # Not byte-aligned
        )

        writer = NumpyBitWriter()
        # CABAC alignment bits (1 followed by zeros to byte boundary)
        writer.write_bit(1)  # cabac_alignment_one_bit
        writer.write_bits(0, 4)  # cabac_alignment_zero_bits to align
        # ... CABAC slice data ...
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Data should start at byte boundary (bit 32)
        assert slice_data.data_start_bit % 8 == 0

    def test_rbsp_trailing_bits_detection(self):
        """Correctly detect RBSP trailing bits pattern."""
        from slice.slice_data import detect_rbsp_trailing_bits

        # Valid trailing bits: 1 followed by zeros
        valid_trailing = bytes([0x80])  # 10000000
        assert detect_rbsp_trailing_bits(valid_trailing, start_bit=0) is True

        # More complex: data followed by trailing bits
        data_and_trailing = bytes([0xFF, 0x40])  # 11111111 01000000
        # Bit 9 is the stop bit (1), followed by zeros
        assert detect_rbsp_trailing_bits(data_and_trailing, start_bit=9) is True


class TestMultipleMacroblocksInSequence:
    """Tests for parsing multiple consecutive macroblocks.

    H.264 Spec: Section 7.3.4
    """

    def _create_default_sps(self) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,  # 4 MBs wide
            pic_height_in_map_units_minus1=2,  # 3 MBs tall = 12 total
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
        )

    def test_parse_sequence_of_i_mbs(self):
        """Parse sequence of I-macroblocks."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
            slice_qp_delta=0,
        )

        writer = NumpyBitWriter()
        # Create data for 3 I_16x16 macroblocks
        for _ in range(3):
            writer.write_ue(1)  # mb_type = I_16x16_0_0_0
            # ... residual data would follow
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Should have parsed 3 macroblocks
        assert len(slice_data.macroblocks) >= 3
        assert all(mb.mb_type == 'I_16x16' for mb in slice_data.macroblocks[:3])

    def test_parse_mixed_skip_and_coded_mbs(self):
        """Parse P-slice with mixed skipped and coded MBs."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=0,
            num_ref_idx_l0_active_minus1=0,
        )

        writer = NumpyBitWriter()
        # Skip 2 MBs, code 1, skip 1, code 1
        writer.write_ue(2)  # mb_skip_run = 2 (skip MB 0, 1)
        writer.write_ue(1)  # mb_type for MB 2
        # ... residual for MB 2 ...
        writer.write_ue(1)  # mb_skip_run = 1 (skip MB 3)
        writer.write_ue(0)  # mb_type for MB 4
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Verify sequence: skip 0,1 -> code 2 -> skip 3 -> code 4
        assert 0 in slice_data.skipped_mb_addresses
        assert 1 in slice_data.skipped_mb_addresses
        assert 3 in slice_data.skipped_mb_addresses
        assert 2 not in slice_data.skipped_mb_addresses
        assert 4 not in slice_data.skipped_mb_addresses

    def test_parse_all_mbs_in_small_slice(self):
        """Parse complete small slice (2x2 MBs)."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=1,  # 2x2 = 4 MBs
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
        )

        writer = NumpyBitWriter()
        # 4 I-macroblocks
        for _ in range(4):
            writer.write_ue(0)  # mb_type = I_4x4
            # ... prediction modes and residual ...
        # RBSP trailing bits
        writer.write_bit(1)
        writer.write_bits(0, 7)  # Pad to byte
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Should have all 4 MBs
        assert len(slice_data.macroblocks) == 4
        assert slice_data.last_mb_addr == 3
        assert slice_data.end_of_slice_detected is True

    def test_mb_processing_order(self):
        """Verify macroblocks are processed in correct raster order."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()  # 4x3 MBs
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
        )

        writer = NumpyBitWriter()
        # Parse first row (4 MBs)
        for _ in range(4):
            writer.write_ue(0)  # I_4x4
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # Verify raster scan order: 0, 1, 2, 3 (first row)
        for i, mb in enumerate(slice_data.macroblocks[:4]):
            assert mb.mb_addr == i

    def test_qp_tracking_across_mbs(self):
        """Track QP changes across macroblock sequence."""
        from slice.slice_data import parse_slice_data, SliceData

        sps = self._create_default_sps()
        pps = PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            pic_init_qp_minus26=0,  # QP = 26
        )
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
            slice_qp_delta=0,  # Slice QP = 26
        )

        writer = NumpyBitWriter()
        # MB 0: mb_qp_delta = 0, QP = 26
        writer.write_ue(0)  # mb_type
        # ... residual with mb_qp_delta = 0 ...

        # MB 1: mb_qp_delta = 2, QP = 28
        writer.write_ue(0)  # mb_type
        # ... residual with mb_qp_delta = 2 ...
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # QP should be tracked per-MB
        assert slice_data.macroblocks[0].qp == 26
        assert slice_data.macroblocks[1].qp == 28


class TestMultipleSlicesInFrame:
    """Tests for parsing and handling multiple slices within a single frame.

    H.264 Spec: Section 7.4.3
    A frame can be divided into multiple slices, each with its own header.
    """

    def _create_default_sps(self) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,  # 4 MBs wide
            pic_height_in_map_units_minus1=3,  # 4 MBs tall = 16 total
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            deblocking_filter_control_present_flag=False,
        )

    def test_two_slices_horizontal_split(self):
        """Frame split horizontally into two slices."""
        from slice.slice_data import parse_slice_data, SliceData
        from slice.frame_assembly import FrameSliceAssembler

        sps = self._create_default_sps()  # 4x4 = 16 MBs
        pps = self._create_default_pps()

        assembler = FrameSliceAssembler(sps, pps)

        # Slice 0: MBs 0-7 (top two rows)
        header0 = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
        )
        # Slice 1: MBs 8-15 (bottom two rows)
        header1 = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=8,
        )

        # Add slices to assembler
        assembler.add_slice(header0, mb_count=8)
        assembler.add_slice(header1, mb_count=8)

        assert assembler.is_frame_complete()
        assert assembler.get_slice_count() == 2

    def test_four_slices_per_row(self):
        """Each row is a separate slice."""
        from slice.frame_assembly import FrameSliceAssembler

        sps = self._create_default_sps()  # 4x4 = 16 MBs
        pps = self._create_default_pps()

        assembler = FrameSliceAssembler(sps, pps)

        # One slice per row
        for row in range(4):
            header = SliceHeader(
                slice_type=SliceType.I,
                first_mb_in_slice=row * 4,
            )
            assembler.add_slice(header, mb_count=4)

        assert assembler.is_frame_complete()
        assert assembler.get_slice_count() == 4

    def test_slices_out_of_order(self):
        """Slices arriving out of first_mb_in_slice order (ASO)."""
        from slice.frame_assembly import FrameSliceAssembler

        sps = self._create_default_sps()
        pps = self._create_default_pps()

        assembler = FrameSliceAssembler(sps, pps)

        # Slices arrive out of order: 8, 0, 12, 4
        headers = [
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=8),
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=0),
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=12),
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=4),
        ]

        for header in headers:
            assembler.add_slice(header, mb_count=4)

        assert assembler.is_frame_complete()
        # Should have detected ASO
        assert assembler.uses_arbitrary_slice_order()

    def test_overlapping_slices_error(self):
        """Overlapping slices (without redundant_pic_cnt) should error."""
        from slice.frame_assembly import FrameSliceAssembler, SliceOverlapError

        sps = self._create_default_sps()
        pps = self._create_default_pps()

        assembler = FrameSliceAssembler(sps, pps)

        # First slice: MBs 0-7
        assembler.add_slice(
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=0),
            mb_count=8
        )

        # Second slice overlaps: MBs 4-11
        with pytest.raises(SliceOverlapError):
            assembler.add_slice(
                SliceHeader(slice_type=SliceType.I, first_mb_in_slice=4),
                mb_count=8
            )

    def test_incomplete_frame_missing_slice(self):
        """Frame with missing slice should report incomplete."""
        from slice.frame_assembly import FrameSliceAssembler

        sps = self._create_default_sps()
        pps = self._create_default_pps()

        assembler = FrameSliceAssembler(sps, pps)

        # Only add first and third slices
        assembler.add_slice(
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=0),
            mb_count=4
        )
        assembler.add_slice(
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=8),
            mb_count=4
        )

        assert not assembler.is_frame_complete()
        missing = assembler.get_missing_mb_ranges()
        assert (4, 7) in missing  # MBs 4-7 missing

    def test_slice_boundary_deblocking_flag(self):
        """Track slice boundaries for deblocking decisions."""
        from slice.frame_assembly import FrameSliceAssembler

        sps = self._create_default_sps()
        pps = self._create_default_pps()

        assembler = FrameSliceAssembler(sps, pps)

        assembler.add_slice(
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=0),
            mb_count=8
        )
        assembler.add_slice(
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=8),
            mb_count=8
        )

        # Check slice boundary detection
        assert assembler.is_slice_boundary(mb_a=7, mb_b=8)
        assert not assembler.is_slice_boundary(mb_a=0, mb_b=1)


class TestSliceTypeMixing:
    """Tests for mixing different slice types in a frame.

    H.264 Spec allows mixing slice types (e.g., I and P slices) in some profiles.
    """

    def _create_default_sps(self) -> SPS:
        return SPS(
            profile_idc=77,  # Main profile
            level_idc=30,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=3,
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
        )

    def test_i_and_p_slices_same_frame(self):
        """Mix of I and P slices in same frame."""
        from slice.frame_assembly import FrameSliceAssembler

        sps = self._create_default_sps()
        pps = self._create_default_pps()

        assembler = FrameSliceAssembler(sps, pps)

        # First slice is I
        assembler.add_slice(
            SliceHeader(slice_type=SliceType.I, first_mb_in_slice=0),
            mb_count=8
        )
        # Second slice is P
        assembler.add_slice(
            SliceHeader(slice_type=SliceType.P, first_mb_in_slice=8),
            mb_count=8
        )

        assert assembler.is_frame_complete()
        assert assembler.has_mixed_slice_types()

    def test_all_i_slices_is_idr_candidate(self):
        """Frame with only I slices could be IDR."""
        from slice.frame_assembly import FrameSliceAssembler

        sps = self._create_default_sps()
        pps = self._create_default_pps()

        assembler = FrameSliceAssembler(sps, pps)

        for row in range(4):
            assembler.add_slice(
                SliceHeader(slice_type=SliceType.I, first_mb_in_slice=row * 4),
                mb_count=4
            )

        assert assembler.all_slices_intra()


class TestSliceDataEdgeCases:
    """Edge case tests for slice data parsing."""

    def _create_default_sps(self) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=2,
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
        )

    def test_single_mb_slice(self):
        """Parse slice containing single macroblock."""
        from slice.slice_data import parse_slice_data

        sps = self._create_default_sps()
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=5,  # Single MB in middle
        )

        writer = NumpyBitWriter()
        writer.write_ue(0)  # mb_type = I_4x4
        # Stop bit
        writer.write_bit(1)
        writer.write_bits(0, 7)
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        assert len(slice_data.macroblocks) == 1
        assert slice_data.macroblocks[0].mb_addr == 5

    def test_last_mb_only_slice(self):
        """Parse slice containing only last MB in frame."""
        from slice.slice_data import parse_slice_data

        sps = self._create_default_sps()  # 4x3 = 12 MBs
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=11,  # Last MB
        )

        writer = NumpyBitWriter()
        writer.write_ue(0)  # mb_type
        writer.write_bit(1)  # Stop bit
        writer.write_bits(0, 7)
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        assert slice_data.macroblocks[0].mb_addr == 11
        assert slice_data.end_of_slice_detected

    def test_entire_frame_skipped_p_slice(self):
        """P-slice where all MBs are skipped."""
        from slice.slice_data import parse_slice_data

        sps = SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=1,  # 2x2 = 4 MBs
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=0,
        )

        writer = NumpyBitWriter()
        writer.write_ue(4)  # mb_skip_run = 4 (all MBs skipped)
        writer.write_bit(1)  # Stop bit
        writer.write_bits(0, 7)
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        assert len(slice_data.skipped_mb_addresses) == 4
        assert slice_data.skipped_mb_addresses == [0, 1, 2, 3]

    def test_mb_skip_run_exceeds_remaining_mbs(self):
        """Handle mb_skip_run that would exceed frame boundaries."""
        from slice.slice_data import parse_slice_data, SliceDataError

        sps = SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=1,  # 2x2 = 4 MBs
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )
        pps = self._create_default_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=2,  # Start at MB 2
        )

        writer = NumpyBitWriter()
        writer.write_ue(10)  # mb_skip_run = 10 (but only 2 MBs left)
        rbsp = writer.to_bytes()

        # Should handle gracefully or raise error
        with pytest.raises(SliceDataError):
            parse_slice_data(rbsp, header, sps, pps)

    def test_qp_delta_wrapping(self):
        """QP delta causing wrap around (52 wraps to 0)."""
        from slice.slice_data import parse_slice_data

        sps = self._create_default_sps()
        pps = PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            pic_init_qp_minus26=24,  # QP = 50
        )
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
            slice_qp_delta=0,  # Slice QP = 50
        )

        writer = NumpyBitWriter()
        writer.write_ue(0)  # mb_type
        # mb_qp_delta that would push QP > 51
        writer.write_se(5)  # QP would be 55, should wrap to 3
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data(rbsp, header, sps, pps)

        # QP = (50 + 5) % 52 = 3
        assert slice_data.macroblocks[0].qp == 3


class TestSliceGroupMapIntegration:
    """Tests for FMO slice group map integration with slice data."""

    def _create_fmo_sps(self) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=3,
            frame_mbs_only_flag=True,
        )

    def _create_fmo_pps(self, map_type: int = 0) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_slice_groups_minus1=1,  # 2 groups
            slice_group_map_type=map_type,
            run_length_minus1=[7, 7] if map_type == 0 else None,
        )

    def test_slice_respects_fmo_group_assignment(self):
        """Slice should only process MBs in its slice group."""
        from slice.slice_data import parse_slice_data_fmo

        sps = self._create_fmo_sps()  # 4x4 = 16 MBs
        pps = self._create_fmo_pps(map_type=0)  # Interleaved
        # With run_length=8, group 0 has MBs 0-7, group 1 has MBs 8-15

        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,  # Slice group 0
        )

        writer = NumpyBitWriter()
        for _ in range(8):
            writer.write_ue(0)  # mb_type
        rbsp = writer.to_bytes()

        slice_data = parse_slice_data_fmo(rbsp, header, sps, pps, slice_group_id=0)

        # Should only contain MBs from group 0
        assert len(slice_data.macroblocks) == 8
        assert all(mb.mb_addr < 8 for mb in slice_data.macroblocks)

    def test_next_mb_addr_skips_other_groups(self):
        """NextMbAddress should skip MBs in other slice groups."""
        from slice.slice_data import get_next_mb_addr_fmo

        sps = self._create_fmo_sps()
        pps = self._create_fmo_pps(map_type=0)

        # In group 0 (MBs 0-7), next after 7 should be end
        next_addr = get_next_mb_addr_fmo(
            current_addr=7,
            slice_group_id=0,
            sps=sps,
            pps=pps
        )

        # No more MBs in group 0
        assert next_addr is None or next_addr == -1


class TestCABACSliceDataParsing:
    """Tests for slice data parsing with CABAC entropy coding."""

    def _create_cabac_sps(self) -> SPS:
        return SPS(
            profile_idc=77,  # Main profile
            level_idc=30,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=2,
            frame_mbs_only_flag=True,
        )

    def _create_cabac_pps(self) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=True,  # CABAC
            cabac_init_idc=0,
        )

    def test_cabac_byte_alignment_before_data(self):
        """CABAC slice data must start at byte boundary."""
        from slice.slice_data import parse_cabac_slice_data

        sps = self._create_cabac_sps()
        pps = self._create_cabac_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
            header_bit_size=27,  # Not byte-aligned
        )

        # CABAC data should start at bit 32 (next byte boundary)
        writer = NumpyBitWriter()
        # Alignment bits (1 + zeros to boundary)
        writer.write_bit(1)
        writer.write_bits(0, 4)  # Pad to byte
        # ... CABAC encoded data ...
        rbsp = writer.to_bytes()

        slice_data = parse_cabac_slice_data(rbsp, header, sps, pps)

        assert slice_data.data_start_bit % 8 == 0

    def test_cabac_end_of_slice_flag(self):
        """CABAC uses end_of_slice_flag instead of RBSP trailing bits."""
        from slice.slice_data import parse_cabac_slice_data

        sps = self._create_cabac_sps()
        pps = self._create_cabac_pps()
        header = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
        )

        # Create CABAC bitstream with end_of_slice_flag
        writer = NumpyBitWriter()
        # ... CABAC encoded data ...
        # end_of_slice_flag = 1 is encoded last
        rbsp = writer.to_bytes()

        slice_data = parse_cabac_slice_data(rbsp, header, sps, pps)

        assert slice_data.end_of_slice_detected

    def test_cabac_mb_skip_flag(self):
        """CABAC uses mb_skip_flag instead of mb_skip_run."""
        from slice.slice_data import parse_cabac_slice_data

        sps = self._create_cabac_sps()
        pps = self._create_cabac_pps()
        header = SliceHeader(
            slice_type=SliceType.P,
            first_mb_in_slice=0,
        )

        # In CABAC, each MB has a skip flag
        # This is different from CAVLC's run-length encoding
        writer = NumpyBitWriter()
        # mb_skip_flag would be CABAC encoded here
        rbsp = writer.to_bytes()

        slice_data = parse_cabac_slice_data(rbsp, header, sps, pps)

        # Parser should handle individual skip flags
        assert hasattr(slice_data, 'skipped_mb_addresses')


class TestASOSliceDataParsing:
    """Tests for Arbitrary Slice Order slice data parsing."""

    def _create_default_sps(self) -> SPS:
        return SPS(
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=3,
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        return PPS(
            pic_parameter_set_id=0,
            entropy_coding_mode_flag=False,
        )

    def test_aso_detection_from_first_mb_order(self):
        """Detect ASO when slices arrive out of first_mb order."""
        from slice.aso import detect_aso_usage

        slice_order = [8, 0, 4, 12]  # Out of order

        assert detect_aso_usage(slice_order) is True

    def test_aso_in_order_detection(self):
        """No ASO when slices arrive in order."""
        from slice.aso import detect_aso_usage

        slice_order = [0, 4, 8, 12]  # In order

        assert detect_aso_usage(slice_order) is False

    def test_redundant_slice_handling(self):
        """Handle redundant slices (redundant_pic_cnt > 0)."""
        from slice.slice_data import parse_slice_data_with_redundant

        sps = self._create_default_sps()
        pps = self._create_default_pps()

        # Primary slice
        header_primary = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
            redundant_pic_cnt=0,
        )

        # Redundant slice for same region
        header_redundant = SliceHeader(
            slice_type=SliceType.I,
            first_mb_in_slice=0,
            redundant_pic_cnt=1,
        )

        writer = NumpyBitWriter()
        writer.write_ue(0)  # mb_type
        rbsp = writer.to_bytes()

        primary_data = parse_slice_data_with_redundant(
            rbsp, header_primary, sps, pps
        )
        redundant_data = parse_slice_data_with_redundant(
            rbsp, header_redundant, sps, pps
        )

        assert primary_data.is_primary()
        assert not redundant_data.is_primary()
        assert redundant_data.redundant_pic_cnt == 1
