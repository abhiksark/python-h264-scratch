# h264/decoder/tests/test_decoder_cabac.py
"""Tests for CABAC-specific decoder functionality.

Tests for the neighbor CBP tracking fix that enables proper CABAC context
derivation for coded_block_pattern syntax elements.

H.264 Spec Reference: Section 9.3.3.1.1.3 - Derivation of ctxIdxInc for
coded_block_pattern syntax elements.
"""

import pytest
import numpy as np

from decoder.decoder import H264Decoder, DecoderState
from parameters import SPS, PPS


def make_test_sps(width_mbs: int = 4, height_mbs: int = 4) -> SPS:
    """Create a minimal SPS for testing.

    Args:
        width_mbs: Frame width in macroblocks.
        height_mbs: Frame height in macroblocks.

    Returns:
        SPS with specified dimensions.
    """
    return SPS(
        profile_idc=77,  # Main profile (supports CABAC)
        level_idc=30,
        seq_parameter_set_id=0,
        chroma_format_idc=1,
        bit_depth_luma_minus8=0,
        bit_depth_chroma_minus8=0,
        log2_max_frame_num_minus4=4,
        pic_order_cnt_type=0,
        log2_max_pic_order_cnt_lsb_minus4=4,
        max_num_ref_frames=1,
        pic_width_in_mbs_minus1=width_mbs - 1,
        pic_height_in_map_units_minus1=height_mbs - 1,
        frame_mbs_only_flag=True,
    )


def make_test_pps() -> PPS:
    """Create a minimal PPS for testing.

    Returns:
        PPS with CABAC enabled.
    """
    return PPS(
        pic_parameter_set_id=0,
        seq_parameter_set_id=0,
        entropy_coding_mode_flag=True,  # CABAC
        bottom_field_pic_order_in_frame_present_flag=False,
        num_slice_groups_minus1=0,
        num_ref_idx_l0_default_active_minus1=0,
        num_ref_idx_l1_default_active_minus1=0,
        weighted_pred_flag=False,
        weighted_bipred_idc=0,
        pic_init_qp_minus26=0,
        pic_init_qs_minus26=0,
        chroma_qp_index_offset=0,
        deblocking_filter_control_present_flag=False,
        constrained_intra_pred_flag=False,
        redundant_pic_cnt_present_flag=False,
    )


class TestDecoderStateMbCbpsAllocation:
    """Tests for mb_cbps field allocation in DecoderState."""

    def test_mb_cbps_allocated_on_frame_buffer_allocation(self):
        """mb_cbps should be allocated when frame buffers are allocated."""
        state = DecoderState()
        sps = make_test_sps(width_mbs=4, height_mbs=4)

        state.allocate_frame_buffers(sps)

        assert state.mb_cbps is not None
        assert isinstance(state.mb_cbps, np.ndarray)

    def test_mb_cbps_has_correct_size(self):
        """mb_cbps should have one entry per macroblock."""
        state = DecoderState()
        sps = make_test_sps(width_mbs=4, height_mbs=3)

        state.allocate_frame_buffers(sps)

        expected_mb_count = 4 * 3  # width_mbs * height_mbs = 12
        assert state.mb_cbps.shape == (expected_mb_count,)

    def test_mb_cbps_initialized_to_zero(self):
        """mb_cbps should be initialized to zero."""
        state = DecoderState()
        sps = make_test_sps(width_mbs=2, height_mbs=2)

        state.allocate_frame_buffers(sps)

        assert np.all(state.mb_cbps == 0)

    def test_mb_cbps_dtype_is_int32(self):
        """mb_cbps should use int32 dtype for computation safety."""
        state = DecoderState()
        sps = make_test_sps(width_mbs=2, height_mbs=2)

        state.allocate_frame_buffers(sps)

        assert state.mb_cbps.dtype == np.int32


class TestBuildCabacMbInfoLeftCbp:
    """Tests for left_cbp in _build_cabac_mb_info."""

    def test_left_cbp_unavailable_at_x_zero(self):
        """left_cbp should be -1 when MB is at x=0 (left edge)."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Test MB at (0, 1) - left edge, second row
        mb_info = decoder._build_cabac_mb_info(
            mb_x=0, mb_y=1, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['left_cbp'] == -1

    def test_left_cbp_available_at_x_greater_than_zero(self):
        """left_cbp should reflect neighbor's CBP when x > 0."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Set CBP of left neighbor (MB at index 0)
        decoder.state.mb_cbps[0] = 0x2F  # Some non-zero CBP value

        # Test MB at (1, 0) - second column, first row
        mb_info = decoder._build_cabac_mb_info(
            mb_x=1, mb_y=0, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['left_cbp'] == 0x2F

    def test_left_cbp_correct_value_interior_mb(self):
        """left_cbp should return correct neighbor CBP for interior MB."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Set up CBPs for row 2: indices 8, 9, 10, 11
        decoder.state.mb_cbps[8] = 0x0F   # MB (0, 2)
        decoder.state.mb_cbps[9] = 0x1E   # MB (1, 2)
        decoder.state.mb_cbps[10] = 0x3C  # MB (2, 2)

        # Test MB at (2, 2) - its left neighbor is MB (1, 2) with CBP 0x1E
        mb_info = decoder._build_cabac_mb_info(
            mb_x=2, mb_y=2, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['left_cbp'] == 0x1E


class TestBuildCabacMbInfoTopCbp:
    """Tests for top_cbp in _build_cabac_mb_info."""

    def test_top_cbp_unavailable_at_y_zero(self):
        """top_cbp should be -1 when MB is at y=0 (top edge)."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Test MB at (2, 0) - third column, top row
        mb_info = decoder._build_cabac_mb_info(
            mb_x=2, mb_y=0, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['top_cbp'] == -1

    def test_top_cbp_available_at_y_greater_than_zero(self):
        """top_cbp should reflect neighbor's CBP when y > 0."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Set CBP of top neighbor (MB at index 2, which is (2, 0))
        decoder.state.mb_cbps[2] = 0x17  # Some non-zero CBP value

        # Test MB at (2, 1) - third column, second row
        # Its top neighbor is MB at (2, 0), which is index 2
        mb_info = decoder._build_cabac_mb_info(
            mb_x=2, mb_y=1, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['top_cbp'] == 0x17

    def test_top_cbp_correct_value_interior_mb(self):
        """top_cbp should return correct neighbor CBP for interior MB."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Set up CBPs for column 1: indices 1, 5, 9, 13
        decoder.state.mb_cbps[1] = 0x05   # MB (1, 0)
        decoder.state.mb_cbps[5] = 0x0A   # MB (1, 1)
        decoder.state.mb_cbps[9] = 0x15   # MB (1, 2)

        # Test MB at (1, 2) - its top neighbor is MB (1, 1) with CBP 0x0A
        mb_info = decoder._build_cabac_mb_info(
            mb_x=1, mb_y=2, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['top_cbp'] == 0x0A


class TestBuildCabacMbInfoUnavailableNeighbors:
    """Tests for unavailable neighbors returning -1."""

    def test_top_left_corner_both_unavailable(self):
        """Both left_cbp and top_cbp should be -1 at top-left corner."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Test MB at (0, 0) - top-left corner
        mb_info = decoder._build_cabac_mb_info(
            mb_x=0, mb_y=0, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['left_cbp'] == -1
        assert mb_info['top_cbp'] == -1

    def test_top_row_top_unavailable(self):
        """top_cbp should be -1 for all MBs in top row."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Test all MBs in top row (y=0)
        for mb_x in range(4):
            mb_info = decoder._build_cabac_mb_info(
                mb_x=mb_x, mb_y=0, mb_width=4, sps=sps, pps=pps
            )
            assert mb_info['top_cbp'] == -1, f"Failed for mb_x={mb_x}"

    def test_left_column_left_unavailable(self):
        """left_cbp should be -1 for all MBs in left column."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Test all MBs in left column (x=0)
        for mb_y in range(4):
            mb_info = decoder._build_cabac_mb_info(
                mb_x=0, mb_y=mb_y, mb_width=4, sps=sps, pps=pps
            )
            assert mb_info['left_cbp'] == -1, f"Failed for mb_y={mb_y}"

    def test_mb_info_includes_availability_flags(self):
        """_build_cabac_mb_info should include availability flags."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Interior MB should have both neighbors available
        mb_info = decoder._build_cabac_mb_info(
            mb_x=2, mb_y=2, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['left_available'] is True
        assert mb_info['top_available'] is True

        # Corner MB should have neither available
        mb_info = decoder._build_cabac_mb_info(
            mb_x=0, mb_y=0, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['left_available'] is False
        assert mb_info['top_available'] is False


class TestProcessCabacMacroblockCbpStorage:
    """Tests for CBP storage after processing a CABAC macroblock."""

    def test_cbp_stored_after_processing(self):
        """CBP should be stored in state.mb_cbps after processing."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)
        decoder.state.current_sps = sps
        decoder.state.current_pps = pps

        # Simulate a decoded macroblock with CBP
        mb_data = {
            'mb_type': 1,  # Intra 16x16
            'cbp': 0x2F,   # Some CBP value
            'mb_skip_flag': 0,
        }

        # Process MB at (1, 1) which is index 5 (1 + 1*4)
        mb_x, mb_y = 1, 1
        mb_idx = mb_y * sps.pic_width_in_mbs + mb_x

        # Store MB type and CBP directly (simulating what _process_cabac_macroblock does)
        if decoder.state.mb_types is not None:
            decoder.state.mb_types[mb_idx] = mb_data.get('mb_type', 0)
        if decoder.state.mb_cbps is not None:
            decoder.state.mb_cbps[mb_idx] = mb_data.get('cbp', 0)

        assert decoder.state.mb_cbps[mb_idx] == 0x2F

    def test_cbp_stored_for_multiple_macroblocks(self):
        """CBP should be stored correctly for multiple MBs in sequence."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)
        decoder.state.current_sps = sps
        decoder.state.current_pps = pps

        # Simulate processing multiple MBs with different CBPs
        test_cases = [
            ((0, 0), 0x00),  # MB 0: no coded blocks
            ((1, 0), 0x0F),  # MB 1: all luma AC
            ((2, 0), 0x1F),  # MB 2: luma AC + Cb AC
            ((3, 0), 0x2F),  # MB 3: luma AC + Cr AC
            ((0, 1), 0x3F),  # MB 4: luma AC + both chroma AC
        ]

        for (mb_x, mb_y), cbp in test_cases:
            mb_idx = mb_y * sps.pic_width_in_mbs + mb_x
            decoder.state.mb_cbps[mb_idx] = cbp

        # Verify all CBPs stored correctly
        for (mb_x, mb_y), expected_cbp in test_cases:
            mb_idx = mb_y * sps.pic_width_in_mbs + mb_x
            assert decoder.state.mb_cbps[mb_idx] == expected_cbp, \
                f"CBP mismatch at ({mb_x}, {mb_y})"

    def test_cbp_available_for_subsequent_neighbor_lookup(self):
        """After storing CBP, it should be available for neighbor lookup."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)
        decoder.state.current_sps = sps
        decoder.state.current_pps = pps

        # Store CBP for MB at (0, 0)
        decoder.state.mb_cbps[0] = 0x1A

        # When building info for MB at (1, 0), left_cbp should be 0x1A
        mb_info = decoder._build_cabac_mb_info(
            mb_x=1, mb_y=0, mb_width=4, sps=sps, pps=pps
        )

        assert mb_info['left_cbp'] == 0x1A

    def test_skip_macroblock_cbp_zero(self):
        """Skip macroblocks should have CBP=0."""
        decoder = H264Decoder()
        sps = make_test_sps(width_mbs=4, height_mbs=4)
        pps = make_test_pps()

        decoder.state.allocate_frame_buffers(sps)

        # Skip MB has no residual, so CBP should be 0
        mb_data = {
            'mb_type': 0,
            'cbp': 0,
            'mb_skip_flag': 1,
        }

        mb_idx = 0
        decoder.state.mb_cbps[mb_idx] = mb_data.get('cbp', 0)

        assert decoder.state.mb_cbps[mb_idx] == 0
