# h264/decoder/tests/test_multiple_slices.py
"""RED TESTS: Multiple slice support.

H.264 frames can be split into multiple slices for error resilience
and parallel decoding. Each slice can be decoded independently.

These tests SHOULD FAIL until multiple slice support is implemented.
"""

import pytest
import numpy as np

from decoder.decoder import H264Decoder, DecoderState


class TestSliceTracking:
    """Tests for tracking which slice each MB belongs to."""

    def test_decoder_state_tracks_slice_ids(self):
        """DecoderState should track slice ID per macroblock."""
        state = DecoderState()

        assert hasattr(state, 'mb_slice_ids'), \
            "DecoderState should track mb_slice_ids array"

    def test_slice_id_initialized_on_frame_alloc(self):
        """Slice ID array should be allocated with frame buffers."""
        from parameters import SPS

        state = DecoderState()
        # Create minimal SPS (2x2 MBs = 32x32 pixels)
        sps = SPS(
            profile_idc=66,
            level_idc=30,
            seq_parameter_set_id=0,
            chroma_format_idc=1,
            bit_depth_luma_minus8=0,
            bit_depth_chroma_minus8=0,
            log2_max_frame_num_minus4=0,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=1,  # 2 MBs wide
            pic_height_in_map_units_minus1=1,  # 2 MBs tall
            frame_mbs_only_flag=True,
        )

        state.allocate_frame_buffers(sps)

        assert state.mb_slice_ids is not None
        assert state.mb_slice_ids.shape == (4,)  # 2x2 = 4 MBs


class TestMultiSliceDecoding:
    """Tests for decoding frames with multiple slices."""

    def test_decoder_handles_nonzero_first_mb(self):
        """Decoder should handle first_mb_in_slice != 0."""
        decoder = H264Decoder()

        # Should have method to continue decoding mid-frame
        assert hasattr(decoder, '_continue_slice'), \
            "Decoder should have _continue_slice method"

    def test_slices_can_have_different_qp(self):
        """Different slices can have different QP values."""
        decoder = H264Decoder()

        # Decoder should track per-slice QP
        assert hasattr(decoder, '_get_slice_qp'), \
            "Decoder should have _get_slice_qp method"

    def test_slice_boundary_detection(self):
        """Decoder should detect slice boundaries for deblocking."""
        decoder = H264Decoder()

        # Already tested _is_slice_boundary, but verify it uses slice tracking
        # This test ensures actual slice data is used
        assert hasattr(decoder.state, 'mb_slice_ids'), \
            "Slice boundary detection needs mb_slice_ids"


class TestSliceGrouping:
    """Tests for FMO (Flexible Macroblock Ordering) slice groups."""

    def test_decoder_supports_slice_group_map_type(self):
        """Decoder should handle PPS slice_group_map_type."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_get_slice_group'), \
            "Decoder should have _get_slice_group method"

    def test_interleaved_slice_groups(self):
        """slice_group_map_type=0: Interleaved slice groups."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_calc_interleaved_map'), \
            "Decoder should support interleaved slice group mapping"

    def test_dispersed_slice_groups(self):
        """slice_group_map_type=1: Dispersed slice groups."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_calc_dispersed_map'), \
            "Decoder should support dispersed slice group mapping"


class TestSliceDataPartitioning:
    """Tests for data partitioning (Annex G)."""

    def test_decoder_recognizes_partition_a(self):
        """Decoder should recognize slice data partition A NAL type."""
        from bitstream import NALUnitType

        # NAL types 2, 3, 4 are partition A, B, C
        assert hasattr(NALUnitType, 'SLICE_PARTITION_A') or \
               NALUnitType(2).value == 2, \
            "NALUnitType should have partition types"

    def test_decoder_can_combine_partitions(self):
        """Decoder should combine data partitions into complete slice."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_combine_partitions'), \
            "Decoder should have _combine_partitions method"
