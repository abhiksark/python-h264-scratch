# h264/decoder/tests/test_decoder.py
"""Tests for H.264 decoder."""

import pytest
import numpy as np

from bitstream import BitWriter, BitReader, BITSTRING_AVAILABLE
from decoder import (
    H264Decoder,
    DecodedFrame,
    DecoderState,
    decode_h264_bytes,
)
from color import ColorMatrix
from parameters import SPS, PPS

# Skip all tests if bitstring not available
pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestDecodedFrame:
    """Tests for DecodedFrame dataclass."""

    def test_basic_frame(self):
        """Create basic decoded frame."""
        luma = np.full((16, 16), 128, dtype=np.uint8)
        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        frame = DecodedFrame(
            frame_num=0,
            poc=0,
            luma=luma,
            cb=cb,
            cr=cr,
            width=16,
            height=16,
        )

        assert frame.frame_num == 0
        assert frame.shape == (16, 16)
        assert frame.luma.shape == (16, 16)
        assert frame.cb.shape == (8, 8)
        assert frame.cr.shape == (8, 8)

    def test_to_rgb_conversion(self):
        """Test YUV to RGB conversion."""
        # Gray frame (Y=128, Cb=Cr=128)
        luma = np.full((16, 16), 128, dtype=np.uint8)
        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        frame = DecodedFrame(
            frame_num=0,
            poc=0,
            luma=luma,
            cb=cb,
            cr=cr,
            width=16,
            height=16,
        )

        rgb = frame.to_rgb()
        assert rgb.shape == (16, 16, 3)
        assert rgb.dtype == np.uint8
        # Gray should produce roughly equal RGB values
        assert np.allclose(rgb[:, :, 0], rgb[:, :, 1], atol=2)
        assert np.allclose(rgb[:, :, 1], rgb[:, :, 2], atol=2)

    def test_to_rgb_bt709(self):
        """Test RGB conversion with BT.709 standard."""
        luma = np.full((16, 16), 128, dtype=np.uint8)
        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        frame = DecodedFrame(
            frame_num=0,
            poc=0,
            luma=luma,
            cb=cb,
            cr=cr,
            width=16,
            height=16,
        )

        rgb = frame.to_rgb(color_matrix=ColorMatrix.BT709)
        assert rgb.shape == (16, 16, 3)


class TestDecoderState:
    """Tests for DecoderState management."""

    def test_empty_state(self):
        """Initial state is empty."""
        state = DecoderState()
        assert len(state.sps_dict) == 0
        assert len(state.pps_dict) == 0
        assert state.current_sps is None
        assert state.current_pps is None

    def test_sps_storage(self):
        """SPS can be stored and retrieved."""
        state = DecoderState()

        # Create minimal SPS
        sps = SPS(
            profile_idc=66,
            level_idc=30,
            seq_parameter_set_id=0,
            chroma_format_idc=1,
            bit_depth_luma_minus8=0,
            bit_depth_chroma_minus8=0,
            log2_max_frame_num_minus4=4,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=3,  # 4 MBs = 64 pixels
            pic_height_in_map_units_minus1=3,  # 4 MBs = 64 pixels
            frame_mbs_only_flag=True,
        )

        state.sps_dict[0] = sps
        retrieved = state.get_sps(0)
        assert retrieved.profile_idc == 66
        assert retrieved.pic_width_in_mbs == 4

    def test_sps_not_found(self):
        """Missing SPS raises error."""
        state = DecoderState()
        with pytest.raises(ValueError, match="SPS 0 not found"):
            state.get_sps(0)

    def test_pps_storage(self):
        """PPS can be stored and retrieved."""
        state = DecoderState()

        pps = PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
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

        state.pps_dict[0] = pps
        retrieved = state.get_pps(0)
        assert retrieved.entropy_coding_mode_flag is False

    def test_pps_not_found(self):
        """Missing PPS raises error."""
        state = DecoderState()
        with pytest.raises(ValueError, match="PPS 0 not found"):
            state.get_pps(0)

    def test_allocate_frame_buffers(self):
        """Frame buffers allocated correctly."""
        state = DecoderState()

        sps = SPS(
            profile_idc=66,
            level_idc=30,
            seq_parameter_set_id=0,
            chroma_format_idc=1,
            bit_depth_luma_minus8=0,
            bit_depth_chroma_minus8=0,
            log2_max_frame_num_minus4=4,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=3,  # 4 MBs = 64 pixels
            pic_height_in_map_units_minus1=3,  # 4 map units = 64 pixels (frame_mbs_only)
            frame_mbs_only_flag=True,
        )

        state.allocate_frame_buffers(sps)

        # pic_width_in_mbs=4, frame_height_in_mbs=4 (since frame_mbs_only)
        assert state.frame_luma.shape == (64, 64)
        assert state.frame_cb.shape == (32, 32)
        assert state.frame_cr.shape == (32, 32)
        assert state.nz_counts.shape == (16, 24)  # 16 MBs, 24 blocks each


class TestH264Decoder:
    """Tests for H264Decoder class."""

    def test_decoder_initialization(self):
        """Decoder initializes correctly."""
        decoder = H264Decoder()
        assert decoder.state is not None
        assert len(decoder.state.sps_dict) == 0
        assert len(decoder.state.pps_dict) == 0

    def test_empty_input(self):
        """Empty input yields no frames."""
        decoder = H264Decoder()
        frames = list(decoder.decode_bytes(b""))
        assert len(frames) == 0

    def test_invalid_data(self):
        """Invalid data yields no frames."""
        decoder = H264Decoder()
        frames = list(decoder.decode_bytes(b"\x00\x00\x01\xff"))
        assert len(frames) == 0


class TestNeighborCalculation:
    """Tests for neighbor calculation logic."""

    def test_top_left_corner(self):
        """Top-left MB has no neighbors."""
        decoder = H264Decoder()
        neighbors = decoder._get_mb_neighbors(0, 0, 4)

        assert neighbors["top_available"] is False
        assert neighbors["left_available"] is False
        assert neighbors["top_left_available"] is False

    def test_top_row(self):
        """Top row MBs have no top neighbor."""
        decoder = H264Decoder()
        neighbors = decoder._get_mb_neighbors(2, 0, 4)

        assert neighbors["top_available"] is False
        assert neighbors["left_available"] is True
        assert neighbors["top_left_available"] is False

    def test_left_column(self):
        """Left column MBs have no left neighbor."""
        decoder = H264Decoder()
        neighbors = decoder._get_mb_neighbors(0, 2, 4)

        assert neighbors["top_available"] is True
        assert neighbors["left_available"] is False
        assert neighbors["top_left_available"] is False

    def test_interior_mb(self):
        """Interior MBs have all neighbors."""
        decoder = H264Decoder()
        neighbors = decoder._get_mb_neighbors(2, 2, 4)

        assert neighbors["top_available"] is True
        assert neighbors["left_available"] is True
        assert neighbors["top_left_available"] is True

    def test_top_right_availability(self):
        """Top-right corner availability."""
        decoder = H264Decoder()

        # MB at (3, 1) - rightmost column, has no top-right
        neighbors = decoder._get_mb_neighbors(3, 1, 4)
        assert neighbors["top_right_available"] is False

        # MB at (2, 1) - not rightmost, has top-right
        neighbors = decoder._get_mb_neighbors(2, 1, 4)
        assert neighbors["top_right_available"] is True


class TestDecoderHelpers:
    """Tests for decoder helper functions."""

    def test_decode_h264_bytes_empty(self):
        """Helper with empty input returns empty list."""
        frames = decode_h264_bytes(b"")
        assert frames == []
