# h264/parameters/tests/test_sps.py
"""Tests for SPS (Sequence Parameter Set) parsing."""

import pytest

from bitstream import BitWriter, BITSTRING_AVAILABLE
from parameters.sps import (
    SPS,
    VUIParameters,
    parse_sps,
    PROFILE_BASELINE,
    PROFILE_MAIN,
    PROFILE_HIGH,
)

# Skip all tests if bitstring not available
pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestSPSDataclass:
    """Tests for SPS dataclass properties."""

    def test_profile_name_baseline(self):
        """Baseline profile name."""
        sps = SPS(profile_idc=66)
        assert sps.profile_name == "Baseline"

    def test_profile_name_main(self):
        """Main profile name."""
        sps = SPS(profile_idc=77)
        assert sps.profile_name == "Main"

    def test_profile_name_high(self):
        """High profile name."""
        sps = SPS(profile_idc=100)
        assert sps.profile_name == "High"

    def test_profile_name_unknown(self):
        """Unknown profile name."""
        sps = SPS(profile_idc=99)
        assert "Unknown" in sps.profile_name

    def test_level_string(self):
        """Level string formatting."""
        sps = SPS(level_idc=30)
        assert sps.level == "3.0"

        sps = SPS(level_idc=41)
        assert sps.level == "4.1"

    def test_max_frame_num(self):
        """Calculate max_frame_num from log2 value."""
        sps = SPS(log2_max_frame_num_minus4=0)  # 2^4 = 16
        assert sps.max_frame_num == 16

        sps = SPS(log2_max_frame_num_minus4=4)  # 2^8 = 256
        assert sps.max_frame_num == 256

    def test_max_pic_order_cnt_lsb(self):
        """Calculate max POC LSB for type 0."""
        sps = SPS(pic_order_cnt_type=0, log2_max_pic_order_cnt_lsb_minus4=0)
        assert sps.max_pic_order_cnt_lsb == 16

        sps = SPS(pic_order_cnt_type=0, log2_max_pic_order_cnt_lsb_minus4=4)
        assert sps.max_pic_order_cnt_lsb == 256

    def test_max_pic_order_cnt_lsb_non_type0(self):
        """max_pic_order_cnt_lsb is 0 for non-type-0."""
        sps = SPS(pic_order_cnt_type=2)
        assert sps.max_pic_order_cnt_lsb == 0

    def test_dimensions_in_mbs(self):
        """Calculate MB dimensions."""
        sps = SPS(pic_width_in_mbs_minus1=3, pic_height_in_map_units_minus1=2)
        assert sps.pic_width_in_mbs == 4
        assert sps.pic_height_in_map_units == 3

    def test_frame_height_frame_only(self):
        """Frame height for frame-only mode."""
        sps = SPS(
            pic_height_in_map_units_minus1=2,  # 3 map units
            frame_mbs_only_flag=True
        )
        assert sps.frame_height_in_mbs == 3

    def test_frame_height_interlaced(self):
        """Frame height for interlaced mode (field coding)."""
        sps = SPS(
            pic_height_in_map_units_minus1=2,  # 3 map units
            frame_mbs_only_flag=False
        )
        assert sps.frame_height_in_mbs == 6  # 2 * 3

    def test_pixel_dimensions(self):
        """Calculate pixel dimensions."""
        sps = SPS(
            pic_width_in_mbs_minus1=7,  # 8 MBs = 128 pixels
            pic_height_in_map_units_minus1=5,  # 6 MBs = 96 pixels
            frame_mbs_only_flag=True
        )
        assert sps.width == 128
        assert sps.height == 96

    def test_cropped_dimensions(self):
        """Calculate cropped dimensions."""
        sps = SPS(
            pic_width_in_mbs_minus1=7,  # 128 pixels
            pic_height_in_map_units_minus1=5,  # 96 pixels
            frame_mbs_only_flag=True,
            chroma_format_idc=1,  # 4:2:0
            frame_cropping_flag=True,
            frame_crop_left_offset=0,
            frame_crop_right_offset=2,  # 4 pixels off right
            frame_crop_top_offset=0,
            frame_crop_bottom_offset=4,  # 8 pixels off bottom
        )
        assert sps.cropped_width == 124
        assert sps.cropped_height == 88

    def test_chroma_format_name(self):
        """Chroma format name lookup."""
        assert SPS(chroma_format_idc=0).chroma_format_name == "Monochrome"
        assert SPS(chroma_format_idc=1).chroma_format_name == "4:2:0"
        assert SPS(chroma_format_idc=2).chroma_format_name == "4:2:2"
        assert SPS(chroma_format_idc=3).chroma_format_name == "4:4:4"

    def test_bit_depth(self):
        """Bit depth calculation."""
        sps = SPS(bit_depth_luma_minus8=0, bit_depth_chroma_minus8=0)
        assert sps.bit_depth_luma == 8
        assert sps.bit_depth_chroma == 8

        sps = SPS(bit_depth_luma_minus8=2, bit_depth_chroma_minus8=2)
        assert sps.bit_depth_luma == 10
        assert sps.bit_depth_chroma == 10

    def test_repr(self):
        """String representation."""
        sps = SPS(
            seq_parameter_set_id=0,
            profile_idc=66,
            level_idc=30,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=2,
            frame_mbs_only_flag=True,
            chroma_format_idc=1,
        )
        s = repr(sps)
        assert "id=0" in s
        assert "Baseline" in s
        assert "3.0" in s
        assert "64x48" in s  # 4*16 x 3*16
        assert "4:2:0" in s


class TestVUIParameters:
    """Tests for VUI parameters."""

    def test_frame_rate_calculation(self):
        """Calculate frame rate from timing info."""
        vui = VUIParameters(
            timing_info_present_flag=True,
            num_units_in_tick=1001,
            time_scale=60000,
        )
        # 60000 / (2 * 1001) = 29.97 fps
        assert abs(vui.frame_rate - 29.97) < 0.01

    def test_frame_rate_no_timing(self):
        """Frame rate is None without timing info."""
        vui = VUIParameters(timing_info_present_flag=False)
        assert vui.frame_rate is None


class TestParseSPS:
    """Tests for SPS parsing."""

    def _create_baseline_sps_rbsp(
        self,
        width_mbs=4,
        height_mbs=3,
        profile=66,
        level=30,
        sps_id=0,
    ) -> bytes:
        """Create a minimal Baseline profile SPS RBSP for testing."""
        writer = BitWriter()

        # Profile and level
        writer.write_bits(profile, 8)  # profile_idc
        writer.write_flag(False)  # constraint_set0_flag
        writer.write_flag(False)  # constraint_set1_flag
        writer.write_flag(False)  # constraint_set2_flag
        writer.write_flag(False)  # constraint_set3_flag
        writer.write_flag(False)  # constraint_set4_flag
        writer.write_flag(False)  # constraint_set5_flag
        writer.write_bits(0, 2)  # reserved_zero_2bits
        writer.write_bits(level, 8)  # level_idc

        writer.write_ue(sps_id)  # seq_parameter_set_id

        # Frame numbering (no high profile extensions for baseline)
        writer.write_ue(0)  # log2_max_frame_num_minus4

        # POC type 2 (simplest - derived from frame_num)
        writer.write_ue(2)  # pic_order_cnt_type

        # Reference frames
        writer.write_ue(1)  # max_num_ref_frames
        writer.write_flag(False)  # gaps_in_frame_num_value_allowed_flag

        # Dimensions
        writer.write_ue(width_mbs - 1)  # pic_width_in_mbs_minus1
        writer.write_ue(height_mbs - 1)  # pic_height_in_map_units_minus1

        # Coding flags
        writer.write_flag(True)  # frame_mbs_only_flag
        writer.write_flag(False)  # direct_8x8_inference_flag

        # No cropping
        writer.write_flag(False)  # frame_cropping_flag

        # No VUI
        writer.write_flag(False)  # vui_parameters_present_flag

        return writer.to_bytes()

    def test_parse_minimal_baseline_sps(self):
        """Parse minimal Baseline profile SPS."""
        rbsp = self._create_baseline_sps_rbsp()
        sps = parse_sps(rbsp)

        assert sps.profile_idc == PROFILE_BASELINE
        assert sps.level_idc == 30
        assert sps.seq_parameter_set_id == 0
        assert sps.pic_width_in_mbs == 4
        assert sps.frame_height_in_mbs == 3
        assert sps.width == 64
        assert sps.height == 48
        assert sps.frame_mbs_only_flag is True

    def test_parse_different_dimensions(self):
        """Parse SPS with different dimensions."""
        rbsp = self._create_baseline_sps_rbsp(width_mbs=8, height_mbs=6)
        sps = parse_sps(rbsp)

        assert sps.pic_width_in_mbs == 8
        assert sps.frame_height_in_mbs == 6
        assert sps.width == 128
        assert sps.height == 96

    def test_parse_hd_dimensions(self):
        """Parse SPS with HD dimensions (1920x1080)."""
        # 1920/16=120 MBs, 1088/16=68 MBs (with cropping for 1080)
        rbsp = self._create_baseline_sps_rbsp(width_mbs=120, height_mbs=68)
        sps = parse_sps(rbsp)

        assert sps.width == 1920
        assert sps.height == 1088

    def test_parse_different_sps_id(self):
        """Parse SPS with different ID."""
        rbsp = self._create_baseline_sps_rbsp(sps_id=5)
        sps = parse_sps(rbsp)

        assert sps.seq_parameter_set_id == 5

    def test_parse_different_level(self):
        """Parse SPS with different level."""
        rbsp = self._create_baseline_sps_rbsp(level=41)
        sps = parse_sps(rbsp)

        assert sps.level_idc == 41
        assert sps.level == "4.1"

    def test_parse_with_cropping(self):
        """Parse SPS with frame cropping."""
        writer = BitWriter()

        # Basic header
        writer.write_bits(66, 8)  # profile_idc (Baseline)
        writer.write_bits(0, 6)  # constraint flags
        writer.write_bits(0, 2)  # reserved
        writer.write_bits(30, 8)  # level_idc

        writer.write_ue(0)  # sps_id
        writer.write_ue(0)  # log2_max_frame_num_minus4
        writer.write_ue(2)  # pic_order_cnt_type
        writer.write_ue(1)  # max_num_ref_frames
        writer.write_flag(False)  # gaps

        writer.write_ue(7)  # 8 MBs width
        writer.write_ue(5)  # 6 MBs height

        writer.write_flag(True)  # frame_mbs_only
        writer.write_flag(False)  # direct_8x8

        # Cropping enabled
        writer.write_flag(True)  # frame_cropping_flag
        writer.write_ue(0)  # left
        writer.write_ue(0)  # right
        writer.write_ue(0)  # top
        writer.write_ue(4)  # bottom (8 pixels for 4:2:0)

        writer.write_flag(False)  # vui

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        assert sps.frame_cropping_flag is True
        assert sps.frame_crop_bottom_offset == 4
        assert sps.height == 96  # 6 * 16
        assert sps.cropped_height == 88  # 96 - 2*4

    def test_parse_with_poc_type_0(self):
        """Parse SPS with POC type 0."""
        writer = BitWriter()

        writer.write_bits(66, 8)
        writer.write_bits(0, 6)
        writer.write_bits(0, 2)
        writer.write_bits(30, 8)

        writer.write_ue(0)  # sps_id
        writer.write_ue(4)  # log2_max_frame_num_minus4 = 4 -> max=256

        # POC type 0
        writer.write_ue(0)  # pic_order_cnt_type
        writer.write_ue(4)  # log2_max_pic_order_cnt_lsb_minus4 = 4

        writer.write_ue(1)  # max_num_ref_frames
        writer.write_flag(False)  # gaps

        writer.write_ue(3)  # width
        writer.write_ue(2)  # height

        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        assert sps.pic_order_cnt_type == 0
        assert sps.log2_max_pic_order_cnt_lsb_minus4 == 4
        assert sps.max_pic_order_cnt_lsb == 256

    def test_parse_too_short_raises(self):
        """Parsing too-short RBSP raises error."""
        with pytest.raises(ValueError):
            parse_sps(b'\x00\x00')


class TestRealWorldSPS:
    """Test with real-world SPS patterns."""

    def test_typical_720p_sps(self):
        """Parse typical 720p Baseline SPS."""
        writer = BitWriter()

        # Baseline, level 3.1
        writer.write_bits(66, 8)
        writer.write_bits(0, 6)
        writer.write_bits(0, 2)
        writer.write_bits(31, 8)

        writer.write_ue(0)  # sps_id
        writer.write_ue(0)  # log2_max_frame_num_minus4
        writer.write_ue(2)  # poc type 2
        writer.write_ue(3)  # max 3 ref frames
        writer.write_flag(False)

        # 1280x720 = 80x45 MBs
        writer.write_ue(79)  # width - 1
        writer.write_ue(44)  # height - 1

        writer.write_flag(True)  # frame only
        writer.write_flag(True)  # direct_8x8
        writer.write_flag(False)  # no crop
        writer.write_flag(False)  # no vui

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        assert sps.width == 1280
        assert sps.height == 720
        assert sps.profile_name == "Baseline"
        assert sps.level == "3.1"

    def test_typical_1080p_sps_with_cropping(self):
        """Parse typical 1080p SPS with cropping."""
        writer = BitWriter()

        # Main profile, level 4.0
        writer.write_bits(77, 8)
        writer.write_bits(0, 6)
        writer.write_bits(0, 2)
        writer.write_bits(40, 8)

        writer.write_ue(0)  # sps_id
        writer.write_ue(0)  # log2_max_frame_num_minus4
        writer.write_ue(0)  # poc type 0
        writer.write_ue(4)  # log2_max_poc_lsb_minus4
        writer.write_ue(4)  # max 4 ref frames
        writer.write_flag(False)

        # 1920x1088 = 120x68 MBs (crop to 1080)
        writer.write_ue(119)  # width - 1
        writer.write_ue(67)  # height - 1

        writer.write_flag(True)  # frame only
        writer.write_flag(True)  # direct_8x8

        # Cropping for 1080 (1088 - 1080 = 8 pixels = 4 crop units for 4:2:0)
        writer.write_flag(True)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_ue(4)  # 8 pixels off bottom

        writer.write_flag(False)  # no vui

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        assert sps.width == 1920
        assert sps.height == 1088
        assert sps.cropped_width == 1920
        assert sps.cropped_height == 1080
        assert sps.profile_name == "Main"
        assert sps.level == "4.0"
