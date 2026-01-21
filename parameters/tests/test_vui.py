# h264/parameters/tests/test_vui.py
"""Tests for VUI (Video Usability Information) parsing (TDD - RED phase).

H.264 Spec Reference:
- Annex E.1.1: VUI parameters syntax
- Annex E.2: VUI parameters semantics
- Table E-1: Meaning of sample aspect ratio indicator

These tests are written TDD-style and should FAIL until the
VUI parsing features are properly implemented.
"""

import pytest

from bitstream import BitWriter, BITSTRING_AVAILABLE


pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


# H.264 Spec Table E-1: Sample aspect ratio indicators
SAR_EXTENDED = 255  # Extended_SAR: sar_width and sar_height present
SAR_SQUARE = 1      # 1:1 (square pixels)
SAR_12_11 = 2       # 12:11 (NTSC 4:3)
SAR_10_11 = 3       # 10:11 (NTSC 16:9)
SAR_16_11 = 4       # 16:11 (PAL 4:3)
SAR_40_33 = 5       # 40:33 (PAL 16:9)
SAR_24_11 = 6       # 24:11
SAR_20_11 = 7       # 20:11
SAR_32_11 = 8       # 32:11
SAR_80_33 = 9       # 80:33
SAR_18_11 = 10      # 18:11
SAR_15_11 = 11      # 15:11
SAR_64_33 = 12      # 64:33
SAR_160_99 = 13     # 160:99

# H.264 Spec Table E-2: Video format indicator
VIDEO_FORMAT_COMPONENT = 0
VIDEO_FORMAT_PAL = 1
VIDEO_FORMAT_NTSC = 2
VIDEO_FORMAT_SECAM = 3
VIDEO_FORMAT_MAC = 4
VIDEO_FORMAT_UNSPECIFIED = 5

# H.264 Spec Table E-3: Colour primaries
COLOUR_PRIMARIES_BT709 = 1
COLOUR_PRIMARIES_UNSPECIFIED = 2
COLOUR_PRIMARIES_BT470M = 4
COLOUR_PRIMARIES_BT470BG = 5
COLOUR_PRIMARIES_SMPTE170M = 6
COLOUR_PRIMARIES_SMPTE240M = 7
COLOUR_PRIMARIES_FILM = 8
COLOUR_PRIMARIES_BT2020 = 9

# H.264 Spec Table E-4: Transfer characteristics
TRANSFER_BT709 = 1
TRANSFER_UNSPECIFIED = 2
TRANSFER_BT470M = 4
TRANSFER_BT470BG = 5
TRANSFER_SMPTE170M = 6
TRANSFER_SMPTE240M = 7
TRANSFER_LINEAR = 8
TRANSFER_LOG_100 = 9
TRANSFER_LOG_316 = 10
TRANSFER_IEC61966 = 11
TRANSFER_BT1361 = 12
TRANSFER_SRGB = 13
TRANSFER_BT2020_10BIT = 14
TRANSFER_BT2020_12BIT = 15

# H.264 Spec Table E-5: Matrix coefficients
MATRIX_GBR = 0
MATRIX_BT709 = 1
MATRIX_UNSPECIFIED = 2
MATRIX_FCC = 4
MATRIX_BT470BG = 5
MATRIX_SMPTE170M = 6
MATRIX_SMPTE240M = 7
MATRIX_YCGCO = 8
MATRIX_BT2020_NCL = 9
MATRIX_BT2020_CL = 10


class TestVUIAspectRatioParsing:
    """Test VUI aspect_ratio_info parsing.

    H.264 Spec: Annex E.1.1, E.2.1
    """

    @pytest.mark.xfail(reason="VUI aspect_ratio parsing not fully implemented")
    def test_aspect_ratio_not_present(self):
        """When aspect_ratio_info_present_flag=0, SAR is unspecified."""
        from parameters.sps import VUIParameters, parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio_info_present_flag
        # Continue with remaining VUI fields (minimal)
        writer.write_flag(False)  # overscan_info_present_flag
        writer.write_flag(False)  # video_signal_type_present_flag
        writer.write_flag(False)  # chroma_loc_info_present_flag
        writer.write_flag(False)  # timing_info_present_flag
        writer.write_flag(False)  # nal_hrd_parameters_present_flag
        writer.write_flag(False)  # vcl_hrd_parameters_present_flag
        writer.write_flag(False)  # pic_struct_present_flag
        writer.write_flag(False)  # bitstream_restriction_flag

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.aspect_ratio_info_present_flag is False
        assert vui.aspect_ratio_idc == 0
        assert vui.sar_width == 0
        assert vui.sar_height == 0

    @pytest.mark.xfail(reason="VUI aspect_ratio parsing not fully implemented")
    def test_aspect_ratio_square_pixels(self):
        """Parse aspect_ratio_idc=1 (1:1 square pixels)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(True)  # aspect_ratio_info_present_flag
        writer.write_bits(SAR_SQUARE, 8)  # aspect_ratio_idc = 1
        writer.write_flag(False)  # overscan_info_present_flag
        writer.write_flag(False)  # video_signal_type_present_flag
        writer.write_flag(False)  # chroma_loc_info_present_flag
        writer.write_flag(False)  # timing_info_present_flag
        writer.write_flag(False)  # nal_hrd_parameters_present_flag
        writer.write_flag(False)  # vcl_hrd_parameters_present_flag
        writer.write_flag(False)  # pic_struct_present_flag
        writer.write_flag(False)  # bitstream_restriction_flag

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.aspect_ratio_info_present_flag is True
        assert vui.aspect_ratio_idc == SAR_SQUARE
        # For SAR_SQUARE, effective SAR is 1:1
        assert vui.sample_aspect_ratio == (1, 1)

    @pytest.mark.xfail(reason="VUI aspect_ratio parsing not fully implemented")
    def test_aspect_ratio_extended_sar(self):
        """Parse Extended_SAR (aspect_ratio_idc=255) with custom width/height."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(True)  # aspect_ratio_info_present_flag
        writer.write_bits(SAR_EXTENDED, 8)  # aspect_ratio_idc = 255
        writer.write_bits(64, 16)  # sar_width
        writer.write_bits(45, 16)  # sar_height
        writer.write_flag(False)  # overscan_info_present_flag
        writer.write_flag(False)  # video_signal_type_present_flag
        writer.write_flag(False)  # chroma_loc_info_present_flag
        writer.write_flag(False)  # timing_info_present_flag
        writer.write_flag(False)  # nal_hrd_parameters_present_flag
        writer.write_flag(False)  # vcl_hrd_parameters_present_flag
        writer.write_flag(False)  # pic_struct_present_flag
        writer.write_flag(False)  # bitstream_restriction_flag

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.aspect_ratio_idc == SAR_EXTENDED
        assert vui.sar_width == 64
        assert vui.sar_height == 45
        assert vui.sample_aspect_ratio == (64, 45)

    @pytest.mark.xfail(reason="VUI aspect_ratio parsing not fully implemented")
    def test_aspect_ratio_ntsc_4_3(self):
        """Parse aspect_ratio_idc=2 (12:11 NTSC 4:3)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(True)
        writer.write_bits(SAR_12_11, 8)  # 12:11
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.sample_aspect_ratio == (12, 11)

    @pytest.mark.xfail(reason="VUI aspect_ratio parsing not fully implemented")
    def test_aspect_ratio_pal_16_9(self):
        """Parse aspect_ratio_idc=5 (40:33 PAL 16:9)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(True)
        writer.write_bits(SAR_40_33, 8)  # 40:33
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.sample_aspect_ratio == (40, 33)


class TestVUITimingInfoParsing:
    """Test VUI timing_info parsing for frame rate detection.

    H.264 Spec: Annex E.2.1
    """

    @pytest.mark.xfail(reason="VUI timing_info parsing not fully implemented")
    def test_timing_info_not_present(self):
        """When timing_info_present_flag=0, frame rate is unknown."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing_info_present_flag
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.timing_info_present_flag is False
        assert vui.frame_rate is None

    @pytest.mark.xfail(reason="VUI timing_info parsing not fully implemented")
    def test_timing_info_29_97fps(self):
        """Parse timing_info for 29.97 fps (NTSC)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(True)   # timing_info_present_flag
        writer.write_bits(1001, 32)  # num_units_in_tick
        writer.write_bits(60000, 32)  # time_scale
        writer.write_flag(True)  # fixed_frame_rate_flag
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.timing_info_present_flag is True
        assert vui.num_units_in_tick == 1001
        assert vui.time_scale == 60000
        assert vui.fixed_frame_rate_flag is True
        # Frame rate = time_scale / (2 * num_units_in_tick) = 60000 / 2002 = 29.97
        assert abs(vui.frame_rate - 29.97) < 0.01

    @pytest.mark.xfail(reason="VUI timing_info parsing not fully implemented")
    def test_timing_info_25fps(self):
        """Parse timing_info for 25 fps (PAL)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(True)   # timing_info_present_flag
        writer.write_bits(1, 32)  # num_units_in_tick
        writer.write_bits(50, 32)  # time_scale
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        # Frame rate = 50 / (2 * 1) = 25 fps
        assert abs(vui.frame_rate - 25.0) < 0.001

    @pytest.mark.xfail(reason="VUI timing_info parsing not fully implemented")
    def test_timing_info_23_976fps(self):
        """Parse timing_info for 23.976 fps (film)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(True)
        writer.write_bits(1001, 32)  # num_units_in_tick
        writer.write_bits(48000, 32)  # time_scale
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        # Frame rate = 48000 / (2 * 1001) = 23.976
        assert abs(vui.frame_rate - 23.976) < 0.001

    @pytest.mark.xfail(reason="VUI timing_info parsing not fully implemented")
    def test_timing_info_60fps(self):
        """Parse timing_info for 60 fps."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(True)
        writer.write_bits(1, 32)
        writer.write_bits(120, 32)  # time_scale = 120
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        # Frame rate = 120 / (2 * 1) = 60 fps
        assert abs(vui.frame_rate - 60.0) < 0.001

    @pytest.mark.xfail(reason="VUI timing_info parsing not fully implemented")
    def test_timing_info_variable_frame_rate(self):
        """Variable frame rate (fixed_frame_rate_flag=0)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(True)
        writer.write_bits(1001, 32)
        writer.write_bits(60000, 32)
        writer.write_flag(False)  # fixed_frame_rate_flag = 0
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.fixed_frame_rate_flag is False
        # Frame rate is still calculable but may vary
        assert vui.frame_rate is not None


class TestVUIPicStructParsing:
    """Test VUI pic_struct_present_flag parsing.

    H.264 Spec: Annex D.2.2
    When pic_struct_present_flag=1, pic_struct is present in pic_timing SEI.
    """

    @pytest.mark.xfail(reason="VUI pic_struct parsing not fully implemented")
    def test_pic_struct_not_present(self):
        """pic_struct_present_flag=0 means no pic_struct in SEI."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct_present_flag
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.pic_struct_present_flag is False

    @pytest.mark.xfail(reason="VUI pic_struct parsing not fully implemented")
    def test_pic_struct_present(self):
        """pic_struct_present_flag=1 enables pic_struct in SEI."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(True)   # pic_struct_present_flag
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.pic_struct_present_flag is True


class TestVUIVideoSignalTypeParsing:
    """Test VUI video_signal_type parsing.

    H.264 Spec: Annex E.2.1
    """

    @pytest.mark.xfail(reason="VUI video_signal_type parsing not fully implemented")
    def test_video_signal_type_not_present(self):
        """When video_signal_type_present_flag=0, defaults apply."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal_type_present_flag
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.video_signal_type_present_flag is False
        assert vui.video_format == VIDEO_FORMAT_UNSPECIFIED
        assert vui.video_full_range_flag is False
        assert vui.colour_description_present_flag is False

    @pytest.mark.xfail(reason="VUI video_signal_type parsing not fully implemented")
    def test_video_signal_type_pal(self):
        """Parse video_signal_type with PAL format."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(True)   # video_signal_type_present_flag
        writer.write_bits(VIDEO_FORMAT_PAL, 3)  # video_format
        writer.write_flag(False)  # video_full_range_flag
        writer.write_flag(False)  # colour_description_present_flag
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.video_signal_type_present_flag is True
        assert vui.video_format == VIDEO_FORMAT_PAL

    @pytest.mark.xfail(reason="VUI video_signal_type parsing not fully implemented")
    def test_video_full_range_flag(self):
        """Parse video_full_range_flag=1 (full range 0-255)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(True)   # video_signal_type_present_flag
        writer.write_bits(VIDEO_FORMAT_UNSPECIFIED, 3)
        writer.write_flag(True)   # video_full_range_flag
        writer.write_flag(False)  # colour_description_present_flag
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.video_full_range_flag is True


class TestVUIColourDescriptionParsing:
    """Test VUI colour_description parsing.

    H.264 Spec: Annex E.2.1
    """

    @pytest.mark.xfail(reason="VUI colour_description parsing not fully implemented")
    def test_colour_description_bt709(self):
        """Parse BT.709 colour description (HD standard)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(True)   # video_signal_type_present_flag
        writer.write_bits(VIDEO_FORMAT_UNSPECIFIED, 3)
        writer.write_flag(False)  # video_full_range_flag
        writer.write_flag(True)   # colour_description_present_flag
        writer.write_bits(COLOUR_PRIMARIES_BT709, 8)  # colour_primaries
        writer.write_bits(TRANSFER_BT709, 8)  # transfer_characteristics
        writer.write_bits(MATRIX_BT709, 8)  # matrix_coefficients
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.colour_description_present_flag is True
        assert vui.colour_primaries == COLOUR_PRIMARIES_BT709
        assert vui.transfer_characteristics == TRANSFER_BT709
        assert vui.matrix_coefficients == MATRIX_BT709

    @pytest.mark.xfail(reason="VUI colour_description parsing not fully implemented")
    def test_colour_description_smpte170m(self):
        """Parse SMPTE 170M colour description (NTSC/PAL SD)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(True)   # video_signal_type_present_flag
        writer.write_bits(VIDEO_FORMAT_NTSC, 3)
        writer.write_flag(False)
        writer.write_flag(True)
        writer.write_bits(COLOUR_PRIMARIES_SMPTE170M, 8)
        writer.write_bits(TRANSFER_SMPTE170M, 8)
        writer.write_bits(MATRIX_SMPTE170M, 8)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.colour_primaries == COLOUR_PRIMARIES_SMPTE170M
        assert vui.transfer_characteristics == TRANSFER_SMPTE170M
        assert vui.matrix_coefficients == MATRIX_SMPTE170M

    @pytest.mark.xfail(reason="VUI colour_description parsing not fully implemented")
    def test_colour_description_bt2020(self):
        """Parse BT.2020 colour description (UHD/HDR)."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(True)
        writer.write_bits(VIDEO_FORMAT_UNSPECIFIED, 3)
        writer.write_flag(False)
        writer.write_flag(True)
        writer.write_bits(COLOUR_PRIMARIES_BT2020, 8)
        writer.write_bits(TRANSFER_BT2020_10BIT, 8)
        writer.write_bits(MATRIX_BT2020_NCL, 8)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(False)

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.colour_primaries == COLOUR_PRIMARIES_BT2020
        assert vui.transfer_characteristics == TRANSFER_BT2020_10BIT
        assert vui.matrix_coefficients == MATRIX_BT2020_NCL


class TestVUIChromaLocInfoParsing:
    """Test VUI chroma_loc_info parsing.

    H.264 Spec: Annex E.2.1
    """

    @pytest.mark.xfail(reason="VUI chroma_loc_info parsing not fully implemented")
    def test_chroma_loc_info_not_present(self):
        """When chroma_loc_info_present_flag=0, defaults apply."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc_info_present_flag
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.chroma_loc_info_present_flag is False
        assert vui.chroma_sample_loc_type_top_field == 0
        assert vui.chroma_sample_loc_type_bottom_field == 0

    @pytest.mark.xfail(reason="VUI chroma_loc_info parsing not fully implemented")
    def test_chroma_loc_info_present(self):
        """Parse chroma sample location types."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(True)   # chroma_loc_info_present_flag
        writer.write_ue(1)  # chroma_sample_loc_type_top_field
        writer.write_ue(2)  # chroma_sample_loc_type_bottom_field
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.chroma_loc_info_present_flag is True
        assert vui.chroma_sample_loc_type_top_field == 1
        assert vui.chroma_sample_loc_type_bottom_field == 2


class TestVUIBitstreamRestrictionParsing:
    """Test VUI bitstream_restriction parsing.

    H.264 Spec: Annex E.2.1
    """

    @pytest.mark.xfail(reason="VUI bitstream_restriction parsing not fully implemented")
    def test_bitstream_restriction_not_present(self):
        """When bitstream_restriction_flag=0, no restrictions specified."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction_flag

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.bitstream_restriction_flag is False

    @pytest.mark.xfail(reason="VUI bitstream_restriction parsing not fully implemented")
    def test_bitstream_restriction_present(self):
        """Parse full bitstream_restriction data."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(True)   # bitstream_restriction_flag
        writer.write_flag(True)   # motion_vectors_over_pic_boundaries_flag
        writer.write_ue(0)  # max_bytes_per_pic_denom
        writer.write_ue(0)  # max_bits_per_mb_denom
        writer.write_ue(11)  # log2_max_mv_length_horizontal
        writer.write_ue(11)  # log2_max_mv_length_vertical
        writer.write_ue(2)  # max_num_reorder_frames
        writer.write_ue(4)  # max_dec_frame_buffering

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.bitstream_restriction_flag is True
        assert vui.motion_vectors_over_pic_boundaries_flag is True
        assert vui.max_bytes_per_pic_denom == 0
        assert vui.max_bits_per_mb_denom == 0
        assert vui.log2_max_mv_length_horizontal == 11
        assert vui.log2_max_mv_length_vertical == 11
        assert vui.max_num_reorder_frames == 2
        assert vui.max_dec_frame_buffering == 4

    @pytest.mark.xfail(reason="VUI bitstream_restriction parsing not fully implemented")
    def test_max_num_reorder_frames_for_baseline(self):
        """Baseline profile should have max_num_reorder_frames=0."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(True)   # bitstream_restriction_flag
        writer.write_flag(True)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_ue(11)
        writer.write_ue(11)
        writer.write_ue(0)  # max_num_reorder_frames = 0 for baseline
        writer.write_ue(1)

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.max_num_reorder_frames == 0


class TestVUIWithHRDParameters:
    """Test VUI parsing with HRD parameters present.

    H.264 Spec: Annex E.1.1, E.1.2
    """

    @pytest.mark.xfail(reason="VUI with HRD parsing not fully implemented")
    def test_vui_with_nal_hrd_parameters(self):
        """Parse VUI when nal_hrd_parameters_present_flag=1."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(True)   # nal_hrd_parameters_present_flag
        # HRD parameters
        writer.write_ue(0)  # cpb_cnt_minus1
        writer.write_bits(4, 4)  # bit_rate_scale
        writer.write_bits(6, 4)  # cpb_size_scale
        writer.write_ue(62499)  # bit_rate_value_minus1[0]
        writer.write_ue(62499)  # cpb_size_value_minus1[0]
        writer.write_flag(False)  # cbr_flag[0]
        writer.write_bits(23, 5)  # initial_cpb_removal_delay_length_minus1
        writer.write_bits(23, 5)  # cpb_removal_delay_length_minus1
        writer.write_bits(23, 5)  # dpb_output_delay_length_minus1
        writer.write_bits(24, 5)  # time_offset_length
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # low_delay_hrd_flag
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.nal_hrd_parameters_present_flag is True
        assert vui.nal_hrd_parameters is not None
        assert vui.nal_hrd_parameters.cpb_cnt_minus1 == 0

    @pytest.mark.xfail(reason="VUI with HRD parsing not fully implemented")
    def test_vui_with_vcl_hrd_parameters(self):
        """Parse VUI when vcl_hrd_parameters_present_flag=1."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(True)   # vcl_hrd_parameters_present_flag
        # HRD parameters
        writer.write_ue(0)
        writer.write_bits(4, 4)
        writer.write_bits(6, 4)
        writer.write_ue(62499)
        writer.write_ue(62499)
        writer.write_flag(True)  # cbr_flag
        writer.write_bits(23, 5)
        writer.write_bits(23, 5)
        writer.write_bits(23, 5)
        writer.write_bits(24, 5)
        writer.write_flag(False)  # low_delay_hrd_flag
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.vcl_hrd_parameters_present_flag is True
        assert vui.vcl_hrd_parameters is not None

    @pytest.mark.xfail(reason="VUI with HRD parsing not fully implemented")
    def test_low_delay_hrd_flag(self):
        """Parse low_delay_hrd_flag when HRD present."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(True)   # nal_hrd
        # Minimal HRD
        writer.write_ue(0)
        writer.write_bits(0, 4)
        writer.write_bits(0, 4)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_flag(False)
        writer.write_bits(0, 5)
        writer.write_bits(0, 5)
        writer.write_bits(0, 5)
        writer.write_bits(0, 5)
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(True)   # low_delay_hrd_flag
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.low_delay_hrd_flag is True


class TestVUIIntegrationWithSPS:
    """Test VUI parsing integrated with SPS parsing."""

    @pytest.mark.xfail(reason="VUI sample_aspect_ratio property not implemented")
    def test_sps_with_full_vui_sample_aspect_ratio(self):
        """Parse SPS with VUI and verify sample_aspect_ratio property."""
        from parameters.sps import parse_sps

        writer = BitWriter()
        # SPS header
        writer.write_bits(66, 8)  # profile_idc (Baseline)
        writer.write_bits(0, 6)   # constraint flags
        writer.write_bits(0, 2)   # reserved
        writer.write_bits(30, 8)  # level_idc
        writer.write_ue(0)  # sps_id
        writer.write_ue(0)  # log2_max_frame_num_minus4
        writer.write_ue(2)  # pic_order_cnt_type
        writer.write_ue(1)  # max_num_ref_frames
        writer.write_flag(False)  # gaps
        writer.write_ue(79)  # 1280 pixels / 16 - 1
        writer.write_ue(44)  # 720 pixels / 16 - 1
        writer.write_flag(True)  # frame_mbs_only
        writer.write_flag(False)  # direct_8x8
        writer.write_flag(False)  # cropping
        writer.write_flag(True)   # vui_parameters_present_flag

        # VUI parameters
        writer.write_flag(True)   # aspect_ratio_info_present_flag
        writer.write_bits(SAR_SQUARE, 8)
        writer.write_flag(False)  # overscan
        writer.write_flag(True)   # video_signal_type_present_flag
        writer.write_bits(VIDEO_FORMAT_UNSPECIFIED, 3)
        writer.write_flag(False)  # full_range
        writer.write_flag(True)   # colour_description_present_flag
        writer.write_bits(COLOUR_PRIMARIES_BT709, 8)
        writer.write_bits(TRANSFER_BT709, 8)
        writer.write_bits(MATRIX_BT709, 8)
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(True)   # timing_info_present_flag
        writer.write_bits(1001, 32)
        writer.write_bits(60000, 32)
        writer.write_flag(True)   # fixed_frame_rate
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        assert sps.vui_parameters_present_flag is True
        assert sps.vui_parameters is not None
        assert sps.vui_parameters.aspect_ratio_idc == SAR_SQUARE
        # Test the sample_aspect_ratio property that converts idc to tuple
        assert sps.vui_parameters.sample_aspect_ratio == (1, 1)

    @pytest.mark.xfail(reason="VUI bitstream_restriction storage not implemented")
    def test_sps_vui_bitstream_restriction_fields(self):
        """Parse SPS with VUI bitstream_restriction and verify all fields stored."""
        from parameters.sps import parse_sps

        writer = BitWriter()
        # SPS header
        writer.write_bits(66, 8)
        writer.write_bits(0, 6)
        writer.write_bits(0, 2)
        writer.write_bits(30, 8)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_ue(2)
        writer.write_ue(1)
        writer.write_flag(False)
        writer.write_ue(79)
        writer.write_ue(44)
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_flag(True)   # vui_parameters_present_flag

        # VUI parameters with bitstream restriction
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(False)  # nal_hrd
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(True)   # bitstream_restriction_flag
        writer.write_flag(True)   # motion_vectors_over_pic_boundaries_flag
        writer.write_ue(0)  # max_bytes_per_pic_denom
        writer.write_ue(0)  # max_bits_per_mb_denom
        writer.write_ue(11)  # log2_max_mv_length_horizontal
        writer.write_ue(11)  # log2_max_mv_length_vertical
        writer.write_ue(2)  # max_num_reorder_frames
        writer.write_ue(4)  # max_dec_frame_buffering

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        assert sps.vui_parameters.bitstream_restriction_flag is True
        # These fields should be stored and accessible
        assert sps.vui_parameters.max_num_reorder_frames == 2
        assert sps.vui_parameters.max_dec_frame_buffering == 4
        assert sps.vui_parameters.log2_max_mv_length_horizontal == 11


class TestVUIHelperMethods:
    """Test VUI helper methods and properties."""

    @pytest.mark.xfail(reason="VUI helper methods not fully implemented")
    def test_display_aspect_ratio_calculation(self):
        """Calculate display aspect ratio from SAR and dimensions."""
        from parameters.sps import VUIParameters

        vui = VUIParameters(
            aspect_ratio_info_present_flag=True,
            aspect_ratio_idc=SAR_SQUARE
        )
        # For 1920x1080 with 1:1 SAR, DAR should be 16:9
        dar = vui.get_display_aspect_ratio(1920, 1080)
        assert abs(dar - (16 / 9)) < 0.01

    @pytest.mark.xfail(reason="VUI helper methods not fully implemented")
    def test_is_hd_content(self):
        """Detect HD content from VUI colour description."""
        from parameters.sps import VUIParameters

        vui = VUIParameters(
            video_signal_type_present_flag=True,
            colour_description_present_flag=True,
            colour_primaries=COLOUR_PRIMARIES_BT709,
            transfer_characteristics=TRANSFER_BT709,
            matrix_coefficients=MATRIX_BT709
        )
        assert vui.is_hd_colorspace is True

    @pytest.mark.xfail(reason="VUI helper methods not fully implemented")
    def test_is_sd_content(self):
        """Detect SD content from VUI colour description."""
        from parameters.sps import VUIParameters

        vui = VUIParameters(
            video_signal_type_present_flag=True,
            colour_description_present_flag=True,
            colour_primaries=COLOUR_PRIMARIES_SMPTE170M,
            transfer_characteristics=TRANSFER_SMPTE170M,
            matrix_coefficients=MATRIX_SMPTE170M
        )
        assert vui.is_sd_colorspace is True
