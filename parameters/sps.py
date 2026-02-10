# h264/parameters/sps.py
"""Sequence Parameter Set (SPS) parsing for H.264.

The SPS contains sequence-level parameters like video dimensions,
profile/level, and reference frame configuration.

H.264 Spec Reference:
- Section 7.3.2.1: Sequence parameter set RBSP syntax
- Section 7.4.2.1: Sequence parameter set RBSP semantics
- Annex A: Profiles, levels, and compliance

Profile IDCs (Annex A.2):
- 66: Baseline Profile
- 77: Main Profile
- 88: Extended Profile
- 100: High Profile
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from bitstream import BitReader

logger = logging.getLogger(__name__)


# Profile IDC values
PROFILE_BASELINE = 66
PROFILE_MAIN = 77
PROFILE_EXTENDED = 88
PROFILE_HIGH = 100
PROFILE_HIGH_10 = 110
PROFILE_HIGH_422 = 122
PROFILE_HIGH_444 = 244


@dataclass
class VUIParameters:
    """Video Usability Information (VUI) parameters.

    H.264 Spec: Annex E
    """
    aspect_ratio_info_present_flag: bool = False
    aspect_ratio_idc: int = 0
    sar_width: int = 0
    sar_height: int = 0

    overscan_info_present_flag: bool = False
    overscan_appropriate_flag: bool = False

    video_signal_type_present_flag: bool = False
    video_format: int = 5  # Unspecified
    video_full_range_flag: bool = False
    colour_description_present_flag: bool = False
    colour_primaries: int = 2  # Unspecified
    transfer_characteristics: int = 2  # Unspecified
    matrix_coefficients: int = 2  # Unspecified

    chroma_loc_info_present_flag: bool = False
    chroma_sample_loc_type_top_field: int = 0
    chroma_sample_loc_type_bottom_field: int = 0

    timing_info_present_flag: bool = False
    num_units_in_tick: int = 0
    time_scale: int = 0
    fixed_frame_rate_flag: bool = False

    nal_hrd_parameters_present_flag: bool = False
    vcl_hrd_parameters_present_flag: bool = False

    pic_struct_present_flag: bool = False
    bitstream_restriction_flag: bool = False

    @property
    def frame_rate(self) -> Optional[float]:
        """Calculate frame rate from timing info if available."""
        if self.timing_info_present_flag and self.num_units_in_tick > 0:
            # time_scale / (2 * num_units_in_tick) for progressive
            return self.time_scale / (2 * self.num_units_in_tick)
        return None


@dataclass
class SPS:
    """Sequence Parameter Set (SPS) data structure.

    Contains all decoded SPS fields needed for video decoding.

    Attributes:
        profile_idc: Profile identifier (66=Baseline, 77=Main, etc.)
        level_idc: Level identifier (e.g., 30 = Level 3.0)
        seq_parameter_set_id: SPS identifier (0-31)

        log2_max_frame_num_minus4: For frame_num calculation
        pic_order_cnt_type: POC type (0, 1, or 2)

        max_num_ref_frames: Max reference frames
        pic_width_in_mbs_minus1: Picture width in macroblocks - 1
        pic_height_in_map_units_minus1: Picture height in map units - 1
        frame_mbs_only_flag: True if only frame coding (no field)
    """
    # Profile and level
    profile_idc: int = 0
    constraint_set0_flag: bool = False
    constraint_set1_flag: bool = False
    constraint_set2_flag: bool = False
    constraint_set3_flag: bool = False
    constraint_set4_flag: bool = False
    constraint_set5_flag: bool = False
    level_idc: int = 0
    seq_parameter_set_id: int = 0

    # Chroma format (for High profile and above)
    chroma_format_idc: int = 1  # Default 4:2:0
    separate_colour_plane_flag: bool = False
    bit_depth_luma_minus8: int = 0
    bit_depth_chroma_minus8: int = 0
    qpprime_y_zero_transform_bypass_flag: bool = False
    seq_scaling_matrix_present_flag: bool = False

    # Frame numbering
    log2_max_frame_num_minus4: int = 0

    # Picture order count
    pic_order_cnt_type: int = 0
    log2_max_pic_order_cnt_lsb_minus4: int = 0
    delta_pic_order_always_zero_flag: bool = False
    offset_for_non_ref_pic: int = 0
    offset_for_top_to_bottom_field: int = 0
    num_ref_frames_in_pic_order_cnt_cycle: int = 0
    offset_for_ref_frame: List[int] = field(default_factory=list)

    # Reference frames
    max_num_ref_frames: int = 0
    gaps_in_frame_num_value_allowed_flag: bool = False

    # Picture dimensions
    pic_width_in_mbs_minus1: int = 0
    pic_height_in_map_units_minus1: int = 0
    frame_mbs_only_flag: bool = True
    mb_adaptive_frame_field_flag: bool = False

    # Other flags
    direct_8x8_inference_flag: bool = False

    # Frame cropping
    frame_cropping_flag: bool = False
    frame_crop_left_offset: int = 0
    frame_crop_right_offset: int = 0
    frame_crop_top_offset: int = 0
    frame_crop_bottom_offset: int = 0

    # VUI
    vui_parameters_present_flag: bool = False
    vui_parameters: Optional[VUIParameters] = None

    # Scaling lists (High profile)
    scaling_lists_4x4: List[List[int]] = field(default_factory=list)
    scaling_lists_8x8: List[List[int]] = field(default_factory=list)

    # Derived values
    @property
    def profile_name(self) -> str:
        """Human-readable profile name."""
        profiles = {
            66: "Baseline",
            77: "Main",
            88: "Extended",
            100: "High",
            110: "High 10",
            122: "High 4:2:2",
            244: "High 4:4:4",
        }
        return profiles.get(self.profile_idc, f"Unknown({self.profile_idc})")

    @property
    def level(self) -> str:
        """Human-readable level string (e.g., '3.0')."""
        return f"{self.level_idc // 10}.{self.level_idc % 10}"

    @property
    def max_frame_num(self) -> int:
        """Maximum frame_num value."""
        return 1 << (self.log2_max_frame_num_minus4 + 4)

    @property
    def max_pic_order_cnt_lsb(self) -> int:
        """Maximum pic_order_cnt_lsb value (for POC type 0)."""
        if self.pic_order_cnt_type == 0:
            return 1 << (self.log2_max_pic_order_cnt_lsb_minus4 + 4)
        return 0

    @property
    def pic_width_in_mbs(self) -> int:
        """Picture width in macroblocks."""
        return self.pic_width_in_mbs_minus1 + 1

    @property
    def pic_height_in_map_units(self) -> int:
        """Picture height in map units."""
        return self.pic_height_in_map_units_minus1 + 1

    @property
    def frame_height_in_mbs(self) -> int:
        """Frame height in macroblocks."""
        return (2 - (1 if self.frame_mbs_only_flag else 0)) * self.pic_height_in_map_units

    @property
    def width(self) -> int:
        """Picture width in pixels (before cropping)."""
        return self.pic_width_in_mbs * 16

    @property
    def height(self) -> int:
        """Picture height in pixels (before cropping)."""
        return self.frame_height_in_mbs * 16

    @property
    def cropped_width(self) -> int:
        """Picture width in pixels (after cropping)."""
        crop_unit_x, _ = self._get_crop_units()
        return self.width - crop_unit_x * (
            self.frame_crop_left_offset + self.frame_crop_right_offset
        )

    @property
    def cropped_height(self) -> int:
        """Picture height in pixels (after cropping)."""
        _, crop_unit_y = self._get_crop_units()
        return self.height - crop_unit_y * (
            self.frame_crop_top_offset + self.frame_crop_bottom_offset
        )

    def _get_subsampling_factors(self) -> Tuple[int, int]:
        if self.separate_colour_plane_flag or self.chroma_format_idc == 0:
            return 1, 1
        if self.chroma_format_idc == 1:
            return 2, 2
        if self.chroma_format_idc == 2:
            return 2, 1
        if self.chroma_format_idc == 3:
            return 1, 1
        raise ValueError(f"Invalid chroma_format_idc: {self.chroma_format_idc}")

    def _get_crop_units(self) -> Tuple[int, int]:
        if self.separate_colour_plane_flag or self.chroma_format_idc == 0:
            crop_unit_x = 1
        else:
            sub_w, _ = self._get_subsampling_factors()
            crop_unit_x = sub_w

        if self.separate_colour_plane_flag or self.chroma_format_idc == 0:
            crop_unit_y = 2 - (1 if self.frame_mbs_only_flag else 0)
        else:
            _, sub_h = self._get_subsampling_factors()
            crop_unit_y = sub_h * (2 - (1 if self.frame_mbs_only_flag else 0))

        return int(crop_unit_x), int(crop_unit_y)

    def get_chroma_crop_left(self) -> int:
        if not self.frame_cropping_flag:
            return 0
        if self.chroma_format_idc == 0 and not self.separate_colour_plane_flag:
            return 0
        crop_unit_x, _ = self._get_crop_units()
        sub_w, _ = self._get_subsampling_factors()
        return int((self.frame_crop_left_offset * crop_unit_x) // sub_w)

    def get_chroma_crop_right(self) -> int:
        if not self.frame_cropping_flag:
            return 0
        if self.chroma_format_idc == 0 and not self.separate_colour_plane_flag:
            return 0
        crop_unit_x, _ = self._get_crop_units()
        sub_w, _ = self._get_subsampling_factors()
        return int((self.frame_crop_right_offset * crop_unit_x) // sub_w)

    def get_chroma_crop_top(self) -> int:
        if not self.frame_cropping_flag:
            return 0
        if self.chroma_format_idc == 0 and not self.separate_colour_plane_flag:
            return 0
        _, crop_unit_y = self._get_crop_units()
        _, sub_h = self._get_subsampling_factors()
        return int((self.frame_crop_top_offset * crop_unit_y) // sub_h)

    def get_chroma_crop_bottom(self) -> int:
        if not self.frame_cropping_flag:
            return 0
        if self.chroma_format_idc == 0 and not self.separate_colour_plane_flag:
            return 0
        _, crop_unit_y = self._get_crop_units()
        _, sub_h = self._get_subsampling_factors()
        return int((self.frame_crop_bottom_offset * crop_unit_y) // sub_h)

    @property
    def chroma_format_name(self) -> str:
        """Human-readable chroma format."""
        formats = {0: "Monochrome", 1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}
        return formats.get(self.chroma_format_idc, "Unknown")

    @property
    def bit_depth_luma(self) -> int:
        """Luma bit depth."""
        return 8 + self.bit_depth_luma_minus8

    @property
    def bit_depth_chroma(self) -> int:
        """Chroma bit depth."""
        return 8 + self.bit_depth_chroma_minus8

    def __repr__(self) -> str:
        return (
            f"SPS(id={self.seq_parameter_set_id}, "
            f"profile={self.profile_name}, level={self.level}, "
            f"size={self.cropped_width}x{self.cropped_height}, "
            f"chroma={self.chroma_format_name})"
        )


def _is_high_profile(profile_idc: int) -> bool:
    """Check if profile requires extended SPS parsing."""
    return profile_idc in (
        100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135
    )


def _parse_hrd_parameters(reader: BitReader) -> dict:
    """Parse Hypothetical Reference Decoder (HRD) parameters.

    H.264 Spec: Annex E.1.2
    """
    hrd = {}
    hrd['cpb_cnt_minus1'] = reader.read_ue()
    hrd['bit_rate_scale'] = reader.read_bits(4)
    hrd['cpb_size_scale'] = reader.read_bits(4)

    cpb_cnt = hrd['cpb_cnt_minus1'] + 1
    hrd['bit_rate_value_minus1'] = []
    hrd['cpb_size_value_minus1'] = []
    hrd['cbr_flag'] = []

    for _ in range(cpb_cnt):
        hrd['bit_rate_value_minus1'].append(reader.read_ue())
        hrd['cpb_size_value_minus1'].append(reader.read_ue())
        hrd['cbr_flag'].append(reader.read_flag())

    hrd['initial_cpb_removal_delay_length_minus1'] = reader.read_bits(5)
    hrd['cpb_removal_delay_length_minus1'] = reader.read_bits(5)
    hrd['dpb_output_delay_length_minus1'] = reader.read_bits(5)
    hrd['time_offset_length'] = reader.read_bits(5)

    return hrd


def _parse_vui_parameters(reader: BitReader) -> VUIParameters:
    """Parse Video Usability Information (VUI) parameters.

    H.264 Spec: Annex E.1.1
    """
    vui = VUIParameters()

    vui.aspect_ratio_info_present_flag = reader.read_flag()
    if vui.aspect_ratio_info_present_flag:
        vui.aspect_ratio_idc = reader.read_bits(8)
        if vui.aspect_ratio_idc == 255:  # Extended_SAR
            vui.sar_width = reader.read_bits(16)
            vui.sar_height = reader.read_bits(16)

    vui.overscan_info_present_flag = reader.read_flag()
    if vui.overscan_info_present_flag:
        vui.overscan_appropriate_flag = reader.read_flag()

    vui.video_signal_type_present_flag = reader.read_flag()
    if vui.video_signal_type_present_flag:
        vui.video_format = reader.read_bits(3)
        vui.video_full_range_flag = reader.read_flag()
        vui.colour_description_present_flag = reader.read_flag()
        if vui.colour_description_present_flag:
            vui.colour_primaries = reader.read_bits(8)
            vui.transfer_characteristics = reader.read_bits(8)
            vui.matrix_coefficients = reader.read_bits(8)

    vui.chroma_loc_info_present_flag = reader.read_flag()
    if vui.chroma_loc_info_present_flag:
        vui.chroma_sample_loc_type_top_field = reader.read_ue()
        vui.chroma_sample_loc_type_bottom_field = reader.read_ue()

    vui.timing_info_present_flag = reader.read_flag()
    if vui.timing_info_present_flag:
        vui.num_units_in_tick = reader.read_bits(32)
        vui.time_scale = reader.read_bits(32)
        vui.fixed_frame_rate_flag = reader.read_flag()

    vui.nal_hrd_parameters_present_flag = reader.read_flag()
    if vui.nal_hrd_parameters_present_flag:
        _parse_hrd_parameters(reader)  # Parse but don't store for now

    vui.vcl_hrd_parameters_present_flag = reader.read_flag()
    if vui.vcl_hrd_parameters_present_flag:
        _parse_hrd_parameters(reader)

    if vui.nal_hrd_parameters_present_flag or vui.vcl_hrd_parameters_present_flag:
        reader.read_flag()  # low_delay_hrd_flag

    vui.pic_struct_present_flag = reader.read_flag()

    vui.bitstream_restriction_flag = reader.read_flag()
    if vui.bitstream_restriction_flag:
        reader.read_flag()  # motion_vectors_over_pic_boundaries_flag
        reader.read_ue()    # max_bytes_per_pic_denom
        reader.read_ue()    # max_bits_per_mb_denom
        reader.read_ue()    # log2_max_mv_length_horizontal
        reader.read_ue()    # log2_max_mv_length_vertical
        reader.read_ue()    # max_num_reorder_frames
        reader.read_ue()    # max_dec_frame_buffering

    return vui


def _parse_scaling_list(reader: BitReader, size: int) -> list:
    """Parse scaling list.

    H.264 Spec: Section 7.3.2.1.1
    """
    scaling_list = [0] * size
    last_scale = 8
    next_scale = 8

    for j in range(size):
        if next_scale != 0:
            delta_scale = reader.read_se()
            next_scale = (last_scale + delta_scale + 256) % 256

        scaling_list[j] = last_scale if next_scale == 0 else next_scale
        last_scale = scaling_list[j]

    return scaling_list


def parse_sps(rbsp: bytes) -> SPS:
    """Parse Sequence Parameter Set from RBSP bytes.

    Args:
        rbsp: Raw Byte Sequence Payload (NAL unit data after header,
              with emulation prevention bytes already removed)

    Returns:
        Parsed SPS object

    Raises:
        ValueError: If RBSP is malformed or too short

    H.264 Spec: Section 7.3.2.1
    """
    if len(rbsp) < 3:
        raise ValueError("SPS RBSP too short")

    reader = BitReader(rbsp)
    sps = SPS()

    # Profile and level
    sps.profile_idc = reader.read_bits(8)
    sps.constraint_set0_flag = reader.read_flag()
    sps.constraint_set1_flag = reader.read_flag()
    sps.constraint_set2_flag = reader.read_flag()
    sps.constraint_set3_flag = reader.read_flag()
    sps.constraint_set4_flag = reader.read_flag()
    sps.constraint_set5_flag = reader.read_flag()
    reader.read_bits(2)  # reserved_zero_2bits
    sps.level_idc = reader.read_bits(8)

    sps.seq_parameter_set_id = reader.read_ue()

    logger.debug(
        f"Parsing SPS: profile={sps.profile_idc}, level={sps.level_idc}, "
        f"id={sps.seq_parameter_set_id}"
    )

    # High profile extensions
    if _is_high_profile(sps.profile_idc):
        sps.chroma_format_idc = reader.read_ue()

        if sps.chroma_format_idc == 3:  # 4:4:4
            sps.separate_colour_plane_flag = reader.read_flag()

        sps.bit_depth_luma_minus8 = reader.read_ue()
        sps.bit_depth_chroma_minus8 = reader.read_ue()
        sps.qpprime_y_zero_transform_bypass_flag = reader.read_flag()
        sps.seq_scaling_matrix_present_flag = reader.read_flag()

        if sps.seq_scaling_matrix_present_flag:
            # Parse and store scaling lists
            n_scaling_lists = 12 if sps.chroma_format_idc == 3 else 8
            for i in range(n_scaling_lists):
                seq_scaling_list_present_flag = reader.read_flag()
                if seq_scaling_list_present_flag:
                    size = 16 if i < 6 else 64
                    scaling_list = _parse_scaling_list(reader, size)
                    if i < 6:
                        sps.scaling_lists_4x4.append(scaling_list)
                    else:
                        sps.scaling_lists_8x8.append(scaling_list)
                else:
                    # Use default (flat) scaling list
                    if i < 6:
                        sps.scaling_lists_4x4.append([16] * 16)
                    else:
                        sps.scaling_lists_8x8.append([16] * 64)

    # Frame numbering
    sps.log2_max_frame_num_minus4 = reader.read_ue()

    # Picture order count
    sps.pic_order_cnt_type = reader.read_ue()

    if sps.pic_order_cnt_type == 0:
        sps.log2_max_pic_order_cnt_lsb_minus4 = reader.read_ue()
    elif sps.pic_order_cnt_type == 1:
        sps.delta_pic_order_always_zero_flag = reader.read_flag()
        sps.offset_for_non_ref_pic = reader.read_se()
        sps.offset_for_top_to_bottom_field = reader.read_se()
        sps.num_ref_frames_in_pic_order_cnt_cycle = reader.read_ue()
        sps.offset_for_ref_frame = []
        for _ in range(sps.num_ref_frames_in_pic_order_cnt_cycle):
            sps.offset_for_ref_frame.append(reader.read_se())

    # Reference frames
    sps.max_num_ref_frames = reader.read_ue()
    sps.gaps_in_frame_num_value_allowed_flag = reader.read_flag()

    # Picture dimensions
    sps.pic_width_in_mbs_minus1 = reader.read_ue()
    sps.pic_height_in_map_units_minus1 = reader.read_ue()

    logger.debug(
        f"SPS dimensions: {sps.pic_width_in_mbs}x{sps.frame_height_in_mbs} MBs "
        f"= {sps.width}x{sps.height} pixels"
    )

    # Frame/field coding
    sps.frame_mbs_only_flag = reader.read_flag()
    if not sps.frame_mbs_only_flag:
        sps.mb_adaptive_frame_field_flag = reader.read_flag()

    sps.direct_8x8_inference_flag = reader.read_flag()

    # Frame cropping
    sps.frame_cropping_flag = reader.read_flag()
    if sps.frame_cropping_flag:
        sps.frame_crop_left_offset = reader.read_ue()
        sps.frame_crop_right_offset = reader.read_ue()
        sps.frame_crop_top_offset = reader.read_ue()
        sps.frame_crop_bottom_offset = reader.read_ue()

        logger.debug(
            f"SPS cropping: left={sps.frame_crop_left_offset}, "
            f"right={sps.frame_crop_right_offset}, "
            f"top={sps.frame_crop_top_offset}, "
            f"bottom={sps.frame_crop_bottom_offset}"
        )

    # VUI parameters
    sps.vui_parameters_present_flag = reader.read_flag()
    if sps.vui_parameters_present_flag:
        sps.vui_parameters = _parse_vui_parameters(reader)
        if sps.vui_parameters.frame_rate:
            logger.debug(f"SPS frame rate: {sps.vui_parameters.frame_rate:.2f} fps")

    logger.info(f"Parsed {sps}")
    return sps
