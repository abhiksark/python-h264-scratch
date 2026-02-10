# h264/parameters/pps.py
"""Picture Parameter Set (PPS) parsing for H.264.

The PPS contains picture-level parameters like quantization settings,
entropy coding mode, and deblocking filter configuration.

H.264 Spec Reference:
- Section 7.3.2.2: Picture parameter set RBSP syntax
- Section 7.4.2.2: Picture parameter set RBSP semantics
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from bitstream import BitReader

logger = logging.getLogger(__name__)


@dataclass
class PPS:
    """Picture Parameter Set (PPS) data structure.

    Contains all decoded PPS fields needed for video decoding.

    Attributes:
        pic_parameter_set_id: PPS identifier (0-255)
        seq_parameter_set_id: Referenced SPS identifier (0-31)
        entropy_coding_mode_flag: False=CAVLC, True=CABAC
        bottom_field_pic_order_in_frame_present_flag: POC for bottom field
        num_slice_groups_minus1: 0 for baseline (no FMO)
        num_ref_idx_l0_default_active_minus1: Default L0 reference count - 1
        num_ref_idx_l1_default_active_minus1: Default L1 reference count - 1
        weighted_pred_flag: Weighted prediction for P slices
        weighted_bipred_idc: Weighted bi-prediction mode
        pic_init_qp_minus26: Initial QP offset (-26 to +25)
        pic_init_qs_minus26: Initial QS offset for SP/SI slices
        chroma_qp_index_offset: Chroma QP offset (-12 to +12)
        deblocking_filter_control_present_flag: Deblocking params in slice header
        constrained_intra_pred_flag: Intra pred uses only intra neighbors
        redundant_pic_cnt_present_flag: Redundant picture count in slice
    """
    pic_parameter_set_id: int = 0
    seq_parameter_set_id: int = 0

    # Entropy coding
    entropy_coding_mode_flag: bool = False  # False = CAVLC, True = CABAC

    # Field/frame coding
    bottom_field_pic_order_in_frame_present_flag: bool = False

    # Slice groups (FMO - not used in Baseline)
    num_slice_groups_minus1: int = 0
    slice_group_map_type: int = 0
    run_length_minus1: List[int] = field(default_factory=list)
    top_left: List[int] = field(default_factory=list)
    bottom_right: List[int] = field(default_factory=list)
    slice_group_change_direction_flag: bool = False
    slice_group_change_rate_minus1: int = 0
    slice_group_change_cycle: int = 0
    slice_group_id: List[int] = field(default_factory=list)

    # Reference pictures
    num_ref_idx_l0_default_active_minus1: int = 0
    num_ref_idx_l1_default_active_minus1: int = 0

    # Weighted prediction
    weighted_pred_flag: bool = False
    weighted_bipred_idc: int = 0

    # Quantization
    pic_init_qp_minus26: int = 0
    pic_init_qs_minus26: int = 0
    chroma_qp_index_offset: int = 0
    second_chroma_qp_index_offset: int = 0  # For High profile

    # Deblocking
    deblocking_filter_control_present_flag: bool = False

    # Intra prediction
    constrained_intra_pred_flag: bool = False

    # Redundant pictures
    redundant_pic_cnt_present_flag: bool = False

    # Transform (High profile)
    transform_8x8_mode_flag: bool = False
    pic_scaling_matrix_present_flag: bool = False
    pic_scaling_list_present_flag: List[bool] = field(default_factory=list)

    # Scaling lists (High profile)
    scaling_lists_4x4: List[List[int]] = field(default_factory=list)
    scaling_lists_8x8: List[List[int]] = field(default_factory=list)

    @property
    def entropy_coding_mode(self) -> str:
        """Human-readable entropy coding mode."""
        return "CABAC" if self.entropy_coding_mode_flag else "CAVLC"

    @property
    def num_ref_idx_l0_default_active(self) -> int:
        """Default number of L0 reference pictures."""
        return self.num_ref_idx_l0_default_active_minus1 + 1

    @property
    def num_ref_idx_l1_default_active(self) -> int:
        """Default number of L1 reference pictures."""
        return self.num_ref_idx_l1_default_active_minus1 + 1

    @property
    def pic_init_qp(self) -> int:
        """Initial QP value (26 + offset)."""
        return 26 + self.pic_init_qp_minus26

    @property
    def weighted_bipred_mode(self) -> str:
        """Human-readable weighted bipred mode."""
        modes = {0: "Default", 1: "Explicit", 2: "Implicit"}
        return modes.get(self.weighted_bipred_idc, "Unknown")

    def __post_init__(self) -> None:
        if self.second_chroma_qp_index_offset == 0 and self.chroma_qp_index_offset != 0:
            self.second_chroma_qp_index_offset = self.chroma_qp_index_offset

    def __repr__(self) -> str:
        return (
            f"PPS(id={self.pic_parameter_set_id}, "
            f"sps_id={self.seq_parameter_set_id}, "
            f"entropy={self.entropy_coding_mode}, "
            f"init_qp={self.pic_init_qp})"
        )


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


def should_parse_second_chroma_offset(profile_idc: int) -> bool:
    return profile_idc in (100, 110, 122, 244)


def validate_second_chroma_offset(value: int) -> bool:
    return -12 <= int(value) <= 12


def parse_pps(rbsp: bytes, is_high_profile: bool = False) -> PPS:
    """Parse Picture Parameter Set from RBSP bytes.

    Args:
        rbsp: Raw Byte Sequence Payload (NAL unit data after header,
              with emulation prevention bytes already removed)
        is_high_profile: Whether the referenced SPS is High profile or above

    Returns:
        Parsed PPS object

    Raises:
        ValueError: If RBSP is malformed or too short

    H.264 Spec: Section 7.3.2.2
    """
    if len(rbsp) < 1:
        raise ValueError("PPS RBSP too short")

    reader = BitReader(rbsp)
    pps = PPS()

    # Basic identifiers
    pps.pic_parameter_set_id = reader.read_ue()
    pps.seq_parameter_set_id = reader.read_ue()

    logger.debug(
        f"Parsing PPS: id={pps.pic_parameter_set_id}, "
        f"sps_id={pps.seq_parameter_set_id}"
    )

    # Entropy coding mode
    pps.entropy_coding_mode_flag = reader.read_flag()

    # Field coding
    pps.bottom_field_pic_order_in_frame_present_flag = reader.read_flag()

    # Slice groups (FMO)
    pps.num_slice_groups_minus1 = reader.read_ue()

    if pps.num_slice_groups_minus1 > 0:
        # FMO (Flexible Macroblock Ordering) - rarely used, not in Baseline
        pps.slice_group_map_type = reader.read_ue()

        if pps.slice_group_map_type == 0:
            pps.run_length_minus1 = []
            for _ in range(pps.num_slice_groups_minus1 + 1):
                pps.run_length_minus1.append(reader.read_ue())
        elif pps.slice_group_map_type == 2:
            pps.top_left = []
            pps.bottom_right = []
            for _ in range(pps.num_slice_groups_minus1):
                pps.top_left.append(reader.read_ue())
                pps.bottom_right.append(reader.read_ue())
        elif pps.slice_group_map_type in (3, 4, 5):
            pps.slice_group_change_direction_flag = reader.read_flag()
            pps.slice_group_change_rate_minus1 = reader.read_ue()
        elif pps.slice_group_map_type == 6:
            pic_size_in_map_units = reader.read_ue() + 1
            num_slice_groups = pps.num_slice_groups_minus1 + 1
            bits = (num_slice_groups - 1).bit_length() or 1
            pps.slice_group_id = []
            for _ in range(pic_size_in_map_units):
                pps.slice_group_id.append(reader.read_bits(bits))

    # Reference picture defaults
    pps.num_ref_idx_l0_default_active_minus1 = reader.read_ue()
    pps.num_ref_idx_l1_default_active_minus1 = reader.read_ue()

    # Weighted prediction
    pps.weighted_pred_flag = reader.read_flag()
    pps.weighted_bipred_idc = reader.read_bits(2)

    # Quantization
    pps.pic_init_qp_minus26 = reader.read_se()
    pps.pic_init_qs_minus26 = reader.read_se()
    pps.chroma_qp_index_offset = reader.read_se()

    logger.debug(
        f"PPS QP: init_qp={pps.pic_init_qp}, "
        f"chroma_offset={pps.chroma_qp_index_offset}"
    )

    # Deblocking and other flags
    pps.deblocking_filter_control_present_flag = reader.read_flag()
    pps.constrained_intra_pred_flag = reader.read_flag()
    pps.redundant_pic_cnt_present_flag = reader.read_flag()

    parsed_second_offset = False

    # High profile extensions
    if is_high_profile and reader.bits_remaining > 0:
        # Check if there's more data (more_rbsp_data)
        try:
            pps.transform_8x8_mode_flag = reader.read_flag()
            pps.pic_scaling_matrix_present_flag = reader.read_flag()

            if pps.pic_scaling_matrix_present_flag:
                n_scaling_lists = 6 + (2 if pps.transform_8x8_mode_flag else 0)
                pps.pic_scaling_list_present_flag = []

                for i in range(n_scaling_lists):
                    present = reader.read_flag()
                    pps.pic_scaling_list_present_flag.append(present)
                    if present:
                        size = 16 if i < 6 else 64
                        scaling_list = _parse_scaling_list(reader, size)
                        if i < 6:
                            pps.scaling_lists_4x4.append(scaling_list)
                        else:
                            pps.scaling_lists_8x8.append(scaling_list)
                    else:
                        # Use default (flat) scaling list
                        if i < 6:
                            pps.scaling_lists_4x4.append([16] * 16)
                        else:
                            pps.scaling_lists_8x8.append([16] * 64)

            pps.second_chroma_qp_index_offset = reader.read_se()
            parsed_second_offset = True
        except Exception:
            # Not enough data for high profile extensions
            pass

    if not parsed_second_offset:
        pps.second_chroma_qp_index_offset = pps.chroma_qp_index_offset

    logger.info(f"Parsed {pps}")
    return pps
