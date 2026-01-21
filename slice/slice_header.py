# h264/slice/slice_header.py
"""Slice header parsing for H.264.

The slice header contains per-slice parameters like slice type, QP delta,
reference picture management, and deblocking filter settings.

H.264 Spec Reference:
- Section 7.3.3: Slice header syntax
- Section 7.4.3: Slice header semantics
- Table 7-6: Slice type values

Slice Types:
- 0, 5: P slice (predictive)
- 1, 6: B slice (bi-predictive)
- 2, 7: I slice (intra)
- 3, 8: SP slice (switching P)
- 4, 9: SI slice (switching I)

Values 5-9 indicate all MBs in slice are of that type.
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Tuple

from bitstream import BitReader
from parameters import SPS, PPS

logger = logging.getLogger(__name__)


class SliceType(IntEnum):
    """Slice type values from H.264 Table 7-6."""
    P = 0
    B = 1
    I = 2
    SP = 3
    SI = 4
    # Values 5-9 are "all MBs same type" versions
    P_ALL = 5
    B_ALL = 6
    I_ALL = 7
    SP_ALL = 8
    SI_ALL = 9

    @classmethod
    def normalize(cls, slice_type: int) -> int:
        """Normalize slice type to 0-4 range."""
        if slice_type > 4:
            return slice_type - 5
        return slice_type

    @classmethod
    def is_i_slice(cls, slice_type: int) -> bool:
        """Check if slice is I or SI type."""
        normalized = cls.normalize(slice_type)
        return normalized in (cls.I, cls.SI)

    @classmethod
    def is_p_slice(cls, slice_type: int) -> bool:
        """Check if slice is P or SP type."""
        normalized = cls.normalize(slice_type)
        return normalized in (cls.P, cls.SP)

    @classmethod
    def is_b_slice(cls, slice_type: int) -> bool:
        """Check if slice is B type."""
        normalized = cls.normalize(slice_type)
        return normalized == cls.B


@dataclass
class RefPicListModification:
    """Reference picture list modification data.

    H.264 Spec: Section 7.3.3.1
    """
    modification_of_pic_nums_idc: List[int] = field(default_factory=list)
    abs_diff_pic_num_minus1: List[int] = field(default_factory=list)
    long_term_pic_num: List[int] = field(default_factory=list)


@dataclass
class DecRefPicMarking:
    """Decoded reference picture marking data.

    H.264 Spec: Section 7.3.3.3
    """
    no_output_of_prior_pics_flag: bool = False
    long_term_reference_flag: bool = False
    adaptive_ref_pic_marking_mode_flag: bool = False

    # For adaptive marking
    memory_management_control_operations: List[int] = field(default_factory=list)
    difference_of_pic_nums_minus1: List[int] = field(default_factory=list)
    long_term_pic_num: List[int] = field(default_factory=list)
    long_term_frame_idx: List[int] = field(default_factory=list)
    max_long_term_frame_idx_plus1: List[int] = field(default_factory=list)


@dataclass
class SliceHeader:
    """Slice header data structure.

    Contains all decoded slice header fields needed for decoding.

    Attributes:
        first_mb_in_slice: Address of first macroblock in slice
        slice_type: Type of slice (I, P, B, SI, SP)
        pic_parameter_set_id: PPS reference
        frame_num: Frame number for reference management
        slice_qp_delta: QP adjustment for this slice
        header_bit_size: Size of slice header in bits (for locating slice data)
    """
    # Basic slice identification
    first_mb_in_slice: int = 0
    slice_type: int = 0
    pic_parameter_set_id: int = 0

    # Header size for locating slice data
    header_bit_size: int = 0

    # Colour plane (for separate_colour_plane_flag)
    colour_plane_id: int = 0

    # Frame/field identification
    frame_num: int = 0
    field_pic_flag: bool = False
    bottom_field_flag: bool = False

    # IDR-specific
    idr_pic_id: int = 0

    # Picture order count
    pic_order_cnt_lsb: int = 0
    delta_pic_order_cnt_bottom: int = 0
    delta_pic_order_cnt: List[int] = field(default_factory=list)

    # Redundant picture
    redundant_pic_cnt: int = 0

    # B-slice specific
    direct_spatial_mv_pred_flag: bool = False

    # Reference picture management
    num_ref_idx_active_override_flag: bool = False
    num_ref_idx_l0_active_minus1: int = 0
    num_ref_idx_l1_active_minus1: int = 0

    # Reference picture list modification
    ref_pic_list_modification_l0: Optional[RefPicListModification] = None
    ref_pic_list_modification_l1: Optional[RefPicListModification] = None

    # Prediction weights (for weighted prediction)
    weighted_pred_table: Optional['WeightTable'] = None

    # Decoded reference picture marking
    dec_ref_pic_marking: Optional[DecRefPicMarking] = None

    # CABAC (not used in baseline)
    cabac_init_idc: int = 0

    # Quantization
    slice_qp_delta: int = 0

    # SP/SI slices
    sp_for_switch_flag: bool = False
    slice_qs_delta: int = 0

    # Deblocking filter
    disable_deblocking_filter_idc: int = 0
    slice_alpha_c0_offset_div2: int = 0
    slice_beta_offset_div2: int = 0

    # Slice group (for FMO)
    slice_group_change_cycle: int = 0

    # Derived values (set during parsing)
    _sps: Optional[SPS] = field(default=None, repr=False)
    _pps: Optional[PPS] = field(default=None, repr=False)
    _is_idr: bool = False

    @property
    def slice_type_name(self) -> str:
        """Human-readable slice type."""
        names = {0: "P", 1: "B", 2: "I", 3: "SP", 4: "SI",
                 5: "P", 6: "B", 7: "I", 8: "SP", 9: "SI"}
        return names.get(self.slice_type, f"Unknown({self.slice_type})")

    @property
    def is_i_slice(self) -> bool:
        """Check if this is an I or SI slice."""
        return SliceType.is_i_slice(self.slice_type)

    @property
    def is_p_slice(self) -> bool:
        """Check if this is a P or SP slice."""
        return SliceType.is_p_slice(self.slice_type)

    @property
    def is_b_slice(self) -> bool:
        """Check if this is a B slice."""
        return SliceType.is_b_slice(self.slice_type)

    @property
    def is_reference(self) -> bool:
        """Check if this slice is used as reference."""
        # Based on nal_ref_idc (needs to be set externally)
        return True  # Default assumption

    @property
    def slice_qp(self) -> int:
        """Calculate actual slice QP."""
        if self._pps:
            return 26 + self._pps.pic_init_qp_minus26 + self.slice_qp_delta
        return 26 + self.slice_qp_delta

    @property
    def num_ref_idx_l0_active(self) -> int:
        """Active L0 reference count."""
        return self.num_ref_idx_l0_active_minus1 + 1

    @property
    def num_ref_idx_l1_active(self) -> int:
        """Active L1 reference count."""
        return self.num_ref_idx_l1_active_minus1 + 1

    @property
    def deblocking_enabled(self) -> bool:
        """Check if deblocking filter is enabled."""
        return self.disable_deblocking_filter_idc != 1

    @property
    def alpha_offset(self) -> int:
        """Deblocking alpha offset."""
        return self.slice_alpha_c0_offset_div2 * 2

    @property
    def beta_offset(self) -> int:
        """Deblocking beta offset."""
        return self.slice_beta_offset_div2 * 2

    def __repr__(self) -> str:
        return (
            f"SliceHeader(type={self.slice_type_name}, "
            f"first_mb={self.first_mb_in_slice}, "
            f"frame_num={self.frame_num}, "
            f"qp_delta={self.slice_qp_delta})"
        )


def _parse_ref_pic_list_modification(
    reader: BitReader,
    slice_type: int
) -> Tuple[Optional[RefPicListModification], Optional[RefPicListModification]]:
    """Parse reference picture list modification.

    H.264 Spec: Section 7.3.3.1
    """
    mod_l0 = None
    mod_l1 = None

    normalized_type = SliceType.normalize(slice_type)

    # L0 modification (for P, B, SP slices)
    if normalized_type in (SliceType.P, SliceType.B, SliceType.SP):
        ref_pic_list_modification_flag_l0 = reader.read_flag()
        if ref_pic_list_modification_flag_l0:
            mod_l0 = RefPicListModification()
            while True:
                idc = reader.read_ue()
                if idc == 3:  # End of modification
                    break
                mod_l0.modification_of_pic_nums_idc.append(idc)
                if idc in (0, 1):
                    mod_l0.abs_diff_pic_num_minus1.append(reader.read_ue())
                elif idc == 2:
                    mod_l0.long_term_pic_num.append(reader.read_ue())

    # L1 modification (for B slices)
    if normalized_type == SliceType.B:
        ref_pic_list_modification_flag_l1 = reader.read_flag()
        if ref_pic_list_modification_flag_l1:
            mod_l1 = RefPicListModification()
            while True:
                idc = reader.read_ue()
                if idc == 3:
                    break
                mod_l1.modification_of_pic_nums_idc.append(idc)
                if idc in (0, 1):
                    mod_l1.abs_diff_pic_num_minus1.append(reader.read_ue())
                elif idc == 2:
                    mod_l1.long_term_pic_num.append(reader.read_ue())

    return mod_l0, mod_l1


def _parse_pred_weight_table(
    reader: BitReader,
    slice_type: int,
    num_ref_idx_l0_active: int,
    num_ref_idx_l1_active: int,
    chroma_format_idc: int
) -> None:
    """Parse prediction weight table (skip for baseline).

    H.264 Spec: Section 7.3.3.2
    """
    luma_log2_weight_denom = reader.read_ue()

    if chroma_format_idc != 0:
        chroma_log2_weight_denom = reader.read_ue()

    # L0 weights
    for _ in range(num_ref_idx_l0_active):
        luma_weight_l0_flag = reader.read_flag()
        if luma_weight_l0_flag:
            reader.read_se()  # luma_weight_l0
            reader.read_se()  # luma_offset_l0
        if chroma_format_idc != 0:
            chroma_weight_l0_flag = reader.read_flag()
            if chroma_weight_l0_flag:
                for _ in range(2):  # Cb, Cr
                    reader.read_se()  # chroma_weight_l0
                    reader.read_se()  # chroma_offset_l0

    # L1 weights (B slices only)
    if SliceType.normalize(slice_type) == SliceType.B:
        for _ in range(num_ref_idx_l1_active):
            luma_weight_l1_flag = reader.read_flag()
            if luma_weight_l1_flag:
                reader.read_se()
                reader.read_se()
            if chroma_format_idc != 0:
                chroma_weight_l1_flag = reader.read_flag()
                if chroma_weight_l1_flag:
                    for _ in range(2):
                        reader.read_se()
                        reader.read_se()


def _parse_dec_ref_pic_marking(
    reader: BitReader,
    is_idr: bool
) -> DecRefPicMarking:
    """Parse decoded reference picture marking.

    H.264 Spec: Section 7.3.3.3
    """
    marking = DecRefPicMarking()

    if is_idr:
        marking.no_output_of_prior_pics_flag = reader.read_flag()
        marking.long_term_reference_flag = reader.read_flag()
    else:
        marking.adaptive_ref_pic_marking_mode_flag = reader.read_flag()
        if marking.adaptive_ref_pic_marking_mode_flag:
            while True:
                mmco = reader.read_ue()
                if mmco == 0:
                    break
                marking.memory_management_control_operations.append(mmco)

                if mmco in (1, 3):
                    marking.difference_of_pic_nums_minus1.append(reader.read_ue())
                if mmco == 2:
                    marking.long_term_pic_num.append(reader.read_ue())
                if mmco in (3, 6):
                    marking.long_term_frame_idx.append(reader.read_ue())
                if mmco == 4:
                    marking.max_long_term_frame_idx_plus1.append(reader.read_ue())

    return marking


def parse_slice_header(
    rbsp: bytes,
    sps: SPS,
    pps: PPS,
    nal_unit_type: int,
    nal_ref_idc: int
) -> SliceHeader:
    """Parse slice header from RBSP bytes.

    Args:
        rbsp: Raw Byte Sequence Payload (slice data after NAL header)
        sps: Sequence Parameter Set for this slice
        pps: Picture Parameter Set for this slice
        nal_unit_type: NAL unit type (1=non-IDR, 5=IDR)
        nal_ref_idc: NAL reference indicator

    Returns:
        Parsed SliceHeader object

    H.264 Spec: Section 7.3.3
    """
    if len(rbsp) < 1:
        raise ValueError("Slice header RBSP too short")

    reader = BitReader(rbsp)
    header = SliceHeader()
    header._sps = sps
    header._pps = pps

    is_idr = nal_unit_type == 5
    header._is_idr = is_idr

    # Basic slice identification
    header.first_mb_in_slice = reader.read_ue()
    header.slice_type = reader.read_ue()
    header.pic_parameter_set_id = reader.read_ue()

    logger.debug(
        f"Parsing slice header: type={header.slice_type_name}, "
        f"first_mb={header.first_mb_in_slice}, pps_id={header.pic_parameter_set_id}"
    )

    # Colour plane (for High 4:4:4 with separate_colour_plane_flag)
    if sps.separate_colour_plane_flag:
        header.colour_plane_id = reader.read_bits(2)

    # Frame number
    frame_num_bits = sps.log2_max_frame_num_minus4 + 4
    header.frame_num = reader.read_bits(frame_num_bits)

    # Field/frame identification
    if not sps.frame_mbs_only_flag:
        header.field_pic_flag = reader.read_flag()
        if header.field_pic_flag:
            header.bottom_field_flag = reader.read_flag()

    # IDR picture ID
    if is_idr:
        header.idr_pic_id = reader.read_ue()

    # Picture order count
    if sps.pic_order_cnt_type == 0:
        poc_lsb_bits = sps.log2_max_pic_order_cnt_lsb_minus4 + 4
        header.pic_order_cnt_lsb = reader.read_bits(poc_lsb_bits)

        if pps.bottom_field_pic_order_in_frame_present_flag and not header.field_pic_flag:
            header.delta_pic_order_cnt_bottom = reader.read_se()

    elif sps.pic_order_cnt_type == 1:
        if not sps.delta_pic_order_always_zero_flag:
            header.delta_pic_order_cnt = [reader.read_se()]
            if pps.bottom_field_pic_order_in_frame_present_flag and not header.field_pic_flag:
                header.delta_pic_order_cnt.append(reader.read_se())

    # Redundant picture
    if pps.redundant_pic_cnt_present_flag:
        header.redundant_pic_cnt = reader.read_ue()

    normalized_type = SliceType.normalize(header.slice_type)

    # B slice: direct spatial prediction flag
    if normalized_type == SliceType.B:
        header.direct_spatial_mv_pred_flag = reader.read_flag()

    # Reference index override
    if normalized_type in (SliceType.P, SliceType.SP, SliceType.B):
        header.num_ref_idx_active_override_flag = reader.read_flag()

        if header.num_ref_idx_active_override_flag:
            header.num_ref_idx_l0_active_minus1 = reader.read_ue()
            if normalized_type == SliceType.B:
                header.num_ref_idx_l1_active_minus1 = reader.read_ue()
        else:
            # Use PPS defaults
            header.num_ref_idx_l0_active_minus1 = pps.num_ref_idx_l0_default_active_minus1
            header.num_ref_idx_l1_active_minus1 = pps.num_ref_idx_l1_default_active_minus1

    # Reference picture list modification
    if normalized_type != SliceType.I and normalized_type != SliceType.SI:
        mod_l0, mod_l1 = _parse_ref_pic_list_modification(reader, header.slice_type)
        header.ref_pic_list_modification_l0 = mod_l0
        header.ref_pic_list_modification_l1 = mod_l1

    # Prediction weight table
    if ((pps.weighted_pred_flag and normalized_type in (SliceType.P, SliceType.SP)) or
        (pps.weighted_bipred_idc == 1 and normalized_type == SliceType.B)):
        _parse_pred_weight_table(
            reader, header.slice_type,
            header.num_ref_idx_l0_active,
            header.num_ref_idx_l1_active,
            sps.chroma_format_idc
        )

    # Decoded reference picture marking
    if nal_ref_idc != 0:
        header.dec_ref_pic_marking = _parse_dec_ref_pic_marking(reader, is_idr)

    # CABAC init (not for baseline)
    if pps.entropy_coding_mode_flag and not header.is_i_slice:
        header.cabac_init_idc = reader.read_ue()

    # Slice QP
    header.slice_qp_delta = reader.read_se()

    logger.debug(f"Slice QP delta: {header.slice_qp_delta}, actual QP: {header.slice_qp}")

    # SP/SI slice QS
    if normalized_type in (SliceType.SP, SliceType.SI):
        if normalized_type == SliceType.SP:
            header.sp_for_switch_flag = reader.read_flag()
        header.slice_qs_delta = reader.read_se()

    # Deblocking filter
    if pps.deblocking_filter_control_present_flag:
        header.disable_deblocking_filter_idc = reader.read_ue()
        if header.disable_deblocking_filter_idc != 1:
            header.slice_alpha_c0_offset_div2 = reader.read_se()
            header.slice_beta_offset_div2 = reader.read_se()

    # Slice group change cycle (for FMO)
    if pps.num_slice_groups_minus1 > 0 and pps.slice_group_map_type in (3, 4, 5):
        # Calculate bits needed
        pic_size_in_map_units = sps.pic_width_in_mbs * sps.pic_height_in_map_units
        # This formula is complex, simplified for basic support
        slice_group_change_cycle_bits = (
            (pic_size_in_map_units - 1).bit_length() + 1
        )
        header.slice_group_change_cycle = reader.read_bits(slice_group_change_cycle_bits)

    # Store header size for locating slice data
    header.header_bit_size = reader.position

    logger.info(f"Parsed {header}")
    return header
