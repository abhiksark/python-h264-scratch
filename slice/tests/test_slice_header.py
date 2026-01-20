# h264/slice/tests/test_slice_header.py
"""Tests for slice header parsing."""

import pytest

from bitstream import BitWriter, BITSTRING_AVAILABLE
from parameters import SPS, PPS
from slice.slice_header import (
    SliceHeader,
    SliceType,
    RefPicListModification,
    DecRefPicMarking,
    parse_slice_header,
)

# Skip all tests if bitstring not available
pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestSliceType:
    """Tests for SliceType enum and helpers."""

    def test_normalize_low_values(self):
        """Values 0-4 stay unchanged."""
        assert SliceType.normalize(0) == 0
        assert SliceType.normalize(2) == 2
        assert SliceType.normalize(4) == 4

    def test_normalize_high_values(self):
        """Values 5-9 map to 0-4."""
        assert SliceType.normalize(5) == 0
        assert SliceType.normalize(7) == 2
        assert SliceType.normalize(9) == 4

    def test_is_i_slice(self):
        """Detect I and SI slices."""
        assert SliceType.is_i_slice(2) is True
        assert SliceType.is_i_slice(4) is True
        assert SliceType.is_i_slice(7) is True  # I_ALL
        assert SliceType.is_i_slice(9) is True  # SI_ALL
        assert SliceType.is_i_slice(0) is False  # P
        assert SliceType.is_i_slice(1) is False  # B

    def test_is_p_slice(self):
        """Detect P and SP slices."""
        assert SliceType.is_p_slice(0) is True
        assert SliceType.is_p_slice(3) is True
        assert SliceType.is_p_slice(5) is True  # P_ALL
        assert SliceType.is_p_slice(8) is True  # SP_ALL
        assert SliceType.is_p_slice(2) is False  # I
        assert SliceType.is_p_slice(1) is False  # B

    def test_is_b_slice(self):
        """Detect B slices."""
        assert SliceType.is_b_slice(1) is True
        assert SliceType.is_b_slice(6) is True  # B_ALL
        assert SliceType.is_b_slice(0) is False  # P
        assert SliceType.is_b_slice(2) is False  # I


class TestSliceHeaderDataclass:
    """Tests for SliceHeader dataclass properties."""

    def test_slice_type_name(self):
        """Slice type name lookup."""
        assert SliceHeader(slice_type=0).slice_type_name == "P"
        assert SliceHeader(slice_type=1).slice_type_name == "B"
        assert SliceHeader(slice_type=2).slice_type_name == "I"
        assert SliceHeader(slice_type=7).slice_type_name == "I"

    def test_is_i_slice_property(self):
        """Check is_i_slice property."""
        assert SliceHeader(slice_type=2).is_i_slice is True
        assert SliceHeader(slice_type=7).is_i_slice is True
        assert SliceHeader(slice_type=0).is_i_slice is False

    def test_is_p_slice_property(self):
        """Check is_p_slice property."""
        assert SliceHeader(slice_type=0).is_p_slice is True
        assert SliceHeader(slice_type=5).is_p_slice is True
        assert SliceHeader(slice_type=2).is_p_slice is False

    def test_is_b_slice_property(self):
        """Check is_b_slice property."""
        assert SliceHeader(slice_type=1).is_b_slice is True
        assert SliceHeader(slice_type=6).is_b_slice is True
        assert SliceHeader(slice_type=0).is_b_slice is False

    def test_slice_qp_without_pps(self):
        """Slice QP calculation without PPS."""
        header = SliceHeader(slice_qp_delta=0)
        assert header.slice_qp == 26

        header = SliceHeader(slice_qp_delta=5)
        assert header.slice_qp == 31

        header = SliceHeader(slice_qp_delta=-10)
        assert header.slice_qp == 16

    def test_slice_qp_with_pps(self):
        """Slice QP calculation with PPS."""
        pps = PPS(pic_init_qp_minus26=4)  # init_qp = 30
        header = SliceHeader(slice_qp_delta=2)
        header._pps = pps
        assert header.slice_qp == 32  # 26 + 4 + 2

    def test_num_ref_idx_active(self):
        """Reference count calculation."""
        header = SliceHeader(
            num_ref_idx_l0_active_minus1=2,
            num_ref_idx_l1_active_minus1=1
        )
        assert header.num_ref_idx_l0_active == 3
        assert header.num_ref_idx_l1_active == 2

    def test_deblocking_enabled(self):
        """Deblocking enabled check."""
        assert SliceHeader(disable_deblocking_filter_idc=0).deblocking_enabled is True
        assert SliceHeader(disable_deblocking_filter_idc=1).deblocking_enabled is False
        assert SliceHeader(disable_deblocking_filter_idc=2).deblocking_enabled is True

    def test_deblocking_offsets(self):
        """Deblocking offset calculation."""
        header = SliceHeader(
            slice_alpha_c0_offset_div2=2,
            slice_beta_offset_div2=-1
        )
        assert header.alpha_offset == 4
        assert header.beta_offset == -2

    def test_repr(self):
        """String representation."""
        header = SliceHeader(
            slice_type=2,
            first_mb_in_slice=0,
            frame_num=5,
            slice_qp_delta=-2
        )
        s = repr(header)
        assert "type=I" in s
        assert "first_mb=0" in s
        assert "frame_num=5" in s
        assert "qp_delta=-2" in s


class TestParseSliceHeader:
    """Tests for slice header parsing."""

    def _create_default_sps(self) -> SPS:
        """Create default SPS for testing."""
        return SPS(
            profile_idc=66,
            level_idc=30,
            log2_max_frame_num_minus4=0,  # max_frame_num = 16
            pic_order_cnt_type=2,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=2,
            frame_mbs_only_flag=True,
        )

    def _create_default_pps(self) -> PPS:
        """Create default PPS for testing."""
        return PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            num_ref_idx_l0_default_active_minus1=0,
            num_ref_idx_l1_default_active_minus1=0,
            pic_init_qp_minus26=0,
            deblocking_filter_control_present_flag=True,
        )

    def _create_i_slice_rbsp(
        self,
        first_mb=0,
        frame_num=0,
        qp_delta=0,
        deblocking_idc=0,
        alpha_offset=0,
        beta_offset=0,
        is_idr=False,
        idr_pic_id=0,
    ) -> bytes:
        """Create minimal I-slice header RBSP."""
        writer = BitWriter()

        writer.write_ue(first_mb)  # first_mb_in_slice
        writer.write_ue(2)  # slice_type = I
        writer.write_ue(0)  # pic_parameter_set_id

        # frame_num (4 bits for log2_max_frame_num_minus4=0)
        writer.write_bits(frame_num, 4)

        # IDR-specific
        if is_idr:
            writer.write_ue(idr_pic_id)  # idr_pic_id

        # No POC for type 2

        # dec_ref_pic_marking for reference slices
        if is_idr:
            writer.write_flag(False)  # no_output_of_prior_pics_flag
            writer.write_flag(False)  # long_term_reference_flag
        else:
            writer.write_flag(False)  # adaptive_ref_pic_marking_mode_flag

        # slice_qp_delta
        writer.write_se(qp_delta)

        # Deblocking
        writer.write_ue(deblocking_idc)
        if deblocking_idc != 1:
            writer.write_se(alpha_offset)
            writer.write_se(beta_offset)

        return writer.to_bytes()

    def test_parse_minimal_i_slice(self):
        """Parse minimal I-slice header."""
        sps = self._create_default_sps()
        pps = self._create_default_pps()
        rbsp = self._create_i_slice_rbsp()

        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=1, nal_ref_idc=3)

        assert header.first_mb_in_slice == 0
        assert header.slice_type == 2
        assert header.is_i_slice is True
        assert header.frame_num == 0
        assert header.slice_qp_delta == 0
        assert header.slice_qp == 26

    def test_parse_i_slice_different_first_mb(self):
        """Parse I-slice with non-zero first MB."""
        sps = self._create_default_sps()
        pps = self._create_default_pps()
        rbsp = self._create_i_slice_rbsp(first_mb=5)

        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=1, nal_ref_idc=3)

        assert header.first_mb_in_slice == 5

    def test_parse_i_slice_with_qp_delta(self):
        """Parse I-slice with QP delta."""
        sps = self._create_default_sps()
        pps = self._create_default_pps()
        rbsp = self._create_i_slice_rbsp(qp_delta=-5)

        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=1, nal_ref_idc=3)

        assert header.slice_qp_delta == -5
        assert header.slice_qp == 21

    def test_parse_i_slice_with_frame_num(self):
        """Parse I-slice with specific frame number."""
        sps = self._create_default_sps()
        pps = self._create_default_pps()
        rbsp = self._create_i_slice_rbsp(frame_num=7)

        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=1, nal_ref_idc=3)

        assert header.frame_num == 7

    def test_parse_idr_slice(self):
        """Parse IDR slice header."""
        sps = self._create_default_sps()
        pps = self._create_default_pps()
        rbsp = self._create_i_slice_rbsp(is_idr=True, idr_pic_id=3)

        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=5, nal_ref_idc=3)

        assert header.idr_pic_id == 3
        assert header._is_idr is True

    def test_parse_deblocking_disabled(self):
        """Parse slice with deblocking disabled."""
        sps = self._create_default_sps()
        pps = self._create_default_pps()
        rbsp = self._create_i_slice_rbsp(deblocking_idc=1)

        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=1, nal_ref_idc=3)

        assert header.disable_deblocking_filter_idc == 1
        assert header.deblocking_enabled is False

    def test_parse_deblocking_with_offsets(self):
        """Parse slice with deblocking offsets."""
        sps = self._create_default_sps()
        pps = self._create_default_pps()
        rbsp = self._create_i_slice_rbsp(
            deblocking_idc=0,
            alpha_offset=2,
            beta_offset=-1
        )

        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=1, nal_ref_idc=3)

        assert header.slice_alpha_c0_offset_div2 == 2
        assert header.slice_beta_offset_div2 == -1
        assert header.alpha_offset == 4
        assert header.beta_offset == -2

    def test_parse_with_pps_qp_offset(self):
        """Parse slice with PPS QP offset."""
        sps = self._create_default_sps()
        pps = PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            pic_init_qp_minus26=4,  # init_qp = 30
            deblocking_filter_control_present_flag=True,
        )
        rbsp = self._create_i_slice_rbsp(qp_delta=2)

        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=1, nal_ref_idc=3)

        assert header.slice_qp == 32  # 26 + 4 + 2

    def test_parse_too_short_raises(self):
        """Parsing too-short RBSP raises error."""
        sps = self._create_default_sps()
        pps = self._create_default_pps()

        with pytest.raises(ValueError):
            parse_slice_header(b'', sps, pps, nal_unit_type=1, nal_ref_idc=3)


class TestParseSliceHeaderPOC:
    """Tests for POC parsing in slice header."""

    def test_parse_poc_type_0(self):
        """Parse slice with POC type 0."""
        sps = SPS(
            profile_idc=66,
            level_idc=30,
            log2_max_frame_num_minus4=0,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,  # max_poc_lsb = 16
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=2,
            frame_mbs_only_flag=True,
        )
        pps = PPS(
            pic_parameter_set_id=0,
            deblocking_filter_control_present_flag=True,
        )

        writer = BitWriter()
        writer.write_ue(0)  # first_mb
        writer.write_ue(2)  # I slice
        writer.write_ue(0)  # pps_id
        writer.write_bits(0, 4)  # frame_num
        writer.write_bits(8, 4)  # pic_order_cnt_lsb = 8
        writer.write_flag(False)  # adaptive_ref_pic_marking
        writer.write_se(0)  # qp_delta
        writer.write_ue(0)  # deblocking
        writer.write_se(0)  # alpha
        writer.write_se(0)  # beta

        rbsp = writer.to_bytes()
        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=1, nal_ref_idc=3)

        assert header.pic_order_cnt_lsb == 8


class TestParseSliceHeaderPSlice:
    """Tests for P-slice header parsing."""

    def test_parse_p_slice_basic(self):
        """Parse basic P-slice header."""
        sps = SPS(
            profile_idc=66,
            level_idc=30,
            log2_max_frame_num_minus4=0,
            pic_order_cnt_type=2,
            pic_width_in_mbs_minus1=3,
            pic_height_in_map_units_minus1=2,
            frame_mbs_only_flag=True,
        )
        pps = PPS(
            pic_parameter_set_id=0,
            num_ref_idx_l0_default_active_minus1=0,
            deblocking_filter_control_present_flag=True,
        )

        writer = BitWriter()
        writer.write_ue(0)  # first_mb
        writer.write_ue(0)  # P slice
        writer.write_ue(0)  # pps_id
        writer.write_bits(1, 4)  # frame_num = 1
        writer.write_flag(False)  # num_ref_idx_active_override
        writer.write_flag(False)  # ref_pic_list_modification_flag_l0
        writer.write_flag(False)  # adaptive_ref_pic_marking
        writer.write_se(-2)  # qp_delta
        writer.write_ue(0)  # deblocking
        writer.write_se(0)
        writer.write_se(0)

        rbsp = writer.to_bytes()
        header = parse_slice_header(rbsp, sps, pps, nal_unit_type=1, nal_ref_idc=3)

        assert header.slice_type == 0
        assert header.is_p_slice is True
        assert header.frame_num == 1
        assert header.slice_qp_delta == -2
        assert header.num_ref_idx_l0_active == 1  # From PPS default


class TestDecRefPicMarking:
    """Tests for decoded reference picture marking."""

    def test_dec_ref_pic_marking_idr(self):
        """Parse dec_ref_pic_marking for IDR slice."""
        marking = DecRefPicMarking(
            no_output_of_prior_pics_flag=True,
            long_term_reference_flag=False
        )
        assert marking.no_output_of_prior_pics_flag is True
        assert marking.long_term_reference_flag is False

    def test_dec_ref_pic_marking_non_idr(self):
        """Parse dec_ref_pic_marking for non-IDR slice."""
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[1, 0]
        )
        assert marking.adaptive_ref_pic_marking_mode_flag is True
        assert len(marking.memory_management_control_operations) == 2
