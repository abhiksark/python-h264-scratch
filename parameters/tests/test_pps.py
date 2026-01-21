# h264/parameters/tests/test_pps.py
"""Tests for PPS (Picture Parameter Set) parsing."""

import pytest

from bitstream import BitWriter, BITSTRING_AVAILABLE
from parameters.pps import PPS, parse_pps

# Skip all tests if bitstring not available
pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestPPSDataclass:
    """Tests for PPS dataclass properties."""

    def test_entropy_coding_mode_cavlc(self):
        """CAVLC entropy coding mode."""
        pps = PPS(entropy_coding_mode_flag=False)
        assert pps.entropy_coding_mode == "CAVLC"

    def test_entropy_coding_mode_cabac(self):
        """CABAC entropy coding mode."""
        pps = PPS(entropy_coding_mode_flag=True)
        assert pps.entropy_coding_mode == "CABAC"

    def test_num_ref_idx_default_active(self):
        """Reference index count calculation."""
        pps = PPS(
            num_ref_idx_l0_default_active_minus1=2,
            num_ref_idx_l1_default_active_minus1=1
        )
        assert pps.num_ref_idx_l0_default_active == 3
        assert pps.num_ref_idx_l1_default_active == 2

    def test_pic_init_qp(self):
        """Initial QP calculation."""
        pps = PPS(pic_init_qp_minus26=0)
        assert pps.pic_init_qp == 26

        pps = PPS(pic_init_qp_minus26=-10)
        assert pps.pic_init_qp == 16

        pps = PPS(pic_init_qp_minus26=10)
        assert pps.pic_init_qp == 36

    def test_weighted_bipred_mode(self):
        """Weighted bipred mode names."""
        assert PPS(weighted_bipred_idc=0).weighted_bipred_mode == "Default"
        assert PPS(weighted_bipred_idc=1).weighted_bipred_mode == "Explicit"
        assert PPS(weighted_bipred_idc=2).weighted_bipred_mode == "Implicit"

    def test_repr(self):
        """String representation."""
        pps = PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            pic_init_qp_minus26=0
        )
        s = repr(pps)
        assert "id=0" in s
        assert "sps_id=0" in s
        assert "CAVLC" in s
        assert "init_qp=26" in s


class TestParsePPS:
    """Tests for PPS parsing."""

    def _create_baseline_pps_rbsp(
        self,
        pps_id=0,
        sps_id=0,
        init_qp_offset=0,
        chroma_qp_offset=0,
        num_ref_l0=0,
    ) -> bytes:
        """Create a minimal Baseline profile PPS RBSP for testing."""
        writer = BitWriter()

        # Basic identifiers
        writer.write_ue(pps_id)  # pic_parameter_set_id
        writer.write_ue(sps_id)  # seq_parameter_set_id

        # Entropy coding (CAVLC for baseline)
        writer.write_flag(False)  # entropy_coding_mode_flag

        # Field coding
        writer.write_flag(False)  # bottom_field_pic_order_in_frame_present_flag

        # No FMO
        writer.write_ue(0)  # num_slice_groups_minus1

        # Reference defaults
        writer.write_ue(num_ref_l0)  # num_ref_idx_l0_default_active_minus1
        writer.write_ue(0)  # num_ref_idx_l1_default_active_minus1

        # No weighted prediction
        writer.write_flag(False)  # weighted_pred_flag
        writer.write_bits(0, 2)  # weighted_bipred_idc

        # Quantization
        writer.write_se(init_qp_offset)  # pic_init_qp_minus26
        writer.write_se(0)  # pic_init_qs_minus26
        writer.write_se(chroma_qp_offset)  # chroma_qp_index_offset

        # Flags
        writer.write_flag(True)  # deblocking_filter_control_present_flag
        writer.write_flag(False)  # constrained_intra_pred_flag
        writer.write_flag(False)  # redundant_pic_cnt_present_flag

        return writer.to_bytes()

    def test_parse_minimal_baseline_pps(self):
        """Parse minimal Baseline profile PPS."""
        rbsp = self._create_baseline_pps_rbsp()
        pps = parse_pps(rbsp)

        assert pps.pic_parameter_set_id == 0
        assert pps.seq_parameter_set_id == 0
        assert pps.entropy_coding_mode_flag is False
        assert pps.entropy_coding_mode == "CAVLC"
        assert pps.pic_init_qp == 26
        assert pps.deblocking_filter_control_present_flag is True

    def test_parse_different_pps_id(self):
        """Parse PPS with different ID."""
        rbsp = self._create_baseline_pps_rbsp(pps_id=5, sps_id=2)
        pps = parse_pps(rbsp)

        assert pps.pic_parameter_set_id == 5
        assert pps.seq_parameter_set_id == 2

    def test_parse_different_init_qp(self):
        """Parse PPS with different initial QP."""
        rbsp = self._create_baseline_pps_rbsp(init_qp_offset=-10)
        pps = parse_pps(rbsp)

        assert pps.pic_init_qp_minus26 == -10
        assert pps.pic_init_qp == 16

        rbsp = self._create_baseline_pps_rbsp(init_qp_offset=15)
        pps = parse_pps(rbsp)

        assert pps.pic_init_qp_minus26 == 15
        assert pps.pic_init_qp == 41

    def test_parse_chroma_qp_offset(self):
        """Parse PPS with chroma QP offset."""
        rbsp = self._create_baseline_pps_rbsp(chroma_qp_offset=-5)
        pps = parse_pps(rbsp)

        assert pps.chroma_qp_index_offset == -5

        rbsp = self._create_baseline_pps_rbsp(chroma_qp_offset=5)
        pps = parse_pps(rbsp)

        assert pps.chroma_qp_index_offset == 5

    def test_parse_different_ref_count(self):
        """Parse PPS with different reference count."""
        rbsp = self._create_baseline_pps_rbsp(num_ref_l0=3)
        pps = parse_pps(rbsp)

        assert pps.num_ref_idx_l0_default_active_minus1 == 3
        assert pps.num_ref_idx_l0_default_active == 4

    def test_parse_cabac_pps(self):
        """Parse PPS with CABAC entropy coding."""
        writer = BitWriter()

        writer.write_ue(0)  # pps_id
        writer.write_ue(0)  # sps_id
        writer.write_flag(True)  # entropy_coding_mode_flag (CABAC)
        writer.write_flag(False)  # bottom_field
        writer.write_ue(0)  # num_slice_groups
        writer.write_ue(0)  # l0 ref
        writer.write_ue(0)  # l1 ref
        writer.write_flag(False)  # weighted_pred
        writer.write_bits(0, 2)  # weighted_bipred_idc
        writer.write_se(0)  # init_qp
        writer.write_se(0)  # init_qs
        writer.write_se(0)  # chroma_offset
        writer.write_flag(True)  # deblocking
        writer.write_flag(False)  # constrained_intra
        writer.write_flag(False)  # redundant

        rbsp = writer.to_bytes()
        pps = parse_pps(rbsp)

        assert pps.entropy_coding_mode_flag is True
        assert pps.entropy_coding_mode == "CABAC"

    def test_parse_weighted_prediction(self):
        """Parse PPS with weighted prediction enabled."""
        writer = BitWriter()

        writer.write_ue(0)  # pps_id
        writer.write_ue(0)  # sps_id
        writer.write_flag(False)  # entropy_coding
        writer.write_flag(False)  # bottom_field
        writer.write_ue(0)  # num_slice_groups
        writer.write_ue(0)  # l0 ref
        writer.write_ue(0)  # l1 ref
        writer.write_flag(True)  # weighted_pred_flag
        writer.write_bits(2, 2)  # weighted_bipred_idc (implicit)
        writer.write_se(0)  # init_qp
        writer.write_se(0)  # init_qs
        writer.write_se(0)  # chroma_offset
        writer.write_flag(False)  # deblocking
        writer.write_flag(False)  # constrained_intra
        writer.write_flag(False)  # redundant

        rbsp = writer.to_bytes()
        pps = parse_pps(rbsp)

        assert pps.weighted_pred_flag is True
        assert pps.weighted_bipred_idc == 2
        assert pps.weighted_bipred_mode == "Implicit"

    def test_parse_constrained_intra(self):
        """Parse PPS with constrained intra prediction."""
        writer = BitWriter()

        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_flag(False)
        writer.write_flag(False)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_flag(False)
        writer.write_bits(0, 2)
        writer.write_se(0)
        writer.write_se(0)
        writer.write_se(0)
        writer.write_flag(False)
        writer.write_flag(True)  # constrained_intra_pred_flag
        writer.write_flag(False)

        rbsp = writer.to_bytes()
        pps = parse_pps(rbsp)

        assert pps.constrained_intra_pred_flag is True

    def test_parse_too_short_raises(self):
        """Parsing too-short RBSP raises error."""
        with pytest.raises(ValueError):
            parse_pps(b'')


class TestRealWorldPPS:
    """Test with real-world PPS patterns."""

    def test_typical_baseline_pps(self):
        """Parse typical Baseline profile PPS."""
        writer = BitWriter()

        # Typical baseline PPS
        writer.write_ue(0)  # pps_id
        writer.write_ue(0)  # sps_id
        writer.write_flag(False)  # CAVLC
        writer.write_flag(False)
        writer.write_ue(0)  # no FMO
        writer.write_ue(0)  # 1 L0 ref
        writer.write_ue(0)  # 1 L1 ref
        writer.write_flag(False)
        writer.write_bits(0, 2)
        writer.write_se(0)  # init_qp = 26
        writer.write_se(0)
        writer.write_se(0)  # chroma offset = 0
        writer.write_flag(True)  # deblocking in slice header
        writer.write_flag(False)
        writer.write_flag(False)

        rbsp = writer.to_bytes()
        pps = parse_pps(rbsp)

        assert pps.entropy_coding_mode == "CAVLC"
        assert pps.pic_init_qp == 26
        assert pps.deblocking_filter_control_present_flag is True
        assert pps.num_slice_groups_minus1 == 0

    def test_typical_main_pps(self):
        """Parse typical Main profile PPS with CABAC."""
        writer = BitWriter()

        writer.write_ue(0)  # pps_id
        writer.write_ue(0)  # sps_id
        writer.write_flag(True)  # CABAC
        writer.write_flag(False)
        writer.write_ue(0)  # no FMO
        writer.write_ue(2)  # 3 L0 refs
        writer.write_ue(1)  # 2 L1 refs
        writer.write_flag(False)
        writer.write_bits(1, 2)  # explicit weighted bipred
        writer.write_se(-2)  # init_qp = 24
        writer.write_se(0)
        writer.write_se(-2)  # chroma offset = -2
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)

        rbsp = writer.to_bytes()
        pps = parse_pps(rbsp)

        assert pps.entropy_coding_mode == "CABAC"
        assert pps.pic_init_qp == 24
        assert pps.num_ref_idx_l0_default_active == 3
        assert pps.num_ref_idx_l1_default_active == 2
        assert pps.weighted_bipred_idc == 1
        assert pps.chroma_qp_index_offset == -2


class TestPPSScalingListStorage:
    """Tests for scaling list storage in PPS (High profile)."""

    def test_pps_has_scaling_lists_4x4_field(self):
        """PPS should have scaling_lists_4x4 field."""
        pps = PPS()
        assert hasattr(pps, 'scaling_lists_4x4')

    def test_pps_has_scaling_lists_8x8_field(self):
        """PPS should have scaling_lists_8x8 field."""
        pps = PPS()
        assert hasattr(pps, 'scaling_lists_8x8')

    def test_pps_scaling_lists_default_empty(self):
        """PPS scaling lists should default to empty lists."""
        pps = PPS()
        assert pps.scaling_lists_4x4 == []
        assert pps.scaling_lists_8x8 == []
