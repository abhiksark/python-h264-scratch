# h264/decoder/tests/test_profile_constraints.py
"""TDD RED TESTS: H.264 profile and level constraint validation.

Tests for profile-specific feature restrictions and level limits:
- Baseline profile: No CABAC, No B-frames, No weighted prediction
- Main profile: CABAC allowed, B-frames allowed
- High profile: 8x8 transform, scaling lists, High bit depth
- Level constraints: MaxMBPS, MaxFS, MaxDpbMbs, etc.

H.264 Spec References:
- Annex A: Profiles and levels
- Table A-1: Level limits
- Table A-2: Profile-specific features

These tests SHOULD FAIL until the corresponding validation is implemented.
"""

import pytest
import numpy as np


# =============================================================================
# Test: Baseline profile restrictions
# =============================================================================

class TestBaselineProfileRestrictions:
    """Tests for Baseline profile (profile_idc=66) restrictions.

    Baseline is the simplest profile, used for low-latency applications.
    Key restrictions:
    - entropy_coding_mode_flag must be 0 (CAVLC only, no CABAC)
    - No B-slices allowed
    - No weighted prediction
    - No MBAFF
    - No direct_8x8_inference required

    H.264 Spec: Annex A.2.1
    """

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_no_cabac(self):
        """Baseline profile must use CAVLC (no CABAC)."""
        from decoder.profile_validator import validate_baseline_constraints
        from parameters.pps import PPS

        # CABAC is forbidden in Baseline
        pps_cabac = PPS(entropy_coding_mode_flag=True)

        with pytest.raises(ValueError, match="CABAC.*Baseline"):
            validate_baseline_constraints(pps_cabac, profile_idc=66)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_cavlc_allowed(self):
        """Baseline profile allows CAVLC."""
        from decoder.profile_validator import validate_baseline_constraints
        from parameters.pps import PPS

        pps_cavlc = PPS(entropy_coding_mode_flag=False)

        # Should not raise
        validate_baseline_constraints(pps_cavlc, profile_idc=66)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_no_b_slices(self):
        """Baseline profile forbids B-slices."""
        from decoder.profile_validator import validate_slice_type_for_profile
        from slice.slice_header import SliceHeader

        # B-slice (type 1 or 6)
        b_slice = SliceHeader(slice_type=1)

        with pytest.raises(ValueError, match="B.slice.*Baseline"):
            validate_slice_type_for_profile(b_slice, profile_idc=66)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_i_p_slices_allowed(self):
        """Baseline profile allows I and P slices."""
        from decoder.profile_validator import validate_slice_type_for_profile
        from slice.slice_header import SliceHeader

        # I-slice (type 2 or 7)
        i_slice = SliceHeader(slice_type=2)
        validate_slice_type_for_profile(i_slice, profile_idc=66)

        # P-slice (type 0 or 5)
        p_slice = SliceHeader(slice_type=0)
        validate_slice_type_for_profile(p_slice, profile_idc=66)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_no_weighted_pred(self):
        """Baseline profile forbids weighted prediction."""
        from decoder.profile_validator import validate_baseline_constraints
        from parameters.pps import PPS

        pps_weighted = PPS(weighted_pred_flag=True)

        with pytest.raises(ValueError, match="weighted.*Baseline"):
            validate_baseline_constraints(pps_weighted, profile_idc=66)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_no_weighted_bipred(self):
        """Baseline profile forbids weighted bi-prediction."""
        from decoder.profile_validator import validate_baseline_constraints
        from parameters.pps import PPS

        pps_bipred = PPS(weighted_bipred_idc=1)

        with pytest.raises(ValueError, match="weighted.*Baseline"):
            validate_baseline_constraints(pps_bipred, profile_idc=66)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_no_mbaff(self):
        """Baseline profile forbids MBAFF."""
        from decoder.profile_validator import validate_baseline_constraints
        from parameters.sps import SPS

        sps_mbaff = SPS(
            profile_idc=66,
            frame_mbs_only_flag=False,
            mb_adaptive_frame_field_flag=True,
        )

        with pytest.raises(ValueError, match="MBAFF.*Baseline"):
            validate_baseline_constraints(sps_mbaff, profile_idc=66)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_fmo_allowed(self):
        """Baseline profile allows FMO (Flexible Macroblock Ordering)."""
        from decoder.profile_validator import validate_baseline_constraints
        from parameters.pps import PPS

        pps_fmo = PPS(num_slice_groups_minus1=1)

        # Should not raise - FMO is allowed in Baseline
        validate_baseline_constraints(pps_fmo, profile_idc=66)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_aso_allowed(self):
        """Baseline profile allows ASO (Arbitrary Slice Ordering)."""
        from decoder.profile_validator import is_feature_allowed

        result = is_feature_allowed(
            feature="ASO",
            profile_idc=66
        )

        assert result is True

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_baseline_redundant_slices_allowed(self):
        """Baseline profile allows redundant slices."""
        from decoder.profile_validator import is_feature_allowed

        result = is_feature_allowed(
            feature="redundant_slices",
            profile_idc=66
        )

        assert result is True


# =============================================================================
# Test: Main profile constraints
# =============================================================================

class TestMainProfileConstraints:
    """Tests for Main profile (profile_idc=77) constraints.

    Main profile adds:
    - CABAC entropy coding
    - B-slices
    - Weighted prediction
    - Interlaced (MBAFF/PAFF)

    But no:
    - FMO
    - ASO
    - Redundant slices

    H.264 Spec: Annex A.2.2
    """

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_main_cabac_allowed(self):
        """Main profile allows CABAC."""
        from decoder.profile_validator import validate_main_constraints
        from parameters.pps import PPS

        pps_cabac = PPS(entropy_coding_mode_flag=True)

        # Should not raise
        validate_main_constraints(pps_cabac, profile_idc=77)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_main_b_slices_allowed(self):
        """Main profile allows B-slices."""
        from decoder.profile_validator import validate_slice_type_for_profile
        from slice.slice_header import SliceHeader

        b_slice = SliceHeader(slice_type=1)

        # Should not raise
        validate_slice_type_for_profile(b_slice, profile_idc=77)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_main_weighted_pred_allowed(self):
        """Main profile allows weighted prediction."""
        from decoder.profile_validator import validate_main_constraints
        from parameters.pps import PPS

        pps_weighted = PPS(weighted_pred_flag=True)

        # Should not raise
        validate_main_constraints(pps_weighted, profile_idc=77)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_main_mbaff_allowed(self):
        """Main profile allows MBAFF."""
        from decoder.profile_validator import validate_main_constraints
        from parameters.sps import SPS

        sps_mbaff = SPS(
            profile_idc=77,
            frame_mbs_only_flag=False,
            mb_adaptive_frame_field_flag=True,
        )

        # Should not raise
        validate_main_constraints(sps_mbaff, profile_idc=77)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_main_no_fmo(self):
        """Main profile forbids FMO."""
        from decoder.profile_validator import validate_main_constraints
        from parameters.pps import PPS

        pps_fmo = PPS(num_slice_groups_minus1=1)

        with pytest.raises(ValueError, match="FMO.*Main"):
            validate_main_constraints(pps_fmo, profile_idc=77)

    @pytest.mark.xfail(reason="Profile validation not implemented")
    def test_main_no_redundant_slices(self):
        """Main profile forbids redundant slices."""
        from decoder.profile_validator import validate_main_constraints
        from parameters.pps import PPS

        pps_redundant = PPS(redundant_pic_cnt_present_flag=True)

        with pytest.raises(ValueError, match="redundant.*Main"):
            validate_main_constraints(pps_redundant, profile_idc=77)


# =============================================================================
# Test: High profile specific features
# =============================================================================

class TestHighProfileFeatures:
    """Tests for High profile (profile_idc=100) specific features.

    High profile adds:
    - 8x8 transform
    - Custom scaling lists
    - Chroma format (4:2:2, 4:4:4 in subprofiles)
    - More reference frames

    H.264 Spec: Annex A.2.4
    """

    @pytest.mark.xfail(reason="High profile features not implemented")
    def test_high_8x8_transform_allowed(self):
        """High profile allows 8x8 transform."""
        from decoder.profile_validator import is_feature_allowed

        result = is_feature_allowed(
            feature="transform_8x8_mode",
            profile_idc=100
        )

        assert result is True

    @pytest.mark.xfail(reason="High profile features not implemented")
    def test_high_8x8_transform_flag_in_pps(self):
        """High profile PPS can have transform_8x8_mode_flag."""
        from parameters.pps import PPS

        pps = PPS(transform_8x8_mode_flag=True)

        assert pps.transform_8x8_mode_flag is True

    @pytest.mark.xfail(reason="High profile features not implemented")
    def test_baseline_no_8x8_transform(self):
        """Baseline profile forbids 8x8 transform."""
        from decoder.profile_validator import is_feature_allowed

        result = is_feature_allowed(
            feature="transform_8x8_mode",
            profile_idc=66
        )

        assert result is False

    @pytest.mark.xfail(reason="High profile features not implemented")
    def test_high_scaling_lists_allowed(self):
        """High profile allows custom scaling lists."""
        from decoder.profile_validator import is_feature_allowed

        result = is_feature_allowed(
            feature="scaling_lists",
            profile_idc=100
        )

        assert result is True

    @pytest.mark.xfail(reason="High profile features not implemented")
    def test_high_profile_sps_has_extended_fields(self):
        """High profile SPS has extended fields (chroma, bit depth, etc.)."""
        from parameters.sps import parse_sps_high_profile_extensions

        # High profile SPS includes:
        # - chroma_format_idc
        # - bit_depth_luma_minus8
        # - bit_depth_chroma_minus8
        # - qpprime_y_zero_transform_bypass_flag
        # - seq_scaling_matrix_present_flag
        assert callable(parse_sps_high_profile_extensions)

    @pytest.mark.xfail(reason="High profile features not implemented")
    def test_high_profile_420_chroma(self):
        """High profile default is 4:2:0 chroma."""
        from parameters.sps import SPS

        sps = SPS(profile_idc=100)

        # Default chroma_format_idc is 1 (4:2:0)
        assert sps.chroma_format_idc == 1

    @pytest.mark.xfail(reason="High profile features not implemented")
    def test_high_profile_8bit_depth(self):
        """High profile default is 8-bit depth."""
        from parameters.sps import SPS

        sps = SPS(profile_idc=100)

        assert sps.bit_depth_luma == 8
        assert sps.bit_depth_chroma == 8

    @pytest.mark.xfail(reason="High profile features not implemented")
    def test_high_direct_8x8_inference_required(self):
        """High profile requires direct_8x8_inference_flag when using B-slices."""
        from decoder.profile_validator import validate_high_constraints
        from parameters.sps import SPS

        # High profile with B-slices needs direct_8x8_inference
        sps_no_inference = SPS(
            profile_idc=100,
            direct_8x8_inference_flag=False,
        )

        # Should warn or fail when B-slices are used
        with pytest.raises(ValueError, match="direct_8x8"):
            validate_high_constraints(sps_no_inference, has_b_slices=True)


# =============================================================================
# Test: High 10 profile (10-bit support)
# =============================================================================

class TestHigh10Profile:
    """Tests for High 10 profile (profile_idc=110) features.

    High 10 adds:
    - 10-bit luma/chroma support (bit_depth 8-10)

    H.264 Spec: Annex A.2.5
    """

    @pytest.mark.xfail(reason="High 10 profile not implemented")
    def test_high10_10bit_luma_allowed(self):
        """High 10 profile allows 10-bit luma."""
        from decoder.profile_validator import validate_bit_depth
        from parameters.sps import SPS

        sps = SPS(
            profile_idc=110,
            bit_depth_luma_minus8=2,  # 10-bit
        )

        # Should not raise
        validate_bit_depth(sps)

    @pytest.mark.xfail(reason="High 10 profile not implemented")
    def test_high10_10bit_chroma_allowed(self):
        """High 10 profile allows 10-bit chroma."""
        from decoder.profile_validator import validate_bit_depth
        from parameters.sps import SPS

        sps = SPS(
            profile_idc=110,
            bit_depth_chroma_minus8=2,  # 10-bit
        )

        # Should not raise
        validate_bit_depth(sps)

    @pytest.mark.xfail(reason="High 10 profile not implemented")
    def test_high_basic_no_10bit(self):
        """Standard High profile (100) does not allow 10-bit."""
        from decoder.profile_validator import validate_bit_depth
        from parameters.sps import SPS

        sps = SPS(
            profile_idc=100,
            bit_depth_luma_minus8=2,  # 10-bit
        )

        with pytest.raises(ValueError, match="10.bit.*High.100"):
            validate_bit_depth(sps)

    @pytest.mark.xfail(reason="High 10 profile not implemented")
    def test_high10_qp_offset_for_bit_depth(self):
        """10-bit requires QP offset: QP' = QP + 6*(bit_depth - 8)."""
        from dequant.qp import calculate_qp_prime

        qp = 26
        bit_depth = 10

        qp_prime = calculate_qp_prime(qp, bit_depth)

        # QP' = 26 + 6*(10-8) = 26 + 12 = 38
        assert qp_prime == 38

    @pytest.mark.xfail(reason="High 10 profile not implemented")
    def test_high10_extended_qp_range(self):
        """10-bit extends QP range: 0 to 51 + 6*(bit_depth - 8)."""
        from dequant.qp import get_max_qp_for_bit_depth

        max_qp_8bit = get_max_qp_for_bit_depth(8)
        max_qp_10bit = get_max_qp_for_bit_depth(10)

        assert max_qp_8bit == 51
        assert max_qp_10bit == 63  # 51 + 12


# =============================================================================
# Test: constraint_set flags validation
# =============================================================================

class TestConstraintSetFlags:
    """Tests for constraint_set flags validation.

    constraint_set0_flag through constraint_set5_flag indicate
    compliance with specific profile constraints.

    H.264 Spec: Section 7.4.2.1
    """

    @pytest.mark.xfail(reason="Constraint flags validation not implemented")
    def test_constraint_set0_means_baseline(self):
        """constraint_set0_flag=1 means Baseline constraints apply."""
        from decoder.profile_validator import get_implied_profiles

        profiles = get_implied_profiles(
            profile_idc=77,  # Main
            constraint_set0_flag=True,
        )

        # With constraint_set0_flag, Baseline constraints also apply
        assert 66 in profiles  # Baseline
        assert 77 in profiles  # Main

    @pytest.mark.xfail(reason="Constraint flags validation not implemented")
    def test_constraint_set1_means_main(self):
        """constraint_set1_flag=1 means Main constraints apply."""
        from decoder.profile_validator import get_implied_profiles

        profiles = get_implied_profiles(
            profile_idc=100,  # High
            constraint_set1_flag=True,
        )

        # With constraint_set1_flag, Main constraints also apply
        assert 77 in profiles

    @pytest.mark.xfail(reason="Constraint flags validation not implemented")
    def test_constraint_set3_intra_only(self):
        """constraint_set3_flag in High profile means Intra-only."""
        from decoder.profile_validator import is_intra_only_profile

        result = is_intra_only_profile(
            profile_idc=100,
            constraint_set3_flag=True,
        )

        assert result is True

    @pytest.mark.xfail(reason="Constraint flags validation not implemented")
    def test_constrained_baseline_profile(self):
        """Constrained Baseline = Baseline with constraint_set1_flag."""
        from decoder.profile_validator import get_profile_name

        name = get_profile_name(
            profile_idc=66,
            constraint_set1_flag=True,
        )

        assert name == "Constrained Baseline"

    @pytest.mark.xfail(reason="Constraint flags validation not implemented")
    def test_constraint_flags_stored_in_sps(self):
        """SPS stores all constraint flags."""
        from parameters.sps import SPS

        sps = SPS(
            constraint_set0_flag=True,
            constraint_set1_flag=False,
            constraint_set2_flag=True,
            constraint_set3_flag=False,
            constraint_set4_flag=True,
            constraint_set5_flag=False,
        )

        assert sps.constraint_set0_flag is True
        assert sps.constraint_set1_flag is False
        assert sps.constraint_set2_flag is True
        assert sps.constraint_set3_flag is False
        assert sps.constraint_set4_flag is True
        assert sps.constraint_set5_flag is False


# =============================================================================
# Test: Level constraints
# =============================================================================

class TestLevelConstraints:
    """Tests for level_idc constraints.

    Levels define limits on:
    - MaxMBPS: Max macroblock processing speed
    - MaxFS: Max frame size in macroblocks
    - MaxDpbMbs: Max decoded picture buffer size
    - MaxBR: Max bitrate
    - MaxCPB: Max coded picture buffer size

    H.264 Spec: Table A-1
    """

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_max_frame_size_level30(self):
        """Level 3.0 allows max 1620 macroblocks (e.g., 720x576)."""
        from decoder.level_validator import get_max_frame_size

        max_fs = get_max_frame_size(level_idc=30)

        assert max_fs == 1620

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_max_frame_size_level40(self):
        """Level 4.0 allows max 8192 macroblocks (e.g., 1920x1088)."""
        from decoder.level_validator import get_max_frame_size

        max_fs = get_max_frame_size(level_idc=40)

        assert max_fs == 8192

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_max_frame_size_level51(self):
        """Level 5.1 allows max 36864 macroblocks (e.g., 4096x2160)."""
        from decoder.level_validator import get_max_frame_size

        max_fs = get_max_frame_size(level_idc=51)

        assert max_fs == 36864

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_validate_frame_size(self):
        """Frame size validation against level."""
        from decoder.level_validator import validate_frame_size

        # 720p = 45x80 = 3600 MBs - needs level 3.1 or higher
        with pytest.raises(ValueError, match="exceeds"):
            validate_frame_size(
                pic_width_in_mbs=80,
                pic_height_in_mbs=45,
                level_idc=30  # Max 1620 MBs
            )

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_max_dpb_mbs_level30(self):
        """Level 3.0 MaxDpbMbs = 8100."""
        from decoder.level_validator import get_max_dpb_mbs

        max_dpb = get_max_dpb_mbs(level_idc=30)

        assert max_dpb == 8100

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_max_dpb_mbs_level40(self):
        """Level 4.0 MaxDpbMbs = 32768."""
        from decoder.level_validator import get_max_dpb_mbs

        max_dpb = get_max_dpb_mbs(level_idc=40)

        assert max_dpb == 32768

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_derive_max_ref_frames(self):
        """Derive max reference frames from level and frame size."""
        from decoder.level_validator import derive_max_ref_frames

        # For 1080p at level 4.0:
        # MaxDpbMbs = 32768, FrameSize = 8160 MBs (1920x1088)
        # MaxRefFrames = min(16, floor(32768 / 8160)) = 4
        max_ref = derive_max_ref_frames(
            level_idc=40,
            pic_width_in_mbs=120,
            pic_height_in_mbs=68,
        )

        assert max_ref == 4

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_max_mbps_level30(self):
        """Level 3.0 MaxMBPS = 40500 MB/s."""
        from decoder.level_validator import get_max_mbps

        max_mbps = get_max_mbps(level_idc=30)

        assert max_mbps == 40500

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_max_mbps_level40(self):
        """Level 4.0 MaxMBPS = 245760 MB/s."""
        from decoder.level_validator import get_max_mbps

        max_mbps = get_max_mbps(level_idc=40)

        assert max_mbps == 245760

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_validate_mbps(self):
        """Validate macroblock processing speed."""
        from decoder.level_validator import validate_mbps

        # 1080p60 = 120*68*60 = 489,600 MB/s
        # Needs level 4.2 or higher
        with pytest.raises(ValueError, match="MaxMBPS"):
            validate_mbps(
                pic_width_in_mbs=120,
                pic_height_in_mbs=68,
                frame_rate=60.0,
                level_idc=40  # Max 245760 MB/s
            )

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_max_bitrate_level30(self):
        """Level 3.0 MaxBR = 10000 kbps (Baseline/Main)."""
        from decoder.level_validator import get_max_bitrate

        max_br = get_max_bitrate(level_idc=30, profile_idc=66)

        assert max_br == 10000  # kbps

    @pytest.mark.xfail(reason="Level validation not implemented")
    def test_level_max_bitrate_high_profile(self):
        """High profile has higher bitrate limits (1.25x Main)."""
        from decoder.level_validator import get_max_bitrate

        max_br_main = get_max_bitrate(level_idc=40, profile_idc=77)
        max_br_high = get_max_bitrate(level_idc=40, profile_idc=100)

        # High profile allows 1.25x the Main profile bitrate
        assert max_br_high == int(max_br_main * 1.25)


# =============================================================================
# Test: Level 1b special handling
# =============================================================================

class TestLevel1bHandling:
    """Tests for special Level 1b handling.

    Level 1b is indicated by level_idc=11 with constraint_set3_flag=1,
    or by level_idc=9 in some older encoders.

    H.264 Spec: Table A-1 footnote
    """

    @pytest.mark.xfail(reason="Level 1b handling not implemented")
    def test_level_1b_from_11_with_constraint(self):
        """Level 1b: level_idc=11 with constraint_set3_flag=1."""
        from decoder.level_validator import get_effective_level

        effective = get_effective_level(
            level_idc=11,
            constraint_set3_flag=True
        )

        assert effective == "1b"

    @pytest.mark.xfail(reason="Level 1b handling not implemented")
    def test_level_1b_from_9(self):
        """Level 1b: level_idc=9 (legacy encoding)."""
        from decoder.level_validator import get_effective_level

        effective = get_effective_level(level_idc=9)

        assert effective == "1b"

    @pytest.mark.xfail(reason="Level 1b handling not implemented")
    def test_level_1b_limits(self):
        """Level 1b has specific limits between 1 and 1.1."""
        from decoder.level_validator import get_max_frame_size

        max_fs_1 = get_max_frame_size(level_idc=10)  # Level 1
        max_fs_1b = get_max_frame_size(level_idc=9)  # Level 1b
        max_fs_11 = get_max_frame_size(level_idc=11)  # Level 1.1

        # 1b is between 1 and 1.1
        assert max_fs_1 <= max_fs_1b <= max_fs_11


# =============================================================================
# Test: Profile/level combination validation
# =============================================================================

class TestProfileLevelCombinations:
    """Tests for valid profile/level combinations."""

    @pytest.mark.xfail(reason="Profile/level validation not implemented")
    def test_baseline_all_levels_valid(self):
        """Baseline profile supports all levels."""
        from decoder.profile_validator import is_valid_profile_level

        for level in [10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51]:
            assert is_valid_profile_level(profile_idc=66, level_idc=level)

    @pytest.mark.xfail(reason="Profile/level validation not implemented")
    def test_high_10_requires_level_30_plus(self):
        """High 10 profile typically requires Level 3.0 or higher."""
        from decoder.profile_validator import is_valid_profile_level

        # Level 2.x may not support High 10 features
        assert not is_valid_profile_level(profile_idc=110, level_idc=20)

        # Level 3.0+ should work
        assert is_valid_profile_level(profile_idc=110, level_idc=30)

    @pytest.mark.xfail(reason="Profile/level validation not implemented")
    def test_decoder_rejects_invalid_combination(self):
        """Decoder rejects invalid profile/level combinations."""
        from decoder import H264Decoder
        from parameters.sps import SPS

        decoder = H264Decoder()

        sps_invalid = SPS(
            profile_idc=110,  # High 10
            level_idc=10,     # Level 1.0 (too low)
        )

        with pytest.raises(ValueError, match="profile.*level"):
            decoder.validate_sps(sps_invalid)


# =============================================================================
# Test: Feature availability by profile
# =============================================================================

class TestFeatureAvailabilityMatrix:
    """Tests for feature availability across profiles."""

    @pytest.mark.xfail(reason="Feature matrix not implemented")
    def test_feature_matrix_cabac(self):
        """CABAC availability by profile."""
        from decoder.profile_validator import is_feature_allowed

        # Baseline: no CABAC
        assert not is_feature_allowed("CABAC", profile_idc=66)

        # Main: yes CABAC
        assert is_feature_allowed("CABAC", profile_idc=77)

        # High: yes CABAC
        assert is_feature_allowed("CABAC", profile_idc=100)

    @pytest.mark.xfail(reason="Feature matrix not implemented")
    def test_feature_matrix_b_slices(self):
        """B-slice availability by profile."""
        from decoder.profile_validator import is_feature_allowed

        # Baseline: no B-slices
        assert not is_feature_allowed("B_slices", profile_idc=66)

        # Main: yes B-slices
        assert is_feature_allowed("B_slices", profile_idc=77)

        # High: yes B-slices
        assert is_feature_allowed("B_slices", profile_idc=100)

    @pytest.mark.xfail(reason="Feature matrix not implemented")
    def test_feature_matrix_fmo(self):
        """FMO availability by profile."""
        from decoder.profile_validator import is_feature_allowed

        # Baseline: yes FMO
        assert is_feature_allowed("FMO", profile_idc=66)

        # Main: no FMO
        assert not is_feature_allowed("FMO", profile_idc=77)

        # Extended: yes FMO
        assert is_feature_allowed("FMO", profile_idc=88)

    @pytest.mark.xfail(reason="Feature matrix not implemented")
    def test_feature_matrix_8x8_transform(self):
        """8x8 transform availability by profile."""
        from decoder.profile_validator import is_feature_allowed

        # Baseline: no 8x8
        assert not is_feature_allowed("transform_8x8", profile_idc=66)

        # Main: no 8x8
        assert not is_feature_allowed("transform_8x8", profile_idc=77)

        # High: yes 8x8
        assert is_feature_allowed("transform_8x8", profile_idc=100)

    @pytest.mark.xfail(reason="Feature matrix not implemented")
    def test_feature_matrix_weighted_pred(self):
        """Weighted prediction availability by profile."""
        from decoder.profile_validator import is_feature_allowed

        # Baseline: no weighted pred
        assert not is_feature_allowed("weighted_pred", profile_idc=66)

        # Main: yes weighted pred
        assert is_feature_allowed("weighted_pred", profile_idc=77)

    @pytest.mark.xfail(reason="Feature matrix not implemented")
    def test_feature_matrix_scaling_lists(self):
        """Scaling list availability by profile."""
        from decoder.profile_validator import is_feature_allowed

        # Baseline: no scaling lists
        assert not is_feature_allowed("scaling_lists", profile_idc=66)

        # Main: no scaling lists
        assert not is_feature_allowed("scaling_lists", profile_idc=77)

        # High: yes scaling lists
        assert is_feature_allowed("scaling_lists", profile_idc=100)


# =============================================================================
# Test: Extended profile specifics
# =============================================================================

class TestExtendedProfile:
    """Tests for Extended profile (profile_idc=88) specifics.

    Extended profile is unique - it supports:
    - SI and SP slices (switching slices)
    - FMO, ASO, redundant slices
    - Data partitioning

    But NOT:
    - CABAC
    - 8x8 transform

    H.264 Spec: Annex A.2.3
    """

    @pytest.mark.xfail(reason="Extended profile not implemented")
    def test_extended_no_cabac(self):
        """Extended profile uses CAVLC only (like Baseline)."""
        from decoder.profile_validator import is_feature_allowed

        assert not is_feature_allowed("CABAC", profile_idc=88)

    @pytest.mark.xfail(reason="Extended profile not implemented")
    def test_extended_si_sp_slices(self):
        """Extended profile supports SI and SP slices."""
        from decoder.profile_validator import is_feature_allowed

        assert is_feature_allowed("SI_slices", profile_idc=88)
        assert is_feature_allowed("SP_slices", profile_idc=88)

    @pytest.mark.xfail(reason="Extended profile not implemented")
    def test_extended_b_slices(self):
        """Extended profile supports B-slices."""
        from decoder.profile_validator import is_feature_allowed

        assert is_feature_allowed("B_slices", profile_idc=88)

    @pytest.mark.xfail(reason="Extended profile not implemented")
    def test_extended_data_partitioning(self):
        """Extended profile supports data partitioning."""
        from decoder.profile_validator import is_feature_allowed

        assert is_feature_allowed("data_partitioning", profile_idc=88)

    @pytest.mark.xfail(reason="Extended profile not implemented")
    def test_extended_no_8x8_transform(self):
        """Extended profile does not support 8x8 transform."""
        from decoder.profile_validator import is_feature_allowed

        assert not is_feature_allowed("transform_8x8", profile_idc=88)


# =============================================================================
# Test: Decoder compliance checking
# =============================================================================

class TestDecoderCompliance:
    """Tests for decoder compliance checking at runtime."""

    @pytest.mark.xfail(reason="Compliance checking not implemented")
    def test_decoder_warns_on_profile_violation(self):
        """Decoder warns when bitstream violates declared profile."""
        from decoder import H264Decoder
        from parameters.sps import SPS
        from parameters.pps import PPS

        decoder = H264Decoder()

        # Declare Baseline profile
        sps = SPS(profile_idc=66)
        decoder.state.sps_dict[0] = sps

        # But PPS uses CABAC (Baseline violation)
        pps = PPS(seq_parameter_set_id=0, entropy_coding_mode_flag=True)

        # Should warn or raise
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            decoder.state.pps_dict[0] = pps
            assert len(w) > 0
            assert "CABAC" in str(w[0].message) or "profile" in str(w[0].message)

    @pytest.mark.xfail(reason="Compliance checking not implemented")
    def test_decoder_tracks_profile_compliance(self):
        """Decoder tracks whether stream is profile-compliant."""
        from decoder import H264Decoder

        decoder = H264Decoder()

        # Initially compliant
        assert decoder.is_profile_compliant()

        # After violation
        decoder.record_profile_violation("CABAC in Baseline")
        assert not decoder.is_profile_compliant()

    @pytest.mark.xfail(reason="Compliance checking not implemented")
    def test_strict_mode_rejects_violations(self):
        """Strict mode decoder rejects profile violations."""
        from decoder import H264Decoder
        from parameters.sps import SPS
        from parameters.pps import PPS

        decoder = H264Decoder(strict_profile_compliance=True)

        sps = SPS(profile_idc=66)  # Baseline
        decoder.state.sps_dict[0] = sps

        pps = PPS(seq_parameter_set_id=0, entropy_coding_mode_flag=True)

        with pytest.raises(ValueError, match="profile"):
            decoder.state.pps_dict[0] = pps
