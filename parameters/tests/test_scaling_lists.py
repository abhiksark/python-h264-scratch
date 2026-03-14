# h264/parameters/tests/test_scaling_lists.py
"""Tests for H.264 scaling list handling (TDD - RED phase).

H.264 Spec Reference:
- Section 7.3.2.1.1: Scaling list syntax
- Section 7.4.2.1.1: Scaling list semantics
- Section 8.5.9: Derivation process for scaling functions
- Table 7-3: Default_4x4_Intra and Default_4x4_Inter
- Table 7-4: Default_8x8_Intra and Default_8x8_Inter

These tests are written TDD-style and should FAIL until the
scaling list handling is properly implemented.
"""

import numpy as np
import pytest

from bitstream import BitWriter, BITSTRING_AVAILABLE


pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


# H.264 Spec Table 7-3: Default scaling lists for 4x4 blocks
DEFAULT_4x4_INTRA = [
    6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42
]

DEFAULT_4x4_INTER = [
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34
]

# H.264 Spec Table 7-4: Default scaling lists for 8x8 blocks
DEFAULT_8x8_INTRA = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23,
    23, 23, 23, 23, 23, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27,
    27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42
]

DEFAULT_8x8_INTER = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21,
    21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 24, 24, 24, 24,
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35
]

# Flat scaling list (all 16s) - used when seq_scaling_matrix_present_flag=0
FLAT_4x4 = [16] * 16
FLAT_8x8 = [16] * 64


class TestDefaultScalingListSelection:
    """Test default scaling list selection logic.

    H.264 Spec: Section 7.4.2.1.1
    When seq_scaling_matrix_present_flag is 0 (or not present for
    non-High profiles), flat scaling lists should be used.

    When seq_scaling_list_present_flag[i] is 0, fallback rules apply:
    - For i=0,1,2 (Intra Y 4x4): use Default_4x4_Intra or fall back to i-1
    - For i=3,4,5 (Inter Y 4x4): use Default_4x4_Inter or fall back to i-1
    - For i=6 (Intra Y 8x8): use Default_8x8_Intra
    - For i=7 (Inter Y 8x8): use Default_8x8_Inter
    """

    def test_baseline_profile_uses_flat_scaling(self):
        """Baseline profile (no scaling matrix) should use flat lists."""
        from parameters.sps import parse_sps
        from parameters.scaling import get_scaling_list_4x4, get_scaling_list_8x8

        # Create Baseline SPS (profile_idc=66)
        writer = BitWriter()
        writer.write_bits(66, 8)  # profile_idc
        writer.write_bits(0, 6)  # constraint flags
        writer.write_bits(0, 2)  # reserved
        writer.write_bits(30, 8)  # level_idc
        writer.write_ue(0)  # sps_id
        writer.write_ue(0)  # log2_max_frame_num_minus4
        writer.write_ue(2)  # pic_order_cnt_type
        writer.write_ue(1)  # max_num_ref_frames
        writer.write_flag(False)  # gaps
        writer.write_ue(7)  # width - 1
        writer.write_ue(5)  # height - 1
        writer.write_flag(True)  # frame_mbs_only
        writer.write_flag(False)  # direct_8x8
        writer.write_flag(False)  # no cropping
        writer.write_flag(False)  # no vui

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        # Should return flat scaling lists
        for i in range(6):
            assert get_scaling_list_4x4(sps, None, i) == FLAT_4x4
        for i in range(2):
            assert get_scaling_list_8x8(sps, None, i) == FLAT_8x8

    def test_high_profile_no_matrix_uses_flat_scaling(self):
        """High profile with scaling_matrix_present=0 uses FLAT lists.

        JM reference decoder: when neither SPS nor PPS signals scaling
        matrices (both present_flags=0), flat lists (all 16s) are used.
        """
        from parameters.sps import parse_sps
        from parameters.scaling import (
            get_scaling_list_4x4,
            FLAT_4x4,
        )

        writer = BitWriter()
        writer.write_bits(100, 8)  # High profile
        writer.write_bits(0, 6)
        writer.write_bits(0, 2)
        writer.write_bits(40, 8)
        writer.write_ue(0)
        writer.write_ue(1)  # chroma_format_idc
        writer.write_ue(0)  # bit_depth_luma
        writer.write_ue(0)  # bit_depth_chroma
        writer.write_flag(False)  # qpprime bypass
        writer.write_flag(False)  # seq_scaling_matrix_present_flag = 0
        writer.write_ue(0)
        writer.write_ue(2)
        writer.write_ue(1)
        writer.write_flag(False)
        writer.write_ue(7)
        writer.write_ue(5)
        writer.write_flag(True)
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        # JM: when neither SPS nor PPS has scaling matrices, use flat
        for i in range(6):
            assert get_scaling_list_4x4(sps, None, i) == FLAT_4x4

    def test_use_default_scaling_matrix_flag_selects_spec_defaults(self):
        """When use_default_scaling_matrix_flag is set, use spec-defined lists."""
        from parameters.scaling import (
            get_scaling_list_4x4,
            DEFAULT_4x4_INTRA,
            DEFAULT_4x4_INTER,
        )

        # This tests the case where next_scale==0 and j==0 during parsing
        # In this case, use_default_scaling_matrix_flag is inferred and
        # the spec-defined default should be used
        # Lists 0,1,2 are Intra, lists 3,4,5 are Inter
        assert DEFAULT_4x4_INTRA == [
            6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42
        ]
        assert DEFAULT_4x4_INTER == [
            10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34
        ]


class TestCustomScalingListParsing4x4:
    """Test custom 4x4 scaling list parsing from bitstream.

    H.264 Spec: Section 7.3.2.1.1
    """

    def test_parse_custom_4x4_scaling_list(self):
        """Parse a custom 4x4 scaling list with delta_scale values."""
        from parameters.scaling import parse_scaling_list

        writer = BitWriter()
        # Write a custom scaling list: first value 16, rest with deltas
        # delta_scale = next_scale - last_scale (mod 256)
        # If we want [16, 16, 16, ...], all deltas should be 0 except first
        writer.write_se(8)  # First: next_scale = 8 + 8 = 16
        for _ in range(15):
            writer.write_se(0)  # Rest: delta=0 means use last_scale

        data = writer.to_bytes()
        scaling_list, use_default = parse_scaling_list(data, size=16)

        assert len(scaling_list) == 16
        assert use_default is False
        assert scaling_list == [16] * 16

    def test_parse_4x4_list_with_varying_values(self):
        """Parse 4x4 list with varying scaling values."""
        from parameters.scaling import parse_scaling_list

        # Create a custom pattern: 10, 12, 14, 16, ...
        writer = BitWriter()
        writer.write_se(2)  # 8 + 2 = 10
        writer.write_se(2)  # 10 + 2 = 12
        writer.write_se(2)  # 12 + 2 = 14
        writer.write_se(2)  # 14 + 2 = 16
        for _ in range(12):
            writer.write_se(0)  # Rest stay at 16

        data = writer.to_bytes()
        scaling_list, use_default = parse_scaling_list(data, size=16)

        assert scaling_list[0] == 10
        assert scaling_list[1] == 12
        assert scaling_list[2] == 14
        assert scaling_list[3] == 16

    def test_parse_4x4_list_detects_use_default_flag(self):
        """Detect use_default_scaling_matrix_flag when next_scale=0 at j=0."""
        from parameters.scaling import parse_scaling_list

        writer = BitWriter()
        # delta_scale = -8 makes next_scale = 8 + (-8) = 0 at j=0
        writer.write_se(-8)  # This signals use_default_scaling_matrix_flag

        data = writer.to_bytes()
        scaling_list, use_default = parse_scaling_list(data, size=16)

        assert use_default is True
        # When use_default is True, return spec-defined default list

    def test_parse_4x4_intra_list_from_sps(self):
        """Parse Intra 4x4 scaling list from SPS."""
        from parameters.sps import parse_sps

        writer = BitWriter()
        writer.write_bits(100, 8)  # High profile
        writer.write_bits(0, 8)
        writer.write_bits(40, 8)
        writer.write_ue(0)
        writer.write_ue(1)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_flag(False)
        writer.write_flag(True)  # seq_scaling_matrix_present

        # List 0 (Intra Y 4x4) present with custom values
        writer.write_flag(True)  # list present
        # Write delta_scale values for a flat list of 16s
        writer.write_se(8)  # 8 + 8 = 16
        for _ in range(15):
            writer.write_se(0)

        # Lists 1-7 not present (use fallback)
        for _ in range(7):
            writer.write_flag(False)

        # Standard SPS fields
        writer.write_ue(0)
        writer.write_ue(2)
        writer.write_ue(1)
        writer.write_flag(False)
        writer.write_ue(7)
        writer.write_ue(5)
        writer.write_flag(True)
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        assert sps.seq_scaling_matrix_present_flag is True
        assert len(sps.scaling_lists_4x4) >= 1
        assert sps.scaling_lists_4x4[0] == [16] * 16


class TestCustomScalingListParsing8x8:
    """Test custom 8x8 scaling list parsing from bitstream."""

    def test_parse_custom_8x8_scaling_list(self):
        """Parse a custom 8x8 scaling list."""
        from parameters.scaling import parse_scaling_list

        writer = BitWriter()
        writer.write_se(8)  # First: 8 + 8 = 16
        for _ in range(63):
            writer.write_se(0)

        data = writer.to_bytes()
        scaling_list, use_default = parse_scaling_list(data, size=64)

        assert len(scaling_list) == 64
        assert use_default is False
        assert scaling_list == [16] * 64

    def test_parse_8x8_intra_list_from_sps(self):
        """Parse Intra 8x8 scaling list from SPS."""
        from parameters.sps import parse_sps

        writer = BitWriter()
        writer.write_bits(100, 8)  # High profile
        writer.write_bits(0, 8)
        writer.write_bits(40, 8)
        writer.write_ue(0)
        writer.write_ue(1)  # chroma_format_idc=1 means 8 scaling lists
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_flag(False)
        writer.write_flag(True)  # seq_scaling_matrix_present

        # Lists 0-5 (4x4) not present
        for _ in range(6):
            writer.write_flag(False)

        # List 6 (Intra Y 8x8) present
        writer.write_flag(True)
        writer.write_se(8)
        for _ in range(63):
            writer.write_se(0)

        # List 7 not present
        writer.write_flag(False)

        writer.write_ue(0)
        writer.write_ue(2)
        writer.write_ue(1)
        writer.write_flag(False)
        writer.write_ue(7)
        writer.write_ue(5)
        writer.write_flag(True)
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)

        rbsp = writer.to_bytes()
        sps = parse_sps(rbsp)

        assert len(sps.scaling_lists_8x8) >= 1
        assert sps.scaling_lists_8x8[0] == [16] * 64


class TestScalingListFallbackRules:
    """Test scaling list fallback rules.

    H.264 Spec: Section 7.4.2.1.1
    When seq_scaling_list_present_flag[i] = 0:
    - i=0: Use Default_4x4_Intra
    - i=1,2: Use scaling list i-1 (fallback to previous)
    - i=3: Use Default_4x4_Inter
    - i=4,5: Use scaling list i-1 (fallback to previous)
    - i=6: Use Default_8x8_Intra
    - i=7: Use scaling list i-1 (fallback to list 6)
    """

    def test_fallback_4x4_list_0_to_default_intra(self):
        """List 0 not present -> use Default_4x4_Intra."""
        from parameters.scaling import get_effective_scaling_list_4x4

        # Create SPS with scaling_matrix_present but list 0 not present
        # and use_default_scaling_matrix_flag set
        result = get_effective_scaling_list_4x4(
            sps_list=None,
            pps_list=None,
            list_idx=0,
            use_default_flag=True
        )
        assert result == DEFAULT_4x4_INTRA

    def test_fallback_4x4_list_1_to_list_0(self):
        """List 1 not present -> fall back to list 0."""
        from parameters.scaling import get_effective_scaling_list_4x4

        custom_list_0 = [20] * 16
        result = get_effective_scaling_list_4x4(
            sps_list=[custom_list_0, None],  # List 0 present, list 1 absent
            pps_list=None,
            list_idx=1,
            use_default_flag=False
        )
        assert result == custom_list_0

    def test_fallback_4x4_list_3_to_default_inter(self):
        """List 3 not present -> use Default_4x4_Inter."""
        from parameters.scaling import get_effective_scaling_list_4x4

        result = get_effective_scaling_list_4x4(
            sps_list=[None, None, None, None],
            pps_list=None,
            list_idx=3,
            use_default_flag=True
        )
        assert result == DEFAULT_4x4_INTER

    def test_fallback_8x8_list_6_to_default_intra(self):
        """8x8 list 0 (index 6) not present -> use Default_8x8_Intra."""
        from parameters.scaling import get_effective_scaling_list_8x8

        result = get_effective_scaling_list_8x8(
            sps_list=None,
            pps_list=None,
            list_idx=0,
            use_default_flag=True
        )
        assert result == DEFAULT_8x8_INTRA

    def test_fallback_8x8_list_7_to_list_6(self):
        """8x8 list 1 (index 7) not present -> fall back to list 0 (index 6)."""
        from parameters.scaling import get_effective_scaling_list_8x8

        custom_list_6 = [32] * 64
        result = get_effective_scaling_list_8x8(
            sps_list=[custom_list_6, None],
            pps_list=None,
            list_idx=1,
            use_default_flag=False
        )
        assert result == custom_list_6


class TestPPSOverridesSPS:
    """Test that PPS scaling lists override SPS scaling lists.

    H.264 Spec: Section 7.4.2.2
    PPS can have its own scaling lists that override SPS lists.
    """

    def test_pps_scaling_list_overrides_sps(self):
        """PPS scaling list should override SPS when present."""
        from parameters.scaling import get_effective_scaling_list_4x4

        sps_list = [[16] * 16] * 6
        pps_list = [[32] * 16] * 6

        result = get_effective_scaling_list_4x4(
            sps_list=sps_list,
            pps_list=pps_list,
            list_idx=0,
            use_default_flag=False
        )
        assert result == [32] * 16

    def test_pps_absent_falls_back_to_sps(self):
        """When PPS list absent, fall back to SPS list."""
        from parameters.scaling import get_effective_scaling_list_4x4

        sps_list = [[24] * 16] * 6
        pps_list = [None] * 6  # PPS doesn't define any lists

        result = get_effective_scaling_list_4x4(
            sps_list=sps_list,
            pps_list=pps_list,
            list_idx=0,
            use_default_flag=False
        )
        assert result == [24] * 16

    def test_pps_partial_override(self):
        """PPS can override only some lists, others fall back to SPS."""
        from parameters.scaling import get_effective_scaling_list_4x4

        sps_list = [[16] * 16, [18] * 16, [20] * 16, [22] * 16, [24] * 16, [26] * 16]
        pps_list = [[32] * 16, None, [40] * 16, None, None, None]

        # List 0: PPS overrides
        assert get_effective_scaling_list_4x4(sps_list, pps_list, 0, False) == [32] * 16
        # List 1: Falls back to SPS
        assert get_effective_scaling_list_4x4(sps_list, pps_list, 1, False) == [18] * 16
        # List 2: PPS overrides
        assert get_effective_scaling_list_4x4(sps_list, pps_list, 2, False) == [40] * 16

    def test_parse_pps_with_scaling_lists(self):
        """Parse PPS with custom scaling lists."""
        from parameters.pps import parse_pps

        writer = BitWriter()
        writer.write_ue(0)  # pps_id
        writer.write_ue(0)  # sps_id
        writer.write_flag(True)  # CABAC
        writer.write_flag(False)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_ue(0)
        writer.write_flag(False)
        writer.write_bits(0, 2)
        writer.write_se(0)
        writer.write_se(0)
        writer.write_se(0)
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_flag(False)

        # High profile extensions
        writer.write_flag(True)  # transform_8x8_mode_flag
        writer.write_flag(True)  # pic_scaling_matrix_present_flag

        # 8 scaling lists (6 for 4x4 + 2 for 8x8 when transform_8x8 enabled)
        # List 0 present with custom values
        writer.write_flag(True)
        writer.write_se(12)  # 8 + 12 = 20
        for _ in range(15):
            writer.write_se(0)

        # Lists 1-7 not present
        for _ in range(7):
            writer.write_flag(False)

        writer.write_se(0)  # second_chroma_qp_index_offset

        rbsp = writer.to_bytes()
        pps = parse_pps(rbsp, is_high_profile=True)

        assert pps.pic_scaling_matrix_present_flag is True
        assert len(pps.scaling_lists_4x4) >= 1
        assert pps.scaling_lists_4x4[0] == [20] * 16


class TestScalingListInDequantization:
    """Test scaling list application in dequantization.

    H.264 Spec: Section 8.5.9, 8.5.11
    """

    def test_dequant_4x4_with_scaling_list(self):
        """Dequantize 4x4 block using custom scaling list."""
        from dequant import dequant_4x4_with_scaling_list

        coeffs = np.ones((4, 4), dtype=np.int32)
        scaling_list = [32] * 16  # Double the flat list values

        result_flat = dequant_4x4_with_scaling_list(coeffs, qp=26, scaling_list=FLAT_4x4)
        result_custom = dequant_4x4_with_scaling_list(coeffs, qp=26, scaling_list=scaling_list)

        # Custom list with higher values should produce larger output
        assert np.sum(np.abs(result_custom)) > np.sum(np.abs(result_flat))

    def test_dequant_8x8_with_scaling_list(self):
        """Dequantize 8x8 block using custom scaling list."""
        from dequant import dequant_8x8

        coeffs = np.ones((8, 8), dtype=np.int32)

        result_flat = dequant_8x8(coeffs, qp=26, scaling_list=FLAT_8x8)
        result_custom = dequant_8x8(coeffs, qp=26, scaling_list=[32] * 64)

        assert np.sum(np.abs(result_custom)) > np.sum(np.abs(result_flat))

    def test_scaling_list_position_dependent(self):
        """Scaling list values should be applied position-dependently."""
        from dequant import dequant_4x4_with_scaling_list

        # Create a scaling list with different values at different positions
        scaling_list = list(range(16, 32))  # 16, 17, 18, ..., 31
        coeffs = np.ones((4, 4), dtype=np.int32)

        result = dequant_4x4_with_scaling_list(coeffs, qp=26, scaling_list=scaling_list)

        # Due to different scaling values, output should vary by position
        # The flat portion of the scale matrix is affected by scaling_list
        assert result[0, 0] != result[3, 3]

    def test_intra_inter_scaling_lists_differ(self):
        """Intra and Inter scaling lists produce different results."""
        from dequant import dequant_4x4_with_scaling_list

        coeffs = np.ones((4, 4), dtype=np.int32)

        result_intra = dequant_4x4_with_scaling_list(
            coeffs, qp=26, scaling_list=DEFAULT_4x4_INTRA
        )
        result_inter = dequant_4x4_with_scaling_list(
            coeffs, qp=26, scaling_list=DEFAULT_4x4_INTER
        )

        # Intra and Inter defaults differ, so results should differ
        assert not np.array_equal(result_intra, result_inter)


class TestUseDefaultScalingMatrixFlag:
    """Test use_default_scaling_matrix_flag handling.

    H.264 Spec: Section 7.3.2.1.1, 7.4.2.1.1
    When parsing scaling_list(), if delta_scale causes next_scale=0
    at the first position (j=0), use_default_scaling_matrix_flag is
    inferred as 1, meaning use the spec-defined default list.
    """

    def test_delta_minus8_at_start_sets_use_default_flag(self):
        """delta_scale=-8 at j=0 should set use_default_scaling_matrix_flag."""
        from parameters.scaling import parse_scaling_list

        writer = BitWriter()
        writer.write_se(-8)  # next_scale = 8 + (-8) = 0 at j=0

        data = writer.to_bytes()
        _, use_default = parse_scaling_list(data, size=16)

        assert use_default is True

    def test_use_default_flag_returns_spec_default_4x4_intra(self):
        """When use_default=True for list 0, return Default_4x4_Intra."""
        from parameters.scaling import get_scaling_list_with_default

        result = get_scaling_list_with_default(
            parsed_list=None,
            use_default_flag=True,
            list_idx=0  # Intra Y 4x4
        )
        assert result == DEFAULT_4x4_INTRA

    def test_use_default_flag_returns_spec_default_4x4_inter(self):
        """When use_default=True for list 3, return Default_4x4_Inter."""
        from parameters.scaling import get_scaling_list_with_default

        result = get_scaling_list_with_default(
            parsed_list=None,
            use_default_flag=True,
            list_idx=3  # Inter Y 4x4
        )
        assert result == DEFAULT_4x4_INTER

    def test_use_default_flag_returns_spec_default_8x8_intra(self):
        """When use_default=True for list 6, return Default_8x8_Intra."""
        from parameters.scaling import get_scaling_list_with_default

        result = get_scaling_list_with_default(
            parsed_list=None,
            use_default_flag=True,
            list_idx=6  # Intra Y 8x8
        )
        assert result == DEFAULT_8x8_INTRA

    def test_use_default_flag_returns_spec_default_8x8_inter(self):
        """When use_default=True for list 7, return Default_8x8_Inter."""
        from parameters.scaling import get_scaling_list_with_default

        result = get_scaling_list_with_default(
            parsed_list=None,
            use_default_flag=True,
            list_idx=7  # Inter Y 8x8
        )
        assert result == DEFAULT_8x8_INTER

    def test_next_scale_zero_mid_list_uses_last_scale(self):
        """next_scale=0 at j>0 means use last_scale, not default flag."""
        from parameters.scaling import parse_scaling_list

        writer = BitWriter()
        writer.write_se(8)   # j=0: next_scale = 16
        writer.write_se(4)   # j=1: next_scale = 20
        writer.write_se(-20) # j=2: next_scale = 0, use last_scale (20)
        # For remaining positions, since next_scale=0, they all use 20
        for _ in range(13):
            pass  # No more reads needed when next_scale=0

        data = writer.to_bytes()
        scaling_list, use_default = parse_scaling_list(data, size=16)

        assert use_default is False
        assert scaling_list[0] == 16
        assert scaling_list[1] == 20
        assert scaling_list[2] == 20  # Uses last_scale since next_scale=0


class TestScalingListModuleInterface:
    """Test the scaling list module interface.

    These tests verify the module provides the expected functions.
    """

    def test_module_exports_default_lists(self):
        """Module should export default scaling list constants."""
        from parameters import scaling

        assert hasattr(scaling, 'DEFAULT_4x4_INTRA')
        assert hasattr(scaling, 'DEFAULT_4x4_INTER')
        assert hasattr(scaling, 'DEFAULT_8x8_INTRA')
        assert hasattr(scaling, 'DEFAULT_8x8_INTER')
        assert hasattr(scaling, 'FLAT_4x4')
        assert hasattr(scaling, 'FLAT_8x8')

    def test_module_exports_parse_function(self):
        """Module should export parse_scaling_list function."""
        from parameters import scaling

        assert hasattr(scaling, 'parse_scaling_list')
        assert callable(scaling.parse_scaling_list)

    def test_module_exports_getter_functions(self):
        """Module should export scaling list getter functions."""
        from parameters import scaling

        assert hasattr(scaling, 'get_scaling_list_4x4')
        assert hasattr(scaling, 'get_scaling_list_8x8')
        assert hasattr(scaling, 'get_effective_scaling_list_4x4')
        assert hasattr(scaling, 'get_effective_scaling_list_8x8')


class TestScalingListEdgeCases:
    """Test edge cases in scaling list handling."""

    def test_all_zeros_scaling_list_invalid(self):
        """Scaling list with all zeros should be handled gracefully."""
        from parameters.scaling import parse_scaling_list

        # This would require delta_scale values that wrap around
        # A well-formed bitstream shouldn't produce this, but we should handle it
        writer = BitWriter()
        writer.write_se(-8)  # next_scale = 0 at j=0 -> use_default

        data = writer.to_bytes()
        _, use_default = parse_scaling_list(data, size=16)
        assert use_default is True

    def test_maximum_scaling_values(self):
        """Scaling values up to 255 should be handled."""
        from parameters.scaling import parse_scaling_list

        writer = BitWriter()
        # Start at 8, add delta to get 255
        writer.write_se(247)  # 8 + 247 = 255
        for _ in range(15):
            writer.write_se(0)

        data = writer.to_bytes()
        scaling_list, _ = parse_scaling_list(data, size=16)
        assert scaling_list[0] == 255

    def test_chroma_scaling_lists_4_2_0(self):
        """Test chroma scaling lists for 4:2:0 format."""
        from parameters.scaling import get_chroma_scaling_list_4x4

        # For 4:2:0, chroma uses same scaling as luma lists 0 (Intra) and 3 (Inter)
        # Cb/Cr Intra: derived from list 0
        # Cb/Cr Inter: derived from list 3
        cb_intra = get_chroma_scaling_list_4x4(
            sps_lists=None,
            pps_lists=None,
            chroma_format_idc=1,
            is_intra=True,
            cb_cr=0  # Cb
        )
        # Should fall back to appropriate default
        assert len(cb_intra) == 16
