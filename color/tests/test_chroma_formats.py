# h264/color/tests/test_chroma_formats.py
"""TDD RED TESTS: H.264 chroma format variations and profile-specific color handling.

Tests for chroma sampling formats beyond the standard 4:2:0:
- 4:0:0 (Monochrome): No chroma planes
- 4:2:2: Chroma subsampled horizontally only
- 4:4:4: No chroma subsampling
- separate_colour_plane_flag: Treat Y/Cb/Cr as independent planes

H.264 Spec References:
- Section 6.2: Source, decoded, and output picture formats
- Section 7.4.2.1.1: chroma_format_idc semantics
- Section 8.5.7: Chroma samples derivation
- Table 8-15: ChromaArrayType and SubWidthC/SubHeightC

These tests SHOULD FAIL until the corresponding features are implemented.
"""

import numpy as np
import pytest


# =============================================================================
# Test: Monochrome (4:0:0) format handling
# =============================================================================

class TestMonochromeFormat:
    """Tests for 4:0:0 (monochrome) chroma format.

    When chroma_format_idc=0, there are no chroma components.
    The decoder must handle this gracefully.

    H.264 Spec: Section 7.4.2.1.1, Table 6-1
    """

    @pytest.mark.xfail(reason="Monochrome format not implemented")
    def test_monochrome_no_chroma_arrays(self):
        """Monochrome has no chroma arrays (ChromaArrayType=0)."""
        from color.chroma_format import get_chroma_array_type

        chroma_type = get_chroma_array_type(
            chroma_format_idc=0,
            separate_colour_plane_flag=False
        )

        assert chroma_type == 0

    @pytest.mark.xfail(reason="Monochrome format not implemented")
    def test_monochrome_chroma_dimensions_zero(self):
        """Monochrome has zero chroma dimensions."""
        from color.chroma_format import get_chroma_dimensions

        luma_width, luma_height = 1920, 1080
        cb_w, cb_h, cr_w, cr_h = get_chroma_dimensions(
            luma_width, luma_height,
            chroma_format_idc=0
        )

        assert cb_w == 0
        assert cb_h == 0
        assert cr_w == 0
        assert cr_h == 0

    @pytest.mark.xfail(reason="Monochrome format not implemented")
    def test_monochrome_decoded_frame_no_cb_cr(self):
        """Decoded monochrome frame has None for Cb/Cr planes."""
        from decoder import DecodedFrame

        luma = np.full((16, 16), 128, dtype=np.uint8)

        frame = DecodedFrame(
            frame_num=0,
            poc=0,
            luma=luma,
            cb=None,  # No chroma
            cr=None,
            width=16,
            height=16,
            chroma_format_idc=0,
        )

        assert frame.cb is None
        assert frame.cr is None

    @pytest.mark.xfail(reason="Monochrome format not implemented")
    def test_monochrome_to_rgb_grayscale(self):
        """Monochrome to RGB produces grayscale image."""
        from color.chroma_format import monochrome_to_rgb

        luma = np.arange(256, dtype=np.uint8).reshape(16, 16)

        rgb = monochrome_to_rgb(luma)

        assert rgb.shape == (16, 16, 3)
        # All channels should equal luma
        np.testing.assert_array_equal(rgb[:, :, 0], luma)
        np.testing.assert_array_equal(rgb[:, :, 1], luma)
        np.testing.assert_array_equal(rgb[:, :, 2], luma)

    @pytest.mark.xfail(reason="Monochrome format not implemented")
    def test_monochrome_macroblock_only_luma(self):
        """Monochrome macroblock has only 16 4x4 luma blocks (no chroma)."""
        from reconstruct.macroblock import get_block_count_for_chroma_format

        luma_blocks, chroma_blocks = get_block_count_for_chroma_format(
            chroma_format_idc=0
        )

        assert luma_blocks == 16
        assert chroma_blocks == 0

    @pytest.mark.xfail(reason="Monochrome format not implemented")
    def test_monochrome_cbp_no_chroma_bits(self):
        """Coded block pattern for monochrome has no chroma coded bits."""
        from entropy.cbp import decode_cbp_monochrome

        # For monochrome, CBP only indicates luma blocks
        # No bits allocated for chroma
        cbp_luma = decode_cbp_monochrome(cbp_value=15)

        assert cbp_luma == 15  # All luma blocks coded
        # No chroma CBP


# =============================================================================
# Test: 4:2:2 chroma format
# =============================================================================

class TestChroma422Format:
    """Tests for 4:2:2 chroma format.

    In 4:2:2, chroma is subsampled only horizontally:
    - SubWidthC = 2, SubHeightC = 1
    - Chroma planes are (W/2, H) - half width, full height

    H.264 Spec: Table 6-1
    """

    @pytest.mark.xfail(reason="4:2:2 format not implemented")
    def test_422_subsampling_factors(self):
        """4:2:2 has SubWidthC=2, SubHeightC=1."""
        from color.chroma_format import get_subsampling_factors

        sub_width, sub_height = get_subsampling_factors(chroma_format_idc=2)

        assert sub_width == 2
        assert sub_height == 1

    @pytest.mark.xfail(reason="4:2:2 format not implemented")
    def test_422_chroma_dimensions(self):
        """4:2:2 chroma has half width, full height."""
        from color.chroma_format import get_chroma_dimensions

        luma_width, luma_height = 1920, 1080
        cb_w, cb_h, cr_w, cr_h = get_chroma_dimensions(
            luma_width, luma_height,
            chroma_format_idc=2
        )

        assert cb_w == 960
        assert cb_h == 1080
        assert cr_w == 960
        assert cr_h == 1080

    @pytest.mark.xfail(reason="4:2:2 format not implemented")
    def test_422_macroblock_chroma_blocks(self):
        """4:2:2 macroblock has 8 chroma blocks (4 Cb + 4 Cr)."""
        from reconstruct.macroblock import get_block_count_for_chroma_format

        luma_blocks, chroma_blocks = get_block_count_for_chroma_format(
            chroma_format_idc=2
        )

        # Luma: 16 4x4 blocks
        # Chroma: 4 4x4 blocks each for Cb and Cr (8x16 per component)
        assert luma_blocks == 16
        assert chroma_blocks == 8  # 4 Cb + 4 Cr

    @pytest.mark.xfail(reason="4:2:2 format not implemented")
    def test_422_chroma_dc_block_size(self):
        """4:2:2 chroma DC block is 2x4 (after Hadamard)."""
        from transform.hadamard import get_chroma_dc_dimensions

        dc_width, dc_height = get_chroma_dc_dimensions(chroma_format_idc=2)

        # 4:2:2 has 2x4 chroma DC block (half width, double height of 4:2:0)
        assert dc_width == 2
        assert dc_height == 4

    @pytest.mark.xfail(reason="4:2:2 format not implemented")
    def test_422_upsample_horizontal_only(self):
        """4:2:2 upsampling only in horizontal direction."""
        from color.chroma_format import upsample_chroma_422

        # Chroma plane at half width, full height
        chroma = np.array([
            [100, 200],
            [50, 150],
            [75, 175],
            [25, 125],
        ], dtype=np.uint8)

        upsampled = upsample_chroma_422(chroma, target_width=4, target_height=4)

        assert upsampled.shape == (4, 4)
        # Height unchanged, width doubled

    @pytest.mark.xfail(reason="4:2:2 format not implemented")
    def test_422_ycbcr_to_rgb(self):
        """4:2:2 YCbCr to RGB conversion."""
        from color.chroma_format import ycbcr_422_to_rgb

        y = np.full((4, 4), 128, dtype=np.uint8)
        cb = np.full((4, 2), 128, dtype=np.uint8)  # Half width
        cr = np.full((4, 2), 128, dtype=np.uint8)

        rgb = ycbcr_422_to_rgb(y, cb, cr)

        assert rgb.shape == (4, 4, 3)
        # Gray input should produce gray output
        assert np.allclose(rgb[:, :, 0], 128, atol=1)

    @pytest.mark.xfail(reason="4:2:2 format not implemented")
    def test_422_cbp_encoding(self):
        """4:2:2 has different CBP encoding than 4:2:0."""
        from entropy.cbp import get_cbp_table_for_chroma_format

        cbp_table = get_cbp_table_for_chroma_format(chroma_format_idc=2)

        # 4:2:2 CBP can indicate 8 chroma blocks
        assert cbp_table is not None

    @pytest.mark.xfail(reason="4:2:2 format not implemented")
    def test_422_intra_chroma_modes(self):
        """4:2:2 supports same intra chroma prediction modes."""
        from intra.chroma_pred import get_supported_chroma_modes

        modes = get_supported_chroma_modes(chroma_format_idc=2)

        # DC, Horizontal, Vertical, Plane
        assert 0 in modes  # DC
        assert 1 in modes  # Horizontal
        assert 2 in modes  # Vertical
        assert 3 in modes  # Plane


# =============================================================================
# Test: 4:4:4 chroma format
# =============================================================================

class TestChroma444Format:
    """Tests for 4:4:4 chroma format.

    In 4:4:4, no chroma subsampling:
    - SubWidthC = 1, SubHeightC = 1
    - Chroma planes have same size as luma

    H.264 Spec: Table 6-1
    """

    @pytest.mark.xfail(reason="4:4:4 format not implemented")
    def test_444_subsampling_factors(self):
        """4:4:4 has SubWidthC=1, SubHeightC=1 (no subsampling)."""
        from color.chroma_format import get_subsampling_factors

        sub_width, sub_height = get_subsampling_factors(chroma_format_idc=3)

        assert sub_width == 1
        assert sub_height == 1

    @pytest.mark.xfail(reason="4:4:4 format not implemented")
    def test_444_chroma_dimensions_equal_luma(self):
        """4:4:4 chroma has same dimensions as luma."""
        from color.chroma_format import get_chroma_dimensions

        luma_width, luma_height = 1920, 1080
        cb_w, cb_h, cr_w, cr_h = get_chroma_dimensions(
            luma_width, luma_height,
            chroma_format_idc=3
        )

        assert cb_w == luma_width
        assert cb_h == luma_height
        assert cr_w == luma_width
        assert cr_h == luma_height

    @pytest.mark.xfail(reason="4:4:4 format not implemented")
    def test_444_macroblock_full_chroma(self):
        """4:4:4 macroblock has 16 blocks each for Cb and Cr."""
        from reconstruct.macroblock import get_block_count_for_chroma_format

        luma_blocks, chroma_blocks = get_block_count_for_chroma_format(
            chroma_format_idc=3
        )

        # Luma: 16 4x4 blocks
        # Chroma: 16 4x4 blocks each for Cb and Cr (16x16 per component)
        assert luma_blocks == 16
        assert chroma_blocks == 32  # 16 Cb + 16 Cr

    @pytest.mark.xfail(reason="4:4:4 format not implemented")
    def test_444_chroma_dc_block_size(self):
        """4:4:4 chroma DC block is 4x4 (after Hadamard)."""
        from transform.hadamard import get_chroma_dc_dimensions

        dc_width, dc_height = get_chroma_dc_dimensions(chroma_format_idc=3)

        # 4:4:4 has 4x4 chroma DC block (same as luma)
        assert dc_width == 4
        assert dc_height == 4

    @pytest.mark.xfail(reason="4:4:4 format not implemented")
    def test_444_no_upsampling_needed(self):
        """4:4:4 requires no chroma upsampling."""
        from color.chroma_format import upsample_chroma_444

        chroma = np.full((16, 16), 128, dtype=np.uint8)

        upsampled = upsample_chroma_444(chroma, target_width=16, target_height=16)

        # Should return same array (no upsampling)
        np.testing.assert_array_equal(upsampled, chroma)

    @pytest.mark.xfail(reason="4:4:4 format not implemented")
    def test_444_ycbcr_to_rgb_no_upsample(self):
        """4:4:4 YCbCr to RGB without upsampling."""
        from color.chroma_format import ycbcr_444_to_rgb

        y = np.full((16, 16), 128, dtype=np.uint8)
        cb = np.full((16, 16), 128, dtype=np.uint8)  # Same size
        cr = np.full((16, 16), 128, dtype=np.uint8)

        rgb = ycbcr_444_to_rgb(y, cb, cr)

        assert rgb.shape == (16, 16, 3)
        # Gray input should produce gray output
        assert np.allclose(rgb, 128, atol=1)

    @pytest.mark.xfail(reason="4:4:4 format not implemented")
    def test_444_supports_8x8_chroma_transform(self):
        """4:4:4 High profile supports 8x8 transform for chroma."""
        from transform.transform_size import supports_8x8_chroma

        # 4:4:4 in High profile can use 8x8 transform for chroma
        assert supports_8x8_chroma(
            chroma_format_idc=3,
            transform_8x8_mode_flag=True
        )

    @pytest.mark.xfail(reason="4:4:4 format not implemented")
    def test_444_intra_prediction_8x8_chroma(self):
        """4:4:4 supports 8x8 intra prediction for chroma."""
        from intra.intra_8x8 import supports_8x8_intra_chroma

        result = supports_8x8_intra_chroma(chroma_format_idc=3)

        assert result is True


# =============================================================================
# Test: separate_colour_plane_flag handling
# =============================================================================

class TestSeparateColourPlaneFlag:
    """Tests for separate_colour_plane_flag (High 4:4:4 Predictive profile).

    When separate_colour_plane_flag=1 with chroma_format_idc=3:
    - Y, Cb, Cr are coded as separate monochrome pictures
    - ChromaArrayType = 0 (treated as monochrome for each plane)

    H.264 Spec: Section 7.4.2.1.1
    """

    @pytest.mark.xfail(reason="separate_colour_plane_flag not implemented")
    def test_separate_planes_chroma_array_type_zero(self):
        """With separate_colour_plane_flag, ChromaArrayType=0."""
        from color.chroma_format import get_chroma_array_type

        chroma_type = get_chroma_array_type(
            chroma_format_idc=3,
            separate_colour_plane_flag=True
        )

        # Each plane coded independently, treated as monochrome
        assert chroma_type == 0

    @pytest.mark.xfail(reason="separate_colour_plane_flag not implemented")
    def test_separate_planes_three_slice_groups(self):
        """Separate planes decode as three independent slices."""
        from slice.slice_header import get_colour_plane_id

        # colour_plane_id indicates which plane (0=Y, 1=Cb, 2=Cr)
        for plane_id in [0, 1, 2]:
            result = get_colour_plane_id(plane_id)
            assert result in ['Y', 'Cb', 'Cr']

    @pytest.mark.xfail(reason="separate_colour_plane_flag not implemented")
    def test_separate_planes_sps_flag_parsing(self):
        """SPS correctly parses separate_colour_plane_flag."""
        from parameters.sps import SPS

        sps = SPS(
            chroma_format_idc=3,
            separate_colour_plane_flag=True,
        )

        assert sps.chroma_format_idc == 3
        assert sps.separate_colour_plane_flag is True

    @pytest.mark.xfail(reason="separate_colour_plane_flag not implemented")
    def test_separate_planes_each_plane_full_resolution(self):
        """Each plane has full luma resolution."""
        from color.chroma_format import get_plane_dimensions_separate

        luma_width, luma_height = 1920, 1080

        y_dims = get_plane_dimensions_separate(luma_width, luma_height, plane_id=0)
        cb_dims = get_plane_dimensions_separate(luma_width, luma_height, plane_id=1)
        cr_dims = get_plane_dimensions_separate(luma_width, luma_height, plane_id=2)

        # All planes have same dimensions
        assert y_dims == (1920, 1080)
        assert cb_dims == (1920, 1080)
        assert cr_dims == (1920, 1080)

    @pytest.mark.xfail(reason="separate_colour_plane_flag not implemented")
    def test_separate_planes_independent_qp(self):
        """Each plane can have independent QP in separate mode."""
        from slice.slice_header import parse_slice_qp_separate_planes

        # With separate planes, each slice can have different QP
        y_qp = parse_slice_qp_separate_planes(slice_qp_y=26, plane_id=0)
        cb_qp = parse_slice_qp_separate_planes(slice_qp_y=26, plane_id=1)
        cr_qp = parse_slice_qp_separate_planes(slice_qp_y=26, plane_id=2)

        assert y_qp == 26
        # Cb and Cr may have offset
        assert isinstance(cb_qp, int)
        assert isinstance(cr_qp, int)


# =============================================================================
# Test: chroma_qp_index_offset handling
# =============================================================================

class TestChromaQPIndexOffset:
    """Tests for chroma_qp_index_offset from PPS.

    chroma_qp_index_offset adjusts chroma QP relative to luma QP.
    Range: -12 to +12

    H.264 Spec: Section 7.4.2.2, Table 8-15
    """

    @pytest.mark.xfail(reason="chroma_qp_index_offset handling not fully implemented")
    def test_chroma_qp_offset_zero_default(self):
        """Default chroma_qp_index_offset is 0."""
        from parameters.pps import PPS

        pps = PPS()

        assert pps.chroma_qp_index_offset == 0

    @pytest.mark.xfail(reason="chroma_qp_index_offset handling not fully implemented")
    def test_chroma_qp_offset_positive(self):
        """Positive offset increases chroma QP."""
        from dequant.chroma_qp import calculate_chroma_qp

        luma_qp = 26
        offset = 6

        chroma_qp = calculate_chroma_qp(luma_qp, chroma_qp_index_offset=offset)

        # QPc = Clip3(0, 51, QPy + offset) then apply Table 8-15
        # For QPy=26, offset=6: QPI=32, QPc from table
        assert chroma_qp > calculate_chroma_qp(luma_qp, chroma_qp_index_offset=0)

    @pytest.mark.xfail(reason="chroma_qp_index_offset handling not fully implemented")
    def test_chroma_qp_offset_negative(self):
        """Negative offset decreases chroma QP."""
        from dequant.chroma_qp import calculate_chroma_qp

        luma_qp = 26
        offset = -6

        chroma_qp = calculate_chroma_qp(luma_qp, chroma_qp_index_offset=offset)

        # Lower QPI leads to lower QPc
        assert chroma_qp < calculate_chroma_qp(luma_qp, chroma_qp_index_offset=0)

    @pytest.mark.xfail(reason="chroma_qp_index_offset handling not fully implemented")
    def test_chroma_qp_offset_minimum(self):
        """chroma_qp_index_offset minimum is -12."""
        from dequant.chroma_qp import calculate_chroma_qp

        result = calculate_chroma_qp(luma_qp=26, chroma_qp_index_offset=-12)

        # Should not crash, QPc still valid
        assert 0 <= result <= 51

    @pytest.mark.xfail(reason="chroma_qp_index_offset handling not fully implemented")
    def test_chroma_qp_offset_maximum(self):
        """chroma_qp_index_offset maximum is +12."""
        from dequant.chroma_qp import calculate_chroma_qp

        result = calculate_chroma_qp(luma_qp=26, chroma_qp_index_offset=12)

        # Should not crash, QPc still valid
        assert 0 <= result <= 51

    @pytest.mark.xfail(reason="chroma_qp_index_offset handling not fully implemented")
    def test_chroma_qp_offset_clipping_at_low_qp(self):
        """QPI clipped to 0 when luma_qp + offset < 0."""
        from dequant.chroma_qp import calculate_chroma_qp

        # With QPy=5 and offset=-12, QPI would be -7, clipped to 0
        result = calculate_chroma_qp(luma_qp=5, chroma_qp_index_offset=-12)

        # QPc = QPc(0) = 0
        assert result == 0

    @pytest.mark.xfail(reason="chroma_qp_index_offset handling not fully implemented")
    def test_chroma_qp_offset_clipping_at_high_qp(self):
        """QPI clipped to 51 when luma_qp + offset > 51."""
        from dequant.chroma_qp import calculate_chroma_qp

        # With QPy=45 and offset=12, QPI would be 57, clipped to 51
        result = calculate_chroma_qp(luma_qp=45, chroma_qp_index_offset=12)

        # QPc = QPc(51) = 39
        assert result == 39

    @pytest.mark.xfail(reason="chroma_qp_index_offset handling not fully implemented")
    def test_chroma_qp_table_mapping(self):
        """Verify chroma QP mapping table (Table 8-15)."""
        from dequant.chroma_qp import get_qpc_from_table

        # Some known mappings from H.264 Table 8-15
        assert get_qpc_from_table(0) == 0
        assert get_qpc_from_table(29) == 29
        assert get_qpc_from_table(30) == 29  # First difference
        assert get_qpc_from_table(51) == 39  # Maximum

    @pytest.mark.xfail(reason="chroma_qp_index_offset handling not fully implemented")
    def test_chroma_qp_applies_to_dequant(self):
        """Chroma QP offset affects dequantization."""
        from dequant.dequant import dequant_dc_2x2
        from dequant.chroma_qp import calculate_chroma_qp

        dc_coeffs = np.ones((2, 2), dtype=np.int32) * 10

        # Without offset
        chroma_qp_0 = calculate_chroma_qp(26, chroma_qp_index_offset=0)
        result_0 = dequant_dc_2x2(dc_coeffs, qp=chroma_qp_0)

        # With positive offset (higher QP = larger dequant values)
        chroma_qp_6 = calculate_chroma_qp(26, chroma_qp_index_offset=6)
        result_6 = dequant_dc_2x2(dc_coeffs, qp=chroma_qp_6)

        # Higher QP produces larger dequantized values
        assert np.sum(np.abs(result_6)) > np.sum(np.abs(result_0))


# =============================================================================
# Test: second_chroma_qp_index_offset (High profile)
# =============================================================================

class TestSecondChromaQPIndexOffset:
    """Tests for second_chroma_qp_index_offset (High profile feature).

    High profile adds second_chroma_qp_index_offset for Cr component,
    allowing different QP offsets for Cb and Cr.

    H.264 Spec: Section 7.4.2.2
    """

    @pytest.mark.xfail(reason="second_chroma_qp_index_offset not implemented")
    def test_second_offset_pps_field_exists(self):
        """PPS should have second_chroma_qp_index_offset field."""
        from parameters.pps import PPS

        pps = PPS()

        assert hasattr(pps, 'second_chroma_qp_index_offset')

    @pytest.mark.xfail(reason="second_chroma_qp_index_offset not implemented")
    def test_second_offset_default_equals_first(self):
        """Default second_chroma_qp_index_offset equals first offset."""
        from parameters.pps import PPS

        pps = PPS(chroma_qp_index_offset=3)

        # When not explicitly set, second offset defaults to first
        assert pps.second_chroma_qp_index_offset == pps.chroma_qp_index_offset

    @pytest.mark.xfail(reason="second_chroma_qp_index_offset not implemented")
    def test_separate_cb_cr_qp_calculation(self):
        """Cb and Cr can have different QP with second offset."""
        from dequant.chroma_qp import calculate_cb_qp, calculate_cr_qp

        luma_qp = 26
        cb_offset = 2
        cr_offset = -2

        cb_qp = calculate_cb_qp(
            luma_qp,
            chroma_qp_index_offset=cb_offset
        )
        cr_qp = calculate_cr_qp(
            luma_qp,
            second_chroma_qp_index_offset=cr_offset
        )

        assert cb_qp != cr_qp

    @pytest.mark.xfail(reason="second_chroma_qp_index_offset not implemented")
    def test_second_offset_only_high_profile(self):
        """second_chroma_qp_index_offset only present in High profile PPS."""
        from parameters.pps import should_parse_second_chroma_offset

        # Baseline profile - no second offset
        assert not should_parse_second_chroma_offset(profile_idc=66)

        # Main profile - no second offset
        assert not should_parse_second_chroma_offset(profile_idc=77)

        # High profile - has second offset
        assert should_parse_second_chroma_offset(profile_idc=100)

    @pytest.mark.xfail(reason="second_chroma_qp_index_offset not implemented")
    def test_second_offset_affects_cr_dequant(self):
        """second_chroma_qp_index_offset affects Cr dequantization."""
        from dequant.dequant import dequant_dc_2x2
        from dequant.chroma_qp import calculate_cr_qp

        dc_coeffs = np.ones((2, 2), dtype=np.int32) * 10

        # Same first offset, different second offset
        cr_qp_low = calculate_cr_qp(26, second_chroma_qp_index_offset=-6)
        cr_qp_high = calculate_cr_qp(26, second_chroma_qp_index_offset=6)

        result_low = dequant_dc_2x2(dc_coeffs, qp=cr_qp_low)
        result_high = dequant_dc_2x2(dc_coeffs, qp=cr_qp_high)

        # Higher QP = larger dequantized values
        assert np.sum(np.abs(result_high)) > np.sum(np.abs(result_low))

    @pytest.mark.xfail(reason="second_chroma_qp_index_offset not implemented")
    def test_second_offset_range_validation(self):
        """second_chroma_qp_index_offset must be in [-12, +12]."""
        from parameters.pps import validate_second_chroma_offset

        # Valid values
        assert validate_second_chroma_offset(-12) is True
        assert validate_second_chroma_offset(0) is True
        assert validate_second_chroma_offset(12) is True

        # Invalid values (should raise or return False)
        assert validate_second_chroma_offset(-13) is False
        assert validate_second_chroma_offset(13) is False


# =============================================================================
# Test: Chroma format integration
# =============================================================================

class TestChromaFormatIntegration:
    """Integration tests for chroma format handling across the decoder."""

    @pytest.mark.xfail(reason="Multi-format chroma support not implemented")
    def test_decoder_accepts_all_chroma_formats(self):
        """Decoder should accept all valid chroma_format_idc values."""
        from decoder import H264Decoder

        decoder = H264Decoder()

        # Valid chroma formats: 0, 1, 2, 3
        for chroma_format in [0, 1, 2, 3]:
            decoder.configure_chroma_format(chroma_format_idc=chroma_format)
            assert decoder.chroma_format_idc == chroma_format

    @pytest.mark.xfail(reason="Multi-format chroma support not implemented")
    def test_chroma_format_from_sps(self):
        """Decoder extracts chroma format from SPS."""
        from parameters.sps import SPS
        from decoder import DecoderState

        sps = SPS(
            profile_idc=100,  # High profile
            chroma_format_idc=2,  # 4:2:2
        )

        state = DecoderState()
        state.apply_sps(sps)

        assert state.chroma_format_idc == 2

    @pytest.mark.xfail(reason="Multi-format chroma support not implemented")
    def test_frame_buffer_allocation_by_chroma_format(self):
        """Frame buffers allocated based on chroma format."""
        from decoder import DecoderState
        from parameters.sps import SPS

        # 4:2:0 (default)
        sps_420 = SPS(
            pic_width_in_mbs_minus1=7,
            pic_height_in_map_units_minus1=5,
            frame_mbs_only_flag=True,
            chroma_format_idc=1,
        )
        state_420 = DecoderState()
        state_420.allocate_frame_buffers(sps_420)

        # 4:4:4
        sps_444 = SPS(
            pic_width_in_mbs_minus1=7,
            pic_height_in_map_units_minus1=5,
            frame_mbs_only_flag=True,
            chroma_format_idc=3,
        )
        state_444 = DecoderState()
        state_444.allocate_frame_buffers(sps_444)

        # 4:2:0 chroma is 1/4 of luma
        assert state_420.frame_cb.shape == (48, 64)

        # 4:4:4 chroma equals luma
        assert state_444.frame_cb.shape == (96, 128)

    @pytest.mark.xfail(reason="Multi-format chroma support not implemented")
    def test_crop_parameters_scale_by_chroma_format(self):
        """Frame crop parameters scale differently by chroma format."""
        from parameters.sps import SPS

        # 4:2:0: crop units are 2x2 for chroma
        sps_420 = SPS(
            chroma_format_idc=1,
            frame_cropping_flag=True,
            frame_crop_right_offset=4,  # 8 luma pixels
            frame_crop_bottom_offset=4,  # 8 luma pixels
        )
        assert sps_420.get_chroma_crop_right() == 4  # 4 chroma pixels

        # 4:4:4: crop units are 1x1 for chroma
        sps_444 = SPS(
            chroma_format_idc=3,
            frame_cropping_flag=True,
            frame_crop_right_offset=4,  # 4 luma pixels
            frame_crop_bottom_offset=4,  # 4 luma pixels
        )
        assert sps_444.get_chroma_crop_right() == 4  # Same as luma


# =============================================================================
# Test: Chroma intra prediction modes
# =============================================================================

class TestChromaIntraPrediction:
    """Tests for chroma intra prediction in different chroma formats."""

    @pytest.mark.xfail(reason="Chroma format-specific prediction not implemented")
    def test_chroma_pred_block_size_420(self):
        """4:2:0 chroma prediction uses 8x8 blocks."""
        from intra.chroma_pred import get_chroma_pred_block_size

        block_w, block_h = get_chroma_pred_block_size(chroma_format_idc=1)

        assert block_w == 8
        assert block_h == 8

    @pytest.mark.xfail(reason="Chroma format-specific prediction not implemented")
    def test_chroma_pred_block_size_422(self):
        """4:2:2 chroma prediction uses 8x16 blocks."""
        from intra.chroma_pred import get_chroma_pred_block_size

        block_w, block_h = get_chroma_pred_block_size(chroma_format_idc=2)

        assert block_w == 8
        assert block_h == 16

    @pytest.mark.xfail(reason="Chroma format-specific prediction not implemented")
    def test_chroma_pred_block_size_444(self):
        """4:4:4 chroma prediction uses 16x16 blocks."""
        from intra.chroma_pred import get_chroma_pred_block_size

        block_w, block_h = get_chroma_pred_block_size(chroma_format_idc=3)

        assert block_w == 16
        assert block_h == 16

    @pytest.mark.xfail(reason="Chroma format-specific prediction not implemented")
    def test_chroma_dc_prediction_422(self):
        """DC prediction for 4:2:2 (8x16 block)."""
        from intra.chroma_pred import predict_chroma_dc_422

        # Left neighbor (16 pixels)
        left = np.full(16, 100, dtype=np.uint8)
        # Top neighbor (8 pixels)
        top = np.full(8, 200, dtype=np.uint8)

        pred = predict_chroma_dc_422(left, top, left_available=True, top_available=True)

        assert pred.shape == (16, 8)
        # DC should be average of available neighbors
        expected_dc = (100 * 16 + 200 * 8) // 24
        assert np.allclose(pred, expected_dc, atol=1)

    @pytest.mark.xfail(reason="Chroma format-specific prediction not implemented")
    def test_chroma_plane_prediction_444(self):
        """Plane prediction for 4:4:4 (16x16 block)."""
        from intra.chroma_pred import predict_chroma_plane_444

        # Neighbors for 16x16 block
        left = np.arange(16, dtype=np.uint8) * 8 + 64
        top = np.arange(16, dtype=np.uint8) * 8 + 64
        top_left = np.uint8(64)

        pred = predict_chroma_plane_444(left, top, top_left)

        assert pred.shape == (16, 16)
        assert pred.dtype == np.uint8
