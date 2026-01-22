# h264/deblock/tests/test_deblock_8x8.py
"""Tests for H.264 deblocking filter with 8x8 transforms (High Profile).

H.264 Spec Reference: Section 8.7 - Deblocking filter process
                      Section 8.7.2.4 - Filtering for 8x8 transform blocks

When transform_size_8x8_flag is set to 1 (High Profile I_8x8 macroblocks):
- Internal edges are only filtered at 8x8 block boundaries (not 4x4)
- Vertical edges at x=0 and x=8 (instead of x=0,4,8,12 for 4x4)
- Horizontal edges at y=0 and y=8 (instead of y=0,4,8,12 for 4x4)
- bS calculation uses 8x8 block coefficients rather than 4x4 blocks
"""

import pytest
import numpy as np


class TestBoundaryStrength8x8:
    """Tests for boundary strength (bS) calculation with 8x8 transforms."""

    def test_bs_4_for_i_8x8_macroblock(self):
        """bS=4 for all edges within I_8x8 intra macroblock.

        H.264 Spec: For intra macroblocks, bS=4 at all edges.
        """
        from deblock.boundary import calc_boundary_strength_8x8

        # I_8x8 macroblock with transform_size_8x8_flag=1
        bs = calc_boundary_strength_8x8(
            is_intra_p=True,
            is_intra_q=True,
            transform_8x8_p=True,
            transform_8x8_q=True,
            has_coeff_8x8_p=False,
            has_coeff_8x8_q=False,
            mv_p=(0, 0),
            mv_q=(0, 0),
            ref_p=0,
            ref_q=0,
        )
        assert bs == 4

    def test_bs_uses_8x8_block_coefficients(self):
        """bS calculation uses 8x8 block coefficient flags, not 4x4.

        When transform_size_8x8_flag=1, we check if any coefficient
        in the entire 8x8 block is non-zero, not individual 4x4 sub-blocks.
        """
        from deblock.boundary import calc_boundary_strength_8x8

        # Inter block with 8x8 transform, has coefficients
        bs = calc_boundary_strength_8x8(
            is_intra_p=False,
            is_intra_q=False,
            transform_8x8_p=True,
            transform_8x8_q=True,
            has_coeff_8x8_p=True,  # 8x8 block has coefficients
            has_coeff_8x8_q=False,
            mv_p=(0, 0),
            mv_q=(0, 0),
            ref_p=0,
            ref_q=0,
        )
        assert bs == 3

    def test_bs_with_mixed_transform_sizes(self):
        """bS at boundary between 8x8 and 4x4 transform blocks.

        When neighbors have different transform sizes, bS must still
        be calculated correctly using appropriate coefficient flags.

        H.264 Spec: Edge between differently sized transforms still
        uses block-level coefficient checks.
        """
        from deblock.boundary import calc_boundary_strength_8x8

        # Left block uses 4x4, right uses 8x8
        bs = calc_boundary_strength_8x8(
            is_intra_p=False,
            is_intra_q=False,
            transform_8x8_p=False,  # 4x4 transform
            transform_8x8_q=True,   # 8x8 transform
            has_coeff_8x8_p=True,   # For 4x4, this checks relevant 4x4 block
            has_coeff_8x8_q=False,
            mv_p=(0, 0),
            mv_q=(0, 0),
            ref_p=0,
            ref_q=0,
        )
        assert bs == 3

    def test_bs_0_for_similar_8x8_inter_blocks(self):
        """bS=0 when both 8x8 blocks have no coeffs and similar MVs.

        For inter-coded 8x8 blocks with no residual and similar motion.
        """
        from deblock.boundary import calc_boundary_strength_8x8

        bs = calc_boundary_strength_8x8(
            is_intra_p=False,
            is_intra_q=False,
            transform_8x8_p=True,
            transform_8x8_q=True,
            has_coeff_8x8_p=False,
            has_coeff_8x8_q=False,
            mv_p=(0, 0),
            mv_q=(2, 2),  # MV diff < 4 quarter-pixels
            ref_p=0,
            ref_q=0,
        )
        assert bs == 0


class TestEdgeSelection8x8:
    """Tests for filter edge selection with 8x8 transforms."""

    def test_luma_edges_8x8_vertical(self):
        """Only filter at x=0, 8 for 8x8 transforms, not x=4, 12.

        H.264 Spec Section 8.7.2.4: When transform_size_8x8_flag=1,
        internal vertical edges are only filtered at 8-pixel boundaries.
        """
        from deblock.deblock import get_luma_edge_positions_8x8

        v_edges = get_luma_edge_positions_8x8('vertical')

        # Should return edges at x=0 and x=8 only
        edge_x_positions = set(x for x, y in v_edges)

        # Must NOT include x=4 or x=12 (4x4 boundaries)
        assert 4 not in edge_x_positions, "x=4 should NOT be filtered for 8x8"
        assert 12 not in edge_x_positions, "x=12 should NOT be filtered for 8x8"

        # Must include x=0 (MB boundary) and x=8 (8x8 block boundary)
        assert 0 in edge_x_positions, "x=0 (MB boundary) must be filtered"
        assert 8 in edge_x_positions, "x=8 (8x8 boundary) must be filtered"

    def test_luma_edges_8x8_horizontal(self):
        """Only filter at y=0, 8 for 8x8 transforms, not y=4, 12.

        H.264 Spec Section 8.7.2.4: When transform_size_8x8_flag=1,
        internal horizontal edges are only filtered at 8-pixel boundaries.
        """
        from deblock.deblock import get_luma_edge_positions_8x8

        h_edges = get_luma_edge_positions_8x8('horizontal')

        # Should return edges at y=0 and y=8 only
        edge_y_positions = set(y for x, y in h_edges)

        # Must NOT include y=4 or y=12 (4x4 boundaries)
        assert 4 not in edge_y_positions, "y=4 should NOT be filtered for 8x8"
        assert 12 not in edge_y_positions, "y=12 should NOT be filtered for 8x8"

        # Must include y=0 (MB boundary) and y=8 (8x8 block boundary)
        assert 0 in edge_y_positions, "y=0 (MB boundary) must be filtered"
        assert 8 in edge_y_positions, "y=8 (8x8 boundary) must be filtered"

    def test_edge_count_8x8_vs_4x4(self):
        """8x8 transforms filter fewer edges than 4x4.

        4x4: 3 internal x positions (4,8,12) x 4 y positions = 12 internal edges
        8x8: 1 internal x position (8) x 2 y positions = 2 internal edges
        """
        from deblock.deblock import get_luma_edge_positions, get_luma_edge_positions_8x8

        edges_4x4 = get_luma_edge_positions('vertical')
        edges_8x8 = get_luma_edge_positions_8x8('vertical')

        # 8x8 should have fewer edges than 4x4
        assert len(edges_8x8) < len(edges_4x4)

        # Internal edges only (exclude MB boundary at x=0)
        internal_8x8_edges = [(x, y) for x, y in edges_8x8 if x != 0]
        internal_4x4_edges = [(x, y) for x, y in edges_4x4 if x != 0]

        # 8x8 has only 2 internal edge positions vs 12 for 4x4
        assert len(internal_8x8_edges) == 2
        assert len(internal_4x4_edges) == 12
        assert len(internal_8x8_edges) < len(internal_4x4_edges)


class TestDeblocking8x8BlockBoundaries:
    """Tests for deblocking at actual 8x8 block boundaries."""

    def test_deblock_8x8_internal_edge(self):
        """Deblock at x=8 internal edge for 8x8 transform macroblock.

        The filter should be applied at the 8x8 block boundary (x=8)
        but NOT at 4x4 sub-block boundaries (x=4, x=12).
        """
        from deblock.deblock import deblock_macroblock_8x8

        # Create MB with edge between two 8x8 blocks
        # Use values within alpha threshold (alpha=15 for QP=26)
        # to ensure filtering occurs
        luma = np.zeros((16, 16), dtype=np.uint8)
        luma[:, :8] = 125   # Left 8x8 block
        luma[:, 8:] = 135   # Right 8x8 block (diff=10 < alpha=15)

        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        block_info = {
            'is_intra': True,
            'transform_size_8x8_flag': True,
            'has_coeff_8x8': np.zeros(4, dtype=bool),  # 4 8x8 blocks
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        luma_out, _, _ = deblock_macroblock_8x8(
            luma, cb, cr, block_info, qp=26
        )

        # Pixels at x=7 and x=8 should be filtered (smoothed)
        # The difference should be less than original (125 vs 135)
        original_diff = abs(int(luma[8, 7]) - int(luma[8, 8]))
        filtered_diff = abs(int(luma_out[8, 7]) - int(luma_out[8, 8]))
        assert filtered_diff < original_diff, "Edge at x=8 should be filtered"

    def test_no_filter_at_4x4_subblock_boundary_for_8x8(self):
        """Do NOT filter at x=4 for 8x8 transform blocks.

        When transform_size_8x8_flag=1, the 4x4 sub-block boundaries
        (x=4, x=12) should NOT be filtered.
        """
        from deblock.deblock import deblock_macroblock_8x8

        # Create MB with edge at 4x4 boundary within first 8x8 block
        luma = np.zeros((16, 16), dtype=np.uint8)
        luma[:, :4] = 100
        luma[:, 4:8] = 200
        luma[:, 8:] = 128

        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        block_info = {
            'is_intra': True,
            'transform_size_8x8_flag': True,
            'has_coeff_8x8': np.zeros(4, dtype=bool),
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        luma_out, _, _ = deblock_macroblock_8x8(
            luma, cb, cr, block_info, qp=26
        )

        # Edge at x=4 should NOT be filtered for 8x8 transforms
        # Original values should be preserved
        np.testing.assert_array_equal(
            luma_out[:, 3:5],
            luma[:, 3:5],
            err_msg="x=4 edge should NOT be filtered for 8x8 transform"
        )


class TestMixed4x4And8x8Deblocking:
    """Tests for deblocking when 4x4 and 8x8 transforms are mixed."""

    def test_mb_boundary_4x4_to_8x8(self):
        """Filter MB boundary between 4x4 and 8x8 transform macroblocks.

        When left MB uses 4x4 and right MB uses 8x8, the MB boundary
        should still be filtered correctly.
        """
        from deblock.deblock import deblock_mb_boundary_mixed_transform

        left_mb_info = {
            'is_intra': False,
            'transform_size_8x8_flag': False,  # 4x4 transform
            'has_coeff': np.zeros(16, dtype=bool),  # 16 4x4 blocks
            'mvs': np.zeros((16, 2), dtype=np.int32),
            'refs': np.zeros(16, dtype=np.int32),
        }

        right_mb_info = {
            'is_intra': True,
            'transform_size_8x8_flag': True,  # 8x8 transform
            'has_coeff_8x8': np.zeros(4, dtype=bool),  # 4 8x8 blocks
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        # Create frame with two MBs
        luma = np.zeros((16, 32), dtype=np.uint8)
        luma[:, :16] = 100   # Left MB (4x4)
        luma[:, 16:] = 200   # Right MB (8x8)

        bs = deblock_mb_boundary_mixed_transform(
            left_mb_info, right_mb_info, direction='vertical'
        )

        # bS should be 4 because right MB is intra
        assert bs == 4

    def test_edge_within_8x8_neighbor_4x4(self):
        """Internal edge decision when MB has 8x8 but neighbor row has 4x4.

        The edge filtering decision should consider both blocks'
        transform sizes when determining which edges to filter.
        """
        from deblock.deblock import get_filter_edges_for_mb_pair

        current_mb = {
            'transform_size_8x8_flag': True,
        }
        left_mb = {
            'transform_size_8x8_flag': False,
        }

        # Get which edges should be filtered at MB boundary
        edges = get_filter_edges_for_mb_pair(current_mb, left_mb, 'vertical')

        # MB boundary (x=0) should always be filtered
        assert any(x == 0 for x, y in edges)


class TestBsWhenTransformSizeDiffers:
    """Tests for bS calculation when transform_size_8x8_flag differs."""

    def test_bs_i_8x8_vs_i_4x4_boundary(self):
        """bS calculation at boundary between I_8x8 and I_4x4 blocks.

        Both are intra, so bS should be 4 regardless of transform size.
        """
        from deblock.boundary import calc_boundary_strength_mixed_transform

        bs = calc_boundary_strength_mixed_transform(
            is_intra_p=True,
            is_intra_q=True,
            transform_8x8_p=True,   # I_8x8
            transform_8x8_q=False,  # I_4x4
        )
        assert bs == 4

    def test_bs_inter_8x8_with_coeff_vs_inter_4x4_no_coeff(self):
        """bS for inter 8x8 (with coeff) vs inter 4x4 (no coeff).

        When one side has coefficients, bS should be 3.
        The transform size affects which coefficient flag to check.
        """
        from deblock.boundary import calc_boundary_strength_mixed_transform

        bs = calc_boundary_strength_mixed_transform(
            is_intra_p=False,
            is_intra_q=False,
            transform_8x8_p=True,
            transform_8x8_q=False,
            has_coeff_p=True,   # 8x8 block has coefficients
            has_coeff_q=False,  # 4x4 block has no coefficients
            mv_p=(0, 0),
            mv_q=(0, 0),
            ref_p=0,
            ref_q=0,
        )
        assert bs == 3

    def test_bs_mapping_8x8_to_4x4_indices(self):
        """Correct 8x8 to 4x4 block index mapping for bS lookup.

        When checking bS at an edge, need to map 8x8 block index to
        corresponding 4x4 sub-blocks for coefficient flag lookup.

        8x8 blocks:   4x4 blocks:
        [0][1]        [0  1  4  5 ]
        [2][3]        [2  3  6  7 ]
                      [8  9  12 13]
                      [10 11 14 15]

        8x8 block 0 contains 4x4 blocks 0, 1, 2, 3
        8x8 block 1 contains 4x4 blocks 4, 5, 6, 7
        8x8 block 2 contains 4x4 blocks 8, 9, 10, 11
        8x8 block 3 contains 4x4 blocks 12, 13, 14, 15
        """
        from deblock.boundary import get_4x4_blocks_for_8x8

        # Map 8x8 block index to constituent 4x4 block indices
        blocks_8x8_0 = get_4x4_blocks_for_8x8(0)
        blocks_8x8_1 = get_4x4_blocks_for_8x8(1)
        blocks_8x8_2 = get_4x4_blocks_for_8x8(2)
        blocks_8x8_3 = get_4x4_blocks_for_8x8(3)

        assert set(blocks_8x8_0) == {0, 1, 2, 3}
        assert set(blocks_8x8_1) == {4, 5, 6, 7}
        assert set(blocks_8x8_2) == {8, 9, 10, 11}
        assert set(blocks_8x8_3) == {12, 13, 14, 15}


class TestChromaDeblocking8x8:
    """Tests for chroma deblocking with 8x8 transforms."""

    def test_chroma_edges_independent_of_luma_transform_size(self):
        """Chroma deblocking is always at 4x4 chroma block boundaries.

        Even when luma uses 8x8 transforms, chroma always filters
        at its native 4x4 block boundaries (which correspond to
        8x8 luma regions due to 4:2:0 subsampling).

        H.264 Spec: Chroma filtering is independent of transform_size_8x8_flag.
        """
        from deblock.deblock import get_chroma_edge_positions

        # Chroma edges for 8x8 chroma plane (one per MB)
        v_edges = get_chroma_edge_positions('vertical')
        h_edges = get_chroma_edge_positions('horizontal')

        # Should include edge at x=4 (middle of 8x8 chroma block)
        edge_x = set(x for x, y in v_edges)
        edge_y = set(y for x, y in h_edges)

        assert 4 in edge_x, "Chroma should filter at x=4"
        assert 4 in edge_y, "Chroma should filter at y=4"


class TestBlockInfoStructure8x8:
    """Tests for 8x8-specific block_info structure."""

    def test_block_info_has_coeff_8x8_array(self):
        """block_info for 8x8 should have has_coeff_8x8 with 4 elements.

        For 8x8 transforms, coefficient flags are per-8x8-block (4 per MB)
        instead of per-4x4-block (16 per MB).
        """
        from deblock.deblock import create_block_info_8x8

        block_info = create_block_info_8x8(
            is_intra=True,
            coeffs_8x8=[True, False, True, False],  # 4 8x8 blocks
        )

        assert 'has_coeff_8x8' in block_info
        assert len(block_info['has_coeff_8x8']) == 4
        np.testing.assert_array_equal(
            block_info['has_coeff_8x8'],
            [True, False, True, False]
        )
        assert block_info['transform_size_8x8_flag'] is True


class TestCustomAlphaBetaOffsets:
    """Tests for deblocking with custom slice alpha/beta offsets.

    H.264 Spec Section 7.4.3: slice_alpha_c0_offset_div2, slice_beta_offset_div2
    These parameters allow per-slice fine-tuning of deblocking strength.
    Range: [-6, 6] (actual offset is 2x the signaled value)
    """

    def test_alpha_offset_increases_filtering(self):
        """Positive alpha offset increases filtering threshold.

        Higher alpha means more edges will be filtered.
        """
        from deblock.thresholds import get_effective_alpha

        base_qp = 26
        base_alpha = get_effective_alpha(base_qp, offset=0)
        increased_alpha = get_effective_alpha(base_qp, offset=4)

        assert increased_alpha > base_alpha

    def test_alpha_offset_decreases_filtering(self):
        """Negative alpha offset decreases filtering threshold.

        Lower alpha means fewer edges will be filtered.
        """
        from deblock.thresholds import get_effective_alpha

        base_qp = 26
        base_alpha = get_effective_alpha(base_qp, offset=0)
        decreased_alpha = get_effective_alpha(base_qp, offset=-4)

        assert decreased_alpha < base_alpha

    def test_beta_offset_positive(self):
        """Positive beta offset increases internal edge sensitivity."""
        from deblock.thresholds import get_effective_beta

        base_qp = 26
        base_beta = get_effective_beta(base_qp, offset=0)
        increased_beta = get_effective_beta(base_qp, offset=4)

        assert increased_beta > base_beta

    def test_beta_offset_negative(self):
        """Negative beta offset decreases internal edge sensitivity."""
        from deblock.thresholds import get_effective_beta

        base_qp = 26
        base_beta = get_effective_beta(base_qp, offset=0)
        decreased_beta = get_effective_beta(base_qp, offset=-4)

        assert decreased_beta < base_beta

    def test_offset_clamps_effective_qp(self):
        """Effective QP with offset clamps to [0, 51]."""
        from deblock.thresholds import get_effective_alpha, get_effective_beta

        # Very low QP with negative offset should clamp to 0
        alpha_low = get_effective_alpha(base_qp=2, offset=-6)
        # Should use QP=0 threshold, not negative
        assert alpha_low >= 0

        # Very high QP with positive offset should clamp to 51
        alpha_high = get_effective_alpha(base_qp=50, offset=6)
        # Should use QP=51 threshold, not higher
        from deblock.thresholds import ALPHA_TABLE
        assert alpha_high == ALPHA_TABLE[51]

    def test_deblock_8x8_with_custom_offsets(self):
        """Deblock 8x8 transform MB with custom alpha/beta offsets."""
        from deblock.deblock import deblock_macroblock_8x8

        # Create MB with visible edge
        luma = np.zeros((16, 16), dtype=np.uint8)
        luma[:, :8] = 100
        luma[:, 8:] = 200

        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        block_info = {
            'is_intra': True,
            'transform_size_8x8_flag': True,
            'has_coeff_8x8': np.zeros(4, dtype=bool),
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        # Strong filtering with positive offset
        luma_strong, _, _ = deblock_macroblock_8x8(
            luma.copy(), cb, cr, block_info, qp=26,
            alpha_offset=4, beta_offset=4
        )

        # Weak filtering with negative offset
        luma_weak, _, _ = deblock_macroblock_8x8(
            luma.copy(), cb, cr, block_info, qp=26,
            alpha_offset=-4, beta_offset=-4
        )

        # Strong filtering should smooth more
        strong_diff = abs(int(luma_strong[8, 7]) - int(luma_strong[8, 8]))
        weak_diff = abs(int(luma_weak[8, 7]) - int(luma_weak[8, 8]))

        assert strong_diff <= weak_diff

    def test_offset_range_validation(self):
        """Validate offset range [-6, 6] (div2 values, so [-12, 12] effective)."""
        from deblock.thresholds import validate_offset

        # Valid offsets
        for offset in range(-6, 7):
            assert validate_offset(offset) is True

        # Invalid offsets
        with pytest.raises(ValueError):
            validate_offset(-7)

        with pytest.raises(ValueError):
            validate_offset(7)

    def test_chroma_qp_indexing_with_offset(self):
        """Chroma uses different QP mapping with offset applied after.

        H.264 Spec Table 8-15: QpC = Qpc(qpI)
        Offset is applied to QPy first, then mapping is done.
        """
        from deblock.thresholds import get_chroma_qp_with_offset

        # Chroma QP mapping differs from luma
        chroma_qp_base = get_chroma_qp_with_offset(luma_qp=30, offset=0)
        chroma_qp_offset = get_chroma_qp_with_offset(luma_qp=30, offset=4)

        # Chroma QP should increase with offset
        assert chroma_qp_offset > chroma_qp_base

    def test_frame_level_default_offsets_from_pps(self):
        """PPS can specify default alpha/beta offsets for all slices."""
        from deblock.thresholds import get_deblock_params_from_pps

        # Mock PPS with offsets
        pps = {
            'deblocking_filter_control_present_flag': True,
            'default_alpha_c0_offset_div2': 2,
            'default_beta_offset_div2': -1,
        }

        alpha_offset, beta_offset = get_deblock_params_from_pps(pps)

        assert alpha_offset == 4   # 2 * 2
        assert beta_offset == -2   # -1 * 2


class TestDeblockDisableFilterIDC:
    """Tests for disable_deblocking_filter_idc parameter.

    H.264 Spec Section 7.4.3:
    - idc=0: Filter all edges
    - idc=1: Disable all filtering
    - idc=2: Disable filtering at slice boundaries only
    """

    def test_filter_idc_0_all_edges(self):
        """idc=0 filters all edges including slice boundaries."""
        from deblock.deblock import should_filter_edge_with_idc

        # Internal edge
        assert should_filter_edge_with_idc(
            is_slice_boundary=False,
            disable_deblocking_filter_idc=0
        ) is True

        # Slice boundary edge
        assert should_filter_edge_with_idc(
            is_slice_boundary=True,
            disable_deblocking_filter_idc=0
        ) is True

    def test_filter_idc_1_no_filtering(self):
        """idc=1 disables all filtering."""
        from deblock.deblock import should_filter_edge_with_idc

        assert should_filter_edge_with_idc(
            is_slice_boundary=False,
            disable_deblocking_filter_idc=1
        ) is False

        assert should_filter_edge_with_idc(
            is_slice_boundary=True,
            disable_deblocking_filter_idc=1
        ) is False

    def test_filter_idc_2_skip_slice_boundary(self):
        """idc=2 filters all edges except slice boundaries."""
        from deblock.deblock import should_filter_edge_with_idc

        # Internal edge - filter
        assert should_filter_edge_with_idc(
            is_slice_boundary=False,
            disable_deblocking_filter_idc=2
        ) is True

        # Slice boundary - skip
        assert should_filter_edge_with_idc(
            is_slice_boundary=True,
            disable_deblocking_filter_idc=2
        ) is False

    def test_deblock_8x8_respects_filter_idc(self):
        """deblock_macroblock_8x8 should respect filter_idc parameter."""
        from deblock.deblock import deblock_macroblock_8x8

        luma = np.zeros((16, 16), dtype=np.uint8)
        luma[:, :8] = 100
        luma[:, 8:] = 200

        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        block_info = {
            'is_intra': True,
            'transform_size_8x8_flag': True,
            'has_coeff_8x8': np.zeros(4, dtype=bool),
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        # With idc=1, output should match input (no filtering)
        luma_out, _, _ = deblock_macroblock_8x8(
            luma.copy(), cb, cr, block_info, qp=26,
            disable_deblocking_filter_idc=1
        )

        np.testing.assert_array_equal(luma_out, luma)


class TestDeblock8x8EdgeCases:
    """Edge case tests for 8x8 transform deblocking."""

    def test_all_zeros_macroblock(self):
        """Deblocking all-zero MB should produce all zeros."""
        from deblock.deblock import deblock_macroblock_8x8

        luma = np.zeros((16, 16), dtype=np.uint8)
        cb = np.zeros((8, 8), dtype=np.uint8)
        cr = np.zeros((8, 8), dtype=np.uint8)

        block_info = {
            'is_intra': False,
            'transform_size_8x8_flag': True,
            'has_coeff_8x8': np.zeros(4, dtype=bool),
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        luma_out, cb_out, cr_out = deblock_macroblock_8x8(
            luma, cb, cr, block_info, qp=26
        )

        np.testing.assert_array_equal(luma_out, 0)
        np.testing.assert_array_equal(cb_out, 0)
        np.testing.assert_array_equal(cr_out, 0)

    def test_all_255_macroblock(self):
        """Deblocking all-255 MB should produce all 255."""
        from deblock.deblock import deblock_macroblock_8x8

        luma = np.full((16, 16), 255, dtype=np.uint8)
        cb = np.full((8, 8), 255, dtype=np.uint8)
        cr = np.full((8, 8), 255, dtype=np.uint8)

        block_info = {
            'is_intra': False,
            'transform_size_8x8_flag': True,
            'has_coeff_8x8': np.zeros(4, dtype=bool),
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        luma_out, cb_out, cr_out = deblock_macroblock_8x8(
            luma, cb, cr, block_info, qp=26
        )

        np.testing.assert_array_equal(luma_out, 255)
        np.testing.assert_array_equal(cb_out, 255)
        np.testing.assert_array_equal(cr_out, 255)

    def test_extreme_qp_values(self):
        """Test deblocking at extreme QP values (0 and 51)."""
        from deblock.deblock import deblock_macroblock_8x8

        luma = np.zeros((16, 16), dtype=np.uint8)
        luma[:, :8] = 100
        luma[:, 8:] = 200

        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        block_info = {
            'is_intra': True,
            'transform_size_8x8_flag': True,
            'has_coeff_8x8': np.zeros(4, dtype=bool),
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        # QP=0: minimal filtering
        luma_qp0, _, _ = deblock_macroblock_8x8(
            luma.copy(), cb, cr, block_info, qp=0
        )

        # QP=51: maximum filtering
        luma_qp51, _, _ = deblock_macroblock_8x8(
            luma.copy(), cb, cr, block_info, qp=51
        )

        # Higher QP should result in more filtering (smoother edge)
        diff_qp0 = abs(int(luma_qp0[8, 7]) - int(luma_qp0[8, 8]))
        diff_qp51 = abs(int(luma_qp51[8, 7]) - int(luma_qp51[8, 8]))

        assert diff_qp51 <= diff_qp0

    def test_alternating_pattern_8x8(self):
        """Deblock MB with alternating 8x8 block values."""
        from deblock.deblock import deblock_macroblock_8x8

        # Checkerboard pattern at 8x8 block level
        # Use values within alpha threshold (alpha=15 for QP=26)
        luma = np.zeros((16, 16), dtype=np.uint8)
        luma[:8, :8] = 120     # Top-left 8x8
        luma[:8, 8:] = 130     # Top-right 8x8 (diff=10 < alpha=15)
        luma[8:, :8] = 130     # Bottom-left 8x8
        luma[8:, 8:] = 120     # Bottom-right 8x8

        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        block_info = {
            'is_intra': True,
            'transform_size_8x8_flag': True,
            'has_coeff_8x8': np.zeros(4, dtype=bool),
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        luma_out, _, _ = deblock_macroblock_8x8(
            luma, cb, cr, block_info, qp=26
        )

        # Edges at x=8 and y=8 should be filtered
        # Values near boundaries should be smoothed
        assert luma_out[7, 8] != 130 or luma_out[8, 8] != 120

    def test_single_pixel_difference(self):
        """Edge with minimal difference should filter minimally."""
        from deblock.deblock import deblock_macroblock_8x8

        luma = np.full((16, 16), 128, dtype=np.uint8)
        # Just 1 pixel difference at boundary
        luma[:, :8] = 127
        luma[:, 8:] = 129

        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        block_info = {
            'is_intra': True,
            'transform_size_8x8_flag': True,
            'has_coeff_8x8': np.zeros(4, dtype=bool),
            'mvs': np.zeros((4, 2), dtype=np.int32),
            'refs': np.zeros(4, dtype=np.int32),
        }

        luma_out, _, _ = deblock_macroblock_8x8(
            luma, cb, cr, block_info, qp=26
        )

        # With small difference, filtering should be minimal
        # Output should be close to input
        diff = np.abs(luma_out.astype(np.int32) - luma.astype(np.int32))
        assert np.max(diff) <= 2


class TestDeblock8x8InterPrediction:
    """Tests for deblocking 8x8 transform blocks with inter prediction."""

    def test_bs_with_bi_prediction_8x8(self):
        """bS calculation for bi-predicted 8x8 blocks.

        B-frames with bi-prediction have two MVs and two refs per block.
        bS depends on whether either ref differs or either MV differs significantly.
        """
        from deblock.boundary import calc_boundary_strength_8x8_bipred

        # Both references same, MVs similar -> bS=0
        bs = calc_boundary_strength_8x8_bipred(
            is_intra_p=False, is_intra_q=False,
            transform_8x8_p=True, transform_8x8_q=True,
            has_coeff_8x8_p=False, has_coeff_8x8_q=False,
            mv_l0_p=(0, 0), mv_l0_q=(1, 1),  # L0 MVs
            mv_l1_p=(0, 0), mv_l1_q=(1, 1),  # L1 MVs
            ref_l0_p=0, ref_l0_q=0,
            ref_l1_p=1, ref_l1_q=1,
        )
        assert bs == 0

        # Different L0 refs -> bS=2
        bs = calc_boundary_strength_8x8_bipred(
            is_intra_p=False, is_intra_q=False,
            transform_8x8_p=True, transform_8x8_q=True,
            has_coeff_8x8_p=False, has_coeff_8x8_q=False,
            mv_l0_p=(0, 0), mv_l0_q=(0, 0),
            mv_l1_p=(0, 0), mv_l1_q=(0, 0),
            ref_l0_p=0, ref_l0_q=1,  # Different L0 ref
            ref_l1_p=1, ref_l1_q=1,
        )
        assert bs == 2

    def test_bs_direct_mode_8x8(self):
        """bS for 8x8 blocks using B_Direct mode."""
        from deblock.boundary import calc_boundary_strength_8x8_direct

        # Direct mode blocks may have derived MVs
        bs = calc_boundary_strength_8x8_direct(
            is_intra_p=False, is_intra_q=False,
            transform_8x8_p=True, transform_8x8_q=True,
            has_coeff_8x8_p=False, has_coeff_8x8_q=False,
            is_direct_p=True, is_direct_q=True,
            collocated_ref_p=0, collocated_ref_q=0,
        )

        # Direct mode with same collocated ref -> similar to skip
        assert bs in [0, 1, 2]

    def test_bs_p_skip_vs_8x8_intra(self):
        """bS between P_Skip and I_8x8 should be 4."""
        from deblock.boundary import calc_boundary_strength_8x8

        bs = calc_boundary_strength_8x8(
            is_intra_p=False,  # P_Skip
            is_intra_q=True,   # I_8x8
            transform_8x8_p=False,
            transform_8x8_q=True,
            has_coeff_8x8_p=False,
            has_coeff_8x8_q=True,
            mv_p=(0, 0),
            mv_q=(0, 0),
            ref_p=0,
            ref_q=0,
        )

        assert bs == 4  # Intra boundary
