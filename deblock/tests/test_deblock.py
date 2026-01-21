# h264/deblock/tests/test_deblock.py
"""Tests for H.264 deblocking filter.

H.264 Spec Reference: Section 8.7 - Deblocking filter process
"""

import pytest
import numpy as np


class TestBoundaryStrength:
    """Tests for boundary strength (bS) calculation."""

    def test_bs_4_for_intra_boundary(self):
        """bS=4 when either block is intra-coded."""
        from deblock.boundary import calc_boundary_strength

        # Intra block on left, inter on right
        bs = calc_boundary_strength(
            is_intra_p=True, is_intra_q=False,
            has_coeff_p=False, has_coeff_q=False,
            mv_p=(0, 0), mv_q=(0, 0),
            ref_p=0, ref_q=0
        )
        assert bs == 4

    def test_bs_4_for_both_intra(self):
        """bS=4 when both blocks are intra-coded."""
        from deblock.boundary import calc_boundary_strength

        bs = calc_boundary_strength(
            is_intra_p=True, is_intra_q=True,
            has_coeff_p=False, has_coeff_q=False,
            mv_p=(0, 0), mv_q=(0, 0),
            ref_p=0, ref_q=0
        )
        assert bs == 4

    def test_bs_3_when_either_has_coeffs(self):
        """bS=3 when either block has non-zero coefficients."""
        from deblock.boundary import calc_boundary_strength

        bs = calc_boundary_strength(
            is_intra_p=False, is_intra_q=False,
            has_coeff_p=True, has_coeff_q=False,
            mv_p=(0, 0), mv_q=(0, 0),
            ref_p=0, ref_q=0
        )
        assert bs == 3

    def test_bs_2_different_refs(self):
        """bS=2 when reference frames differ."""
        from deblock.boundary import calc_boundary_strength

        bs = calc_boundary_strength(
            is_intra_p=False, is_intra_q=False,
            has_coeff_p=False, has_coeff_q=False,
            mv_p=(0, 0), mv_q=(0, 0),
            ref_p=0, ref_q=1  # Different refs
        )
        assert bs == 2

    def test_bs_1_large_mv_diff(self):
        """bS=1 when MV differs by >= 4 quarter-pixels."""
        from deblock.boundary import calc_boundary_strength

        bs = calc_boundary_strength(
            is_intra_p=False, is_intra_q=False,
            has_coeff_p=False, has_coeff_q=False,
            mv_p=(0, 0), mv_q=(4, 0),  # 4 quarter-pixels = 1 full pixel
            ref_p=0, ref_q=0
        )
        assert bs == 1

    def test_bs_0_similar_blocks(self):
        """bS=0 when blocks are similar (no filtering needed)."""
        from deblock.boundary import calc_boundary_strength

        bs = calc_boundary_strength(
            is_intra_p=False, is_intra_q=False,
            has_coeff_p=False, has_coeff_q=False,
            mv_p=(0, 0), mv_q=(0, 0),
            ref_p=0, ref_q=0
        )
        assert bs == 0


class TestFilterThresholds:
    """Tests for alpha/beta threshold tables."""

    def test_alpha_table_exists(self):
        """Alpha threshold table indexed by QP."""
        from deblock.thresholds import ALPHA_TABLE
        assert len(ALPHA_TABLE) == 52  # QP 0-51

    def test_beta_table_exists(self):
        """Beta threshold table indexed by QP."""
        from deblock.thresholds import BETA_TABLE
        assert len(BETA_TABLE) == 52

    def test_alpha_increases_with_qp(self):
        """Higher QP means more filtering (higher alpha)."""
        from deblock.thresholds import ALPHA_TABLE
        assert ALPHA_TABLE[0] < ALPHA_TABLE[51]

    def test_alpha_at_qp_26(self):
        """Alpha at typical QP=26."""
        from deblock.thresholds import ALPHA_TABLE
        # From H.264 spec table 8-16
        assert ALPHA_TABLE[26] == 15

    def test_beta_at_qp_26(self):
        """Beta at typical QP=26."""
        from deblock.thresholds import BETA_TABLE
        # From H.264 spec table 8-16
        assert BETA_TABLE[26] == 6

    def test_get_effective_qp(self):
        """Effective QP includes slice offsets."""
        from deblock.thresholds import get_effective_qp

        # Base QP with no offset
        qp = get_effective_qp(base_qp=26, offset=0)
        assert qp == 26

        # QP with positive offset
        qp = get_effective_qp(base_qp=26, offset=2)
        assert qp == 28

        # QP clamped to valid range
        qp = get_effective_qp(base_qp=50, offset=5)
        assert qp == 51  # Clamped to max


class TestFilterDecision:
    """Tests for filter on/off decision."""

    def test_no_filter_when_bs_0(self):
        """Don't filter when bS=0."""
        from deblock.filter import should_filter_edge

        result = should_filter_edge(
            bs=0, alpha=7, beta=7,
            p0=100, p1=100, q0=100, q1=100
        )
        assert result is False

    def test_filter_when_bs_nonzero_and_threshold_met(self):
        """Filter when bS>0 and pixel differences within thresholds."""
        from deblock.filter import should_filter_edge

        # Small pixel differences, should filter
        result = should_filter_edge(
            bs=2, alpha=7, beta=7,
            p0=100, p1=102, q0=103, q1=101
        )
        assert result is True

    def test_no_filter_when_diff_exceeds_alpha(self):
        """Don't filter when |p0-q0| >= alpha."""
        from deblock.filter import should_filter_edge

        # Large difference across boundary
        result = should_filter_edge(
            bs=2, alpha=7, beta=7,
            p0=100, p1=102, q0=120, q1=118  # |100-120| = 20 > 7
        )
        assert result is False

    def test_no_filter_when_diff_exceeds_beta(self):
        """Don't filter when |p1-p0| >= beta or |q1-q0| >= beta."""
        from deblock.filter import should_filter_edge

        # Large gradient on p side
        result = should_filter_edge(
            bs=2, alpha=7, beta=7,
            p0=100, p1=120, q0=102, q1=101  # |120-100| = 20 > 7
        )
        assert result is False


class TestLumaFiltering:
    """Tests for luma sample filtering."""

    def test_strong_filter_bs4(self):
        """Strong filtering when bS=4."""
        from deblock.filter import filter_luma_edge_strong

        # Edge pixels: p2 p1 p0 | q0 q1 q2
        p = np.array([50, 60, 70], dtype=np.int32)
        q = np.array([130, 120, 110], dtype=np.int32)

        p_new, q_new = filter_luma_edge_strong(p, q)

        # Strong filter smooths toward the average
        # p0 (70) and q0 (130) should both move toward ~100
        assert p_new[2] > 70   # p0 increased toward q0
        assert q_new[0] < 130  # q0 decreased toward p0

        # Verify significant smoothing occurred
        # New values should be closer to average (100)
        avg = (70 + 130) // 2
        assert abs(p_new[2] - avg) < abs(70 - avg)
        assert abs(q_new[0] - avg) < abs(130 - avg)

    def test_normal_filter_bs1_3(self):
        """Normal filtering when bS=1,2,3."""
        from deblock.filter import filter_luma_edge_normal

        p = np.array([60, 70], dtype=np.int32)  # p1, p0
        q = np.array([130, 120], dtype=np.int32)  # q0, q1

        p0_new, q0_new = filter_luma_edge_normal(p, q, bs=2, tc=4)

        # Normal filter adjusts p0 and q0 only
        # Change is limited by tc (threshold)
        assert abs(p0_new - 70) <= 4
        assert abs(q0_new - 130) <= 4


class TestChromaFiltering:
    """Tests for chroma sample filtering."""

    def test_chroma_filter_applied(self):
        """Chroma filtering uses same bS but different thresholds."""
        from deblock.filter import filter_chroma_edge

        p = np.array([60, 70], dtype=np.int32)
        q = np.array([130, 120], dtype=np.int32)

        p0_new, q0_new = filter_chroma_edge(p, q, bs=2, tc=3)

        # Chroma filter modifies p0 and q0
        assert p0_new != 70 or q0_new != 130


class TestEdgeOrdering:
    """Tests for edge filtering order."""

    def test_vertical_edges_first(self):
        """Vertical edges are filtered before horizontal."""
        from deblock.deblock import get_edge_filter_order

        order = get_edge_filter_order()
        assert order[0] == 'vertical'
        assert order[1] == 'horizontal'

    def test_16_edges_per_direction_luma(self):
        """16 vertical and 16 horizontal edges per luma MB."""
        from deblock.deblock import get_luma_edge_positions

        v_edges = get_luma_edge_positions('vertical')
        h_edges = get_luma_edge_positions('horizontal')

        # 4 edges at x=0,4,8,12 across 16 rows = 16 edge segments
        # But edge at x=0 is MB boundary, internal edges at 4,8,12
        assert len(v_edges) == 16  # 4 columns x 4 rows
        assert len(h_edges) == 16


class TestMBDeblocking:
    """Integration tests for macroblock deblocking."""

    def test_deblock_uniform_mb_no_change(self):
        """Uniform macroblock should not change significantly."""
        from deblock.deblock import deblock_macroblock

        luma = np.full((16, 16), 128, dtype=np.uint8)
        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        # All blocks are inter with no coefficients, same MV
        block_info = {
            'is_intra': False,
            'has_coeff': np.zeros((16,), dtype=bool),
            'mvs': np.zeros((16, 2), dtype=np.int32),
            'refs': np.zeros((16,), dtype=np.int32),
        }

        luma_out, cb_out, cr_out = deblock_macroblock(
            luma, cb, cr, block_info, qp=26
        )

        # Should be nearly unchanged (bS=0 everywhere)
        np.testing.assert_array_almost_equal(luma_out, luma, decimal=0)
