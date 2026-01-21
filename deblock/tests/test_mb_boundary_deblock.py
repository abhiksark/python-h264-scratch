# h264/deblock/tests/test_mb_boundary_deblock.py
"""RED TESTS: Macroblock boundary deblocking.

The deblocking filter applies to edges between macroblocks, not just
internal 4x4 block edges. MB boundary filtering requires neighbor info.

These tests SHOULD FAIL until MB boundary deblocking is implemented.
"""

import pytest
import numpy as np


class TestMBBoundaryStrength:
    """Tests for bS calculation at MB boundaries."""

    def test_bs_at_left_mb_boundary(self):
        """Calculate bS using left neighbor MB info."""
        from deblock.boundary import calc_mb_boundary_strength

        # Current MB is inter, left neighbor is intra
        bs = calc_mb_boundary_strength(
            current_mb={'is_intra': False, 'has_coeff': False, 'mv': (0, 0), 'ref': 0},
            neighbor_mb={'is_intra': True, 'has_coeff': False, 'mv': (0, 0), 'ref': 0},
            is_mb_edge=True,
        )

        # Intra neighbor means bS=4
        assert bs == 4

    def test_bs_at_top_mb_boundary(self):
        """Calculate bS using top neighbor MB info."""
        from deblock.boundary import calc_mb_boundary_strength

        # Both inter, but different refs
        bs = calc_mb_boundary_strength(
            current_mb={'is_intra': False, 'has_coeff': False, 'mv': (0, 0), 'ref': 0},
            neighbor_mb={'is_intra': False, 'has_coeff': False, 'mv': (0, 0), 'ref': 1},
            is_mb_edge=True,
        )

        assert bs == 2  # Different refs

    def test_mb_boundary_uses_edge_blocks(self):
        """MB boundary bS uses the 4x4 blocks at the edge."""
        from deblock.boundary import get_boundary_block_pairs

        # For vertical edge at left MB boundary (x=0)
        pairs = get_boundary_block_pairs(direction='vertical', edge_pos=0)

        # Should return pairs of (neighbor_block, current_block) indices
        # For left boundary, neighbor blocks are 5,7,13,15 (right edge of left MB)
        # and current blocks are 0,2,8,10 (left edge of current MB)
        assert len(pairs) == 4  # 4 block pairs along MB edge


class TestMBEdgeFiltering:
    """Tests for filtering at MB boundaries."""

    def test_filter_left_mb_edge(self):
        """Filter vertical edge at left MB boundary."""
        from deblock.deblock import filter_mb_boundary

        # Two adjacent MBs
        left_mb_luma = np.full((16, 16), 120, dtype=np.uint8)
        current_mb_luma = np.full((16, 16), 130, dtype=np.uint8)

        # Filter the boundary between them
        left_filtered, current_filtered = filter_mb_boundary(
            neighbor_luma=left_mb_luma,
            current_luma=current_mb_luma,
            direction='vertical',
            bs=4,
            alpha=15,
            beta=6,
        )

        # Right edge of left MB should change
        assert left_filtered[0, 15] != 120 or current_filtered[0, 0] != 130

    def test_filter_top_mb_edge(self):
        """Filter horizontal edge at top MB boundary."""
        from deblock.deblock import filter_mb_boundary

        top_mb_luma = np.full((16, 16), 125, dtype=np.uint8)
        current_mb_luma = np.full((16, 16), 130, dtype=np.uint8)

        top_filtered, current_filtered = filter_mb_boundary(
            neighbor_luma=top_mb_luma,
            current_luma=current_mb_luma,
            direction='horizontal',
            bs=3,
            alpha=15,
            beta=6,
        )

        # Bottom edge of top MB should change
        assert top_filtered[15, 0] != 125 or current_filtered[0, 0] != 130


class TestFrameMBBoundaryDeblock:
    """Tests for frame-level MB boundary filtering."""

    def test_deblock_frame_filters_mb_boundaries(self):
        """deblock_frame should filter between MBs."""
        from deblock.deblock import deblock_frame

        # 32x32 frame = 2x2 MBs
        luma = np.zeros((32, 32), dtype=np.uint8)
        # Left two MBs = 120, right two MBs = 130
        luma[:, :16] = 120
        luma[:, 16:] = 130

        cb = np.full((16, 16), 128, dtype=np.uint8)
        cr = np.full((16, 16), 128, dtype=np.uint8)

        luma_out, _, _ = deblock_frame(
            luma=luma, cb=cb, cr=cr, mb_info=None, qp=26
        )

        # The vertical boundary at x=16 should be filtered
        assert luma_out[0, 15] != 120 or luma_out[0, 16] != 130, \
            "MB boundary should be filtered"

    def test_deblock_respects_left_neighbor(self):
        """Deblocking at x=0 edge should use left MB's rightmost blocks."""
        from deblock.deblock import deblock_macroblock_with_neighbors

        # Create frame context
        frame_luma = np.zeros((32, 32), dtype=np.uint8)
        frame_luma[:, :16] = 100  # Left MB
        frame_luma[:, 16:] = 110  # Right MB (current)

        # Deblock right MB (1, 0) with left neighbor info
        deblock_macroblock_with_neighbors(
            frame_luma=frame_luma,
            mb_x=1,
            mb_y=0,
            current_info={'is_intra': True},
            left_info={'is_intra': True},
            top_info=None,
            qp=26,
        )

        # Left edge of current MB should be filtered
        assert frame_luma[0, 16] != 110 or frame_luma[0, 15] != 100

    def test_deblock_respects_top_neighbor(self):
        """Deblocking at y=0 edge should use top MB's bottom blocks."""
        from deblock.deblock import deblock_macroblock_with_neighbors

        frame_luma = np.zeros((32, 32), dtype=np.uint8)
        frame_luma[:16, :] = 100  # Top MB
        frame_luma[16:, :] = 110  # Bottom MB (current)

        deblock_macroblock_with_neighbors(
            frame_luma=frame_luma,
            mb_x=0,
            mb_y=1,
            current_info={'is_intra': True},
            left_info=None,
            top_info={'is_intra': True},
            qp=26,
        )

        # Top edge of current MB should be filtered
        assert frame_luma[16, 0] != 110 or frame_luma[15, 0] != 100


class TestMBBoundaryOrder:
    """Tests for correct filtering order at MB boundaries."""

    def test_vertical_before_horizontal_at_boundary(self):
        """Vertical MB edges filtered before horizontal."""
        from deblock.deblock import get_mb_edge_filter_order

        order = get_mb_edge_filter_order()

        # Should filter: left boundary (V), internal V, top boundary (H), internal H
        assert order[0] == 'left_boundary'
        assert order[1] == 'internal_vertical'
        assert order[2] == 'top_boundary'
        assert order[3] == 'internal_horizontal'

    def test_no_left_boundary_for_first_column(self):
        """First column MBs have no left boundary to filter."""
        from deblock.deblock import should_filter_mb_edge

        assert not should_filter_mb_edge(mb_x=0, mb_y=0, direction='left')
        assert should_filter_mb_edge(mb_x=1, mb_y=0, direction='left')

    def test_no_top_boundary_for_first_row(self):
        """First row MBs have no top boundary to filter."""
        from deblock.deblock import should_filter_mb_edge

        assert not should_filter_mb_edge(mb_x=0, mb_y=0, direction='top')
        assert should_filter_mb_edge(mb_x=0, mb_y=1, direction='top')
