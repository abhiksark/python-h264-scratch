# h264/deblock/tests/test_chroma_deblock.py
"""RED TESTS: Chroma plane deblocking.

The H.264 deblocking filter applies to both luma and chroma planes.
Chroma uses the same bS but different thresholds (indexed by QPC).

These tests SHOULD FAIL until chroma deblocking is fully implemented.
"""

import pytest
import numpy as np


class TestChromaThresholds:
    """Tests for chroma-specific filter thresholds."""

    def test_chroma_qp_table_exists(self):
        """QPC table maps luma QP to chroma QP."""
        from deblock.thresholds import QPC_TABLE

        # H.264 Table 8-15: QPC as function of qPI
        assert len(QPC_TABLE) == 52
        # QPC = QP for QP <= 29
        assert QPC_TABLE[26] == 26
        # QPC diverges for higher QP
        assert QPC_TABLE[51] < 51

    def test_get_chroma_qp(self):
        """Function to get chroma QP from luma QP."""
        from deblock.thresholds import get_chroma_qp

        # For low QP, chroma QP equals luma QP
        assert get_chroma_qp(20) == 20

        # For high QP, chroma QP is lower
        qpc = get_chroma_qp(45)
        assert qpc < 45

    def test_chroma_alpha_uses_qpc(self):
        """Chroma filtering uses alpha[QPC], not alpha[QP]."""
        from deblock.thresholds import get_alpha, get_chroma_qp

        qp = 40
        qpc = get_chroma_qp(qp)

        # Chroma should use QPC-indexed threshold
        alpha_luma = get_alpha(qp)
        alpha_chroma = get_alpha(qpc)

        # At high QP, these will differ
        assert alpha_chroma <= alpha_luma


class TestChromaEdgeFiltering:
    """Tests for filtering chroma block edges."""

    def test_filter_chroma_vertical_edge(self):
        """Filter vertical edge in chroma plane."""
        from deblock.filter import filter_chroma_vertical_edge

        # 8x8 chroma block with edge at x=4
        cb = np.zeros((8, 8), dtype=np.uint8)
        cb[:, :4] = 120
        cb[:, 4:] = 130

        filtered = filter_chroma_vertical_edge(
            cb, edge_x=4, bs=2, alpha=15, beta=6, tc=3
        )

        # Edge should be smoothed
        assert filtered[0, 3] != 120 or filtered[0, 4] != 130

    def test_filter_chroma_horizontal_edge(self):
        """Filter horizontal edge in chroma plane."""
        from deblock.filter import filter_chroma_horizontal_edge

        cr = np.zeros((8, 8), dtype=np.uint8)
        cr[:4, :] = 120
        cr[4:, :] = 130

        filtered = filter_chroma_horizontal_edge(
            cr, edge_y=4, bs=2, alpha=15, beta=6, tc=3
        )

        assert filtered[3, 0] != 120 or filtered[4, 0] != 130

    def test_chroma_uses_same_bs_as_luma(self):
        """Chroma edges use same bS as corresponding luma edges."""
        from deblock.deblock import get_chroma_bs_from_luma

        # bS is determined by luma block properties
        # Luma has 4x4 bS values (one per 4x4 block in 16x16 MB)
        # Chroma 4x4 block corresponds to 2x2 group of luma 4x4 blocks
        luma_bs = np.array([
            [4, 3, 2, 1],
            [3, 2, 1, 0],
            [2, 1, 0, 1],
            [1, 0, 1, 2],
        ])

        chroma_bs = get_chroma_bs_from_luma(luma_bs)

        # Chroma takes max bS from corresponding 2x2 luma blocks
        assert chroma_bs[0, 0] == 4  # max of top-left 2x2: max(4,3,3,2) = 4
        assert chroma_bs[0, 1] == 2  # max of top-right 2x2: max(2,1,1,0) = 2


class TestChromaMBDeblocking:
    """Tests for complete chroma MB deblocking."""

    def test_deblock_cb_plane(self):
        """Deblock entire Cb plane of macroblock."""
        from deblock.deblock import deblock_chroma_mb

        cb = np.zeros((8, 8), dtype=np.uint8)
        # Create blocking pattern
        cb[0:4, 0:4] = 120
        cb[0:4, 4:8] = 125
        cb[4:8, 0:4] = 125
        cb[4:8, 4:8] = 120

        block_info = {
            'is_intra': True,
            'has_coeff': np.zeros(4, dtype=bool),
        }

        filtered = deblock_chroma_mb(cb, block_info, qp=26)

        # Edges should be filtered
        assert not np.array_equal(filtered, cb)

    def test_deblock_cr_plane(self):
        """Deblock entire Cr plane of macroblock."""
        from deblock.deblock import deblock_chroma_mb

        cr = np.full((8, 8), 128, dtype=np.uint8)
        cr[0:4, :] = 130
        cr[4:8, :] = 126

        block_info = {'is_intra': True, 'has_coeff': np.zeros(4, dtype=bool)}

        filtered = deblock_chroma_mb(cr, block_info, qp=26)

        # Should smooth the horizontal edge
        assert filtered[3, 0] != 130 or filtered[4, 0] != 126


class TestChromaFrameDeblocking:
    """Tests for frame-level chroma deblocking."""

    def test_deblock_frame_filters_chroma(self):
        """deblock_frame should filter both Cb and Cr planes."""
        from deblock.deblock import deblock_frame

        # Create test frame (2x2 MBs = 32x32 luma, 16x16 chroma)
        luma = np.full((32, 32), 128, dtype=np.uint8)
        cb = np.zeros((16, 16), dtype=np.uint8)
        cr = np.zeros((16, 16), dtype=np.uint8)

        # Add chroma blocking artifacts at internal edges (x=4, y=4 within each 8x8)
        # For 4:2:0, each 8x8 chroma block is one MB's chroma
        # Edge at x=4 within first 8x8 block
        cb[0:8, 0:4] = 120
        cb[0:8, 4:8] = 130
        # Edge at y=4 within first 8x8 block
        cb[0:4, 8:16] = 125
        cb[4:8, 8:16] = 130
        cb[8:16, :] = 128

        cr[0:4, :] = 125
        cr[4:8, :] = 130
        cr[8:16, :] = 128

        luma_out, cb_out, cr_out = deblock_frame(
            luma=luma, cb=cb, cr=cr, mb_info=None, qp=26
        )

        # Chroma planes should be modified at internal edges
        assert not np.array_equal(cb_out, cb), \
            "Cb plane should be filtered"
        assert not np.array_equal(cr_out, cr), \
            "Cr plane should be filtered"
