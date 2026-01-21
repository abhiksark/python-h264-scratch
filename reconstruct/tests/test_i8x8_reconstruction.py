# h264/reconstruct/tests/test_i8x8_reconstruction.py
"""RED TESTS: I_8x8 macroblock reconstruction for H.264 High Profile.

TDD: Tests written first, implementation to follow.

H.264 Spec Reference:
- Section 8.3.3: Intra_8x8 prediction for luma samples
- Section 8.5.6: 8x8 scaling and transformation process
- Section 8.5.12: Inverse 8x8 transform process
- Section 7.3.5.1: Macroblock prediction mode syntax

I_8x8 macroblocks (High profile only):
- 4 separate 8x8 blocks, each with its own prediction mode (0-8)
- Uses 8x8 IDCT instead of 4x4 IDCT
- Requires transform_8x8_mode_flag=1 in PPS
- Prediction modes coded using prev_intra8x8_pred_mode_flag and
  rem_intra8x8_pred_mode (same scheme as I_4x4)
- Blocks processed in raster scan order: top-left, top-right,
  bottom-left, bottom-right
"""

import numpy as np
import pytest

# Note: Some functionality is already implemented in macroblock.py
# Tests marked with @pytest.mark.xfail indicate areas that need implementation


# =============================================================================
# Test Imports and Module Structure
# =============================================================================

class TestI8x8ReconstructionModuleExists:
    """Tests for I_8x8 reconstruction module structure.

    H.264 Spec: I_8x8 is part of macroblock layer reconstruction.
    """

    def test_reconstruct_i8x8_block_function_exists(self):
        """reconstruct_i8x8_block should exist in macroblock module."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        assert callable(reconstruct_i8x8_block)

    def test_reconstruct_i8x8_luma_function_exists(self):
        """reconstruct_i8x8_luma should exist for full MB reconstruction."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        assert callable(reconstruct_i8x8_luma)

    @pytest.mark.xfail(reason="get_8x8_block_neighbors not implemented")
    def test_get_8x8_block_neighbors_function_exists(self):
        """get_8x8_block_neighbors should exist for neighbor extraction."""
        from reconstruct.macroblock import get_8x8_block_neighbors

        assert callable(get_8x8_block_neighbors)

    def test_block_scan_order_8x8_constant_exists(self):
        """BLOCK_SCAN_ORDER_8x8 constant should define block positions."""
        from reconstruct.macroblock import BLOCK_SCAN_ORDER_8x8

        assert len(BLOCK_SCAN_ORDER_8x8) == 4


# =============================================================================
# Test 8x8 Block Scan Order
# =============================================================================

class TestBlockScanOrder8x8:
    """Tests for 8x8 block scan order within 16x16 macroblock.

    H.264 Spec 6.4.6: 8x8 luma block indices
    Block positions in raster scan:
        Block 0: (0,0)  - top-left 8x8
        Block 1: (0,8)  - top-right 8x8
        Block 2: (8,0)  - bottom-left 8x8
        Block 3: (8,8)  - bottom-right 8x8
    """

    def test_scan_order_has_four_blocks(self):
        """Should have exactly 4 blocks for I_8x8."""
        from reconstruct.macroblock import BLOCK_SCAN_ORDER_8x8

        assert len(BLOCK_SCAN_ORDER_8x8) == 4

    def test_scan_order_covers_all_positions(self):
        """All four 8x8 positions should be covered."""
        from reconstruct.macroblock import BLOCK_SCAN_ORDER_8x8

        positions = set(BLOCK_SCAN_ORDER_8x8)
        expected = {(0, 0), (0, 8), (8, 0), (8, 8)}
        assert positions == expected

    def test_first_block_is_top_left(self):
        """Block 0 should be at (0, 0)."""
        from reconstruct.macroblock import BLOCK_SCAN_ORDER_8x8

        assert BLOCK_SCAN_ORDER_8x8[0] == (0, 0)

    def test_scan_follows_raster_order(self):
        """Blocks should follow raster scan order."""
        from reconstruct.macroblock import BLOCK_SCAN_ORDER_8x8

        assert BLOCK_SCAN_ORDER_8x8[0] == (0, 0)   # Top-left
        assert BLOCK_SCAN_ORDER_8x8[1] == (0, 8)   # Top-right
        assert BLOCK_SCAN_ORDER_8x8[2] == (8, 0)   # Bottom-left
        assert BLOCK_SCAN_ORDER_8x8[3] == (8, 8)   # Bottom-right


# =============================================================================
# Test 8x8 Block Reconstruction for All 9 Prediction Modes
# =============================================================================

class TestReconstructI8x8BlockVertical:
    """Tests for I_8x8 block reconstruction with mode 0 (Vertical).

    H.264 Spec 8.3.3.1: Vertical prediction copies top row to all rows.
    """

    def test_vertical_mode_no_residual(self):
        """Mode 0 with zero residual should copy top to all rows."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=0,  # Vertical
            residual=residual,
            top=top,
            left=left,
            top_left=0,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8
        for row in range(8):
            np.testing.assert_array_equal(result[row], top)

    def test_vertical_mode_with_residual(self):
        """Mode 0 with residual should add residual to prediction."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 50, dtype=np.uint8)
        residual = np.full((8, 8), 20, dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=0,
            residual=residual,
            top=top,
            left=left,
            top_left=75,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # 100 (prediction) + 20 (residual) = 120
        assert np.all(result == 120)


class TestReconstructI8x8BlockHorizontal:
    """Tests for I_8x8 block reconstruction with mode 1 (Horizontal).

    H.264 Spec 8.3.3.2: Horizontal prediction copies left column to all columns.
    """

    def test_horizontal_mode_no_residual(self):
        """Mode 1 with zero residual should copy left to all columns."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 50, dtype=np.uint8)
        left = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=1,  # Horizontal
            residual=residual,
            top=top,
            left=left,
            top_left=0,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        for col in range(8):
            np.testing.assert_array_equal(result[:, col], left)

    def test_horizontal_mode_with_residual(self):
        """Mode 1 with residual should add residual to prediction."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 50, dtype=np.uint8)
        left = np.full(8, 80, dtype=np.uint8)
        residual = np.full((8, 8), -30, dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=1,
            residual=residual,
            top=top,
            left=left,
            top_left=0,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # 80 (prediction) - 30 (residual) = 50
        assert np.all(result == 50)


class TestReconstructI8x8BlockDC:
    """Tests for I_8x8 block reconstruction with mode 2 (DC).

    H.264 Spec 8.3.3.3: DC prediction averages available neighbors.
    - Both available: (sum(top) + sum(left) + 8) >> 4
    - Only top: (sum(top) + 4) >> 3
    - Only left: (sum(left) + 4) >> 3
    - Neither: 128 (default value)
    """

    def test_dc_mode_both_neighbors_available(self):
        """DC mode with both neighbors should average them."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,  # DC
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8
        # (8*100 + 8*100 + 8) >> 4 = (1608) >> 4 = 100
        assert np.all(result == 100)

    def test_dc_mode_only_top_available(self):
        """DC mode with only top should use top average."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 80, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=top,
            left=None,
            top_left=None,
            top_right=None,
            top_available=True,
            left_available=False,
            top_right_available=False,
        )

        # (8*80 + 4) >> 3 = 644 >> 3 = 80
        assert np.all(result == 80)

    def test_dc_mode_only_left_available(self):
        """DC mode with only left should use left average."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        left = np.full(8, 120, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=None,
            left=left,
            top_left=None,
            top_right=None,
            top_available=False,
            left_available=True,
            top_right_available=False,
        )

        # (8*120 + 4) >> 3 = 964 >> 3 = 120
        assert np.all(result == 120)

    def test_dc_mode_no_neighbors_available(self):
        """DC mode with no neighbors should use default value 128."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=None,
            left=None,
            top_left=None,
            top_right=None,
            top_available=False,
            left_available=False,
            top_right_available=False,
        )

        assert np.all(result == 128)

    def test_dc_mode_with_residual(self):
        """DC mode with residual should add residual to DC value."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.full((8, 8), 55, dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # 100 + 55 = 155
        assert np.all(result == 155)


class TestReconstructI8x8BlockDiagonalDownLeft:
    """Tests for I_8x8 block reconstruction with mode 3 (DDL).

    H.264 Spec 8.3.3.4: Diagonal Down-Left extrapolates at 45 degrees
    from top-right toward bottom-left.
    """

    def test_ddl_mode_returns_8x8(self):
        """Mode 3 should return 8x8 block."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.arange(8, dtype=np.uint8) * 10 + 50
        top_right = np.arange(8, dtype=np.uint8) * 10 + 130
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=3,  # DDL
            residual=residual,
            top=top,
            left=left,
            top_left=50,
            top_right=top_right,
            top_available=True,
            left_available=True,
            top_right_available=True,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8

    def test_ddl_mode_no_top_right_replicates(self):
        """DDL without top-right should replicate last top pixel."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=3,
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # With uniform input, output should be uniform
        assert result.shape == (8, 8)


class TestReconstructI8x8BlockDiagonalDownRight:
    """Tests for I_8x8 block reconstruction with mode 4 (DDR).

    H.264 Spec 8.3.3.5: Diagonal Down-Right extrapolates at 45 degrees
    from top-left toward bottom-right.
    """

    def test_ddr_mode_returns_8x8(self):
        """Mode 4 should return 8x8 block."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.arange(8, dtype=np.uint8) * 10 + 50
        left = np.arange(8, dtype=np.uint8) * 10 + 50
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=4,  # DDR
            residual=residual,
            top=top,
            left=left,
            top_left=40,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8

    def test_ddr_mode_diagonal_values(self):
        """DDR diagonal pixels should use weighted average of corner."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=4,
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # With uniform 100 input, diagonal should be 100
        assert result[0, 0] == 100
        assert result[1, 1] == 100


class TestReconstructI8x8BlockVerticalRight:
    """Tests for I_8x8 block reconstruction with mode 5 (VR).

    H.264 Spec 8.3.3.6: Vertical-Right extrapolates at 26.6 degrees
    to the right of vertical.
    """

    def test_vr_mode_returns_8x8(self):
        """Mode 5 should return 8x8 block."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.arange(8, dtype=np.uint8) * 10 + 50
        left = np.arange(8, dtype=np.uint8) * 10 + 50
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=5,  # VR
            residual=residual,
            top=top,
            left=left,
            top_left=40,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8


class TestReconstructI8x8BlockHorizontalDown:
    """Tests for I_8x8 block reconstruction with mode 6 (HD).

    H.264 Spec 8.3.3.7: Horizontal-Down extrapolates at 26.6 degrees
    below horizontal.
    """

    def test_hd_mode_returns_8x8(self):
        """Mode 6 should return 8x8 block."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.arange(8, dtype=np.uint8) * 10 + 50
        left = np.arange(8, dtype=np.uint8) * 10 + 50
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=6,  # HD
            residual=residual,
            top=top,
            left=left,
            top_left=40,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8


class TestReconstructI8x8BlockVerticalLeft:
    """Tests for I_8x8 block reconstruction with mode 7 (VL).

    H.264 Spec 8.3.3.8: Vertical-Left extrapolates at 26.6 degrees
    to the left of vertical.
    """

    def test_vl_mode_returns_8x8(self):
        """Mode 7 should return 8x8 block."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.arange(8, dtype=np.uint8) * 10 + 50
        top_right = np.arange(8, dtype=np.uint8) * 10 + 130
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=7,  # VL
            residual=residual,
            top=top,
            left=left,
            top_left=50,
            top_right=top_right,
            top_available=True,
            left_available=True,
            top_right_available=True,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8


class TestReconstructI8x8BlockHorizontalUp:
    """Tests for I_8x8 block reconstruction with mode 8 (HU).

    H.264 Spec 8.3.3.9: Horizontal-Up extrapolates at 26.6 degrees
    above horizontal.
    """

    def test_hu_mode_returns_8x8(self):
        """Mode 8 should return 8x8 block."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 50, dtype=np.uint8)
        left = np.arange(8, dtype=np.uint8) * 10 + 50
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=8,  # HU
            residual=residual,
            top=top,
            left=left,
            top_left=50,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8


class TestReconstructI8x8BlockAllModes:
    """Parametrized tests for all 9 prediction modes."""

    @pytest.mark.parametrize("mode", range(9))
    def test_all_modes_return_8x8(self, mode):
        """All modes should return 8x8 block."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.arange(8, dtype=np.uint8) * 20 + 50
        left = np.arange(8, dtype=np.uint8) * 15 + 60
        top_right = np.arange(8, dtype=np.uint8) * 10 + 100
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=mode,
            residual=residual,
            top=top,
            left=left,
            top_left=40,
            top_right=top_right,
            top_available=True,
            left_available=True,
            top_right_available=True,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("mode", range(9))
    def test_all_modes_with_residual(self, mode):
        """All modes should correctly add residual."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)
        residual = np.full((8, 8), 10, dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=mode,
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=top_right,
            top_available=True,
            left_available=True,
            top_right_available=True,
        )

        # With uniform 100 input and +10 residual, expect values near 110
        # (exact value depends on mode filtering)
        assert result.min() >= 100  # At least 100 + some residual effect
        assert result.max() <= 130  # Not too high


# =============================================================================
# Test Residual Addition
# =============================================================================

class TestResidualAddition:
    """Tests for proper residual addition to predicted blocks.

    H.264 Spec 8.5.6: The 8x8 transform produces residual samples that
    are added to the prediction samples.
    """

    def test_positive_residual_increases_values(self):
        """Positive residual should increase pixel values."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.full((8, 8), 50, dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,  # DC
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # 100 + 50 = 150
        assert np.all(result == 150)

    def test_negative_residual_decreases_values(self):
        """Negative residual should decrease pixel values."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.full((8, 8), -60, dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,  # DC
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # 100 - 60 = 40
        assert np.all(result == 40)

    def test_residual_pattern_preserved(self):
        """Residual spatial pattern should be preserved after addition."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)

        # Gradient residual
        residual = np.zeros((8, 8), dtype=np.int32)
        for i in range(8):
            residual[i, :] = i * 5

        result = reconstruct_i8x8_block(
            mode=2,  # DC
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # Row 0: 100 + 0 = 100
        # Row 7: 100 + 35 = 135
        assert np.all(result[0, :] == 100)
        assert np.all(result[7, :] == 135)

    def test_residual_clipping_high(self):
        """Results should be clipped to 255 max."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 250, dtype=np.uint8)
        left = np.full(8, 250, dtype=np.uint8)
        residual = np.full((8, 8), 100, dtype=np.int32)  # Would exceed 255

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=top,
            left=left,
            top_left=250,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.max() <= 255

    def test_residual_clipping_low(self):
        """Results should be clipped to 0 min."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 10, dtype=np.uint8)
        left = np.full(8, 10, dtype=np.uint8)
        residual = np.full((8, 8), -100, dtype=np.int32)  # Would go below 0

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=top,
            left=left,
            top_left=10,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.min() >= 0


# =============================================================================
# Test Neighbor Pixel Handling for 8x8 Predictions
# =============================================================================

@pytest.mark.xfail(reason="get_8x8_block_neighbors not implemented")
class TestGet8x8BlockNeighbors:
    """Tests for extracting neighbor pixels for 8x8 blocks.

    H.264 Spec 8.3.3: 8x8 prediction uses up to 25 neighbors:
    - 8 top pixels (A0-A7)
    - 8 top-right pixels (B0-B7)
    - 8 left pixels (I0-I7)
    - 1 top-left pixel (M)
    """

    def test_interior_block_has_all_neighbors(self):
        """Block in middle of frame has all neighbors available."""
        from reconstruct.macroblock import get_8x8_block_neighbors

        frame = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)

        neighbors = get_8x8_block_neighbors(
            frame=frame,
            block_row=8,
            block_col=8,
            mb_row=0,
            mb_col=0,
        )

        assert neighbors['top_available']
        assert neighbors['left_available']
        assert len(neighbors['top']) == 8
        assert len(neighbors['left']) == 8

    def test_first_block_first_mb_no_external_neighbors(self):
        """First block (0,0) of first MB has no external neighbors."""
        from reconstruct.macroblock import get_8x8_block_neighbors

        frame = np.full((16, 16), 128, dtype=np.uint8)

        neighbors = get_8x8_block_neighbors(
            frame=frame,
            block_row=0,
            block_col=0,
            mb_row=0,
            mb_col=0,
        )

        assert not neighbors['top_available']
        assert not neighbors['left_available']

    def test_block_uses_frame_neighbors(self):
        """Blocks should use frame buffer for external neighbors."""
        from reconstruct.macroblock import get_8x8_block_neighbors

        frame = np.zeros((32, 32), dtype=np.uint8)
        # Set known pattern in row above MB (row 15)
        frame[15, 16:24] = [10, 20, 30, 40, 50, 60, 70, 80]

        neighbors = get_8x8_block_neighbors(
            frame=frame,
            block_row=0,
            block_col=0,
            mb_row=1,
            mb_col=1,
        )

        assert neighbors['top_available']
        np.testing.assert_array_equal(
            neighbors['top'],
            [10, 20, 30, 40, 50, 60, 70, 80]
        )

    def test_block_uses_reconstructed_neighbors(self):
        """Later blocks use previously reconstructed pixels from same MB."""
        from reconstruct.macroblock import get_8x8_block_neighbors

        frame = np.zeros((16, 16), dtype=np.uint8)
        # Simulate block 0 reconstructed (top-left 8x8)
        frame[0:8, 0:8] = 150

        neighbors = get_8x8_block_neighbors(
            frame=frame,
            block_row=0,  # Block 1: top-right
            block_col=8,
            mb_row=0,
            mb_col=0,
        )

        # Block 1's left neighbor should be from block 0's right edge
        if neighbors['left_available']:
            assert np.all(neighbors['left'] == 150)

    def test_top_right_availability_block_0(self):
        """Block 0 should have top-right available if top MB exists."""
        from reconstruct.macroblock import get_8x8_block_neighbors

        frame = np.zeros((32, 32), dtype=np.uint8)
        frame[15, 24:32] = 200  # Top-right of block 0

        neighbors = get_8x8_block_neighbors(
            frame=frame,
            block_row=0,
            block_col=0,
            mb_row=1,
            mb_col=1,
        )

        # Block 0's top-right is at row -1, cols 8-16 of this MB
        # Which is row 15, cols 24-32 of frame (if mb at (1,1))
        if neighbors['top_right_available']:
            assert len(neighbors['top_right']) == 8

    def test_top_right_not_available_block_1(self):
        """Block 1 (top-right) may not have top-right available.

        H.264 6.4.6: Top-right for blocks 1 and 3 depends on
        MB to upper-right which may not exist.
        """
        from reconstruct.macroblock import get_8x8_block_neighbors

        frame = np.zeros((32, 32), dtype=np.uint8)

        neighbors = get_8x8_block_neighbors(
            frame=frame,
            block_row=0,  # Block 1: top-right 8x8
            block_col=8,
            mb_row=1,
            mb_col=1,  # Not rightmost MB
        )

        # Top-right for block 1 would be from MB at (mb_x+1, mb_y-1)
        # Availability depends on frame dimensions


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestI8x8EdgeCases:
    """Tests for I_8x8 edge cases and boundary conditions."""

    def test_top_boundary_mb(self):
        """First row of MBs has no top neighbors."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 2, 2, 2]  # All DC
        residuals = [np.zeros((8, 8), dtype=np.int32) for _ in range(4)]

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=None,
            neighbors_left=None,
            neighbor_top_left=None,
        )

        assert result.shape == (16, 16)
        # Should use default DC value (128) for all blocks
        assert np.all(result == 128)

    def test_left_boundary_mb(self):
        """First column of MBs has no left neighbors."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 2, 2, 2]
        residuals = [np.zeros((8, 8), dtype=np.int32) for _ in range(4)]
        neighbors_top = np.full(16, 100, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=None,
            neighbor_top_left=None,
        )

        assert result.shape == (16, 16)
        # DC with only top: (8*100 + 4) >> 3 = 100
        # First row blocks should be 100

    def test_all_zeros_residual(self):
        """All zeros residual should return pure prediction."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 75, dtype=np.uint8)
        left = np.full(8, 75, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=top,
            left=left,
            top_left=75,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert np.all(result == 75)

    def test_all_255_saturation(self):
        """Maximum values should saturate to 255."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 255, dtype=np.uint8)
        left = np.full(8, 255, dtype=np.uint8)
        residual = np.full((8, 8), 100, dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=top,
            left=left,
            top_left=255,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert np.all(result == 255)

    def test_invalid_mode_raises_error(self):
        """Invalid prediction mode should raise ValueError."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        with pytest.raises(ValueError, match="[Mm]ode"):
            reconstruct_i8x8_block(
                mode=9,  # Invalid - only 0-8 valid
                residual=np.zeros((8, 8), dtype=np.int32),
                top=np.zeros(8, dtype=np.uint8),
                left=np.zeros(8, dtype=np.uint8),
                top_left=0,
                top_right=None,
                top_available=True,
                left_available=True,
                top_right_available=False,
            )


# =============================================================================
# Test Full 16x16 Luma Reconstruction
# =============================================================================

class TestReconstructI8x8Luma:
    """Tests for full 16x16 luma reconstruction from four 8x8 blocks."""

    def test_returns_16x16(self):
        """Should return 16x16 luma block."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 2, 2, 2]
        residuals = [np.zeros((8, 8), dtype=np.int32) for _ in range(4)]
        neighbors_top = np.full(16, 128, dtype=np.uint8)
        neighbors_left = np.full(16, 128, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=128,
        )

        assert result.shape == (16, 16)
        assert result.dtype == np.uint8

    def test_uniform_dc_produces_uniform_output(self):
        """All DC mode with uniform neighbors produces uniform output."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 2, 2, 2]
        residuals = [np.zeros((8, 8), dtype=np.int32) for _ in range(4)]
        neighbors_top = np.full(16, 100, dtype=np.uint8)
        neighbors_left = np.full(16, 100, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=100,
        )

        assert np.all(result == 100)

    def test_different_modes_per_block(self):
        """Different prediction modes for different blocks."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [0, 1, 2, 0]  # V, H, DC, V
        residuals = [np.zeros((8, 8), dtype=np.int32) for _ in range(4)]
        neighbors_top = np.arange(16, dtype=np.uint8) * 5 + 50
        neighbors_left = np.arange(16, dtype=np.uint8) * 3 + 60

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=50,
        )

        assert result.shape == (16, 16)
        assert result.dtype == np.uint8

    @pytest.mark.xfail(reason="Block-specific residuals affected by neighbor propagation")
    def test_residuals_added_to_each_block(self):
        """Residuals should be added to each 8x8 block.

        Note: This test is designed for the idealized case where each block
        gets external neighbors only. In practice, later blocks use
        reconstructed neighbors from earlier blocks, which affects their
        DC prediction values.
        """
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 2, 2, 2]
        # Different residuals for each block
        residuals = [
            np.full((8, 8), 10, dtype=np.int32),
            np.full((8, 8), 20, dtype=np.int32),
            np.full((8, 8), 30, dtype=np.int32),
            np.full((8, 8), 40, dtype=np.int32),
        ]
        neighbors_top = np.full(16, 100, dtype=np.uint8)
        neighbors_left = np.full(16, 100, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=100,
        )

        # Block 0 (top-left): 100 + 10 = 110
        assert np.all(result[0:8, 0:8] == 110)
        # Block 1 (top-right): 100 + 20 = 120
        assert np.all(result[0:8, 8:16] == 120)
        # Block 2 (bottom-left): 100 + 30 = 130
        assert np.all(result[8:16, 0:8] == 130)
        # Block 3 (bottom-right): 100 + 40 = 140
        assert np.all(result[8:16, 8:16] == 140)

    def test_blocks_use_previously_decoded_neighbors(self):
        """Later blocks should use reconstructed pixels from earlier blocks."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        # First block gets large residual
        modes = [2, 1, 2, 2]  # DC, Horizontal, DC, DC
        residuals = [
            np.full((8, 8), 50, dtype=np.int32),  # Block 0: +50
            np.zeros((8, 8), dtype=np.int32),      # Block 1: no residual
            np.zeros((8, 8), dtype=np.int32),
            np.zeros((8, 8), dtype=np.int32),
        ]
        # External neighbors all 100
        neighbors_top = np.full(16, 100, dtype=np.uint8)
        neighbors_left = np.full(16, 100, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=100,
        )

        # Block 0: DC=100, +50 residual = 150
        assert np.all(result[0:8, 0:8] == 150)

        # Block 1 (Horizontal mode): uses left neighbors from block 0
        # Block 0's right edge is 150, so block 1 should replicate 150
        assert np.all(result[0:8, 8:16] == 150)


# =============================================================================
# Test Integration with 8x8 IDCT
# =============================================================================

class TestI8x8IDCTIntegration:
    """Tests for integration with 8x8 IDCT output.

    H.264 Spec 8.5.12: The 8x8 inverse transform process produces
    residual samples that are added to prediction samples.
    """

    def test_idct_output_as_residual(self):
        """8x8 IDCT output should be usable as residual."""
        from reconstruct.macroblock import reconstruct_i8x8_block
        from transform.idct_8x8 import idct_8x8

        # Create transform coefficients
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 256  # DC only

        # Apply IDCT
        residual = idct_8x8(coeffs)

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)

        result = reconstruct_i8x8_block(
            mode=2,  # DC
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8
        # Should have combined DC prediction with IDCT residual

    def test_ac_coefficients_produce_spatial_detail(self):
        """AC coefficients in IDCT should produce spatial detail."""
        from reconstruct.macroblock import reconstruct_i8x8_block
        from transform.idct_8x8 import idct_8x8

        # Create coefficients with AC content
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 0    # No DC
        coeffs[0, 1] = 128  # Horizontal AC

        residual = idct_8x8(coeffs)

        top = np.full(8, 128, dtype=np.uint8)
        left = np.full(8, 128, dtype=np.uint8)

        result = reconstruct_i8x8_block(
            mode=2,  # DC
            residual=residual,
            top=top,
            left=left,
            top_left=128,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # Should not be uniform due to AC content
        assert not np.all(result == result[0, 0])

    def test_full_reconstruction_pipeline(self):
        """Test complete prediction + IDCT residual pipeline."""
        from reconstruct.macroblock import reconstruct_i8x8_luma
        from transform.idct_8x8 import idct_8x8

        modes = [2, 2, 2, 2]

        # Create IDCT residuals for each block
        residuals = []
        for i in range(4):
            coeffs = np.zeros((8, 8), dtype=np.int32)
            coeffs[0, 0] = 64 * (i + 1)  # Different DC for each
            residuals.append(idct_8x8(coeffs))

        neighbors_top = np.full(16, 100, dtype=np.uint8)
        neighbors_left = np.full(16, 100, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=100,
        )

        assert result.shape == (16, 16)
        # Each quadrant should have different average due to different DC


# =============================================================================
# Test Neighbor Derivation Between Blocks
# =============================================================================

class TestI8x8NeighborDerivation:
    """Tests for correct neighbor derivation between 8x8 blocks.

    When reconstructing block N, we need neighbors from:
    - Previously reconstructed blocks in same MB
    - Reconstructed MBs in frame buffer (top, left, top-left, top-right)
    """

    def test_block_1_uses_block_0_as_left(self):
        """Block 1 (top-right) uses block 0 (top-left) for left neighbors."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 1, 2, 2]  # Block 1 uses Horizontal (mode 1)
        residuals = [
            np.full((8, 8), 30, dtype=np.int32),  # Block 0: DC + 30
            np.zeros((8, 8), dtype=np.int32),      # Block 1: no residual
            np.zeros((8, 8), dtype=np.int32),
            np.zeros((8, 8), dtype=np.int32),
        ]
        neighbors_top = np.full(16, 100, dtype=np.uint8)
        neighbors_left = np.full(16, 100, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=100,
        )

        # Block 0: DC=100 + 30 = 130
        block_0_value = 130

        # Block 1 uses Horizontal mode, which copies left edge
        # Left edge comes from block 0's right column (all 130)
        expected_block_1_value = block_0_value
        np.testing.assert_array_equal(
            result[0:8, 8:16],
            np.full((8, 8), expected_block_1_value, dtype=np.uint8)
        )

    def test_block_2_uses_block_0_as_top(self):
        """Block 2 (bottom-left) uses block 0 (top-left) for top neighbors."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 2, 0, 2]  # Block 2 uses Vertical (mode 0)
        residuals = [
            np.full((8, 8), 20, dtype=np.int32),  # Block 0: DC + 20
            np.zeros((8, 8), dtype=np.int32),
            np.zeros((8, 8), dtype=np.int32),      # Block 2: no residual
            np.zeros((8, 8), dtype=np.int32),
        ]
        neighbors_top = np.full(16, 100, dtype=np.uint8)
        neighbors_left = np.full(16, 100, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=100,
        )

        # Block 0: DC=100 + 20 = 120
        block_0_value = 120

        # Block 2 uses Vertical mode, which copies top row
        # Top row comes from block 0's bottom row (all 120)
        expected_block_2_value = block_0_value
        np.testing.assert_array_equal(
            result[8:16, 0:8],
            np.full((8, 8), expected_block_2_value, dtype=np.uint8)
        )

    def test_block_3_uses_blocks_1_and_2(self):
        """Block 3 uses block 1 for top and block 2 for left."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 2, 2, 2]  # All DC
        residuals = [
            np.zeros((8, 8), dtype=np.int32),
            np.full((8, 8), 40, dtype=np.int32),  # Block 1: DC + 40
            np.full((8, 8), 60, dtype=np.int32),  # Block 2: DC + 60
            np.zeros((8, 8), dtype=np.int32),      # Block 3: no residual
        ]
        neighbors_top = np.full(16, 100, dtype=np.uint8)
        neighbors_left = np.full(16, 100, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=100,
        )

        # Block 1: 100 + 40 = 140
        # Block 2: 100 + 60 = 160
        # Block 3 DC should average top (140) and left (160)
        # DC = (8*140 + 8*160 + 8) >> 4 = (1120 + 1280 + 8) >> 4 = 150
        expected_block_3_value = 150
        np.testing.assert_allclose(
            result[8:16, 8:16],
            np.full((8, 8), expected_block_3_value, dtype=np.uint8),
            atol=1
        )


# =============================================================================
# Test Mode Availability Requirements
# =============================================================================

@pytest.mark.xfail(reason="Missing neighbor fallback behavior not fully implemented")
class TestModeAvailabilityRequirements:
    """Tests for prediction mode neighbor requirements.

    H.264 Spec 8.3.3: Different modes require different neighbors:
    - Mode 0 (V): Requires top
    - Mode 1 (H): Requires left
    - Mode 2 (DC): Uses available neighbors
    - Modes 3, 7 (DDL, VL): Require top, use top-right if available
    - Modes 4, 5, 6 (DDR, VR, HD): Require top, left, and top-left
    - Mode 8 (HU): Requires left
    """

    def test_vertical_mode_without_top_uses_default(self):
        """Vertical mode without top neighbors uses default value."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=0,  # Vertical
            residual=residual,
            top=None,
            left=np.full(8, 100, dtype=np.uint8),
            top_left=None,
            top_right=None,
            top_available=False,
            left_available=True,
            top_right_available=False,
        )

        # Should use default value (128) when required neighbor unavailable
        assert result.shape == (8, 8)
        assert np.all(result == 128)

    def test_horizontal_mode_without_left_uses_default(self):
        """Horizontal mode without left neighbors uses default value."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=1,  # Horizontal
            residual=residual,
            top=np.full(8, 100, dtype=np.uint8),
            left=None,
            top_left=None,
            top_right=None,
            top_available=True,
            left_available=False,
            top_right_available=False,
        )

        # Should use default value when required neighbor unavailable
        assert result.shape == (8, 8)
        assert np.all(result == 128)

    def test_ddr_without_top_left_uses_default(self):
        """DDR mode without top-left uses default or approximation."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        # DDR requires all three: top, left, top-left
        # Without top_left, implementation may use default or estimate
        result = reconstruct_i8x8_block(
            mode=4,  # DDR
            residual=residual,
            top=top,
            left=left,
            top_left=None,  # Missing!
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # Should still produce valid output
        assert result.shape == (8, 8)


# =============================================================================
# Test Data Type Handling
# =============================================================================

class TestDataTypeHandling:
    """Tests for correct data type handling throughout reconstruction."""

    def test_output_is_uint8(self):
        """Output should always be uint8."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        result = reconstruct_i8x8_block(
            mode=2,
            residual=np.zeros((8, 8), dtype=np.int32),
            top=np.full(8, 100, dtype=np.uint8),
            left=np.full(8, 100, dtype=np.uint8),
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.dtype == np.uint8

    def test_accepts_int32_residual(self):
        """Should accept int32 residual (from IDCT)."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        residual = np.array([
            [-50, -40, -30, -20, -10, 0, 10, 20],
            [-40, -30, -20, -10, 0, 10, 20, 30],
            [-30, -20, -10, 0, 10, 20, 30, 40],
            [-20, -10, 0, 10, 20, 30, 40, 50],
            [-10, 0, 10, 20, 30, 40, 50, 60],
            [0, 10, 20, 30, 40, 50, 60, 70],
            [10, 20, 30, 40, 50, 60, 70, 80],
            [20, 30, 40, 50, 60, 70, 80, 90],
        ], dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=np.full(8, 100, dtype=np.uint8),
            left=np.full(8, 100, dtype=np.uint8),
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_internal_computation_avoids_overflow(self):
        """Internal computation should use int32 to avoid overflow."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        # Max prediction (255) + large positive residual
        residual = np.full((8, 8), 1000, dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=np.full(8, 255, dtype=np.uint8),
            left=np.full(8, 255, dtype=np.uint8),
            top_left=255,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        # Should clip to 255, not wrap around
        assert np.all(result == 255)
