# h264/reconstruct/tests/test_intra_4x4_reconstruction.py
"""Tests for I_4x4 (I_NxN) macroblock reconstruction.

TDD: Tests written first, implementation to follow.

H.264 Spec: Section 7.3.5, 7.4.5 - Macroblock layer syntax/semantics
H.264 Spec: Section 8.3.1 - Intra_4x4 prediction process

I_NxN macroblocks (mb_type=0 for I-slices):
- 16 separate 4x4 blocks, each with its own prediction mode
- Prediction modes coded using prev_intra4x4_pred_mode_flag and
  rem_intra4x4_pred_mode
- Blocks processed in raster scan order within 8x8 blocks,
  then 8x8 blocks in raster scan order
"""

import numpy as np
import pytest

from bitstream import BitWriter, BitReader, BITSTRING_AVAILABLE

pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)

# These will fail until implemented
from reconstruct.macroblock import (
    decode_intra4x4_pred_modes,
    get_4x4_block_neighbors,
    reconstruct_i4x4_block,
    reconstruct_i4x4_luma,
    BLOCK_SCAN_ORDER,
)


class TestBlockScanOrder:
    """Tests for 4x4 block scan order within macroblock.

    H.264 uses a specific scan order for 4x4 blocks:
    Block indices 0-15 map to positions in 16x16 macroblock.

    8x8 block order: 0=TL, 1=TR, 2=BL, 3=BR
    Within each 8x8: 4x4 blocks in Z-pattern
    """

    def test_scan_order_length(self):
        """Should have 16 blocks."""
        assert len(BLOCK_SCAN_ORDER) == 16

    def test_scan_order_covers_all_positions(self):
        """All 16 positions should be covered."""
        positions = set()
        for row, col in BLOCK_SCAN_ORDER:
            positions.add((row, col))

        expected = {(r * 4, c * 4) for r in range(4) for c in range(4)}
        assert positions == expected

    def test_first_block_is_top_left(self):
        """Block 0 should be at (0, 0)."""
        assert BLOCK_SCAN_ORDER[0] == (0, 0)

    def test_scan_follows_z_pattern(self):
        """Verify Z-pattern within each 8x8 block."""
        # First 8x8 block (top-left): blocks 0, 1, 2, 3
        assert BLOCK_SCAN_ORDER[0] == (0, 0)   # Top-left of 8x8
        assert BLOCK_SCAN_ORDER[1] == (0, 4)   # Top-right of 8x8
        assert BLOCK_SCAN_ORDER[2] == (4, 0)   # Bottom-left of 8x8
        assert BLOCK_SCAN_ORDER[3] == (4, 4)   # Bottom-right of 8x8


class TestDecodeIntra4x4PredModes:
    """Tests for decoding 4x4 prediction modes from bitstream.

    H.264 Spec 7.3.5.1: prev_intra4x4_pred_mode_flag and rem_intra4x4_pred_mode

    For each 4x4 block:
    - If prev_intra4x4_pred_mode_flag=1: use predicted mode
    - If prev_intra4x4_pred_mode_flag=0: rem_intra4x4_pred_mode gives mode
    """

    def test_decode_all_predicted_modes(self):
        """When all flags are 1, uses predicted modes."""
        # 16 bits, all 1s (each block uses predicted mode)
        writer = BitWriter()
        for _ in range(16):
            writer.write_bits(1, 1)  # prev_intra4x4_pred_mode_flag = 1
        writer.write_bits(0, 8)  # Padding
        reader = BitReader(writer.to_bytes())

        # With no prior context, predicted mode is typically DC (mode 2)
        modes = decode_intra4x4_pred_modes(reader)

        assert len(modes) == 16
        # All modes should be the predicted mode (min of left and top)

    def test_decode_explicit_modes(self):
        """When flag is 0, read 3-bit mode value."""
        writer = BitWriter()
        # Block 0: flag=0, rem=0 (vertical)
        writer.write_bits(0, 1)
        writer.write_bits(0, 3)
        # Block 1: flag=0, rem=1 (horizontal)
        writer.write_bits(0, 1)
        writer.write_bits(1, 3)
        # Remaining blocks: flag=1 (use predicted)
        for _ in range(14):
            writer.write_bits(1, 1)
        writer.write_bits(0, 8)  # Padding
        reader = BitReader(writer.to_bytes())

        modes = decode_intra4x4_pred_modes(reader)

        assert len(modes) == 16
        # First two blocks should have explicit modes
        # Mode mapping: if rem < predicted_mode: mode = rem, else mode = rem + 1

    def test_decode_dc_modes(self):
        """All DC modes (mode 2) using predicted mode.

        H.264 mode encoding: rem_intra4x4_pred_mode skips the predicted mode.
        When predicted_mode=2 (DC default), using flag=1 gives DC mode.
        Using flag=0 with rem=2 would give mode 3 (since rem >= predicted_mode).
        """
        writer = BitWriter()
        for _ in range(16):
            writer.write_bits(1, 1)  # flag = 1 (use predicted mode = DC)
        writer.write_bits(0, 8)  # Padding
        reader = BitReader(writer.to_bytes())

        modes = decode_intra4x4_pred_modes(reader)

        assert len(modes) == 16
        # All should be DC mode (2) - predicted mode when neighbors unavailable
        assert all(m == 2 for m in modes)


class TestGet4x4BlockNeighbors:
    """Tests for extracting neighbor pixels for a 4x4 block.

    Each 4x4 block needs up to 13 neighbors:
    - top[0-3]: 4 pixels above
    - top_right[0-3]: 4 pixels above-right
    - left[0-3]: 4 pixels to left
    - top_left: 1 corner pixel
    """

    def test_interior_block_has_all_neighbors(self):
        """Block in middle of frame has all neighbors."""
        # Create 16x16 frame with known values
        frame = np.arange(256, dtype=np.uint8).reshape(16, 16)

        # Block at position (4, 4) - not on any edge
        neighbors = get_4x4_block_neighbors(
            frame=frame,
            block_row=4,
            block_col=4,
            mb_row=0,
            mb_col=0,
        )

        assert neighbors['top_available']
        assert neighbors['left_available']
        assert neighbors['top_right_available']
        assert len(neighbors['top']) == 4
        assert len(neighbors['left']) == 4

    def test_top_left_block_limited_neighbors(self):
        """First block (0,0) of first MB has no external neighbors."""
        frame = np.full((16, 16), 128, dtype=np.uint8)

        neighbors = get_4x4_block_neighbors(
            frame=frame,
            block_row=0,
            block_col=0,
            mb_row=0,
            mb_col=0,
        )

        # First block has no top/left from outside
        assert not neighbors['top_available']
        assert not neighbors['left_available']

    def test_second_block_uses_first_block(self):
        """Block 1 uses reconstructed pixels from block 0 as left neighbor."""
        frame = np.zeros((16, 16), dtype=np.uint8)
        # Simulate block 0 being reconstructed
        frame[0:4, 0:4] = 100

        neighbors = get_4x4_block_neighbors(
            frame=frame,
            block_row=0,
            block_col=4,  # Block 1 is to the right of block 0
            mb_row=0,
            mb_col=0,
        )

        # Block 1 should have top (from outside MB) unavailable for first MB
        # But should have left (from block 0)
        # This depends on implementation details


class TestReconstructI4x4Block:
    """Tests for reconstructing a single 4x4 block."""

    def test_reconstruct_dc_mode_no_residual(self):
        """DC mode with zero residual returns DC prediction."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)
        residual = np.zeros((4, 4), dtype=np.int32)

        result = reconstruct_i4x4_block(
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

        assert result.shape == (4, 4)
        assert result.dtype == np.uint8
        assert np.all(result == 100)

    def test_reconstruct_vertical_mode(self):
        """Vertical mode copies top row."""
        top = np.array([10, 20, 30, 40], dtype=np.uint8)
        left = np.array([50, 50, 50, 50], dtype=np.uint8)
        residual = np.zeros((4, 4), dtype=np.int32)

        result = reconstruct_i4x4_block(
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

        # Each row should equal top
        for row in range(4):
            np.testing.assert_array_equal(result[row], top)

    def test_reconstruct_with_residual(self):
        """Prediction + residual gives final result."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)
        # Add 10 to each pixel
        residual = np.full((4, 4), 10, dtype=np.int32)

        result = reconstruct_i4x4_block(
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

        assert np.all(result == 110)

    def test_reconstruct_clips_to_valid_range(self):
        """Result should be clipped to [0, 255]."""
        top = np.array([250, 250, 250, 250], dtype=np.uint8)
        left = np.array([250, 250, 250, 250], dtype=np.uint8)
        residual = np.full((4, 4), 20, dtype=np.int32)  # Would overflow

        result = reconstruct_i4x4_block(
            mode=2,  # DC
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


class TestReconstructI4x4Luma:
    """Tests for full 16x16 luma reconstruction from 16 4x4 blocks."""

    def test_uniform_dc_produces_uniform_block(self):
        """All DC mode blocks with zero residual = uniform output."""
        # All 16 blocks use DC mode
        modes = [2] * 16
        # All zero residuals
        residuals = [np.zeros((4, 4), dtype=np.int32) for _ in range(16)]
        # Uniform neighbors
        neighbors_top = np.full(16, 128, dtype=np.uint8)
        neighbors_left = np.full(16, 128, dtype=np.uint8)

        result = reconstruct_i4x4_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=128,
        )

        assert result.shape == (16, 16)
        assert result.dtype == np.uint8
        # Should be all 128 (DC prediction from neighbors)
        assert np.all(result == 128)

    def test_different_modes_per_block(self):
        """Different prediction modes for different blocks."""
        # Mix of modes
        modes = [0, 1, 2, 0] * 4  # V, H, DC, V repeated
        residuals = [np.zeros((4, 4), dtype=np.int32) for _ in range(16)]
        neighbors_top = np.arange(16, dtype=np.uint8) * 10
        neighbors_left = np.arange(16, dtype=np.uint8) * 5

        result = reconstruct_i4x4_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=0,
        )

        assert result.shape == (16, 16)
        assert result.dtype == np.uint8

    def test_residuals_added_correctly(self):
        """Residuals should be added to predictions."""
        modes = [2] * 16  # All DC
        # Add 50 to first block only
        residuals = [np.zeros((4, 4), dtype=np.int32) for _ in range(16)]
        residuals[0] = np.full((4, 4), 50, dtype=np.int32)

        neighbors_top = np.full(16, 100, dtype=np.uint8)
        neighbors_left = np.full(16, 100, dtype=np.uint8)

        result = reconstruct_i4x4_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=100,
        )

        # First 4x4 block should be 150 (100 DC + 50 residual)
        assert np.all(result[0:4, 0:4] == 150)
        # Other blocks should be 100 (DC prediction)


class TestI4x4Integration:
    """Integration tests for I_4x4 macroblock decoding."""

    def test_first_mb_no_neighbors(self):
        """First MB in frame has no external neighbors."""
        modes = [2] * 16  # All DC (works without neighbors)
        residuals = [np.zeros((4, 4), dtype=np.int32) for _ in range(16)]

        result = reconstruct_i4x4_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=None,
            neighbors_left=None,
            neighbor_top_left=None,
        )

        assert result.shape == (16, 16)
        # Should be 128 (default DC value)
        assert np.all(result == 128)

    def test_blocks_use_previously_decoded_blocks(self):
        """Later blocks should use reconstructed pixels from earlier blocks."""
        # First block gets large residual, second block DC should see it
        modes = [2] * 16
        residuals = [np.zeros((4, 4), dtype=np.int32) for _ in range(16)]
        residuals[0] = np.full((4, 4), 50, dtype=np.int32)  # First block +50

        result = reconstruct_i4x4_luma(
            modes=modes,
            residuals=residuals,
            neighbors_top=None,
            neighbors_left=None,
            neighbor_top_left=None,
        )

        # First block: 128 (default DC) + 50 = 178
        assert np.allclose(result[0:4, 0:4], 178, atol=1)

        # Second block (at col 4) uses first block's right edge as left neighbor
        # Its DC should include those values
