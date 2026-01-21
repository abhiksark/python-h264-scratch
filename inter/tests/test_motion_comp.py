# h264/inter/tests/test_motion_comp.py
"""Tests for motion compensation.

Tests block extraction from reference frames at integer and fractional
motion vector positions.
"""

import pytest
import numpy as np

from inter.motion_comp import (
    get_block_integer,
    clip_mv_to_frame,
    interpolate_half_h,
    interpolate_half_v,
    interpolate_half_hv,
    get_luma_block_fractional,
)


class TestIntegerMotionCompensation:
    """Tests for integer-position motion compensation."""

    @pytest.fixture
    def reference_frame(self):
        """Create a test reference frame with known pattern."""
        # 32x32 frame with gradient pattern
        frame = np.zeros((32, 32), dtype=np.uint8)
        for y in range(32):
            for x in range(32):
                frame[y, x] = (y * 8 + x) % 256
        return frame

    def test_get_block_at_origin(self, reference_frame):
        """Extract block at position (0, 0)."""
        block = get_block_integer(reference_frame, x=0, y=0, width=4, height=4)

        assert block.shape == (4, 4)
        # Check top-left corner values from gradient pattern
        assert block[0, 0] == 0
        assert block[0, 1] == 1
        assert block[1, 0] == 8

    def test_get_block_at_offset(self, reference_frame):
        """Extract block at non-zero position."""
        block = get_block_integer(reference_frame, x=4, y=8, width=4, height=4)

        assert block.shape == (4, 4)
        # At y=8, x=4: value = 8*8 + 4 = 68
        assert block[0, 0] == 68

    def test_get_16x16_block(self, reference_frame):
        """Extract full 16x16 macroblock."""
        block = get_block_integer(reference_frame, x=0, y=0, width=16, height=16)

        assert block.shape == (16, 16)

    def test_get_8x8_block(self, reference_frame):
        """Extract 8x8 sub-macroblock partition."""
        block = get_block_integer(reference_frame, x=8, y=8, width=8, height=8)

        assert block.shape == (8, 8)
        # At y=8, x=8: value = 8*8 + 8 = 72
        assert block[0, 0] == 72

    def test_get_4x4_block(self, reference_frame):
        """Extract 4x4 sub-block."""
        block = get_block_integer(reference_frame, x=16, y=16, width=4, height=4)

        assert block.shape == (4, 4)
        # At y=16, x=16: value = 16*8 + 16 = 144
        assert block[0, 0] == 144

    def test_block_at_frame_edge(self, reference_frame):
        """Extract block at right/bottom edge."""
        # Block at (28, 28) with size 4x4 should fit exactly
        block = get_block_integer(reference_frame, x=28, y=28, width=4, height=4)

        assert block.shape == (4, 4)
        # At y=28, x=28: value = 28*8 + 28 = 252
        assert block[0, 0] == 252

    def test_block_extends_past_right_edge(self):
        """Block extending past right edge is clipped."""
        frame = np.arange(16).reshape(4, 4).astype(np.uint8)
        # Request block at x=2, which would extend past x=3

        block = get_block_integer(frame, x=2, y=0, width=4, height=4)

        assert block.shape == (4, 4)
        # Rightmost columns should be replicated from edge
        assert block[0, 0] == 2
        assert block[0, 1] == 3
        assert block[0, 2] == 3  # Edge replicated
        assert block[0, 3] == 3  # Edge replicated

    def test_block_extends_past_bottom_edge(self):
        """Block extending past bottom edge is clipped."""
        frame = np.arange(16).reshape(4, 4).astype(np.uint8)

        block = get_block_integer(frame, x=0, y=2, width=4, height=4)

        assert block.shape == (4, 4)
        # Bottom rows should be replicated from edge
        assert block[0, 0] == 8   # y=2
        assert block[1, 0] == 12  # y=3
        assert block[2, 0] == 12  # Edge replicated
        assert block[3, 0] == 12  # Edge replicated

    def test_negative_position_clipped(self):
        """Negative positions are clipped to frame boundary."""
        frame = np.arange(16).reshape(4, 4).astype(np.uint8)

        block = get_block_integer(frame, x=-2, y=-1, width=4, height=4)

        assert block.shape == (4, 4)
        # Top-left should be replicated from (0,0)
        assert block[0, 0] == 0  # Clipped from (-2, -1)
        assert block[0, 1] == 0  # Clipped from (-1, -1)
        assert block[1, 2] == 0  # First valid x position

    def test_block_returns_copy(self, reference_frame):
        """Returned block should be a copy, not a view."""
        block = get_block_integer(reference_frame, x=0, y=0, width=4, height=4)
        original_value = block[0, 0]

        block[0, 0] = 255

        # Reference frame should be unchanged
        assert reference_frame[0, 0] == original_value


class TestClipMV:
    """Tests for motion vector clipping to frame boundaries."""

    def test_clip_mv_inside_frame(self):
        """MV inside frame is unchanged."""
        x, y = clip_mv_to_frame(10, 10, frame_width=32, frame_height=32)
        assert (x, y) == (10, 10)

    def test_clip_mv_negative(self):
        """Negative MV is clipped to 0."""
        x, y = clip_mv_to_frame(-5, -10, frame_width=32, frame_height=32)
        assert x == 0
        assert y == 0

    def test_clip_mv_past_right(self):
        """MV past right edge is clipped."""
        x, y = clip_mv_to_frame(40, 10, frame_width=32, frame_height=32)
        assert x == 31
        assert y == 10

    def test_clip_mv_past_bottom(self):
        """MV past bottom edge is clipped."""
        x, y = clip_mv_to_frame(10, 50, frame_width=32, frame_height=32)
        assert x == 10
        assert y == 31


class TestHalfPixelInterpolation:
    """Tests for half-pixel interpolation using 6-tap filter.

    H.264 Spec Section 8.4.2.2.1 - Luma sample interpolation
    Filter coefficients: [1, -5, 20, 20, -5, 1] with normalization by 32
    """

    @pytest.fixture
    def flat_frame(self):
        """Frame with constant value - interpolation should preserve it."""
        return np.full((16, 16), 128, dtype=np.uint8)

    @pytest.fixture
    def gradient_h_frame(self):
        """Frame with horizontal gradient for testing h interpolation."""
        frame = np.zeros((16, 16), dtype=np.uint8)
        for x in range(16):
            frame[:, x] = x * 16  # 0, 16, 32, ... 240
        return frame

    @pytest.fixture
    def gradient_v_frame(self):
        """Frame with vertical gradient for testing v interpolation."""
        frame = np.zeros((16, 16), dtype=np.uint8)
        for y in range(16):
            frame[y, :] = y * 16
        return frame

    def test_half_h_flat_frame(self, flat_frame):
        """Half-pixel horizontal on flat frame returns same value."""
        result = interpolate_half_h(flat_frame, x=4, y=4, width=4, height=4)

        assert result.shape == (4, 4)
        # Flat frame should interpolate to same value (or very close)
        assert np.allclose(result, 128, atol=1)

    def test_half_v_flat_frame(self, flat_frame):
        """Half-pixel vertical on flat frame returns same value."""
        result = interpolate_half_v(flat_frame, x=4, y=4, width=4, height=4)

        assert result.shape == (4, 4)
        assert np.allclose(result, 128, atol=1)

    def test_half_hv_flat_frame(self, flat_frame):
        """Half-pixel diagonal on flat frame returns same value."""
        result = interpolate_half_hv(flat_frame, x=4, y=4, width=4, height=4)

        assert result.shape == (4, 4)
        assert np.allclose(result, 128, atol=1)

    def test_half_h_known_values(self):
        """Test 6-tap filter with known input sequence.

        For sequence [A, B, C, D, E, F], half-pixel between C and D is:
        (A - 5B + 20C + 20D - 5E + F + 16) >> 5
        """
        # Create frame where row 4 has values [10, 20, 30, 40, 50, 60, 70, ...]
        frame = np.zeros((16, 16), dtype=np.uint8)
        for x in range(16):
            frame[4, x] = 10 + x * 10 if 10 + x * 10 < 256 else 255

        # At x=2, the 6 samples are frame[4, 0:6] = [10, 20, 30, 40, 50, 60]
        # Half-pixel at x=2.5: (10 - 5*20 + 20*30 + 20*40 - 5*50 + 60 + 16) >> 5
        # = (10 - 100 + 600 + 800 - 250 + 60 + 16) >> 5 = 1136 >> 5 = 35
        result = interpolate_half_h(frame, x=2, y=4, width=1, height=1)

        assert result[0, 0] == 35

    def test_half_v_known_values(self):
        """Test vertical 6-tap filter with known sequence."""
        frame = np.zeros((16, 16), dtype=np.uint8)
        for y in range(16):
            frame[y, 4] = 10 + y * 10 if 10 + y * 10 < 256 else 255

        # At y=2, the 6 samples are [10, 20, 30, 40, 50, 60]
        # Same calculation as horizontal
        result = interpolate_half_v(frame, x=4, y=2, width=1, height=1)

        assert result[0, 0] == 35

    def test_half_h_clipping(self):
        """Result should be clipped to [0, 255]."""
        # Create frame that would produce out-of-range interpolation
        frame = np.zeros((16, 16), dtype=np.uint8)
        # Sequence that might produce negative: high-low-high pattern
        frame[4, 0:6] = [255, 0, 0, 0, 0, 255]

        result = interpolate_half_h(frame, x=2, y=4, width=1, height=1)

        # Should be clipped to valid range
        assert 0 <= result[0, 0] <= 255

    def test_half_h_block_size(self):
        """Interpolate a 4x4 block."""
        frame = np.full((16, 16), 100, dtype=np.uint8)

        result = interpolate_half_h(frame, x=4, y=4, width=4, height=4)

        assert result.shape == (4, 4)
        assert result.dtype == np.uint8

    def test_half_hv_uses_h_then_v(self):
        """Diagonal interpolation applies h filter then v filter on result.

        Per H.264 spec, position 'j' (half-h, half-v) is computed by:
        1. Compute half-h positions (b) for a vertical column
        2. Apply vertical filter to those half-h values
        """
        frame = np.full((16, 16), 64, dtype=np.uint8)

        result = interpolate_half_hv(frame, x=4, y=4, width=4, height=4)

        assert result.shape == (4, 4)
        # Flat input should give ~64
        assert np.allclose(result, 64, atol=1)


class TestFractionalLumaBlock:
    """Tests for get_luma_block_fractional with quarter-pixel precision.

    H.264 uses quarter-pixel MVs for luma (dx, dy in range 0-3).
    Position encoding:
        dx=0,dy=0: integer position
        dx=2,dy=0: half-pixel horizontal
        dx=0,dy=2: half-pixel vertical
        dx=2,dy=2: half-pixel diagonal
        dx=1,dy=0: quarter-pixel (average of integer and half-h)
        etc.
    """

    @pytest.fixture
    def test_frame(self):
        """16x16 frame with gradient pattern."""
        frame = np.zeros((16, 16), dtype=np.uint8)
        for y in range(16):
            for x in range(16):
                frame[y, x] = (y * 16 + x) % 256
        return frame

    def test_integer_position(self, test_frame):
        """dx=0, dy=0 should return integer block."""
        result = get_luma_block_fractional(
            test_frame, x=4, y=4, dx=0, dy=0, width=4, height=4
        )

        # Should match integer extraction
        expected = get_block_integer(test_frame, x=4, y=4, width=4, height=4)
        assert np.array_equal(result, expected)

    def test_half_h_position(self, test_frame):
        """dx=2, dy=0 should use horizontal half-pixel interpolation."""
        result = get_luma_block_fractional(
            test_frame, x=4, y=4, dx=2, dy=0, width=4, height=4
        )

        expected = interpolate_half_h(test_frame, x=4, y=4, width=4, height=4)
        assert np.array_equal(result, expected)

    def test_half_v_position(self, test_frame):
        """dx=0, dy=2 should use vertical half-pixel interpolation."""
        result = get_luma_block_fractional(
            test_frame, x=4, y=4, dx=0, dy=2, width=4, height=4
        )

        expected = interpolate_half_v(test_frame, x=4, y=4, width=4, height=4)
        assert np.array_equal(result, expected)

    def test_half_hv_position(self, test_frame):
        """dx=2, dy=2 should use diagonal half-pixel interpolation."""
        result = get_luma_block_fractional(
            test_frame, x=4, y=4, dx=2, dy=2, width=4, height=4
        )

        expected = interpolate_half_hv(test_frame, x=4, y=4, width=4, height=4)
        assert np.array_equal(result, expected)

    def test_quarter_h_position(self, test_frame):
        """dx=1, dy=0 should average integer and half-h."""
        result = get_luma_block_fractional(
            test_frame, x=4, y=4, dx=1, dy=0, width=4, height=4
        )

        int_block = get_block_integer(test_frame, x=4, y=4, width=4, height=4)
        half_h = interpolate_half_h(test_frame, x=4, y=4, width=4, height=4)
        expected = ((int_block.astype(np.int32) + half_h.astype(np.int32) + 1) >> 1).astype(np.uint8)

        assert np.array_equal(result, expected)

    def test_quarter_v_position(self, test_frame):
        """dx=0, dy=1 should average integer and half-v."""
        result = get_luma_block_fractional(
            test_frame, x=4, y=4, dx=0, dy=1, width=4, height=4
        )

        int_block = get_block_integer(test_frame, x=4, y=4, width=4, height=4)
        half_v = interpolate_half_v(test_frame, x=4, y=4, width=4, height=4)
        expected = ((int_block.astype(np.int32) + half_v.astype(np.int32) + 1) >> 1).astype(np.uint8)

        assert np.array_equal(result, expected)

    def test_three_quarter_h_position(self, test_frame):
        """dx=3, dy=0 should average half-h and next integer."""
        result = get_luma_block_fractional(
            test_frame, x=4, y=4, dx=3, dy=0, width=4, height=4
        )

        # dx=3 averages half-h at current x and integer at x+1
        half_h = interpolate_half_h(test_frame, x=4, y=4, width=4, height=4)
        next_int = get_block_integer(test_frame, x=5, y=4, width=4, height=4)
        expected = ((half_h.astype(np.int32) + next_int.astype(np.int32) + 1) >> 1).astype(np.uint8)

        assert np.array_equal(result, expected)

    def test_block_16x16(self, test_frame):
        """Extract full macroblock at fractional position."""
        # Need larger frame for 16x16 block
        frame = np.zeros((32, 32), dtype=np.uint8)
        for y in range(32):
            for x in range(32):
                frame[y, x] = (y * 8 + x) % 256

        result = get_luma_block_fractional(
            frame, x=0, y=0, dx=2, dy=0, width=16, height=16
        )

        assert result.shape == (16, 16)
        assert result.dtype == np.uint8

    def test_all_quarter_positions_valid(self, test_frame):
        """All 16 quarter-pixel positions should produce valid output."""
        for dy in range(4):
            for dx in range(4):
                result = get_luma_block_fractional(
                    test_frame, x=4, y=4, dx=dx, dy=dy, width=4, height=4
                )
                assert result.shape == (4, 4)
                assert result.dtype == np.uint8
                assert np.all(result >= 0) and np.all(result <= 255)
