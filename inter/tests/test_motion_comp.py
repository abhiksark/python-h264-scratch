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
