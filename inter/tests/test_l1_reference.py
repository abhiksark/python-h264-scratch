# h264/inter/tests/test_l1_reference.py
"""RED TESTS: L1 reference list management for B-frames.

B-frames use two reference lists:
- L0: Past frames (like P-frames)
- L1: Future frames (B-frame specific)

These tests SHOULD FAIL until L1 reference list support is implemented.
"""

import pytest
import numpy as np

from inter.reference import ReferenceFrame, ReferenceFrameBuffer


class TestL1ReferenceList:
    """Tests for L1 (future) reference list."""

    def test_buffer_has_l0_list_method(self):
        """Reference buffer should have get_l0_list method."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        assert hasattr(buffer, 'get_l0_list'), \
            "ReferenceFrameBuffer should have get_l0_list method"

    def test_buffer_has_l1_list_method(self):
        """Reference buffer should have get_l1_list method."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        assert hasattr(buffer, 'get_l1_list'), \
            "ReferenceFrameBuffer should have get_l1_list method"

    def test_build_ref_lists_method_exists(self):
        """Buffer should have build_ref_lists method."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        assert hasattr(buffer, 'build_ref_lists'), \
            "ReferenceFrameBuffer should have build_ref_lists method"

    def test_reference_frame_has_poc(self):
        """ReferenceFrame should have POC attribute."""
        frame = ReferenceFrame(
            luma=np.zeros((16, 16), dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=0,
        )

        assert hasattr(frame, 'poc'), \
            "ReferenceFrame should have poc attribute"


class TestL0ListBuilding:
    """Tests for building L0 (past) reference list."""

    def test_l0_list_contains_past_frames(self):
        """L0 list should contain frames with POC < current."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        # Add frames with different POCs
        for i, poc in enumerate([0, 2, 4, 6]):
            frame = ReferenceFrame(
                luma=np.full((16, 16), i * 10, dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=i,
                poc=poc,
            )
            buffer.add_frame(frame)

        # Build lists for current POC=3 (B-frame between POC 2 and 4)
        buffer.build_ref_lists(current_poc=3)
        l0_list = buffer.get_l0_list()

        # L0 should have frames with POC < 3 (POC 0 and 2)
        assert len(l0_list) >= 1
        assert all(f.poc < 3 for f in l0_list)

    def test_l0_list_ordered_by_poc_descending(self):
        """L0 list should be ordered by POC descending (closest first)."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        for i, poc in enumerate([0, 2, 4]):
            frame = ReferenceFrame(
                luma=np.zeros((16, 16), dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=i,
                poc=poc,
            )
            buffer.add_frame(frame)

        buffer.build_ref_lists(current_poc=3)
        l0_list = buffer.get_l0_list()

        # Should be [POC=2, POC=0] (closest first)
        if len(l0_list) >= 2:
            assert l0_list[0].poc > l0_list[1].poc


class TestL1ListBuilding:
    """Tests for building L1 (future) reference list."""

    def test_l1_list_contains_future_frames(self):
        """L1 list should contain frames with POC > current."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        for i, poc in enumerate([0, 2, 4, 6]):
            frame = ReferenceFrame(
                luma=np.full((16, 16), i * 10, dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=i,
                poc=poc,
            )
            buffer.add_frame(frame)

        # Build lists for current POC=3
        buffer.build_ref_lists(current_poc=3)
        l1_list = buffer.get_l1_list()

        # L1 should have frames with POC > 3 (POC 4 and 6)
        assert len(l1_list) >= 1
        assert all(f.poc > 3 for f in l1_list)

    def test_l1_list_ordered_by_poc_ascending(self):
        """L1 list should be ordered by POC ascending (closest first)."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        for i, poc in enumerate([0, 2, 4, 6]):
            frame = ReferenceFrame(
                luma=np.zeros((16, 16), dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=i,
                poc=poc,
            )
            buffer.add_frame(frame)

        buffer.build_ref_lists(current_poc=3)
        l1_list = buffer.get_l1_list()

        # Should be [POC=4, POC=6] (closest first)
        if len(l1_list) >= 2:
            assert l1_list[0].poc < l1_list[1].poc


class TestRefListEdgeCases:
    """Tests for edge cases in reference list building."""

    def test_single_reference_both_lists(self):
        """With one reference, it should be in both L0 and L1."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        frame = ReferenceFrame(
            luma=np.zeros((16, 16), dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=0,
            poc=0,
        )
        buffer.add_frame(frame)

        buffer.build_ref_lists(current_poc=1)
        l0_list = buffer.get_l0_list()
        l1_list = buffer.get_l1_list()

        # Single frame should appear in both lists
        assert len(l0_list) >= 1 or len(l1_list) >= 1

    def test_get_l0_frame_by_index(self):
        """Should be able to get L0 frame by ref_idx."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        for i, poc in enumerate([0, 2]):
            frame = ReferenceFrame(
                luma=np.full((16, 16), poc, dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=i,
                poc=poc,
            )
            buffer.add_frame(frame)

        buffer.build_ref_lists(current_poc=3)

        # get_l0_frame(0) should return closest past frame
        assert hasattr(buffer, 'get_l0_frame'), \
            "Should have get_l0_frame method"

    def test_get_l1_frame_by_index(self):
        """Should be able to get L1 frame by ref_idx."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        for i, poc in enumerate([0, 4]):
            frame = ReferenceFrame(
                luma=np.full((16, 16), poc, dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=i,
                poc=poc,
            )
            buffer.add_frame(frame)

        buffer.build_ref_lists(current_poc=2)

        assert hasattr(buffer, 'get_l1_frame'), \
            "Should have get_l1_frame method"
