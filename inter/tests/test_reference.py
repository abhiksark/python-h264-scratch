# h264/inter/tests/test_reference.py
"""Tests for reference frame buffer.

Tests the storage and retrieval of decoded frames for inter prediction.
"""

import pytest
import numpy as np

from inter.reference import ReferenceFrame, ReferenceFrameBuffer


class TestReferenceFrame:
    """Tests for ReferenceFrame data class."""

    def test_create_reference_frame(self):
        """Create a reference frame with all components."""
        luma = np.zeros((16, 16), dtype=np.uint8)
        cb = np.zeros((8, 8), dtype=np.uint8)
        cr = np.zeros((8, 8), dtype=np.uint8)

        frame = ReferenceFrame(
            luma=luma,
            cb=cb,
            cr=cr,
            frame_num=0
        )

        assert frame.luma.shape == (16, 16)
        assert frame.cb.shape == (8, 8)
        assert frame.cr.shape == (8, 8)
        assert frame.frame_num == 0

    def test_reference_frame_stores_data(self):
        """Verify frame data is stored correctly."""
        luma = np.full((16, 16), 128, dtype=np.uint8)
        cb = np.full((8, 8), 64, dtype=np.uint8)
        cr = np.full((8, 8), 192, dtype=np.uint8)

        frame = ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=5)

        assert np.all(frame.luma == 128)
        assert np.all(frame.cb == 64)
        assert np.all(frame.cr == 192)
        assert frame.frame_num == 5

    def test_reference_frame_different_sizes(self):
        """Support various frame sizes."""
        # 32x32 frame (4:2:0)
        luma = np.zeros((32, 32), dtype=np.uint8)
        cb = np.zeros((16, 16), dtype=np.uint8)
        cr = np.zeros((16, 16), dtype=np.uint8)

        frame = ReferenceFrame(luma=luma, cb=cb, cr=cr, frame_num=0)

        assert frame.luma.shape == (32, 32)
        assert frame.cb.shape == (16, 16)


class TestReferenceFrameBuffer:
    """Tests for ReferenceFrameBuffer management."""

    def _make_frame(self, frame_num: int, fill_value: int = 0) -> ReferenceFrame:
        """Helper to create test frames."""
        return ReferenceFrame(
            luma=np.full((16, 16), fill_value, dtype=np.uint8),
            cb=np.full((8, 8), fill_value, dtype=np.uint8),
            cr=np.full((8, 8), fill_value, dtype=np.uint8),
            frame_num=frame_num
        )

    def test_create_empty_buffer(self):
        """Create buffer with specified max size."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        assert buffer.max_frames == 4
        assert len(buffer) == 0

    def test_add_single_frame(self):
        """Add one frame to buffer."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        frame = self._make_frame(frame_num=0, fill_value=100)

        buffer.add_frame(frame)

        assert len(buffer) == 1

    def test_get_frame_by_ref_idx(self):
        """Retrieve frame by reference index (0 = most recent)."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        frame0 = self._make_frame(frame_num=0, fill_value=100)
        frame1 = self._make_frame(frame_num=1, fill_value=200)

        buffer.add_frame(frame0)
        buffer.add_frame(frame1)

        # ref_idx 0 = most recent (frame1)
        retrieved = buffer.get_frame(ref_idx=0)
        assert np.all(retrieved.luma == 200)
        assert retrieved.frame_num == 1

        # ref_idx 1 = previous (frame0)
        retrieved = buffer.get_frame(ref_idx=1)
        assert np.all(retrieved.luma == 100)
        assert retrieved.frame_num == 0

    def test_buffer_overflow_evicts_oldest(self):
        """When buffer is full, oldest frame is evicted."""
        buffer = ReferenceFrameBuffer(max_frames=2)

        buffer.add_frame(self._make_frame(0, fill_value=100))
        buffer.add_frame(self._make_frame(1, fill_value=150))
        buffer.add_frame(self._make_frame(2, fill_value=200))  # Evicts frame 0

        assert len(buffer) == 2

        # ref_idx 0 = most recent (frame 2)
        assert buffer.get_frame(0).frame_num == 2

        # ref_idx 1 = previous (frame 1)
        assert buffer.get_frame(1).frame_num == 1

    def test_get_frame_invalid_index_raises(self):
        """Invalid reference index raises error."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        buffer.add_frame(self._make_frame(0))

        with pytest.raises(IndexError):
            buffer.get_frame(ref_idx=1)  # Only one frame in buffer

        with pytest.raises(IndexError):
            buffer.get_frame(ref_idx=-1)  # Negative index

    def test_clear_buffer(self):
        """Clear removes all frames."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        buffer.add_frame(self._make_frame(0))
        buffer.add_frame(self._make_frame(1))

        buffer.clear()

        assert len(buffer) == 0

    def test_buffer_max_frames_one(self):
        """Buffer with max_frames=1 (minimal case)."""
        buffer = ReferenceFrameBuffer(max_frames=1)

        buffer.add_frame(self._make_frame(0, fill_value=50))
        assert buffer.get_frame(0).frame_num == 0

        buffer.add_frame(self._make_frame(1, fill_value=100))
        assert len(buffer) == 1
        assert buffer.get_frame(0).frame_num == 1

    def test_get_frame_by_frame_num(self):
        """Retrieve frame by frame_num (for reference picture matching)."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        buffer.add_frame(self._make_frame(5, fill_value=50))
        buffer.add_frame(self._make_frame(7, fill_value=70))
        buffer.add_frame(self._make_frame(9, fill_value=90))

        frame = buffer.get_frame_by_num(7)
        assert frame is not None
        assert frame.frame_num == 7
        assert np.all(frame.luma == 70)

    def test_get_frame_by_frame_num_not_found(self):
        """Return None when frame_num not in buffer."""
        buffer = ReferenceFrameBuffer(max_frames=4)
        buffer.add_frame(self._make_frame(5))

        assert buffer.get_frame_by_num(10) is None

    def test_frame_order_preserved(self):
        """Frames maintain decode order (newest first in ref list)."""
        buffer = ReferenceFrameBuffer(max_frames=4)

        for i in range(4):
            buffer.add_frame(self._make_frame(i, fill_value=i * 50))

        # Verify order: ref_idx 0 = frame 3, ref_idx 3 = frame 0
        for ref_idx in range(4):
            expected_frame_num = 3 - ref_idx
            assert buffer.get_frame(ref_idx).frame_num == expected_frame_num
