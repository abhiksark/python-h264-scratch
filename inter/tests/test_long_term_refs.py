# h264/inter/tests/test_long_term_refs.py
"""RED TESTS: Long-Term Reference Picture Management.

Long-term reference pictures persist across IDR boundaries and are explicitly
managed by the encoder. They provide:
- Stable reference frames for periodic scene content
- Better compression for video with repetitive patterns
- Support for scene cuts and fast seeking

H.264 Spec Reference:
- Section 8.2.4.2: Initialization of reference picture lists
- Section 8.2.5: Decoded reference picture marking process

Key concepts:
- Long-term references are identified by LongTermPicNum (not FrameNum)
- max_long_term_frame_idx limits how many long-term refs can exist
- Short-term refs can be converted to long-term via MMCO
- Long-term refs are NOT affected by sliding window marking

These tests SHOULD FAIL until long-term reference support is implemented.
"""

import pytest
import numpy as np

from inter.reference import ReferenceFrame, ReferenceFrameBuffer


class TestLongTermRefModuleExists:
    """Tests for long-term reference module existence."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_long_term_ref_manager_exists(self):
        """LongTermRefManager class should exist."""
        try:
            from inter.long_term_refs import LongTermRefManager
            assert LongTermRefManager is not None
        except ImportError:
            pytest.fail("inter.long_term_refs module should exist")

    @pytest.mark.xfail(reason="Not implemented")
    def test_manager_has_mark_long_term_method(self):
        """Manager should have mark_as_long_term() method."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        assert hasattr(manager, 'mark_as_long_term'), \
            "Should have mark_as_long_term() method"

    @pytest.mark.xfail(reason="Not implemented")
    def test_manager_has_unmark_long_term_method(self):
        """Manager should have unmark_long_term() method."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        assert hasattr(manager, 'unmark_long_term'), \
            "Should have unmark_long_term() method"

    @pytest.mark.xfail(reason="Not implemented")
    def test_manager_has_get_long_term_ref_method(self):
        """Manager should have get_long_term_ref() method."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        assert hasattr(manager, 'get_long_term_ref'), \
            "Should have get_long_term_ref() method"


class TestMarkingLongTermReference:
    """Tests for marking pictures as long-term references."""

    def _make_frame(self, frame_num: int, poc: int) -> ReferenceFrame:
        """Helper to create test reference frames."""
        return ReferenceFrame(
            luma=np.full((16, 16), frame_num * 10, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=frame_num,
            poc=poc,
        )

    @pytest.mark.xfail(reason="Not implemented")
    def test_mark_short_term_as_long_term(self):
        """Convert a short-term ref to long-term."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)

        # Mark as long-term with long_term_frame_idx=0
        manager.mark_as_long_term(
            pic_num=5,  # Identifies the short-term ref
            long_term_frame_idx=0
        )

        # Should now be retrievable as long-term
        lt_ref = manager.get_long_term_ref(long_term_pic_num=0)
        assert lt_ref is not None
        assert lt_ref.frame_num == 5

    @pytest.mark.xfail(reason="Not implemented")
    def test_mark_long_term_removes_from_short_term(self):
        """Marking as long-term removes from short-term list."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)

        # Verify it's in short-term list
        assert manager.has_short_term_ref(pic_num=5)

        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=0)

        # Should no longer be in short-term list
        assert not manager.has_short_term_ref(pic_num=5)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mark_long_term_assigns_long_term_frame_idx(self):
        """Long-term ref should have correct long_term_frame_idx."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)

        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=3)

        lt_ref = manager.get_long_term_ref(long_term_pic_num=3)
        assert lt_ref is not None

    @pytest.mark.xfail(reason="Not implemented")
    def test_long_term_pic_num_equals_frame_idx(self):
        """For frames, LongTermPicNum = long_term_frame_idx."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)

        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=7)

        # LongTermPicNum should equal long_term_frame_idx for frames
        lt_ref = manager.get_long_term_ref(long_term_pic_num=7)
        assert lt_ref is not None

    @pytest.mark.xfail(reason="Not implemented")
    def test_mark_nonexistent_short_term_raises(self):
        """Marking non-existent short-term as long-term should fail."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        with pytest.raises(ValueError):
            manager.mark_as_long_term(pic_num=99, long_term_frame_idx=0)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mark_replaces_existing_long_term(self):
        """New long-term at same idx replaces existing."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        frame1 = self._make_frame(5, 10)
        frame2 = self._make_frame(8, 16)
        manager.add_short_term_ref(frame1)
        manager.add_short_term_ref(frame2)

        # Mark frame1 as long-term idx=0
        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=0)

        # Mark frame2 as long-term idx=0 (replaces frame1)
        manager.mark_as_long_term(pic_num=8, long_term_frame_idx=0)

        lt_ref = manager.get_long_term_ref(long_term_pic_num=0)
        assert lt_ref.frame_num == 8  # Should be frame2


class TestMaxLongTermFrameIdxHandling:
    """Tests for max_long_term_frame_idx management."""

    def _make_frame(self, frame_num: int, poc: int) -> ReferenceFrame:
        """Helper to create test reference frames."""
        return ReferenceFrame(
            luma=np.full((16, 16), frame_num * 10, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=frame_num,
            poc=poc,
        )

    @pytest.mark.xfail(reason="Not implemented")
    def test_set_max_long_term_frame_idx(self):
        """Setting max_long_term_frame_idx limits allowed indices."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        # Set max to 2 (allows idx 0, 1, 2)
        manager.set_max_long_term_frame_idx(max_idx=2)

        assert manager.get_max_long_term_frame_idx() == 2

    @pytest.mark.xfail(reason="Not implemented")
    def test_mark_exceeding_max_idx_raises(self):
        """Marking with idx > max should fail."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=2)

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)

        with pytest.raises(ValueError):
            manager.mark_as_long_term(pic_num=5, long_term_frame_idx=5)

    @pytest.mark.xfail(reason="Not implemented")
    def test_reduce_max_removes_excess_long_term(self):
        """Reducing max_idx removes long-term refs with idx > new max."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=5)

        # Add some long-term refs
        for i in range(4):
            frame = self._make_frame(i, i * 2)
            manager.add_short_term_ref(frame)
            manager.mark_as_long_term(pic_num=i, long_term_frame_idx=i)

        # Refs at idx 0, 1, 2, 3
        assert manager.get_long_term_ref(long_term_pic_num=3) is not None

        # Reduce max to 1 (only idx 0, 1 allowed)
        manager.set_max_long_term_frame_idx(max_idx=1)

        # Refs at idx 2, 3 should be removed
        assert manager.get_long_term_ref(long_term_pic_num=0) is not None
        assert manager.get_long_term_ref(long_term_pic_num=1) is not None

        with pytest.raises((ValueError, KeyError)):
            manager.get_long_term_ref(long_term_pic_num=2)

        with pytest.raises((ValueError, KeyError)):
            manager.get_long_term_ref(long_term_pic_num=3)

    @pytest.mark.xfail(reason="Not implemented")
    def test_max_idx_minus1_disables_long_term(self):
        """max_long_term_frame_idx = -1 means no long-term refs allowed."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        # max_long_term_frame_idx_plus1 = 0 means no long-term refs
        # which translates to max_idx = -1 (no valid index)
        manager.set_max_long_term_frame_idx(max_idx=-1)

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)

        with pytest.raises(ValueError):
            manager.mark_as_long_term(pic_num=5, long_term_frame_idx=0)

    @pytest.mark.xfail(reason="Not implemented")
    def test_max_idx_from_mmco4(self):
        """MMCO 4 sets max_long_term_frame_idx_plus1."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        # MMCO 4 with max_long_term_frame_idx_plus1 = 3
        # means max_idx = 2
        manager.process_mmco4(max_long_term_frame_idx_plus1=3)

        assert manager.get_max_long_term_frame_idx() == 2

    @pytest.mark.xfail(reason="Not implemented")
    def test_max_idx_zero_from_mmco4(self):
        """MMCO 4 with value 0 removes all long-term refs."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=5)

        # Add long-term ref
        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)
        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=0)

        # MMCO 4 with value 0 - removes all long-term
        manager.process_mmco4(max_long_term_frame_idx_plus1=0)

        assert manager.get_max_long_term_frame_idx() == -1
        assert manager.get_long_term_count() == 0


class TestLongTermReferenceListBuilding:
    """Tests for building reference lists including long-term refs."""

    def _make_frame(self, frame_num: int, poc: int) -> ReferenceFrame:
        """Helper to create test reference frames."""
        return ReferenceFrame(
            luma=np.full((16, 16), frame_num * 10, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=frame_num,
            poc=poc,
        )

    @pytest.mark.xfail(reason="Not implemented")
    def test_l0_list_includes_long_term(self):
        """L0 list should include long-term refs after short-term."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        # Add short-term refs
        for poc in [0, 2, 4]:
            frame = self._make_frame(poc // 2, poc)
            manager.add_short_term_ref(frame)

        # Mark one as long-term
        manager.mark_as_long_term(pic_num=0, long_term_frame_idx=0)

        # Build L0 for current POC=5
        l0_list = manager.build_l0_list(
            current_poc=5,
            num_ref_idx_l0_active=4
        )

        # Short-term should come first, then long-term
        # Short-term: POC 4, 2 (descending, closest first)
        # Long-term: idx 0 (was POC=0)
        l0_pocs = [f.poc for f in l0_list]

        # Verify long-term is included
        assert 0 in l0_pocs  # The long-term ref (originally POC=0)

    @pytest.mark.xfail(reason="Not implemented")
    def test_long_term_sorted_by_pic_num_ascending(self):
        """Long-term refs sorted by LongTermPicNum ascending."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=5)

        # Add and mark multiple long-term refs
        for i in [3, 1, 2]:  # Out of order
            frame = self._make_frame(i * 10, i * 100)
            manager.add_short_term_ref(frame)
            manager.mark_as_long_term(pic_num=i * 10, long_term_frame_idx=i)

        # Get long-term list
        lt_list = manager.get_long_term_list()

        # Should be sorted by LongTermPicNum ascending: 1, 2, 3
        lt_pic_nums = [manager.get_long_term_pic_num(f) for f in lt_list]
        assert lt_pic_nums == [1, 2, 3]

    @pytest.mark.xfail(reason="Not implemented")
    def test_mixed_short_and_long_term_in_l0(self):
        """L0 with both short-term and long-term refs."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=2)

        # Short-term refs
        for poc in [2, 4, 6]:
            frame = self._make_frame(poc // 2, poc)
            manager.add_short_term_ref(frame)

        # Long-term ref
        frame_lt = self._make_frame(100, 0)  # Old frame, now long-term
        manager.add_long_term_ref(frame_lt, long_term_frame_idx=0)

        l0_list = manager.build_l0_list(current_poc=7, num_ref_idx_l0_active=4)

        # Should have: [POC=6, POC=4, POC=2, LT_idx=0]
        assert len(l0_list) == 4

    @pytest.mark.xfail(reason="Not implemented")
    def test_l1_list_includes_long_term(self):
        """L1 list should include long-term refs."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=2)

        # Refs for B-frame at POC=5
        for poc in [0, 2, 8, 10]:  # Past and future
            frame = self._make_frame(poc, poc)
            manager.add_short_term_ref(frame)

        # Long-term ref
        frame_lt = self._make_frame(100, 50)
        manager.add_long_term_ref(frame_lt, long_term_frame_idx=0)

        l1_list = manager.build_l1_list(current_poc=5, num_ref_idx_l1_active=4)

        # L1: future short-term first, then long-term
        # Short-term future: POC 8, 10
        # Long-term: idx 0
        assert len(l1_list) >= 1


class TestLongTermToShortTermConversion:
    """Tests for converting long-term refs back to short-term."""

    def _make_frame(self, frame_num: int, poc: int) -> ReferenceFrame:
        """Helper to create test reference frames."""
        return ReferenceFrame(
            luma=np.full((16, 16), frame_num * 10, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=frame_num,
            poc=poc,
        )

    @pytest.mark.xfail(reason="Not implemented")
    def test_unmark_long_term_basic(self):
        """Unmark a long-term ref (doesn't convert, just removes)."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=2)

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)
        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=0)

        # Unmark - removes from long-term
        manager.unmark_long_term(long_term_pic_num=0)

        with pytest.raises((ValueError, KeyError)):
            manager.get_long_term_ref(long_term_pic_num=0)

    @pytest.mark.xfail(reason="Not implemented")
    def test_unmarked_frame_is_released(self):
        """Unmarking releases the frame (not converted to short-term)."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=2)

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)
        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=0)

        initial_total = manager.get_total_ref_count()

        manager.unmark_long_term(long_term_pic_num=0)

        # Total refs should decrease
        assert manager.get_total_ref_count() == initial_total - 1

    @pytest.mark.xfail(reason="Not implemented")
    def test_unmark_nonexistent_raises(self):
        """Unmarking non-existent long-term should raise."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=2)

        with pytest.raises((ValueError, KeyError)):
            manager.unmark_long_term(long_term_pic_num=99)


class TestIDRHandling:
    """Tests for IDR picture handling with long-term refs."""

    def _make_frame(self, frame_num: int, poc: int) -> ReferenceFrame:
        """Helper to create test reference frames."""
        return ReferenceFrame(
            luma=np.full((16, 16), frame_num * 10, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=frame_num,
            poc=poc,
        )

    @pytest.mark.xfail(reason="Not implemented")
    def test_idr_clears_short_term_refs(self):
        """IDR clears all short-term references."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        for i in range(3):
            manager.add_short_term_ref(self._make_frame(i, i * 2))

        assert manager.get_short_term_count() == 3

        manager.process_idr()

        assert manager.get_short_term_count() == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_idr_clears_long_term_refs_by_default(self):
        """IDR with long_term_reference_flag=0 clears long-term refs."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=2)

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)
        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=0)

        assert manager.get_long_term_count() == 1

        manager.process_idr(long_term_reference_flag=False)

        assert manager.get_long_term_count() == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_idr_with_long_term_flag_marks_idr_as_long_term(self):
        """IDR with long_term_reference_flag=1 marks IDR as long-term."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=2)

        # Process IDR with flag=True
        idr_frame = self._make_frame(0, 0)
        manager.process_idr(
            long_term_reference_flag=True,
            idr_frame=idr_frame
        )

        # IDR should be marked as long-term with idx=0
        lt_ref = manager.get_long_term_ref(long_term_pic_num=0)
        assert lt_ref is not None
        assert lt_ref.frame_num == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_idr_sets_max_long_term_idx(self):
        """IDR sets max_long_term_frame_idx appropriately."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=5)

        # IDR with flag=True sets max to 0
        idr_frame = self._make_frame(0, 0)
        manager.process_idr(
            long_term_reference_flag=True,
            idr_frame=idr_frame
        )

        assert manager.get_max_long_term_frame_idx() == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_idr_with_no_output_of_prior_pics(self):
        """IDR with no_output_of_prior_pics_flag handling."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()

        # Add frames to output queue
        for i in range(3):
            manager.add_to_output_queue(self._make_frame(i, i * 2))

        assert manager.get_output_queue_size() == 3

        # IDR with no_output_of_prior_pics=True
        manager.process_idr(no_output_of_prior_pics_flag=True)

        # Output queue should be cleared
        assert manager.get_output_queue_size() == 0


class TestSlidingWindowMarking:
    """Tests for sliding window reference picture marking."""

    def _make_frame(self, frame_num: int, poc: int) -> ReferenceFrame:
        """Helper to create test reference frames."""
        return ReferenceFrame(
            luma=np.full((16, 16), frame_num * 10, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=frame_num,
            poc=poc,
        )

    @pytest.mark.xfail(reason="Not implemented")
    def test_sliding_window_evicts_oldest_short_term(self):
        """Sliding window evicts oldest short-term ref."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_num_ref_frames(max_refs=2)

        # Add 2 short-term refs (at capacity)
        manager.add_short_term_ref(self._make_frame(0, 0))
        manager.add_short_term_ref(self._make_frame(1, 2))

        assert manager.get_short_term_count() == 2

        # Add another - should evict oldest (frame_num=0)
        manager.add_short_term_ref(self._make_frame(2, 4))

        assert manager.get_short_term_count() == 2
        assert not manager.has_short_term_ref(pic_num=0)
        assert manager.has_short_term_ref(pic_num=1)
        assert manager.has_short_term_ref(pic_num=2)

    @pytest.mark.xfail(reason="Not implemented")
    def test_sliding_window_doesnt_evict_long_term(self):
        """Sliding window does NOT evict long-term refs."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_num_ref_frames(max_refs=2)
        manager.set_max_long_term_frame_idx(max_idx=1)

        # Add short-term and mark as long-term
        manager.add_short_term_ref(self._make_frame(0, 0))
        manager.mark_as_long_term(pic_num=0, long_term_frame_idx=0)

        # Add short-term
        manager.add_short_term_ref(self._make_frame(1, 2))

        # Add another short-term (at capacity with 1 LT + 1 ST)
        manager.add_short_term_ref(self._make_frame(2, 4))

        # Long-term should still exist
        lt_ref = manager.get_long_term_ref(long_term_pic_num=0)
        assert lt_ref is not None

    @pytest.mark.xfail(reason="Not implemented")
    def test_sliding_window_considers_total_refs(self):
        """Sliding window counts both short and long-term."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_num_ref_frames(max_refs=3)
        manager.set_max_long_term_frame_idx(max_idx=1)

        # 1 long-term
        manager.add_short_term_ref(self._make_frame(0, 0))
        manager.mark_as_long_term(pic_num=0, long_term_frame_idx=0)

        # 2 short-term (total = 3, at capacity)
        manager.add_short_term_ref(self._make_frame(1, 2))
        manager.add_short_term_ref(self._make_frame(2, 4))

        assert manager.get_total_ref_count() == 3

        # Adding one more should evict oldest SHORT-term
        manager.add_short_term_ref(self._make_frame(3, 6))

        assert manager.get_total_ref_count() == 3
        assert not manager.has_short_term_ref(pic_num=1)  # Evicted


class TestReferenceFrameAttributes:
    """Tests for long-term attributes on ReferenceFrame."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_reference_frame_has_is_long_term_attr(self):
        """ReferenceFrame should have is_long_term attribute."""
        frame = ReferenceFrame(
            luma=np.zeros((16, 16), dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=0,
            poc=0,
        )

        assert hasattr(frame, 'is_long_term'), \
            "ReferenceFrame should have is_long_term attribute"

    @pytest.mark.xfail(reason="Not implemented")
    def test_reference_frame_has_long_term_frame_idx(self):
        """ReferenceFrame should have long_term_frame_idx attribute."""
        frame = ReferenceFrame(
            luma=np.zeros((16, 16), dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=0,
            poc=0,
        )

        assert hasattr(frame, 'long_term_frame_idx'), \
            "ReferenceFrame should have long_term_frame_idx attribute"

    @pytest.mark.xfail(reason="Not implemented")
    def test_new_frame_is_short_term_by_default(self):
        """New ReferenceFrame should be short-term by default."""
        frame = ReferenceFrame(
            luma=np.zeros((16, 16), dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=0,
            poc=0,
        )

        assert frame.is_long_term is False
        assert frame.long_term_frame_idx is None


class TestLongTermInReferenceLists:
    """Tests for long-term refs in L0/L1 list reordering."""

    def _make_frame(self, frame_num: int, poc: int) -> ReferenceFrame:
        """Helper to create test reference frames."""
        return ReferenceFrame(
            luma=np.full((16, 16), frame_num * 10, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=frame_num,
            poc=poc,
        )

    @pytest.mark.xfail(reason="Not implemented")
    def test_reorder_command_puts_long_term_first(self):
        """Reorder command can put long-term ref at position 0."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=2)

        # Short-term refs
        for poc in [2, 4, 6]:
            manager.add_short_term_ref(self._make_frame(poc // 2, poc))

        # Long-term ref
        manager.add_short_term_ref(self._make_frame(0, 0))
        manager.mark_as_long_term(pic_num=0, long_term_frame_idx=0)

        # Build default L0
        l0_list = manager.build_l0_list(current_poc=7, num_ref_idx_l0_active=4)

        # Reorder to put long-term first
        reorder_commands = [
            {'modification_of_pic_nums_idc': 2, 'long_term_pic_num': 0},
        ]

        reordered = manager.reorder_l0(l0_list, reorder_commands)

        # Long-term should now be first
        assert reordered[0].is_long_term
        assert reordered[0].long_term_frame_idx == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_long_term_pic_num_for_reorder(self):
        """Long-term reorder uses LongTermPicNum (not PicNum)."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=5)

        # Multiple long-term refs
        for i in [1, 3]:
            frame = self._make_frame(i * 10, i * 100)
            manager.add_short_term_ref(frame)
            manager.mark_as_long_term(pic_num=i * 10, long_term_frame_idx=i)

        # Reorder by long_term_pic_num=3
        reorder_commands = [
            {'modification_of_pic_nums_idc': 2, 'long_term_pic_num': 3},
        ]

        l0_list = manager.build_l0_list(current_poc=500, num_ref_idx_l0_active=2)
        reordered = manager.reorder_l0(l0_list, reorder_commands)

        # Frame with long_term_frame_idx=3 should be first
        assert reordered[0].long_term_frame_idx == 3


class TestIntegrationWithDecoder:
    """Tests for long-term ref integration with decoder."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_decoder_has_long_term_manager(self):
        """Decoder should have long-term reference manager."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        assert hasattr(decoder, 'long_term_manager') or \
               hasattr(decoder.state, 'long_term_refs'), \
            "Decoder should track long-term references"

    @pytest.mark.xfail(reason="Not implemented")
    def test_decoder_processes_long_term_marking(self):
        """Decoder should process MMCO long-term marking."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        assert hasattr(decoder, '_process_long_term_marking'), \
            "Decoder should have _process_long_term_marking method"

    @pytest.mark.xfail(reason="Not implemented")
    def test_decoder_builds_lists_with_long_term(self):
        """Decoder should include long-term refs in reference lists."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        assert hasattr(decoder, '_build_ref_lists_with_long_term'), \
            "Decoder should support building lists with long-term refs"


class TestEdgeCases:
    """Tests for edge cases in long-term reference handling."""

    def _make_frame(self, frame_num: int, poc: int) -> ReferenceFrame:
        """Helper to create test reference frames."""
        return ReferenceFrame(
            luma=np.full((16, 16), frame_num * 10, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=frame_num,
            poc=poc,
        )

    @pytest.mark.xfail(reason="Not implemented")
    def test_all_refs_long_term(self):
        """Handle case where all refs are long-term."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_num_ref_frames(max_refs=4)
        manager.set_max_long_term_frame_idx(max_idx=3)

        # Mark all as long-term
        for i in range(4):
            frame = self._make_frame(i, i * 2)
            manager.add_short_term_ref(frame)
            manager.mark_as_long_term(pic_num=i, long_term_frame_idx=i)

        assert manager.get_short_term_count() == 0
        assert manager.get_long_term_count() == 4

        l0_list = manager.build_l0_list(current_poc=10, num_ref_idx_l0_active=4)
        assert len(l0_list) == 4

    @pytest.mark.xfail(reason="Not implemented")
    def test_no_short_term_refs(self):
        """Handle case with no short-term refs (only long-term)."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=1)

        frame = self._make_frame(0, 0)
        manager.add_long_term_ref(frame, long_term_frame_idx=0)

        l0_list = manager.build_l0_list(current_poc=10, num_ref_idx_l0_active=1)
        assert len(l0_list) == 1
        assert l0_list[0].is_long_term

    @pytest.mark.xfail(reason="Not implemented")
    def test_reassign_long_term_idx(self):
        """Same frame can be reassigned to different long_term_frame_idx."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=5)

        frame = self._make_frame(5, 10)
        manager.add_short_term_ref(frame)
        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=0)

        # Get ref to verify
        lt_ref = manager.get_long_term_ref(long_term_pic_num=0)
        assert lt_ref is not None

        # Via MMCO, could reassign to different idx (spec allows this)
        # This would typically happen via MMCO 2 (unmark) then MMCO 3 (mark new)
        manager.unmark_long_term(long_term_pic_num=0)

        # Re-add and mark with different idx (unusual but valid)
        manager.add_short_term_ref(frame)
        manager.mark_as_long_term(pic_num=5, long_term_frame_idx=3)

        lt_ref = manager.get_long_term_ref(long_term_pic_num=3)
        assert lt_ref is not None
        assert lt_ref.frame_num == 5

    @pytest.mark.xfail(reason="Not implemented")
    def test_sparse_long_term_indices(self):
        """Long-term indices can be sparse (gaps in numbering)."""
        from inter.long_term_refs import LongTermRefManager

        manager = LongTermRefManager()
        manager.set_max_long_term_frame_idx(max_idx=10)

        # Add at indices 0, 5, 10 (sparse)
        for idx in [0, 5, 10]:
            frame = self._make_frame(idx, idx * 2)
            manager.add_short_term_ref(frame)
            manager.mark_as_long_term(pic_num=idx, long_term_frame_idx=idx)

        assert manager.get_long_term_count() == 3

        # Should be able to retrieve each
        for idx in [0, 5, 10]:
            lt_ref = manager.get_long_term_ref(long_term_pic_num=idx)
            assert lt_ref is not None
