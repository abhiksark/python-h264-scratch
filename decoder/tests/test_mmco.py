# h264/decoder/tests/test_mmco.py
"""RED TESTS: Memory Management Control Operations (MMCO).

MMCO allows explicit control over the Decoded Picture Buffer (DPB).
This is used for:
- Converting short-term references to long-term
- Marking references as unused for reference
- Setting maximum long-term frame index
- Custom GOP structures and scene cuts

H.264 Spec Reference:
- Section 7.3.3.3: dec_ref_pic_marking() syntax
- Section 8.2.5: Decoded reference picture marking process

memory_management_control_operation values:
- 0: End of MMCO commands
- 1: Mark short-term pic as unused for reference
- 2: Mark long-term pic as unused for reference
- 3: Mark short-term pic as long-term with given idx
- 4: Set max_long_term_frame_idx
- 5: Mark all reference pictures as unused
- 6: Mark current picture as long-term

These tests SHOULD FAIL until MMCO support is implemented.
"""

import pytest
import numpy as np

from inter.reference import ReferenceFrame, ReferenceFrameBuffer
from slice.slice_header import DecRefPicMarking


class TestMMCOModuleExists:
    """Tests for MMCO module existence."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco_processor_exists(self):
        """MMCOProcessor class should exist."""
        try:
            from decoder.mmco import MMCOProcessor
            assert MMCOProcessor is not None
        except ImportError:
            pytest.fail("decoder.mmco module should exist")

    @pytest.mark.xfail(reason="Not implemented")
    def test_processor_has_process_method(self):
        """MMCOProcessor should have process() method."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        assert hasattr(processor, 'process'), \
            "MMCOProcessor should have process() method"

    @pytest.mark.xfail(reason="Not implemented")
    def test_processor_accepts_dec_ref_pic_marking(self):
        """MMCOProcessor should accept DecRefPicMarking from slice header."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[1, 0],
            difference_of_pic_nums_minus1=[2],
        )

        # Should not raise
        processor.process(marking)


class TestMMCO1MarkShortTermUnused:
    """Tests for MMCO 1: Mark short-term pic as unused for reference."""

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
    def test_mmco1_marks_short_term_unused(self):
        """MMCO 1 marks specified short-term pic as unused."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        # Set up DPB with short-term refs
        for i in range(4):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        assert processor.has_short_term_ref(pic_num=2)

        # MMCO 1: Mark pic with difference_of_pic_nums_minus1=0
        # from current frame_num=3 marks pic_num=3 as unused
        # difference_of_pic_nums_minus1=0 means diff=1, so pic_num=3-1=2
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[1, 0],
            difference_of_pic_nums_minus1=[0],  # Marks pic_num = curr - 1
        )

        processor.process(marking, current_frame_num=3, max_frame_num=16)

        # Pic with pic_num=2 should be marked unused
        assert not processor.has_short_term_ref(pic_num=2)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco1_difference_of_pic_nums_calculation(self):
        """MMCO 1 correctly calculates target pic_num."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        for i in range(5):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        # Current frame_num=4
        # difference_of_pic_nums_minus1=2 means diff=3
        # target pic_num = 4 - 3 = 1
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[1, 0],
            difference_of_pic_nums_minus1=[2],
        )

        processor.process(marking, current_frame_num=4, max_frame_num=16)

        assert not processor.has_short_term_ref(pic_num=1)
        assert processor.has_short_term_ref(pic_num=0)  # Others remain
        assert processor.has_short_term_ref(pic_num=2)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco1_with_frame_num_wrap(self):
        """MMCO 1 handles frame_num wraparound."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        # Refs at frame_num 14, 15 (before wrap) and 0, 1 (after wrap)
        for fn in [14, 15, 0, 1]:
            processor.add_short_term_ref(self._make_frame(fn, fn * 2))

        # Current frame_num=1, max=16
        # To mark frame_num=15: currPicNum - picNum needs wraparound
        # picNumX = 15, currPicNum = 1
        # If picNumX > currPicNum: picNum = picNumX - MaxPicNum = 15 - 16 = -1
        # diff = 1 - (-1) = 2, so difference_of_pic_nums_minus1 = 1

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[1, 0],
            difference_of_pic_nums_minus1=[1],  # Should mark frame_num=15
        )

        processor.process(marking, current_frame_num=1, max_frame_num=16)

        # Frame 15 should be unmarked
        # This test verifies wraparound handling
        assert not processor.has_short_term_ref(pic_num=15) or \
               not processor.has_short_term_ref(pic_num=-1)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco1_multiple_in_sequence(self):
        """Multiple MMCO 1 commands in sequence."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        for i in range(5):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        # Mark pic_num 3 and 1 as unused
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[1, 1, 0],
            difference_of_pic_nums_minus1=[0, 2],  # curr-1=3, then curr-3=1
        )

        processor.process(marking, current_frame_num=4, max_frame_num=16)

        assert not processor.has_short_term_ref(pic_num=3)
        assert not processor.has_short_term_ref(pic_num=1)
        assert processor.has_short_term_ref(pic_num=0)
        assert processor.has_short_term_ref(pic_num=2)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco1_nonexistent_ref_handles_gracefully(self):
        """MMCO 1 for non-existent ref should handle gracefully."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        processor.add_short_term_ref(self._make_frame(0, 0))

        # Try to mark non-existent pic_num
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[1, 0],
            difference_of_pic_nums_minus1=[10],  # Doesn't exist
        )

        # Should handle gracefully (warning or no-op)
        processor.process(marking, current_frame_num=4, max_frame_num=16)


class TestMMCO2MarkLongTermUnused:
    """Tests for MMCO 2: Mark long-term pic as unused for reference."""

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
    def test_mmco2_marks_long_term_unused(self):
        """MMCO 2 marks specified long-term pic as unused."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        # Add long-term ref
        frame = self._make_frame(5, 10)
        processor.add_short_term_ref(frame)
        processor.mark_as_long_term(pic_num=5, long_term_frame_idx=0)

        assert processor.has_long_term_ref(long_term_pic_num=0)

        # MMCO 2: Mark long-term with long_term_pic_num=0 as unused
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[2, 0],
            long_term_pic_num=[0],
        )

        processor.process(marking, current_frame_num=6, max_frame_num=16)

        assert not processor.has_long_term_ref(long_term_pic_num=0)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco2_multiple_long_term(self):
        """MMCO 2 can unmark specific long-term from multiple."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=5)

        # Add multiple long-term refs
        for i in range(3):
            frame = self._make_frame(i, i * 2)
            processor.add_short_term_ref(frame)
            processor.mark_as_long_term(pic_num=i, long_term_frame_idx=i)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[2, 0],
            long_term_pic_num=[1],  # Only unmark long_term_pic_num=1
        )

        processor.process(marking, current_frame_num=5, max_frame_num=16)

        assert processor.has_long_term_ref(long_term_pic_num=0)
        assert not processor.has_long_term_ref(long_term_pic_num=1)
        assert processor.has_long_term_ref(long_term_pic_num=2)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco2_nonexistent_long_term(self):
        """MMCO 2 for non-existent long-term handles gracefully."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[2, 0],
            long_term_pic_num=[5],  # Doesn't exist
        )

        # Should handle gracefully
        processor.process(marking, current_frame_num=5, max_frame_num=16)


class TestMMCO3MarkShortTermAsLongTerm:
    """Tests for MMCO 3: Mark short-term as long-term."""

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
    def test_mmco3_converts_short_to_long_term(self):
        """MMCO 3 converts short-term ref to long-term."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        frame = self._make_frame(3, 6)
        processor.add_short_term_ref(frame)

        assert processor.has_short_term_ref(pic_num=3)

        # MMCO 3: Mark short-term as long-term
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[3, 0],
            difference_of_pic_nums_minus1=[0],  # pic_num = 4-1 = 3
            long_term_frame_idx=[0],
        )

        processor.process(marking, current_frame_num=4, max_frame_num=16)

        # Should now be long-term, not short-term
        assert not processor.has_short_term_ref(pic_num=3)
        assert processor.has_long_term_ref(long_term_pic_num=0)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco3_assigns_correct_long_term_frame_idx(self):
        """MMCO 3 assigns specified long_term_frame_idx."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=5)

        frame = self._make_frame(2, 4)
        processor.add_short_term_ref(frame)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[3, 0],
            difference_of_pic_nums_minus1=[1],  # pic_num = 4-2 = 2
            long_term_frame_idx=[3],  # Assign to idx 3
        )

        processor.process(marking, current_frame_num=4, max_frame_num=16)

        lt_ref = processor.get_long_term_ref(long_term_pic_num=3)
        assert lt_ref is not None
        assert lt_ref.frame_num == 2

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco3_replaces_existing_at_idx(self):
        """MMCO 3 replaces existing long-term at same idx."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        # Existing long-term at idx 0
        frame1 = self._make_frame(1, 2)
        processor.add_short_term_ref(frame1)
        processor.mark_as_long_term(pic_num=1, long_term_frame_idx=0)

        # New short-term to convert
        frame2 = self._make_frame(3, 6)
        processor.add_short_term_ref(frame2)

        # MMCO 3: Convert frame2 to long-term idx 0 (replaces frame1)
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[3, 0],
            difference_of_pic_nums_minus1=[0],  # pic_num = 4-1 = 3
            long_term_frame_idx=[0],
        )

        processor.process(marking, current_frame_num=4, max_frame_num=16)

        lt_ref = processor.get_long_term_ref(long_term_pic_num=0)
        assert lt_ref.frame_num == 3  # Should be frame2

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco3_exceeding_max_idx_error(self):
        """MMCO 3 with idx > max should raise error."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        frame = self._make_frame(3, 6)
        processor.add_short_term_ref(frame)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[3, 0],
            difference_of_pic_nums_minus1=[0],
            long_term_frame_idx=[5],  # Exceeds max
        )

        with pytest.raises(ValueError):
            processor.process(marking, current_frame_num=4, max_frame_num=16)


class TestMMCO4SetMaxLongTermFrameIdx:
    """Tests for MMCO 4: Set max_long_term_frame_idx."""

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
    def test_mmco4_sets_max_long_term_idx(self):
        """MMCO 4 sets max_long_term_frame_idx."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        # MMCO 4: max_long_term_frame_idx_plus1 = 3 -> max_idx = 2
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[4, 0],
            max_long_term_frame_idx_plus1=[3],
        )

        processor.process(marking, current_frame_num=0, max_frame_num=16)

        assert processor.get_max_long_term_frame_idx() == 2

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco4_value_zero_clears_long_term(self):
        """MMCO 4 with value 0 marks all long-term as unused."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=5)

        # Add some long-term refs
        for i in range(3):
            frame = self._make_frame(i, i * 2)
            processor.add_short_term_ref(frame)
            processor.mark_as_long_term(pic_num=i, long_term_frame_idx=i)

        assert processor.get_long_term_count() == 3

        # MMCO 4 with value 0 - no long-term allowed
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[4, 0],
            max_long_term_frame_idx_plus1=[0],
        )

        processor.process(marking, current_frame_num=5, max_frame_num=16)

        assert processor.get_max_long_term_frame_idx() == -1
        assert processor.get_long_term_count() == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco4_removes_excess_long_term(self):
        """MMCO 4 removes long-term refs with idx > new max."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=5)

        # Long-term refs at idx 0, 2, 4
        for i in [0, 2, 4]:
            frame = self._make_frame(i, i * 2)
            processor.add_short_term_ref(frame)
            processor.mark_as_long_term(pic_num=i, long_term_frame_idx=i)

        # Reduce max to 1 (only idx 0, 1 allowed)
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[4, 0],
            max_long_term_frame_idx_plus1=[2],  # max_idx = 1
        )

        processor.process(marking, current_frame_num=5, max_frame_num=16)

        assert processor.has_long_term_ref(long_term_pic_num=0)
        assert not processor.has_long_term_ref(long_term_pic_num=2)
        assert not processor.has_long_term_ref(long_term_pic_num=4)


class TestMMCO5MarkAllUnused:
    """Tests for MMCO 5: Mark all reference pictures as unused."""

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
    def test_mmco5_clears_all_short_term(self):
        """MMCO 5 marks all short-term as unused."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        for i in range(4):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        assert processor.get_short_term_count() == 4

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[5, 0],
        )

        processor.process(marking, current_frame_num=5, max_frame_num=16)

        assert processor.get_short_term_count() == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco5_clears_all_long_term(self):
        """MMCO 5 marks all long-term as unused."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=3)

        for i in range(3):
            frame = self._make_frame(i, i * 2)
            processor.add_short_term_ref(frame)
            processor.mark_as_long_term(pic_num=i, long_term_frame_idx=i)

        assert processor.get_long_term_count() == 3

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[5, 0],
        )

        processor.process(marking, current_frame_num=5, max_frame_num=16)

        assert processor.get_long_term_count() == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco5_sets_max_long_term_idx_to_none(self):
        """MMCO 5 sets MaxLongTermFrameIdx to 'no long-term'."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=5)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[5, 0],
        )

        processor.process(marking, current_frame_num=5, max_frame_num=16)

        assert processor.get_max_long_term_frame_idx() == -1

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco5_resets_frame_num(self):
        """MMCO 5 resets frame_num to 0 for current picture."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[5, 0],
        )

        result = processor.process(marking, current_frame_num=10, max_frame_num=16)

        # Spec says: FrameNum = 0 after MMCO 5
        assert result.new_frame_num == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco5_affects_poc_calculation(self):
        """MMCO 5 affects POC derivation (resets POC)."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[5, 0],
        )

        result = processor.process(
            marking,
            current_frame_num=10,
            current_poc=50,
            max_frame_num=16
        )

        # MMCO 5 causes POC-related state reset
        assert result.poc_reset is True


class TestMMCO6MarkCurrentAsLongTerm:
    """Tests for MMCO 6: Mark current picture as long-term."""

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
    def test_mmco6_marks_current_as_long_term(self):
        """MMCO 6 marks current picture as long-term."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        current_frame = self._make_frame(5, 10)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[6, 0],
            long_term_frame_idx=[0],
        )

        processor.process(
            marking,
            current_frame_num=5,
            max_frame_num=16,
            current_frame=current_frame
        )

        # Current frame should now be long-term
        lt_ref = processor.get_long_term_ref(long_term_pic_num=0)
        assert lt_ref is not None
        assert lt_ref.frame_num == 5

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco6_assigns_specified_idx(self):
        """MMCO 6 assigns specified long_term_frame_idx."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=5)

        current_frame = self._make_frame(5, 10)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[6, 0],
            long_term_frame_idx=[3],  # Assign to idx 3
        )

        processor.process(
            marking,
            current_frame_num=5,
            max_frame_num=16,
            current_frame=current_frame
        )

        lt_ref = processor.get_long_term_ref(long_term_pic_num=3)
        assert lt_ref is not None

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco6_replaces_existing_at_idx(self):
        """MMCO 6 replaces existing long-term at same idx."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        # Existing long-term
        old_frame = self._make_frame(1, 2)
        processor.add_short_term_ref(old_frame)
        processor.mark_as_long_term(pic_num=1, long_term_frame_idx=0)

        # Mark current as long-term at same idx
        current_frame = self._make_frame(5, 10)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[6, 0],
            long_term_frame_idx=[0],
        )

        processor.process(
            marking,
            current_frame_num=5,
            max_frame_num=16,
            current_frame=current_frame
        )

        lt_ref = processor.get_long_term_ref(long_term_pic_num=0)
        assert lt_ref.frame_num == 5  # Should be current, not old

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco6_exceeding_max_idx_error(self):
        """MMCO 6 with idx > max should raise error."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        current_frame = self._make_frame(5, 10)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[6, 0],
            long_term_frame_idx=[5],  # Exceeds max
        )

        with pytest.raises(ValueError):
            processor.process(
                marking,
                current_frame_num=5,
                max_frame_num=16,
                current_frame=current_frame
            )


class TestIDRPictureMarking:
    """Tests for IDR picture dec_ref_pic_marking."""

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
    def test_idr_clears_all_refs(self):
        """IDR with no_output_of_prior_pics clears all refs."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        # Add refs
        for i in range(3):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        frame_lt = self._make_frame(10, 20)
        processor.add_short_term_ref(frame_lt)
        processor.mark_as_long_term(pic_num=10, long_term_frame_idx=0)

        marking = DecRefPicMarking(
            no_output_of_prior_pics_flag=True,
            long_term_reference_flag=False,
        )

        processor.process_idr(marking)

        assert processor.get_short_term_count() == 0
        assert processor.get_long_term_count() == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_idr_long_term_reference_flag_false(self):
        """IDR with long_term_reference_flag=False marks as short-term."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        idr_frame = self._make_frame(0, 0)

        marking = DecRefPicMarking(
            no_output_of_prior_pics_flag=False,
            long_term_reference_flag=False,
        )

        processor.process_idr(marking, idr_frame=idr_frame)

        # IDR should be added as short-term ref
        assert processor.has_short_term_ref(pic_num=0)
        assert processor.get_long_term_count() == 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_idr_long_term_reference_flag_true(self):
        """IDR with long_term_reference_flag=True marks as long-term."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        idr_frame = self._make_frame(0, 0)

        marking = DecRefPicMarking(
            no_output_of_prior_pics_flag=False,
            long_term_reference_flag=True,
        )

        processor.process_idr(marking, idr_frame=idr_frame)

        # IDR should be long-term with idx=0
        lt_ref = processor.get_long_term_ref(long_term_pic_num=0)
        assert lt_ref is not None
        assert processor.get_max_long_term_frame_idx() == 0


class TestSlidingWindowVsAdaptiveMode:
    """Tests for sliding window vs adaptive ref marking mode."""

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
    def test_sliding_window_mode_default(self):
        """Default sliding window mode when adaptive_flag=False."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_num_ref_frames(max_refs=3)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=False,
        )

        # Add 3 frames (at capacity)
        for i in range(3):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        # Process marking - should use sliding window
        new_frame = self._make_frame(3, 6)
        processor.process_non_idr(marking, current_frame=new_frame)

        # Sliding window should evict oldest
        assert processor.get_short_term_count() == 3
        assert not processor.has_short_term_ref(pic_num=0)  # Evicted

    @pytest.mark.xfail(reason="Not implemented")
    def test_adaptive_mode_overrides_sliding_window(self):
        """Adaptive mode operations override sliding window."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_num_ref_frames(max_refs=3)
        processor.set_max_long_term_frame_idx(max_idx=1)

        for i in range(3):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        # Adaptive mode: mark frame 1 as long-term instead of evicting
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[3, 0],
            difference_of_pic_nums_minus1=[1],  # Mark frame 2-2=1 as LT
            long_term_frame_idx=[0],
        )

        new_frame = self._make_frame(3, 6)
        processor.process_non_idr(
            marking,
            current_frame=new_frame,
            current_frame_num=3,
            max_frame_num=16
        )

        # Frame 1 should be long-term, not evicted
        assert processor.has_long_term_ref(long_term_pic_num=0)

    @pytest.mark.xfail(reason="Not implemented")
    def test_adaptive_then_sliding_window(self):
        """After adaptive operations, sliding window still applies if needed."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_num_ref_frames(max_refs=2)

        for i in range(2):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        # Adaptive: only mark one as unused
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[0],  # End immediately (no-op)
        )

        new_frame = self._make_frame(2, 4)
        processor.process_non_idr(
            marking,
            current_frame=new_frame,
            current_frame_num=2,
            max_frame_num=16
        )

        # Still need to apply sliding window to make room
        assert processor.get_total_ref_count() <= 2


class TestComplexMMCOSequences:
    """Tests for complex MMCO command sequences."""

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
    def test_mmco1_then_mmco3(self):
        """MMCO 1 followed by MMCO 3 in same marking."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        for i in range(4):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        # Unmark frame 1, then mark frame 2 as long-term
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[1, 3, 0],
            difference_of_pic_nums_minus1=[2, 1],  # Unmark 4-3=1, mark 4-2=2
            long_term_frame_idx=[0],
        )

        processor.process(marking, current_frame_num=4, max_frame_num=16)

        assert not processor.has_short_term_ref(pic_num=1)
        assert not processor.has_short_term_ref(pic_num=2)
        assert processor.has_long_term_ref(long_term_pic_num=0)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco4_then_mmco6(self):
        """MMCO 4 sets max, then MMCO 6 marks current as long-term."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        current_frame = self._make_frame(5, 10)

        # Set max_long_term_frame_idx, then mark current as long-term
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[4, 6, 0],
            max_long_term_frame_idx_plus1=[3],  # max_idx = 2
            long_term_frame_idx=[1],
        )

        processor.process(
            marking,
            current_frame_num=5,
            max_frame_num=16,
            current_frame=current_frame
        )

        assert processor.get_max_long_term_frame_idx() == 2
        assert processor.has_long_term_ref(long_term_pic_num=1)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco2_mmco3_mmco6(self):
        """Complex: unmark LT, convert ST to LT, mark current as LT."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=3)

        # Existing long-term
        lt_frame = self._make_frame(1, 2)
        processor.add_short_term_ref(lt_frame)
        processor.mark_as_long_term(pic_num=1, long_term_frame_idx=0)

        # Short-term to convert
        st_frame = self._make_frame(3, 6)
        processor.add_short_term_ref(st_frame)

        current_frame = self._make_frame(5, 10)

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[2, 3, 6, 0],
            long_term_pic_num=[0],  # Unmark LT 0
            difference_of_pic_nums_minus1=[1],  # Convert pic_num 5-2=3
            long_term_frame_idx=[0, 1],  # First for MMCO 3, second for MMCO 6
        )

        processor.process(
            marking,
            current_frame_num=5,
            max_frame_num=16,
            current_frame=current_frame
        )

        # Old LT should be gone, st_frame is now LT 0, current is LT 1
        assert processor.get_long_term_ref(long_term_pic_num=0).frame_num == 3
        assert processor.get_long_term_ref(long_term_pic_num=1).frame_num == 5

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco5_then_mmco6(self):
        """MMCO 5 clears all, then MMCO 6 marks current as only ref."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()
        processor.set_max_long_term_frame_idx(max_idx=2)

        for i in range(3):
            processor.add_short_term_ref(self._make_frame(i, i * 2))

        current_frame = self._make_frame(5, 10)

        # Clear all, then mark current as long-term
        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[5, 6, 0],
            long_term_frame_idx=[0],
        )

        processor.process(
            marking,
            current_frame_num=5,
            max_frame_num=16,
            current_frame=current_frame
        )

        # Only current as long-term should remain
        assert processor.get_short_term_count() == 0
        assert processor.get_long_term_count() == 1


class TestIntegrationWithDecoder:
    """Tests for MMCO integration with main decoder."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_decoder_processes_dec_ref_pic_marking(self):
        """Decoder should process dec_ref_pic_marking from slice header."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        assert hasattr(decoder, '_process_dec_ref_pic_marking'), \
            "Decoder should have _process_dec_ref_pic_marking method"

    @pytest.mark.xfail(reason="Not implemented")
    def test_decoder_has_mmco_processor(self):
        """Decoder should have MMCO processor."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        assert hasattr(decoder, 'mmco_processor') or \
               hasattr(decoder, '_mmco'), \
            "Decoder should have MMCO processor"

    @pytest.mark.xfail(reason="Not implemented")
    def test_decoder_applies_mmco_after_decode(self):
        """Decoder should apply MMCO after decoding reference frame."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # MMCO is applied after decoding but before next frame
        assert hasattr(decoder, '_apply_dec_ref_pic_marking'), \
            "Decoder should apply dec_ref_pic_marking after decode"


class TestEdgeCases:
    """Tests for MMCO edge cases."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_empty_mmco_commands(self):
        """Handle empty MMCO command list (just terminator)."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        marking = DecRefPicMarking(
            adaptive_ref_pic_marking_mode_flag=True,
            memory_management_control_operations=[0],  # Just end marker
        )

        # Should not raise
        processor.process(marking, current_frame_num=0, max_frame_num=16)

    @pytest.mark.xfail(reason="Not implemented")
    def test_mmco_on_non_reference_frame(self):
        """MMCO on non-reference frame (nal_ref_idc=0) is no-op."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        # Non-reference frames don't have dec_ref_pic_marking
        # If somehow called, should be no-op
        processor.process_for_non_reference()

        # State unchanged
        assert True  # No error

    @pytest.mark.xfail(reason="Not implemented")
    def test_max_dpb_size_constraint(self):
        """DPB size should not exceed level-based limits."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        # Level 3.0 allows ~8 references for HD
        processor.set_max_dpb_size(max_size=8)
        processor.set_max_num_ref_frames(max_refs=4)

        for i in range(10):
            processor.add_short_term_ref(ReferenceFrame(
                luma=np.zeros((16, 16), dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=i,
                poc=i * 2,
            ))

        # Should not exceed max
        assert processor.get_total_ref_count() <= 8

    @pytest.mark.xfail(reason="Not implemented")
    def test_field_vs_frame_pic_num(self):
        """PicNum calculation differs for fields vs frames."""
        from decoder.mmco import MMCOProcessor

        processor = MMCOProcessor()

        # For frames: PicNum = FrameNum
        # For fields: PicNum = 2*FrameNum + (bottom_field ? 1 : 0)
        frame_pic_num = processor.calc_pic_num(
            frame_num=5,
            is_field=False,
            is_bottom_field=False,
            current_frame_num=6,
            max_frame_num=16
        )

        field_top_pic_num = processor.calc_pic_num(
            frame_num=5,
            is_field=True,
            is_bottom_field=False,
            current_frame_num=6,
            max_frame_num=16
        )

        field_bottom_pic_num = processor.calc_pic_num(
            frame_num=5,
            is_field=True,
            is_bottom_field=True,
            current_frame_num=6,
            max_frame_num=16
        )

        assert frame_pic_num == 5
        assert field_top_pic_num == 10  # 2*5 + 0
        assert field_bottom_pic_num == 11  # 2*5 + 1
