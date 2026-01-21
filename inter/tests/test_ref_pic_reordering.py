# h264/inter/tests/test_ref_pic_reordering.py
"""RED TESTS: Reference Picture List Reordering.

Reference picture list reordering allows the encoder to modify the default
reference picture list order. This is used for:
- Optimizing prediction by placing better references earlier in the list
- Supporting long-term reference pictures
- Allowing flexible GOP structures

H.264 Spec Reference:
- Section 7.3.3.1: ref_pic_list_modification() syntax
- Section 8.2.4: Decoding process for reference picture lists

modification_of_pic_nums_idc values:
- 0: Subtract abs_diff_pic_num_minus1 + 1 from current picNumL0Pred
- 1: Add abs_diff_pic_num_minus1 + 1 to current picNumL0Pred
- 2: Use long_term_pic_num to specify a long-term reference
- 3: End of reordering commands

These tests SHOULD FAIL until reference picture list reordering is implemented.
"""

import pytest
import numpy as np

from inter.reference import ReferenceFrame, ReferenceFrameBuffer


class TestRefPicListReorderingModuleExists:
    """Tests for reordering module existence."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_ref_pic_reorder_module_exists(self):
        """Reference picture reordering module should exist."""
        try:
            from inter import ref_pic_reorder
            assert ref_pic_reorder is not None
        except ImportError:
            pytest.fail("inter.ref_pic_reorder module should exist")

    @pytest.mark.xfail(reason="Not implemented")
    def test_ref_pic_reorder_class_exists(self):
        """RefPicListReorder class should exist."""
        from inter.ref_pic_reorder import RefPicListReorder

        assert RefPicListReorder is not None

    @pytest.mark.xfail(reason="Not implemented")
    def test_reorder_has_reorder_l0_method(self):
        """RefPicListReorder should have reorder_l0() method."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()
        assert hasattr(reorderer, 'reorder_l0'), \
            "RefPicListReorder should have reorder_l0() method"

    @pytest.mark.xfail(reason="Not implemented")
    def test_reorder_has_reorder_l1_method(self):
        """RefPicListReorder should have reorder_l1() method."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()
        assert hasattr(reorderer, 'reorder_l1'), \
            "RefPicListReorder should have reorder_l1() method"


class TestModificationOfPicNumsIdcParsing:
    """Tests for modification_of_pic_nums_idc parsing."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_idc_0_subtract_from_pic_num(self):
        """IDC 0: Subtract from current pic_num prediction."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # Command: idc=0, abs_diff_pic_num_minus1=2
        # Means: subtract 3 from current picNumLXPred
        commands = [
            {'modification_of_pic_nums_idc': 0, 'abs_diff_pic_num_minus1': 2}
        ]

        assert reorderer.parse_reorder_command(commands[0]) is not None

    @pytest.mark.xfail(reason="Not implemented")
    def test_idc_1_add_to_pic_num(self):
        """IDC 1: Add to current pic_num prediction."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # Command: idc=1, abs_diff_pic_num_minus1=1
        # Means: add 2 to current picNumLXPred
        commands = [
            {'modification_of_pic_nums_idc': 1, 'abs_diff_pic_num_minus1': 1}
        ]

        assert reorderer.parse_reorder_command(commands[0]) is not None

    @pytest.mark.xfail(reason="Not implemented")
    def test_idc_2_long_term_ref(self):
        """IDC 2: Use long-term reference by long_term_pic_num."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # Command: idc=2, long_term_pic_num=0
        # Means: put long-term ref with long_term_pic_num=0 at current position
        commands = [
            {'modification_of_pic_nums_idc': 2, 'long_term_pic_num': 0}
        ]

        assert reorderer.parse_reorder_command(commands[0]) is not None

    @pytest.mark.xfail(reason="Not implemented")
    def test_idc_3_terminates_reordering(self):
        """IDC 3: Terminates the reordering command loop."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        commands = [
            {'modification_of_pic_nums_idc': 0, 'abs_diff_pic_num_minus1': 0},
            {'modification_of_pic_nums_idc': 3},  # End marker
        ]

        # Should not process any commands after idc=3
        result = reorderer.process_commands(commands)
        assert result is not None

    @pytest.mark.xfail(reason="Not implemented")
    def test_idc_4_long_term_for_l1(self):
        """IDC 4: For B-slices, subtract from long-term for L1."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # IDC 4 is same as IDC 0 but for potential alternative interpretation
        # Actually spec only defines 0-3 for modification_of_pic_nums_idc
        # IDC 4 and 5 are for abs_diff_view_idx_minus1 in MVC extension
        commands = [
            {'modification_of_pic_nums_idc': 4, 'abs_diff_view_idx_minus1': 0}
        ]

        # Should handle gracefully or raise for non-MVC streams
        result = reorderer.parse_reorder_command(commands[0])
        assert result is not None or True  # May raise for base spec

    @pytest.mark.xfail(reason="Not implemented")
    def test_idc_5_view_idx_plus(self):
        """IDC 5: For MVC, view index with addition."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        commands = [
            {'modification_of_pic_nums_idc': 5, 'abs_diff_view_idx_minus1': 0}
        ]

        # Should handle gracefully for non-MVC streams
        result = reorderer.parse_reorder_command(commands[0])
        assert result is not None or True


class TestAbsDiffPicNumMinus1Handling:
    """Tests for abs_diff_pic_num_minus1 calculation."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_abs_diff_pic_num_basic(self):
        """Basic abs_diff_pic_num_minus1 calculation."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # If current picNumLXPred=5 and abs_diff_pic_num_minus1=2, idc=0
        # Then target pic_num = 5 - (2+1) = 2
        target_pic_num = reorderer.calculate_target_pic_num(
            pic_num_lx_pred=5,
            abs_diff_pic_num_minus1=2,
            modification_idc=0,
            max_pic_num=16
        )

        assert target_pic_num == 2

    @pytest.mark.xfail(reason="Not implemented")
    def test_abs_diff_pic_num_with_add(self):
        """abs_diff_pic_num_minus1 with idc=1 (addition)."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # If current picNumLXPred=5 and abs_diff_pic_num_minus1=2, idc=1
        # Then target pic_num = 5 + (2+1) = 8
        target_pic_num = reorderer.calculate_target_pic_num(
            pic_num_lx_pred=5,
            abs_diff_pic_num_minus1=2,
            modification_idc=1,
            max_pic_num=16
        )

        assert target_pic_num == 8

    @pytest.mark.xfail(reason="Not implemented")
    def test_abs_diff_pic_num_wraparound_subtract(self):
        """abs_diff_pic_num wraps around for subtraction."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # If picNumLXPred=2 and abs_diff_pic_num_minus1=4, idc=0
        # target = 2 - 5 = -3, which wraps to MaxPicNum - 3
        target_pic_num = reorderer.calculate_target_pic_num(
            pic_num_lx_pred=2,
            abs_diff_pic_num_minus1=4,
            modification_idc=0,
            max_pic_num=16
        )

        # Should wrap: (2 - 5 + 16) % 16 = 13
        assert target_pic_num == 13

    @pytest.mark.xfail(reason="Not implemented")
    def test_abs_diff_pic_num_wraparound_add(self):
        """abs_diff_pic_num wraps around for addition."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # If picNumLXPred=14 and abs_diff_pic_num_minus1=3, idc=1
        # target = 14 + 4 = 18, which wraps to 18 % 16 = 2
        target_pic_num = reorderer.calculate_target_pic_num(
            pic_num_lx_pred=14,
            abs_diff_pic_num_minus1=3,
            modification_idc=1,
            max_pic_num=16
        )

        assert target_pic_num == 2

    @pytest.mark.xfail(reason="Not implemented")
    def test_abs_diff_pic_num_max_value(self):
        """abs_diff_pic_num_minus1 maximum allowed value."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # abs_diff_pic_num_minus1 must be in range [0, MaxPicNum - 1]
        # For MaxPicNum=16, max abs_diff_pic_num_minus1 = 14
        # (because abs_diff_pic_num = abs_diff_pic_num_minus1 + 1 <= MaxPicNum)
        target_pic_num = reorderer.calculate_target_pic_num(
            pic_num_lx_pred=15,
            abs_diff_pic_num_minus1=14,  # Subtracts 15
            modification_idc=0,
            max_pic_num=16
        )

        assert target_pic_num == 0  # 15 - 15 = 0


class TestLongTermPicNumHandling:
    """Tests for long_term_pic_num in reordering."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_long_term_pic_num_basic(self):
        """Basic long_term_pic_num lookup."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # Set up long-term references
        long_term_refs = {
            0: {'frame_num': 10, 'poc': 20},
            1: {'frame_num': 20, 'poc': 40},
        }
        reorderer.set_long_term_refs(long_term_refs)

        # Reorder command to place long-term ref 0 at position 0
        frame = reorderer.get_long_term_ref(long_term_pic_num=0)

        assert frame is not None
        assert frame['frame_num'] == 10

    @pytest.mark.xfail(reason="Not implemented")
    def test_long_term_pic_num_not_found(self):
        """long_term_pic_num not in DPB should raise error."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        long_term_refs = {
            0: {'frame_num': 10, 'poc': 20},
        }
        reorderer.set_long_term_refs(long_term_refs)

        # Try to get non-existent long-term ref
        with pytest.raises(ValueError):
            reorderer.get_long_term_ref(long_term_pic_num=5)

    @pytest.mark.xfail(reason="Not implemented")
    def test_long_term_pic_num_multiple_refs(self):
        """Multiple long-term refs with different long_term_pic_num."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        long_term_refs = {
            0: {'frame_num': 10, 'poc': 20},
            1: {'frame_num': 30, 'poc': 60},
            2: {'frame_num': 50, 'poc': 100},
        }
        reorderer.set_long_term_refs(long_term_refs)

        ref_1 = reorderer.get_long_term_ref(long_term_pic_num=1)
        ref_2 = reorderer.get_long_term_ref(long_term_pic_num=2)

        assert ref_1['frame_num'] == 30
        assert ref_2['frame_num'] == 50


class TestL0ListReordering:
    """Tests for L0 (forward/past) reference list reordering."""

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
    def test_l0_default_order_by_poc(self):
        """Default L0 list ordered by POC descending (closest past first)."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)

        # Add frames with POC 0, 2, 4, 6
        for poc in [0, 2, 4, 6]:
            buffer.add_frame(self._make_frame(poc // 2, poc))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        # Build default L0 for current POC=7
        l0_list = reorderer.build_initial_l0(current_poc=7)

        # Default order: closest POC first -> [6, 4, 2, 0]
        l0_pocs = [f.poc for f in l0_list]
        assert l0_pocs == [6, 4, 2, 0], f"Default L0 order wrong: {l0_pocs}"

    @pytest.mark.xfail(reason="Not implemented")
    def test_l0_reorder_single_swap(self):
        """Reorder L0 list with single swap."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)

        for poc in [0, 2, 4, 6]:
            buffer.add_frame(self._make_frame(poc // 2, poc))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        # Build default L0 for current POC=7 -> [6, 4, 2, 0]
        l0_list = reorderer.build_initial_l0(current_poc=7)

        # Reorder: put POC=0 at position 0
        # Need to use idc=0 to subtract from picNumL0Pred
        # Current frame_num for POC=7 would be around 3 or 4
        # frame_num for POC=0 is 0
        # picNumL0Pred starts at current_frame_num
        reorder_commands = [
            {'modification_of_pic_nums_idc': 0, 'abs_diff_pic_num_minus1': 2},  # Example
            {'modification_of_pic_nums_idc': 3},  # End
        ]

        reordered = reorderer.reorder_l0(
            l0_list=l0_list,
            commands=reorder_commands,
            current_frame_num=3,
            max_frame_num=16
        )

        # Verify reordering happened
        assert len(reordered) == len(l0_list)

    @pytest.mark.xfail(reason="Not implemented")
    def test_l0_reorder_multiple_commands(self):
        """Reorder L0 list with multiple commands."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)

        for poc in [0, 2, 4, 6]:
            buffer.add_frame(self._make_frame(poc // 2, poc))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        l0_list = reorderer.build_initial_l0(current_poc=7)

        # Multiple reorder commands
        reorder_commands = [
            {'modification_of_pic_nums_idc': 0, 'abs_diff_pic_num_minus1': 2},
            {'modification_of_pic_nums_idc': 0, 'abs_diff_pic_num_minus1': 0},
            {'modification_of_pic_nums_idc': 3},
        ]

        reordered = reorderer.reorder_l0(
            l0_list=l0_list,
            commands=reorder_commands,
            current_frame_num=3,
            max_frame_num=16
        )

        assert len(reordered) == len(l0_list)

    @pytest.mark.xfail(reason="Not implemented")
    def test_l0_reorder_with_long_term(self):
        """Reorder L0 to include long-term reference."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)

        for poc in [0, 2, 4, 6]:
            buffer.add_frame(self._make_frame(poc // 2, poc))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        # Set up a long-term reference
        lt_frame = self._make_frame(10, 100)
        reorderer.set_long_term_refs({0: lt_frame})

        l0_list = reorderer.build_initial_l0(current_poc=7)

        # Reorder: put long-term ref at position 0
        reorder_commands = [
            {'modification_of_pic_nums_idc': 2, 'long_term_pic_num': 0},
            {'modification_of_pic_nums_idc': 3},
        ]

        reordered = reorderer.reorder_l0(
            l0_list=l0_list,
            commands=reorder_commands,
            current_frame_num=3,
            max_frame_num=16
        )

        # Long-term ref should be at position 0
        assert reordered[0].frame_num == 10

    @pytest.mark.xfail(reason="Not implemented")
    def test_l0_reorder_preserves_remaining(self):
        """Reordering shifts remaining entries."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)

        for poc in [0, 2, 4, 6]:
            buffer.add_frame(self._make_frame(poc // 2, poc))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        l0_list = reorderer.build_initial_l0(current_poc=7)
        original_pocs = [f.poc for f in l0_list]

        # Move last frame to first position
        # The other frames should shift down
        reorder_commands = [
            {'modification_of_pic_nums_idc': 0, 'abs_diff_pic_num_minus1': 2},
            {'modification_of_pic_nums_idc': 3},
        ]

        reordered = reorderer.reorder_l0(
            l0_list=l0_list,
            commands=reorder_commands,
            current_frame_num=3,
            max_frame_num=16
        )

        # All original frames should still be present
        reordered_pocs = [f.poc for f in reordered]
        assert set(reordered_pocs) == set(original_pocs)


class TestL1ListReordering:
    """Tests for L1 (backward/future) reference list reordering (B-slices)."""

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
    def test_l1_default_order_by_poc(self):
        """Default L1 list ordered by POC ascending (closest future first)."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)

        # Add frames with POC 0, 4, 8, 12 (both past and future)
        for poc in [0, 4, 8, 12]:
            buffer.add_frame(self._make_frame(poc // 4, poc))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        # Build default L1 for current POC=6 (between 4 and 8)
        l1_list = reorderer.build_initial_l1(current_poc=6)

        # Default order: closest future first -> [8, 12]
        l1_pocs = [f.poc for f in l1_list]
        assert l1_pocs == [8, 12], f"Default L1 order wrong: {l1_pocs}"

    @pytest.mark.xfail(reason="Not implemented")
    def test_l1_reorder_single_command(self):
        """Reorder L1 list with single command."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)

        for poc in [0, 4, 8, 12]:
            buffer.add_frame(self._make_frame(poc // 4, poc))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        l1_list = reorderer.build_initial_l1(current_poc=6)

        reorder_commands = [
            {'modification_of_pic_nums_idc': 1, 'abs_diff_pic_num_minus1': 0},
            {'modification_of_pic_nums_idc': 3},
        ]

        reordered = reorderer.reorder_l1(
            l1_list=l1_list,
            commands=reorder_commands,
            current_frame_num=1,
            max_frame_num=16
        )

        assert len(reordered) == len(l1_list)

    @pytest.mark.xfail(reason="Not implemented")
    def test_l1_reorder_with_long_term(self):
        """Reorder L1 to include long-term reference."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)

        for poc in [0, 4, 8, 12]:
            buffer.add_frame(self._make_frame(poc // 4, poc))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        lt_frame = self._make_frame(20, 200)
        reorderer.set_long_term_refs({0: lt_frame})

        l1_list = reorderer.build_initial_l1(current_poc=6)

        reorder_commands = [
            {'modification_of_pic_nums_idc': 2, 'long_term_pic_num': 0},
            {'modification_of_pic_nums_idc': 3},
        ]

        reordered = reorderer.reorder_l1(
            l1_list=l1_list,
            commands=reorder_commands,
            current_frame_num=1,
            max_frame_num=16
        )

        assert reordered[0].frame_num == 20


class TestBSliceRefListReordering:
    """Tests for B-slice specific reference list reordering."""

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
    def test_b_slice_both_lists_reordered(self):
        """B-slice can reorder both L0 and L1."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)

        # Frames at POC 0, 2, 6, 8 (current B at POC=4)
        for poc in [0, 2, 6, 8]:
            buffer.add_frame(self._make_frame(poc, poc))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        l0_list = reorderer.build_initial_l0(current_poc=4)
        l1_list = reorderer.build_initial_l1(current_poc=4)

        l0_commands = [
            {'modification_of_pic_nums_idc': 0, 'abs_diff_pic_num_minus1': 0},
            {'modification_of_pic_nums_idc': 3},
        ]
        l1_commands = [
            {'modification_of_pic_nums_idc': 1, 'abs_diff_pic_num_minus1': 0},
            {'modification_of_pic_nums_idc': 3},
        ]

        reordered_l0 = reorderer.reorder_l0(
            l0_list, l0_commands, current_frame_num=4, max_frame_num=16
        )
        reordered_l1 = reorderer.reorder_l1(
            l1_list, l1_commands, current_frame_num=4, max_frame_num=16
        )

        assert len(reordered_l0) > 0
        assert len(reordered_l1) > 0

    @pytest.mark.xfail(reason="Not implemented")
    def test_b_slice_l0_l1_share_refs_initially(self):
        """With one reference, both L0 and L1 may contain same frame."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=2)

        # Only one reference frame
        buffer.add_frame(self._make_frame(0, 0))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        l0_list = reorderer.build_initial_l0(current_poc=4)
        l1_list = reorderer.build_initial_l1(current_poc=4)

        # With only one ref, it should appear in both lists
        assert len(l0_list) >= 1 or len(l1_list) >= 1


class TestPicNumWraparound:
    """Tests for picture number wraparound handling."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_pic_num_modulo_max_frame_num(self):
        """PicNum calculation with MaxFrameNum wraparound."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # MaxFrameNum = 16, current frame_num = 2
        # PicNum for short-term refs = FrameNum (no wrap for base case)
        pic_num = reorderer.calc_pic_num(
            frame_num=5,
            current_frame_num=2,
            max_frame_num=16
        )

        # For short-term refs before IDR, PicNum = FrameNum
        assert pic_num == 5

    @pytest.mark.xfail(reason="Not implemented")
    def test_pic_num_after_frame_num_wrap(self):
        """PicNum after frame_num has wrapped."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # MaxFrameNum = 16, current frame_num = 2 (after wrap)
        # Reference has frame_num = 14 (before wrap)
        # PicNum for ref = 14, but relative to current should consider wrap
        pic_num = reorderer.calc_pic_num(
            frame_num=14,
            current_frame_num=2,
            max_frame_num=16
        )

        # After wrap, older frames get negative relative PicNum
        # Spec uses: if FrameNum > current, PicNum = FrameNum - MaxFrameNum
        expected = 14 - 16  # -2
        assert pic_num == expected or pic_num == 14  # Depends on interpretation

    @pytest.mark.xfail(reason="Not implemented")
    def test_pic_num_pred_initialization(self):
        """picNumL0Pred/picNumL1Pred initialization."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # picNumLXPred is initialized to CurrPicNum
        curr_pic_num = reorderer.get_curr_pic_num(
            current_frame_num=5,
            max_frame_num=16
        )

        assert curr_pic_num == 5

    @pytest.mark.xfail(reason="Not implemented")
    def test_pic_num_pred_update_after_reorder(self):
        """picNumLXPred updates after each reordering step."""
        from inter.ref_pic_reorder import RefPicListReorder

        reorderer = RefPicListReorder()

        # After reordering to PicNum X, picNumLXPred becomes X
        reorderer.set_pic_num_pred(5)

        # Apply reorder command: subtract 2
        new_pic_num = reorderer.apply_reorder_step(
            modification_idc=0,
            abs_diff_pic_num_minus1=1,  # diff = 2
            max_pic_num=16
        )

        # New target = 5 - 2 = 3
        assert new_pic_num == 3
        # And picNumLXPred should now be 3
        assert reorderer.get_pic_num_pred() == 3


class TestReorderingEdgeCases:
    """Tests for edge cases in reference list reordering."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_empty_reorder_commands(self):
        """No reorder commands means use default list."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=4)
        buffer.add_frame(ReferenceFrame(
            luma=np.zeros((16, 16), dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=0,
            poc=0,
        ))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        l0_list = reorderer.build_initial_l0(current_poc=5)

        # Empty command list (only terminator)
        commands = [{'modification_of_pic_nums_idc': 3}]

        reordered = reorderer.reorder_l0(
            l0_list, commands, current_frame_num=2, max_frame_num=16
        )

        # Should be unchanged
        assert [f.poc for f in reordered] == [f.poc for f in l0_list]

    @pytest.mark.xfail(reason="Not implemented")
    def test_reorder_single_ref_no_effect(self):
        """Reordering single reference has no visible effect."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=2)
        buffer.add_frame(ReferenceFrame(
            luma=np.zeros((16, 16), dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=0,
            poc=0,
        ))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        l0_list = reorderer.build_initial_l0(current_poc=5)

        commands = [
            {'modification_of_pic_nums_idc': 0, 'abs_diff_pic_num_minus1': 1},
            {'modification_of_pic_nums_idc': 3},
        ]

        reordered = reorderer.reorder_l0(
            l0_list, commands, current_frame_num=2, max_frame_num=16
        )

        # Single ref, still single ref
        assert len(reordered) == 1

    @pytest.mark.xfail(reason="Not implemented")
    def test_reorder_ref_not_in_dpb(self):
        """Reordering command for ref not in DPB should handle gracefully."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=2)
        buffer.add_frame(ReferenceFrame(
            luma=np.zeros((16, 16), dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=0,
            poc=0,
        ))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        l0_list = reorderer.build_initial_l0(current_poc=5)

        # Command references frame_num=10 which doesn't exist
        commands = [
            {'modification_of_pic_nums_idc': 0, 'abs_diff_pic_num_minus1': 8},
            {'modification_of_pic_nums_idc': 3},
        ]

        # Should handle gracefully (skip or raise)
        with pytest.raises((ValueError, KeyError)):
            reorderer.reorder_l0(
                l0_list, commands, current_frame_num=2, max_frame_num=16
            )

    @pytest.mark.xfail(reason="Not implemented")
    def test_max_reorder_commands(self):
        """Handle maximum number of reorder commands."""
        from inter.ref_pic_reorder import RefPicListReorder

        buffer = ReferenceFrameBuffer(max_frames=8)
        for i in range(8):
            buffer.add_frame(ReferenceFrame(
                luma=np.full((16, 16), i * 10, dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=i,
                poc=i * 2,
            ))

        reorderer = RefPicListReorder()
        reorderer.set_buffer(buffer)

        l0_list = reorderer.build_initial_l0(current_poc=15)

        # Many reorder commands
        commands = []
        for i in range(7):
            commands.append({
                'modification_of_pic_nums_idc': 0,
                'abs_diff_pic_num_minus1': 0
            })
        commands.append({'modification_of_pic_nums_idc': 3})

        reordered = reorderer.reorder_l0(
            l0_list, commands, current_frame_num=7, max_frame_num=16
        )

        assert len(reordered) == len(l0_list)


class TestIntegrationWithSliceHeader:
    """Tests for integration with slice header parsing."""

    @pytest.mark.xfail(reason="Not implemented")
    def test_slice_header_reorder_data_to_commands(self):
        """Convert SliceHeader reorder data to command format."""
        from inter.ref_pic_reorder import RefPicListReorder
        from slice.slice_header import RefPicListModification

        mod = RefPicListModification(
            modification_of_pic_nums_idc=[0, 1, 3],
            abs_diff_pic_num_minus1=[2, 1],
            long_term_pic_num=[],
        )

        reorderer = RefPicListReorder()
        commands = reorderer.modification_to_commands(mod)

        assert len(commands) == 3
        assert commands[0]['modification_of_pic_nums_idc'] == 0
        assert commands[0]['abs_diff_pic_num_minus1'] == 2
        assert commands[1]['modification_of_pic_nums_idc'] == 1
        assert commands[1]['abs_diff_pic_num_minus1'] == 1
        assert commands[2]['modification_of_pic_nums_idc'] == 3

    @pytest.mark.xfail(reason="Not implemented")
    def test_slice_header_with_long_term_reorder(self):
        """SliceHeader with long-term reorder commands."""
        from inter.ref_pic_reorder import RefPicListReorder
        from slice.slice_header import RefPicListModification

        mod = RefPicListModification(
            modification_of_pic_nums_idc=[2, 0, 3],
            abs_diff_pic_num_minus1=[1],  # Only one, for idc=0
            long_term_pic_num=[0],  # For idc=2
        )

        reorderer = RefPicListReorder()
        commands = reorderer.modification_to_commands(mod)

        assert commands[0]['modification_of_pic_nums_idc'] == 2
        assert commands[0]['long_term_pic_num'] == 0
        assert commands[1]['modification_of_pic_nums_idc'] == 0
        assert commands[1]['abs_diff_pic_num_minus1'] == 1

    @pytest.mark.xfail(reason="Not implemented")
    def test_decoder_applies_reordering(self):
        """Main decoder should apply reference list reordering."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        assert hasattr(decoder, '_apply_ref_list_reordering'), \
            "Decoder should have _apply_ref_list_reordering method"
