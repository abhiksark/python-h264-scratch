# h264/slice/tests/test_aso.py
"""RED TDD tests for H.264 Arbitrary Slice Order (ASO).

ASO allows slices within a frame to arrive and be processed in any order.
This is useful for error resilience and parallel encoding/decoding.

H.264 Spec References:
- Section 7.3.3: Slice header syntax (first_mb_in_slice)
- Section 7.4.3: Slice header semantics
- Section 8.2.2: Slice decoding process
- Annex A: Profiles (ASO allowed in Extended profile, some Main)

Key ASO concepts:
1. first_mb_in_slice identifies where slice starts in raster scan
2. Slices can arrive in any order (not necessarily MB address order)
3. Frame assembly requires tracking which MBs belong to which slice
4. redundant_pic_cnt allows redundant slices for same MBs

These tests SHOULD FAIL until ASO support is implemented.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from slice.slice_header import SliceHeader, parse_slice_header


# -----------------------------------------------------------------------------
# Test fixtures and helpers
# -----------------------------------------------------------------------------

@dataclass
class MockSlice:
    """Mock slice data for testing ASO."""
    first_mb_in_slice: int
    slice_type: int
    frame_num: int
    redundant_pic_cnt: int = 0
    slice_qp_delta: int = 0
    mb_count: int = 1  # Number of MBs in this slice


def create_mock_slice_header(
    first_mb: int,
    slice_type: int = 2,  # I slice
    frame_num: int = 0,
    redundant_pic_cnt: int = 0,
    slice_qp_delta: int = 0,
) -> SliceHeader:
    """Create a mock SliceHeader for testing."""
    return SliceHeader(
        first_mb_in_slice=first_mb,
        slice_type=slice_type,
        frame_num=frame_num,
        redundant_pic_cnt=redundant_pic_cnt,
        slice_qp_delta=slice_qp_delta,
    )


# -----------------------------------------------------------------------------
# SliceOrderDetector - Detects if ASO is being used
# -----------------------------------------------------------------------------

class TestSliceOrderingDetection:
    """Tests for detecting slice ordering patterns.

    H.264 Spec: Section 7.4.3
    first_mb_in_slice specifies address of first macroblock in slice.
    If slices arrive out of order, decoder must detect and handle this.
    """

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_detect_ascending_order(self):
        """Slices arriving in ascending first_mb order are in-order."""
        from slice.aso import SliceOrderDetector

        detector = SliceOrderDetector()

        # Slices 0, 10, 20 - ascending order
        slices = [
            create_mock_slice_header(first_mb=0),
            create_mock_slice_header(first_mb=10),
            create_mock_slice_header(first_mb=20),
        ]

        for s in slices:
            detector.add_slice(s)

        assert detector.is_in_order() is True
        assert detector.uses_aso() is False

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_detect_out_of_order_slices(self):
        """Slices arriving out of first_mb order trigger ASO detection."""
        from slice.aso import SliceOrderDetector

        detector = SliceOrderDetector()

        # Slices arrive as 20, 0, 10 - out of order
        slices = [
            create_mock_slice_header(first_mb=20),
            create_mock_slice_header(first_mb=0),
            create_mock_slice_header(first_mb=10),
        ]

        for s in slices:
            detector.add_slice(s)

        assert detector.is_in_order() is False
        assert detector.uses_aso() is True

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_detect_interleaved_slices(self):
        """Detect interleaved slice pattern (common ASO pattern)."""
        from slice.aso import SliceOrderDetector

        detector = SliceOrderDetector()

        # Odd slices first, then even - interleaved pattern
        slices = [
            create_mock_slice_header(first_mb=10),  # MB 10
            create_mock_slice_header(first_mb=30),  # MB 30
            create_mock_slice_header(first_mb=0),   # MB 0
            create_mock_slice_header(first_mb=20),  # MB 20
        ]

        for s in slices:
            detector.add_slice(s)

        assert detector.uses_aso() is True
        assert detector.get_order_pattern() == "interleaved"

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_detect_reverse_order_slices(self):
        """Detect reverse order (last slice first)."""
        from slice.aso import SliceOrderDetector

        detector = SliceOrderDetector()

        # Reverse order: 30, 20, 10, 0
        slices = [
            create_mock_slice_header(first_mb=30),
            create_mock_slice_header(first_mb=20),
            create_mock_slice_header(first_mb=10),
            create_mock_slice_header(first_mb=0),
        ]

        for s in slices:
            detector.add_slice(s)

        assert detector.uses_aso() is True
        assert detector.get_order_pattern() == "reverse"

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_single_slice_is_in_order(self):
        """Single slice frame is always in-order."""
        from slice.aso import SliceOrderDetector

        detector = SliceOrderDetector()
        detector.add_slice(create_mock_slice_header(first_mb=0))

        assert detector.is_in_order() is True
        assert detector.uses_aso() is False

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_reset_detector_for_new_frame(self):
        """Detector should reset for new frame."""
        from slice.aso import SliceOrderDetector

        detector = SliceOrderDetector()
        detector.add_slice(create_mock_slice_header(first_mb=10, frame_num=0))
        detector.add_slice(create_mock_slice_header(first_mb=0, frame_num=0))

        assert detector.uses_aso() is True

        # New frame should reset
        detector.reset()
        detector.add_slice(create_mock_slice_header(first_mb=0, frame_num=1))
        detector.add_slice(create_mock_slice_header(first_mb=10, frame_num=1))

        assert detector.uses_aso() is False


# -----------------------------------------------------------------------------
# FrameAssembler - Assembles frame from out-of-order slices
# -----------------------------------------------------------------------------

class TestFrameAssemblyOutOfOrder:
    """Tests for assembling frames from out-of-order slices.

    H.264 Spec: Section 8.2.2
    Slices can be decoded independently. Frame assembly must handle
    slices arriving in any order.
    """

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_frame_assembler_exists(self):
        """FrameAssembler class should exist for ASO support."""
        from decoder.frame import FrameAssembler

        assembler = FrameAssembler(width_mbs=4, height_mbs=3)
        assert assembler is not None

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_assemble_slices_out_of_order(self):
        """Assemble frame from out-of-order slices."""
        from decoder.frame import FrameAssembler

        # 4x3 MB frame = 12 MBs total
        assembler = FrameAssembler(width_mbs=4, height_mbs=3)

        # Create mock decoded MB data
        mb_data_slice1 = np.ones((16, 64), dtype=np.uint8) * 100  # MBs 4-7
        mb_data_slice0 = np.ones((16, 64), dtype=np.uint8) * 50   # MBs 0-3
        mb_data_slice2 = np.ones((16, 64), dtype=np.uint8) * 150  # MBs 8-11

        # Add slices out of order
        assembler.add_slice(first_mb=4, luma_data=mb_data_slice1)
        assembler.add_slice(first_mb=0, luma_data=mb_data_slice0)
        assembler.add_slice(first_mb=8, luma_data=mb_data_slice2)

        # Assemble frame
        frame_luma = assembler.assemble_luma()

        # Verify correct assembly
        assert frame_luma[0, 0] == 50     # First row (slice 0)
        assert frame_luma[16, 0] == 100   # Second row (slice 1)
        assert frame_luma[32, 0] == 150   # Third row (slice 2)

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_assemble_preserves_mb_boundaries(self):
        """MB boundaries should be preserved during assembly."""
        from decoder.frame import FrameAssembler

        assembler = FrameAssembler(width_mbs=2, height_mbs=2)

        # Each slice has unique pattern per MB
        slice1_data = np.zeros((16, 32), dtype=np.uint8)
        slice1_data[:, :16] = 111  # MB 2
        slice1_data[:, 16:] = 112  # MB 3

        slice0_data = np.zeros((16, 32), dtype=np.uint8)
        slice0_data[:, :16] = 101  # MB 0
        slice0_data[:, 16:] = 102  # MB 1

        assembler.add_slice(first_mb=2, luma_data=slice1_data)
        assembler.add_slice(first_mb=0, luma_data=slice0_data)

        frame = assembler.assemble_luma()

        # Check MB corners
        assert frame[0, 0] == 101      # MB 0
        assert frame[0, 16] == 102     # MB 1
        assert frame[16, 0] == 111     # MB 2
        assert frame[16, 16] == 112    # MB 3

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_assemble_chroma_planes(self):
        """Chroma planes should also be assembled correctly."""
        from decoder.frame import FrameAssembler

        assembler = FrameAssembler(width_mbs=2, height_mbs=2)

        cb_slice1 = np.ones((8, 16), dtype=np.uint8) * 128
        cr_slice1 = np.ones((8, 16), dtype=np.uint8) * 130
        cb_slice0 = np.ones((8, 16), dtype=np.uint8) * 120
        cr_slice0 = np.ones((8, 16), dtype=np.uint8) * 122

        assembler.add_slice(first_mb=2, cb_data=cb_slice1, cr_data=cr_slice1)
        assembler.add_slice(first_mb=0, cb_data=cb_slice0, cr_data=cr_slice0)

        frame_cb = assembler.assemble_cb()
        frame_cr = assembler.assemble_cr()

        assert frame_cb[0, 0] == 120   # Slice 0
        assert frame_cb[8, 0] == 128   # Slice 1
        assert frame_cr[0, 0] == 122
        assert frame_cr[8, 0] == 130

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_assembly_completion_check(self):
        """Should detect when frame assembly is complete."""
        from decoder.frame import FrameAssembler

        assembler = FrameAssembler(width_mbs=2, height_mbs=2)

        assert assembler.is_complete() is False

        # Add partial data
        assembler.add_slice(first_mb=0, luma_data=np.zeros((16, 32), dtype=np.uint8))
        assert assembler.is_complete() is False

        # Complete the frame
        assembler.add_slice(first_mb=2, luma_data=np.zeros((16, 32), dtype=np.uint8))
        assert assembler.is_complete() is True


# -----------------------------------------------------------------------------
# first_mb_in_slice validation
# -----------------------------------------------------------------------------

class TestFirstMbInSliceValidation:
    """Tests for first_mb_in_slice validation.

    H.264 Spec: Section 7.4.3
    first_mb_in_slice shall be in range [0, PicSizeInMbs - 1].
    """

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_valid_first_mb_at_start(self):
        """first_mb_in_slice=0 is always valid."""
        from slice.aso import validate_first_mb_in_slice

        total_mbs = 12  # 4x3 frame
        assert validate_first_mb_in_slice(0, total_mbs) is True

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_valid_first_mb_at_end(self):
        """first_mb_in_slice at last MB is valid."""
        from slice.aso import validate_first_mb_in_slice

        total_mbs = 12
        assert validate_first_mb_in_slice(11, total_mbs) is True

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_invalid_first_mb_out_of_range(self):
        """first_mb_in_slice >= total_mbs is invalid."""
        from slice.aso import validate_first_mb_in_slice

        total_mbs = 12
        assert validate_first_mb_in_slice(12, total_mbs) is False
        assert validate_first_mb_in_slice(100, total_mbs) is False

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_invalid_negative_first_mb(self):
        """Negative first_mb_in_slice is invalid."""
        from slice.aso import validate_first_mb_in_slice

        total_mbs = 12
        assert validate_first_mb_in_slice(-1, total_mbs) is False

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_first_mb_gap_detection(self):
        """Detect gaps between slices."""
        from slice.aso import SliceValidator

        validator = SliceValidator(total_mbs=12)

        # First slice covers MBs 0-3
        validator.add_slice(first_mb=0, mb_count=4)
        # Gap: MBs 4-5 missing
        # Next slice starts at MB 6
        validator.add_slice(first_mb=6, mb_count=6)

        assert validator.has_gaps() is True
        assert validator.get_missing_mbs() == [4, 5]

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_first_mb_no_gaps(self):
        """No gaps when slices cover all MBs."""
        from slice.aso import SliceValidator

        validator = SliceValidator(total_mbs=12)

        validator.add_slice(first_mb=0, mb_count=4)
        validator.add_slice(first_mb=4, mb_count=4)
        validator.add_slice(first_mb=8, mb_count=4)

        assert validator.has_gaps() is False
        assert validator.get_missing_mbs() == []


# -----------------------------------------------------------------------------
# redundant_pic_cnt handling
# -----------------------------------------------------------------------------

class TestRedundantPicCntHandling:
    """Tests for redundant_pic_cnt slice handling.

    H.264 Spec: Section 7.4.3
    redundant_pic_cnt identifies redundant slices (same MBs, different encoding).
    Primary slices have redundant_pic_cnt=0, redundant slices have >0.
    """

    def test_redundant_pic_cnt_in_slice_header(self):
        """SliceHeader should have redundant_pic_cnt field.

        Note: This field already exists in SliceHeader, but ASO logic
        to use it for redundant slice selection is not implemented.
        """
        header = SliceHeader()
        assert hasattr(header, 'redundant_pic_cnt')
        assert header.redundant_pic_cnt == 0  # Default is primary

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_identify_primary_slice(self):
        """Identify primary slice (redundant_pic_cnt=0)."""
        from slice.aso import is_primary_slice

        primary = create_mock_slice_header(first_mb=0, redundant_pic_cnt=0)
        assert is_primary_slice(primary) is True

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_identify_redundant_slice(self):
        """Identify redundant slice (redundant_pic_cnt>0)."""
        from slice.aso import is_primary_slice, is_redundant_slice

        redundant = create_mock_slice_header(first_mb=0, redundant_pic_cnt=1)
        assert is_primary_slice(redundant) is False
        assert is_redundant_slice(redundant) is True

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_prefer_primary_over_redundant(self):
        """Primary slice should be preferred over redundant."""
        from slice.aso import SliceSelector

        selector = SliceSelector()

        # Add redundant slice first
        redundant = create_mock_slice_header(first_mb=0, redundant_pic_cnt=1)
        selector.add_slice(redundant, data=b'redundant_data')

        # Then add primary slice for same MBs
        primary = create_mock_slice_header(first_mb=0, redundant_pic_cnt=0)
        selector.add_slice(primary, data=b'primary_data')

        # Should select primary
        selected = selector.get_slice_for_mb(0)
        assert selected.redundant_pic_cnt == 0

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_use_redundant_when_primary_missing(self):
        """Use redundant slice when primary is missing/corrupted."""
        from slice.aso import SliceSelector

        selector = SliceSelector()

        # Only add redundant slice
        redundant = create_mock_slice_header(first_mb=0, redundant_pic_cnt=1)
        selector.add_slice(redundant, data=b'redundant_data')

        # Should use redundant since no primary available
        selected = selector.get_slice_for_mb(0)
        assert selected.redundant_pic_cnt == 1

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_multiple_redundant_levels(self):
        """Handle multiple levels of redundancy."""
        from slice.aso import SliceSelector

        selector = SliceSelector()

        # Add slices with increasing redundancy levels
        selector.add_slice(
            create_mock_slice_header(first_mb=0, redundant_pic_cnt=2),
            data=b'level2'
        )
        selector.add_slice(
            create_mock_slice_header(first_mb=0, redundant_pic_cnt=1),
            data=b'level1'
        )

        # Should prefer lower redundant_pic_cnt
        selected = selector.get_slice_for_mb(0)
        assert selected.redundant_pic_cnt == 1


# -----------------------------------------------------------------------------
# Slice-to-macroblock mapping
# -----------------------------------------------------------------------------

class TestSliceToMacroblockMapping:
    """Tests for mapping between slices and macroblocks.

    H.264 Spec: Section 8.2.2
    Each slice contains a contiguous range of macroblocks in raster scan order
    (unless FMO is used).
    """

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_slice_mb_mapping_exists(self):
        """SliceMBMapper class should exist."""
        from slice.aso import SliceMBMapper

        mapper = SliceMBMapper(width_mbs=4, height_mbs=3)
        assert mapper is not None

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_map_single_slice_whole_frame(self):
        """Single slice covering whole frame."""
        from slice.aso import SliceMBMapper

        mapper = SliceMBMapper(width_mbs=4, height_mbs=3)  # 12 MBs

        header = create_mock_slice_header(first_mb=0)
        mapper.add_slice(header, mb_count=12)

        # All MBs should map to slice 0
        for mb_idx in range(12):
            assert mapper.get_slice_id(mb_idx) == 0

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_map_multiple_slices(self):
        """Multiple slices covering frame."""
        from slice.aso import SliceMBMapper

        mapper = SliceMBMapper(width_mbs=4, height_mbs=3)

        # Slice 0: MBs 0-3
        mapper.add_slice(create_mock_slice_header(first_mb=0), mb_count=4)
        # Slice 1: MBs 4-7
        mapper.add_slice(create_mock_slice_header(first_mb=4), mb_count=4)
        # Slice 2: MBs 8-11
        mapper.add_slice(create_mock_slice_header(first_mb=8), mb_count=4)

        assert mapper.get_slice_id(0) == 0
        assert mapper.get_slice_id(3) == 0
        assert mapper.get_slice_id(4) == 1
        assert mapper.get_slice_id(7) == 1
        assert mapper.get_slice_id(8) == 2
        assert mapper.get_slice_id(11) == 2

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_get_mbs_for_slice(self):
        """Get list of MBs belonging to a slice."""
        from slice.aso import SliceMBMapper

        mapper = SliceMBMapper(width_mbs=4, height_mbs=3)

        mapper.add_slice(create_mock_slice_header(first_mb=0), mb_count=4)
        mapper.add_slice(create_mock_slice_header(first_mb=4), mb_count=8)

        mbs_slice0 = mapper.get_mbs_for_slice(0)
        mbs_slice1 = mapper.get_mbs_for_slice(1)

        assert mbs_slice0 == [0, 1, 2, 3]
        assert mbs_slice1 == [4, 5, 6, 7, 8, 9, 10, 11]

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_mb_to_position_conversion(self):
        """Convert MB index to (x, y) position."""
        from slice.aso import SliceMBMapper

        mapper = SliceMBMapper(width_mbs=4, height_mbs=3)

        assert mapper.mb_index_to_position(0) == (0, 0)
        assert mapper.mb_index_to_position(3) == (3, 0)
        assert mapper.mb_index_to_position(4) == (0, 1)
        assert mapper.mb_index_to_position(11) == (3, 2)

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_position_to_mb_conversion(self):
        """Convert (x, y) position to MB index."""
        from slice.aso import SliceMBMapper

        mapper = SliceMBMapper(width_mbs=4, height_mbs=3)

        assert mapper.position_to_mb_index(0, 0) == 0
        assert mapper.position_to_mb_index(3, 0) == 3
        assert mapper.position_to_mb_index(0, 1) == 4
        assert mapper.position_to_mb_index(3, 2) == 11


# -----------------------------------------------------------------------------
# Partial frame handling
# -----------------------------------------------------------------------------

class TestPartialFrameHandling:
    """Tests for handling frames with missing slices.

    When ASO is used, some slices may be lost due to network errors.
    Decoder should handle partial frames gracefully.
    """

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_detect_incomplete_frame(self):
        """Detect when frame is missing slices."""
        from slice.aso import FrameCompletionTracker

        tracker = FrameCompletionTracker(total_mbs=12)

        # Add slice covering only first 4 MBs
        tracker.mark_decoded(first_mb=0, mb_count=4)

        assert tracker.is_complete() is False
        assert tracker.get_decoded_mb_count() == 4
        assert tracker.get_completion_percentage() == pytest.approx(33.33, rel=0.1)

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_partial_frame_concealment_strategy(self):
        """Get concealment strategy for missing MBs."""
        from slice.aso import FrameCompletionTracker, ConcealmentStrategy

        tracker = FrameCompletionTracker(total_mbs=12)
        tracker.mark_decoded(first_mb=0, mb_count=4)
        tracker.mark_decoded(first_mb=8, mb_count=4)
        # MBs 4-7 missing

        missing = tracker.get_missing_mbs()
        assert missing == [4, 5, 6, 7]

        # Should suggest concealment strategy
        strategy = tracker.get_concealment_strategy()
        assert strategy in [
            ConcealmentStrategy.COPY_FROM_ABOVE,
            ConcealmentStrategy.COPY_FROM_LEFT,
            ConcealmentStrategy.INTERPOLATE,
            ConcealmentStrategy.SKIP,
        ]

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_timeout_waiting_for_slices(self):
        """Handle timeout when waiting for missing slices."""
        from slice.aso import FrameCompletionTracker

        tracker = FrameCompletionTracker(total_mbs=12, timeout_ms=100)
        tracker.mark_decoded(first_mb=0, mb_count=6)

        # Simulate waiting
        import time
        time.sleep(0.15)  # 150ms > timeout

        assert tracker.has_timed_out() is True
        assert tracker.should_output_partial() is True

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_progressive_frame_output(self):
        """Output frame progressively as slices arrive."""
        from slice.aso import ProgressiveFrameAssembler

        assembler = ProgressiveFrameAssembler(width_mbs=4, height_mbs=3)

        # Add first slice
        assembler.add_slice(first_mb=0, luma_data=np.ones((16, 64), dtype=np.uint8) * 100)

        # Should be able to get partial frame
        partial = assembler.get_current_frame()
        assert partial is not None
        assert partial[0, 0] == 100

        # Uncoded areas should be marked
        assert partial[16, 0] == 0  # Or some placeholder value


# -----------------------------------------------------------------------------
# Slice boundary detection
# -----------------------------------------------------------------------------

class TestSliceBoundaryDetection:
    """Tests for detecting slice boundaries.

    H.264 Spec: Section 8.7.2
    Deblocking filter may not cross slice boundaries (depending on idc).
    """

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_horizontal_slice_boundary(self):
        """Detect horizontal slice boundary between MBs."""
        from slice.aso import SliceBoundaryDetector

        detector = SliceBoundaryDetector(width_mbs=4, height_mbs=3)

        # Slice 0: row 0, Slice 1: rows 1-2
        detector.add_slice(first_mb=0, mb_count=4)   # MBs 0-3
        detector.add_slice(first_mb=4, mb_count=8)   # MBs 4-11

        # Boundary between MB 3 (row 0) and MB 4 (row 1)
        assert detector.is_slice_boundary(mb_a=3, mb_b=7) is True
        # No boundary within same slice
        assert detector.is_slice_boundary(mb_a=0, mb_b=1) is False

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_vertical_slice_boundary(self):
        """Detect vertical slice boundary between MBs."""
        from slice.aso import SliceBoundaryDetector

        detector = SliceBoundaryDetector(width_mbs=4, height_mbs=3)

        # Vertical split: left 2 columns, right 2 columns
        detector.add_slice(first_mb=0, mb_count=2)   # MBs 0-1
        detector.add_slice(first_mb=2, mb_count=2)   # MBs 2-3
        detector.add_slice(first_mb=4, mb_count=2)   # MBs 4-5
        detector.add_slice(first_mb=6, mb_count=2)   # etc.

        # Boundary between MB 1 and MB 2
        assert detector.is_slice_boundary(mb_a=1, mb_b=2) is True

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_get_boundary_edges(self):
        """Get all edges that are slice boundaries."""
        from slice.aso import SliceBoundaryDetector

        detector = SliceBoundaryDetector(width_mbs=4, height_mbs=2)

        detector.add_slice(first_mb=0, mb_count=4)  # Row 0
        detector.add_slice(first_mb=4, mb_count=4)  # Row 1

        boundaries = detector.get_all_boundary_edges()

        # Should have 4 horizontal boundary edges (between rows 0 and 1)
        horizontal_boundaries = [
            (mb_a, mb_b) for mb_a, mb_b in boundaries
            if mb_b == mb_a + 4  # Below
        ]
        assert len(horizontal_boundaries) == 4

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_deblocking_respects_boundaries(self):
        """Deblocking filter respects slice boundaries when idc=2."""
        from slice.aso import SliceBoundaryDetector
        from deblock.deblock import should_deblock_edge

        detector = SliceBoundaryDetector(width_mbs=4, height_mbs=2)
        detector.add_slice(first_mb=0, mb_count=4)
        detector.add_slice(first_mb=4, mb_count=4)

        # disable_deblocking_filter_idc = 2 means no deblock at slice boundaries
        disable_idc = 2

        # Edge within slice should be deblocked
        assert should_deblock_edge(
            mb_a=0, mb_b=1,
            boundary_detector=detector,
            disable_idc=disable_idc
        ) is True

        # Edge at slice boundary should NOT be deblocked
        assert should_deblock_edge(
            mb_a=3, mb_b=7,
            boundary_detector=detector,
            disable_idc=disable_idc
        ) is False


# -----------------------------------------------------------------------------
# Multiple slices covering same MBs (redundant slices)
# -----------------------------------------------------------------------------

class TestRedundantSlicesSameMBs:
    """Tests for handling multiple slices covering same macroblocks.

    H.264 Spec: Section 7.4.3
    With redundant_pic_cnt, multiple slices can cover same MBs.
    Primary (cnt=0) is preferred; redundant (cnt>0) used for error recovery.
    """

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_detect_overlapping_slices(self):
        """Detect when slices overlap (cover same MBs)."""
        from slice.aso import SliceOverlapDetector

        detector = SliceOverlapDetector(total_mbs=12)

        # Primary slice for MBs 0-5
        detector.add_slice(
            create_mock_slice_header(first_mb=0, redundant_pic_cnt=0),
            mb_count=6
        )
        # Redundant slice for MBs 0-5
        detector.add_slice(
            create_mock_slice_header(first_mb=0, redundant_pic_cnt=1),
            mb_count=6
        )

        assert detector.has_overlapping_slices() is True
        assert detector.get_overlapping_mbs() == list(range(6))

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_overlapping_non_redundant_is_error(self):
        """Overlapping non-redundant slices is a bitstream error."""
        from slice.aso import SliceOverlapDetector, BitstreamError

        detector = SliceOverlapDetector(total_mbs=12)

        # Two primary slices covering same MBs - error
        detector.add_slice(
            create_mock_slice_header(first_mb=0, redundant_pic_cnt=0),
            mb_count=6
        )

        with pytest.raises(BitstreamError):
            detector.add_slice(
                create_mock_slice_header(first_mb=0, redundant_pic_cnt=0),
                mb_count=6
            )

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_partial_overlap_handling(self):
        """Handle partial overlap between redundant and primary."""
        from slice.aso import SliceOverlapDetector

        detector = SliceOverlapDetector(total_mbs=12)

        # Primary covers MBs 0-3
        detector.add_slice(
            create_mock_slice_header(first_mb=0, redundant_pic_cnt=0),
            mb_count=4
        )
        # Redundant covers MBs 2-7 (overlaps 2-3)
        detector.add_slice(
            create_mock_slice_header(first_mb=2, redundant_pic_cnt=1),
            mb_count=6
        )

        overlap = detector.get_overlapping_mbs()
        assert overlap == [2, 3]

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_select_best_quality_slice(self):
        """Select slice with best quality for each MB."""
        from slice.aso import SliceQualitySelector

        selector = SliceQualitySelector(total_mbs=12)

        # Primary with higher QP (lower quality)
        primary = create_mock_slice_header(first_mb=0, redundant_pic_cnt=0, slice_qp_delta=10)
        # Redundant with lower QP (higher quality)
        redundant = create_mock_slice_header(first_mb=0, redundant_pic_cnt=1, slice_qp_delta=-5)

        selector.add_slice(primary, mb_count=4)
        selector.add_slice(redundant, mb_count=4)

        # By default, prefer primary regardless of quality
        selected = selector.select_slice(mb_idx=0, prefer_quality=False)
        assert selected.redundant_pic_cnt == 0

        # With quality preference, might select redundant
        selected = selector.select_slice(mb_idx=0, prefer_quality=True)
        # Implementation might still prefer primary (spec compliant)
        # or could prefer lower QP (quality mode)


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------

class TestASOIntegration:
    """Integration tests for ASO with decoder."""

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_decoder_enables_aso_mode(self):
        """Decoder should have ASO mode toggle."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder(aso_enabled=True)
        assert decoder.aso_enabled is True

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_decoder_auto_detects_aso(self):
        """Decoder should auto-detect ASO when slices arrive out of order."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Simulate receiving out-of-order slices
        # (Would need actual bitstream data in real test)
        assert hasattr(decoder, '_detect_aso_mode')

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_aso_with_fmo(self):
        """ASO combined with FMO (Flexible Macroblock Ordering)."""
        from slice.aso import SliceMBMapper

        # FMO with 2 slice groups (checkerboard)
        mapper = SliceMBMapper(
            width_mbs=4,
            height_mbs=4,
            slice_group_map_type=1,  # Dispersed
            num_slice_groups=2
        )

        # With FMO, MB assignment is more complex
        # Dispersed pattern: alternating assignment
        assert mapper.get_slice_group(mb_idx=0) == 0
        assert mapper.get_slice_group(mb_idx=1) == 1  # or similar pattern

    @pytest.mark.xfail(reason="ASO not implemented")
    def test_aso_performance_tracking(self):
        """Track performance impact of ASO."""
        from slice.aso import ASOPerformanceTracker

        tracker = ASOPerformanceTracker()

        # Simulate decoding with ASO
        tracker.start_frame()
        tracker.slice_received(first_mb=10)
        tracker.slice_received(first_mb=0)
        tracker.slice_received(first_mb=20)
        tracker.end_frame()

        stats = tracker.get_stats()
        assert 'reorder_count' in stats
        assert stats['reorder_count'] == 2  # 2 reorderings needed
