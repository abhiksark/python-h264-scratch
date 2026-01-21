# h264/decoder/tests/test_poc.py
"""RED TESTS: Picture Order Count (POC) calculation.

POC is used for:
- Display ordering (B-frames are decoded out of display order)
- Reference picture list construction (L0/L1 ordering by POC)
- Temporal direct mode MV derivation

H.264 Spec Reference: Section 8.2.1 - Decoding process for picture order count

POC Types:
- Type 0: Uses pic_order_cnt_lsb in slice header with MSB wraparound
- Type 1: Uses delta_pic_order_cnt with reference frame cycle
- Type 2: Derives POC directly from frame_num (no B-frames)

These tests SHOULD FAIL until proper POC calculation is implemented.
"""

import pytest
import numpy as np

from parameters.sps import SPS
from slice.slice_header import SliceHeader


class TestPOCCalculatorExists:
    """Tests for POC calculator module existence."""

    def test_poc_module_exists(self):
        """POC calculator module should exist."""
        try:
            from decoder import poc
            assert poc is not None
        except ImportError:
            pytest.fail("decoder.poc module should exist")

    def test_poc_calculator_class_exists(self):
        """POCCalculator class should exist."""
        try:
            from decoder.poc import POCCalculator
            assert POCCalculator is not None
        except ImportError:
            pytest.fail("POCCalculator class should exist in decoder.poc")

    def test_poc_calculator_has_calculate_method(self):
        """POCCalculator should have calculate() method."""
        from decoder.poc import POCCalculator

        calc = POCCalculator()
        assert hasattr(calc, 'calculate'), \
            "POCCalculator should have calculate() method"


class TestPOCType0Calculation:
    """Tests for POC type 0 (pic_order_cnt_lsb based).

    POC type 0 uses:
    - pic_order_cnt_lsb from slice header (explicit LSB)
    - Derived MSB based on previous frame's POC values

    H.264 Spec: Section 8.2.1.1
    """

    def test_poc_type_0_first_frame(self):
        """First frame (IDR) should have POC=0."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,  # MaxPicOrderCntLsb = 16
        )
        header = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,  # I-slice
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=0,
        )

        calc = POCCalculator()
        poc = calc.calculate(sps, header, is_idr=True)

        assert poc == 0, "First IDR frame should have POC=0"

    def test_poc_type_0_sequential_frames(self):
        """Sequential frames should have incrementing POC."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,  # MaxPicOrderCntLsb = 16
        )

        calc = POCCalculator()

        # Frame 0: IDR, POC=0
        header0 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=0,
        )
        poc0 = calc.calculate(sps, header0, is_idr=True)

        # Frame 1: P, POC=2
        header1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=2,
        )
        poc1 = calc.calculate(sps, header1, is_idr=False)

        # Frame 2: P, POC=4
        header2 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=2,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=4,
        )
        poc2 = calc.calculate(sps, header2, is_idr=False)

        assert poc0 == 0
        assert poc1 == 2
        assert poc2 == 4

    def test_poc_type_0_b_frame_order(self):
        """B-frames have POC between their reference frames.

        Decode order: I0, P2, B1, P4, B3
        Display order (POC): 0, 1, 2, 3, 4
        """
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,  # MaxPicOrderCntLsb = 16
        )

        calc = POCCalculator()

        # I-frame (decode 0, display 0)
        h_i = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=0,
        )
        poc_i = calc.calculate(sps, h_i, is_idr=True)

        # P-frame (decode 1, display 2)
        h_p1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=2,
        )
        poc_p1 = calc.calculate(sps, h_p1, is_idr=False)

        # B-frame (decode 2, display 1)
        h_b1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=1,
            pic_parameter_set_id=0,
            frame_num=2,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=1,
        )
        poc_b1 = calc.calculate(sps, h_b1, is_idr=False)

        assert poc_i == 0, "I-frame POC should be 0"
        assert poc_p1 == 2, "P-frame POC should be 2"
        assert poc_b1 == 1, "B-frame POC should be 1 (between 0 and 2)"


class TestPOCType0Wraparound:
    """Tests for POC type 0 MSB wraparound handling.

    When pic_order_cnt_lsb wraps around, the MSB must be adjusted.

    H.264 Spec: Section 8.2.1.1 equations (8-3) through (8-6)
    """

    def test_poc_lsb_wraparound_forward(self):
        """POC MSB should increment when LSB wraps forward."""
        from decoder.poc import POCCalculator

        # MaxPicOrderCntLsb = 16 (4-bit LSB)
        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,
        )

        calc = POCCalculator()

        # Start at POC=14 (LSB=14, MSB=0)
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=14,
        )
        calc.calculate(sps, h1, is_idr=True)

        # Next frame POC=2 (LSB=2, MSB should become 16)
        # LSB jumps backward by 12 (> MaxPicOrderCntLsb/2=8)
        # So MSB increments: POC = 16 + 2 = 18
        h2 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=2,
        )
        poc2 = calc.calculate(sps, h2, is_idr=False)

        assert poc2 == 18, f"POC should wrap to 18, got {poc2}"

    def test_poc_lsb_wraparound_backward(self):
        """POC MSB should decrement when LSB wraps backward."""
        from decoder.poc import POCCalculator

        # MaxPicOrderCntLsb = 16
        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,
        )

        calc = POCCalculator()

        # Start with MSB=16, LSB=2 (POC=18)
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=2,
        )
        calc.calculate(sps, h1, is_idr=True)

        # If next LSB=14 (jumps forward by 12 > MaxPicOrderCntLsb/2=8)
        # MSB should decrement: POC = 0 + 14 = 14
        # But wait, after IDR we start fresh, so this is MSB=0
        # Let's test a different scenario - after establishing MSB=16
        h2 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=4,  # Normal increment
        )
        poc2 = calc.calculate(sps, h2, is_idr=False)

        # Now from LSB=4 (POC=4), if we see LSB=14 (jump of 10 > 8)
        # MSB should decrement to -16: POC = -16 + 14 = -2
        h3 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=2,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=14,
        )
        poc3 = calc.calculate(sps, h3, is_idr=False)

        # The spec says if (pic_order_cnt_lsb - prevPicOrderCntLsb) > MaxPicOrderCntLsb/2
        # then PicOrderCntMsb = prevPicOrderCntMsb - MaxPicOrderCntLsb
        assert poc3 == -2, f"POC should wrap to -2, got {poc3}"

    def test_large_poc_lsb_range(self):
        """Test with larger MaxPicOrderCntLsb value."""
        from decoder.poc import POCCalculator

        # MaxPicOrderCntLsb = 256 (8-bit LSB)
        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
        )

        calc = POCCalculator()

        # IDR at POC=200
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=200,
        )
        poc1 = calc.calculate(sps, h1, is_idr=True)

        # Next at LSB=10 (wraparound, jump back 190 > 128)
        h2 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=10,
        )
        poc2 = calc.calculate(sps, h2, is_idr=False)

        assert poc1 == 200
        assert poc2 == 256 + 10, f"POC should be 266, got {poc2}"


class TestPOCType1Calculation:
    """Tests for POC type 1 (delta_pic_order_cnt based).

    POC type 1 uses:
    - delta_pic_order_cnt from slice header
    - Reference frame cycle from SPS
    - Expected POC derived from frame_num and cycle

    H.264 Spec: Section 8.2.1.2
    """

    def test_poc_type_1_basic(self):
        """Basic POC type 1 calculation."""
        from decoder.poc import POCCalculator

        # POC type 1 with simple cycle
        sps = SPS(
            pic_order_cnt_type=1,
            delta_pic_order_always_zero_flag=False,
            offset_for_non_ref_pic=0,
            offset_for_top_to_bottom_field=0,
            num_ref_frames_in_pic_order_cnt_cycle=2,
            offset_for_ref_frame=[2, 2],  # ExpectedDeltaPerPicOrderCntCycle = 4
        )

        header = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            delta_pic_order_cnt=[0],
        )

        calc = POCCalculator()
        poc = calc.calculate(sps, header, is_idr=True)

        assert poc == 0, "IDR frame should have POC=0"

    def test_poc_type_1_with_delta(self):
        """POC type 1 with non-zero delta_pic_order_cnt."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=1,
            delta_pic_order_always_zero_flag=False,
            offset_for_non_ref_pic=0,
            offset_for_top_to_bottom_field=0,
            num_ref_frames_in_pic_order_cnt_cycle=1,
            offset_for_ref_frame=[2],
        )

        calc = POCCalculator()

        # IDR
        h0 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            delta_pic_order_cnt=[0],
        )
        poc0 = calc.calculate(sps, h0, is_idr=True)

        # Non-IDR with delta
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            delta_pic_order_cnt=[2],
        )
        poc1 = calc.calculate(sps, h1, is_idr=False)

        assert poc0 == 0
        # expectedPicOrderCnt = frameNumInPicOrderCntCycle * ExpectedDeltaPerPicOrderCntCycle
        #                     + sum(offset_for_ref_frame[0:absFrameNum%cycle])
        # For frame_num=1: expected = 2 + delta = 2 + 2 = 4
        assert poc1 == 4, f"POC should be 4, got {poc1}"

    def test_poc_type_1_delta_always_zero(self):
        """POC type 1 with delta_pic_order_always_zero_flag set."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=1,
            delta_pic_order_always_zero_flag=True,  # delta is always 0
            offset_for_non_ref_pic=0,
            offset_for_top_to_bottom_field=0,
            num_ref_frames_in_pic_order_cnt_cycle=1,
            offset_for_ref_frame=[2],
        )

        calc = POCCalculator()

        h0 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
        )
        poc0 = calc.calculate(sps, h0, is_idr=True)

        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
        )
        poc1 = calc.calculate(sps, h1, is_idr=False)

        assert poc0 == 0
        assert poc1 == 2, f"POC should be 2, got {poc1}"


class TestPOCType2Calculation:
    """Tests for POC type 2 (frame_num based).

    POC type 2 derives POC directly from frame_num:
    - Reference frames: POC = 2 * frame_num
    - Non-reference frames: POC = 2 * frame_num - 1

    H.264 Spec: Section 8.2.1.3
    """

    def test_poc_type_2_reference_frame(self):
        """Reference frames have POC = 2 * frame_num."""
        from decoder.poc import POCCalculator

        sps = SPS(pic_order_cnt_type=2)

        calc = POCCalculator()

        # IDR (reference)
        h0 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,  # I-slice (reference)
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
        )
        poc0 = calc.calculate(sps, h0, is_idr=True)

        # P-slice (reference)
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,  # P-slice (reference)
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
        )
        poc1 = calc.calculate(sps, h1, is_idr=False)

        h2 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=2,
            slice_qp_delta=0,
            header_bit_size=0,
        )
        poc2 = calc.calculate(sps, h2, is_idr=False)

        assert poc0 == 0, "frame_num=0 reference POC should be 0"
        assert poc1 == 2, "frame_num=1 reference POC should be 2"
        assert poc2 == 4, "frame_num=2 reference POC should be 4"

    def test_poc_type_2_non_reference_frame(self):
        """Non-reference frames have POC = 2 * frame_num - 1."""
        from decoder.poc import POCCalculator

        sps = SPS(pic_order_cnt_type=2)

        calc = POCCalculator()

        # Non-reference I-slice (nal_ref_idc=0)
        h = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
        )

        # POC type 2 needs nal_ref_idc info
        poc = calc.calculate(sps, h, is_idr=False, nal_ref_idc=0)

        assert poc == 1, "frame_num=1 non-reference POC should be 1"

    def test_poc_type_2_no_b_frames(self):
        """POC type 2 cannot have B-frames (decode order = display order)."""
        from decoder.poc import POCCalculator

        sps = SPS(pic_order_cnt_type=2)
        calc = POCCalculator()

        # Sequence of reference frames
        pocs = []
        for i in range(5):
            h = SliceHeader(
                first_mb_in_slice=0,
                slice_type=2 if i == 0 else 0,
                pic_parameter_set_id=0,
                frame_num=i,
                slice_qp_delta=0,
                header_bit_size=0,
                idr_pic_flag=(i == 0),
            )
            pocs.append(calc.calculate(sps, h, is_idr=(i == 0)))

        # POCs should be strictly increasing for reference frames
        assert pocs == [0, 2, 4, 6, 8], f"POCs should be [0,2,4,6,8], got {pocs}"


class TestPOCIDRReset:
    """Tests for POC reset on IDR frames."""

    def test_idr_resets_poc_state(self):
        """IDR frame should reset POC calculation state."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,
        )

        calc = POCCalculator()

        # Build up some state
        for i in range(10):
            h = SliceHeader(
                first_mb_in_slice=0,
                slice_type=2 if i == 0 else 0,
                pic_parameter_set_id=0,
                frame_num=i,
                slice_qp_delta=0,
                header_bit_size=0,
                idr_pic_flag=(i == 0),
                pic_order_cnt_lsb=i * 2 % 16,
            )
            calc.calculate(sps, h, is_idr=(i == 0))

        # New IDR should reset to 0
        h_idr = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=0,
        )
        poc = calc.calculate(sps, h_idr, is_idr=True)

        assert poc == 0, "IDR should reset POC to 0"

    def test_idr_resets_msb(self):
        """IDR should reset PicOrderCntMsb to 0."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,  # MaxPicOrderCntLsb = 16
        )

        calc = POCCalculator()

        # Build up MSB > 0
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=14,
        )
        calc.calculate(sps, h1, is_idr=True)

        # Wraparound - MSB becomes 16
        h2 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=2,
        )
        poc2 = calc.calculate(sps, h2, is_idr=False)
        assert poc2 == 18  # MSB=16, LSB=2

        # New IDR at LSB=4 should have POC=4, not 20
        h_idr = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=4,
        )
        poc_idr = calc.calculate(sps, h_idr, is_idr=True)

        assert poc_idr == 4, f"IDR should reset MSB, POC should be 4, got {poc_idr}"


class TestBFrameDisplayOrder:
    """Tests for B-frame display vs decode order."""

    def test_typical_ibbp_pattern(self):
        """Test typical IBBP GOP pattern.

        Decode order: I0, P3, B1, B2, P6, B4, B5, ...
        Display order: I0, B1, B2, P3, B4, B5, P6, ...
        """
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,  # MaxPicOrderCntLsb = 256
        )

        calc = POCCalculator()

        # Decode I (display 0)
        h_i = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=0,
        )
        poc_i = calc.calculate(sps, h_i, is_idr=True)

        # Decode P (display 3)
        h_p = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=6,  # POC=6 but display position 3
        )
        poc_p = calc.calculate(sps, h_p, is_idr=False)

        # Decode B1 (display 1)
        h_b1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=1,
            pic_parameter_set_id=0,
            frame_num=2,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=2,
        )
        poc_b1 = calc.calculate(sps, h_b1, is_idr=False)

        # Decode B2 (display 2)
        h_b2 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=1,
            pic_parameter_set_id=0,
            frame_num=3,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=4,
        )
        poc_b2 = calc.calculate(sps, h_b2, is_idr=False)

        assert poc_i == 0, "I-frame POC"
        assert poc_p == 6, "P-frame POC"
        assert poc_b1 == 2, "B1-frame POC"
        assert poc_b2 == 4, "B2-frame POC"

        # Verify display order by POC
        display_order = sorted(
            [('I', poc_i), ('P', poc_p), ('B1', poc_b1), ('B2', poc_b2)],
            key=lambda x: x[1]
        )
        expected = [('I', 0), ('B1', 2), ('B2', 4), ('P', 6)]
        assert display_order == expected

    def test_hierarchical_b_frames(self):
        """Test hierarchical B-frame structure.

        GOP with hierarchical B: I0, P8, B4, B2, B6, B1, B3, B5, B7
        """
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
        )

        calc = POCCalculator()

        frames = [
            (2, True, 0),   # IDR, poc=0
            (0, False, 16),  # P, poc=16
            (1, False, 8),   # B, poc=8
            (1, False, 4),   # B, poc=4
            (1, False, 12),  # B, poc=12
        ]

        pocs = []
        for i, (slice_type, is_idr, poc_lsb) in enumerate(frames):
            h = SliceHeader(
                first_mb_in_slice=0,
                slice_type=slice_type,
                pic_parameter_set_id=0,
                frame_num=i,
                slice_qp_delta=0,
                header_bit_size=0,
                idr_pic_flag=is_idr,
                pic_order_cnt_lsb=poc_lsb,
            )
            pocs.append(calc.calculate(sps, h, is_idr=is_idr))

        assert pocs == [0, 16, 8, 4, 12], f"Hierarchical B POCs: {pocs}"


class TestReferenceListOrderingByPOC:
    """Tests for reference list ordering based on POC."""

    def test_l0_list_poc_order(self):
        """L0 list should be ordered by POC descending (closest past first)."""
        from inter.reference import ReferenceFrame, ReferenceFrameBuffer

        buffer = ReferenceFrameBuffer(max_frames=5)

        # Add frames with POC 0, 2, 4, 6
        for poc in [0, 2, 4, 6]:
            frame = ReferenceFrame(
                luma=np.zeros((16, 16), dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=poc // 2,
                poc=poc,
            )
            buffer.add_frame(frame)

        # Current frame at POC=5
        buffer.build_ref_lists(current_poc=5)
        l0 = buffer.get_l0_list()

        # L0 should have POC 4, 2, 0 (past frames, closest first)
        l0_pocs = [f.poc for f in l0]
        assert l0_pocs == [4, 2, 0], f"L0 POCs should be [4,2,0], got {l0_pocs}"

    def test_l1_list_poc_order(self):
        """L1 list should be ordered by POC ascending (closest future first)."""
        from inter.reference import ReferenceFrame, ReferenceFrameBuffer

        buffer = ReferenceFrameBuffer(max_frames=5)

        # Add frames with POC 0, 4, 8, 12
        for poc in [0, 4, 8, 12]:
            frame = ReferenceFrame(
                luma=np.zeros((16, 16), dtype=np.uint8),
                cb=np.zeros((8, 8), dtype=np.uint8),
                cr=np.zeros((8, 8), dtype=np.uint8),
                frame_num=poc // 4,
                poc=poc,
            )
            buffer.add_frame(frame)

        # Current frame at POC=2
        buffer.build_ref_lists(current_poc=2)
        l1 = buffer.get_l1_list()

        # L1 should have POC 4, 8, 12 (future frames, closest first)
        l1_pocs = [f.poc for f in l1]
        assert l1_pocs == [4, 8, 12], f"L1 POCs should be [4,8,12], got {l1_pocs}"

    def test_ref_list_for_b_between_i_and_p(self):
        """B-frame between I and P should have correct L0/L1."""
        from inter.reference import ReferenceFrame, ReferenceFrameBuffer

        buffer = ReferenceFrameBuffer(max_frames=4)

        # I at POC=0
        i_frame = ReferenceFrame(
            luma=np.full((16, 16), 100, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=0,
            poc=0,
        )
        buffer.add_frame(i_frame)

        # P at POC=4
        p_frame = ReferenceFrame(
            luma=np.full((16, 16), 200, dtype=np.uint8),
            cb=np.zeros((8, 8), dtype=np.uint8),
            cr=np.zeros((8, 8), dtype=np.uint8),
            frame_num=1,
            poc=4,
        )
        buffer.add_frame(p_frame)

        # B at POC=2
        buffer.build_ref_lists(current_poc=2)

        l0 = buffer.get_l0_list()
        l1 = buffer.get_l1_list()

        # L0[0] should be I (POC=0), L1[0] should be P (POC=4)
        assert len(l0) == 1 and l0[0].poc == 0, "L0[0] should be I-frame (POC=0)"
        assert len(l1) == 1 and l1[0].poc == 4, "L1[0] should be P-frame (POC=4)"


class TestPOCStatePersistence:
    """Tests for POC calculator state management."""

    def test_calculator_tracks_prev_poc_lsb(self):
        """POCCalculator should track previous pic_order_cnt_lsb."""
        from decoder.poc import POCCalculator

        calc = POCCalculator()

        assert hasattr(calc, 'prev_poc_lsb') or hasattr(calc, '_prev_poc_lsb'), \
            "POCCalculator should track prev_poc_lsb"

    def test_calculator_tracks_prev_poc_msb(self):
        """POCCalculator should track previous PicOrderCntMsb."""
        from decoder.poc import POCCalculator

        calc = POCCalculator()

        assert hasattr(calc, 'prev_poc_msb') or hasattr(calc, '_prev_poc_msb'), \
            "POCCalculator should track prev_poc_msb"

    def test_calculator_has_reset_method(self):
        """POCCalculator should have reset() method for IDR handling."""
        from decoder.poc import POCCalculator

        calc = POCCalculator()

        assert hasattr(calc, 'reset'), \
            "POCCalculator should have reset() method"

    def test_reset_clears_state(self):
        """reset() should clear all POC state."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,
        )

        calc = POCCalculator()

        # Build up some state
        h = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=10,
        )
        calc.calculate(sps, h, is_idr=True)

        # Reset
        calc.reset()

        # State should be cleared
        prev_lsb = getattr(calc, 'prev_poc_lsb', getattr(calc, '_prev_poc_lsb', None))
        prev_msb = getattr(calc, 'prev_poc_msb', getattr(calc, '_prev_poc_msb', None))

        assert prev_lsb == 0, "reset() should clear prev_poc_lsb"
        assert prev_msb == 0, "reset() should clear prev_poc_msb"


class TestPOCEdgeCases:
    """Tests for edge cases in POC calculation."""

    def test_negative_poc_allowed(self):
        """POC can be negative (for B-frames before IDR in display order)."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,  # MaxPicOrderCntLsb = 16
        )

        calc = POCCalculator()

        # IDR with non-zero POC LSB
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=4,
        )
        calc.calculate(sps, h1, is_idr=True)

        # B-frame that would display before IDR
        # LSB=14 from LSB=4 is a forward jump of 10 > 8, so MSB decrements
        h2 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=1,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=14,
        )
        poc = calc.calculate(sps, h2, is_idr=False)

        # MSB = 0 - 16 = -16, POC = -16 + 14 = -2
        assert poc == -2, f"Negative POC should be allowed, got {poc}"

    def test_poc_zero_lsb_bits(self):
        """Handle case where log2_max_pic_order_cnt_lsb_minus4 = 0."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,  # 4 bits, max=16
        )

        calc = POCCalculator()

        h = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=15,  # Max valid value
        )
        poc = calc.calculate(sps, h, is_idr=True)

        assert poc == 15

    def test_max_poc_lsb_boundary(self):
        """Test at MaxPicOrderCntLsb boundary."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,  # MaxPicOrderCntLsb = 16
        )

        calc = POCCalculator()

        # At exactly MaxPicOrderCntLsb/2 boundary
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=0,
        )
        calc.calculate(sps, h1, is_idr=True)

        # Jump of exactly 8 (= MaxPicOrderCntLsb/2) - should NOT trigger wraparound
        h2 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=0,
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            pic_order_cnt_lsb=8,
        )
        poc2 = calc.calculate(sps, h2, is_idr=False)

        # At boundary, no wraparound
        assert poc2 == 8, f"At boundary should not wrap, got {poc2}"

    def test_multiple_slices_same_picture(self):
        """Multiple slices in same picture should have same POC."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=0,
        )

        calc = POCCalculator()

        # First slice of picture
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=4,
        )
        poc1 = calc.calculate(sps, h1, is_idr=True)

        # Second slice of same picture (same frame_num and pic_order_cnt_lsb)
        h2 = SliceHeader(
            first_mb_in_slice=10,  # Different starting MB
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,  # Same frame_num
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            pic_order_cnt_lsb=4,  # Same POC LSB
        )
        poc2 = calc.calculate(sps, h2, is_idr=True, same_picture=True)

        assert poc1 == poc2, "Multiple slices of same picture should have same POC"


class TestPOCIntegrationWithDecoder:
    """Tests for POC integration with main decoder."""

    def test_decoder_uses_poc_calculator(self):
        """Main decoder should use POCCalculator for POC calculation."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Decoder should have a POCCalculator instance
        assert hasattr(decoder, 'poc_calculator') or hasattr(decoder, '_poc_calculator'), \
            "H264Decoder should have poc_calculator attribute"

    def test_decoded_frame_has_correct_poc(self):
        """Decoded frames should have POC from POCCalculator."""
        from decoder.poc import POCCalculator

        # This test verifies the contract between decoder and POCCalculator
        calc = POCCalculator()

        sps = SPS(
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
        )

        # Simulate decoding sequence: I(0), P(4), B(2)
        headers = [
            (2, True, 0),   # I, poc_lsb=0
            (0, False, 4),  # P, poc_lsb=4
            (1, False, 2),  # B, poc_lsb=2
        ]

        pocs = []
        for slice_type, is_idr, poc_lsb in headers:
            h = SliceHeader(
                first_mb_in_slice=0,
                slice_type=slice_type,
                pic_parameter_set_id=0,
                frame_num=len(pocs),
                slice_qp_delta=0,
                header_bit_size=0,
                idr_pic_flag=is_idr,
                pic_order_cnt_lsb=poc_lsb,
            )
            pocs.append(calc.calculate(sps, h, is_idr=is_idr))

        assert pocs == [0, 4, 2], f"POCs should be [0, 4, 2], got {pocs}"


class TestPOCType1ExpectedDelta:
    """Tests for POC type 1 ExpectedDeltaPerPicOrderCntCycle calculation."""

    def test_expected_delta_calculation(self):
        """ExpectedDeltaPerPicOrderCntCycle is sum of offset_for_ref_frame."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=1,
            delta_pic_order_always_zero_flag=False,
            offset_for_non_ref_pic=0,
            offset_for_top_to_bottom_field=0,
            num_ref_frames_in_pic_order_cnt_cycle=4,
            offset_for_ref_frame=[1, 2, 3, 4],  # Sum = 10
        )

        calc = POCCalculator()

        # ExpectedDeltaPerPicOrderCntCycle should be calculated
        expected_delta = calc._compute_expected_delta(sps)

        assert expected_delta == 10, f"Expected delta should be 10, got {expected_delta}"

    def test_poc_type_1_frame_num_offset(self):
        """POC type 1 with non-zero offset_for_non_ref_pic."""
        from decoder.poc import POCCalculator

        sps = SPS(
            pic_order_cnt_type=1,
            delta_pic_order_always_zero_flag=False,
            offset_for_non_ref_pic=-2,  # Non-ref pics have POC offset
            offset_for_top_to_bottom_field=0,
            num_ref_frames_in_pic_order_cnt_cycle=1,
            offset_for_ref_frame=[2],
        )

        calc = POCCalculator()

        # IDR
        h0 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            idr_pic_flag=True,
            delta_pic_order_cnt=[0],
        )
        poc0 = calc.calculate(sps, h0, is_idr=True)

        # Non-reference frame
        h1 = SliceHeader(
            first_mb_in_slice=0,
            slice_type=1,  # B-slice (non-reference)
            pic_parameter_set_id=0,
            frame_num=1,
            slice_qp_delta=0,
            header_bit_size=0,
            delta_pic_order_cnt=[0],
        )
        poc1 = calc.calculate(sps, h1, is_idr=False, nal_ref_idc=0)

        # Non-ref should have offset_for_non_ref_pic applied
        assert poc0 == 0
        # expectedPicOrderCnt = 2 (from cycle) - 2 (offset_for_non_ref_pic) = 0
        assert poc1 == 0, f"Non-ref POC should have offset applied, got {poc1}"
