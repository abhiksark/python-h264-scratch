# h264/decoder/poc.py
"""Picture Order Count (POC) calculator.

POC is critical for B-frame decoding because:
- Decode order ≠ Display order
- B-frames are decoded AFTER their reference frames but displayed BETWEEN them
- POC determines the correct display order

Example: IBBP decode/display pattern
  Decode order: I0, P3, B1, B2  (P must be decoded before B can reference it)
  Display order: I0, B1, B2, P3  (sorted by POC: 0, 2, 4, 6)

H.264 Spec Reference: Section 8.2.1 - Decoding process for picture order count

POC Types:
- Type 0: Explicit LSB in slice header, MSB derived with wraparound detection
- Type 1: Delta from expected POC, uses reference frame cycle
- Type 2: Directly from frame_num, no B-frames possible
"""

from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class POCCalculator:
    """Stateful Picture Order Count calculator.

    Maintains state between frames for POC type 0 (MSB tracking)
    and type 1 (prev_frame_num tracking).

    Usage:
        calc = POCCalculator()
        for each frame:
            poc = calc.calculate(sps, header, is_idr)
            # Use POC for reference list ordering
    """

    # State for POC type 0 (LSB-based)
    prev_pic_order_cnt_lsb: int = 0
    prev_pic_order_cnt_msb: int = 0

    # State for POC type 1 (delta-based)
    prev_frame_num: int = 0
    prev_frame_num_offset: int = 0

    def reset(self) -> None:
        """Reset all state (called on IDR or explicit reset)."""
        self.prev_pic_order_cnt_lsb = 0
        self.prev_pic_order_cnt_msb = 0
        self.prev_frame_num = 0
        self.prev_frame_num_offset = 0
        logger.debug("POC state reset")

    def calculate(
        self,
        sps: 'SPS',
        header: 'SliceHeader',
        is_idr: bool,
        nal_ref_idc: int = 1,
    ) -> int:
        """Calculate POC for current picture.

        Args:
            sps: Sequence Parameter Set with POC configuration
            header: Slice header with POC-related fields
            is_idr: True if this is an IDR (Instantaneous Decoder Refresh) picture
            nal_ref_idc: NAL reference indicator (0 = non-reference, >0 = reference)

        Returns:
            Picture Order Count for this frame
        """
        poc_type = getattr(sps, 'pic_order_cnt_type', 0)

        if poc_type == 0:
            return self._calculate_type0(sps, header, is_idr)
        elif poc_type == 1:
            return self._calculate_type1(sps, header, is_idr, nal_ref_idc)
        elif poc_type == 2:
            return self._calculate_type2(sps, header, is_idr, nal_ref_idc)
        else:
            raise ValueError(f"Invalid pic_order_cnt_type: {poc_type}")

    def _calculate_type0(
        self,
        sps: 'SPS',
        header: 'SliceHeader',
        is_idr: bool,
    ) -> int:
        """POC Type 0: LSB-based with MSB wraparound detection.

        H.264 Spec: Section 8.2.1.1

        The encoder sends pic_order_cnt_lsb (the low bits of POC).
        The decoder derives the MSB by detecting wraparound:

        - If LSB decreased by more than MaxPicOrderCntLsb/2:
          We wrapped forward → increment MSB
        - If LSB increased by more than MaxPicOrderCntLsb/2:
          We wrapped backward → decrement MSB

        Why this works:
        - Normal forward progression: LSB increases or wraps with small decrease
        - Backward jump (rare): LSB increases significantly

        Example (MaxPicOrderCntLsb=16):
          prev_lsb=14, curr_lsb=2: decrease of 12 > 8 → MSB += 16
          Final POC = MSB + LSB = 16 + 2 = 18
        """
        # Get MaxPicOrderCntLsb from SPS
        log2_max = getattr(sps, 'log2_max_pic_order_cnt_lsb_minus4', 0) + 4
        max_pic_order_cnt_lsb = 1 << log2_max

        # Get current LSB from slice header
        pic_order_cnt_lsb = getattr(header, 'pic_order_cnt_lsb', 0)

        if is_idr:
            # IDR: Reset everything, MSB = 0
            pic_order_cnt_msb = 0
            logger.debug(f"POC Type 0 IDR: LSB={pic_order_cnt_lsb}, MSB=0")
        else:
            # Non-IDR: Derive MSB from wraparound detection
            # H.264 equations (8-3) through (8-5)

            prev_lsb = self.prev_pic_order_cnt_lsb
            prev_msb = self.prev_pic_order_cnt_msb

            # Wraparound detection using the "half range" rule
            if (pic_order_cnt_lsb < prev_lsb and
                    (prev_lsb - pic_order_cnt_lsb) >= max_pic_order_cnt_lsb // 2):
                # LSB decreased significantly → we wrapped forward
                # Example: prev=14, curr=2, diff=12 > 8 → MSB increases
                pic_order_cnt_msb = prev_msb + max_pic_order_cnt_lsb
                logger.debug(f"POC Type 0 forward wrap: {prev_lsb}→{pic_order_cnt_lsb}, "
                           f"MSB: {prev_msb}→{pic_order_cnt_msb}")

            elif (pic_order_cnt_lsb > prev_lsb and
                    (pic_order_cnt_lsb - prev_lsb) > max_pic_order_cnt_lsb // 2):
                # LSB increased significantly → we wrapped backward (rare)
                # Example: prev=2, curr=14, diff=12 > 8 → MSB decreases
                pic_order_cnt_msb = prev_msb - max_pic_order_cnt_lsb
                logger.debug(f"POC Type 0 backward wrap: {prev_lsb}→{pic_order_cnt_lsb}, "
                           f"MSB: {prev_msb}→{pic_order_cnt_msb}")
            else:
                # No wraparound, MSB unchanged
                pic_order_cnt_msb = prev_msb

        # Final POC = MSB + LSB
        poc = pic_order_cnt_msb + pic_order_cnt_lsb

        # Update state for next frame
        self.prev_pic_order_cnt_lsb = pic_order_cnt_lsb
        self.prev_pic_order_cnt_msb = pic_order_cnt_msb

        logger.debug(f"POC Type 0: LSB={pic_order_cnt_lsb}, MSB={pic_order_cnt_msb}, POC={poc}")
        return poc

    def _calculate_type1(
        self,
        sps: 'SPS',
        header: 'SliceHeader',
        is_idr: bool,
        nal_ref_idc: int,
    ) -> int:
        """POC Type 1: Delta-based with reference frame cycle.

        H.264 Spec: Section 8.2.1.2

        Uses delta_pic_order_cnt from slice header plus an "expected POC"
        derived from frame_num and a reference frame cycle defined in SPS.

        The SPS contains:
        - offset_for_ref_frame[]: POC increment per reference frame in cycle
        - ExpectedDeltaPerPicOrderCntCycle = sum(offset_for_ref_frame)

        This type is more complex but allows flexible POC patterns.
        """
        # Get cycle parameters from SPS
        num_ref_frames_in_cycle = getattr(sps, 'num_ref_frames_in_pic_order_cnt_cycle', 0)
        offset_for_ref_frame = getattr(sps, 'offset_for_ref_frame', [])
        offset_for_non_ref_pic = getattr(sps, 'offset_for_non_ref_pic', 0)
        delta_pic_order_always_zero_flag = getattr(sps, 'delta_pic_order_always_zero_flag', False)
        log2_max_frame_num = getattr(sps, 'log2_max_frame_num_minus4', 0) + 4
        max_frame_num = 1 << log2_max_frame_num

        if is_idr:
            # IDR: Reset state
            self.prev_frame_num = 0
            self.prev_frame_num_offset = 0
            return 0

        frame_num = getattr(header, 'frame_num', 0)

        # Calculate frame_num_offset for wraparound
        if frame_num < self.prev_frame_num:
            frame_num_offset = self.prev_frame_num_offset + max_frame_num
        else:
            frame_num_offset = self.prev_frame_num_offset

        # Calculate absFrameNum
        if num_ref_frames_in_cycle > 0:
            abs_frame_num = frame_num_offset + frame_num
        else:
            abs_frame_num = 0

        # Non-reference pictures don't contribute to cycle
        if nal_ref_idc == 0 and abs_frame_num > 0:
            abs_frame_num -= 1

        # Calculate expected POC from cycle
        expected_pic_order_cnt = 0
        if abs_frame_num > 0:
            # Sum of offset_for_ref_frame for complete cycles
            expected_delta_per_cycle = sum(offset_for_ref_frame) if offset_for_ref_frame else 0

            pic_order_cnt_cycle_cnt = (abs_frame_num - 1) // num_ref_frames_in_cycle if num_ref_frames_in_cycle > 0 else 0
            frame_num_in_cycle = (abs_frame_num - 1) % num_ref_frames_in_cycle if num_ref_frames_in_cycle > 0 else 0

            expected_pic_order_cnt = pic_order_cnt_cycle_cnt * expected_delta_per_cycle

            # Add partial cycle contributions
            for i in range(frame_num_in_cycle + 1):
                if i < len(offset_for_ref_frame):
                    expected_pic_order_cnt += offset_for_ref_frame[i]

        # Add offset for non-reference pictures
        if nal_ref_idc == 0:
            expected_pic_order_cnt += offset_for_non_ref_pic

        # Add delta from slice header (if not always zero)
        if delta_pic_order_always_zero_flag:
            delta = 0
        else:
            delta_list = getattr(header, 'delta_pic_order_cnt', [0])
            delta = delta_list[0] if delta_list else 0

        poc = expected_pic_order_cnt + delta

        # Update state
        self.prev_frame_num = frame_num
        self.prev_frame_num_offset = frame_num_offset

        logger.debug(f"POC Type 1: frame_num={frame_num}, expected={expected_pic_order_cnt}, "
                    f"delta={delta}, POC={poc}")
        return poc

    def _calculate_type2(
        self,
        sps: 'SPS',
        header: 'SliceHeader',
        is_idr: bool,
        nal_ref_idc: int,
    ) -> int:
        """POC Type 2: Frame-num based (simplest, no B-frames).

        H.264 Spec: Section 8.2.1.3

        Directly derives POC from frame_num:
        - Reference frames: POC = 2 * temp_poc
        - Non-reference frames: POC = 2 * temp_poc - 1

        Where temp_poc handles frame_num wraparound.

        Constraint: This type cannot support B-frames because
        decode order equals display order (POC is monotonic with frame_num).
        """
        log2_max_frame_num = getattr(sps, 'log2_max_frame_num_minus4', 0) + 4
        max_frame_num = 1 << log2_max_frame_num

        if is_idr:
            # IDR: Reset state
            self.prev_frame_num = 0
            self.prev_frame_num_offset = 0
            temp_poc = 0
        else:
            frame_num = getattr(header, 'frame_num', 0)

            # Handle frame_num wraparound
            if frame_num < self.prev_frame_num:
                frame_num_offset = self.prev_frame_num_offset + max_frame_num
            else:
                frame_num_offset = self.prev_frame_num_offset

            temp_poc = frame_num_offset + frame_num

            # Update state
            self.prev_frame_num = frame_num
            self.prev_frame_num_offset = frame_num_offset

        # Reference vs non-reference distinction
        if nal_ref_idc != 0:
            # Reference picture
            poc = 2 * temp_poc
        else:
            # Non-reference picture
            poc = 2 * temp_poc - 1

        logger.debug(f"POC Type 2: temp_poc={temp_poc}, is_ref={nal_ref_idc!=0}, POC={poc}")
        return poc


# Convenience function for simple cases
def calculate_poc(
    sps: 'SPS',
    header: 'SliceHeader',
    is_idr: bool,
    calculator: Optional[POCCalculator] = None,
) -> int:
    """Calculate POC using a calculator instance.

    Args:
        sps: Sequence Parameter Set
        header: Slice header
        is_idr: True if IDR picture
        calculator: Optional existing calculator (creates new if None)

    Returns:
        Picture Order Count
    """
    if calculator is None:
        calculator = POCCalculator()
    return calculator.calculate(sps, header, is_idr)
