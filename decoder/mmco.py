
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from inter.reference import ReferenceFrame
from slice.slice_header import DecRefPicMarking


@dataclass
class MMCOResult:
    new_frame_num: Optional[int] = None
    poc_reset: bool = False


class MMCOProcessor:
    def __init__(self):
        self._short_term: List[ReferenceFrame] = []
        self._long_term: Dict[int, ReferenceFrame] = {}
        self._max_long_term_frame_idx: int = -1
        self._max_num_ref_frames: Optional[int] = None
        self._max_dpb_size: Optional[int] = None

    def add_short_term_ref(self, frame: ReferenceFrame) -> None:
        frame_num = int(frame.frame_num)
        self._short_term = [f for f in self._short_term if int(f.frame_num) != frame_num]
        self._short_term.append(frame)
        self._enforce_limits()

    def has_short_term_ref(self, pic_num: int) -> bool:
        target = int(pic_num)
        return any(int(f.frame_num) == target for f in self._short_term)

    def has_long_term_ref(self, long_term_pic_num: int) -> bool:
        return int(long_term_pic_num) in self._long_term

    def get_long_term_ref(self, long_term_pic_num: int) -> Optional[ReferenceFrame]:
        return self._long_term.get(int(long_term_pic_num))

    def mark_as_long_term(self, pic_num: int, long_term_frame_idx: int) -> None:
        idx = int(long_term_frame_idx)
        if self._max_long_term_frame_idx >= 0 and idx > self._max_long_term_frame_idx:
            raise ValueError("long_term_frame_idx exceeds max_long_term_frame_idx")
        if self._max_long_term_frame_idx < 0:
            self._max_long_term_frame_idx = idx

        pic_num_i = int(pic_num)
        st_ref: Optional[ReferenceFrame] = None
        for f in self._short_term:
            if int(f.frame_num) == pic_num_i:
                st_ref = f
                break
        if st_ref is None:
            return

        self._short_term = [f for f in self._short_term if int(f.frame_num) != pic_num_i]
        self._long_term[idx] = st_ref
        self._enforce_limits()

    def calc_pic_num(
        self,
        frame_num: int,
        is_field: bool,
        is_bottom_field: bool,
        current_frame_num: int,
        max_frame_num: int,
    ) -> int:
        frame_num_i = int(frame_num)
        current_frame_num_i = int(current_frame_num)
        max_frame_num_i = int(max_frame_num)
        if max_frame_num_i <= 0:
            raise ValueError("max_frame_num must be > 0")

        frame_num_wrap = frame_num_i
        if frame_num_i > current_frame_num_i:
            frame_num_wrap = frame_num_i - max_frame_num_i

        if not bool(is_field):
            return int(frame_num_wrap)
        return int(2 * frame_num_wrap + (1 if is_bottom_field else 0))

    def process_for_non_reference(self) -> None:
        return None

    def _mark_short_term_unused(self, pic_num: int) -> None:
        target = int(pic_num)
        self._short_term = [f for f in self._short_term if int(f.frame_num) != target]

    def _mark_current_as_long_term(self, current_frame: ReferenceFrame, long_term_frame_idx: int) -> None:
        idx = int(long_term_frame_idx)
        if self._max_long_term_frame_idx >= 0 and idx > self._max_long_term_frame_idx:
            raise ValueError("long_term_frame_idx exceeds max_long_term_frame_idx")
        if self._max_long_term_frame_idx < 0:
            self._max_long_term_frame_idx = idx
        self._long_term[idx] = current_frame
        self._short_term = [f for f in self._short_term if int(f.frame_num) != int(current_frame.frame_num)]
        self._enforce_limits()

    def process_idr(self, marking: DecRefPicMarking, idr_frame: Optional[ReferenceFrame] = None) -> None:
        self._short_term.clear()
        self._long_term.clear()
        self._max_long_term_frame_idx = -1

        if idr_frame is None:
            return

        if bool(getattr(marking, "long_term_reference_flag", False)):
            self._max_long_term_frame_idx = 0
            self._long_term[0] = idr_frame
            self._enforce_limits()
            return

        self.add_short_term_ref(idr_frame)

    def process(
        self,
        marking: DecRefPicMarking,
        current_frame_num: Optional[int] = None,
        max_frame_num: Optional[int] = None,
        current_poc: Optional[int] = None,
        current_frame: Optional[ReferenceFrame] = None,
    ) -> MMCOResult:
        _ = current_poc
        result = MMCOResult()

        if marking is None:
            return result
        if not bool(getattr(marking, "adaptive_ref_pic_marking_mode_flag", False)):
            return result

        ops = list(getattr(marking, "memory_management_control_operations", []) or [])
        diff_list = list(getattr(marking, "difference_of_pic_nums_minus1", []) or [])
        lt_pic_num_list = list(getattr(marking, "long_term_pic_num", []) or [])
        lt_frame_idx_list = list(getattr(marking, "long_term_frame_idx", []) or [])
        max_lt_idx_plus1_list = list(getattr(marking, "max_long_term_frame_idx_plus1", []) or [])

        diff_i = 0
        lt_pic_i = 0
        lt_idx_i = 0
        max_lt_i = 0

        for op in ops:
            op_i = int(op)
            if op_i == 0:
                break

            if op_i == 1:
                if current_frame_num is None:
                    diff_i += 1 if diff_i < len(diff_list) else 0
                    continue
                diff = int(diff_list[diff_i]) + 1 if diff_i < len(diff_list) else 1
                diff_i += 1 if diff_i < len(diff_list) else 0
                target = int(current_frame_num) - diff
                if max_frame_num is not None and int(max_frame_num) > 0:
                    target = target % int(max_frame_num)
                self._mark_short_term_unused(target)
                continue

            if op_i == 2:
                if lt_pic_i >= len(lt_pic_num_list):
                    continue
                lt_pic = int(lt_pic_num_list[lt_pic_i])
                lt_pic_i += 1
                if lt_pic in self._long_term:
                    del self._long_term[lt_pic]
                continue

            if op_i == 3:
                if current_frame_num is None:
                    diff_i += 1 if diff_i < len(diff_list) else 0
                    lt_idx_i += 1 if lt_idx_i < len(lt_frame_idx_list) else 0
                    continue
                diff = int(diff_list[diff_i]) + 1 if diff_i < len(diff_list) else 1
                diff_i += 1 if diff_i < len(diff_list) else 0
                if lt_idx_i >= len(lt_frame_idx_list):
                    continue
                lt_idx = int(lt_frame_idx_list[lt_idx_i])
                lt_idx_i += 1
                target = int(current_frame_num) - diff
                if max_frame_num is not None and int(max_frame_num) > 0:
                    target = target % int(max_frame_num)
                self.mark_as_long_term(target, lt_idx)
                continue

            if op_i == 4:
                if max_lt_i >= len(max_lt_idx_plus1_list):
                    continue
                plus1 = int(max_lt_idx_plus1_list[max_lt_i])
                max_lt_i += 1
                new_max = -1 if plus1 == 0 else plus1 - 1
                self.set_max_long_term_frame_idx(new_max)
                continue

            if op_i == 5:
                self._short_term.clear()
                self._long_term.clear()
                self._max_long_term_frame_idx = -1
                result.new_frame_num = 0
                result.poc_reset = True
                continue

            if op_i == 6:
                if lt_idx_i >= len(lt_frame_idx_list):
                    continue
                lt_idx = int(lt_frame_idx_list[lt_idx_i])
                lt_idx_i += 1
                if current_frame is None:
                    continue
                self._mark_current_as_long_term(current_frame, lt_idx)
                continue

        self._enforce_limits()
        return result

    def process_non_idr(
        self,
        marking: DecRefPicMarking,
        current_frame: Optional[ReferenceFrame] = None,
        current_frame_num: Optional[int] = None,
        max_frame_num: Optional[int] = None,
        current_poc: Optional[int] = None,
    ) -> MMCOResult:
        result = MMCOResult()
        if marking is not None and bool(getattr(marking, "adaptive_ref_pic_marking_mode_flag", False)):
            result = self.process(
                marking,
                current_frame_num=current_frame_num,
                max_frame_num=max_frame_num,
                current_poc=current_poc,
                current_frame=current_frame,
            )

        if current_frame is None:
            return result

        if any(int(f.frame_num) == int(current_frame.frame_num) for f in self._long_term.values()):
            self._enforce_limits()
            return result

        self.add_short_term_ref(current_frame)
        return result

    def set_max_num_ref_frames(self, max_refs: int) -> None:
        self._max_num_ref_frames = int(max_refs)
        self._enforce_limits()

    def set_max_dpb_size(self, max_size: int) -> None:
        self._max_dpb_size = int(max_size)
        self._enforce_limits()

    def set_max_long_term_frame_idx(self, max_idx: int) -> None:
        self._max_long_term_frame_idx = int(max_idx)
        if self._max_long_term_frame_idx < 0:
            self._long_term.clear()
        else:
            for idx in list(self._long_term.keys()):
                if int(idx) > self._max_long_term_frame_idx:
                    del self._long_term[int(idx)]
        self._enforce_limits()

    def get_max_long_term_frame_idx(self) -> int:
        return int(self._max_long_term_frame_idx)

    def get_short_term_count(self) -> int:
        return len(self._short_term)

    def get_long_term_count(self) -> int:
        return len(self._long_term)

    def get_total_ref_count(self) -> int:
        return len(self._short_term) + len(self._long_term)

    def _enforce_limits(self) -> None:
        if self._max_long_term_frame_idx < 0:
            self._long_term.clear()
        else:
            for idx in list(self._long_term.keys()):
                if int(idx) > self._max_long_term_frame_idx:
                    del self._long_term[int(idx)]

        limit: Optional[int] = None
        if self._max_num_ref_frames is not None:
            limit = int(self._max_num_ref_frames)
        if self._max_dpb_size is not None:
            limit = int(self._max_dpb_size) if limit is None else min(limit, int(self._max_dpb_size))

        if limit is None:
            return

        while self.get_total_ref_count() > limit:
            if self._short_term:
                self._short_term.pop(0)
                continue
            if self._long_term:
                del self._long_term[sorted(self._long_term.keys())[-1]]
                continue
            break
