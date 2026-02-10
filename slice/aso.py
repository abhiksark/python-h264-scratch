from __future__ import annotations

import enum
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from decoder.frame import FrameAssembler
from slice.slice_header import SliceHeader


class ConcealmentStrategy(enum.Enum):
    COPY_FROM_ABOVE = "copy_from_above"
    COPY_FROM_LEFT = "copy_from_left"
    INTERPOLATE = "interpolate"
    SKIP = "skip"


class BitstreamError(Exception):
    pass


class SliceOrderDetector:
    def __init__(self):
        self._first_mbs: List[int] = []

    def add_slice(self, header: SliceHeader) -> None:
        self._first_mbs.append(int(header.first_mb_in_slice))

    def reset(self) -> None:
        self._first_mbs.clear()

    def is_in_order(self) -> bool:
        if len(self._first_mbs) <= 1:
            return True
        return all(self._first_mbs[i] <= self._first_mbs[i + 1] for i in range(len(self._first_mbs) - 1))

    def uses_aso(self) -> bool:
        return not self.is_in_order()

    def get_order_pattern(self) -> str:
        if len(self._first_mbs) <= 1:
            return "ascending"

        sorted_vals = sorted(self._first_mbs)
        if self._first_mbs == sorted_vals:
            return "ascending"
        if self._first_mbs == list(reversed(sorted_vals)):
            return "reverse"

        even = sorted_vals[0::2]
        odd = sorted_vals[1::2]
        if self._first_mbs == even + odd or self._first_mbs == odd + even:
            return "interleaved"

        return "out-of-order"


def validate_first_mb_in_slice(first_mb_in_slice: int, total_mbs: int) -> bool:
    first_mb_in_slice = int(first_mb_in_slice)
    total_mbs = int(total_mbs)
    return 0 <= first_mb_in_slice < total_mbs


class SliceValidator:
    def __init__(self, total_mbs: int):
        self.total_mbs = int(total_mbs)
        self._covered = np.zeros((self.total_mbs,), dtype=bool)

    def add_slice(self, first_mb: int, mb_count: int) -> None:
        start = int(first_mb)
        count = int(mb_count)
        if count <= 0 or self.total_mbs <= 0:
            return
        start = max(0, start)
        end = min(self.total_mbs, start + count)
        if end > start:
            self._covered[start:end] = True

    def has_gaps(self) -> bool:
        if self.total_mbs <= 0:
            return False
        return not bool(self._covered.all())

    def get_missing_mbs(self) -> List[int]:
        return [int(i) for i, covered in enumerate(self._covered) if not bool(covered)]


def is_primary_slice(header: SliceHeader) -> bool:
    return int(getattr(header, "redundant_pic_cnt", 0)) == 0


def is_redundant_slice(header: SliceHeader) -> bool:
    return int(getattr(header, "redundant_pic_cnt", 0)) > 0


class SliceSelector:
    def __init__(self):
        self._by_first_mb: Dict[int, List[SliceHeader]] = {}

    def add_slice(self, header: SliceHeader, data: bytes) -> None:
        first_mb = int(header.first_mb_in_slice)
        self._by_first_mb.setdefault(first_mb, []).append(header)

    def get_slice_for_mb(self, mb_idx: int) -> Optional[SliceHeader]:
        candidates = self._by_first_mb.get(int(mb_idx), [])
        if not candidates:
            return None

        for header in candidates:
            if is_primary_slice(header):
                return header

        return min(candidates, key=lambda h: int(getattr(h, "redundant_pic_cnt", 0)))


class SliceMBMapper:
    def __init__(
        self,
        width_mbs: int,
        height_mbs: int,
        slice_group_map_type: int = 0,
        num_slice_groups: int = 1,
    ):
        self.width_mbs = int(width_mbs)
        self.height_mbs = int(height_mbs)
        self.total_mbs = self.width_mbs * self.height_mbs

        self.slice_group_map_type = int(slice_group_map_type)
        self.num_slice_groups = int(num_slice_groups)

        self._mb_to_slice_id: List[int] = [-1] * self.total_mbs
        self._next_slice_id = 0

    def add_slice(self, header: SliceHeader, mb_count: int) -> int:
        slice_id = self._next_slice_id
        self._next_slice_id += 1

        first_mb = int(header.first_mb_in_slice)
        count = int(mb_count)
        if count <= 0 or self.total_mbs <= 0:
            return slice_id

        start = max(0, first_mb)
        end = min(self.total_mbs, start + count)
        for mb in range(start, end):
            self._mb_to_slice_id[mb] = slice_id

        return slice_id

    def get_slice_id(self, mb_idx: int) -> int:
        idx = int(mb_idx)
        if idx < 0 or idx >= self.total_mbs:
            return -1
        return int(self._mb_to_slice_id[idx])

    def get_mbs_for_slice(self, slice_id: int) -> List[int]:
        sid = int(slice_id)
        return [i for i, v in enumerate(self._mb_to_slice_id) if int(v) == sid]

    def mb_index_to_position(self, mb_idx: int) -> Tuple[int, int]:
        idx = int(mb_idx)
        return idx % self.width_mbs, idx // self.width_mbs

    def position_to_mb_index(self, x: int, y: int) -> int:
        return int(y) * self.width_mbs + int(x)

    def get_slice_group(self, mb_idx: int) -> int:
        if self.num_slice_groups <= 1:
            return 0

        idx = int(mb_idx)
        x = idx % self.width_mbs
        y = idx // self.width_mbs

        if self.slice_group_map_type == 1:
            return int((x + ((y * self.num_slice_groups) // 2)) % self.num_slice_groups)

        return int(idx % self.num_slice_groups)


class FrameCompletionTracker:
    def __init__(self, total_mbs: int, timeout_ms: int = 0):
        self.total_mbs = int(total_mbs)
        self.timeout_ms = int(timeout_ms)
        self._covered = np.zeros((self.total_mbs,), dtype=bool)
        self._start_time = time.monotonic()

    def mark_decoded(self, first_mb: int, mb_count: int) -> None:
        start = int(first_mb)
        count = int(mb_count)
        if count <= 0 or self.total_mbs <= 0:
            return

        start = max(0, start)
        end = min(self.total_mbs, start + count)
        if end > start:
            self._covered[start:end] = True

    def is_complete(self) -> bool:
        if self.total_mbs <= 0:
            return True
        return bool(self._covered.all())

    def get_decoded_mb_count(self) -> int:
        return int(self._covered.sum())

    def get_completion_percentage(self) -> float:
        if self.total_mbs <= 0:
            return 100.0
        return (float(self.get_decoded_mb_count()) / float(self.total_mbs)) * 100.0

    def get_missing_mbs(self) -> List[int]:
        return [int(i) for i, covered in enumerate(self._covered) if not bool(covered)]

    def get_concealment_strategy(self) -> ConcealmentStrategy:
        if self.is_complete():
            return ConcealmentStrategy.SKIP
        return ConcealmentStrategy.SKIP

    def has_timed_out(self) -> bool:
        if self.timeout_ms <= 0:
            return False
        elapsed_ms = (time.monotonic() - self._start_time) * 1000.0
        return elapsed_ms >= float(self.timeout_ms)

    def should_output_partial(self) -> bool:
        return self.has_timed_out() and not self.is_complete()


class ProgressiveFrameAssembler:
    def __init__(self, width_mbs: int, height_mbs: int):
        self._assembler = FrameAssembler(width_mbs=width_mbs, height_mbs=height_mbs)

    def add_slice(
        self,
        first_mb: int,
        luma_data: Optional[np.ndarray] = None,
        cb_data: Optional[np.ndarray] = None,
        cr_data: Optional[np.ndarray] = None,
    ) -> None:
        self._assembler.add_slice(first_mb=first_mb, luma_data=luma_data, cb_data=cb_data, cr_data=cr_data)

    def get_current_frame(self) -> np.ndarray:
        return self._assembler.assemble_luma()


class SliceBoundaryDetector:
    def __init__(self, width_mbs: int, height_mbs: int):
        self.width_mbs = int(width_mbs)
        self.height_mbs = int(height_mbs)
        self.total_mbs = self.width_mbs * self.height_mbs

        self._mb_to_slice_id: List[int] = [-1] * self.total_mbs
        self._next_slice_id = 0

    def add_slice(self, first_mb: int, mb_count: int) -> int:
        slice_id = self._next_slice_id
        self._next_slice_id += 1

        start = max(0, int(first_mb))
        end = min(self.total_mbs, start + int(mb_count))
        for mb in range(start, end):
            self._mb_to_slice_id[mb] = slice_id

        return slice_id

    def _get_slice_id(self, mb_idx: int) -> int:
        idx = int(mb_idx)
        if idx < 0 or idx >= self.total_mbs:
            return -1
        return int(self._mb_to_slice_id[idx])

    def is_slice_boundary(self, mb_a: int, mb_b: int) -> bool:
        a = self._get_slice_id(mb_a)
        b = self._get_slice_id(mb_b)
        if a < 0 or b < 0:
            return False
        return a != b

    def get_all_boundary_edges(self) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        for y in range(self.height_mbs):
            for x in range(self.width_mbs):
                mb = y * self.width_mbs + x
                if x + 1 < self.width_mbs:
                    right = mb + 1
                    if self.is_slice_boundary(mb_a=mb, mb_b=right):
                        edges.append((mb, right))
                if y + 1 < self.height_mbs:
                    below = mb + self.width_mbs
                    if self.is_slice_boundary(mb_a=mb, mb_b=below):
                        edges.append((mb, below))
        return edges


class SliceOverlapDetector:
    def __init__(self, total_mbs: int):
        self.total_mbs = int(total_mbs)
        self._owner: List[Optional[SliceHeader]] = [None] * self.total_mbs
        self._overlaps: set[int] = set()

    def add_slice(self, header: SliceHeader, mb_count: int) -> None:
        start = max(0, int(header.first_mb_in_slice))
        end = min(self.total_mbs, start + int(mb_count))

        for mb in range(start, end):
            existing = self._owner[mb]
            if existing is None:
                self._owner[mb] = header
                continue

            existing_primary = is_primary_slice(existing)
            new_primary = is_primary_slice(header)
            if existing_primary and new_primary:
                raise BitstreamError("Overlapping primary slices")

            self._overlaps.add(int(mb))

            if existing_primary:
                continue
            if new_primary:
                self._owner[mb] = header
                continue

            if int(getattr(header, "redundant_pic_cnt", 0)) < int(getattr(existing, "redundant_pic_cnt", 0)):
                self._owner[mb] = header

    def has_overlapping_slices(self) -> bool:
        return bool(self._overlaps)

    def get_overlapping_mbs(self) -> List[int]:
        return sorted(int(i) for i in self._overlaps)


class SliceQualitySelector:
    def __init__(self, total_mbs: int):
        self.total_mbs = int(total_mbs)
        self._candidates: List[List[SliceHeader]] = [[] for _ in range(self.total_mbs)]

    def add_slice(self, header: SliceHeader, mb_count: int) -> None:
        start = max(0, int(header.first_mb_in_slice))
        end = min(self.total_mbs, start + int(mb_count))
        for mb in range(start, end):
            self._candidates[mb].append(header)

    def select_slice(self, mb_idx: int, prefer_quality: bool = False) -> Optional[SliceHeader]:
        idx = int(mb_idx)
        if idx < 0 or idx >= self.total_mbs:
            return None

        candidates = self._candidates[idx]
        if not candidates:
            return None

        if not prefer_quality:
            for header in candidates:
                if is_primary_slice(header):
                    return header
            return min(candidates, key=lambda h: int(getattr(h, "redundant_pic_cnt", 0)))

        return min(
            candidates,
            key=lambda h: (int(getattr(h, "slice_qp_delta", 0)), int(getattr(h, "redundant_pic_cnt", 0))),
        )


class ASOPerformanceTracker:
    def __init__(self):
        self._arrival_order: List[int] = []
        self._stats: Dict[str, int] = {"reorder_count": 0}

    def start_frame(self) -> None:
        self._arrival_order = []

    def slice_received(self, first_mb: int) -> None:
        self._arrival_order.append(int(first_mb))

    def end_frame(self) -> None:
        sorted_vals = sorted(self._arrival_order)
        reorder_count = 0
        for i, v in enumerate(self._arrival_order):
            if i >= len(sorted_vals):
                break
            if int(v) != int(sorted_vals[i]):
                reorder_count += 1
        self._stats = {"reorder_count": int(reorder_count)}

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)
