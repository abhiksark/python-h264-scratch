# h264/inter/reference.py
"""Reference frame buffer for inter prediction.

Manages decoded frames that can be used as references for P and B frame
prediction. Implements a FIFO buffer with configurable maximum size.

H.264 Spec Reference: Section 8.2 - Decoded reference picture marking
"""

import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReferenceFrame:
    """A decoded frame that can be used for inter prediction.

    Attributes:
        luma: Y plane (height x width), uint8
        cb: Cb chroma plane (height/2 x width/2 for 4:2:0), uint8
        cr: Cr chroma plane (height/2 x width/2 for 4:2:0), uint8
        frame_num: Frame number from slice header, used for reference matching
        poc: Picture Order Count for display ordering and B-frame references
        mv_field: Motion vector field for temporal direct mode (optional)
    """
    luma: np.ndarray
    cb: np.ndarray
    cr: np.ndarray
    frame_num: int
    poc: int = 0
    mv_field: Optional[np.ndarray] = None
    ref_idx_field: Optional[np.ndarray] = None

    def _ensure_fields(self) -> None:
        """Lazily allocate MV and ref_idx fields at 8x8 block granularity."""
        if self.mv_field is None:
            # Store per-8x8-block: 2 entries per MB in each dimension
            blk_h = (self.luma.shape[0] // 16) * 2
            blk_w = (self.luma.shape[1] // 16) * 2
            self.mv_field = np.zeros((blk_h, blk_w, 2), dtype=np.int16)
            self.ref_idx_field = np.full(
                (blk_h, blk_w), -1, dtype=np.int8
            )

    def store_mv(
        self, mb_x: int, mb_y: int, mvx: int, mvy: int, ref_idx: int = 0,
        sub_idx: int = -1,
    ) -> None:
        """Store MV and ref_idx for a macroblock or sub-block.

        Args:
            mb_x, mb_y: Macroblock position
            mvx, mvy: Motion vector
            ref_idx: L0 reference index
            sub_idx: Sub-block index (0-3). If -1, store for all 4 sub-blocks.
        """
        self._ensure_fields()

        if sub_idx == -1:
            # Store same value for all 4 sub-blocks of the MB
            for si in range(4):
                by = mb_y * 2 + si // 2
                bx = mb_x * 2 + si % 2
                if (0 <= by < self.mv_field.shape[0] and
                        0 <= bx < self.mv_field.shape[1]):
                    self.mv_field[by, bx] = [mvx, mvy]
                    self.ref_idx_field[by, bx] = ref_idx
        else:
            by = mb_y * 2 + sub_idx // 2
            bx = mb_x * 2 + sub_idx % 2
            if (0 <= by < self.mv_field.shape[0] and
                    0 <= bx < self.mv_field.shape[1]):
                self.mv_field[by, bx] = [mvx, mvy]
                self.ref_idx_field[by, bx] = ref_idx

    def get_colocated_mv(
        self, mb_x: int, mb_y: int, sub_idx: int = 0,
    ) -> tuple:
        """Get co-located MV for direct mode.

        Args:
            mb_x, mb_y: Macroblock position
            sub_idx: Sub-block index (0-3) for 8x8 granularity

        Returns:
            Tuple of (mvx, mvy), or (0, 0) if not available
        """
        if self.mv_field is None:
            return (0, 0)

        by = mb_y * 2 + sub_idx // 2
        bx = mb_x * 2 + sub_idx % 2
        if 0 <= by < self.mv_field.shape[0] and 0 <= bx < self.mv_field.shape[1]:
            return int(self.mv_field[by, bx, 0]), int(self.mv_field[by, bx, 1])
        return (0, 0)

    def get_colocated_ref_idx(
        self, mb_x: int, mb_y: int, sub_idx: int = 0,
    ) -> int:
        """Get co-located L0 ref_idx for colZeroFlag check.

        Args:
            mb_x, mb_y: Macroblock position
            sub_idx: Sub-block index (0-3) for 8x8 granularity

        Returns:
            L0 ref_idx, or -1 if not available

        H.264 Spec: Section 8.4.1.2.2 - colZeroFlag requires refIdx check
        """
        if self.ref_idx_field is None:
            return -1

        by = mb_y * 2 + sub_idx // 2
        bx = mb_x * 2 + sub_idx % 2
        if (0 <= by < self.ref_idx_field.shape[0] and
                0 <= bx < self.ref_idx_field.shape[1]):
            return int(self.ref_idx_field[by, bx])
        return -1


class ReferenceFrameBuffer:
    """Buffer for storing reference frames.

    Implements a FIFO buffer where:
    - Newest frame is at ref_idx=0
    - When buffer is full, oldest frame is evicted
    - Frames can be retrieved by ref_idx or frame_num

    H.264 uses this for reference picture list construction.
    The buffer size is limited by max_num_ref_frames from SPS.
    """

    def __init__(self, max_frames: int):
        """Initialize buffer with maximum capacity.

        Args:
            max_frames: Maximum number of reference frames to store
                       (from SPS max_num_ref_frames)
        """
        self._max_frames = max_frames
        self._frames: List[ReferenceFrame] = []

    @property
    def max_frames(self) -> int:
        """Maximum number of frames this buffer can hold."""
        return self._max_frames

    def __len__(self) -> int:
        """Number of frames currently in buffer."""
        return len(self._frames)

    def add_frame(self, frame: ReferenceFrame) -> None:
        """Add a decoded frame to the buffer.

        The frame is inserted at the front (ref_idx=0).
        If buffer is full, the oldest frame is evicted.

        Args:
            frame: Decoded reference frame to add
        """
        # Insert at front (newest)
        self._frames.insert(0, frame)

        # Evict oldest if over capacity
        if len(self._frames) > self._max_frames:
            evicted = self._frames.pop()
            logger.debug(f"Evicted frame {evicted.frame_num} from reference buffer")

        logger.debug(f"Added frame {frame.frame_num} to reference buffer "
                    f"(size: {len(self._frames)}/{self._max_frames})")

    def get_frame(self, ref_idx: int) -> ReferenceFrame:
        """Get frame by reference index.

        Uses L0 reference list when available (after build_p_slice_ref_list
        or build_ref_lists), falls back to raw DPB order.

        When num_ref_idx_l0_active exceeds the list size, extra entries
        are padding (H.264 8.2.4.1). Clamp to the last available frame.

        Args:
            ref_idx: Reference index (0 = most recent)

        Returns:
            The reference frame at the given index

        Raises:
            IndexError: If buffer is empty or ref_idx is negative
        """
        # Use L0 list when available (P-slice ref list reordering)
        l0_list = getattr(self, '_l0_list', [])
        if l0_list:
            return self.get_l0_frame(ref_idx)

        if not self._frames:
            raise IndexError("Reference buffer is empty")
        if ref_idx < 0:
            raise IndexError(f"Negative reference index {ref_idx}")
        if ref_idx >= len(self._frames):
            logger.debug(
                f"ref_idx {ref_idx} exceeds DPB size {len(self._frames)}, "
                f"clamping to {len(self._frames) - 1}"
            )
            ref_idx = len(self._frames) - 1
        return self._frames[ref_idx]

    def get_frame_by_num(self, frame_num: int) -> Optional[ReferenceFrame]:
        """Get frame by frame_num.

        Used for reference picture matching when reordering reference lists.

        Args:
            frame_num: The frame_num to search for

        Returns:
            The reference frame with matching frame_num, or None if not found
        """
        for frame in self._frames:
            if frame.frame_num == frame_num:
                return frame
        return None

    def build_p_slice_ref_list(
        self,
        current_frame_num: int,
        max_frame_num: int,
        modification: 'RefPicListModification' = None,
    ) -> None:
        """Build L0 reference list for P-slices.

        Default order: descending frame_num (most recent first).
        Optionally applies ref_pic_list_modification.
        Builds _l0_list without modifying the DPB (_frames).

        Args:
            current_frame_num: frame_num of current slice
            max_frame_num: MaxFrameNum from SPS
            modification: Parsed ref_pic_list_modification_l0

        H.264 Spec: Section 8.2.4.1, 8.2.4.3.1
        """
        # Default: sorted by frame_num descending (work on a copy)
        self._l0_list = sorted(
            self._frames, key=lambda f: f.frame_num, reverse=True
        )

        if modification is None or not modification.modification_of_pic_nums_idc:
            return

        # Apply reference list reordering (H.264 8.2.4.3.1)
        # Uses the spec's shift-insert-compact algorithm, NOT simple pop/insert.
        num_active = len(self._l0_list)
        ref_list = list(self._l0_list) + [None]  # Extra slot for shift

        pic_num_no_wrap = current_frame_num
        ref_idx = 0

        for i, idc in enumerate(modification.modification_of_pic_nums_idc):
            if idc == 0 or idc == 1:
                abs_diff = modification.abs_diff_pic_num_minus1[i] + 1
                if idc == 0:
                    pic_num_no_wrap -= abs_diff
                    if pic_num_no_wrap < 0:
                        pic_num_no_wrap += max_frame_num
                else:
                    pic_num_no_wrap += abs_diff
                    if pic_num_no_wrap >= max_frame_num:
                        pic_num_no_wrap -= max_frame_num

                pic_num = pic_num_no_wrap

                # Find the frame with this PicNum in the DPB
                target = None
                for f in self._frames:
                    if f.frame_num == pic_num:
                        target = f
                        break

                if target is None:
                    continue

                # Shift right: make room at ref_idx
                for c in range(num_active, ref_idx, -1):
                    ref_list[c] = ref_list[c - 1]

                # Place target at ref_idx
                ref_list[ref_idx] = target

                # Compact: remove duplicate of target after ref_idx
                n_idx = ref_idx + 1
                for c in range(ref_idx + 1, num_active + 1):
                    if ref_list[c] is not target:
                        ref_list[n_idx] = ref_list[c]
                        n_idx += 1

                ref_idx += 1

        self._l0_list[:] = ref_list[:num_active]

    def remove_by_frame_num(self, frame_num: int) -> bool:
        """Remove a short-term reference frame by frame_num.

        Used by MMCO operation 1 (mark short-term as unused for reference).

        Args:
            frame_num: frame_num of the frame to remove

        Returns:
            True if a frame was removed, False if not found
        """
        for i, f in enumerate(self._frames):
            if f.frame_num == frame_num:
                removed = self._frames.pop(i)
                logger.debug(
                    f"MMCO: removed frame_num={frame_num} (POC={removed.poc}) "
                    f"from reference buffer"
                )
                return True
        return False

    def clear(self) -> None:
        """Remove all frames from buffer."""
        self._frames.clear()
        self._l0_list = []
        self._l1_list = []
        logger.debug("Cleared reference frame buffer")

    def build_ref_lists(self, current_poc: int) -> None:
        """Build L0 and L1 reference lists for B-slices.

        H.264 Section 8.2.4.2.3: Default ref list construction for B-slices.
        L0 = past frames (descending POC) + future frames (ascending POC).
        L1 = future frames (ascending POC) + past frames (descending POC).
        If L0 == L1 and len > 1, swap first two entries of L1.

        Args:
            current_poc: POC of current picture being decoded
        """
        past_frames = sorted(
            [f for f in self._frames if f.poc < current_poc],
            key=lambda f: f.poc, reverse=True,
        )
        future_frames = sorted(
            [f for f in self._frames if f.poc > current_poc],
            key=lambda f: f.poc,
        )

        # L0: past (descending) then future (ascending)
        self._l0_list = past_frames + future_frames

        # L1: future (ascending) then past (descending)
        self._l1_list = future_frames + past_frames

        # H.264 8.2.4.2.3: If L0 and L1 are identical and have > 1 entry,
        # swap the first two entries of L1
        if (len(self._l0_list) > 1
                and len(self._l0_list) == len(self._l1_list)
                and all(a.poc == b.poc
                        for a, b in zip(self._l0_list, self._l1_list))):
            self._l1_list[0], self._l1_list[1] = (
                self._l1_list[1], self._l1_list[0]
            )

        # Fallback: if no frames classified, use all
        if not self._l0_list and not self._l1_list and self._frames:
            self._l0_list = self._frames.copy()
            self._l1_list = self._frames.copy()

        logger.debug(
            f"Built ref lists for POC={current_poc}: "
            f"L0={[f.poc for f in self._l0_list]}, "
            f"L1={[f.poc for f in self._l1_list]}"
        )

    def get_l0_list(self) -> List[ReferenceFrame]:
        """Get L0 (past/forward) reference list.

        Returns:
            List of reference frames for L0 prediction
        """
        return getattr(self, '_l0_list', [])

    def get_l1_list(self) -> List[ReferenceFrame]:
        """Get L1 (future/backward) reference list.

        Returns:
            List of reference frames for L1 prediction
        """
        return getattr(self, '_l1_list', [])

    def get_l0_frame(self, ref_idx: int) -> ReferenceFrame:
        """Get frame from L0 list by index.

        Clamps to list size when num_ref_idx_active exceeds DPB.

        Args:
            ref_idx: Index into L0 list

        Returns:
            Reference frame

        Raises:
            IndexError: If L0 list is empty or ref_idx negative
        """
        l0_list = self.get_l0_list()
        if not l0_list:
            raise IndexError("L0 reference list is empty")
        if ref_idx < 0:
            raise IndexError(f"Negative L0 ref_idx {ref_idx}")
        if ref_idx >= len(l0_list):
            ref_idx = len(l0_list) - 1
        return l0_list[ref_idx]

    def get_l1_frame(self, ref_idx: int) -> ReferenceFrame:
        """Get frame from L1 list by index.

        Clamps to list size when num_ref_idx_active exceeds DPB.

        Args:
            ref_idx: Index into L1 list

        Returns:
            Reference frame

        Raises:
            IndexError: If L1 list is empty or ref_idx negative
        """
        l1_list = self.get_l1_list()
        if not l1_list:
            raise IndexError("L1 reference list is empty")
        if ref_idx < 0:
            raise IndexError(f"Negative L1 ref_idx {ref_idx}")
        if ref_idx >= len(l1_list):
            ref_idx = len(l1_list) - 1
        return l1_list[ref_idx]
