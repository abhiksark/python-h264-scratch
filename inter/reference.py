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
    """
    luma: np.ndarray
    cb: np.ndarray
    cr: np.ndarray
    frame_num: int


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

        ref_idx=0 is the most recently added frame.
        ref_idx=1 is the second most recent, etc.

        Args:
            ref_idx: Reference index (0 = most recent)

        Returns:
            The reference frame at the given index

        Raises:
            IndexError: If ref_idx is out of range
        """
        if ref_idx < 0 or ref_idx >= len(self._frames):
            raise IndexError(
                f"Reference index {ref_idx} out of range "
                f"(buffer has {len(self._frames)} frames)"
            )
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

    def clear(self) -> None:
        """Remove all frames from buffer."""
        self._frames.clear()
        logger.debug("Cleared reference frame buffer")
