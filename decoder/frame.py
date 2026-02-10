from __future__ import annotations

from typing import Optional

import numpy as np


class FrameAssembler:
    def __init__(self, width_mbs: int, height_mbs: int):
        self.width_mbs = int(width_mbs)
        self.height_mbs = int(height_mbs)
        self.total_mbs = self.width_mbs * self.height_mbs

        self._luma = np.zeros((self.height_mbs * 16, self.width_mbs * 16), dtype=np.uint8)
        self._cb = np.zeros((self.height_mbs * 8, self.width_mbs * 8), dtype=np.uint8)
        self._cr = np.zeros((self.height_mbs * 8, self.width_mbs * 8), dtype=np.uint8)

        self._luma_present = np.zeros((self.total_mbs,), dtype=bool)

    def add_slice(
        self,
        first_mb: int,
        luma_data: Optional[np.ndarray] = None,
        cb_data: Optional[np.ndarray] = None,
        cr_data: Optional[np.ndarray] = None,
    ) -> None:
        first_mb = int(first_mb)

        if luma_data is not None:
            self._add_plane(first_mb=first_mb, data=luma_data, block_w=16, block_h=16, dest=self._luma)
            slice_w_mbs = int(luma_data.shape[1] // 16)
            slice_h_mbs = int(luma_data.shape[0] // 16)
            for i in range(slice_w_mbs * slice_h_mbs):
                mb_addr = first_mb + i
                if 0 <= mb_addr < self.total_mbs:
                    self._luma_present[mb_addr] = True

        if cb_data is not None:
            self._add_plane(first_mb=first_mb, data=cb_data, block_w=8, block_h=8, dest=self._cb)

        if cr_data is not None:
            self._add_plane(first_mb=first_mb, data=cr_data, block_w=8, block_h=8, dest=self._cr)

    def _add_plane(self, first_mb: int, data: np.ndarray, block_w: int, block_h: int, dest: np.ndarray) -> None:
        slice_w_mbs = int(data.shape[1] // block_w)
        slice_h_mbs = int(data.shape[0] // block_h)
        mb_count = slice_w_mbs * slice_h_mbs

        for i in range(mb_count):
            mb_addr = first_mb + i
            if mb_addr < 0 or mb_addr >= self.total_mbs:
                continue

            frame_x = mb_addr % self.width_mbs
            frame_y = mb_addr // self.width_mbs

            slice_x = i % slice_w_mbs
            slice_y = i // slice_w_mbs

            src_y0 = slice_y * block_h
            src_y1 = src_y0 + block_h
            src_x0 = slice_x * block_w
            src_x1 = src_x0 + block_w

            dst_y0 = frame_y * block_h
            dst_y1 = dst_y0 + block_h
            dst_x0 = frame_x * block_w
            dst_x1 = dst_x0 + block_w

            dest[dst_y0:dst_y1, dst_x0:dst_x1] = data[src_y0:src_y1, src_x0:src_x1]

    def assemble_luma(self) -> np.ndarray:
        return self._luma

    def assemble_cb(self) -> np.ndarray:
        return self._cb

    def assemble_cr(self) -> np.ndarray:
        return self._cr

    def is_complete(self) -> bool:
        if self.total_mbs == 0:
            return True
        return bool(self._luma_present.all())
