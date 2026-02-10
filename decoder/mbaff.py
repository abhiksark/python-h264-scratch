
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def mb_addr_to_pair_idx(mb_addr: int) -> int:
    mb_addr_i = int(mb_addr)
    if mb_addr_i < 0:
        raise ValueError("mb_addr must be >= 0")
    return mb_addr_i // 2


def is_top_mb_of_pair(mb_addr: int) -> bool:
    mb_addr_i = int(mb_addr)
    if mb_addr_i < 0:
        raise ValueError("mb_addr must be >= 0")
    return (mb_addr_i % 2) == 0


def get_pair_mb_addrs(pair_idx: int) -> Tuple[int, int]:
    pair_idx_i = int(pair_idx)
    if pair_idx_i < 0:
        raise ValueError("pair_idx must be >= 0")
    top = pair_idx_i * 2
    return top, top + 1


def get_pair_position(pair_idx: int, mb_width: int) -> Tuple[int, int]:
    pair_idx_i = int(pair_idx)
    mb_width_i = int(mb_width)
    if pair_idx_i < 0:
        raise ValueError("pair_idx must be >= 0")
    if mb_width_i <= 0:
        raise ValueError("mb_width must be > 0")
    return pair_idx_i % mb_width_i, pair_idx_i // mb_width_i


def parse_mb_field_decoding_flag(
    reader: Any,
    mb_addr: int,
    mbaff_frame_flag: bool,
    pair_field_flags: Optional[Dict[int, bool]] = None,
) -> bool:
    if not mbaff_frame_flag:
        return False
    flags = pair_field_flags if pair_field_flags is not None else {}
    pair_idx = mb_addr_to_pair_idx(mb_addr)
    if not is_top_mb_of_pair(mb_addr):
        return bool(flags.get(pair_idx, False))
    if reader is None:
        flags[pair_idx] = False
        return False
    is_field = bool(reader.read_flag())
    flags[pair_idx] = is_field
    return is_field


def get_left_neighbor_mbaff(
    mb_addr: int,
    current_is_field: bool,
    neighbor_is_field: bool,
    mb_width: int,
) -> Optional[int]:
    _ = current_is_field
    _ = neighbor_is_field
    pair_idx = mb_addr_to_pair_idx(mb_addr)
    pair_x, _pair_y = get_pair_position(pair_idx, mb_width)
    if pair_x == 0:
        return None
    left_pair_idx = pair_idx - 1
    left_top, left_bottom = get_pair_mb_addrs(left_pair_idx)
    return left_top if is_top_mb_of_pair(mb_addr) else left_bottom


def get_top_neighbor_mbaff(
    mb_addr: int,
    current_is_field: bool,
    mb_width: int,
) -> Optional[int]:
    pair_idx = mb_addr_to_pair_idx(mb_addr)
    pair_x, pair_y = get_pair_position(pair_idx, mb_width)
    _ = pair_x
    if current_is_field:
        if pair_y == 0:
            return None
        above_pair_idx = pair_idx - mb_width
        above_top, above_bottom = get_pair_mb_addrs(above_pair_idx)
        return above_top if is_top_mb_of_pair(mb_addr) else above_bottom

    if not is_top_mb_of_pair(mb_addr):
        return int(mb_addr) - 1
    if pair_y == 0:
        return None
    above_pair_idx = pair_idx - mb_width
    _above_top, above_bottom = get_pair_mb_addrs(above_pair_idx)
    return above_bottom


def scale_mv_for_field_prediction(
    mv: np.ndarray,
    current_is_field: bool,
    neighbor_is_field: bool,
) -> np.ndarray:
    mv_i = np.asarray(mv, dtype=np.int32).reshape(2)
    if current_is_field and not neighbor_is_field:
        return np.array([mv_i[0], mv_i[1] // 2], dtype=np.int32)
    if (not current_is_field) and neighbor_is_field:
        return np.array([mv_i[0], mv_i[1] * 2], dtype=np.int32)
    return mv_i


def _median3(a: int, b: int, c: int) -> int:
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return b


def predict_mv_mbaff(
    mv_a: Optional[np.ndarray],
    mv_b: Optional[np.ndarray],
    mv_c: Optional[np.ndarray],
    current_is_field: bool,
    neighbor_a_is_field: Optional[bool],
    neighbor_b_is_field: Optional[bool],
    neighbor_c_is_field: Optional[bool],
) -> Optional[np.ndarray]:
    if mv_a is None and mv_b is None and mv_c is None:
        return None

    def _norm(mv: Optional[np.ndarray], is_field: Optional[bool]) -> Optional[np.ndarray]:
        if mv is None:
            return None
        if is_field is None:
            return np.asarray(mv, dtype=np.int32).reshape(2)
        return scale_mv_for_field_prediction(mv, current_is_field=current_is_field, neighbor_is_field=is_field)

    a = _norm(mv_a, neighbor_a_is_field)
    b = _norm(mv_b, neighbor_b_is_field)
    c = _norm(mv_c, neighbor_c_is_field)

    if b is None and a is not None:
        b = a
    if b is None and c is not None:
        b = c
    if a is None:
        a = b
    if c is None:
        c = b

    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    cx, cy = int(c[0]), int(c[1])
    return np.array([
        _median3(ax, bx, cx),
        _median3(ay, by, cy),
    ], dtype=np.int32)


def derive_direct_mv_mbaff(
    current_is_field: bool,
    colocated_is_field: bool,
    colocated_mv: np.ndarray,
    colocated_ref: int,
    poc_current: int,
    poc_l0: int,
    poc_l1: int,
) -> Tuple[np.ndarray, np.ndarray]:
    _ = colocated_ref
    _ = poc_current
    _ = poc_l0
    _ = poc_l1
    mv = scale_mv_for_field_prediction(
        colocated_mv,
        current_is_field=current_is_field,
        neighbor_is_field=colocated_is_field,
    )
    return mv.copy(), mv.copy()


def get_deblock_edges_mbaff(is_field: bool, direction: str) -> List[Tuple[int, int]]:
    if direction not in {"vertical", "horizontal"}:
        raise ValueError("direction must be 'vertical' or 'horizontal'")

    if direction == "vertical":
        return [(x, 0) for x in (0, 4, 8, 12)]

    if not is_field:
        return [(0, y) for y in (0, 4, 8, 12)]
    return [(0, y) for y in (0, 8)]


def calc_bs_mbaff_boundary(
    current_is_field: bool,
    neighbor_is_field: bool,
    current_is_intra: bool,
    neighbor_is_intra: bool,
    current_mv: np.ndarray,
    neighbor_mv: np.ndarray,
    current_ref: int,
    neighbor_ref: int,
    current_has_coeff: bool,
    neighbor_has_coeff: bool,
) -> int:
    if current_is_intra or neighbor_is_intra:
        return 4
    if current_has_coeff or neighbor_has_coeff:
        return 2
    if current_ref != neighbor_ref:
        return 1
    mv_n = scale_mv_for_field_prediction(neighbor_mv, current_is_field=current_is_field, neighbor_is_field=neighbor_is_field)
    mv_c = np.asarray(current_mv, dtype=np.int32).reshape(2)
    return 1 if bool(np.any(mv_c != mv_n)) else 0


def deblock_mbaff_pair_boundary(
    luma: np.ndarray,
    pair_left: Dict[str, Any],
    pair_right: Dict[str, Any],
    qp: int,
) -> np.ndarray:
    _ = pair_left
    _ = pair_right
    _ = qp
    out = np.asarray(luma).copy()
    if out.ndim != 2 or out.shape[1] < 2:
        return out
    x = out.shape[1] // 2
    if x <= 0 or x >= out.shape[1]:
        return out
    p = out[:, x - 1].astype(np.int16)
    q = out[:, x].astype(np.int16)
    avg = ((p + q) // 2).astype(np.uint8)
    out[:, x - 1] = avg
    out[:, x] = avg
    return out


def is_field_macroblock(mb_addr: int, pair_field_flags: Dict[int, bool]) -> bool:
    return bool(pair_field_flags.get(mb_addr_to_pair_idx(mb_addr), False))


def get_field_line_in_frame(field_line: int, is_bottom_field: bool, pair_y: int) -> int:
    return int(pair_y) * 32 + int(field_line) * 2 + (1 if is_bottom_field else 0)


def motion_comp_field_mb(
    ref_luma: np.ndarray,
    mv: np.ndarray,
    mb_x: int,
    mb_y: int,
    is_bottom_field: bool,
) -> np.ndarray:
    mv_i = np.asarray(mv, dtype=np.int32).reshape(2)
    x0 = int(mb_x) * 16 + int(mv_i[0])
    pair_y = int(mb_y)
    base_y = pair_y * 32 + int(mv_i[1])
    lines = [base_y + i * 2 + (1 if is_bottom_field else 0) for i in range(16)]
    out = np.zeros((16, 16), dtype=np.uint8)
    for i, y in enumerate(lines):
        if 0 <= y < ref_luma.shape[0]:
            out[i, :] = ref_luma[y, x0:x0 + 16]
    return out


def interleave_field_pair(top_field: np.ndarray, bottom_field: np.ndarray) -> np.ndarray:
    top = np.asarray(top_field)
    bottom = np.asarray(bottom_field)
    if top.shape != bottom.shape:
        raise ValueError("top_field and bottom_field must have same shape")
    h, w = top.shape
    frame = np.zeros((h * 2, w), dtype=top.dtype)
    frame[0::2, :] = top
    frame[1::2, :] = bottom
    return frame


def build_ref_list_mbaff(
    ref_buffer: Sequence[Dict[str, Any]],
    current_poc: int,
    current_is_field: bool,
) -> List[Dict[str, Any]]:
    _ = current_is_field
    return sorted(list(ref_buffer), key=lambda r: abs(int(r.get("poc", 0)) - int(current_poc)))


def modify_ref_list_mbaff(
    ref_list: List[Dict[str, Any]],
    modifications: Sequence[Dict[str, Any]],
    current_pic_num: int,
) -> List[Dict[str, Any]]:
    _ = modifications
    _ = current_pic_num
    return list(ref_list)


def get_preferred_ref_parity(current_is_bottom: bool) -> str:
    return "bottom" if current_is_bottom else "top"


def chroma_motion_comp_field(
    ref_chroma: np.ndarray,
    mv: np.ndarray,
    mb_x: int,
    mb_y: int,
    is_bottom_field: bool,
) -> np.ndarray:
    mv_i = np.asarray(mv, dtype=np.int32).reshape(2)
    x0 = int(mb_x) * 8 + int(mv_i[0] // 8)
    pair_y = int(mb_y)
    base_y = pair_y * 16 + int(mv_i[1] // 8)
    lines = [base_y + i * 2 + (1 if is_bottom_field else 0) for i in range(8)]
    out = np.zeros((8, 8), dtype=np.uint8)
    for i, y in enumerate(lines):
        if 0 <= y < ref_chroma.shape[0]:
            out[i, :] = ref_chroma[y, x0:x0 + 8]
    return out


def scale_chroma_mv_field(
    mv: np.ndarray,
    current_is_field: bool,
    ref_is_field: bool,
) -> np.ndarray:
    mv_i = np.asarray(mv, dtype=np.int32).reshape(2)
    if current_is_field and not ref_is_field:
        return np.array([mv_i[0], mv_i[1] // 2], dtype=np.int32)
    if (not current_is_field) and ref_is_field:
        return np.array([mv_i[0], mv_i[1] * 2], dtype=np.int32)
    return mv_i


def get_transform_size_mbaff(is_field: bool, transform_8x8_mode_flag: bool) -> Tuple[int, int]:
    if transform_8x8_mode_flag:
        return 8, 8
    return (4, 8) if is_field else (4, 4)


def get_idct_function_mbaff(is_field: bool):
    _ = is_field
    from transform import idct_4x4

    return idct_4x4


def get_dequant_scale_mbaff(qp: int, is_field: bool):
    _ = is_field
    from dequant import get_scale_matrix

    return get_scale_matrix(int(qp))


def get_intra_neighbors_mbaff(
    mb_addr: int,
    is_field: bool,
    is_bottom_mb: bool,
    frame_luma: np.ndarray,
) -> Dict[str, Any]:
    _ = mb_addr
    _ = is_field
    _ = is_bottom_mb
    _ = frame_luma
    return {"top": None, "left": None}


def intra_16x16_predict_field(
    mode: int,
    top: Optional[np.ndarray],
    left: Optional[np.ndarray],
    top_available: bool,
    left_available: bool,
) -> np.ndarray:
    if mode != 0:
        mode = 0
    if top_available and top is not None and left_available and left is not None:
        dc = (int(np.sum(np.asarray(top, dtype=np.int32))) + int(np.sum(np.asarray(left, dtype=np.int32))) + 16) >> 5
    elif top_available and top is not None:
        dc = (int(np.sum(np.asarray(top, dtype=np.int32))) + 8) >> 4
    elif left_available and left is not None:
        dc = (int(np.sum(np.asarray(left, dtype=np.int32))) + 8) >> 4
    else:
        dc = 128
    return np.full((16, 16), dc, dtype=np.uint8)


def get_intra_4x4_neighbors_mbaff(
    block_idx: int,
    mb_addr: int,
    current_is_field: bool,
    left_mb_is_field: bool,
) -> Dict[str, Any]:
    _ = block_idx
    _ = mb_addr
    _ = current_is_field
    _ = left_mb_is_field
    return {"top": None, "left": None, "top_left": None}


def get_ctx_idx_mb_field_flag(left_is_field: bool, top_is_field: bool) -> int:
    return (1 if left_is_field else 0) + (1 if top_is_field else 0)


def get_cabac_ctx_field_adaptation(element: str, is_field: bool) -> int:
    _ = element
    return 1 if is_field else 0


class MBAFFDecoder:
    def __init__(self):
        self.pair_field_flags: Dict[int, bool] = {}
        self.is_configured: bool = False
        self.header: Any = None

    def set_pair_field_flag(self, pair_idx: int, is_field: bool) -> None:
        self.pair_field_flags[int(pair_idx)] = bool(is_field)

    def is_pair_field_coded(self, pair_idx: int) -> bool:
        return bool(self.pair_field_flags.get(int(pair_idx), False))

    def parse_mb_field_decoding_flag(self, reader: Any, mb_addr: int, mbaff_frame_flag: bool) -> bool:
        return parse_mb_field_decoding_flag(
            reader=reader,
            mb_addr=mb_addr,
            mbaff_frame_flag=mbaff_frame_flag,
            pair_field_flags=self.pair_field_flags,
        )

    def infer_field_flag_for_skip(self, mb_addr: int, prev_pair_field_flag: bool) -> Optional[bool]:
        _ = mb_addr
        return bool(prev_pair_field_flag)

    def process_skip_run(self, start_mb_addr: int, skip_count: int) -> List[bool]:
        results: List[bool] = []
        for i in range(int(skip_count)):
            mb_addr = int(start_mb_addr) + i
            pair_idx = mb_addr_to_pair_idx(mb_addr)
            if pair_idx not in self.pair_field_flags:
                self.pair_field_flags[pair_idx] = self.pair_field_flags.get(pair_idx - 1, False)
            results.append(bool(self.pair_field_flags[pair_idx]))
        return results

    def decode_mbaff_pair(
        self,
        reader: Any,
        pair_idx: int,
        frame_luma: np.ndarray,
        frame_cb: np.ndarray,
        frame_cr: np.ndarray,
        qp: int,
    ) -> Dict[str, Any]:
        _ = frame_luma
        _ = frame_cb
        _ = frame_cr
        _ = qp
        top_mb_addr, bottom_mb_addr = get_pair_mb_addrs(pair_idx)
        is_field = self.parse_mb_field_decoding_flag(reader, top_mb_addr, mbaff_frame_flag=True)
        top_mb_type = reader.read_ue() if reader is not None else None
        bottom_mb_type = reader.read_ue() if reader is not None else None
        return {
            "pair_idx": int(pair_idx),
            "is_field": bool(is_field),
            "top_mb": {"mb_addr": top_mb_addr, "mb_type": top_mb_type},
            "bottom_mb": {"mb_addr": bottom_mb_addr, "mb_type": bottom_mb_type},
        }

    def decode_mbaff_frame(self, reader: Any, frame_width: int, frame_height: int) -> Dict[str, Any]:
        _ = reader
        w = int(frame_width)
        h = int(frame_height)
        luma = np.zeros((h * 16, w * 16), dtype=np.uint8)
        cb = np.zeros((h * 8, w * 8), dtype=np.uint8)
        cr = np.zeros((h * 8, w * 8), dtype=np.uint8)
        return {"luma": luma, "cb": cb, "cr": cr}

    def configure_from_header(self, header: Any) -> None:
        self.header = header
        self.is_configured = True

