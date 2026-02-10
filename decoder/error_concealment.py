from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from inter.mv_prediction import MVCache
from inter.reference import ReferenceFrame, ReferenceFrameBuffer


class ConcealmentStrategy(Enum):
    TEMPORAL = auto()
    SPATIAL = auto()
    ZERO = auto()
    PREVIOUS_MB = auto()


def select_concealment_strategy(
    ref_buffer: Optional[ReferenceFrameBuffer],
    has_neighbors: bool,
) -> ConcealmentStrategy:
    if ref_buffer is not None and len(ref_buffer) > 0:
        return ConcealmentStrategy.TEMPORAL
    if has_neighbors:
        return ConcealmentStrategy.SPATIAL
    return ConcealmentStrategy.ZERO


def detect_missing_macroblocks(expected_mbs: int, decoded_mbs: Sequence[int]) -> List[int]:
    expected = set(range(int(expected_mbs)))
    decoded = set(int(x) for x in decoded_mbs)
    missing = sorted(expected - decoded)
    return missing


def is_macroblock_corrupt(coeffs: np.ndarray, *, max_abs_coeff: int = 4096) -> bool:
    if coeffs is None:
        return True
    arr = np.asarray(coeffs)
    if np.issubdtype(arr.dtype, np.floating):
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return True
    try:
        return bool(np.any(np.abs(arr) > int(max_abs_coeff)))
    except Exception:
        return True


def is_mb_type_valid(
    mb_type: int,
    *,
    is_i_slice: bool,
    is_p_slice: bool = False,
) -> bool:
    try:
        mb_type_i = int(mb_type)
    except Exception:
        return False

    if mb_type_i < 0:
        return False

    if is_i_slice:
        return 0 <= mb_type_i <= 25

    if is_p_slice:
        return 0 <= mb_type_i <= 30

    return False


def _extract_block(
    plane: Optional[np.ndarray],
    x: int,
    y: int,
    h: int,
    w: int,
    *,
    fill: int = 128,
) -> np.ndarray:
    out = np.full((h, w), int(fill), dtype=np.uint8)
    if plane is None:
        return out

    img = np.asarray(plane)
    if img.ndim != 2:
        return out

    x0 = max(int(x), 0)
    y0 = max(int(y), 0)
    x1 = min(int(x) + int(w), img.shape[1])
    y1 = min(int(y) + int(h), img.shape[0])

    if x0 >= x1 or y0 >= y1:
        return out

    out_y0 = y0 - int(y)
    out_x0 = x0 - int(x)
    out[out_y0:out_y0 + (y1 - y0), out_x0:out_x0 + (x1 - x0)] = img[y0:y1, x0:x1].astype(
        np.uint8, copy=False
    )
    return out


def conceal_macroblock_temporal(
    *,
    ref_frame: ReferenceFrame,
    mb_x: int,
    mb_y: int,
    mvx: int = 0,
    mvy: int = 0,
) -> Dict[str, np.ndarray]:
    off_x = int(mvx / 4)
    off_y = int(mvy / 4)

    luma_x = int(mb_x) * 16 + off_x
    luma_y = int(mb_y) * 16 + off_y
    luma = _extract_block(ref_frame.luma, luma_x, luma_y, 16, 16, fill=128)

    chroma_off_x = int(off_x / 2)
    chroma_off_y = int(off_y / 2)
    cb_x = int(mb_x) * 8 + chroma_off_x
    cb_y = int(mb_y) * 8 + chroma_off_y
    cb = _extract_block(ref_frame.cb, cb_x, cb_y, 8, 8, fill=128)
    cr = _extract_block(ref_frame.cr, cb_x, cb_y, 8, 8, fill=128)

    return {"luma": luma, "cb": cb, "cr": cr}


def conceal_macroblock_spatial(
    *,
    frame_luma: np.ndarray,
    frame_cb: np.ndarray,
    frame_cr: np.ndarray,
    mb_x: int,
    mb_y: int,
    neighbors: Dict[str, bool],
) -> Dict[str, np.ndarray]:
    top_avail = bool(neighbors.get("top_available"))
    left_avail = bool(neighbors.get("left_available"))
    bottom_avail = bool(neighbors.get("bottom_available"))
    right_avail = bool(neighbors.get("right_available"))

    vals: List[float] = []

    top_val = None
    left_val = None
    bottom_val = None
    right_val = None

    if top_avail:
        top_val = float(
            np.mean(_extract_block(frame_luma, int(mb_x) * 16, (int(mb_y) - 1) * 16, 16, 16, fill=128))
        )
        vals.append(top_val)
    if left_avail:
        left_val = float(
            np.mean(_extract_block(frame_luma, (int(mb_x) - 1) * 16, int(mb_y) * 16, 16, 16, fill=128))
        )
        vals.append(left_val)
    if bottom_avail:
        bottom_val = float(
            np.mean(_extract_block(frame_luma, int(mb_x) * 16, (int(mb_y) + 1) * 16, 16, 16, fill=128))
        )
        vals.append(bottom_val)
    if right_avail:
        right_val = float(
            np.mean(_extract_block(frame_luma, (int(mb_x) + 1) * 16, int(mb_y) * 16, 16, 16, fill=128))
        )
        vals.append(right_val)

    if not vals:
        luma = np.full((16, 16), 128, dtype=np.uint8)
    elif left_avail and right_avail and not top_avail and not bottom_avail:
        lv = float(left_val if left_val is not None else 128.0)
        rv = float(right_val if right_val is not None else 128.0)
        ramp = np.linspace(lv, rv, 16, dtype=np.float64)
        luma = np.tile(ramp[None, :], (16, 1)).clip(0, 255).astype(np.uint8)
    elif top_avail and bottom_avail and not left_avail and not right_avail:
        tv = float(top_val if top_val is not None else 128.0)
        bv = float(bottom_val if bottom_val is not None else 128.0)
        ramp = np.linspace(tv, bv, 16, dtype=np.float64)
        luma = np.tile(ramp[:, None], (1, 16)).clip(0, 255).astype(np.uint8)
    else:
        avg = float(np.mean(vals))
        luma = np.full((16, 16), int(round(avg)), dtype=np.uint8)

    cb = np.full((8, 8), 128, dtype=np.uint8)
    cr = np.full((8, 8), 128, dtype=np.uint8)
    return {"luma": luma, "cb": cb, "cr": cr}


def conceal_motion_vector(
    *,
    mv_cache: MVCache,
    mb_x: int,
    mb_y: int,
    colocated_ref: Optional[ReferenceFrame] = None,
    use_colocated: bool = False,
) -> Tuple[int, int]:
    if use_colocated and colocated_ref is not None:
        mv = colocated_ref.get_colocated_mv(int(mb_x), int(mb_y))
        if mv != (0, 0):
            return int(mv[0]), int(mv[1])

    candidates: List[Tuple[int, int]] = []

    for nx, ny in (
        (int(mb_x) - 1, int(mb_y)),
        (int(mb_x), int(mb_y) - 1),
        (int(mb_x) + 1, int(mb_y) - 1),
    ):
        if mv_cache.is_available(nx, ny, 0, 0):
            candidates.append(mv_cache.get_mv(nx, ny, 0, 0))

    if not candidates:
        return (0, 0)
    if len(candidates) == 1:
        return int(candidates[0][0]), int(candidates[0][1])
    if len(candidates) == 2:
        ax = int(round((candidates[0][0] + candidates[1][0]) / 2))
        ay = int(round((candidates[0][1] + candidates[1][1]) / 2))
        return ax, ay

    xs = sorted(int(mv[0]) for mv in candidates[:3])
    ys = sorted(int(mv[1]) for mv in candidates[:3])
    return xs[1], ys[1]


def conceal_lost_slice(
    *,
    frame_luma: np.ndarray,
    frame_cb: np.ndarray,
    frame_cr: np.ndarray,
    ref_buffer: Optional[ReferenceFrameBuffer],
    first_mb: int,
    last_mb: int,
    mb_width: int,
) -> None:
    ref_frame = None
    if ref_buffer is not None and len(ref_buffer) > 0:
        ref_frame = ref_buffer.get_frame(0)

    for mb_addr in range(int(first_mb), int(last_mb) + 1):
        mb_x = int(mb_addr) % int(mb_width)
        mb_y = int(mb_addr) // int(mb_width)

        if ref_frame is not None:
            mb = conceal_macroblock_temporal(ref_frame=ref_frame, mb_x=mb_x, mb_y=mb_y)
        else:
            mb = {
                "luma": np.full((16, 16), 128, dtype=np.uint8),
                "cb": np.full((8, 8), 128, dtype=np.uint8),
                "cr": np.full((8, 8), 128, dtype=np.uint8),
            }

        y0 = mb_y * 16
        x0 = mb_x * 16
        frame_luma[y0:y0 + 16, x0:x0 + 16] = mb["luma"]

        y0c = mb_y * 8
        x0c = mb_x * 8
        frame_cb[y0c:y0c + 8, x0c:x0c + 8] = mb["cb"]
        frame_cr[y0c:y0c + 8, x0c:x0c + 8] = mb["cr"]


def conceal_partial_slice(
    *,
    frame_luma: np.ndarray,
    frame_cb: np.ndarray,
    frame_cr: np.ndarray,
    ref_buffer: Optional[ReferenceFrameBuffer],
    decoded_mbs: Sequence[int],
    slice_mbs: Sequence[int],
    mb_width: int,
) -> None:
    decoded = set(int(x) for x in decoded_mbs)
    missing = [int(x) for x in slice_mbs if int(x) not in decoded]

    ref_frame = None
    if ref_buffer is not None and len(ref_buffer) > 0:
        ref_frame = ref_buffer.get_frame(0)

    for mb_addr in missing:
        mb_x = int(mb_addr) % int(mb_width)
        mb_y = int(mb_addr) // int(mb_width)

        if ref_frame is not None:
            mb = conceal_macroblock_temporal(ref_frame=ref_frame, mb_x=mb_x, mb_y=mb_y)
        else:
            mb = {
                "luma": np.full((16, 16), 128, dtype=np.uint8),
                "cb": np.full((8, 8), 128, dtype=np.uint8),
                "cr": np.full((8, 8), 128, dtype=np.uint8),
            }

        y0 = mb_y * 16
        x0 = mb_x * 16
        frame_luma[y0:y0 + 16, x0:x0 + 16] = mb["luma"]

        y0c = mb_y * 8
        x0c = mb_x * 8
        frame_cb[y0c:y0c + 8, x0c:x0c + 8] = mb["cb"]
        frame_cr[y0c:y0c + 8, x0c:x0c + 8] = mb["cr"]


def sanitize_coefficients(coeffs: np.ndarray, *, max_coeff: int = 2047) -> np.ndarray:
    arr = np.asarray(coeffs, dtype=np.float64)
    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=float(max_coeff),
        neginf=float(-int(max_coeff) - 1),
    )

    lo = -int(max_coeff) - 1
    hi = int(max_coeff)
    arr = np.clip(arr, lo, hi)
    return arr.astype(np.int32)


def repair_dc_coefficient(coeffs: np.ndarray, neighbor_dcs: Sequence[int]) -> np.ndarray:
    out = np.array(coeffs, copy=True)
    if neighbor_dcs:
        pred = int(round(float(np.mean([int(x) for x in neighbor_dcs]))))
    else:
        pred = 0
    out[0, 0] = pred
    return out


def is_nal_truncated(nal_data: bytes) -> bool:
    if not nal_data:
        return True

    i = len(nal_data)
    while i > 0 and nal_data[i - 1] == 0:
        i -= 1
    if i == 0:
        return True

    return nal_data[i - 1] != 0x80


def estimate_truncation_extent(
    nal_data: bytes,
    *,
    expected_mbs: int,
    decoded_mbs: int,
) -> Dict[str, Any]:
    missing_mbs = max(0, int(expected_mbs) - int(decoded_mbs))
    estimated_bytes = missing_mbs * 32
    return {"missing_mbs": missing_mbs, "estimated_bytes": estimated_bytes}


def decode_truncated_slice(nal_data: bytes) -> Dict[str, Any]:
    return {
        "decoded_mbs": 0,
        "error_position": len(nal_data),
        "partial_decode": True,
    }


def recover_invalid_mb_type(
    mb_type: int,
    *,
    is_i_slice: bool,
    is_p_slice: bool = False,
) -> int:
    if is_mb_type_valid(mb_type, is_i_slice=is_i_slice, is_p_slice=is_p_slice):
        return int(mb_type)
    return 0


def recover_invalid_cbp(cbp: int) -> int:
    try:
        cbp_i = int(cbp)
    except Exception:
        return 0
    if cbp_i < 0:
        return 0
    if cbp_i > 47:
        return 47
    return cbp_i


def recover_invalid_intra_mode(mode: int, *, block_size: int) -> int:
    try:
        mode_i = int(mode)
    except Exception:
        return 0
    if mode_i < 0:
        return 0
    max_mode = 8
    if mode_i > max_mode:
        return max_mode
    return mode_i


def recover_invalid_mv(
    mvx: int,
    mvy: int,
    *,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int]:
    _ = int(frame_width)
    _ = int(frame_height)

    def clamp(v: int) -> int:
        return max(-8192, min(8191, int(v)))

    return clamp(mvx), clamp(mvy)


def auto_select_concealment(
    *,
    is_i_slice: bool,
    ref_buffer: Optional[ReferenceFrameBuffer],
    mb_x: int,
    mb_y: int,
    has_neighbors: bool,
    mv_cache: Optional[MVCache] = None,
) -> ConcealmentStrategy:
    _ = (int(mb_x), int(mb_y), mv_cache)

    if is_i_slice:
        if has_neighbors:
            return ConcealmentStrategy.SPATIAL
        return ConcealmentStrategy.ZERO

    if ref_buffer is not None and len(ref_buffer) > 0:
        return ConcealmentStrategy.TEMPORAL

    if has_neighbors:
        return ConcealmentStrategy.SPATIAL

    return ConcealmentStrategy.ZERO


def conceal_macroblock(
    *,
    strategy: ConcealmentStrategy,
    mb_x: int,
    mb_y: int,
    ref_buffer: Optional[ReferenceFrameBuffer] = None,
    frame_luma: Optional[np.ndarray] = None,
    frame_cb: Optional[np.ndarray] = None,
    frame_cr: Optional[np.ndarray] = None,
    neighbors: Optional[Dict[str, bool]] = None,
    mv_cache: Optional[MVCache] = None,
    prev_mb: Optional[Dict[str, np.ndarray]] = None,
    mvx: int = 0,
    mvy: int = 0,
) -> Dict[str, np.ndarray]:
    _ = mv_cache

    if strategy == ConcealmentStrategy.TEMPORAL:
        if ref_buffer is None or len(ref_buffer) == 0:
            strategy = ConcealmentStrategy.ZERO
        else:
            ref_frame = ref_buffer.get_frame(0)
            return conceal_macroblock_temporal(
                ref_frame=ref_frame,
                mb_x=int(mb_x),
                mb_y=int(mb_y),
                mvx=int(mvx),
                mvy=int(mvy),
            )

    if strategy == ConcealmentStrategy.SPATIAL:
        if frame_luma is None or frame_cb is None or frame_cr is None or neighbors is None:
            strategy = ConcealmentStrategy.ZERO
        else:
            return conceal_macroblock_spatial(
                frame_luma=frame_luma,
                frame_cb=frame_cb,
                frame_cr=frame_cr,
                mb_x=int(mb_x),
                mb_y=int(mb_y),
                neighbors=neighbors,
            )

    if strategy == ConcealmentStrategy.PREVIOUS_MB:
        if prev_mb is not None and all(k in prev_mb for k in ("luma", "cb", "cr")):
            return {
                "luma": np.array(prev_mb["luma"], copy=True),
                "cb": np.array(prev_mb["cb"], copy=True),
                "cr": np.array(prev_mb["cr"], copy=True),
            }
        strategy = ConcealmentStrategy.ZERO

    return {
        "luma": np.full((16, 16), 128, dtype=np.uint8),
        "cb": np.full((8, 8), 128, dtype=np.uint8),
        "cr": np.full((8, 8), 128, dtype=np.uint8),
    }
