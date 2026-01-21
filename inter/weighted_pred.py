# h264/inter/weighted_pred.py
"""Weighted prediction for inter prediction.

H.264 supports weighted prediction to handle fades, exposure changes, etc:
    pred' = ((w * pred + 2^(ld-1)) >> ld) + o

where w=weight, o=offset, ld=log2_weight_denom

H.264 Spec Reference: Section 7.4.3.2 - Weighted prediction semantics
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class WeightTable:
    """Weight table for weighted prediction.

    Stores per-reference weights and offsets for luma and chroma.
    """

    luma_log2_weight_denom: int = 6
    chroma_log2_weight_denom: int = 6

    # Per-reference weights and offsets (indexed by ref_idx)
    _luma_weights: List[int] = field(default_factory=list)
    _luma_offsets: List[int] = field(default_factory=list)
    _cb_weights: List[int] = field(default_factory=list)
    _cb_offsets: List[int] = field(default_factory=list)
    _cr_weights: List[int] = field(default_factory=list)
    _cr_offsets: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Initialize weight lists."""
        if not self._luma_weights:
            self._luma_weights = []
            self._luma_offsets = []
            self._cb_weights = []
            self._cb_offsets = []
            self._cr_weights = []
            self._cr_offsets = []

    def _ensure_ref_idx(self, ref_idx: int) -> None:
        """Ensure lists are large enough for ref_idx."""
        while len(self._luma_weights) <= ref_idx:
            # Default weight = 2^log2_denom (1.0 scaling)
            self._luma_weights.append(1 << self.luma_log2_weight_denom)
            self._luma_offsets.append(0)
            self._cb_weights.append(1 << self.chroma_log2_weight_denom)
            self._cb_offsets.append(0)
            self._cr_weights.append(1 << self.chroma_log2_weight_denom)
            self._cr_offsets.append(0)

    def set_luma_weight(self, ref_idx: int, weight: int, offset: int) -> None:
        """Set luma weight and offset for a reference."""
        self._ensure_ref_idx(ref_idx)
        self._luma_weights[ref_idx] = weight
        self._luma_offsets[ref_idx] = offset

    def get_luma_weight(self, ref_idx: int) -> Tuple[int, int]:
        """Get luma weight and offset for a reference."""
        self._ensure_ref_idx(ref_idx)
        return self._luma_weights[ref_idx], self._luma_offsets[ref_idx]

    def set_chroma_weight(
        self, ref_idx: int, weight_cb: int, offset_cb: int, weight_cr: int, offset_cr: int
    ) -> None:
        """Set chroma weights and offsets for a reference."""
        self._ensure_ref_idx(ref_idx)
        self._cb_weights[ref_idx] = weight_cb
        self._cb_offsets[ref_idx] = offset_cb
        self._cr_weights[ref_idx] = weight_cr
        self._cr_offsets[ref_idx] = offset_cr

    def get_chroma_weight(self, ref_idx: int) -> Tuple[int, int, int, int]:
        """Get chroma weights and offsets for a reference."""
        self._ensure_ref_idx(ref_idx)
        return (
            self._cb_weights[ref_idx],
            self._cb_offsets[ref_idx],
            self._cr_weights[ref_idx],
            self._cr_offsets[ref_idx],
        )

    @property
    def luma_weight(self) -> List[int]:
        """List of luma weights (for compatibility)."""
        return self._luma_weights

    @property
    def luma_offset(self) -> List[int]:
        """List of luma offsets (for compatibility)."""
        return self._luma_offsets


def parse_pred_weight_table(reader, num_ref_idx_l0: int) -> WeightTable:
    """Parse pred_weight_table from slice header bitstream.

    Args:
        reader: BitReader positioned at pred_weight_table
        num_ref_idx_l0: Number of L0 reference indices

    Returns:
        WeightTable with parsed weights and offsets

    H.264 Spec: Section 7.3.3.2
    """
    # Read denominators
    luma_log2_weight_denom = reader.read_ue()
    chroma_log2_weight_denom = reader.read_ue()

    table = WeightTable(
        luma_log2_weight_denom=luma_log2_weight_denom,
        chroma_log2_weight_denom=chroma_log2_weight_denom,
    )

    # Default weights (1.0 scaling)
    default_luma_weight = 1 << luma_log2_weight_denom
    default_chroma_weight = 1 << chroma_log2_weight_denom

    # Parse L0 weights
    for i in range(num_ref_idx_l0):
        luma_weight_flag = reader.read_bits(1)

        if luma_weight_flag:
            luma_weight = reader.read_se()
            luma_offset = reader.read_se()
            table.set_luma_weight(i, luma_weight, luma_offset)
        else:
            table.set_luma_weight(i, default_luma_weight, 0)

        chroma_weight_flag = reader.read_bits(1)

        if chroma_weight_flag:
            cb_weight = reader.read_se()
            cb_offset = reader.read_se()
            cr_weight = reader.read_se()
            cr_offset = reader.read_se()
            table.set_chroma_weight(i, cb_weight, cb_offset, cr_weight, cr_offset)
        else:
            table.set_chroma_weight(
                i, default_chroma_weight, 0, default_chroma_weight, 0
            )

    return table


def apply_weighted_prediction(
    pred: np.ndarray,
    weight: int,
    offset: int,
    log2_denom: int,
) -> np.ndarray:
    """Apply weighted prediction to a prediction block.

    Formula: result = ((weight * pred + 2^(log2_denom-1)) >> log2_denom) + offset

    Args:
        pred: Prediction block
        weight: Weight value
        offset: Offset value
        log2_denom: Log2 of weight denominator

    Returns:
        Weighted prediction clipped to [0, 255]

    H.264 Spec: Section 8.4.2.3.1
    """
    # Compute with sufficient precision
    rounding = 1 << (log2_denom - 1) if log2_denom > 0 else 0

    result = pred.astype(np.int32) * weight + rounding
    result = (result >> log2_denom) + offset

    # Clip to valid range
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_weighted_prediction_chroma(
    pred_cb: np.ndarray,
    pred_cr: np.ndarray,
    weight_cb: int,
    offset_cb: int,
    weight_cr: int,
    offset_cr: int,
    log2_denom: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply weighted prediction to chroma planes.

    Args:
        pred_cb: Cb prediction block
        pred_cr: Cr prediction block
        weight_cb: Cb weight
        offset_cb: Cb offset
        weight_cr: Cr weight
        offset_cr: Cr offset
        log2_denom: Log2 of chroma weight denominator

    Returns:
        Tuple of (weighted_cb, weighted_cr)
    """
    cb_result = apply_weighted_prediction(pred_cb, weight_cb, offset_cb, log2_denom)
    cr_result = apply_weighted_prediction(pred_cr, weight_cr, offset_cr, log2_denom)
    return cb_result, cr_result


def calc_implicit_weights(
    current_poc: int,
    ref0_poc: int,
    ref1_poc: int,
) -> Tuple[int, int]:
    """Calculate implicit weights from POC distances.

    Used for implicit weighted biprediction (weighted_bipred_idc=2).

    Formula:
        tb = current_poc - ref0_poc
        td = ref1_poc - ref0_poc
        w1 = (tb * 64 + td/2) / td  (rounded)
        w0 = 64 - w1

    Args:
        current_poc: POC of current picture
        ref0_poc: POC of L0 reference
        ref1_poc: POC of L1 reference

    Returns:
        Tuple of (w0, w1) weights in 1/64 scale

    H.264 Spec: Section 8.4.2.3.2
    """
    tb = current_poc - ref0_poc
    td = ref1_poc - ref0_poc

    if td == 0:
        # Fallback for same POC (shouldn't happen normally)
        return 32, 32

    # Calculate w1 with rounding
    # w1 = (tb * 64 + abs(td)/2) / td
    w1 = (tb * 64 + (abs(td) >> 1)) // td

    # Clamp to valid range [-64, 128]
    w1 = max(-64, min(128, w1))

    w0 = 64 - w1

    return w0, w1
