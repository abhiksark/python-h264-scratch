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


def apply_weighted_bipred(
    pred_l0: np.ndarray,
    pred_l1: np.ndarray,
    w0: int,
    o0: int,
    w1: int,
    o1: int,
    log2_denom: int,
) -> np.ndarray:
    """Apply weighted biprediction to L0 and L1 predictions.

    Formula: result = ((w0 * pred_l0 + w1 * pred_l1 + 2^log2_denom) >> (log2_denom+1)) + (o0+o1+1)/2

    Args:
        pred_l0: L0 prediction block
        pred_l1: L1 prediction block
        w0: L0 weight
        o0: L0 offset
        w1: L1 weight
        o1: L1 offset
        log2_denom: Log2 of weight denominator

    Returns:
        Weighted biprediction result clipped to [0, 255]

    H.264 Spec: Section 8.4.2.3.2
    """
    rounding = 1 << log2_denom

    result = (pred_l0.astype(np.int32) * w0 +
              pred_l1.astype(np.int32) * w1 + rounding)
    result = (result >> (log2_denom + 1)) + ((o0 + o1 + 1) >> 1)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_implicit_weighted_bipred(
    pred_l0: np.ndarray,
    pred_l1: np.ndarray,
    current_poc: int,
    l0_poc: int,
    l1_poc: int,
) -> np.ndarray:
    """Apply implicit weighted biprediction using POC distances.

    Args:
        pred_l0: L0 prediction block
        pred_l1: L1 prediction block
        current_poc: POC of current picture
        l0_poc: POC of L0 reference
        l1_poc: POC of L1 reference

    Returns:
        Weighted biprediction result
    """
    w0, w1 = calc_implicit_weights(current_poc, l0_poc, l1_poc)

    # Implicit weights use log2_denom=5 (scale of 64)
    rounding = 1 << 5  # 32

    result = (pred_l0.astype(np.int32) * w0 +
              pred_l1.astype(np.int32) * w1 + rounding)
    result = result >> 6  # Divide by 64

    return np.clip(result, 0, 255).astype(np.uint8)


@dataclass
class WeightTableBSlice:
    """Weight table for B-slice with separate L0 and L1 weights.

    B-slices can have different weights for L0 and L1 references.
    """

    luma_log2_weight_denom: int = 6
    chroma_log2_weight_denom: int = 6

    # L0 weights/offsets
    _l0_luma_weights: List[int] = field(default_factory=list)
    _l0_luma_offsets: List[int] = field(default_factory=list)
    _l0_cb_weights: List[int] = field(default_factory=list)
    _l0_cb_offsets: List[int] = field(default_factory=list)
    _l0_cr_weights: List[int] = field(default_factory=list)
    _l0_cr_offsets: List[int] = field(default_factory=list)

    # L1 weights/offsets
    _l1_luma_weights: List[int] = field(default_factory=list)
    _l1_luma_offsets: List[int] = field(default_factory=list)
    _l1_cb_weights: List[int] = field(default_factory=list)
    _l1_cb_offsets: List[int] = field(default_factory=list)
    _l1_cr_weights: List[int] = field(default_factory=list)
    _l1_cr_offsets: List[int] = field(default_factory=list)

    def __post_init__(self):
        """Initialize weight lists."""
        if not self._l0_luma_weights:
            self._l0_luma_weights = []
            self._l0_luma_offsets = []
            self._l0_cb_weights = []
            self._l0_cb_offsets = []
            self._l0_cr_weights = []
            self._l0_cr_offsets = []
            self._l1_luma_weights = []
            self._l1_luma_offsets = []
            self._l1_cb_weights = []
            self._l1_cb_offsets = []
            self._l1_cr_weights = []
            self._l1_cr_offsets = []

    def _ensure_l0_ref_idx(self, ref_idx: int) -> None:
        """Ensure L0 lists are large enough."""
        while len(self._l0_luma_weights) <= ref_idx:
            self._l0_luma_weights.append(1 << self.luma_log2_weight_denom)
            self._l0_luma_offsets.append(0)
            self._l0_cb_weights.append(1 << self.chroma_log2_weight_denom)
            self._l0_cb_offsets.append(0)
            self._l0_cr_weights.append(1 << self.chroma_log2_weight_denom)
            self._l0_cr_offsets.append(0)

    def _ensure_l1_ref_idx(self, ref_idx: int) -> None:
        """Ensure L1 lists are large enough."""
        while len(self._l1_luma_weights) <= ref_idx:
            self._l1_luma_weights.append(1 << self.luma_log2_weight_denom)
            self._l1_luma_offsets.append(0)
            self._l1_cb_weights.append(1 << self.chroma_log2_weight_denom)
            self._l1_cb_offsets.append(0)
            self._l1_cr_weights.append(1 << self.chroma_log2_weight_denom)
            self._l1_cr_offsets.append(0)

    def set_l0_luma_weight(self, ref_idx: int, weight: int, offset: int) -> None:
        """Set L0 luma weight and offset."""
        self._ensure_l0_ref_idx(ref_idx)
        self._l0_luma_weights[ref_idx] = weight
        self._l0_luma_offsets[ref_idx] = offset

    def get_l0_luma_weight(self, ref_idx: int) -> Tuple[int, int]:
        """Get L0 luma weight and offset."""
        self._ensure_l0_ref_idx(ref_idx)
        return self._l0_luma_weights[ref_idx], self._l0_luma_offsets[ref_idx]

    def set_l1_luma_weight(self, ref_idx: int, weight: int, offset: int) -> None:
        """Set L1 luma weight and offset."""
        self._ensure_l1_ref_idx(ref_idx)
        self._l1_luma_weights[ref_idx] = weight
        self._l1_luma_offsets[ref_idx] = offset

    def get_l1_luma_weight(self, ref_idx: int) -> Tuple[int, int]:
        """Get L1 luma weight and offset."""
        self._ensure_l1_ref_idx(ref_idx)
        return self._l1_luma_weights[ref_idx], self._l1_luma_offsets[ref_idx]


def get_bipred_weights(
    weighted_bipred_idc: int,
    weight_table_l0: Optional[WeightTable],
    weight_table_l1: Optional[WeightTable],
    ref_idx_l0: int,
    ref_idx_l1: int,
    current_poc: int = 0,
    l0_poc: int = 0,
    l1_poc: int = 0,
    log2_denom: int = 6,
) -> Tuple[int, int, int, int]:
    """Get weights and offsets for biprediction based on mode.

    Args:
        weighted_bipred_idc: Bipred mode (0=default, 1=explicit, 2=implicit)
        weight_table_l0: L0 weight table (for idc=1)
        weight_table_l1: L1 weight table (for idc=1)
        ref_idx_l0: L0 reference index
        ref_idx_l1: L1 reference index
        current_poc: Current picture POC (for idc=2)
        l0_poc: L0 reference POC (for idc=2)
        l1_poc: L1 reference POC (for idc=2)
        log2_denom: Weight denominator (for defaults)

    Returns:
        Tuple of (w0, o0, w1, o1)
    """
    default_weight = 1 << log2_denom

    if weighted_bipred_idc == 0:
        # Default: equal weights, no offset
        return default_weight, 0, default_weight, 0

    elif weighted_bipred_idc == 1:
        # Explicit: use weight tables
        if weight_table_l0 is not None:
            w0, o0 = weight_table_l0.get_luma_weight(ref_idx_l0)
        else:
            w0, o0 = default_weight, 0

        if weight_table_l1 is not None:
            w1, o1 = weight_table_l1.get_luma_weight(ref_idx_l1)
        else:
            w1, o1 = default_weight, 0

        return w0, o0, w1, o1

    elif weighted_bipred_idc == 2:
        # Implicit: calculate from POC
        w0, w1 = calc_implicit_weights(current_poc, l0_poc, l1_poc)
        return w0, 0, w1, 0

    else:
        return default_weight, 0, default_weight, 0


def validate_log2_weight_denom(log2_denom: int) -> bool:
    """Validate log2_weight_denom is in valid range [0, 7].

    H.264 Spec: Section 7.4.3.2

    Args:
        log2_denom: Log2 weight denominator

    Returns:
        True if valid

    Raises:
        ValueError: If out of range
    """
    if not 0 <= log2_denom <= 7:
        raise ValueError(f"log2_weight_denom {log2_denom} out of range [0, 7]")
    return True


def validate_weight_offset(weight: int, offset: int, bit_depth: int = 8) -> bool:
    """Validate weight and offset values.

    H.264 Spec: Section 7.4.3.2
    Weight: [-128, 127]
    Offset: depends on bit depth

    Args:
        weight: Weight value
        offset: Offset value
        bit_depth: Sample bit depth (8 or 10)

    Returns:
        True if valid

    Raises:
        ValueError: If out of range
    """
    if not -128 <= weight <= 127:
        raise ValueError(f"Weight {weight} out of range [-128, 127]")

    # Offset range depends on bit depth
    max_offset = 1 << (bit_depth - 1)  # 128 for 8-bit, 512 for 10-bit
    if not -max_offset <= offset < max_offset:
        raise ValueError(f"Offset {offset} out of range for {bit_depth}-bit")

    return True


def calc_implicit_weights_spec(
    current_poc: int,
    l0_poc: int,
    l1_poc: int,
) -> Tuple[int, int]:
    """Calculate implicit weights per H.264 spec exactly.

    H.264 Spec: Section 8.4.2.3.2, equations 8-209 through 8-214

    Args:
        current_poc: POC of current picture
        l0_poc: POC of L0 reference
        l1_poc: POC of L1 reference

    Returns:
        Tuple of (w0, w1)
    """
    tb = max(-128, min(127, current_poc - l0_poc))
    td = max(-128, min(127, l1_poc - l0_poc))

    if td == 0 or abs(td) > 128 or abs(tb) > 128:
        # Fallback to equal weights
        return 32, 32

    tx = (16384 + abs(td // 2)) // td
    dist_scale_factor = max(-1024, min(1023, (tb * tx + 32) >> 6))

    w1 = dist_scale_factor >> 2
    w0 = 64 - w1

    return w0, w1


def apply_weighted_prediction_partition(
    pred: np.ndarray,
    ref_idx: int = 0,
    weight_table: WeightTable = None,
    is_chroma: bool = False,
    weights: dict = None,
    log2_denom: int = 6,
) -> np.ndarray:
    """Apply weighted prediction for a partition using weight table.

    Args:
        pred: Prediction block
        ref_idx: Reference index
        weight_table: Weight table with per-ref weights (legacy)
        is_chroma: True for chroma planes (with weight_table)
        weights: Dict mapping ref_idx to (weight, offset)
        log2_denom: Weight denominator (with weights dict)

    Returns:
        Weighted prediction
    """
    if weights is not None:
        # Use dict-based weights
        w, o = weights.get(ref_idx, (1 << log2_denom, 0))
        return apply_weighted_prediction(pred, w, o, log2_denom)

    if weight_table is not None:
        if is_chroma:
            w, o, _, _ = weight_table.get_chroma_weight(ref_idx)
            denom = weight_table.chroma_log2_weight_denom
        else:
            w, o = weight_table.get_luma_weight(ref_idx)
            denom = weight_table.luma_log2_weight_denom
        return apply_weighted_prediction(pred, w, o, denom)

    # Fallback to default (unity weight)
    return pred.copy()


def get_default_weight(log2_denom: int) -> Tuple[int, int]:
    """Get default weight (1.0 scaling) for given denominator.

    When weighted_pred_flag=0, use default weight (2^log2_denom) with offset 0.

    Args:
        log2_denom: Log2 weight denominator

    Returns:
        Tuple of (weight, offset) where weight=2^log2_denom and offset=0
    """
    return (1 << log2_denom, 0)


def apply_weighted_prediction_10bit(
    pred: np.ndarray,
    weight: int,
    offset: int,
    log2_denom: int,
    bit_depth: int = 10,
) -> np.ndarray:
    """Apply weighted prediction for 10-bit content.

    Args:
        pred: Prediction block
        weight: Weight value
        offset: Offset value
        log2_denom: Log2 of weight denominator
        bit_depth: Bit depth (10)

    Returns:
        Weighted prediction clipped to [0, 2^bit_depth - 1]
    """
    rounding = 1 << (log2_denom - 1) if log2_denom > 0 else 0
    max_val = (1 << bit_depth) - 1

    result = pred.astype(np.int32) * weight + rounding
    result = (result >> log2_denom) + offset

    return np.clip(result, 0, max_val).astype(np.uint16)


def get_weight_offset_range(bit_depth: int = 8) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get valid ranges for weight and offset values.

    H.264 Spec: Weight and offset ranges scale with bit depth.

    Args:
        bit_depth: Sample bit depth (8 or 10)

    Returns:
        Tuple of ((weight_min, weight_max), (offset_min, offset_max))
    """
    # Weight and offset ranges scale with bit depth
    max_val = 1 << (bit_depth - 1)  # 128 for 8-bit, 512 for 10-bit
    weight_range = (-max_val, max_val - 1)
    offset_range = (-max_val, max_val - 1)

    return weight_range, offset_range


def parse_pred_weight_table_b_slice(
    reader,
    num_ref_idx_l0: int,
    num_ref_idx_l1: int,
    chroma_array_type: int = 1,
) -> Tuple[WeightTable, WeightTable]:
    """Parse prediction weight table for B-slices.

    B-slices have separate weight tables for L0 and L1 references.

    H.264 Spec: Section 7.3.3.2

    Args:
        reader: BitReader positioned at pred_weight_table
        num_ref_idx_l0: Number of L0 references
        num_ref_idx_l1: Number of L1 references
        chroma_array_type: Chroma format (0=monochrome, 1=4:2:0, 2=4:2:2, 3=4:4:4)

    Returns:
        Tuple of (table_l0, table_l1) WeightTable objects
    """
    luma_log2_denom = reader.read_ue()
    if chroma_array_type != 0:
        chroma_log2_denom = reader.read_ue()
    else:
        chroma_log2_denom = 0

    table_l0 = WeightTable(
        luma_log2_weight_denom=luma_log2_denom,
        chroma_log2_weight_denom=chroma_log2_denom,
    )
    table_l1 = WeightTable(
        luma_log2_weight_denom=luma_log2_denom,
        chroma_log2_weight_denom=chroma_log2_denom,
    )

    # Parse L0 weights
    for i in range(num_ref_idx_l0):
        luma_weight_flag = reader.read_flag()
        if luma_weight_flag:
            weight = reader.read_se()
            offset = reader.read_se()
            table_l0.set_luma_weight(i, weight, offset)

        if chroma_array_type != 0:
            chroma_weight_flag = reader.read_flag()
            if chroma_weight_flag:
                cb_weight = reader.read_se()
                cb_offset = reader.read_se()
                cr_weight = reader.read_se()
                cr_offset = reader.read_se()
                table_l0.set_chroma_weight(i, cb_weight, cb_offset, cr_weight, cr_offset)

    # Parse L1 weights
    for i in range(num_ref_idx_l1):
        luma_weight_flag = reader.read_flag()
        if luma_weight_flag:
            weight = reader.read_se()
            offset = reader.read_se()
            table_l1.set_luma_weight(i, weight, offset)

        if chroma_array_type != 0:
            chroma_weight_flag = reader.read_flag()
            if chroma_weight_flag:
                cb_weight = reader.read_se()
                cb_offset = reader.read_se()
                cr_weight = reader.read_se()
                cr_offset = reader.read_se()
                table_l1.set_chroma_weight(i, cb_weight, cb_offset, cr_weight, cr_offset)

    return table_l0, table_l1
