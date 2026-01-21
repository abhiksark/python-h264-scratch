# h264/inter/bipred.py
"""Bi-directional prediction for B-frames.

Bi-prediction averages predictions from L0 (forward) and L1 (backward)
reference frames:
    result = (pred_l0 + pred_l1 + 1) >> 1

Weighted bi-prediction applies weights:
    result = ((w0*pred_l0 + w1*pred_l1 + (1<<log2_denom)) >> (log2_denom+1)) + (o0+o1+1)>>1

H.264 Spec Reference: Section 8.4.2.2 - Decoding process for inter prediction samples
"""

from typing import Tuple
import numpy as np


def bipred_average(pred_l0: np.ndarray, pred_l1: np.ndarray) -> np.ndarray:
    """Average L0 and L1 predictions for bi-prediction.

    Uses round-up formula: result = (pred_l0 + pred_l1 + 1) >> 1

    Args:
        pred_l0: L0 (forward) prediction block
        pred_l1: L1 (backward) prediction block

    Returns:
        Averaged prediction block, uint8

    H.264 Spec: Section 8.4.2.2.1
    """
    # Use int16 to avoid overflow during addition
    sum_pred = pred_l0.astype(np.int16) + pred_l1.astype(np.int16) + 1
    result = sum_pred >> 1

    return result.astype(np.uint8)


def bipred_chroma(
    cb_l0: np.ndarray,
    cb_l1: np.ndarray,
    cr_l0: np.ndarray,
    cr_l1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average L0 and L1 predictions for chroma planes.

    Args:
        cb_l0: L0 Cb prediction
        cb_l1: L1 Cb prediction
        cr_l0: L0 Cr prediction
        cr_l1: L1 Cr prediction

    Returns:
        Tuple of (cb_result, cr_result)
    """
    cb_result = bipred_average(cb_l0, cb_l1)
    cr_result = bipred_average(cr_l0, cr_l1)
    return cb_result, cr_result


def weighted_bipred(
    pred_l0: np.ndarray,
    pred_l1: np.ndarray,
    w0: int,
    o0: int,
    w1: int,
    o1: int,
    log2_denom: int,
) -> np.ndarray:
    """Apply weighted bi-prediction.

    Formula for explicit weighted biprediction:
        result = Clip1Y(((w0*pred_l0 + w1*pred_l1 + 2^log2_denom) >> (log2_denom+1)) + ((o0+o1+1)>>1))

    Args:
        pred_l0: L0 prediction block
        pred_l1: L1 prediction block
        w0: L0 weight
        o0: L0 offset
        w1: L1 weight
        o1: L1 offset
        log2_denom: Log2 of weight denominator

    Returns:
        Weighted prediction block, uint8

    H.264 Spec: Section 8.4.2.3.2
    """
    # Convert to int32 for computation
    p0 = pred_l0.astype(np.int32)
    p1 = pred_l1.astype(np.int32)

    # Weighted sum with rounding
    rounding = 1 << log2_denom
    weighted_sum = w0 * p0 + w1 * p1 + rounding

    # Shift and add offset
    result = (weighted_sum >> (log2_denom + 1)) + ((o0 + o1 + 1) >> 1)

    # Clip to valid range
    return np.clip(result, 0, 255).astype(np.uint8)


def weighted_bipred_chroma(
    cb_l0: np.ndarray,
    cb_l1: np.ndarray,
    cr_l0: np.ndarray,
    cr_l1: np.ndarray,
    w0_cb: int,
    o0_cb: int,
    w1_cb: int,
    o1_cb: int,
    w0_cr: int,
    o0_cr: int,
    w1_cr: int,
    o1_cr: int,
    log2_denom: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply weighted bi-prediction to chroma planes.

    Args:
        cb_l0, cb_l1: Cb predictions from L0 and L1
        cr_l0, cr_l1: Cr predictions from L0 and L1
        w0_cb, o0_cb, w1_cb, o1_cb: Cb weights and offsets
        w0_cr, o0_cr, w1_cr, o1_cr: Cr weights and offsets
        log2_denom: Log2 of weight denominator

    Returns:
        Tuple of (cb_result, cr_result)
    """
    cb_result = weighted_bipred(cb_l0, cb_l1, w0_cb, o0_cb, w1_cb, o1_cb, log2_denom)
    cr_result = weighted_bipred(cr_l0, cr_l1, w0_cr, o0_cr, w1_cr, o1_cr, log2_denom)
    return cb_result, cr_result


def calc_implicit_bipred_weights(
    current_poc: int,
    l0_poc: int,
    l1_poc: int,
) -> Tuple[int, int]:
    """Calculate implicit bi-prediction weights from POC distances.

    For implicit weighted prediction (weighted_bipred_idc=2),
    weights are derived from temporal distances.

    Args:
        current_poc: POC of current picture
        l0_poc: POC of L0 reference
        l1_poc: POC of L1 reference

    Returns:
        Tuple of (w0, w1) weights in 1/64 scale

    H.264 Spec: Section 8.4.2.3.2
    """
    tb = current_poc - l0_poc  # Distance to L0
    td = l1_poc - l0_poc  # Distance between references

    if td == 0:
        return 32, 32  # Equal weights if same POC

    # Calculate w1 proportional to tb/td
    # w1 = (tb * 256 + abs(td)/2) / td, then scale to 1/64
    tx = (16384 + abs(td // 2)) // td
    dist_scale_factor = (tb * tx + 32) >> 6

    # Clamp to valid range
    dist_scale_factor = max(-1024, min(1023, dist_scale_factor))

    w1 = dist_scale_factor >> 2
    w0 = 64 - w1

    # Clamp weights
    w0 = max(-64, min(128, w0))
    w1 = max(-64, min(128, w1))

    return w0, w1
