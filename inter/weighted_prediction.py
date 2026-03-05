from typing import Dict, Tuple

import numpy as np

from inter.bipred import weighted_bipred, weighted_bipred_chroma
from inter.weighted_pred import (
    WeightTable,
    WeightTableBSlice,
    apply_weighted_prediction,
    apply_weighted_prediction_chroma,
    validate_log2_weight_denom as _validate_log2_weight_denom,
    validate_weight_offset,
)


def apply_explicit_weights(
    pred: np.ndarray,
    weight: int,
    offset: int,
    log2_weight_denom: int,
) -> np.ndarray:
    return apply_weighted_prediction(
        pred,
        weight=weight,
        offset=offset,
        log2_denom=log2_weight_denom,
    )


def apply_explicit_chroma_weights(
    pred_cb: np.ndarray,
    pred_cr: np.ndarray,
    weight_cb: int,
    offset_cb: int,
    weight_cr: int,
    offset_cr: int,
    chroma_log2_weight_denom: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return apply_weighted_prediction_chroma(
        pred_cb,
        pred_cr,
        weight_cb=weight_cb,
        offset_cb=offset_cb,
        weight_cr=weight_cr,
        offset_cr=offset_cr,
        log2_denom=chroma_log2_weight_denom,
    )


def apply_explicit_weights_with_table(
    pred: np.ndarray,
    table: WeightTable,
    ref_idx: int,
) -> np.ndarray:
    weight, offset = table.get_luma_weight(ref_idx)
    return apply_weighted_prediction(
        pred,
        weight=weight,
        offset=offset,
        log2_denom=table.luma_log2_weight_denom,
    )


def apply_explicit_chroma_with_table(
    pred_cb: np.ndarray,
    pred_cr: np.ndarray,
    table: WeightTable,
    ref_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    weight_cb, offset_cb, weight_cr, offset_cr = table.get_chroma_weight(ref_idx)
    return apply_weighted_prediction_chroma(
        pred_cb,
        pred_cr,
        weight_cb=weight_cb,
        offset_cb=offset_cb,
        weight_cr=weight_cr,
        offset_cr=offset_cr,
        log2_denom=table.chroma_log2_weight_denom,
    )


def apply_explicit_bipred_weights(
    pred_l0: np.ndarray,
    pred_l1: np.ndarray,
    w0: int,
    o0: int,
    w1: int,
    o1: int,
    log2_weight_denom: int,
) -> np.ndarray:
    return weighted_bipred(
        pred_l0,
        pred_l1,
        w0=w0,
        o0=o0,
        w1=w1,
        o1=o1,
        log2_denom=log2_weight_denom,
    )


def apply_explicit_bipred_chroma_weights(
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
    chroma_log2_weight_denom: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return weighted_bipred_chroma(
        cb_l0,
        cb_l1,
        cr_l0,
        cr_l1,
        w0_cb=w0_cb,
        o0_cb=o0_cb,
        w1_cb=w1_cb,
        o1_cb=o1_cb,
        w0_cr=w0_cr,
        o0_cr=o0_cr,
        w1_cr=w1_cr,
        o1_cr=o1_cr,
        log2_denom=chroma_log2_weight_denom,
    )


def validate_log2_weight_denom(log2_weight_denom: int) -> bool:
    return _validate_log2_weight_denom(log2_weight_denom)


def validate_weight(weight: int) -> bool:
    return validate_weight_offset(weight=weight, offset=0)


def validate_offset(offset: int) -> bool:
    return validate_weight_offset(weight=0, offset=offset)


def get_explicit_weight_for_ref(
    ref_idx: int,
    luma_log2_weight_denom: int,
    explicit_weights: Dict[int, Tuple[int, int]],
) -> Tuple[int, int]:
    return explicit_weights.get(ref_idx, (1 << luma_log2_weight_denom, 0))


__all__ = [
    "WeightTableBSlice",
    "apply_explicit_weights",
    "apply_explicit_chroma_weights",
    "apply_explicit_weights_with_table",
    "apply_explicit_chroma_with_table",
    "apply_explicit_bipred_weights",
    "apply_explicit_bipred_chroma_weights",
    "validate_log2_weight_denom",
    "validate_weight",
    "validate_offset",
    "get_explicit_weight_for_ref",
]
