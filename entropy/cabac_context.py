# h264/entropy/cabac_context.py
"""CABAC context model initialization.

CABAC uses 460 context models (ctxIdx 0-459), each with:
- pStateIdx: Probability state index (0-63), higher = more likely MPS
- valMPS: Most Probable Symbol (0 or 1)

Context initialization depends on slice type and QP using m,n parameters:
    preCtxState = Clip3(1, 126, ((m * SliceQP) >> 4) + n)
    valMPS = (preCtxState >= 64) ? 1 : 0
    pStateIdx = (valMPS) ? preCtxState - 64 : 63 - preCtxState

H.264 Spec Reference: Section 9.3.1.1 - Initialization process for context variables
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CABACContext:
    """CABAC context model.

    Attributes:
        pStateIdx: Probability state index (0-63)
        valMPS: Most probable symbol (0 or 1)
    """
    pStateIdx: int
    valMPS: int


# Context index ranges (H.264 Table 9-11)
CTX_MB_TYPE_SI_START = 0        # SI slice mb_type (ctxIdx 0-2)
CTX_MB_TYPE_I_START = 3         # I slice mb_type (ctxIdx 3-13)
CTX_MB_TYPE_P_START = 14        # P slice mb_type (ctxIdx 14-26)
CTX_MB_TYPE_B_START = 27        # B slice mb_type (ctxIdx 27-35)
CTX_SUB_MB_TYPE_P_START = 36    # P slice sub_mb_type (ctxIdx 36-39)
CTX_MVD_START = 40              # mvd (ctxIdx 40-46)
CTX_REF_IDX_START = 54          # ref_idx (ctxIdx 54-59)
CTX_MB_QP_DELTA_START = 60      # mb_qp_delta (ctxIdx 60-63)
CTX_INTRA_CHROMA_PRED_START = 64  # intra_chroma_pred_mode (ctxIdx 64-67)
CTX_PREV_INTRA_PRED_FLAG_START = 68  # prev_intra*_pred_mode_flag (ctxIdx 68-72)
CTX_CODED_BLOCK_PATTERN_START = 73   # coded_block_pattern (ctxIdx 73-84)
CTX_CODED_BLOCK_FLAG_START = 85      # coded_block_flag (ctxIdx 85-104)
CTX_SIG_COEFF_FLAG_START = 105       # significant_coeff_flag (ctxIdx 105-165)
CTX_LAST_SIG_COEFF_START = 166       # last_significant_coeff_flag (ctxIdx 166-226)
CTX_COEFF_ABS_LEVEL_START = 227      # coeff_abs_level_minus1 (ctxIdx 227-275)

# Total number of contexts
NUM_CONTEXTS = 460


def calc_initial_state(m: int, n: int, slice_qp: int) -> Tuple[int, int]:
    """Calculate initial pStateIdx and valMPS from m, n parameters.

    Formula (H.264 equations 9-5 and 9-6):
        preCtxState = Clip3(1, 126, ((m * SliceQP) >> 4) + n)
        valMPS = (preCtxState >= 64) ? 1 : 0
        pStateIdx = (valMPS) ? preCtxState - 64 : 63 - preCtxState

    Args:
        m: Slope parameter from spec tables
        n: Offset parameter from spec tables
        slice_qp: Slice QP value (0-51)

    Returns:
        Tuple of (pStateIdx, valMPS)
    """
    # Equation 9-5: preCtxState calculation
    pre_ctx_state = ((m * slice_qp) >> 4) + n
    pre_ctx_state = max(1, min(126, pre_ctx_state))  # Clip3(1, 126, ...)

    # Equation 9-6: Derive valMPS and pStateIdx
    if pre_ctx_state >= 64:
        val_mps = 1
        p_state_idx = pre_ctx_state - 64
    else:
        val_mps = 0
        p_state_idx = 63 - pre_ctx_state

    return p_state_idx, val_mps


# Initialization parameters (m, n) from H.264 Tables 9-12 through 9-23
# Format: INIT_PARAMS[slice_type][ctxIdx] = (m, n)
# slice_type: 0=P, 1=B, 2=I

# Default initialization for all contexts (can be overridden per slice type)
_DEFAULT_MN = (0, 64)  # Results in pStateIdx=0, valMPS=1

# Simplified initialization table (subset of full spec)
# Full implementation would have all 460 entries from H.264 Tables 9-12 to 9-23
# These are representative values for basic functionality

_INIT_I = [
    # mb_type SI (0-2) - from H.264 Table 9-12
    (20, -15), (2, 54), (3, 74),
    # mb_type I (3-10) - from H.264 Table 9-12
    (20, -15), (2, 54), (3, 74), (-28, 127), (-23, 104),
    (-6, 53), (-1, 54), (7, 51),
    # mb_type I continued (11-13)
    (-11, 72), (-6, 76), (-10, 77),
] + [_DEFAULT_MN] * (NUM_CONTEXTS - 14)

_INIT_P = [
    # mb_type SI (0-2) - from H.264 Table 9-12
    (20, -15), (2, 54), (3, 74),
    # mb_type I (3-10) - from H.264 Table 9-12
    (20, -15), (2, 54), (3, 74), (-28, 127), (-23, 104),
    (-6, 53), (-1, 54), (7, 51),
    # mb_type I continued (11-13)
    (-11, 72), (-6, 76), (-10, 77),
    # mb_type P (14-26)
    (23, 33), (23, 2), (21, 0), (14, 52), (16, 38), (19, 26),
    (20, 22), (14, 36), (17, 27), (14, 52), (16, 38), (19, 26), (20, 22),
    # sub_mb_type P (27-35) - actually B mb_type in spec but reusing
    (20, 40), (-2, 69), (-4, 86), (-11, 72), (-6, 76), (-10, 77),
    (-4, 86), (-7, 77), (-4, 86),
    # sub_mb_type (36-39)
    (21, 33), (18, 37), (15, 44), (16, 42),
    # mvd (40-53)
    (32, 95), (-18, 121), (-2, 77), (29, 89), (-12, 105), (3, 79),
    (32, 95), (-18, 121), (-2, 77), (29, 89), (-12, 105), (3, 79),
    (32, 95), (-18, 121),
    # ref_idx (54-59)
    (0, 41), (0, 63), (0, 63), (0, 41), (0, 63), (0, 63),
    # mb_qp_delta (60-63)
    (0, 45), (0, 61), (0, 68), (0, 73),
] + [_DEFAULT_MN] * (NUM_CONTEXTS - 64)

_INIT_B = [
    # mb_type SI (0-2) - from H.264 Table 9-12
    (20, -15), (2, 54), (3, 74),
    # mb_type I (3-10) - from H.264 Table 9-12
    (20, -15), (2, 54), (3, 74), (-28, 127), (-23, 104),
    (-6, 53), (-1, 54), (7, 51),
    # mb_type I continued (11-13)
    (-11, 72), (-6, 76), (-10, 77),
    # mb_type P (14-26)
    (23, 33), (23, 2), (21, 0), (14, 52), (16, 38), (19, 26),
    (20, 22), (14, 36), (17, 27), (14, 52), (16, 38), (19, 26), (20, 22),
    # mb_type B (27-35)
    (17, 26), (3, 62), (-1, 70), (-2, 75), (5, 65), (5, 64),
    (4, 64), (3, 66), (3, 65),
    # sub_mb_type (36-39)
    (21, 33), (18, 37), (15, 44), (16, 42),
    # mvd (40-53)
    (29, 99), (-22, 124), (-6, 84), (26, 94), (-16, 113), (0, 82),
    (29, 99), (-22, 124), (-6, 84), (26, 94), (-16, 113), (0, 82),
    (29, 99), (-22, 124),
    # ref_idx (54-59)
    (-4, 55), (-2, 69), (-2, 69), (-4, 55), (-2, 69), (-2, 69),
    # mb_qp_delta (60-63)
    (0, 45), (0, 61), (0, 68), (0, 73),
] + [_DEFAULT_MN] * (NUM_CONTEXTS - 64)


def _get_init_params(slice_type: int) -> List[Tuple[int, int]]:
    """Get initialization parameters for slice type.

    Args:
        slice_type: 0=P, 1=B, 2=I (or 5=P, 6=B, 7=I)

    Returns:
        List of (m, n) tuples for each context
    """
    # Normalize slice type (0,5=P, 1,6=B, 2,7=I)
    st = slice_type % 5

    if st == 2:  # I-slice
        return _INIT_I
    elif st == 0:  # P-slice
        return _INIT_P
    else:  # B-slice
        return _INIT_B


def init_context_models(slice_type: int, slice_qp: int) -> List[CABACContext]:
    """Initialize all 460 CABAC context models.

    Args:
        slice_type: Slice type (0=P, 1=B, 2=I)
        slice_qp: Slice QP value (0-51)

    Returns:
        List of 460 initialized CABACContext objects
    """
    init_params = _get_init_params(slice_type)

    contexts = []
    for i in range(NUM_CONTEXTS):
        if i < len(init_params):
            m, n = init_params[i]
        else:
            m, n = _DEFAULT_MN

        p_state_idx, val_mps = calc_initial_state(m, n, slice_qp)
        contexts.append(CABACContext(pStateIdx=p_state_idx, valMPS=val_mps))

    return contexts


def init_context_models_with_idc(
    slice_type: int,
    slice_qp: int,
    cabac_init_idc: int = 0,
) -> List[CABACContext]:
    """Initialize CABAC context models with cabac_init_idc.

    cabac_init_idc selects between different initialization tables
    for P and B slices. I-slices always use idc=0.

    Args:
        slice_type: Slice type (0=P, 1=B, 2=I)
        slice_qp: Slice QP value (0-51)
        cabac_init_idc: Initialization table index (0, 1, or 2)

    Returns:
        List of 460 initialized CABACContext objects

    H.264 Spec Reference: Section 9.3.1.1
    """
    # I-slices always use default initialization
    if slice_type == 2:
        return init_context_models(slice_type, slice_qp)

    # For P/B slices, cabac_init_idc selects the table
    # Currently using same tables, but structure allows different ones
    # The init_idc affects which predefined table is used
    init_params = _get_init_params_with_idc(slice_type, cabac_init_idc)

    contexts = []
    for i in range(NUM_CONTEXTS):
        if i < len(init_params):
            m, n = init_params[i]
        else:
            m, n = _DEFAULT_MN

        p_state_idx, val_mps = calc_initial_state(m, n, slice_qp)
        contexts.append(CABACContext(pStateIdx=p_state_idx, valMPS=val_mps))

    return contexts


def _get_init_params_with_idc(
    slice_type: int,
    cabac_init_idc: int,
) -> List[Tuple[int, int]]:
    """Get initialization parameters based on slice type and cabac_init_idc.

    Args:
        slice_type: Slice type
        cabac_init_idc: Init table index (0, 1, or 2)

    Returns:
        List of (m, n) init value pairs
    """
    # For now, use the same base parameters
    # Different cabac_init_idc values can select different tables
    # This is a simplified implementation
    base_params = _get_init_params(slice_type)

    # Apply slight modifications based on init_idc for P/B slices
    # In full implementation, these would be completely different tables
    if cabac_init_idc == 0:
        return base_params
    elif cabac_init_idc == 1:
        # Table 1 - slightly different initial states
        return [(max(-128, m - 2), n) for m, n in base_params]
    else:  # cabac_init_idc == 2
        # Table 2 - another variation
        return [(min(127, m + 2), n) for m, n in base_params]
