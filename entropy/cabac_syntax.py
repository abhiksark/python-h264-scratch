# h264/entropy/cabac_syntax.py
"""CABAC syntax element decoding.

Decode individual H.264 syntax elements using CABAC.
Each element has specific binarization and context assignment.

H.264 Spec Reference: Section 9.3.3.1 - Decoding process for binary decisions
"""

from typing import List, TYPE_CHECKING

from entropy.cabac_context import (
    CTX_MB_TYPE_I_START,
    CTX_MB_TYPE_P_START,
    CTX_MB_TYPE_B_START,
    CTX_SUB_MB_TYPE_P_START,
    CTX_MVD_START,
    CTX_REF_IDX_START,
    CTX_MB_QP_DELTA_START,
    CTX_INTRA_CHROMA_PRED_START,
    CTX_PREV_INTRA_PRED_FLAG_START,
    CTX_CODED_BLOCK_PATTERN_START,
    CTX_CODED_BLOCK_FLAG_START,
)
from entropy.cabac_binarize import decode_unary, decode_truncated_unary

if TYPE_CHECKING:
    from entropy.cabac_arith import CABACDecoder
    from entropy.cabac_context import CABACContext


# Context indices for mb_skip_flag (P-slice: 11-13, B-slice: 24-26)
CTX_MB_SKIP_FLAG_P = 11
CTX_MB_SKIP_FLAG_B = 24


def decode_mb_skip_flag(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    slice_type: int,
    mb_x: int,
    mb_y: int,
    left_skip: bool,
    top_skip: bool,
) -> int:
    """Decode mb_skip_flag.

    Context depends on neighbor skip status.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        slice_type: 0=P, 1=B
        mb_x, mb_y: Macroblock position
        left_skip: True if left MB is skipped
        top_skip: True if top MB is skipped

    Returns:
        0 or 1
    """
    # Context increment based on neighbors
    ctx_inc = (1 if left_skip else 0) + (1 if top_skip else 0)

    # Base context depends on slice type
    if slice_type == 1:  # B-slice
        ctx_idx = CTX_MB_SKIP_FLAG_B + ctx_inc
    else:  # P-slice
        ctx_idx = CTX_MB_SKIP_FLAG_P + ctx_inc

    return decoder.decode_decision(contexts[ctx_idx])


def decode_mb_type_i(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    ctx_inc: int = 0,
) -> int:
    """Decode mb_type for I-slice.

    Binarization (H.264 Table 9-26):
    - 0: I_4x4 = "0"
    - 1-24: I_16x16 sub-type
    - 25: I_PCM = "1" + terminate

    Context indices (H.264 Table 9-32, ctxIdxOffset=3):
    - binIdx 0: ctxIdx = 3 + ctx_inc (neighbor-dependent, 0-2)
    - binIdx 1: decode_terminate (I_PCM check)
    - binIdx 2: ctxIdx = 3 + 3 = 6 (cbp_luma flag)
    - binIdx 3: ctxIdx = 3 + 4 = 7 (chroma CBP first)
    - binIdx 4: ctxIdx = 3 + 5 = 8 (chroma CBP second, if needed)
    - binIdx 5-6: bypass (pred mode, 2 bits)

    Args:
        decoder: CABAC decoder
        contexts: Context models
        ctx_inc: Context increment for first bin (condTermFlagA + condTermFlagB)

    Returns:
        mb_type value (0-25)
    """
    # First bin: 0=I_4x4, 1=I_16x16 or I_PCM
    ctx_idx = CTX_MB_TYPE_I_START + ctx_inc
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 0  # I_4x4

    # Check for terminate (I_PCM)
    if decoder.decode_terminate() == 1:
        return 25  # I_PCM

    # Decode I_16x16 sub-type (mb_type 1-24)
    # mb_type = 1 + pred_mode + 4*cbp_chroma + 12*(cbp_luma!=0)

    # cbp_luma flag (ctxIdx = 3 + 3 = 6)
    cbp_luma = decoder.decode_decision(contexts[CTX_MB_TYPE_I_START + 3])

    # cbp_chroma: truncated unary max 2 (ctxIdx 7 for first, 8 for second)
    cbp_chroma = 0
    if decoder.decode_decision(contexts[CTX_MB_TYPE_I_START + 4]) == 1:
        if decoder.decode_decision(contexts[CTX_MB_TYPE_I_START + 5]) == 1:
            cbp_chroma = 2
        else:
            cbp_chroma = 1

    # pred_mode: 2 context-based bins (ctxIdx 9 and 10)
    pred_mode_bit0 = decoder.decode_decision(contexts[CTX_MB_TYPE_I_START + 6])
    pred_mode_bit1 = decoder.decode_decision(contexts[CTX_MB_TYPE_I_START + 7])
    pred_mode = pred_mode_bit0 * 2 + pred_mode_bit1

    return 1 + pred_mode + 4 * cbp_chroma + 12 * cbp_luma


def decode_mb_type_p(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
) -> int:
    """Decode mb_type for P-slice.

    P-MB types (0-4) or I-MB types (5+).

    Args:
        decoder: CABAC decoder
        contexts: Context models

    Returns:
        mb_type value
    """
    ctx_idx = CTX_MB_TYPE_P_START

    # First bin distinguishes P-MB from I-MB in P-slice
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        # P-MB type
        ctx_idx = CTX_MB_TYPE_P_START + 1
        if decoder.decode_decision(contexts[ctx_idx]) == 0:
            # P_L0_16x16 or P_8x8
            ctx_idx = CTX_MB_TYPE_P_START + 2
            if decoder.decode_decision(contexts[ctx_idx]) == 0:
                return 0  # P_L0_16x16
            else:
                return 3  # P_8x8
        else:
            # P_L0_L0_16x8 or P_L0_L0_8x16
            ctx_idx = CTX_MB_TYPE_P_START + 3
            if decoder.decode_decision(contexts[ctx_idx]) == 0:
                return 1  # P_L0_L0_16x8
            else:
                return 2  # P_L0_L0_8x16
    else:
        # I-MB in P-slice: decode as I-MB type + offset
        i_type = decode_mb_type_i(decoder, contexts)
        return 5 + i_type


def decode_mb_type_b(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
) -> int:
    """Decode mb_type for B-slice.

    B-MB types (0-22) or I-MB types (23+).

    Args:
        decoder: CABAC decoder
        contexts: Context models

    Returns:
        mb_type value
    """
    ctx_idx = CTX_MB_TYPE_B_START

    # First bin: 0=B_Direct, 1=other
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 0  # B_Direct_16x16

    ctx_idx = CTX_MB_TYPE_B_START + 1
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        # B_L0_16x16 or B_L1_16x16
        ctx_idx = CTX_MB_TYPE_B_START + 3
        if decoder.decode_decision(contexts[ctx_idx]) == 0:
            return 1  # B_L0_16x16
        else:
            return 2  # B_L1_16x16
    else:
        ctx_idx = CTX_MB_TYPE_B_START + 2
        if decoder.decode_decision(contexts[ctx_idx]) == 0:
            # B_Bi_16x16 or partitioned
            ctx_idx = CTX_MB_TYPE_B_START + 3
            if decoder.decode_decision(contexts[ctx_idx]) == 0:
                return 3  # B_Bi_16x16
            else:
                # Partitioned types (4-21) or B_8x8 (22)
                # Simplified: return B_8x8
                return 22
        else:
            # I-MB in B-slice
            i_type = decode_mb_type_i(decoder, contexts)
            return 23 + i_type


def decode_sub_mb_type_p(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
) -> int:
    """Decode sub_mb_type for P-slice P_8x8.

    Args:
        decoder: CABAC decoder
        contexts: Context models

    Returns:
        sub_mb_type (0-3)
    """
    ctx_idx = CTX_SUB_MB_TYPE_P_START

    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 0  # P_L0_8x8

    ctx_idx = CTX_SUB_MB_TYPE_P_START + 1
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 1  # P_L0_8x4

    ctx_idx = CTX_SUB_MB_TYPE_P_START + 2
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 2  # P_L0_4x8

    return 3  # P_L0_4x4


def decode_sub_mb_type_b(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
) -> int:
    """Decode sub_mb_type for B-slice B_8x8.

    Args:
        decoder: CABAC decoder
        contexts: Context models

    Returns:
        sub_mb_type (0-12)
    """
    # Simplified: use truncated unary
    return decode_truncated_unary(
        decoder, max_val=12, ctx_base=CTX_SUB_MB_TYPE_P_START, contexts=contexts
    )


def decode_ref_idx(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    list_idx: int,
) -> int:
    """Decode ref_idx_lX.

    Uses unary binarization.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        list_idx: 0=L0, 1=L1

    Returns:
        Reference index (>= 0)
    """
    ctx_base = CTX_REF_IDX_START + list_idx * 3

    return decode_unary(decoder, ctx_base=ctx_base, contexts=contexts, max_ctx_inc=2)


def decode_mvd_lx(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    list_idx: int,
    comp: int,
) -> int:
    """Decode mvd_lX[comp].

    Uses UEG3 binarization with sign.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        list_idx: 0=L0, 1=L1
        comp: 0=x, 1=y

    Returns:
        Signed MVD value
    """
    from entropy.cabac_binarize import decode_mvd

    return decode_mvd(decoder, contexts=contexts, comp=comp)


def decode_mb_qp_delta(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    ctx_inc_first: int = 0,
) -> int:
    """Decode mb_qp_delta.

    Unary binarization with spec-correct context assignment:
    - binIdx 0: ctxIdx = 60 + ctx_inc_first (0 or 1, from 9.3.3.1.1.10)
    - binIdx 1: ctxIdx = 60 + 2 = 62
    - binIdx >= 2: ctxIdx = 60 + 3 = 63

    H.264 Spec Reference: Table 9-32, Section 9.3.3.1.1.10

    Args:
        decoder: CABAC decoder
        contexts: Context models
        ctx_inc_first: Context increment for first bin (0 if prev qp_delta==0, 1 otherwise)

    Returns:
        Signed QP delta
    """
    # First bin
    ctx_idx = CTX_MB_QP_DELTA_START + ctx_inc_first
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 0

    # Second bin (binIdx=1): ctxIdxInc = 2
    ctx_idx = CTX_MB_QP_DELTA_START + 2
    abs_val = 1
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        # abs_val = 1
        pass
    else:
        # Subsequent bins (binIdx >= 2): ctxIdxInc = 3
        abs_val = 2
        ctx_idx = CTX_MB_QP_DELTA_START + 3
        while decoder.decode_decision(contexts[ctx_idx]) == 1:
            abs_val += 1

    # Map to signed: 1->1, 2->-1, 3->2, 4->-2, ...
    if abs_val % 2 == 1:
        return (abs_val + 1) // 2
    else:
        return -(abs_val // 2)


def decode_cbp_luma(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_type: int,
    left_cbp: int = -1,
    top_cbp: int = -1,
) -> int:
    """Decode coded_block_pattern luma part.

    4 bits, one per 8x8 block. Context depends on neighbor CBP values.

    8x8 block layout in MB:
        +---+---+
        | 0 | 1 |
        +---+---+
        | 2 | 3 |
        +---+---+

    H.264 Spec Reference: Section 9.3.3.1.1.3

    Args:
        decoder: CABAC decoder
        contexts: Context models
        mb_type: Current mb_type
        left_cbp: Left neighbor's CBP luma (-1 if unavailable)
        top_cbp: Top neighbor's CBP luma (-1 if unavailable)

    Returns:
        CBP luma (0-15)
    """
    ctx_base = CTX_CODED_BLOCK_PATTERN_START

    # Decode 4 bits, one per 8x8 block
    # condTermFlagN = !(neighbor_cbp & bit_mask)
    # When unavailable: treat as -1 (all bits set) → !(set) = 0
    # Matches ffmpeg: ctx = !(cbp_a & bit) + 2 * !(cbp_b & bit)
    cbp = 0
    for i in range(4):
        # Derive condTermFlagA (left adjacent block)
        if i == 0:  # Top-left: left neighbor is block 1 of left MB
            if left_cbp < 0:  # Unavailable → treat as all-set
                cond_a = 0
            else:
                cond_a = 0 if (left_cbp & 0x02) else 1
        elif i == 1:  # Top-right: left neighbor is block 0 in same MB
            cond_a = 0 if (cbp & 0x01) else 1
        elif i == 2:  # Bottom-left: left neighbor is block 3 of left MB
            if left_cbp < 0:
                cond_a = 0
            else:
                cond_a = 0 if (left_cbp & 0x08) else 1
        else:  # Bottom-right (i=3): left neighbor is block 2 in same MB
            cond_a = 0 if (cbp & 0x04) else 1

        # Derive condTermFlagB (top adjacent block)
        if i == 0:  # Top-left: top neighbor is block 2 of top MB
            if top_cbp < 0:
                cond_b = 0
            else:
                cond_b = 0 if (top_cbp & 0x04) else 1
        elif i == 1:  # Top-right: top neighbor is block 3 of top MB
            if top_cbp < 0:
                cond_b = 0
            else:
                cond_b = 0 if (top_cbp & 0x08) else 1
        elif i == 2:  # Bottom-left: top neighbor is block 0 in same MB
            cond_b = 0 if (cbp & 0x01) else 1
        else:  # Bottom-right (i=3): top neighbor is block 1 in same MB
            cond_b = 0 if (cbp & 0x02) else 1

        # ctx_inc = condTermFlagA + 2*condTermFlagB (range 0-3)
        ctx_inc = cond_a + 2 * cond_b
        ctx_idx = ctx_base + ctx_inc

        if decoder.decode_decision(contexts[ctx_idx]) == 1:
            cbp |= (1 << i)

    return cbp


def decode_cbp_chroma(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_type: int,
    left_cbp_chroma: int = -1,
    top_cbp_chroma: int = -1,
) -> int:
    """Decode coded_block_pattern chroma part.

    0=none, 1=DC only, 2=DC+AC.
    Context depends on neighbor chroma CBP values.

    H.264 Spec Reference: Section 9.3.3.1.1.3

    Args:
        decoder: CABAC decoder
        contexts: Context models
        mb_type: Current mb_type
        left_cbp_chroma: Left neighbor's CBP chroma (-1 if unavailable)
        top_cbp_chroma: Top neighbor's CBP chroma (-1 if unavailable)

    Returns:
        CBP chroma (0-2)
    """
    ctx_base = CTX_CODED_BLOCK_PATTERN_START + 4

    # First bin: distinguishes 0 vs non-zero
    # condTermFlag = 1 if neighbor chroma CBP > 0, 0 otherwise
    # When unavailable: condTermFlag = 0 (H.264 Section 9.3.3.1.1.3)
    if left_cbp_chroma < 0:
        cond_a = 0  # unavailable → 0
    else:
        cond_a = 1 if left_cbp_chroma > 0 else 0

    if top_cbp_chroma < 0:
        cond_b = 0  # unavailable → 0
    else:
        cond_b = 1 if top_cbp_chroma > 0 else 0

    ctx_inc = cond_a + 2 * cond_b
    ctx_idx = ctx_base + ctx_inc

    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 0

    # Second bin: distinguishes 1 (DC only) vs 2 (DC+AC)
    # condTermFlag = 1 if neighbor chroma CBP == 2, 0 otherwise
    # When unavailable: condTermFlag = 0 (H.264 Section 9.3.3.1.1.3)
    if left_cbp_chroma < 0:
        cond_a = 0  # unavailable → 0
    else:
        cond_a = 1 if left_cbp_chroma == 2 else 0

    if top_cbp_chroma < 0:
        cond_b = 0  # unavailable → 0
    else:
        cond_b = 1 if top_cbp_chroma == 2 else 0

    ctx_inc = cond_a + 2 * cond_b
    ctx_idx = ctx_base + 4 + ctx_inc  # Second set of contexts

    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 1
    else:
        return 2


def decode_prev_intra4x4_pred_mode_flag(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
) -> int:
    """Decode prev_intra4x4_pred_mode_flag.

    Args:
        decoder: CABAC decoder
        contexts: Context models

    Returns:
        0 or 1
    """
    ctx_idx = CTX_PREV_INTRA_PRED_FLAG_START
    return decoder.decode_decision(contexts[ctx_idx])


def decode_rem_intra4x4_pred_mode(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
) -> int:
    """Decode rem_intra4x4_pred_mode.

    3-bit fixed length using regular context-based decode at ctxIdx 69.
    All 3 bins use the same context. Value assembled LSB-first.

    H.264 Spec Reference: Table 9-34, ctxIdxOffset=69

    Args:
        decoder: CABAC decoder
        contexts: Context models

    Returns:
        Remaining prediction mode (0-7)
    """
    ctx_idx = CTX_PREV_INTRA_PRED_FLAG_START + 1  # ctxIdx 69
    value = 0
    value |= decoder.decode_decision(contexts[ctx_idx])
    value |= decoder.decode_decision(contexts[ctx_idx]) << 1
    value |= decoder.decode_decision(contexts[ctx_idx]) << 2
    return value


def decode_intra_chroma_pred_mode(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    ctx_inc: int = 0,
) -> int:
    """Decode intra_chroma_pred_mode.

    Truncated unary (0-3) with spec-correct context assignment:
    - binIdx 0: ctxIdx = 64 + ctx_inc (neighbor-dependent, 0-2)
    - binIdx 1,2: ctxIdx = 64 + 3 = 67

    H.264 Spec Reference: Table 9-32, Section 9.3.3.1.1.8

    Args:
        decoder: CABAC decoder
        contexts: Context models
        ctx_inc: Context increment for first bin (condTermFlagA + condTermFlagB)

    Returns:
        Chroma prediction mode (0-3)
    """
    # First bin: neighbor-dependent context
    ctx_idx = CTX_INTRA_CHROMA_PRED_START + ctx_inc
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 0

    # Subsequent bins use fixed ctxIdxInc = 3
    ctx_idx = CTX_INTRA_CHROMA_PRED_START + 3
    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 1

    if decoder.decode_decision(contexts[ctx_idx]) == 0:
        return 2

    return 3


def decode_coded_block_flag(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    cat: int,
    ctx_block_cat: int,
) -> int:
    """Decode coded_block_flag.

    Indicates if a block has non-zero coefficients.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        cat: Block category (0-4)
        ctx_block_cat: Context offset within category

    Returns:
        0 or 1
    """
    ctx_idx = CTX_CODED_BLOCK_FLAG_START + cat * 4 + ctx_block_cat
    return decoder.decode_decision(contexts[ctx_idx])


def decode_mvd_lx_suffix_bypass(
    decoder: 'CABACDecoder',
    prefix_value: int,
) -> int:
    """Decode MVD suffix using bypass mode.

    For MVD values > 9, the suffix is exp-golomb coded in bypass mode.

    Args:
        decoder: CABAC decoder
        prefix_value: Value decoded from prefix (>= 9)

    Returns:
        Suffix value to add to prefix

    H.264 Spec Reference: Section 9.3.2.3
    """
    if prefix_value < 9:
        return 0

    # Use UEG3 (exp-golomb order 3)
    from entropy.cabac_binarize import decode_exp_golomb_bypass
    suffix = decode_exp_golomb_bypass(decoder, k=3)

    return suffix


def decode_mvd_sign_bypass(
    decoder: 'CABACDecoder',
) -> int:
    """Decode MVD sign using bypass mode.

    Args:
        decoder: CABAC decoder

    Returns:
        Sign value: 0 = positive, 1 = negative
    """
    return decoder.decode_bypass()


def get_ref_idx_ctx_idx(
    list_idx: int,
    bin_idx: int = None,
    ctx_inc: int = None,
) -> int:
    """Get context index for ref_idx decoding.

    Args:
        list_idx: Reference list (0=L0, 1=L1)
        bin_idx: Bin index within binarization (alternative to ctx_inc)
        ctx_inc: Context increment (alternative to bin_idx)

    Returns:
        Context index

    H.264 Spec Reference: Section 9.3.3.1.1.5
    """
    # Context base for ref_idx_lX
    # L0: contexts 54-55, L1: contexts 56-57
    ctx_base = 54 if list_idx == 0 else 56

    # Use ctx_inc if provided, otherwise derive from bin_idx
    if ctx_inc is not None:
        inc = min(2, ctx_inc)
    elif bin_idx is not None:
        inc = min(2, bin_idx)
    else:
        inc = 0

    return ctx_base + inc
