# h264/entropy/cabac_residual.py
"""CABAC residual block decoding.

Decode transform coefficient blocks using CABAC:
1. coded_block_flag - is block coded?
2. significant_coeff_flag - which scan positions have coefficients?
3. last_significant_coeff_flag - where is last coefficient?
4. coeff_abs_level_minus1 - coefficient magnitudes
5. coeff_sign_flag - coefficient signs (bypass)

H.264 Spec Reference: Section 9.3.3.1.3 - Decoding process for significance map
"""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

from entropy.cabac_context import (
    CTX_SIG_COEFF_FLAG_START,
    CTX_LAST_SIG_COEFF_START,
    CTX_COEFF_ABS_LEVEL_START,
)

if TYPE_CHECKING:
    from entropy.cabac_arith import CABACDecoder
    from entropy.cabac_context import CABACContext


# 4x4 zigzag scan order (H.264 Table 8-13)
ZIGZAG_SCAN_4X4 = [
    0, 1, 4, 8,
    5, 2, 3, 6,
    9, 12, 13, 10,
    7, 11, 14, 15,
]

# 2x2 scan order for chroma DC
SCAN_2X2 = [0, 1, 2, 3]

# Context index offsets per block category (H.264 Table 9-39)
# Format: (sig_ctx_offset, last_ctx_offset, abs_ctx_offset)
BLOCK_CAT_CTX_OFFSETS = {
    0: (0, 0, 0),    # Luma DC (I_16x16)
    1: (15, 15, 10), # Luma AC (I_16x16)
    2: (29, 29, 20), # Luma 4x4
    3: (44, 44, 30), # Chroma DC
    4: (47, 47, 39), # Chroma AC
}


def get_scan_order(block_cat: int) -> List[int]:
    """Get scan order for block category.

    Args:
        block_cat: Block category (0-4)

    Returns:
        List of scan positions
    """
    if block_cat == 3:  # Chroma DC
        return SCAN_2X2
    return ZIGZAG_SCAN_4X4


def decode_significant_coeff_flag(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    block_cat: int,
    scan_idx: int,
) -> int:
    """Decode significant_coeff_flag.

    Indicates if coefficient at scan position is non-zero.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        block_cat: Block category
        scan_idx: Position in scan order

    Returns:
        0 or 1
    """
    sig_offset, _, _ = BLOCK_CAT_CTX_OFFSETS.get(block_cat, (0, 0, 0))

    # Context depends on scan position
    ctx_idx = CTX_SIG_COEFF_FLAG_START + sig_offset + min(scan_idx, 14)

    return decoder.decode_decision(contexts[ctx_idx])


def decode_last_significant_coeff_flag(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    block_cat: int,
    scan_idx: int,
) -> int:
    """Decode last_significant_coeff_flag.

    Indicates if this is the last non-zero coefficient.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        block_cat: Block category
        scan_idx: Position in scan order

    Returns:
        0 or 1
    """
    _, last_offset, _ = BLOCK_CAT_CTX_OFFSETS.get(block_cat, (0, 0, 0))

    ctx_idx = CTX_LAST_SIG_COEFF_START + last_offset + min(scan_idx, 14)

    return decoder.decode_decision(contexts[ctx_idx])


def decode_coeff_abs_level_minus1(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    block_cat: int,
    num_decode_abs_level_eq1: int,
    num_decode_abs_level_gt1: int,
) -> int:
    """Decode coeff_abs_level_minus1.

    Decodes |coeff| - 1, so actual level = return + 1.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        block_cat: Block category
        num_decode_abs_level_eq1: Count of level==1 decoded so far
        num_decode_abs_level_gt1: Count of level>1 decoded so far

    Returns:
        coeff_abs_level_minus1 (>= 0)
    """
    _, _, abs_offset = BLOCK_CAT_CTX_OFFSETS.get(block_cat, (0, 0, 0))
    ctx_base = CTX_COEFF_ABS_LEVEL_START + abs_offset

    # Context for first bin depends on previous levels
    if num_decode_abs_level_gt1 > 0:
        ctx_idx_inc = 0
    else:
        ctx_idx_inc = min(4, 1 + num_decode_abs_level_eq1)

    ctx_idx = ctx_base + ctx_idx_inc

    # Decode prefix (coeff_abs_level_greater1_flag sequence)
    prefix = 0
    while prefix < 14:
        if decoder.decode_decision(contexts[ctx_idx]) == 0:
            break
        prefix += 1

        # Update context for subsequent bins
        if prefix == 1:
            ctx_idx_inc = 5 if num_decode_abs_level_gt1 > 0 else 4
        else:
            ctx_idx_inc = min(4, 5 + num_decode_abs_level_gt1)
        ctx_idx = ctx_base + ctx_idx_inc

    if prefix < 14:
        return prefix

    # Decode suffix using exp-golomb bypass
    suffix = 0
    k = 0

    while decoder.decode_bypass() == 1:
        k += 1

    for _ in range(k):
        suffix = (suffix << 1) | decoder.decode_bypass()

    suffix += (1 << k) - 1

    return 14 + suffix


def decode_coeff_sign_flag(decoder: 'CABACDecoder') -> int:
    """Decode coeff_sign_flag using bypass.

    Args:
        decoder: CABAC decoder

    Returns:
        0 (positive) or 1 (negative)
    """
    return decoder.decode_bypass()


def decode_residual_block_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    max_coeff: int,
    block_cat: int,
) -> np.ndarray:
    """Decode a residual coefficient block using CABAC.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        max_coeff: Maximum number of coefficients (4, 15, or 16)
        block_cat: Block category (0-4)

    Returns:
        Array of coefficients in scan order
    """
    coeffs = np.zeros(max_coeff, dtype=np.int32)

    # Get scan order
    scan = get_scan_order(block_cat)[:max_coeff]

    # Track coefficient positions (reverse order for level decoding)
    sig_positions = []

    # Decode significance map
    for i in range(max_coeff - 1):  # Last position always checked if reached
        sig = decode_significant_coeff_flag(decoder, contexts, block_cat, i)

        if sig == 1:
            sig_positions.append(i)

            # Check if this is the last coefficient
            last = decode_last_significant_coeff_flag(decoder, contexts, block_cat, i)
            if last == 1:
                break
    else:
        # Reached last position - check if significant
        if len(sig_positions) == 0 or sig_positions[-1] < max_coeff - 1:
            sig = decode_significant_coeff_flag(
                decoder, contexts, block_cat, max_coeff - 1
            )
            if sig == 1:
                sig_positions.append(max_coeff - 1)

    # Decode coefficient levels (in reverse scan order)
    num_eq1 = 0
    num_gt1 = 0

    for pos in reversed(sig_positions):
        level_m1 = decode_coeff_abs_level_minus1(
            decoder, contexts, block_cat, num_eq1, num_gt1
        )
        level = level_m1 + 1

        # Decode sign
        sign = decode_coeff_sign_flag(decoder)
        if sign == 1:
            level = -level

        coeffs[pos] = level

        # Update counters
        if abs(level) == 1:
            num_eq1 += 1
        else:
            num_gt1 += 1

    return coeffs


def decode_residual_block_cabac_with_cbf(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    max_coeff: int,
    block_cat: int,
    coded_block_flag_ctx_idx: int,
) -> np.ndarray:
    """Decode residual block with coded_block_flag check.

    For CABAC, each block has a coded_block_flag decoded before the
    significance map. If coded_block_flag=0, the block is all zeros.

    H.264 Spec Reference: Section 9.3.3.1.1.9

    Args:
        decoder: CABAC decoder
        contexts: Context models
        max_coeff: Maximum coefficients (4, 15, or 16)
        block_cat: Block category (0-4)
        coded_block_flag_ctx_idx: Context index for coded_block_flag

    Returns:
        Array of coefficients (all zeros if coded_block_flag=0)
    """
    cbf = decoder.decode_decision(contexts[coded_block_flag_ctx_idx])

    if cbf == 0:
        return np.zeros(max_coeff, dtype=np.int32)

    return decode_residual_block_cabac(decoder, contexts, max_coeff, block_cat)


def decode_coeff_abs_level_suffix_bypass(
    decoder: 'CABACDecoder',
    prefix_value: int,
) -> int:
    """Decode coefficient level suffix using bypass mode.

    For levels > 14, the suffix is exp-golomb coded in bypass mode.

    Args:
        decoder: CABAC decoder
        prefix_value: Value decoded from prefix (>= 14)

    Returns:
        Suffix value

    H.264 Spec Reference: Section 9.3.2.3
    """
    if prefix_value < 14:
        return 0

    # Exp-golomb order 0 for suffix
    suffix = 0
    suffix_len = 0

    # Count leading ones in bypass mode
    while decoder.decode_bypass() == 1:
        suffix_len += 1
        if suffix_len > 16:  # Sanity limit
            break

    # Read suffix_len bits
    for _ in range(suffix_len):
        suffix = (suffix << 1) | decoder.decode_bypass()

    # Add offset
    suffix += (1 << suffix_len) - 1

    return suffix


def get_coded_block_flag_ctx_idx(
    block_cat: int,
    block_type: str = None,
    left_cbf: int = 0,
    top_cbf: int = 0,
    left_available: bool = True,
    top_available: bool = True,
) -> int:
    """Get context index for coded_block_flag.

    Context depends on block category and neighbor coded_block_flags.

    Args:
        block_cat: Block category (0-4)
        block_type: Block type string (optional, derived from block_cat if not given)
        left_cbf: Left neighbor coded_block_flag (0 or 1)
        top_cbf: Top neighbor coded_block_flag (0 or 1)
        left_available: Whether left neighbor is available
        top_available: Whether top neighbor is available

    Returns:
        Context index

    H.264 Spec Reference: Section 9.3.3.1.1.9
    """
    # Base context indices for coded_block_flag
    # Table 9-39 in H.264 spec
    CBF_LUMA_DC_BASE = 85
    CBF_LUMA_AC_BASE = 89
    CBF_CHROMA_DC_BASE = 93
    CBF_CHROMA_AC_BASE = 97

    # Select base context based on block_cat or block_type
    if block_type is not None:
        if block_type == 'luma_dc':
            ctx_base = CBF_LUMA_DC_BASE
        elif block_type == 'luma_ac':
            ctx_base = CBF_LUMA_AC_BASE
        elif block_type == 'chroma_dc':
            ctx_base = CBF_CHROMA_DC_BASE
        elif block_type == 'chroma_ac':
            ctx_base = CBF_CHROMA_AC_BASE
        else:
            ctx_base = CBF_LUMA_AC_BASE
    else:
        # Derive from block_cat
        # 0=Luma DC (I_16x16), 1=Luma AC, 2=Luma 4x4
        # 3=Chroma DC, 4=Chroma AC
        if block_cat == 0:
            ctx_base = CBF_LUMA_DC_BASE
        elif block_cat in (1, 2):
            ctx_base = CBF_LUMA_AC_BASE
        elif block_cat == 3:
            ctx_base = CBF_CHROMA_DC_BASE
        else:
            ctx_base = CBF_CHROMA_AC_BASE

    # Context increment based on neighbors (condTermFlagA + condTermFlagB)
    # If neighbor unavailable, treat as 0
    cond_a = 1 if (left_available and left_cbf) else 0
    cond_b = 1 if (top_available and top_cbf) else 0
    ctx_inc = cond_a + cond_b

    return ctx_base + ctx_inc
