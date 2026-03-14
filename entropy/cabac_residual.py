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
    CTX_SIG_COEFF_FLAG_8X8_START,
    CTX_LAST_SIG_COEFF_8X8_START,
    CTX_COEFF_ABS_LEVEL_8X8_START,
    CTX_TRANSFORM_8X8_FLAG_START,
)
from entropy.tables import ZIGZAG_8x8

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
    5: (0, 0, 0),    # Luma 8x8 (uses separate 8x8 context bases)
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

        # All subsequent bins use same context (H.264 Table 9-34)
        # Chroma DC (blockCat 3) has only 4 abs contexts (max_c2=3 per JM)
        max_c2 = 3 if block_cat == 3 else 4
        ctx_idx_inc = 5 + min(max_c2, num_decode_abs_level_gt1)
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
        # Loop completed without finding last=1.
        # Last position is implicitly significant (H.264 Section 9.3.3.1.3).
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
    # Base context indices for coded_block_flag (H.264 Table 9-12)
    # ctxIdx = 85 + 4 * blockCat + ctxIdxInc
    CBF_BASE = {
        0: 85,   # Luma DC (I_16x16)
        1: 89,   # Luma AC (I_16x16)
        2: 93,   # Luma 4x4
        3: 97,   # Chroma DC
        4: 101,  # Chroma AC
    }

    # Select base context
    if block_type is not None:
        block_type_to_cat = {
            'luma_dc': 0, 'luma_ac': 1, 'luma_4x4': 2,
            'chroma_dc': 3, 'chroma_ac': 4,
        }
        cat = block_type_to_cat.get(block_type, block_cat)
        ctx_base = CBF_BASE.get(cat, 89)
    else:
        ctx_base = CBF_BASE.get(block_cat, 89)

    # Context increment: condTermFlagA + 2 * condTermFlagB (H.264 9.3.3.1.1.9)
    cond_a = 1 if (left_available and left_cbf) else 0
    cond_b = 1 if (top_available and top_cbf) else 0
    ctx_inc = cond_a + 2 * cond_b

    return ctx_base + ctx_inc


# =============================================================================
# 8x8 block support (High Profile)
# =============================================================================

# Block category constants for 8x8
BLOCK_CAT_8X8_LUMA = 5
BLOCK_CAT_8X8_CB = 7
BLOCK_CAT_8X8_CR = 8

# 8x8 zigzag scan for CABAC (same as CAVLC, from H.264 Table 8-15)
ZIGZAG_8X8_CABAC = list(ZIGZAG_8x8)

# 8x8 field scan (H.264 Table 8-14) - interlaced video
ZIGZAG_8X8_FIELD = [
    0,  8, 16,  1,  9, 24, 32, 17,
    2, 25, 40, 48, 33, 10,  3, 18,
   26, 41, 56, 49, 34, 11,  4, 19,
   27, 42, 57, 50, 35, 12,  5, 20,
   28, 43, 58, 51, 36, 13,  6, 21,
   29, 44, 59, 52, 37, 14,  7, 22,
   30, 45, 60, 53, 38, 15, 23, 31,
   46, 61, 54, 39, 47, 62, 55, 63,
]

# Context index offsets for 8x8 block categories
# For 8x8, contexts use absolute ctxIdx bases (not relative to 4x4 ranges)
# Format: (sig_base, last_base, abs_base) - absolute ctxIdx values
BLOCK_CAT_CTX_OFFSETS_8X8 = {
    5: (0, 0, 0),  # Luma 8x8: offsets within 8x8 ranges
}

# H.264 Table 9-43: ctxIdxInc for significant_coeff_flag (8x8, frame scan)
# 63 entries for scan positions 0-62 (position 63 is implicit)
SIG_COEFF_FLAG_CTX_INC_8X8 = [
    0,  1,  2,  3,  4,  5,  5,  4,  4,  3,  3,  4,  4,  4,  5,  5,
    4,  4,  4,  4,  3,  3,  6,  7,  7,  7,  8,  9, 10,  9,  8,  7,
    7,  6, 11, 12, 13, 11,  6,  7,  8,  9, 14, 10,  9,  8,  6, 11,
   12, 13, 11,  6,  9, 14, 10,  9, 11, 12, 13, 11, 14, 10, 12,
]

# H.264 Table 9-43: ctxIdxInc for last_significant_coeff_flag (8x8, frame scan)
LAST_SIG_COEFF_FLAG_CTX_INC_8X8 = [
    0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,
    5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7,  8,  8,  8,
]


def is_8x8_block_category(block_cat: int) -> bool:
    """Check if block category is an 8x8 type.

    Args:
        block_cat: Block category

    Returns:
        True for 8x8 categories (5, 7, 8)
    """
    return block_cat in (5, 7, 8)


def get_max_coeff_for_category(block_cat: int) -> int:
    """Get maximum number of coefficients for block category.

    Args:
        block_cat: Block category (0-8)

    Returns:
        Maximum coefficients for this category
    """
    cat_to_max = {
        0: 16,  # Luma DC (I_16x16)
        1: 15,  # Luma AC (I_16x16)
        2: 16,  # Luma 4x4
        3: 4,   # Chroma DC (4:2:0)
        4: 15,  # Chroma AC
        5: 64,  # Luma 8x8
        7: 64,  # Cb 8x8 (4:4:4)
        8: 64,  # Cr 8x8 (4:4:4)
    }
    return cat_to_max.get(block_cat, 16)


def get_scan_order_8x8(field_scan: bool = False) -> List[int]:
    """Get 8x8 scan order.

    Args:
        field_scan: True for field (interlaced) scan, False for frame scan

    Returns:
        List of 64 scan positions
    """
    if field_scan:
        return list(ZIGZAG_8X8_FIELD)
    return list(ZIGZAG_8X8_CABAC)


def get_sig_coeff_flag_8x8_ctx_inc(scan_idx: int) -> int:
    """Get ctxIdxInc for significant_coeff_flag in 8x8 block.

    H.264 Table 9-43.

    Args:
        scan_idx: Position in scan order (0-62)

    Returns:
        ctxIdxInc value
    """
    if scan_idx >= len(SIG_COEFF_FLAG_CTX_INC_8X8):
        return SIG_COEFF_FLAG_CTX_INC_8X8[-1]
    return SIG_COEFF_FLAG_CTX_INC_8X8[scan_idx]


def get_last_sig_coeff_8x8_ctx_idx(scan_idx: int) -> int:
    """Get ctxIdxInc for last_significant_coeff_flag in 8x8 block.

    H.264 Table 9-43.

    Args:
        scan_idx: Position in scan order (0-62)

    Returns:
        ctxIdxInc value
    """
    if scan_idx >= len(LAST_SIG_COEFF_FLAG_CTX_INC_8X8):
        return LAST_SIG_COEFF_FLAG_CTX_INC_8X8[-1]
    return LAST_SIG_COEFF_FLAG_CTX_INC_8X8[scan_idx]


def get_coeff_abs_level_8x8_ctx_inc(
    num_eq1: int,
    num_gt1: int,
    bin_idx: int = 0,
) -> int:
    """Get ctxIdxInc for coeff_abs_level_minus1 in 8x8 block.

    Same logic as 4x4 but used with 8x8 context base.

    Args:
        num_eq1: Count of level==1 decoded so far
        num_gt1: Count of level>1 decoded so far
        bin_idx: Bin index (0 for first bin, >0 for subsequent)

    Returns:
        ctxIdxInc value
    """
    if bin_idx == 0:
        if num_gt1 > 0:
            return 0
        return min(4, 1 + num_eq1)
    return 5 + min(4, num_gt1)


def decode_significant_coeff_flag_8x8(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    block_cat: int,
    scan_idx: int,
) -> int:
    """Decode significant_coeff_flag for 8x8 block.

    Uses Table 9-43 ctxIdxInc mapping.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        block_cat: Block category (5, 7, or 8)
        scan_idx: Position in scan order (0-62)

    Returns:
        0 or 1
    """
    ctx_inc = get_sig_coeff_flag_8x8_ctx_inc(scan_idx)
    ctx_idx = CTX_SIG_COEFF_FLAG_8X8_START + ctx_inc
    return decoder.decode_decision(contexts[ctx_idx])


def decode_last_significant_coeff_flag_8x8(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    block_cat: int,
    scan_idx: int,
) -> int:
    """Decode last_significant_coeff_flag for 8x8 block.

    Uses Table 9-43 ctxIdxInc mapping.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        block_cat: Block category (5, 7, or 8)
        scan_idx: Position in scan order (0-62)

    Returns:
        0 or 1
    """
    ctx_inc = get_last_sig_coeff_8x8_ctx_idx(scan_idx)
    ctx_idx = CTX_LAST_SIG_COEFF_8X8_START + ctx_inc
    return decoder.decode_decision(contexts[ctx_idx])


def decode_coeff_abs_level_minus1_8x8(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    block_cat: int,
    num_decode_abs_level_eq1: int,
    num_decode_abs_level_gt1: int,
) -> int:
    """Decode coeff_abs_level_minus1 for 8x8 block.

    Same algorithm as 4x4 but uses 8x8 context base (ctxIdx 402+).

    Args:
        decoder: CABAC decoder
        contexts: Context models
        block_cat: Block category (5, 7, or 8)
        num_decode_abs_level_eq1: Count of level==1 decoded so far
        num_decode_abs_level_gt1: Count of level>1 decoded so far

    Returns:
        coeff_abs_level_minus1 (>= 0)
    """
    ctx_base = CTX_COEFF_ABS_LEVEL_8X8_START

    ctx_idx_inc = get_coeff_abs_level_8x8_ctx_inc(
        num_decode_abs_level_eq1, num_decode_abs_level_gt1, bin_idx=0
    )
    ctx_idx = ctx_base + ctx_idx_inc

    # Decode prefix
    prefix = 0
    while prefix < 14:
        if decoder.decode_decision(contexts[ctx_idx]) == 0:
            break
        prefix += 1
        ctx_idx_inc = 5 + min(4, num_decode_abs_level_gt1)
        ctx_idx = ctx_base + ctx_idx_inc

    if prefix < 14:
        return prefix

    # Decode suffix using exp-golomb bypass
    k = 0
    while decoder.decode_bypass() == 1:
        k += 1

    suffix = 0
    for _ in range(k):
        suffix = (suffix << 1) | decoder.decode_bypass()

    suffix += (1 << k) - 1
    return 14 + suffix


def decode_significance_map_8x8(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    block_cat: int,
) -> List[int]:
    """Decode significance map for 8x8 block.

    Returns a 64-element list of 0/1 flags indicating which positions
    have non-zero coefficients.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        block_cat: Block category (5, 7, or 8)

    Returns:
        List of 64 significance flags
    """
    sig_map = [0] * 64

    for i in range(63):  # Positions 0-62
        sig = decode_significant_coeff_flag_8x8(decoder, contexts, block_cat, i)
        sig_map[i] = sig

        if sig == 1:
            last = decode_last_significant_coeff_flag_8x8(
                decoder, contexts, block_cat, i
            )
            if last == 1:
                return sig_map

    # Reached position 63: implicitly significant
    sig_map[63] = 1
    return sig_map


def decode_residual_block_8x8(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    block_cat: int = 5,
) -> np.ndarray:
    """Decode an 8x8 residual coefficient block using CABAC.

    No coded_block_flag for 8x8: CBP at MB level gates whether this
    function is called (H.264 Section 7.4.5.3).

    Args:
        decoder: CABAC decoder
        contexts: Context models
        block_cat: Block category (5=Luma 8x8, 7=Cb 8x8, 8=Cr 8x8)

    Returns:
        Array of 64 coefficients in scan order

    H.264 Spec Reference: Section 9.3.3.1.3
    """
    coeffs = np.zeros(64, dtype=np.int32)

    # Decode significance map and collect significant positions
    sig_positions = []

    for i in range(63):  # Positions 0-62
        sig = decode_significant_coeff_flag_8x8(decoder, contexts, block_cat, i)

        if sig == 1:
            sig_positions.append(i)

            last = decode_last_significant_coeff_flag_8x8(
                decoder, contexts, block_cat, i
            )
            if last == 1:
                break
    else:
        # Position 63 is implicitly significant
        sig_positions.append(63)

    # Decode coefficient levels (in reverse scan order)
    num_eq1 = 0
    num_gt1 = 0

    for pos in reversed(sig_positions):
        level_m1 = decode_coeff_abs_level_minus1_8x8(
            decoder, contexts, block_cat, num_eq1, num_gt1
        )
        level = level_m1 + 1

        sign = decode_coeff_sign_flag(decoder)
        if sign == 1:
            level = -level

        coeffs[pos] = level

        if abs(level) == 1:
            num_eq1 += 1
        else:
            num_gt1 += 1

    return coeffs


def get_transform_8x8_flag_ctx_inc(
    left_available: bool,
    left_8x8: int,
    top_available: bool,
    top_8x8: int,
) -> int:
    """Get ctxIdxInc for transform_size_8x8_flag.

    ctxIdxInc = condTermFlagA + condTermFlagB
    condTermFlag = transform_size_8x8_flag of neighbor (0 if unavailable)

    Args:
        left_available: Whether left neighbor is available
        left_8x8: Left neighbor's transform_size_8x8_flag (0 or 1)
        top_available: Whether top neighbor is available
        top_8x8: Top neighbor's transform_size_8x8_flag (0 or 1)

    Returns:
        ctxIdxInc (0, 1, or 2)

    H.264 Spec Reference: Section 9.3.3.1.1.10
    """
    cond_a = left_8x8 if left_available else 0
    cond_b = top_8x8 if top_available else 0
    return cond_a + cond_b


def decode_transform_8x8_flag(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    neighbor_info: dict,
) -> int:
    """Decode transform_size_8x8_flag using CABAC.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        neighbor_info: Dict with keys:
            left_transform_8x8_flag, top_transform_8x8_flag,
            left_available, top_available

    Returns:
        0 (4x4 transform) or 1 (8x8 transform)

    H.264 Spec Reference: Section 7.3.5 / Table 9-11
    """
    ctx_inc = get_transform_8x8_flag_ctx_inc(
        left_available=neighbor_info.get('left_available', True),
        left_8x8=neighbor_info.get('left_transform_8x8_flag', 0),
        top_available=neighbor_info.get('top_available', True),
        top_8x8=neighbor_info.get('top_transform_8x8_flag', 0),
    )
    ctx_idx = CTX_TRANSFORM_8X8_FLAG_START + ctx_inc
    return decoder.decode_decision(contexts[ctx_idx])


def get_coded_block_flag_8x8_ctx_inc(
    left_available: bool,
    left_cbf: int,
    top_available: bool,
    top_cbf: int,
) -> int:
    """Get ctxIdxInc for coded_block_flag of 8x8 block.

    condTermFlagN = coded_block_flag if neighbor available, else 1.
    ctxIdxInc = condTermFlagA + condTermFlagB.

    Args:
        left_available: Whether left neighbor is available
        left_cbf: Left neighbor coded_block_flag
        top_available: Whether top neighbor is available
        top_cbf: Top neighbor coded_block_flag

    Returns:
        ctxIdxInc (0, 1, or 2)

    H.264 Spec Reference: Section 9.3.3.1.1.9
    """
    cond_a = left_cbf if left_available else 1
    cond_b = top_cbf if top_available else 1
    return cond_a + cond_b


def decode_coded_block_flag_8x8(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    block_cat: int,
    neighbor_info: dict,
) -> int:
    """Decode coded_block_flag for 8x8 block.

    Note: In standard High Profile 4:2:0, coded_block_flag is NOT decoded
    for 8x8 blocks (CBP serves this role). This function exists for
    completeness and potential 4:4:4 support.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        block_cat: Block category (5, 7, or 8)
        neighbor_info: Dict with left_cbf, top_cbf, availability flags

    Returns:
        0 or 1
    """
    ctx_inc = get_coded_block_flag_8x8_ctx_inc(
        left_available=neighbor_info.get('left_available', True),
        left_cbf=neighbor_info.get('left_cbf', 0),
        top_available=neighbor_info.get('top_available', True),
        top_cbf=neighbor_info.get('top_cbf', 0),
    )
    # Use luma 4x4 CBF context base (93) for 8x8 as approximation
    ctx_idx = 93 + ctx_inc
    return decoder.decode_decision(contexts[ctx_idx])
