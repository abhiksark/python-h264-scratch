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
