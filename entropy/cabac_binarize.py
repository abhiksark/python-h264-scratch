# h264/entropy/cabac_binarize.py
"""CABAC binarization schemes.

Binarization maps multi-valued syntax elements to/from binary strings:
- Unary (U): Value n encoded as n ones followed by zero
- Truncated Unary (TU): Like unary but max value has no terminator
- Exp-Golomb (UEGk): Unary prefix + exp-golomb suffix for large values
- Fixed Length (FL): Fixed number of bits

H.264 Spec Reference: Section 9.3.2 - Binarization process
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from entropy.cabac_arith import CABACDecoder
    from entropy.cabac_context import CABACContext


# Context index offsets for MVD (H.264 Table 9-34)
CTX_MVD_X_BASE = 40
CTX_MVD_Y_BASE = 47

# Context index offsets for coefficient levels (H.264 Table 9-38)
CTX_COEFF_ABS_LEVEL_BASE = 227


def decode_unary(
    decoder: 'CABACDecoder',
    ctx_base: int,
    contexts: List['CABACContext'],
    max_ctx_inc: int = 2,
) -> int:
    """Decode unary binarized value.

    Unary: n encoded as n ones followed by zero.
    Example: 0 = "0", 1 = "10", 2 = "110", 3 = "1110"

    Args:
        decoder: CABAC arithmetic decoder
        ctx_base: Base context index
        contexts: List of context models
        max_ctx_inc: Maximum context increment (bins after this use max)

    Returns:
        Decoded value
    """
    value = 0

    while True:
        # Context index: ctx_base + min(value, max_ctx_inc)
        ctx_idx = ctx_base + min(value, max_ctx_inc)
        bin_val = decoder.decode_decision(contexts[ctx_idx])

        if bin_val == 0:
            break
        value += 1

    return value


def decode_truncated_unary(
    decoder: 'CABACDecoder',
    max_val: int,
    ctx_base: int,
    contexts: List['CABACContext'],
) -> int:
    """Decode truncated unary binarized value.

    Like unary but bounded. At max_val, no terminating zero needed.
    Example (max=3): 0="0", 1="10", 2="110", 3="111"

    Args:
        decoder: CABAC arithmetic decoder
        max_val: Maximum value
        ctx_base: Base context index
        contexts: List of context models

    Returns:
        Decoded value in [0, max_val]
    """
    value = 0

    while value < max_val:
        ctx_idx = ctx_base + min(value, 2)  # Limited context increment
        bin_val = decoder.decode_decision(contexts[ctx_idx])

        if bin_val == 0:
            break
        value += 1

    return value


def decode_uegk(
    decoder: 'CABACDecoder',
    k: int,
    ctx_base: int,
    contexts: List['CABACContext'],
    uCoff: int = 14,
) -> int:
    """Decode UEGk (Unary/Exp-Golomb order k) binarized value.

    For values < uCoff: truncated unary prefix
    For values >= uCoff: TU prefix + exp-golomb suffix (bypass coded)

    Args:
        decoder: CABAC arithmetic decoder
        k: Exp-Golomb order
        ctx_base: Base context index
        contexts: List of context models
        uCoff: Cutoff for exp-golomb suffix (default 14)

    Returns:
        Decoded unsigned value
    """
    # Decode prefix (truncated unary, max = uCoff)
    prefix = 0
    while prefix < uCoff:
        ctx_idx = ctx_base + min(prefix, 4)
        bin_val = decoder.decode_decision(contexts[ctx_idx])
        if bin_val == 0:
            break
        prefix += 1

    if prefix < uCoff:
        return prefix

    # Decode suffix using bypass (exp-golomb order k)
    suffix = 0
    suffix_len = k

    # Read leading ones to determine suffix length
    while decoder.decode_bypass() == 1:
        suffix_len += 1

    # Read suffix bits
    for _ in range(suffix_len):
        suffix = (suffix << 1) | decoder.decode_bypass()

    # Add offset for this length
    suffix += (1 << suffix_len) - (1 << k)

    return uCoff + suffix


def decode_fixed_length(
    decoder: 'CABACDecoder',
    num_bits: int,
) -> int:
    """Decode fixed-length binarized value.

    Uses bypass mode for all bits.

    Args:
        decoder: CABAC arithmetic decoder
        num_bits: Number of bits to decode

    Returns:
        Decoded value
    """
    value = 0
    for _ in range(num_bits):
        value = (value << 1) | decoder.decode_bypass()
    return value


def decode_signed_value(abs_val: int, sign_flag: int) -> int:
    """Apply sign to absolute value.

    Args:
        abs_val: Absolute value
        sign_flag: 0 for positive, 1 for negative

    Returns:
        Signed value
    """
    if abs_val == 0:
        return 0
    if sign_flag:
        return -abs_val
    return abs_val


def decode_mvd(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    comp: int,
) -> int:
    """Decode motion vector difference component.

    Uses UEG3 binarization with sign.

    Args:
        decoder: CABAC arithmetic decoder
        contexts: List of context models
        comp: Component (0=x, 1=y)

    Returns:
        Signed MVD value
    """
    # Context base depends on component
    ctx_base = CTX_MVD_X_BASE if comp == 0 else CTX_MVD_Y_BASE

    # Decode absolute value using UEGk with k=3
    abs_val = decode_uegk(decoder, k=3, ctx_base=ctx_base, contexts=contexts, uCoff=9)

    if abs_val == 0:
        return 0

    # Decode sign using bypass
    sign_flag = decoder.decode_bypass()

    return decode_signed_value(abs_val, sign_flag)


def decode_coeff_abs_level(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    cat: int,
    num_decode_abs_level_eq1: int = 0,
    num_decode_abs_level_gt1: int = 0,
) -> int:
    """Decode coefficient absolute level minus 1.

    Uses context selection based on previously decoded levels.

    Args:
        decoder: CABAC arithmetic decoder
        contexts: List of context models
        cat: Block category (0-4)
        num_decode_abs_level_eq1: Count of level==1 in block
        num_decode_abs_level_gt1: Count of level>1 in block

    Returns:
        coeff_abs_level_minus1 value (actual level = return + 1)
    """
    # Context for prefix (coeff_abs_level_greater1)
    ctx_idx_inc = min(4, 1 + num_decode_abs_level_gt1)
    ctx_idx = CTX_COEFF_ABS_LEVEL_BASE + cat * 10 + ctx_idx_inc

    # Decode prefix: 0 means level=1, 1... means level>1
    prefix = 0
    while prefix < 14:
        if decoder.decode_decision(contexts[ctx_idx]) == 0:
            break
        prefix += 1
        # Update context for subsequent bins
        ctx_idx_inc = min(4, 5 + num_decode_abs_level_gt1)
        ctx_idx = CTX_COEFF_ABS_LEVEL_BASE + cat * 10 + ctx_idx_inc

    if prefix < 14:
        return prefix

    # Decode suffix using bypass (exp-golomb order 0)
    suffix = 0
    suffix_len = 0

    while decoder.decode_bypass() == 1:
        suffix_len += 1

    for _ in range(suffix_len):
        suffix = (suffix << 1) | decoder.decode_bypass()

    suffix += (1 << suffix_len) - 1

    return 14 + suffix


def decode_exp_golomb_bypass(
    decoder: 'CABACDecoder',
    k: int = 0,
) -> int:
    """Decode exp-golomb coded value using bypass mode.

    UEGk (Unsigned Exp-Golomb order k) binarization.

    Args:
        decoder: CABAC arithmetic decoder
        k: Exp-golomb order (default 0)

    Returns:
        Decoded value

    H.264 Spec Reference: Section 9.3.2.3
    """
    # Count leading zeros
    leading_zeros = 0
    while decoder.decode_bypass() == 0:
        leading_zeros += 1
        if leading_zeros > 32:
            break  # Sanity limit

    # Read (leading_zeros + k) suffix bits
    suffix_len = leading_zeros + k
    suffix = 0
    for _ in range(suffix_len):
        suffix = (suffix << 1) | decoder.decode_bypass()

    # Calculate value
    # value = (1 << leading_zeros) - 1 + suffix for order 0
    # For order k: includes the k offset
    if k == 0:
        value = ((1 << leading_zeros) - 1) + suffix
    else:
        value = ((1 << leading_zeros) - 1) * (1 << k) + suffix

    return value


def binarize_ref_idx(ref_idx: int, max_ref_idx: int) -> List[int]:
    """Binarize reference index using truncated unary.

    Args:
        ref_idx: Reference index value
        max_ref_idx: Maximum reference index (num_ref - 1)

    Returns:
        List of binary values

    H.264 Spec Reference: Section 9.3.2.1
    """
    bins = []

    # Truncated unary: output 1s up to value, then 0 (unless at max)
    for i in range(ref_idx):
        bins.append(1)

    # If not at max, append terminating 0
    if ref_idx < max_ref_idx:
        bins.append(0)

    return bins


def decode_ref_idx_bins(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    ctx_base: int,
    max_ref_idx: int,
) -> int:
    """Decode reference index using truncated unary.

    Args:
        decoder: CABAC arithmetic decoder
        contexts: Context models
        ctx_base: Base context index
        max_ref_idx: Maximum reference index

    Returns:
        Decoded reference index
    """
    if max_ref_idx == 0:
        return 0

    ref_idx = 0

    while ref_idx < max_ref_idx:
        ctx_inc = min(2, ref_idx)
        ctx_idx = ctx_base + ctx_inc

        if decoder.decode_decision(contexts[ctx_idx]) == 0:
            break
        ref_idx += 1

    return ref_idx
