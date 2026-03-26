# h264/entropy/cavlc.py
"""CAVLC (Context-Adaptive Variable-Length Coding) decoder for H.264.

CAVLC is the entropy coding method used in H.264 Baseline profile.
It encodes transform coefficients using context-dependent VLC tables.

H.264 Spec Reference:
- Section 9.2: CAVLC parsing process
- Section 9.2.1: Parsing coeff_token
- Section 9.2.2: Parsing level information
- Section 9.2.3: Parsing total_zeros
- Section 9.2.4: Parsing run_before

Decoding order for a block:
1. coeff_token -> TotalCoeff, TrailingOnes
2. trailing_ones_sign_flag (T1 sign bits)
3. level values (remaining coefficient magnitudes)
4. total_zeros
5. run_before (zero runs between coefficients)
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np

from bitstream import BitReader
from .tables import (
    COEFF_TOKEN_DECODE_0,
    COEFF_TOKEN_DECODE_2,
    COEFF_TOKEN_DECODE_4,
    COEFF_TOKEN_DECODE_CHROMA_DC,
    TOTAL_ZEROS_DECODE_4x4,
    TOTAL_ZEROS_DECODE_CHROMA_DC,
    RUN_BEFORE_DECODE,
    ZIGZAG_4x4,
    ZIGZAG_2x2,
    ZIGZAG_8x8,
)

logger = logging.getLogger(__name__)


def validate_vlc_bits_consumed(
    reader: BitReader,
    pos_before: int,
    expected_bits: int,
    context: str,
    code_value: Any
) -> None:
    """Validate VLC decode consumed expected number of bits.

    Args:
        reader: Bit reader after VLC decode
        pos_before: Bit position before VLC decode
        expected_bits: Expected number of bits consumed
        context: Description for error message (e.g., "coeff_token nC=2")
        code_value: The decoded value

    Raises:
        ValueError: If bit consumption doesn't match expected
    """
    actual_bits = reader.position - pos_before
    if actual_bits != expected_bits:
        raise ValueError(
            f"VLC bit consumption mismatch in {context}: "
            f"expected {expected_bits} bits, consumed {actual_bits} bits "
            f"(value={code_value}, pos_before={pos_before}, pos_after={reader.position})"
        )


def _compute_level_code(level_prefix: int, suffix_length: int, level_suffix: int) -> int:
    """Compute level_code per H.264 spec Section 9.2.2.

    Args:
        level_prefix: Number of leading zeros in level VLC
        suffix_length: Current suffix length state
        level_suffix: Decoded suffix bits

    Returns:
        level_code value

    H.264 Spec Section 9.2.2 (ITU-T H.264):
        levelCode = (Min(15, level_prefix) << suffixLength) + level_suffix
        if level_prefix >= 15 && suffixLength == 0: levelCode += 15
        if level_prefix >= 16: levelCode += (1 << (level_prefix - 3)) - 4096
    """
    level_code = (min(15, level_prefix) << suffix_length) + level_suffix
    if level_prefix >= 15 and suffix_length == 0:
        level_code += 15
    if level_prefix >= 16:
        level_code += (1 << (level_prefix - 3)) - 4096
    return level_code


@dataclass
class CAVLCBlock:
    """Result of CAVLC decoding for one block.

    Attributes:
        total_coeff: Number of non-zero coefficients (0-16)
        trailing_ones: Number of trailing ±1 coefficients (0-3)
        coefficients: Array of coefficient values in scan order
    """
    total_coeff: int
    trailing_ones: int
    coefficients: np.ndarray

    @property
    def is_empty(self) -> bool:
        """Check if block has no non-zero coefficients."""
        return self.total_coeff == 0


def _decode_coeff_token_vlc(
    reader: BitReader,
    decode_table: dict,
    max_bits: int = 16
) -> Tuple[int, int, int]:
    """Decode coeff_token using VLC table lookup.

    Args:
        reader: Bit reader
        decode_table: VLC decode table
        max_bits: Maximum code length

    Returns:
        Tuple of (TotalCoeff, TrailingOnes, num_bits_consumed)
    """
    # Try progressively longer codes
    code = 0
    for num_bits in range(1, max_bits + 1):
        code = (code << 1) | reader.read_bit()

        if (code, num_bits) in decode_table:
            tc, t1 = decode_table[(code, num_bits)]
            logger.debug(f"coeff_token: code={code:0{num_bits}b}, TC={tc}, T1={t1}")
            return tc, t1, num_bits

    raise ValueError(f"Invalid coeff_token: no match found after {max_bits} bits")


def _decode_coeff_token_fixed(reader: BitReader) -> Tuple[int, int]:
    """Decode coeff_token using fixed-length code (nC >= 8).

    For nC >= 8, coeff_token uses 6-bit fixed length per H.264 Table 9-5(e).
    The encoding is NOT a simple bit field split:
    - code 3 (000011): TC=0, T1=0
    - codes 4-5: TC=1, T1=code-4
    - codes 6-8: TC=2, T1=code-6
    - codes >= 9: TC=(code+3)//4, T1=(code+3)%4

    Args:
        reader: Bit reader

    Returns:
        Tuple of (TotalCoeff, TrailingOnes)

    H.264 Spec: Table 9-5(e)
    """
    code = reader.read_bits(6)

    if code < 3:
        raise ValueError(f"Invalid fixed coeff_token code: {code}")

    if code == 3:  # Special case: TotalCoeff = 0
        logger.debug("coeff_token (fixed): TC=0, T1=0")
        return 0, 0

    # For codes 4-5: TC=1
    if code < 6:
        total_coeff = 1
        trailing_ones = code - 4
    # For codes 6-8: TC=2
    elif code < 9:
        total_coeff = 2
        trailing_ones = code - 6
    else:
        # For codes >= 9: TC = (code+3)//4, T1 = (code+3)%4
        adjusted = code + 3
        total_coeff = adjusted >> 2
        trailing_ones = adjusted & 3

    logger.debug(f"coeff_token (fixed): TC={total_coeff}, T1={trailing_ones}")
    return total_coeff, trailing_ones


def decode_coeff_token(reader: BitReader, nC: int) -> Tuple[int, int]:
    """Decode coeff_token based on context nC.

    Args:
        reader: Bit reader
        nC: Context value (average of neighboring non-zero counts)
            -1 for chroma DC, -2 for special cases

    Returns:
        Tuple of (TotalCoeff, TrailingOnes)

    H.264 Spec: Section 9.2.1, Table 9-5

    Note: While H.264 spec says nC >= 8 should use fixed 6-bit table (Table 9-5d),
    x264 and other encoders use VLC table 4 for all nC >= 4. We follow this for
    maximum compatibility.
    """
    pos_before = reader.position
    logger.debug(f"decode_coeff_token: nC={nC}")

    if nC == -1:
        # Chroma DC
        total_coeff, trailing_ones, expected_bits = _decode_coeff_token_vlc(
            reader, COEFF_TOKEN_DECODE_CHROMA_DC, 8)
    elif nC < 2:
        total_coeff, trailing_ones, expected_bits = _decode_coeff_token_vlc(
            reader, COEFF_TOKEN_DECODE_0, 16)
    elif nC < 4:
        total_coeff, trailing_ones, expected_bits = _decode_coeff_token_vlc(
            reader, COEFF_TOKEN_DECODE_2, 14)
    elif nC < 8:
        total_coeff, trailing_ones, expected_bits = _decode_coeff_token_vlc(
            reader, COEFF_TOKEN_DECODE_4, 10)
    else:
        # H.264 Table 9-5(d): Fixed 6-bit encoding for nC >= 8
        # Per JM reference: T1 = code & 3 (bottom 2 bits),
        # TC = (code >> 2) + 1 (top 4 bits + 1)
        # Special case: code >> 2 == 0 AND code & 3 == 3 → TC=0, T1=0
        code = reader.read_bits(6)
        expected_bits = 6
        trailing_ones = code & 3
        tc_pre = code >> 2
        if tc_pre == 0 and trailing_ones == 3:
            total_coeff, trailing_ones = 0, 0
        else:
            total_coeff = tc_pre + 1
        logger.debug(f"Fixed 6-bit: code={code:06b}, TC={total_coeff}, T1={trailing_ones}")

    # Validate bit consumption
    validate_vlc_bits_consumed(
        reader, pos_before, expected_bits,
        context=f"coeff_token nC={nC}",
        code_value=(total_coeff, trailing_ones)
    )

    return total_coeff, trailing_ones


def decode_trailing_ones_signs(
    reader: BitReader,
    trailing_ones: int
) -> List[int]:
    """Decode signs for trailing ones.

    Each trailing one has a 1-bit sign (0 = +1, 1 = -1).
    Signs are read in reverse order.

    Args:
        reader: Bit reader
        trailing_ones: Number of trailing ones (0-3)

    Returns:
        List of signs (+1 or -1) in coefficient order
    """
    if trailing_ones == 0:
        return []

    # Read signs in reverse order
    signs = []
    for _ in range(trailing_ones):
        sign_bit = reader.read_bit()
        signs.append(-1 if sign_bit else 1)

    # Reverse to get coefficient order
    signs.reverse()

    logger.debug(f"Trailing ones signs: {signs}")
    return signs


def decode_levels(
    reader: BitReader,
    total_coeff: int,
    trailing_ones: int
) -> List[int]:
    """Decode level values for non-trailing coefficients.

    Levels are decoded using an adaptive VLC with suffix length
    that increases as larger levels are encountered.

    Args:
        reader: Bit reader
        total_coeff: Total non-zero coefficients
        trailing_ones: Number of trailing ones already decoded

    Returns:
        List of level values (magnitudes with signs)

    H.264 Spec: Section 9.2.2
    """
    num_levels = total_coeff - trailing_ones
    if num_levels == 0:
        return []

    levels = []

    # Initial suffix length
    # If more than 10 coefficients and less than 3 trailing ones, start with 1
    if total_coeff > 10 and trailing_ones < 3:
        suffix_length = 1
    else:
        suffix_length = 0

    for i in range(num_levels):
        pos_before = reader.position

        # Decode level_prefix (number of leading zeros)
        level_prefix = 0
        while reader.read_bit() == 0:
            level_prefix += 1
            if level_prefix > 31:
                raise ValueError("level_prefix too large")

        # Decode level_suffix
        # H.264 Spec 9.2.2: Determine suffix size based on level_prefix
        if level_prefix == 14 and suffix_length == 0:
            suffix_bits = 4
        elif level_prefix >= 15:
            # Escape: level_prefix - 3 bits (per ITU-T H.264 Section 9.2.2)
            suffix_bits = level_prefix - 3
        else:
            suffix_bits = suffix_length

        if suffix_bits > 0:
            level_suffix = reader.read_bits(suffix_bits)
        else:
            level_suffix = 0

        # Validate bit consumption: level_prefix zeros + 1 terminator + suffix_bits
        expected_bits = level_prefix + 1 + suffix_bits
        actual_bits = reader.position - pos_before
        if actual_bits != expected_bits:
            raise ValueError(
                f"Level decode bit mismatch: expected {expected_bits} bits "
                f"(prefix={level_prefix}, suffix_bits={suffix_bits}), "
                f"but consumed {actual_bits} bits (pos_before={pos_before}, pos_after={reader.position})"
            )

        # H.264 Spec 9.2.2: Compute level_code (see Table 9-7)
        level_code = _compute_level_code(level_prefix, suffix_length, level_suffix)

        # Convert level_code to signed level
        if level_code % 2 == 0:
            level = (level_code + 2) >> 1
        else:
            level = -(level_code + 1) >> 1

        # First level must have magnitude > 1 if trailing_ones < 3
        if i == 0 and trailing_ones < 3:
            if level > 0:
                level += 1
            else:
                level -= 1

        levels.append(level)

        # Update suffix_length based on decoded level
        if suffix_length == 0:
            suffix_length = 1

        if suffix_length < 6:
            threshold = 3 << (suffix_length - 1) if suffix_length > 0 else 0
            if abs(levels[-1]) > threshold:
                suffix_length += 1

        logger.debug(f"Level {i}: prefix={level_prefix}, suffix_bits={suffix_bits}, suffix={level_suffix}, "
                    f"code={level_code}, level={level}, consumed={expected_bits} bits, bit_pos={reader.position}")

    return levels


def _decode_total_zeros_vlc(
    reader: BitReader,
    decode_table: dict,
    max_bits: int = 11
) -> Tuple[int, int]:
    """Decode total_zeros using VLC table.

    Args:
        reader: Bit reader
        decode_table: VLC decode table
        max_bits: Maximum code length (default 11 for 4x4 blocks)

    Returns:
        Tuple of (total_zeros_value, num_bits_consumed)

    Note: Max code length is 11 bits for TC=2 in 4x4 blocks (H.264 Table 9-8)
    """
    code = 0
    for num_bits in range(1, max_bits + 1):
        code = (code << 1) | reader.read_bit()

        if (code, num_bits) in decode_table:
            total_zeros = decode_table[(code, num_bits)]
            logger.debug(f"total_zeros: code={code:0{num_bits}b}, value={total_zeros}")
            return total_zeros, num_bits

    # Debug: show what codes are valid for this table
    valid_codes = sorted(decode_table.keys(), key=lambda x: (x[1], x[0]))
    logger.error(f"Invalid total_zeros: read {max_bits} bits, code={code:0{max_bits}b}")
    logger.error(f"Valid codes: {[(f'{c:0{b}b}', b) for c, b in valid_codes[:10]]}")
    raise ValueError(f"Invalid total_zeros: no match found (code={code:0{max_bits}b})")


def decode_total_zeros(
    reader: BitReader,
    total_coeff: int,
    max_coeffs: int = 16
) -> int:
    """Decode total_zeros value.

    total_zeros is the number of zero coefficients before
    the last non-zero coefficient.

    Args:
        reader: Bit reader
        total_coeff: Number of non-zero coefficients
        max_coeffs: Maximum coefficients in block (16 for 4x4, 4 for chroma DC)

    Returns:
        Number of total zeros

    H.264 Spec: Section 9.2.3
    """
    if total_coeff == max_coeffs:
        return 0  # No room for zeros

    pos_before = reader.position
    max_valid_zeros = max_coeffs - total_coeff
    logger.debug(f"decode_total_zeros: TC={total_coeff}, max={max_coeffs}, max_valid_zeros={max_valid_zeros}")

    if max_coeffs == 4:
        # Chroma DC
        decode_table = TOTAL_ZEROS_DECODE_CHROMA_DC.get(total_coeff)
    else:
        # 4x4 block
        decode_table = TOTAL_ZEROS_DECODE_4x4.get(total_coeff)

    if decode_table is None:
        raise ValueError(f"No total_zeros table for TotalCoeff={total_coeff}")

    total_zeros, expected_bits = _decode_total_zeros_vlc(reader, decode_table)

    # Validate bit consumption
    validate_vlc_bits_consumed(
        reader, pos_before, expected_bits,
        context=f"total_zeros TC={total_coeff}",
        code_value=total_zeros
    )

    # Validate: total_zeros cannot exceed max_valid_zeros
    # Some encoders may produce technically invalid streams that FFmpeg tolerates
    # by clamping. We do the same for compatibility.
    if total_zeros > max_valid_zeros:
        logger.warning(f"total_zeros={total_zeros} exceeds max_valid={max_valid_zeros} "
                      f"(TC={total_coeff}, max_coeffs={max_coeffs}), clamping")
        total_zeros = max_valid_zeros

    return total_zeros


def _decode_run_before_vlc(
    reader: BitReader,
    decode_table: dict,
    max_bits: int = 12
) -> Tuple[int, int]:
    """Decode run_before using VLC table.

    Args:
        reader: Bit reader
        decode_table: VLC decode table
        max_bits: Maximum code length (default 12, for run_before=15)

    Returns:
        Tuple of (run_value, num_bits_consumed)
    """
    code = 0
    for num_bits in range(1, max_bits + 1):
        code = (code << 1) | reader.read_bit()

        if (code, num_bits) in decode_table:
            run = decode_table[(code, num_bits)]
            logger.debug(f"run_before: code={code:0{num_bits}b}, value={run}")
            return run, num_bits

    raise ValueError(f"Invalid run_before: no match found")


def decode_run_before(
    reader: BitReader,
    zeros_left: int
) -> int:
    """Decode run_before value.

    run_before is the number of zeros before the current coefficient.

    Args:
        reader: Bit reader
        zeros_left: Remaining zeros to distribute

    Returns:
        Number of zeros before this coefficient

    H.264 Spec: Section 9.2.4
    """
    if zeros_left == 0:
        return 0

    pos_before = reader.position

    # Use table for zerosLeft, capped at 7
    table_idx = min(zeros_left, 7)
    decode_table = RUN_BEFORE_DECODE.get(table_idx)

    if decode_table is None:
        raise ValueError(f"No run_before table for zerosLeft={zeros_left}")

    run, expected_bits = _decode_run_before_vlc(reader, decode_table)

    # Validate bit consumption
    validate_vlc_bits_consumed(
        reader, pos_before, expected_bits,
        context=f"run_before zeros_left={zeros_left}",
        code_value=run
    )

    return run


def decode_residual_block(
    reader: BitReader,
    nC: int,
    max_coeffs: int = 16
) -> CAVLCBlock:
    """Decode a complete residual block using CAVLC.

    Args:
        reader: Bit reader positioned at start of block data
        nC: Context value for coeff_token table selection
        max_coeffs: Maximum coefficients (16 for 4x4, 4 for chroma DC)

    Returns:
        CAVLCBlock with decoded coefficients

    H.264 Spec: Section 9.2
    """
    start_pos = reader.position if hasattr(reader, 'position') else -1
    logger.debug(f"decode_residual_block: nC={nC}, max={max_coeffs}, bit_pos={start_pos}")

    # Step 1: Decode coeff_token
    total_coeff, trailing_ones = decode_coeff_token(reader, nC)
    pos_after_token = reader.position if hasattr(reader, 'position') else -1
    logger.debug(f"[COEFF_TOKEN] TC={total_coeff}, T1={trailing_ones}, bit_pos={pos_after_token}")

    # Empty block
    if total_coeff == 0:
        logger.debug(f"[EARLY_RETURN] total_coeff=0, returning empty block")
        return CAVLCBlock(
            total_coeff=0,
            trailing_ones=0,
            coefficients=np.zeros(max_coeffs, dtype=np.int32)
        )

    # Step 2: Decode trailing ones signs
    t1_signs = decode_trailing_ones_signs(reader, trailing_ones)
    pos_after_t1 = reader.position if hasattr(reader, 'position') else -1
    logger.debug(f"After T1 signs: bit_pos={pos_after_t1}")

    # Step 3: Decode levels
    levels = decode_levels(reader, total_coeff, trailing_ones)
    pos_after_levels = reader.position if hasattr(reader, 'position') else -1
    logger.debug(f"After levels: bit_pos={pos_after_levels}")

    # Combine levels and trailing ones in SCAN order (low position to high)
    # levels are in decode order (high to low), so reverse them
    # t1_signs are already in scan order after reversal in decode_trailing_ones_signs
    # T1s are at higher scan positions than non-T1s, so they come after
    all_coeffs = list(reversed(levels)) + list(t1_signs)

    # Step 4: Decode total_zeros
    if total_coeff < max_coeffs:
        total_zeros = decode_total_zeros(reader, total_coeff, max_coeffs)
    else:
        total_zeros = 0

    pos_after_tz = reader.position if hasattr(reader, 'position') else -1
    logger.debug(f"After total_zeros: bit_pos={pos_after_tz}, TZ={total_zeros}")

    # Step 5: Decode run_before for each coefficient
    zeros_left = total_zeros
    run_befores = []

    for i in range(total_coeff - 1):
        if zeros_left > 0:
            run = decode_run_before(reader, zeros_left)
            if run > zeros_left:
                # Encoder bug or corrupted bitstream - clamp to valid range
                logger.warning(f"run_before={run} > zeros_left={zeros_left} at coeff {i}, clamping to {zeros_left}")
                run = zeros_left
            run_befores.append(run)
            zeros_left -= run
        else:
            run_befores.append(0)

    # Last coefficient gets remaining zeros
    run_befores.append(zeros_left)

    pos_after_run = reader.position if hasattr(reader, 'position') else -1
    logger.debug(f"After run_before: bit_pos={pos_after_run}, runs={run_befores}")

    # Build output array by placing coefficients from high to low scan position
    # H.264 Spec: Coefficients are placed starting at highest position, working down
    coefficients = np.zeros(max_coeffs, dtype=np.int32)

    pos = total_coeff + total_zeros - 1  # Start at highest position
    for i in range(total_coeff):
        # all_coeffs is in scan order (low to high), so index TC-1-i gives decode order
        coeff_idx = total_coeff - 1 - i
        if 0 <= pos < max_coeffs:
            coefficients[pos] = all_coeffs[coeff_idx]
        pos -= 1 + run_befores[i]  # Move left by 1 + zeros before this coeff

    logger.debug(f"Decoded block: TC={total_coeff}, T1={trailing_ones}, "
                f"TZ={total_zeros}, coeffs={coefficients[:total_coeff+total_zeros]}")

    return CAVLCBlock(
        total_coeff=total_coeff,
        trailing_ones=trailing_ones,
        coefficients=coefficients
    )


def decode_residual_4x4(
    reader: BitReader,
    nC: int
) -> np.ndarray:
    """Decode 4x4 residual block and return in raster order.

    Args:
        reader: Bit reader
        nC: Context value

    Returns:
        4x4 coefficient array in raster order
    """
    block = decode_residual_block(reader, nC, max_coeffs=16)

    # Convert from scan order to raster order
    coeffs_4x4 = np.zeros((4, 4), dtype=np.int32)
    for i, pos in enumerate(ZIGZAG_4x4):
        row, col = pos // 4, pos % 4
        coeffs_4x4[row, col] = block.coefficients[i]

    return coeffs_4x4


def decode_chroma_dc(
    reader: BitReader
) -> np.ndarray:
    """Decode 2x2 chroma DC block.

    Args:
        reader: Bit reader

    Returns:
        2x2 coefficient array
    """
    block = decode_residual_block(reader, nC=-1, max_coeffs=4)

    # Convert from scan order to raster order
    coeffs_2x2 = np.zeros((2, 2), dtype=np.int32)
    for i, pos in enumerate(ZIGZAG_2x2):
        row, col = pos // 2, pos % 2
        coeffs_2x2[row, col] = block.coefficients[i]

    return coeffs_2x2


def decode_luma_dc_16x16(
    reader: BitReader,
    nC: int
) -> np.ndarray:
    """Decode 4x4 luma DC block for I16x16 macroblock.

    Args:
        reader: Bit reader
        nC: Context value

    Returns:
        4x4 DC coefficient array
    """
    block = decode_residual_block(reader, nC, max_coeffs=16)

    # Convert from scan order to raster order
    dc_4x4 = np.zeros((4, 4), dtype=np.int32)
    for i, pos in enumerate(ZIGZAG_4x4):
        row, col = pos // 4, pos % 4
        dc_4x4[row, col] = block.coefficients[i]

    return dc_4x4


def calculate_nC(
    nA: Optional[int],
    nB: Optional[int]
) -> int:
    """Calculate context value nC from neighboring block counts.

    Args:
        nA: Non-zero count from left neighbor (None if unavailable)
        nB: Non-zero count from top neighbor (None if unavailable)

    Returns:
        Context value nC

    H.264 Spec: Section 9.2.1
    """
    if nA is not None and nB is not None:
        return (nA + nB + 1) >> 1
    elif nA is not None:
        return nA
    elif nB is not None:
        return nB
    else:
        return 0


def decode_residual_8x8(
    reader: BitReader,
    nC: int
) -> np.ndarray:
    """Decode 8x8 residual block and return in raster order.

    For High profile 8x8 transforms with CAVLC entropy coding.
    Uses the 8x8 diagonal zigzag scan pattern.

    Args:
        reader: Bit reader
        nC: Context value for coeff_token table selection

    Returns:
        8x8 coefficient array in raster order (int32)

    H.264 Spec: Section 9.2, 7.4.5.3.3
    Note: 8x8 CAVLC uses same coeff_token tables as 4x4.
    For full implementation, total_zeros tables 9-9a to 9-9g are needed.
    """
    block = decode_residual_block(reader, nC, max_coeffs=64)

    # Convert from scan order to raster order using 8x8 zigzag
    coeffs_8x8 = np.zeros((8, 8), dtype=np.int32)
    for i, pos in enumerate(ZIGZAG_8x8):
        row, col = pos // 8, pos % 8
        coeffs_8x8[row, col] = block.coefficients[i]

    return coeffs_8x8
