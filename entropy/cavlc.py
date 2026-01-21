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
from typing import List, Tuple, Optional

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
)

logger = logging.getLogger(__name__)


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
) -> Tuple[int, int]:
    """Decode coeff_token using VLC table lookup.

    Args:
        reader: Bit reader
        decode_table: VLC decode table
        max_bits: Maximum code length

    Returns:
        Tuple of (TotalCoeff, TrailingOnes)
    """
    # Try progressively longer codes
    code = 0
    for num_bits in range(1, max_bits + 1):
        code = (code << 1) | reader.read_bit()

        if (code, num_bits) in decode_table:
            tc, t1 = decode_table[(code, num_bits)]
            logger.debug(f"coeff_token: code={code:0{num_bits}b}, TC={tc}, T1={t1}")
            return tc, t1

    raise ValueError(f"Invalid coeff_token: no match found after {max_bits} bits")


def _decode_coeff_token_fixed(reader: BitReader) -> Tuple[int, int]:
    """Decode coeff_token using fixed-length code (nC >= 8).

    For nC >= 8, coeff_token uses 6-bit fixed length:
    - Bits 5-4: TrailingOnes
    - Bits 3-0: TotalCoeff - 1 (if TotalCoeff > 0)
    - All zeros if TotalCoeff = 0

    Args:
        reader: Bit reader

    Returns:
        Tuple of (TotalCoeff, TrailingOnes)
    """
    code = reader.read_bits(6)

    if code == 3:  # Special case: TotalCoeff = 0
        return 0, 0

    trailing_ones = (code >> 4) & 0x3
    total_coeff_minus1 = code & 0x0F
    total_coeff = total_coeff_minus1 + 1

    # Validate
    if trailing_ones > total_coeff:
        trailing_ones = total_coeff

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

    H.264 Spec: Section 9.2.1
    """
    if nC == -1:
        # Chroma DC
        return _decode_coeff_token_vlc(reader, COEFF_TOKEN_DECODE_CHROMA_DC, 8)
    elif nC < 2:
        return _decode_coeff_token_vlc(reader, COEFF_TOKEN_DECODE_0, 16)
    elif nC < 4:
        return _decode_coeff_token_vlc(reader, COEFF_TOKEN_DECODE_2, 14)
    elif nC < 8:
        return _decode_coeff_token_vlc(reader, COEFF_TOKEN_DECODE_4, 10)
    else:
        return _decode_coeff_token_fixed(reader)


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
        # Decode level_prefix (number of leading zeros)
        level_prefix = 0
        while reader.read_bit() == 0:
            level_prefix += 1
            if level_prefix > 15:
                raise ValueError("level_prefix too large")

        # Decode level_suffix
        # H.264 Spec 9.2.2: Determine suffix size and compute level_code
        if level_prefix == 14 and suffix_length == 0:
            # Special case: suffix is 4 bits
            suffix_bits = 4
            level_suffix = reader.read_bits(suffix_bits)
        elif level_prefix >= 15:
            # Extended case: suffix size = level_prefix - 3
            suffix_bits = level_prefix - 3
            level_suffix = reader.read_bits(suffix_bits)
        else:
            # Normal case
            suffix_bits = suffix_length
            if suffix_bits > 0:
                level_suffix = reader.read_bits(suffix_bits)
            else:
                level_suffix = 0

        # H.264 Spec: levelCode = (Min(15, level_prefix) << suffixLength) + level_suffix
        level_code = (min(15, level_prefix) << suffix_length) + level_suffix

        # Additional adjustment for escape codes
        if level_prefix >= 15 and suffix_length == 0:
            level_code += 15

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

        logger.debug(f"Level {i}: prefix={level_prefix}, suffix={level_suffix}, "
                    f"code={level_code}, level={level}, suffix_len={suffix_length}")

    return levels


def _decode_total_zeros_vlc(
    reader: BitReader,
    decode_table: dict,
    max_bits: int = 9
) -> int:
    """Decode total_zeros using VLC table."""
    code = 0
    for num_bits in range(1, max_bits + 1):
        code = (code << 1) | reader.read_bit()

        if (code, num_bits) in decode_table:
            total_zeros = decode_table[(code, num_bits)]
            logger.debug(f"total_zeros: code={code:0{num_bits}b}, value={total_zeros}")
            return total_zeros

    raise ValueError(f"Invalid total_zeros: no match found")


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

    if max_coeffs == 4:
        # Chroma DC
        decode_table = TOTAL_ZEROS_DECODE_CHROMA_DC.get(total_coeff)
    else:
        # 4x4 block
        decode_table = TOTAL_ZEROS_DECODE_4x4.get(total_coeff)

    if decode_table is None:
        raise ValueError(f"No total_zeros table for TotalCoeff={total_coeff}")

    return _decode_total_zeros_vlc(reader, decode_table)


def _decode_run_before_vlc(
    reader: BitReader,
    decode_table: dict,
    max_bits: int = 11
) -> int:
    """Decode run_before using VLC table."""
    code = 0
    for num_bits in range(1, max_bits + 1):
        code = (code << 1) | reader.read_bit()

        if (code, num_bits) in decode_table:
            run = decode_table[(code, num_bits)]
            logger.debug(f"run_before: code={code:0{num_bits}b}, value={run}")
            return run

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

    # Use table for zerosLeft, capped at 7
    table_idx = min(zeros_left, 7)
    decode_table = RUN_BEFORE_DECODE.get(table_idx)

    if decode_table is None:
        raise ValueError(f"No run_before table for zerosLeft={zeros_left}")

    return _decode_run_before_vlc(reader, decode_table)


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
    # Step 1: Decode coeff_token
    total_coeff, trailing_ones = decode_coeff_token(reader, nC)

    # Empty block
    if total_coeff == 0:
        return CAVLCBlock(
            total_coeff=0,
            trailing_ones=0,
            coefficients=np.zeros(max_coeffs, dtype=np.int32)
        )

    # Step 2: Decode trailing ones signs
    t1_signs = decode_trailing_ones_signs(reader, trailing_ones)

    # Step 3: Decode levels
    levels = decode_levels(reader, total_coeff, trailing_ones)

    # Combine levels and trailing ones (in reverse scan order)
    # levels[0] is highest freq, t1_signs[-1] is highest freq T1
    all_coeffs = levels + [s for s in t1_signs]

    # Step 4: Decode total_zeros
    if total_coeff < max_coeffs:
        total_zeros = decode_total_zeros(reader, total_coeff, max_coeffs)
    else:
        total_zeros = 0

    # Step 5: Decode run_before for each coefficient
    zeros_left = total_zeros
    run_befores = []

    for i in range(total_coeff - 1):
        if zeros_left > 0:
            run = decode_run_before(reader, zeros_left)
            run_befores.append(run)
            zeros_left -= run
        else:
            run_befores.append(0)

    # Last coefficient gets remaining zeros
    run_befores.append(zeros_left)

    # Build output array
    coefficients = np.zeros(max_coeffs, dtype=np.int32)

    # Place coefficients in reverse scan order
    pos = 0
    for i in range(total_coeff - 1, -1, -1):
        pos += run_befores[total_coeff - 1 - i]
        if pos < max_coeffs:
            coefficients[pos] = all_coeffs[i]
        pos += 1

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
