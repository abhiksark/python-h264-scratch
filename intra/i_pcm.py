# h264/intra/i_pcm.py
"""I_PCM macroblock support.

I_PCM macroblocks contain raw, uncompressed sample data.
They bypass prediction, transform, and quantization.

H.264 Spec Reference: Section 7.3.5 - Macroblock layer syntax
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class IMBType:
    """Intra macroblock type."""

    name: str
    is_pcm: bool = False
    is_4x4: bool = False
    is_16x16: bool = False
    pred_mode: int = 0  # For I_16x16
    cbp_luma: int = 0   # For I_16x16
    cbp_chroma: int = 0  # For I_16x16


def parse_i_mb_type(mb_type_code: int) -> IMBType:
    """Parse I-macroblock type from mb_type code.

    Args:
        mb_type_code: Decoded mb_type value

    Returns:
        IMBType describing the macroblock

    I-slice mb_type mapping:
        0: I_4x4
        1-24: I_16x16 (various pred_mode/cbp combinations)
        25: I_PCM
    """
    if mb_type_code == 0:
        return IMBType(name="I_4x4", is_4x4=True)

    if mb_type_code == 25:
        return IMBType(name="I_PCM", is_pcm=True)

    if 1 <= mb_type_code <= 24:
        # I_16x16 encoding:
        # mb_type = 1 + pred_mode + cbp_chroma * 4 + cbp_luma_flag * 12
        idx = mb_type_code - 1
        pred_mode = idx % 4
        cbp_chroma = (idx // 4) % 3
        cbp_luma = 15 if idx >= 12 else 0  # 15 = all 4 8x8 blocks

        return IMBType(
            name="I_16x16",
            is_16x16=True,
            pred_mode=pred_mode,
            cbp_luma=cbp_luma,
            cbp_chroma=cbp_chroma,
        )

    raise ValueError(f"Invalid I-slice mb_type: {mb_type_code}")


def align_to_byte(reader) -> None:
    """Align reader to next byte boundary.

    I_PCM data starts at byte-aligned position after mb_type.

    Args:
        reader: BitReader to align
    """
    if reader.position % 8 != 0:
        bits_to_skip = 8 - (reader.position % 8)
        reader.read_bits(bits_to_skip)


def parse_i_pcm_luma(reader, bit_depth: int = 8) -> np.ndarray:
    """Parse luma samples for I_PCM macroblock.

    Args:
        reader: BitReader positioned at luma data
        bit_depth: Sample bit depth (8 or 10)

    Returns:
        16x16 luma array
    """
    luma = np.zeros((16, 16), dtype=np.uint16 if bit_depth > 8 else np.uint8)

    for y in range(16):
        for x in range(16):
            if bit_depth <= 8:
                luma[y, x] = reader.read_bits(8)
            else:
                # For 10-bit, read two bytes
                luma[y, x] = reader.read_bits(bit_depth)

    return luma


def parse_i_pcm_chroma(reader, bit_depth: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Parse chroma samples for I_PCM macroblock.

    Args:
        reader: BitReader positioned at chroma data
        bit_depth: Sample bit depth

    Returns:
        Tuple of (cb, cr) 8x8 arrays
    """
    cb = np.zeros((8, 8), dtype=np.uint16 if bit_depth > 8 else np.uint8)
    cr = np.zeros((8, 8), dtype=np.uint16 if bit_depth > 8 else np.uint8)

    # Read Cb samples
    for y in range(8):
        for x in range(8):
            if bit_depth <= 8:
                cb[y, x] = reader.read_bits(8)
            else:
                cb[y, x] = reader.read_bits(bit_depth)

    # Read Cr samples
    for y in range(8):
        for x in range(8):
            if bit_depth <= 8:
                cr[y, x] = reader.read_bits(8)
            else:
                cr[y, x] = reader.read_bits(bit_depth)

    return cb, cr


def reconstruct_i_pcm(
    luma: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct I_PCM macroblock.

    I_PCM uses raw samples directly - no prediction or transform.

    Args:
        luma: 16x16 luma samples
        cb: 8x8 Cb samples
        cr: 8x8 Cr samples

    Returns:
        Tuple of (luma, cb, cr) unchanged
    """
    return luma.copy(), cb.copy(), cr.copy()


def get_i_pcm_nz_counts() -> np.ndarray:
    """Get non-zero coefficient counts for I_PCM macroblock.

    I_PCM blocks are considered to have 16 coefficients for
    CAVLC context purposes.

    Returns:
        Array of 24 values, all set to 16
    """
    return np.full(24, 16, dtype=np.int32)
