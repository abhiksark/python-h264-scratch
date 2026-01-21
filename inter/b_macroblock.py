# h264/inter/b_macroblock.py
"""B-macroblock type handling for B-frames.

B-macroblocks support more prediction modes than P-macroblocks:
- Direct mode: MVs derived from neighbors or co-located
- L0 only: Forward prediction (like P-frames)
- L1 only: Backward prediction
- Bi-prediction: Average of L0 and L1

H.264 Spec Reference: Table 7-14 - Macroblock types for B slices
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BMacroblockInfo:
    """B-macroblock type information.

    Attributes:
        name: Human-readable MB type name
        mb_type: Raw mb_type value
        is_direct: True if using direct mode (MVs not transmitted)
        pred_mode: "L0", "L1", "Bi", or "Direct"
        num_partitions: Number of partitions (1, 2, or 4)
        partition_size: (width, height) of each partition
        is_skip: True if this is B_Skip (from mb_skip_run)
        is_intra: True if intra-coded in B-slice
    """
    name: str
    mb_type: int
    is_direct: bool = False
    pred_mode: str = "L0"
    num_partitions: int = 1
    partition_size: Optional[Tuple[int, int]] = None
    is_skip: bool = False
    is_intra: bool = False


@dataclass
class BSubMBInfo:
    """B sub-macroblock type information for B_8x8.

    Attributes:
        name: Human-readable sub-MB type name
        sub_mb_type: Raw sub_mb_type value
        is_direct: True if using direct mode
        pred_mode: "L0", "L1", "Bi", or "Direct"
        num_partitions: Number of sub-partitions (1, 2, or 4)
        partition_size: (width, height) of each sub-partition
    """
    name: str
    sub_mb_type: int
    is_direct: bool = False
    pred_mode: str = "L0"
    num_partitions: int = 1
    partition_size: Tuple[int, int] = (8, 8)


# B-macroblock type table (H.264 Table 7-14)
# Format: (name, is_direct, pred_mode, num_partitions, partition_size)
_B_MB_TYPES = {
    0: ("B_Direct_16x16", True, "Direct", 1, (16, 16)),
    1: ("B_L0_16x16", False, "L0", 1, (16, 16)),
    2: ("B_L1_16x16", False, "L1", 1, (16, 16)),
    3: ("B_Bi_16x16", False, "Bi", 1, (16, 16)),
    4: ("B_L0_L0_16x8", False, "L0", 2, (16, 8)),
    5: ("B_L0_L0_8x16", False, "L0", 2, (8, 16)),
    6: ("B_L1_L1_16x8", False, "L1", 2, (16, 8)),
    7: ("B_L1_L1_8x16", False, "L1", 2, (8, 16)),
    8: ("B_L0_L1_16x8", False, "L0_L1", 2, (16, 8)),
    9: ("B_L0_L1_8x16", False, "L0_L1", 2, (8, 16)),
    10: ("B_L1_L0_16x8", False, "L1_L0", 2, (16, 8)),
    11: ("B_L1_L0_8x16", False, "L1_L0", 2, (8, 16)),
    12: ("B_L0_Bi_16x8", False, "L0_Bi", 2, (16, 8)),
    13: ("B_L0_Bi_8x16", False, "L0_Bi", 2, (8, 16)),
    14: ("B_L1_Bi_16x8", False, "L1_Bi", 2, (16, 8)),
    15: ("B_L1_Bi_8x16", False, "L1_Bi", 2, (8, 16)),
    16: ("B_Bi_L0_16x8", False, "Bi_L0", 2, (16, 8)),
    17: ("B_Bi_L0_8x16", False, "Bi_L0", 2, (8, 16)),
    18: ("B_Bi_L1_16x8", False, "Bi_L1", 2, (16, 8)),
    19: ("B_Bi_L1_8x16", False, "Bi_L1", 2, (8, 16)),
    20: ("B_Bi_Bi_16x8", False, "Bi", 2, (16, 8)),
    21: ("B_Bi_Bi_8x16", False, "Bi", 2, (8, 16)),
    22: ("B_8x8", False, "Sub", 4, (8, 8)),
}

# B sub-macroblock type table (H.264 Table 7-17)
_B_SUB_MB_TYPES = {
    0: ("B_Direct_8x8", True, "Direct", 1, (8, 8)),
    1: ("B_L0_8x8", False, "L0", 1, (8, 8)),
    2: ("B_L1_8x8", False, "L1", 1, (8, 8)),
    3: ("B_Bi_8x8", False, "Bi", 1, (8, 8)),
    4: ("B_L0_8x4", False, "L0", 2, (8, 4)),
    5: ("B_L0_4x8", False, "L0", 2, (4, 8)),
    6: ("B_L1_8x4", False, "L1", 2, (8, 4)),
    7: ("B_L1_4x8", False, "L1", 2, (4, 8)),
    8: ("B_Bi_8x4", False, "Bi", 2, (8, 4)),
    9: ("B_Bi_4x8", False, "Bi", 2, (4, 8)),
    10: ("B_L0_4x4", False, "L0", 4, (4, 4)),
    11: ("B_L1_4x4", False, "L1", 4, (4, 4)),
    12: ("B_Bi_4x4", False, "Bi", 4, (4, 4)),
}


def parse_b_mb_type(mb_type_code: int) -> BMacroblockInfo:
    """Parse B-slice macroblock type from mb_type code.

    Args:
        mb_type_code: Decoded mb_type value (0-22 for inter, 23+ for intra)

    Returns:
        BMacroblockInfo describing the macroblock

    H.264 Spec: Table 7-14
    """
    # Intra types in B-slice start at mb_type=23
    if mb_type_code >= 23:
        # Map to I-slice mb_type: 23->0 (I_4x4), 24->1-24 (I_16x16), etc.
        i_mb_type = mb_type_code - 23
        if i_mb_type == 0:
            return BMacroblockInfo(
                name="I_4x4",
                mb_type=mb_type_code,
                pred_mode="Intra",
                is_intra=True,
            )
        elif i_mb_type <= 24:
            return BMacroblockInfo(
                name="I_16x16",
                mb_type=mb_type_code,
                pred_mode="Intra",
                is_intra=True,
            )
        else:
            return BMacroblockInfo(
                name="I_PCM",
                mb_type=mb_type_code,
                pred_mode="Intra",
                is_intra=True,
            )

    if mb_type_code not in _B_MB_TYPES:
        raise ValueError(f"Invalid B-slice mb_type: {mb_type_code}")

    name, is_direct, pred_mode, num_parts, part_size = _B_MB_TYPES[mb_type_code]

    return BMacroblockInfo(
        name=name,
        mb_type=mb_type_code,
        is_direct=is_direct,
        pred_mode=pred_mode,
        num_partitions=num_parts,
        partition_size=part_size,
    )


def parse_b_sub_mb_type(sub_mb_type_code: int) -> BSubMBInfo:
    """Parse B_8x8 sub-macroblock type.

    Args:
        sub_mb_type_code: Decoded sub_mb_type value (0-12)

    Returns:
        BSubMBInfo describing the sub-macroblock

    H.264 Spec: Table 7-17
    """
    if sub_mb_type_code not in _B_SUB_MB_TYPES:
        raise ValueError(f"Invalid B sub_mb_type: {sub_mb_type_code}")

    name, is_direct, pred_mode, num_parts, part_size = _B_SUB_MB_TYPES[sub_mb_type_code]

    return BSubMBInfo(
        name=name,
        sub_mb_type=sub_mb_type_code,
        is_direct=is_direct,
        pred_mode=pred_mode,
        num_partitions=num_parts,
        partition_size=part_size,
    )


def is_b_skip(mb_skip_flag: bool) -> bool:
    """Check if macroblock is B_Skip.

    B_Skip is signaled via mb_skip_run, not mb_type.
    It's equivalent to B_Direct_16x16 with no residual.

    Args:
        mb_skip_flag: True if MB was part of mb_skip_run

    Returns:
        True if this is a B_Skip macroblock
    """
    return mb_skip_flag


def get_b_skip_info() -> BMacroblockInfo:
    """Get BMacroblockInfo for B_Skip.

    Returns:
        BMacroblockInfo representing B_Skip
    """
    return BMacroblockInfo(
        name="B_Skip",
        mb_type=-1,  # Special value - not from mb_type
        is_direct=True,
        pred_mode="Direct",
        num_partitions=1,
        partition_size=(16, 16),
        is_skip=True,
    )


def get_partition_pred_modes(mb_info: BMacroblockInfo) -> list:
    """Get prediction mode for each partition.

    For mixed-mode types like B_L0_L1_16x8, returns separate modes.

    Args:
        mb_info: B-macroblock info

    Returns:
        List of prediction modes, one per partition
    """
    pred_mode = mb_info.pred_mode

    if mb_info.num_partitions == 1:
        return [pred_mode]

    # Split combined modes
    if "_" in pred_mode and pred_mode not in ("L0", "L1"):
        parts = pred_mode.split("_")
        return parts[:mb_info.num_partitions]

    # Same mode for all partitions
    return [pred_mode] * mb_info.num_partitions
