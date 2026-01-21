# h264/inter/p_macroblock.py
"""P-macroblock type definitions and parsing.

Defines macroblock partition types for P-slices and provides
parsing utilities.

H.264 Spec Reference: Table 7-13 (mb_type for P slices)
                      Table 7-17 (sub_mb_type for P slices)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class PMBType:
    """P-macroblock type definition.

    Attributes:
        name: Type name (e.g., "P_L0_16x16")
        num_partitions: Number of motion partitions (1, 2, or 4)
        partition_width: Width of each partition in pixels
        partition_height: Height of each partition in pixels
        ref_idx_forced: If not None, ref_idx is this value (for P_8x8ref0)
        is_intra: True if this is an I-MB embedded in P-slice
        intra_type: For is_intra=True, the I-MB type index
    """
    name: str
    num_partitions: int
    partition_width: int
    partition_height: int
    ref_idx_forced: Optional[int] = None
    is_intra: bool = False
    intra_type: int = 0


@dataclass
class SubMBType:
    """Sub-macroblock type for P_8x8 partitions.

    Attributes:
        name: Type name (e.g., "P_L0_8x8")
        width: Width of each sub-partition
        height: Height of each sub-partition
        num_parts: Number of sub-partitions within this 8x8 block
    """
    name: str
    width: int
    height: int
    num_parts: int


# P-macroblock types (Table 7-13)
P_MB_TYPES = [
    PMBType("P_L0_16x16", num_partitions=1, partition_width=16, partition_height=16),
    PMBType("P_L0_L0_16x8", num_partitions=2, partition_width=16, partition_height=8),
    PMBType("P_L0_L0_8x16", num_partitions=2, partition_width=8, partition_height=16),
    PMBType("P_8x8", num_partitions=4, partition_width=8, partition_height=8),
    PMBType("P_8x8ref0", num_partitions=4, partition_width=8, partition_height=8, ref_idx_forced=0),
]

# Sub-macroblock types for P_8x8 (Table 7-17)
SUB_MB_TYPES = [
    SubMBType("P_L0_8x8", width=8, height=8, num_parts=1),
    SubMBType("P_L0_8x4", width=8, height=4, num_parts=2),
    SubMBType("P_L0_4x8", width=4, height=8, num_parts=2),
    SubMBType("P_L0_4x4", width=4, height=4, num_parts=4),
]


def parse_p_mb_type(mb_type_code: int) -> PMBType:
    """Parse mb_type code in P-slice context.

    In P-slices:
        mb_type 0-4: P-MB types
        mb_type 5+: I-MB types (offset by 5)

    Args:
        mb_type_code: The mb_type value from bitstream

    Returns:
        PMBType describing the macroblock type

    H.264 Spec: Table 7-13
    """
    if mb_type_code < 5:
        return P_MB_TYPES[mb_type_code]

    # I-MB in P-slice (mb_type 5-30 maps to I_4x4, I_16x16, etc.)
    intra_type = mb_type_code - 5
    return PMBType(
        name=f"I_in_P_{intra_type}",
        num_partitions=1,
        partition_width=16,
        partition_height=16,
        is_intra=True,
        intra_type=intra_type,
    )


@dataclass
class PMacroblockInfo:
    """Parsed information for a P-macroblock.

    Attributes:
        mb_type: The macroblock type
        ref_idx: Reference indices for each partition
        mvd: Motion vector differences for each partition (mvdx, mvdy)
        is_skip: True if this is a P_Skip macroblock
        sub_mb_types: For P_8x8, the sub-MB types for each 8x8 block
    """
    mb_type: PMBType = field(default_factory=lambda: P_MB_TYPES[0])
    ref_idx: List[int] = field(default_factory=list)
    mvd: List[Tuple[int, int]] = field(default_factory=list)
    is_skip: bool = False
    sub_mb_types: List[SubMBType] = field(default_factory=list)

    @classmethod
    def create_skip(cls) -> "PMacroblockInfo":
        """Create info for P_Skip macroblock.

        P_Skip has:
        - No explicit mb_type (implied)
        - ref_idx = 0
        - mvd = (0, 0) - actual MV equals prediction
        - No residual data
        """
        return cls(
            mb_type=P_MB_TYPES[0],  # Treated like P_L0_16x16 for motion
            ref_idx=[0],
            mvd=[(0, 0)],
            is_skip=True,
        )

    def get_partition_positions(self) -> List[Tuple[int, int, int, int]]:
        """Get (x, y, width, height) for each partition within the macroblock.

        Returns:
            List of (x, y, w, h) tuples, where x,y are relative to MB origin
        """
        if self.mb_type.num_partitions == 1:
            # 16x16
            return [(0, 0, 16, 16)]

        elif self.mb_type.num_partitions == 2:
            w = self.mb_type.partition_width
            h = self.mb_type.partition_height

            if h == 8:
                # 16x8: top and bottom
                return [(0, 0, 16, 8), (0, 8, 16, 8)]
            else:
                # 8x16: left and right
                return [(0, 0, 8, 16), (8, 0, 8, 16)]

        else:
            # 8x8: four partitions
            return [
                (0, 0, 8, 8),   # Top-left
                (8, 0, 8, 8),   # Top-right
                (0, 8, 8, 8),   # Bottom-left
                (8, 8, 8, 8),   # Bottom-right
            ]

    def get_num_mvs(self) -> int:
        """Get total number of motion vectors needed.

        For most types this equals num_partitions, but for P_8x8 with
        sub-partitions it may be more.
        """
        if self.mb_type.name in ("P_8x8", "P_8x8ref0") and self.sub_mb_types:
            return sum(st.num_parts for st in self.sub_mb_types)
        return self.mb_type.num_partitions
