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

import numpy as np

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


def parse_sub_mb_type(sub_mb_type_code: int) -> SubMBType:
    """Parse sub_mb_type code for P_8x8 sub-macroblocks.

    Args:
        sub_mb_type_code: The sub_mb_type value from bitstream (0-3)

    Returns:
        SubMBType describing the sub-macroblock partition

    H.264 Spec: Table 7-17
    """
    if sub_mb_type_code < 0 or sub_mb_type_code > 3:
        logger.warning(f"Invalid sub_mb_type {sub_mb_type_code}, using 8x8")
        sub_mb_type_code = 0

    sub_type = SUB_MB_TYPES[sub_mb_type_code]
    # Return with expected interface (num_partitions alias)
    return SubMBType(
        name=sub_type.name,
        width=sub_type.width,
        height=sub_type.height,
        num_parts=sub_type.num_parts,
    )


# Alias for test compatibility
parse_sub_mb_type.__doc__ = """Parse sub_mb_type for P_8x8.

Returns SubMBType with:
- name: e.g., "P_L0_8x8"
- partition_width: width of sub-partition
- partition_height: height of sub-partition
- num_partitions: number of sub-partitions
"""


# Add property aliases for SubMBType compatibility
SubMBType.partition_width = property(lambda self: self.width)
SubMBType.partition_height = property(lambda self: self.height)
SubMBType.num_partitions = property(lambda self: self.num_parts)


# Inter CBP mapping table (Table 9-4 in H.264 spec)
# Maps codeNum to actual CBP value for inter macroblocks
# CBP = (chroma_cbp << 4) | luma_cbp
INTER_CBP_TABLE = [
    0,   # codeNum 0 -> CBP 0
    16,  # codeNum 1 -> CBP 16 (chroma DC only)
    1,   # codeNum 2 -> CBP 1
    2,   # codeNum 3 -> CBP 2
    4,   # codeNum 4 -> CBP 4
    8,   # codeNum 5 -> CBP 8
    32,  # codeNum 6 -> CBP 32 (chroma DC+AC)
    3,   # codeNum 7 -> CBP 3
    5,   # codeNum 8 -> CBP 5
    10,  # codeNum 9 -> CBP 10
    12,  # codeNum 10 -> CBP 12
    15,  # codeNum 11 -> CBP 15
    47,  # codeNum 12 -> CBP 47
    7,   # codeNum 13 -> CBP 7
    11,  # codeNum 14 -> CBP 11
    13,  # codeNum 15 -> CBP 13
    14,  # codeNum 16 -> CBP 14
    6,   # codeNum 17 -> CBP 6
    9,   # codeNum 18 -> CBP 9
    31,  # codeNum 19 -> CBP 31
    35,  # codeNum 20 -> CBP 35
    37,  # codeNum 21 -> CBP 37
    42,  # codeNum 22 -> CBP 42
    44,  # codeNum 23 -> CBP 44
    33,  # codeNum 24 -> CBP 33
    34,  # codeNum 25 -> CBP 34
    36,  # codeNum 26 -> CBP 36
    40,  # codeNum 27 -> CBP 40
    39,  # codeNum 28 -> CBP 39
    43,  # codeNum 29 -> CBP 43
    45,  # codeNum 30 -> CBP 45
    46,  # codeNum 31 -> CBP 46
    17,  # codeNum 32 -> CBP 17
    18,  # codeNum 33 -> CBP 18
    20,  # codeNum 34 -> CBP 20
    24,  # codeNum 35 -> CBP 24
    19,  # codeNum 36 -> CBP 19
    21,  # codeNum 37 -> CBP 21
    26,  # codeNum 38 -> CBP 26
    28,  # codeNum 39 -> CBP 28
    23,  # codeNum 40 -> CBP 23
    27,  # codeNum 41 -> CBP 27
    29,  # codeNum 42 -> CBP 29
    30,  # codeNum 43 -> CBP 30
    22,  # codeNum 44 -> CBP 22
    25,  # codeNum 45 -> CBP 25
    38,  # codeNum 46 -> CBP 38
    41,  # codeNum 47 -> CBP 41
]


@dataclass
class PResidual:
    """Residual data for a P-macroblock.

    Attributes:
        luma: 16x16 luma residual or None
        cb: 8x8 Cb residual or None
        cr: 8x8 Cr residual or None
    """
    luma: Optional[np.ndarray] = None
    cb: Optional[np.ndarray] = None
    cr: Optional[np.ndarray] = None


def decode_inter_cbp(reader) -> int:
    """Decode coded_block_pattern for inter macroblock from bitstream.

    Args:
        reader: BitReader positioned at CBP

    Returns:
        CBP value (0-47)

    H.264 Spec: Table 9-4
    """
    code_num = reader.read_ue()
    if code_num >= len(INTER_CBP_TABLE):
        logger.warning(f"Invalid inter CBP codeNum {code_num}, using 0")
        return 0
    return INTER_CBP_TABLE[code_num]


def parse_p_mb_qp_delta(reader) -> int:
    """Parse mb_qp_delta for P-macroblock.

    Args:
        reader: BitReader positioned at mb_qp_delta

    Returns:
        QP delta value (signed)
    """
    return reader.read_se()


def parse_p_residual(reader, cbp: int, qp: int) -> PResidual:
    """Parse residual data for P-macroblock.

    Args:
        reader: BitReader positioned at residual data
        cbp: coded_block_pattern value
        qp: Current QP value

    Returns:
        PResidual with decoded coefficients
    """
    if cbp == 0:
        return PResidual()

    # TODO: Implement full residual parsing
    # For now, return empty residual
    return PResidual()


def get_luma_4x4_indices_for_8x8(block_8x8: int) -> List[int]:
    """Get 4x4 block indices that comprise an 8x8 block.

    8x8 block layout in macroblock:
        0 | 1
        -----
        2 | 3

    4x4 block scan order:
        0  1  4  5
        2  3  6  7
        8  9 12 13
       10 11 14 15

    Args:
        block_8x8: 8x8 block index (0-3)

    Returns:
        List of four 4x4 block indices
    """
    # Map 8x8 block to top-left 4x4 block
    base_indices = {
        0: [0, 1, 4, 5],     # Top-left 8x8
        1: [2, 3, 6, 7],     # Top-right 8x8
        2: [8, 9, 12, 13],   # Bottom-left 8x8
        3: [10, 11, 14, 15], # Bottom-right 8x8
    }
    return base_indices[block_8x8]


def get_4x4_position(block_idx: int) -> Tuple[int, int]:
    """Get (x, y) position of 4x4 block within macroblock.

    Uses raster scan order within each 8x8 block:
        Block indices:     Positions:
        0  1  4  5        (0,0) (4,0) (0,4) (4,4)
        2  3  6  7        (8,0) (12,0) (8,4) (12,4)
        8  9 12 13        (0,8) (4,8) (0,12) (4,12)
       10 11 14 15        (8,8) (12,8) (8,12) (12,12)

    Args:
        block_idx: 4x4 block index (0-15)

    Returns:
        (x, y) position in pixels within macroblock
    """
    # 4x4 block positions (x, y) for indices 0-15
    positions = [
        (0, 0),   (4, 0),   (0, 4),   (4, 4),    # 8x8 block 0
        (8, 0),   (12, 0),  (8, 4),   (12, 4),   # 8x8 block 1
        (0, 8),   (4, 8),   (0, 12),  (4, 12),   # 8x8 block 2
        (8, 8),   (12, 8),  (8, 12),  (12, 12),  # 8x8 block 3
    ]
    return positions[block_idx]


def should_parse_chroma_dc(chroma_cbp: int) -> bool:
    """Check if chroma DC coefficients should be parsed.

    Args:
        chroma_cbp: Chroma CBP value (0, 1, or 2)

    Returns:
        True if DC coefficients should be parsed
    """
    return chroma_cbp >= 1


def should_parse_chroma_ac(chroma_cbp: int) -> bool:
    """Check if chroma AC coefficients should be parsed.

    Args:
        chroma_cbp: Chroma CBP value (0, 1, or 2)

    Returns:
        True if AC coefficients should be parsed
    """
    return chroma_cbp >= 2


def decode_cbp_inter(cbp: int) -> Tuple[List[bool], int]:
    """Decode coded_block_pattern for inter macroblocks.

    For inter MBs, CBP indicates which 8x8 luma blocks and chroma
    components have residual data.

    CBP structure:
        - Bits 0-3: Luma CBP (one bit per 8x8 block)
        - Bits 4-5: Chroma CBP (0=none, 1=DC only, 2=DC+AC)

    Args:
        cbp: The coded_block_pattern value

    Returns:
        Tuple of:
            - luma_cbp: List of 4 booleans (one per 8x8 block)
            - chroma_cbp: 0=none, 1=DC only, 2=DC+AC

    H.264 Spec: Section 7.4.5 - coded_block_pattern semantics
    """
    # Luma: bits 0-3
    luma_cbp = [
        bool(cbp & 1),        # 8x8 block 0 (top-left)
        bool(cbp & 2),        # 8x8 block 1 (top-right)
        bool(cbp & 4),        # 8x8 block 2 (bottom-left)
        bool(cbp & 8),        # 8x8 block 3 (bottom-right)
    ]

    # Chroma: bits 4-5
    chroma_cbp = (cbp >> 4) & 3

    return luma_cbp, chroma_cbp
