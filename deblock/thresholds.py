# h264/deblock/thresholds.py
"""Deblocking filter threshold tables.

H.264 Spec Reference: Section 8.7.2.2 - Tables 8-16, 8-17
"""

# Alpha threshold table indexed by indexA (0-51)
# From H.264 Table 8-16
ALPHA_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     # QP 0-9
    0, 0, 0, 0, 0, 0, 4, 4, 5, 6,     # QP 10-19
    7, 8, 9, 10, 12, 13, 15, 17, 20, 22,  # QP 20-29
    25, 28, 32, 36, 40, 45, 50, 56, 63, 71,  # QP 30-39
    80, 90, 101, 113, 127, 144, 162, 182, 203, 226,  # QP 40-49
    255, 255,  # QP 50-51
]

# Beta threshold table indexed by indexB (0-51)
# From H.264 Table 8-16
BETA_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     # QP 0-9
    0, 0, 0, 0, 0, 0, 2, 2, 2, 3,     # QP 10-19
    3, 3, 3, 4, 4, 4, 6, 6, 7, 7,     # QP 20-29
    8, 8, 9, 9, 10, 10, 11, 11, 12, 12,  # QP 30-39
    13, 13, 14, 14, 15, 15, 16, 16, 17, 17,  # QP 40-49
    18, 18,  # QP 50-51
]

# tc0 table indexed by indexA and bS
# From H.264 Table 8-17
# tc0[indexA][bS] for bS = 1, 2, 3 (bS=0 means no filter, bS=4 uses strong filter)
TC0_TABLE = [
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],  # 0-5
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],  # 6-11
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1],  # 12-17
    [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1],  # 18-23
    [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 2], [1, 1, 2], [1, 1, 2],  # 24-29
    [1, 1, 2], [1, 2, 3], [1, 2, 3], [2, 2, 3], [2, 2, 4], [2, 3, 4],  # 30-35
    [2, 3, 4], [3, 3, 5], [3, 4, 6], [3, 4, 6], [4, 5, 7], [4, 5, 8],  # 36-41
    [4, 6, 9], [5, 7, 10], [6, 8, 11], [6, 8, 13], [7, 10, 14], [8, 11, 16],  # 42-47
    [9, 12, 18], [10, 13, 20], [11, 15, 23], [13, 17, 25],  # 48-51
]


def get_effective_qp(base_qp: int, offset: int) -> int:
    """Calculate effective QP with offset, clamped to valid range.

    Args:
        base_qp: Base quantization parameter
        offset: Slice alpha/beta offset

    Returns:
        Effective QP clamped to [0, 51]
    """
    return max(0, min(51, base_qp + offset))


def get_alpha(index_a: int) -> int:
    """Get alpha threshold for given index."""
    return ALPHA_TABLE[max(0, min(51, index_a))]


def get_beta(index_b: int) -> int:
    """Get beta threshold for given index."""
    return BETA_TABLE[max(0, min(51, index_b))]


def get_tc0(index_a: int, bs: int) -> int:
    """Get tc0 value for given index and boundary strength.

    Args:
        index_a: Index into threshold table
        bs: Boundary strength (1-3)

    Returns:
        tc0 value
    """
    if bs < 1 or bs > 3:
        return 0
    index_a = max(0, min(51, index_a))
    return TC0_TABLE[index_a][bs - 1]
