# h264/parameters/scaling.py
"""H.264 scaling list handling for quantization.

H.264 Spec Reference:
- Section 7.3.2.1.1: Scaling list syntax
- Section 7.4.2.1.1: Scaling list semantics
- Section 8.5.9: Derivation process for scaling functions
- Table 7-3: Default_4x4_Intra and Default_4x4_Inter
- Table 7-4: Default_8x8_Intra and Default_8x8_Inter
"""

import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

from bitstream import BitReader

if TYPE_CHECKING:
    from parameters.sps import SPS
    from parameters.pps import PPS

logger = logging.getLogger(__name__)


# H.264 Spec Table 7-3: Default scaling lists for 4x4 blocks
DEFAULT_4x4_INTRA = [
    6, 13, 13, 20, 20, 20, 28, 28, 28, 28, 32, 32, 32, 37, 37, 42
]

DEFAULT_4x4_INTER = [
    10, 14, 14, 20, 20, 20, 24, 24, 24, 24, 27, 27, 27, 30, 30, 34
]

# H.264 Spec Table 7-4: Default scaling lists for 8x8 blocks
DEFAULT_8x8_INTRA = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23,
    23, 23, 23, 23, 23, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27,
    27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42
]

DEFAULT_8x8_INTER = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21,
    21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 24, 24, 24, 24,
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35
]

# Flat scaling list (all 16s) - used when seq_scaling_matrix_present_flag=0
FLAT_4x4 = [16] * 16
FLAT_8x8 = [16] * 64


def parse_scaling_list(data: bytes, size: int) -> Tuple[List[int], bool]:
    """Parse a scaling list from bitstream.

    H.264 Spec: Section 7.3.2.1.1 scaling_list()

    Args:
        data: Raw bytes containing the scaling list
        size: Size of the scaling list (16 for 4x4, 64 for 8x8)

    Returns:
        Tuple of (scaling_list, use_default_flag)
        - scaling_list: List of scaling values
        - use_default_flag: True if default scaling matrix should be used
    """
    reader = BitReader(data)
    scaling_list = []
    last_scale = 8
    next_scale = 8

    for j in range(size):
        if next_scale != 0:
            delta_scale = reader.read_se()
            next_scale = (last_scale + delta_scale + 256) % 256

            if j == 0 and next_scale == 0:
                # use_default_scaling_matrix_flag inferred
                return [], True

        if next_scale == 0:
            scaling_list.append(last_scale)
        else:
            scaling_list.append(next_scale)
            last_scale = next_scale

    return scaling_list, False


def _is_high_profile(profile_idc: int) -> bool:
    """Check if profile is High or above (has scaling matrix support)."""
    return profile_idc in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135)


def get_scaling_list_with_default(
    parsed_list: Optional[List[int]],
    use_default_flag: bool,
    list_idx: int,
) -> List[int]:
    """Get scaling list, using spec default if use_default_flag is set.

    H.264 Spec: Section 7.4.2.1.1

    Args:
        parsed_list: Parsed scaling list or None
        use_default_flag: If True, use spec-defined default
        list_idx: List index (0-5 for 4x4, 6-7 for 8x8)

    Returns:
        Appropriate scaling list
    """
    if use_default_flag or parsed_list is None:
        # Use spec-defined defaults based on list index
        if list_idx < 3:
            return DEFAULT_4x4_INTRA.copy()
        elif list_idx < 6:
            return DEFAULT_4x4_INTER.copy()
        elif list_idx == 6:
            return DEFAULT_8x8_INTRA.copy()
        else:  # list_idx == 7
            return DEFAULT_8x8_INTER.copy()

    return list(parsed_list)


def get_effective_scaling_list_4x4(
    sps_list: Optional[List[Optional[List[int]]]],
    pps_list: Optional[List[Optional[List[int]]]],
    list_idx: int,
    use_default_flag: bool = False,
) -> List[int]:
    """Get effective 4x4 scaling list with fallback rules.

    H.264 Spec: Section 7.4.2.1.1

    Args:
        sps_list: List of SPS scaling lists (may contain None for absent lists)
        pps_list: List of PPS scaling lists (may contain None)
        list_idx: Index 0-5 for 4x4 lists
        use_default_flag: If True and list is absent, use spec default

    Returns:
        List of 16 scaling values
    """
    # PPS overrides SPS
    if pps_list is not None and len(pps_list) > list_idx and pps_list[list_idx] is not None:
        return list(pps_list[list_idx])

    # Check SPS
    if sps_list is not None and len(sps_list) > list_idx and sps_list[list_idx] is not None:
        return list(sps_list[list_idx])

    # Apply fallback rules
    if use_default_flag:
        # Use spec-defined defaults
        if list_idx < 3:
            return DEFAULT_4x4_INTRA.copy()
        else:
            return DEFAULT_4x4_INTER.copy()

    # Fallback to previous list
    if list_idx == 0:
        return DEFAULT_4x4_INTRA.copy()
    elif list_idx == 3:
        return DEFAULT_4x4_INTER.copy()
    else:
        return get_effective_scaling_list_4x4(sps_list, pps_list, list_idx - 1, use_default_flag)


def get_effective_scaling_list_8x8(
    sps_list: Optional[List[Optional[List[int]]]],
    pps_list: Optional[List[Optional[List[int]]]],
    list_idx: int,
    use_default_flag: bool = False,
) -> List[int]:
    """Get effective 8x8 scaling list with fallback rules.

    H.264 Spec: Section 7.4.2.1.1

    Args:
        sps_list: List of SPS 8x8 scaling lists (may contain None)
        pps_list: List of PPS 8x8 scaling lists (may contain None)
        list_idx: Index 0-1 (corresponding to lists 6-7)
        use_default_flag: If True and list is absent, use spec default

    Returns:
        List of 64 scaling values
    """
    # PPS overrides SPS
    if pps_list is not None and len(pps_list) > list_idx and pps_list[list_idx] is not None:
        return list(pps_list[list_idx])

    # Check SPS
    if sps_list is not None and len(sps_list) > list_idx and sps_list[list_idx] is not None:
        return list(sps_list[list_idx])

    # Apply fallback rules
    if use_default_flag:
        # Use spec-defined defaults
        if list_idx == 0:
            return DEFAULT_8x8_INTRA.copy()
        else:
            return DEFAULT_8x8_INTER.copy()

    # Fallback to previous list
    if list_idx == 0:
        return DEFAULT_8x8_INTRA.copy()
    else:
        # List 7 falls back to list 6
        return get_effective_scaling_list_8x8(sps_list, pps_list, list_idx - 1, use_default_flag)


def get_chroma_scaling_list_4x4(
    sps: 'SPS',
    pps: Optional['PPS'],
    index: int,
    is_cb: bool = True,
) -> List[int]:
    """Get 4x4 scaling list for chroma components.

    For 4:2:0 format, chroma uses the same scaling lists as luma.
    For 4:2:2 and 4:4:4, separate chroma lists may be used.

    Args:
        sps: Sequence parameter set
        pps: Picture parameter set
        index: List index
        is_cb: True for Cb component, False for Cr

    Returns:
        List of 16 scaling values
    """
    # For 4:2:0, chroma uses same lists as luma
    chroma_format_idc = getattr(sps, 'chroma_format_idc', 1)
    if chroma_format_idc == 1:  # 4:2:0
        return get_scaling_list_4x4(sps, pps, index)

    # For other formats, would need additional list indices
    # Currently fallback to luma lists
    return get_scaling_list_4x4(sps, pps, index)


def get_scaling_list_4x4(
    sps: 'SPS',
    pps: Optional['PPS'],
    index: int,
) -> List[int]:
    """Get the effective 4x4 scaling list for a given index.

    H.264 Spec: Section 7.4.2.1.1

    Fallback rules for 4x4:
    - index 0: Default_4x4_Intra or list from SPS
    - index 1,2: Fall back to index-1 (Intra)
    - index 3: Default_4x4_Inter or list from SPS
    - index 4,5: Fall back to index-1 (Inter)

    Args:
        sps: Sequence parameter set
        pps: Picture parameter set (optional, can override SPS)
        index: Scaling list index (0-5)

    Returns:
        List of 16 scaling values
    """
    if index < 0 or index > 5:
        raise ValueError(f"Invalid 4x4 scaling list index: {index}")

    # Check if PPS overrides
    if pps is not None and hasattr(pps, 'pic_scaling_matrix_present_flag'):
        if pps.pic_scaling_matrix_present_flag:
            if hasattr(pps, 'pic_scaling_list_present_flag') and pps.pic_scaling_list_present_flag:
                if len(pps.pic_scaling_list_present_flag) > index and pps.pic_scaling_list_present_flag[index]:
                    if hasattr(pps, 'scaling_lists_4x4') and len(pps.scaling_lists_4x4) > index:
                        return list(pps.scaling_lists_4x4[index])

    # Check SPS scaling lists
    if not _is_high_profile(sps.profile_idc):
        # Non-High profiles don't have scaling matrices
        return FLAT_4x4.copy()

    if not getattr(sps, 'seq_scaling_matrix_present_flag', False):
        # No scaling matrix present, use flat
        return FLAT_4x4.copy()

    # Check if this specific list is present in SPS
    if hasattr(sps, 'seq_scaling_list_present_flag') and sps.seq_scaling_list_present_flag:
        if len(sps.seq_scaling_list_present_flag) > index and sps.seq_scaling_list_present_flag[index]:
            # Check for use_default flag
            if hasattr(sps, 'use_default_scaling_matrix_flag_4x4'):
                if len(sps.use_default_scaling_matrix_flag_4x4) > index:
                    if sps.use_default_scaling_matrix_flag_4x4[index]:
                        # Use spec-defined default
                        return DEFAULT_4x4_INTRA.copy() if index < 3 else DEFAULT_4x4_INTER.copy()

            # Use custom list from SPS
            if hasattr(sps, 'scaling_lists_4x4') and len(sps.scaling_lists_4x4) > index:
                return list(sps.scaling_lists_4x4[index])

    # Apply fallback rules
    if index == 0:
        return DEFAULT_4x4_INTRA.copy()
    elif index == 3:
        return DEFAULT_4x4_INTER.copy()
    else:
        # Fall back to previous list
        return get_scaling_list_4x4(sps, pps, index - 1)


def get_scaling_list_8x8(
    sps: 'SPS',
    pps: Optional['PPS'],
    index: int,
) -> List[int]:
    """Get the effective 8x8 scaling list for a given index.

    H.264 Spec: Section 7.4.2.1.1

    Fallback rules for 8x8:
    - index 0 (list 6): Default_8x8_Intra
    - index 1 (list 7): Default_8x8_Inter

    Args:
        sps: Sequence parameter set
        pps: Picture parameter set (optional, can override SPS)
        index: Scaling list index (0-1, corresponding to lists 6-7)

    Returns:
        List of 64 scaling values
    """
    if index < 0 or index > 1:
        raise ValueError(f"Invalid 8x8 scaling list index: {index}")

    # SPS index for 8x8 lists is 6+index
    sps_index = 6 + index

    # Check if PPS overrides
    if pps is not None and hasattr(pps, 'pic_scaling_matrix_present_flag'):
        if pps.pic_scaling_matrix_present_flag:
            if hasattr(pps, 'pic_scaling_list_present_flag') and pps.pic_scaling_list_present_flag:
                if len(pps.pic_scaling_list_present_flag) > sps_index and pps.pic_scaling_list_present_flag[sps_index]:
                    if hasattr(pps, 'scaling_lists_8x8') and len(pps.scaling_lists_8x8) > index:
                        return list(pps.scaling_lists_8x8[index])

    # Check SPS scaling lists
    if not _is_high_profile(sps.profile_idc):
        # Non-High profiles don't have scaling matrices
        return FLAT_8x8.copy()

    if not getattr(sps, 'seq_scaling_matrix_present_flag', False):
        # No scaling matrix present, use flat
        return FLAT_8x8.copy()

    # Check if this specific list is present in SPS
    if hasattr(sps, 'seq_scaling_list_present_flag') and sps.seq_scaling_list_present_flag:
        if len(sps.seq_scaling_list_present_flag) > sps_index and sps.seq_scaling_list_present_flag[sps_index]:
            # Check for use_default flag
            if hasattr(sps, 'use_default_scaling_matrix_flag_8x8'):
                if len(sps.use_default_scaling_matrix_flag_8x8) > index:
                    if sps.use_default_scaling_matrix_flag_8x8[index]:
                        # Use spec-defined default
                        return DEFAULT_8x8_INTRA.copy() if index == 0 else DEFAULT_8x8_INTER.copy()

            # Use custom list from SPS
            if hasattr(sps, 'scaling_lists_8x8') and len(sps.scaling_lists_8x8) > index:
                return list(sps.scaling_lists_8x8[index])

    # Apply fallback rules - use spec defaults
    return DEFAULT_8x8_INTRA.copy() if index == 0 else DEFAULT_8x8_INTER.copy()
