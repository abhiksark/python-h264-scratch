# h264/entropy/cabac_macroblock.py
"""CABAC macroblock-level decoding.

High-level macroblock decoding using CABAC entropy coding.
Wraps lower-level syntax element decoding with neighbor context.

H.264 Spec Reference: Section 7.3.5 - Macroblock layer syntax
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING

import numpy as np

from entropy.cabac_residual import (
    decode_residual_block_cabac,
    decode_residual_block_cabac_with_cbf,
    decode_residual_block_8x8,
)
from entropy.cabac_syntax import (
    decode_mb_skip_flag,
    decode_mb_type_i,
    decode_mb_type_p,
    decode_mb_type_b,
    decode_sub_mb_type_p,
    decode_sub_mb_type_b,
    decode_intra_chroma_pred_mode,
    decode_cbp_luma,
    decode_cbp_chroma,
    decode_prev_intra4x4_pred_mode_flag,
    decode_rem_intra4x4_pred_mode,
    decode_ref_idx,
    decode_mvd_lx,
    decode_mb_qp_delta,
)

if TYPE_CHECKING:
    from entropy.cabac_arith import CABACDecoder
    from entropy.cabac_context import CABACContext


# Context indices
CTX_TRANSFORM_SIZE_FLAG = 399
CTX_MB_FIELD_DECODING_FLAG = 70
CTX_END_OF_SLICE_FLAG = 276

# Sub-MB index to 4x4 block base position (blk_x, blk_y)
_SUB_MB_BASE = [(0, 0), (2, 0), (0, 2), (2, 2)]

# Sub-partition block offsets within a sub-MB for each sub_mb_type
# P: type 0=8x8, 1=8x4, 2=4x8, 3=4x4
_P_SUB_PART_OFFSETS = {
    0: [(0, 0)],                              # 8x8: 1 MVD
    1: [(0, 0), (0, 1)],                      # 8x4: top, bottom
    2: [(0, 0), (1, 0)],                      # 4x8: left, right
    3: [(0, 0), (1, 0), (0, 1), (1, 1)],      # 4x4: raster scan
}

# Block coverage (rows, cols) for each sub-partition type
_P_SUB_PART_COVERAGE = {
    0: [(2, 2)],                              # 8x8
    1: [(1, 2), (1, 2)],                      # 8x4: each covers 1 row x 2 cols
    2: [(2, 1), (2, 1)],                      # 4x8: each covers 2 rows x 1 col
    3: [(1, 1), (1, 1), (1, 1), (1, 1)],      # 4x4: each covers 1x1
}

# B sub-partition block offsets within a sub-MB (H.264 Table 7-18)
# sub_type 0 (Direct) derives MVDs, not decoded — no entry needed
_B_SUB_PART_OFFSETS = {
    1: [(0, 0)],                              # B_L0_8x8
    2: [(0, 0)],                              # B_L1_8x8
    3: [(0, 0)],                              # B_Bi_8x8
    4: [(0, 0), (0, 1)],                      # B_L0_8x4
    5: [(0, 0), (1, 0)],                      # B_L0_4x8
    6: [(0, 0), (0, 1)],                      # B_L1_8x4
    7: [(0, 0), (1, 0)],                      # B_L1_4x8
    8: [(0, 0), (0, 1)],                      # B_Bi_8x4
    9: [(0, 0), (1, 0)],                      # B_Bi_4x8
    10: [(0, 0), (1, 0), (0, 1), (1, 1)],     # B_L0_4x4
    11: [(0, 0), (1, 0), (0, 1), (1, 1)],     # B_L1_4x4
    12: [(0, 0), (1, 0), (0, 1), (1, 1)],     # B_Bi_4x4
}

_B_SUB_PART_COVERAGE = {
    1: [(2, 2)],                              # 8x8
    2: [(2, 2)],
    3: [(2, 2)],
    4: [(1, 2), (1, 2)],                      # 8x4
    5: [(2, 1), (2, 1)],                      # 4x8
    6: [(1, 2), (1, 2)],
    7: [(2, 1), (2, 1)],
    8: [(1, 2), (1, 2)],
    9: [(2, 1), (2, 1)],
    10: [(1, 1), (1, 1), (1, 1), (1, 1)],     # 4x4
    11: [(1, 1), (1, 1), (1, 1), (1, 1)],
    12: [(1, 1), (1, 1), (1, 1), (1, 1)],
}


def _compute_ref_idx_ctx_inc(
    blk_x: int,
    blk_y: int,
    cur_refs: np.ndarray,
    left_refs: Optional[np.ndarray],
    top_refs: Optional[np.ndarray],
) -> int:
    """Compute ctxIdxInc for ref_idx bin 0 (H.264 Table 9-34).

    condTermFlagA = 1 if left neighbor's ref_idx > 0, else 0.
    condTermFlagB = 1 if top neighbor's ref_idx > 0, else 0.
    ctxIdxInc = condTermFlagA + 2 * condTermFlagB.

    Args:
        blk_x, blk_y: Top-left 4x4 block position of current partition (0-3)
        cur_refs: Current MB's ref_idx grid [4, 4]
        left_refs: Left MB's ref_idx grid [4, 4] or None
        top_refs: Top MB's ref_idx grid [4, 4] or None

    Returns:
        ctxIdxInc: 0, 1, 2, or 3
    """
    # Left neighbor (A)
    cond_a = 0
    if blk_x > 0:
        cond_a = 1 if cur_refs[blk_y, blk_x - 1] > 0 else 0
    elif left_refs is not None:
        cond_a = 1 if left_refs[blk_y, 3] > 0 else 0

    # Top neighbor (B)
    cond_b = 0
    if blk_y > 0:
        cond_b = 1 if cur_refs[blk_y - 1, blk_x] > 0 else 0
    elif top_refs is not None:
        cond_b = 1 if top_refs[3, blk_x] > 0 else 0

    return cond_a + 2 * cond_b


def _compute_mvd_ctx_inc(
    comp: int,
    blk_x: int,
    blk_y: int,
    cur_mvds: np.ndarray,
    left_mvds: Optional[np.ndarray],
    top_mvds: Optional[np.ndarray],
) -> int:
    """Compute ctxIdxInc for MVD bin0 (H.264 Section 9.3.3.1.1.7).

    Args:
        comp: 0=x, 1=y
        blk_x, blk_y: Current 4x4 block position (0-3)
        cur_mvds: Current MB's MVD grid [4, 4, 2]
        left_mvds: Left MB's MVD grid [4, 4, 2] or None
        top_mvds: Top MB's MVD grid [4, 4, 2] or None

    Returns:
        ctxIdxInc: 0, 1, or 2
    """
    # Left neighbor (A)
    abs_a = 0
    if blk_x > 0:
        abs_a = abs(int(cur_mvds[blk_y, blk_x - 1, comp]))
    elif left_mvds is not None:
        abs_a = abs(int(left_mvds[blk_y, 3, comp]))

    # Top neighbor (B)
    abs_b = 0
    if blk_y > 0:
        abs_b = abs(int(cur_mvds[blk_y - 1, blk_x, comp]))
    elif top_mvds is not None:
        abs_b = abs(int(top_mvds[3, blk_x, comp]))

    total = abs_a + abs_b
    if total < 3:
        return 0
    elif total > 32:
        return 2
    else:
        return 1


def decode_mb_skip_flag_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    slice_type: int,
    mb_info: Dict[str, Any],
) -> int:
    """Decode mb_skip_flag with neighbor context from mb_info.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        slice_type: 0=P, 1=B, 2=I
        mb_info: Dict with mb_x, mb_y, left_available, top_available,
                 left_skip, top_skip

    Returns:
        0 (not skipped) or 1 (skipped)
    """
    mb_x = mb_info.get('mb_x', 0)
    mb_y = mb_info.get('mb_y', 0)

    # H.264 9.3.3.1.1.3: condTermFlagN = 1 if neighbor available AND NOT skip
    left_available = mb_info.get('left_available', False)
    top_available = mb_info.get('top_available', False)
    left_cond = left_available and not mb_info.get('left_skip', False)
    top_cond = top_available and not mb_info.get('top_skip', False)

    return decode_mb_skip_flag(
        decoder, contexts, slice_type, mb_x, mb_y, left_cond, top_cond
    )


def decode_mb_type_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    slice_type: int,
    mb_info: Dict[str, Any],
) -> int:
    """Decode mb_type with neighbor context.

    H.264 Section 9.3.3.1.1.3: condTermFlagN = 1 if neighbor available
    and NOT I_NxN (I_4x4/I_8x8), 0 otherwise.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        slice_type: 0=P, 1=B, 2=I
        mb_info: Neighbor information

    Returns:
        mb_type value
    """
    if slice_type == 2:  # I-slice
        # Compute condTermFlagA + condTermFlagB for mb_type context
        left_available = mb_info.get('left_available', False)
        top_available = mb_info.get('top_available', False)
        left_mb_type = mb_info.get('left_mb_type', None)
        top_mb_type = mb_info.get('top_mb_type', None)

        # condTermFlag = 1 if neighbor available AND NOT I_NxN (type 0)
        cond_a = 1 if (left_available and left_mb_type is not None and left_mb_type != 0) else 0
        cond_b = 1 if (top_available and top_mb_type is not None and top_mb_type != 0) else 0
        ctx_inc = cond_a + cond_b

        return decode_mb_type_i(decoder, contexts, ctx_inc=ctx_inc)
    elif slice_type == 0:  # P-slice
        return decode_mb_type_p(decoder, contexts)
    else:  # B-slice
        # H.264 9.3.3.1.1.3: condTermFlagN = 1 if neighbor available
        # and mb_type != B_Direct_16x16 (type 0) and not skip
        left_available = mb_info.get('left_available', False)
        top_available = mb_info.get('top_available', False)
        left_mb_type = mb_info.get('left_mb_type', None)
        top_mb_type = mb_info.get('top_mb_type', None)
        left_skip = mb_info.get('left_skip', False)
        top_skip = mb_info.get('top_skip', False)
        cond_a = 1 if (left_available and left_mb_type is not None
                       and left_mb_type != 0 and not left_skip) else 0
        cond_b = 1 if (top_available and top_mb_type is not None
                       and top_mb_type != 0 and not top_skip) else 0
        ctx_inc = cond_a + cond_b
        return decode_mb_type_b(decoder, contexts, ctx_inc=ctx_inc)


def decode_sub_mb_type_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    slice_type: int,
    mb_type: int = 0,
) -> int:
    """Decode sub_mb_type for P_8x8 or B_8x8 macroblocks.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        slice_type: 0=P, 1=B
        mb_type: Parent macroblock type (not used, for API compatibility)

    Returns:
        sub_mb_type value
    """
    if slice_type == 0:  # P-slice
        return decode_sub_mb_type_p(decoder, contexts)
    else:  # B-slice
        return decode_sub_mb_type_b(decoder, contexts)


def decode_intra_chroma_pred_mode_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_info: Dict[str, Any],
) -> int:
    """Decode intra_chroma_pred_mode with neighbor context.

    H.264 Section 9.3.3.1.1.8:
    ctxIdxInc = condTermFlagA + condTermFlagB
    condTermFlagN = 0 if N unavailable, else (chroma_pred_mode_N != 0 ? 1 : 0)

    Args:
        decoder: CABAC decoder
        contexts: Context models
        mb_info: Neighbor information

    Returns:
        Chroma prediction mode (0=DC, 1=Horizontal, 2=Vertical, 3=Plane)
    """
    left_available = mb_info.get('left_available', False)
    top_available = mb_info.get('top_available', False)

    left_mode = mb_info.get('left_intra_chroma_pred_mode',
                           mb_info.get('left_chroma_mode', 0)) if left_available else 0
    top_mode = mb_info.get('top_intra_chroma_pred_mode',
                          mb_info.get('top_chroma_mode', 0)) if top_available else 0

    # condTermFlag = 1 if neighbor available AND chroma_pred_mode != 0
    cond_a = 1 if (left_available and left_mode != 0) else 0
    cond_b = 1 if (top_available and top_mode != 0) else 0
    ctx_inc = cond_a + cond_b

    return decode_intra_chroma_pred_mode(decoder, contexts, ctx_inc=ctx_inc)


def decode_cbp_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_info: Dict[str, Any],
    is_inter: bool = False,
) -> int:
    """Decode coded_block_pattern.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        mb_info: Neighbor information with cbp values
        is_inter: True for inter macroblocks

    Returns:
        CBP value (bits 0-3: luma, bits 4-5: chroma)
    """
    # Get mb_type from mb_info for context selection
    mb_type = mb_info.get('mb_type', 0)

    # Get neighbor CBP values for context derivation
    left_available = mb_info.get('left_available', False)
    top_available = mb_info.get('top_available', False)

    # Extract neighbor luma CBP (bits 0-3)
    left_cbp = mb_info.get('left_cbp', 0) if left_available else -1
    top_cbp = mb_info.get('top_cbp', 0) if top_available else -1

    # For luma CBP context, we need just the luma bits (0-3)
    left_cbp_luma = (left_cbp & 0x0F) if left_cbp >= 0 else -1
    top_cbp_luma = (top_cbp & 0x0F) if top_cbp >= 0 else -1

    # Extract neighbor chroma CBP (bits 4-5 shifted to 0-1 range)
    left_cbp_chroma = ((left_cbp >> 4) & 0x03) if left_cbp >= 0 else -1
    top_cbp_chroma = ((top_cbp >> 4) & 0x03) if top_cbp >= 0 else -1

    # Decode luma CBP (4 bits) with neighbor context
    cbp_luma = decode_cbp_luma(decoder, contexts, mb_type, left_cbp_luma, top_cbp_luma)

    # Decode chroma CBP (2 bits) with neighbor context
    cbp_chroma = decode_cbp_chroma(decoder, contexts, mb_type, left_cbp_chroma, top_cbp_chroma)

    return cbp_luma | (cbp_chroma << 4)


def decode_transform_size_8x8_flag_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_info: Dict[str, Any],
) -> int:
    """Decode transform_size_8x8_flag.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        mb_info: Neighbor information

    Returns:
        0 (4x4 transform) or 1 (8x8 transform)
    """
    # Context depends on neighbor transform sizes
    left_8x8 = mb_info.get('left_transform_8x8', False) if mb_info.get('left_available', False) else False
    top_8x8 = mb_info.get('top_transform_8x8', False) if mb_info.get('top_available', False) else False

    ctx_inc = (1 if left_8x8 else 0) + (1 if top_8x8 else 0)
    ctx_idx = CTX_TRANSFORM_SIZE_FLAG + ctx_inc

    return decoder.decode_decision(contexts[ctx_idx])


def decode_mb_pred_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_type: int,
    slice_type: int,
    mb_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Decode mb_pred or sub_mb_pred.

    H.264 Section 7.3.5.1 (mb_pred) and 7.3.5.2 (sub_mb_pred).
    Parses ref_idx and MVD for each partition in syntax order.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        mb_type: Macroblock type
        slice_type: Slice type
        mb_info: Neighbor information

    Returns:
        Dict with intra_pred_modes, ref_idx, mvd values, sub_mb_types
    """
    result = {
        'intra_pred_modes': [],
        'ref_idx_l0': [],
        'ref_idx_l1': [],
        'mvd_l0': [],
        'mvd_l1': [],
        'sub_mb_types': [],
    }

    is_intra = _is_intra_mb_type(mb_type, slice_type)

    if is_intra:
        # Intra prediction modes
        intra_base = mb_type
        if slice_type == 0:
            intra_base = mb_type - 5
        elif slice_type == 1:
            intra_base = mb_type - 23

        if intra_base == 0:  # I_NxN (I_4x4 or I_8x8)
            # I_8x8: 4 prediction modes; I_4x4: 16 prediction modes
            t8x8 = mb_info.get('transform_size_8x8_flag', 0)
            num_modes = 4 if t8x8 else 16
            for _ in range(num_modes):
                if decode_prev_intra4x4_pred_mode_flag(decoder, contexts) == 1:
                    result['intra_pred_modes'].append(-1)
                else:
                    mode = decode_rem_intra4x4_pred_mode(decoder, contexts)
                    result['intra_pred_modes'].append(mode)

        result['intra_chroma_pred_mode'] = decode_intra_chroma_pred_mode_cabac(
            decoder, contexts, mb_info
        )
        return result

    num_ref_l0 = mb_info.get('num_ref_idx_l0_active', 1)
    num_ref_l1 = mb_info.get('num_ref_idx_l1_active', 1)

    if slice_type == 0:  # P-slice
        _decode_p_mb_pred(decoder, contexts, mb_type, num_ref_l0, result, mb_info)
    elif slice_type == 1:  # B-slice
        _decode_b_mb_pred(decoder, contexts, mb_type, num_ref_l0, num_ref_l1, result, mb_info)

    return result


def _decode_p_mb_pred(
    decoder: 'CABACDecoder',
    contexts: list,
    mb_type: int,
    num_ref_l0: int,
    result: dict,
    mb_info: dict,
) -> None:
    """Decode P-slice mb_pred or sub_mb_pred."""
    if mb_type == 3:  # P_8x8
        _decode_p_sub_mb_pred(decoder, contexts, num_ref_l0, result, mb_info)
        return
    if mb_type == 4:  # P_8x8ref0
        _decode_p_sub_mb_pred(decoder, contexts, 1, result, mb_info)
        return

    # Regular P partitions: 0=16x16, 1=16x8, 2=8x16
    num_parts = _get_num_partitions(mb_type, 0)

    # Partition top-left block positions: 16x16→(0,0), 16x8→(0,0)/(0,2), 8x16→(0,0)/(2,0)
    part_positions = [(0, 0)]
    if mb_type == 1:  # P_L0_16x8
        part_positions = [(0, 0), (0, 2)]
    elif mb_type == 2:  # P_L0_8x16
        part_positions = [(0, 0), (2, 0)]

    # ref_idx context derivation needs neighbor grids
    cur_refs_l0 = np.zeros((4, 4), dtype=np.int8)
    left_refs_l0 = mb_info.get('left_mb_ref_l0')
    top_refs_l0 = mb_info.get('top_mb_ref_l0')

    # Partition coverage in 4x4 blocks: 16x16→(4,4), 16x8→(4,2), 8x16→(2,4)
    if mb_type == 0:
        part_rows, part_cols = 4, 4
    elif mb_type == 1:
        part_rows, part_cols = 2, 4
    else:
        part_rows, part_cols = 4, 2

    # ref_idx_l0 for all partitions
    for part_idx in range(num_parts):
        blk_x, blk_y = part_positions[part_idx]
        ctx_inc = _compute_ref_idx_ctx_inc(blk_x, blk_y, cur_refs_l0,
                                           left_refs_l0, top_refs_l0)
        ref = decode_ref_idx(decoder, contexts, 0, num_ref_l0, ctx_inc)
        result['ref_idx_l0'].append(ref)
        cur_refs_l0[blk_y:blk_y + part_rows, blk_x:blk_x + part_cols] = ref

    result['ref_idx_grid_l0'] = cur_refs_l0

    # MVD grid for neighbor context computation
    cur_mvds = np.zeros((4, 4, 2), dtype=np.int32)
    left_mvds = mb_info.get('left_mb_mvds_l0')
    top_mvds = mb_info.get('top_mb_mvds_l0')

    # mvd_l0 for all partitions
    for part_idx in range(num_parts):
        blk_x, blk_y = part_positions[part_idx]

        ctx_inc_x = _compute_mvd_ctx_inc(0, blk_x, blk_y, cur_mvds, left_mvds, top_mvds)
        ctx_inc_y = _compute_mvd_ctx_inc(1, blk_x, blk_y, cur_mvds, left_mvds, top_mvds)

        mvd_x = decode_mvd_lx(decoder, contexts, 0, 0, ctx_inc_x)
        mvd_y = decode_mvd_lx(decoder, contexts, 0, 1, ctx_inc_y)
        result['mvd_l0'].append((mvd_x, mvd_y))

        # Fill MVD grid for this partition
        if mb_type == 0:  # P_16x16: all blocks
            cur_mvds[:, :, 0] = mvd_x
            cur_mvds[:, :, 1] = mvd_y
        elif mb_type == 1:  # P_16x8
            cur_mvds[blk_y:blk_y + 2, :, 0] = mvd_x
            cur_mvds[blk_y:blk_y + 2, :, 1] = mvd_y
        elif mb_type == 2:  # P_8x16
            cur_mvds[:, blk_x:blk_x + 2, 0] = mvd_x
            cur_mvds[:, blk_x:blk_x + 2, 1] = mvd_y

    result['mvd_grid_l0'] = cur_mvds


def _decode_p_sub_mb_pred(
    decoder: 'CABACDecoder',
    contexts: list,
    num_ref_l0: int,
    result: dict,
    mb_info: dict,
) -> None:
    """Decode P_8x8 sub_mb_pred (H.264 7.3.5.2)."""
    # sub_mb_type[4]
    sub_types = []
    for _ in range(4):
        sub_types.append(decode_sub_mb_type_p(decoder, contexts))
    result['sub_mb_types'] = sub_types

    # ref_idx_l0[4] — one per 8x8 sub-MB
    cur_refs_l0 = np.zeros((4, 4), dtype=np.int8)
    left_refs_l0 = mb_info.get('left_mb_ref_l0')
    top_refs_l0 = mb_info.get('top_mb_ref_l0')
    for i in range(4):
        base_x, base_y = _SUB_MB_BASE[i]
        ctx_inc = _compute_ref_idx_ctx_inc(base_x, base_y, cur_refs_l0,
                                           left_refs_l0, top_refs_l0)
        ref = decode_ref_idx(decoder, contexts, 0, num_ref_l0, ctx_inc)
        result['ref_idx_l0'].append(ref)
        cur_refs_l0[base_y:base_y + 2, base_x:base_x + 2] = ref
    result['ref_idx_grid_l0'] = cur_refs_l0

    # MVD grid for neighbor context computation
    cur_mvds = np.zeros((4, 4, 2), dtype=np.int32)
    left_mvds = mb_info.get('left_mb_mvds_l0')
    top_mvds = mb_info.get('top_mb_mvds_l0')

    # mvd_l0 for each sub-partition
    for sub_idx in range(4):
        base_x, base_y = _SUB_MB_BASE[sub_idx]
        sub_type = sub_types[sub_idx]
        offsets = _P_SUB_PART_OFFSETS.get(sub_type, [(0, 0)])
        coverage = _P_SUB_PART_COVERAGE.get(sub_type, [(2, 2)])

        for part_i, (dx, dy) in enumerate(offsets):
            blk_x = base_x + dx
            blk_y = base_y + dy

            ctx_inc_x = _compute_mvd_ctx_inc(0, blk_x, blk_y, cur_mvds, left_mvds, top_mvds)
            ctx_inc_y = _compute_mvd_ctx_inc(1, blk_x, blk_y, cur_mvds, left_mvds, top_mvds)

            mvd_x = decode_mvd_lx(decoder, contexts, 0, 0, ctx_inc_x)
            mvd_y = decode_mvd_lx(decoder, contexts, 0, 1, ctx_inc_y)
            result['mvd_l0'].append((mvd_x, mvd_y))

            # Fill MVD grid for this sub-partition
            rows, cols = coverage[part_i]
            cur_mvds[blk_y:blk_y + rows, blk_x:blk_x + cols, 0] = mvd_x
            cur_mvds[blk_y:blk_y + rows, blk_x:blk_x + cols, 1] = mvd_y

    result['mvd_grid_l0'] = cur_mvds


def _decode_b_mb_pred(
    decoder: 'CABACDecoder',
    contexts: list,
    mb_type: int,
    num_ref_l0: int,
    num_ref_l1: int,
    result: dict,
    mb_info: dict,
) -> None:
    """Decode B-slice mb_pred or sub_mb_pred."""
    if mb_type == 0:  # B_Direct_16x16 — no ref_idx or MVD in bitstream
        return

    if mb_type == 22:  # B_8x8
        _decode_b_sub_mb_pred(decoder, contexts, num_ref_l0, num_ref_l1, result, mb_info)
        return

    # Regular B partitions (types 1-21)
    num_parts = _get_num_partitions(mb_type, 1)
    pred_flags = _get_b_pred_flags(mb_type)

    # Partition positions and coverage in 4x4 block units
    if mb_type <= 3:  # 16x16
        part_positions = [(0, 0)]
        part_rows, part_cols = 4, 4
    elif mb_type % 2 == 0:  # 16x8 (even types 4,6,...,20)
        part_positions = [(0, 0), (0, 2)]
        part_rows, part_cols = 2, 4
    else:  # 8x16 (odd types 5,7,...,21)
        part_positions = [(0, 0), (2, 0)]
        part_rows, part_cols = 4, 2

    # ref_idx grids for context derivation (H.264 9.3.3.1.1.3)
    cur_refs_l0 = np.zeros((4, 4), dtype=np.int8)
    cur_refs_l1 = np.zeros((4, 4), dtype=np.int8)
    left_refs_l0 = mb_info.get('left_mb_ref_l0')
    top_refs_l0 = mb_info.get('top_mb_ref_l0')
    left_refs_l1 = mb_info.get('left_mb_ref_l1')
    top_refs_l1 = mb_info.get('top_mb_ref_l1')

    # ref_idx_l0 for all partitions (H.264 7.3.5.1 order)
    for i in range(num_parts):
        if pred_flags[i][0]:  # predFlagL0
            blk_x, blk_y = part_positions[i]
            ctx_inc = _compute_ref_idx_ctx_inc(blk_x, blk_y, cur_refs_l0,
                                               left_refs_l0, top_refs_l0)
            ref = decode_ref_idx(decoder, contexts, 0, num_ref_l0, ctx_inc)
            result['ref_idx_l0'].append(ref)
            cur_refs_l0[blk_y:blk_y + part_rows, blk_x:blk_x + part_cols] = ref

    # ref_idx_l1 for all partitions
    for i in range(num_parts):
        if pred_flags[i][1]:  # predFlagL1
            blk_x, blk_y = part_positions[i]
            ctx_inc = _compute_ref_idx_ctx_inc(blk_x, blk_y, cur_refs_l1,
                                               left_refs_l1, top_refs_l1)
            ref = decode_ref_idx(decoder, contexts, 1, num_ref_l1, ctx_inc)
            result['ref_idx_l1'].append(ref)
            cur_refs_l1[blk_y:blk_y + part_rows, blk_x:blk_x + part_cols] = ref

    result['ref_idx_grid_l0'] = cur_refs_l0
    result['ref_idx_grid_l1'] = cur_refs_l1

    # MVD grids for neighbor context computation (H.264 9.3.3.1.1.7)
    cur_mvds_l0 = np.zeros((4, 4, 2), dtype=np.int32)
    cur_mvds_l1 = np.zeros((4, 4, 2), dtype=np.int32)
    left_mvds_l0 = mb_info.get('left_mb_mvds_l0')
    top_mvds_l0 = mb_info.get('top_mb_mvds_l0')
    left_mvds_l1 = mb_info.get('left_mb_mvds_l1')
    top_mvds_l1 = mb_info.get('top_mb_mvds_l1')

    # mvd_l0 for all partitions
    for i in range(num_parts):
        if pred_flags[i][0]:
            blk_x, blk_y = part_positions[i]
            ctx_inc_x = _compute_mvd_ctx_inc(0, blk_x, blk_y, cur_mvds_l0, left_mvds_l0, top_mvds_l0)
            ctx_inc_y = _compute_mvd_ctx_inc(1, blk_x, blk_y, cur_mvds_l0, left_mvds_l0, top_mvds_l0)
            mvd_x = decode_mvd_lx(decoder, contexts, 0, 0, ctx_inc_x)
            mvd_y = decode_mvd_lx(decoder, contexts, 0, 1, ctx_inc_y)
            result['mvd_l0'].append((mvd_x, mvd_y))
            cur_mvds_l0[blk_y:blk_y + part_rows, blk_x:blk_x + part_cols, 0] = mvd_x
            cur_mvds_l0[blk_y:blk_y + part_rows, blk_x:blk_x + part_cols, 1] = mvd_y

    # mvd_l1 for all partitions
    for i in range(num_parts):
        if pred_flags[i][1]:
            blk_x, blk_y = part_positions[i]
            ctx_inc_x = _compute_mvd_ctx_inc(0, blk_x, blk_y, cur_mvds_l1, left_mvds_l1, top_mvds_l1)
            ctx_inc_y = _compute_mvd_ctx_inc(1, blk_x, blk_y, cur_mvds_l1, left_mvds_l1, top_mvds_l1)
            mvd_x = decode_mvd_lx(decoder, contexts, 1, 0, ctx_inc_x)
            mvd_y = decode_mvd_lx(decoder, contexts, 1, 1, ctx_inc_y)
            result['mvd_l1'].append((mvd_x, mvd_y))
            cur_mvds_l1[blk_y:blk_y + part_rows, blk_x:blk_x + part_cols, 0] = mvd_x
            cur_mvds_l1[blk_y:blk_y + part_rows, blk_x:blk_x + part_cols, 1] = mvd_y

    result['mvd_grid_l0'] = cur_mvds_l0
    result['mvd_grid_l1'] = cur_mvds_l1


def _decode_b_sub_mb_pred(
    decoder: 'CABACDecoder',
    contexts: list,
    num_ref_l0: int,
    num_ref_l1: int,
    result: dict,
    mb_info: dict,
) -> None:
    """Decode B_8x8 sub_mb_pred (H.264 7.3.5.2)."""
    # sub_mb_type[4]
    sub_types = []
    for _ in range(4):
        sub_types.append(decode_sub_mb_type_b(decoder, contexts))
    result['sub_mb_types'] = sub_types

    # ref_idx grids for context derivation (H.264 9.3.3.1.1.3)
    cur_refs_l0 = np.zeros((4, 4), dtype=np.int8)
    cur_refs_l1 = np.zeros((4, 4), dtype=np.int8)
    left_refs_l0 = mb_info.get('left_mb_ref_l0')
    top_refs_l0 = mb_info.get('top_mb_ref_l0')
    left_refs_l1 = mb_info.get('left_mb_ref_l1')
    top_refs_l1 = mb_info.get('top_mb_ref_l1')

    # ref_idx_l0[4] — one per sub-MB, only if sub-partition uses L0
    # B_Direct_8x8 (sub_type=0) derives ref_idx, does not read from bitstream
    for i in range(4):
        base_x, base_y = _SUB_MB_BASE[i]
        if sub_types[i] != 0 and _b_sub_uses_l0(sub_types[i]):
            ctx_inc = _compute_ref_idx_ctx_inc(base_x, base_y, cur_refs_l0,
                                               left_refs_l0, top_refs_l0)
            ref = decode_ref_idx(decoder, contexts, 0, num_ref_l0, ctx_inc)
            result['ref_idx_l0'].append(ref)
            cur_refs_l0[base_y:base_y + 2, base_x:base_x + 2] = ref
        else:
            result['ref_idx_l0'].append(0)  # placeholder or derived

    # ref_idx_l1[4]
    for i in range(4):
        base_x, base_y = _SUB_MB_BASE[i]
        if sub_types[i] != 0 and _b_sub_uses_l1(sub_types[i]):
            ctx_inc = _compute_ref_idx_ctx_inc(base_x, base_y, cur_refs_l1,
                                               left_refs_l1, top_refs_l1)
            ref = decode_ref_idx(decoder, contexts, 1, num_ref_l1, ctx_inc)
            result['ref_idx_l1'].append(ref)
            cur_refs_l1[base_y:base_y + 2, base_x:base_x + 2] = ref
        else:
            result['ref_idx_l1'].append(0)  # placeholder or derived

    result['ref_idx_grid_l0'] = cur_refs_l0
    result['ref_idx_grid_l1'] = cur_refs_l1

    # MVD grids for neighbor context computation
    cur_mvds_l0 = np.zeros((4, 4, 2), dtype=np.int32)
    cur_mvds_l1 = np.zeros((4, 4, 2), dtype=np.int32)
    left_mvds_l0 = mb_info.get('left_mb_mvds_l0')
    top_mvds_l0 = mb_info.get('top_mb_mvds_l0')
    left_mvds_l1 = mb_info.get('left_mb_mvds_l1')
    top_mvds_l1 = mb_info.get('top_mb_mvds_l1')

    # mvd_l0 per sub-partition
    for sub_idx in range(4):
        if _b_sub_uses_l0(sub_types[sub_idx]) and sub_types[sub_idx] != 0:
            base_x, base_y = _SUB_MB_BASE[sub_idx]
            sub_type = sub_types[sub_idx]
            offsets = _B_SUB_PART_OFFSETS.get(sub_type, [(0, 0)])
            coverage = _B_SUB_PART_COVERAGE.get(sub_type, [(2, 2)])

            for part_i, (dx, dy) in enumerate(offsets):
                blk_x = base_x + dx
                blk_y = base_y + dy

                ctx_inc_x = _compute_mvd_ctx_inc(0, blk_x, blk_y, cur_mvds_l0, left_mvds_l0, top_mvds_l0)
                ctx_inc_y = _compute_mvd_ctx_inc(1, blk_x, blk_y, cur_mvds_l0, left_mvds_l0, top_mvds_l0)

                mvd_x = decode_mvd_lx(decoder, contexts, 0, 0, ctx_inc_x)
                mvd_y = decode_mvd_lx(decoder, contexts, 0, 1, ctx_inc_y)
                result['mvd_l0'].append((mvd_x, mvd_y))

                rows, cols = coverage[part_i]
                cur_mvds_l0[blk_y:blk_y + rows, blk_x:blk_x + cols, 0] = mvd_x
                cur_mvds_l0[blk_y:blk_y + rows, blk_x:blk_x + cols, 1] = mvd_y

    # mvd_l1 per sub-partition
    for sub_idx in range(4):
        if _b_sub_uses_l1(sub_types[sub_idx]) and sub_types[sub_idx] != 0:
            base_x, base_y = _SUB_MB_BASE[sub_idx]
            sub_type = sub_types[sub_idx]
            offsets = _B_SUB_PART_OFFSETS.get(sub_type, [(0, 0)])
            coverage = _B_SUB_PART_COVERAGE.get(sub_type, [(2, 2)])

            for part_i, (dx, dy) in enumerate(offsets):
                blk_x = base_x + dx
                blk_y = base_y + dy

                ctx_inc_x = _compute_mvd_ctx_inc(0, blk_x, blk_y, cur_mvds_l1, left_mvds_l1, top_mvds_l1)
                ctx_inc_y = _compute_mvd_ctx_inc(1, blk_x, blk_y, cur_mvds_l1, left_mvds_l1, top_mvds_l1)

                mvd_x = decode_mvd_lx(decoder, contexts, 1, 0, ctx_inc_x)
                mvd_y = decode_mvd_lx(decoder, contexts, 1, 1, ctx_inc_y)
                result['mvd_l1'].append((mvd_x, mvd_y))

                rows, cols = coverage[part_i]
                cur_mvds_l1[blk_y:blk_y + rows, blk_x:blk_x + cols, 0] = mvd_x
                cur_mvds_l1[blk_y:blk_y + rows, blk_x:blk_x + cols, 1] = mvd_y

    result['mvd_grid_l0'] = cur_mvds_l0
    result['mvd_grid_l1'] = cur_mvds_l1


def _p_sub_num_parts(sub_type: int) -> int:
    """Number of sub-partitions for P sub_mb_type."""
    # 0=8x8(1), 1=8x4(2), 2=4x8(2), 3=4x4(4)
    return [1, 2, 2, 4][min(sub_type, 3)]


def _b_sub_num_parts(sub_type: int) -> int:
    """Number of sub-partitions for B sub_mb_type."""
    # H.264 Table 7-18
    # 0=Direct(4), 1=L0_8x8(1), 2=L1_8x8(1), 3=Bi_8x8(1),
    # 4=L0_8x4(2), 5=L0_4x8(2), 6=L1_8x4(2), 7=L1_4x8(2),
    # 8=Bi_8x4(2), 9=Bi_4x8(2), 10=L0_4x4(4), 11=L1_4x4(4), 12=Bi_4x4(4)
    return [4, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4][min(sub_type, 12)]


def _b_sub_uses_l0(sub_type: int) -> bool:
    """Whether B sub_mb_type uses L0 prediction."""
    # Direct(0), L0(1,4,5,10), Bi(3,8,9,12)
    return sub_type in (0, 1, 3, 4, 5, 8, 9, 10, 12)


def _b_sub_uses_l1(sub_type: int) -> bool:
    """Whether B sub_mb_type uses L1 prediction."""
    # Direct(0), L1(2,6,7,11), Bi(3,8,9,12)
    return sub_type in (0, 2, 3, 6, 7, 8, 9, 11, 12)


# B-MB type prediction flags: per partition (predFlagL0, predFlagL1)
# H.264 Table 7-14
_B_PRED_FLAGS = {
    1: [(True, False)],                        # B_L0_16x16
    2: [(False, True)],                        # B_L1_16x16
    3: [(True, True)],                         # B_Bi_16x16
    4: [(True, False), (True, False)],         # B_L0_L0_16x8
    5: [(True, False), (True, False)],         # B_L0_L0_8x16
    6: [(False, True), (False, True)],         # B_L1_L1_16x8
    7: [(False, True), (False, True)],         # B_L1_L1_8x16
    8: [(True, False), (False, True)],         # B_L0_L1_16x8
    9: [(True, False), (False, True)],         # B_L0_L1_8x16
    10: [(False, True), (True, False)],        # B_L1_L0_16x8
    11: [(False, True), (True, False)],        # B_L1_L0_8x16
    12: [(True, False), (True, True)],         # B_L0_Bi_16x8
    13: [(True, False), (True, True)],         # B_L0_Bi_8x16
    14: [(False, True), (True, True)],         # B_L1_Bi_16x8
    15: [(False, True), (True, True)],         # B_L1_Bi_8x16
    16: [(True, True), (True, False)],         # B_Bi_L0_16x8
    17: [(True, True), (True, False)],         # B_Bi_L0_8x16
    18: [(True, True), (False, True)],         # B_Bi_L1_16x8
    19: [(True, True), (False, True)],         # B_Bi_L1_8x16
    20: [(True, True), (True, True)],          # B_Bi_Bi_16x8
    21: [(True, True), (True, True)],          # B_Bi_Bi_8x16
}


def _get_b_pred_flags(mb_type: int) -> list:
    """Get per-partition prediction flags for B mb_type."""
    return _B_PRED_FLAGS.get(mb_type, [])


def decode_macroblock_layer_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    slice_type: int,
    mb_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Decode complete macroblock layer.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        slice_type: Slice type
        mb_info: Neighbor information

    Returns:
        Dict with mb_type, cbp, qp_delta, prediction info, coefficients
    """
    result = {
        'mb_skip_flag': 0,
        'mb_type': 0,
        'cbp': 0,
        'mb_qp_delta': 0,
        'transform_size_8x8_flag': 0,
    }

    # Check for skip (P/B slices only)
    if slice_type in (0, 1):  # P or B
        result['mb_skip_flag'] = decode_mb_skip_flag_cabac(
            decoder, contexts, slice_type, mb_info
        )
        if result['mb_skip_flag'] == 1:
            return result

    # Decode mb_type
    result['mb_type'] = decode_mb_type_cabac(
        decoder, contexts, slice_type, mb_info
    )

    # Check for I_PCM
    if _is_i_pcm(result['mb_type'], slice_type):
        result['i_pcm'] = decode_i_pcm_cabac(decoder, mb_info)
        return result

    # H.264 7.3.5: transform_size_8x8_flag position depends on MB type
    is_intra_nxn = _is_intra_mb_type(result['mb_type'], slice_type) and \
        not _is_i_16x16(result['mb_type'], slice_type) and \
        not _is_i_pcm(result['mb_type'], slice_type)

    # For I_NxN: parse transform_size_8x8_flag BEFORE mb_pred
    if is_intra_nxn and mb_info.get('transform_8x8_mode_flag', False):
        result['transform_size_8x8_flag'] = decode_transform_size_8x8_flag_cabac(
            decoder, contexts, mb_info
        )
        # mb_pred needs this to decide 4 vs 16 intra modes
        mb_info['transform_size_8x8_flag'] = result['transform_size_8x8_flag']

    # Decode mb_pred
    result['mb_pred'] = decode_mb_pred_cabac(
        decoder, contexts, result['mb_type'], slice_type, mb_info
    )

    # Decode CBP (if not I_16x16)
    if not _is_i_16x16(result['mb_type'], slice_type):
        result['cbp'] = decode_cbp_cabac(
            decoder, contexts, mb_info,
            is_inter=not _is_intra_mb_type(result['mb_type'], slice_type)
        )
    else:
        # For I_16x16, CBP is embedded in mb_type
        cbp_luma, cbp_chroma = _extract_i16x16_cbp(result['mb_type'], slice_type)
        result['cbp_luma'] = cbp_luma
        result['cbp_chroma'] = cbp_chroma
        result['cbp'] = cbp_luma | (cbp_chroma << 4)

    # For inter MBs: parse transform_size_8x8_flag AFTER CBP (H.264 7.3.5)
    # Only when noSubMbPartSizeLessThan8x8Flag == 1 (all sub-partitions >= 8x8)
    if not is_intra_nxn and not _is_i_16x16(result['mb_type'], slice_type) and \
            not _is_intra_mb_type(result['mb_type'], slice_type) and \
            mb_info.get('transform_8x8_mode_flag', False):
        cbp_luma = result['cbp'] & 0x0F
        if cbp_luma > 0:
            # H.264 7.4.5: noSubMbPartSizeLessThan8x8Flag
            no_sub_lt_8x8 = True
            sub_types = result.get('sub_mb_types', None)
            if sub_types is not None:
                # P_8x8: sub_mb_type must be 0 (P_L0_8x8)
                # B_8x8: sub_mb_type must be <= 3 (B_*_8x8)
                is_b = (slice_type == 1)
                for st in sub_types:
                    if is_b and st > 3:
                        no_sub_lt_8x8 = False
                        break
                    elif not is_b and st > 0:
                        no_sub_lt_8x8 = False
                        break
            if no_sub_lt_8x8:
                result['transform_size_8x8_flag'] = decode_transform_size_8x8_flag_cabac(
                    decoder, contexts, mb_info
                )

    # Decode mb_qp_delta: H.264 Section 7.3.5.1
    # Condition: CBP_luma > 0 OR CBP_chroma > 0 OR MbPartPredMode == Intra_16x16
    is_16x16 = _is_i_16x16(result['mb_type'], slice_type)
    has_coded_blocks = False
    if is_16x16:
        # I_16x16 always decodes mb_qp_delta (spec: MbPartPredMode == Intra_16x16)
        has_coded_blocks = True
    else:
        has_coded_blocks = result['cbp'] != 0

    if has_coded_blocks:
        # Context for first bin depends on whether prev MB had nonzero qp_delta
        prev_qp_delta = mb_info.get('prev_mb_qp_delta', 0)
        ctx_inc_first = 1 if prev_qp_delta != 0 else 0
        result['mb_qp_delta'] = decode_mb_qp_delta(
            decoder, contexts, ctx_inc_first=ctx_inc_first
        )

    # Decode residual blocks
    if has_coded_blocks:
        result['residual'] = _decode_residual_cabac(
            decoder, contexts, result['mb_type'], slice_type, result['cbp'],
            result.get('transform_size_8x8_flag', 0),
            mb_info=mb_info,
        )
    else:
        result['residual'] = _empty_residual()

    return result


def decode_i_pcm_cabac(
    decoder: 'CABACDecoder',
    mb_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Decode I_PCM macroblock samples.

    Args:
        decoder: CABAC decoder
        mb_info: Macroblock info including bit depth

    Returns:
        Dict with luma and chroma samples
    """
    bit_depth_luma = mb_info.get('bit_depth_luma', 8)
    bit_depth_chroma = mb_info.get('bit_depth_chroma', 8)

    # Byte align
    decoder.byte_align()

    result = {
        'luma_samples': [],
        'cb_samples': [],
        'cr_samples': [],
    }

    # Read 256 luma samples
    for _ in range(256):
        sample = decoder.read_bits(bit_depth_luma)
        result['luma_samples'].append(sample)

    # Read chroma samples (64 each for 4:2:0)
    for _ in range(64):
        sample = decoder.read_bits(bit_depth_chroma)
        result['cb_samples'].append(sample)

    for _ in range(64):
        sample = decoder.read_bits(bit_depth_chroma)
        result['cr_samples'].append(sample)

    # Reinitialize CABAC after I_PCM
    decoder.init_after_pcm()

    return result


def decode_end_of_slice_flag_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'] = None,
) -> int:
    """Decode end_of_slice_flag using terminate mode.

    Args:
        decoder: CABAC decoder
        contexts: Context models (optional, not used - terminate mode)

    Returns:
        0 (continue) or 1 (end of slice)
    """
    return decoder.decode_terminate()


def decode_mb_field_decoding_flag_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_info: Dict[str, Any],
) -> int:
    """Decode mb_field_decoding_flag for MBAFF.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        mb_info: Neighbor information

    Returns:
        0 (frame) or 1 (field)
    """
    # Context depends on neighbor field flags
    left_field = mb_info.get('left_mb_field', False) if mb_info.get('left_available', False) else False
    top_field = mb_info.get('top_mb_field', False) if mb_info.get('top_available', False) else False

    ctx_inc = (1 if left_field else 0) + (1 if top_field else 0)
    ctx_idx = CTX_MB_FIELD_DECODING_FLAG + ctx_inc

    return decoder.decode_decision(contexts[ctx_idx])


# Coded block flag context base indices (H.264 Table 9-39)
# ctxIdx = 85 + ctxBlockCatOffset + ctxIdxInc
# ctxBlockCatOffset = 4 * blockCatIdx
CTX_CBF_LUMA_DC_BASE = 85     # block_cat=0: 85 + 0
CTX_CBF_LUMA_AC_BASE = 89     # block_cat=1: 85 + 4
CTX_CBF_LUMA_4X4_BASE = 93    # block_cat=2: 85 + 8
CTX_CBF_CHROMA_DC_BASE = 97   # block_cat=3: 85 + 12
CTX_CBF_CHROMA_AC_BASE = 101  # block_cat=4: 85 + 16

# Luma 4x4 block neighbor mapping for coded_block_flag context
# Block positions (row, col) from scan order:
#  0:(0,0)   1:(0,4)   4:(0,8)   5:(0,12)
#  2:(4,0)   3:(4,4)   6:(4,8)   7:(4,12)
#  8:(8,0)   9:(8,4)  12:(8,8)  13:(8,12)
# 10:(12,0) 11:(12,4) 14:(12,8) 15:(12,12)
#
# For each block: (left_within_mb_idx or None, top_within_mb_idx or None)
# None means cross-MB neighbor needed
_LUMA_LEFT_NEIGHBOR = [
    None, 0, None, 2, 1, 4, 3, 6,
    None, 8, None, 10, 9, 12, 11, 14,
]
_LUMA_TOP_NEIGHBOR = [
    None, None, 0, 1, None, None, 4, 5,
    2, 3, 8, 9, 6, 7, 12, 13,
]
# For cross-MB: which block index in the left/top MB provides the neighbor
_LUMA_LEFT_MB_BLOCK = [5, None, 7, None, None, None, None, None,
                       13, None, 15, None, None, None, None, None]
_LUMA_TOP_MB_BLOCK = [10, 11, None, None, 14, 15, None, None,
                      None, None, None, None, None, None, None, None]

# Chroma 4x4 block neighbor mapping (2x2 blocks per plane)
# Block positions: 0:(0,0), 1:(0,4), 2:(4,0), 3:(4,4)
_CHROMA_LEFT_NEIGHBOR = [None, 0, None, 2]
_CHROMA_TOP_NEIGHBOR = [None, None, 0, 1]
_CHROMA_LEFT_MB_BLOCK = [1, None, 3, None]
_CHROMA_TOP_MB_BLOCK = [2, 3, None, None]


def _get_cbf_ctx_idx(
    block_cat: int,
    blk_idx: int,
    mb_cbf: list,
    left_mb_cbf: Optional[list],
    top_mb_cbf: Optional[list],
    left_available: bool,
    top_available: bool,
    is_intra: bool = True,
    chroma_ac_offset: int = 0,
) -> int:
    """Compute coded_block_flag context index.

    H.264 Section 9.3.3.1.1.9:
    ctxIdx = ctxIdxBase + condTermFlagA + 2*condTermFlagB

    condTermFlag derivation when neighbor is NOT available:
    - Current MB is intra → condTermFlag = 1
    - Current MB is inter → condTermFlag = 0

    Args:
        block_cat: Block category (0-4)
        blk_idx: Block index (0-15 for luma AC/4x4, 0-3 for chroma AC,
                 0=Cb/1=Cr for chroma DC)
        mb_cbf: Current MB's per-block CBF [0-15: luma, 16-19: Cb AC,
                20-23: Cr AC, 24: luma DC, 25: Cb DC, 26: Cr DC]
        left_mb_cbf: Left MB's per-block CBF list[27] (None if unavailable)
        top_mb_cbf: Top MB's per-block CBF list[27] (None if unavailable)
        left_available: Whether left MB exists
        top_available: Whether top MB exists
        is_intra: Whether current MB is intra (affects unavailable default)

    Returns:
        Context index for coded_block_flag
    """
    CBF_BASE = {0: 85, 1: 89, 2: 93, 3: 97, 4: 101}
    ctx_base = CBF_BASE[block_cat]

    # Default condTermFlag when neighbor is unavailable (H.264 9.3.3.1.1.9)
    unavail_default = 1 if is_intra else 0

    # DC blocks (block_cat 0 or 3): only cross-MB neighbors
    if block_cat == 0:
        # Luma DC (I_16x16): neighbor is left/top MB's luma DC (index 24)
        if left_available and left_mb_cbf is not None and len(left_mb_cbf) > 24:
            cond_a = 1 if left_mb_cbf[24] else 0
        else:
            cond_a = unavail_default
        if top_available and top_mb_cbf is not None and len(top_mb_cbf) > 24:
            cond_b = 1 if top_mb_cbf[24] else 0
        else:
            cond_b = unavail_default
        return ctx_base + cond_a + 2 * cond_b

    if block_cat == 3:
        # Chroma DC: neighbor is left/top MB's same-plane DC
        # blk_idx: 0=Cb, 1=Cr → mb_cbf index 25 or 26
        dc_idx = 25 + blk_idx
        if left_available and left_mb_cbf is not None and len(left_mb_cbf) > dc_idx:
            cond_a = 1 if left_mb_cbf[dc_idx] else 0
        else:
            cond_a = unavail_default
        if top_available and top_mb_cbf is not None and len(top_mb_cbf) > dc_idx:
            cond_b = 1 if top_mb_cbf[dc_idx] else 0
        else:
            cond_b = unavail_default
        return ctx_base + cond_a + 2 * cond_b

    # Luma blocks (block_cat 1 or 2)
    if block_cat in (1, 2):
        left_within = _LUMA_LEFT_NEIGHBOR[blk_idx]
        top_within = _LUMA_TOP_NEIGHBOR[blk_idx]

        # condTermFlagA
        if left_within is not None:
            cond_a = 1 if mb_cbf[left_within] else 0
        elif left_available and left_mb_cbf is not None:
            cross_idx = _LUMA_LEFT_MB_BLOCK[blk_idx]
            cond_a = 1 if left_mb_cbf[cross_idx] else 0
        else:
            cond_a = unavail_default

        # condTermFlagB
        if top_within is not None:
            cond_b = 1 if mb_cbf[top_within] else 0
        elif top_available and top_mb_cbf is not None:
            cross_idx = _LUMA_TOP_MB_BLOCK[blk_idx]
            cond_b = 1 if top_mb_cbf[cross_idx] else 0
        else:
            cond_b = unavail_default

        return ctx_base + cond_a + 2 * cond_b

    # Chroma AC blocks (block_cat 4): indices 16-19 in mb_cbf for within-MB
    # For cross-MB: Cb uses 16-19, Cr uses 20-23 (controlled by chroma_ac_offset)
    left_within = _CHROMA_LEFT_NEIGHBOR[blk_idx]
    top_within = _CHROMA_TOP_NEIGHBOR[blk_idx]

    if left_within is not None:
        cond_a = 1 if mb_cbf[16 + left_within] else 0
    elif left_available and left_mb_cbf is not None:
        cross_idx = _CHROMA_LEFT_MB_BLOCK[blk_idx]
        cond_a = 1 if left_mb_cbf[16 + chroma_ac_offset + cross_idx] else 0
    else:
        cond_a = unavail_default

    if top_within is not None:
        cond_b = 1 if mb_cbf[16 + top_within] else 0
    elif top_available and top_mb_cbf is not None:
        cross_idx = _CHROMA_TOP_MB_BLOCK[blk_idx]
        cond_b = 1 if top_mb_cbf[16 + chroma_ac_offset + cross_idx] else 0
    else:
        cond_b = unavail_default

    return ctx_base + cond_a + 2 * cond_b


def _decode_residual_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_type: int,
    slice_type: int,
    cbp: int,
    transform_size_8x8_flag: int,
    mb_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decode residual coefficient blocks using CABAC.

    Handles the H.264 residual() syntax for all block categories:
    - block_cat=0: Luma DC (I_16x16) -- 16 coefficients
    - block_cat=1: Luma AC (I_16x16) -- 15 coefficients per block
    - block_cat=2: Luma 4x4 (I_4x4/inter) -- 16 coefficients per block
    - block_cat=3: Chroma DC -- 4 coefficients per block
    - block_cat=4: Chroma AC -- 15 coefficients per block

    H.264 Spec Reference: Section 7.3.5.3

    Args:
        decoder: CABAC decoder.
        contexts: Context models.
        mb_type: Macroblock type value.
        slice_type: Slice type (0=P, 1=B, 2=I).
        cbp: Coded block pattern (bits 0-3: luma, bits 4-5: chroma).
        transform_size_8x8_flag: 1 for 8x8 transform (High profile).
        mb_info: Neighbor information for CBF context derivation.

    Returns:
        Dict with residual block arrays keyed by block type.
    """
    residual = _empty_residual()

    is_16x16 = _is_i_16x16(mb_type, slice_type)
    is_intra = _is_intra_mb_type(mb_type, slice_type)

    # Track per-block CBF for within-MB neighbor context
    # Indices 0-15: luma blocks, 16-19: Cb AC, 20-23: Cr AC,
    # 24: luma DC, 25: Cb DC, 26: Cr DC
    mb_cbf = [0] * 27

    # Get neighbor CBF data
    left_available = mb_info.get('left_available', False) if mb_info else False
    top_available = mb_info.get('top_available', False) if mb_info else False
    left_mb_cbf = mb_info.get('left_mb_cbf', None) if mb_info else None
    top_mb_cbf = mb_info.get('top_mb_cbf', None) if mb_info else None

    if is_16x16:
        cbp_luma = cbp & 0x0F
        cbp_chroma = (cbp >> 4) & 0x03

        # Luma DC: block_cat=0, 16 coefficients, with coded_block_flag
        luma_dc_ctx = _get_cbf_ctx_idx(
            0, 0, mb_cbf, left_mb_cbf, top_mb_cbf,
            left_available, top_available, is_intra,
        )
        residual['luma_dc'] = decode_residual_block_cabac_with_cbf(
            decoder, contexts, max_coeff=16, block_cat=0,
            coded_block_flag_ctx_idx=luma_dc_ctx,
        )
        mb_cbf[24] = 1 if (residual['luma_dc'] is not None
                           and np.any(residual['luma_dc'] != 0)) else 0

        # Luma AC: block_cat=1, 15 coefficients per 4x4 block
        if cbp_luma != 0:
            for i in range(16):
                ctx_idx = _get_cbf_ctx_idx(
                    1, i, mb_cbf, left_mb_cbf, top_mb_cbf,
                    left_available, top_available, is_intra,
                )
                residual['luma_ac'][i] = decode_residual_block_cabac_with_cbf(
                    decoder, contexts, max_coeff=15, block_cat=1,
                    coded_block_flag_ctx_idx=ctx_idx,
                )
                # Track CBF for subsequent blocks
                if residual['luma_ac'][i] is not None:
                    mb_cbf[i] = 1 if np.any(residual['luma_ac'][i] != 0) else 0
    else:
        cbp_luma = cbp & 0x0F
        cbp_chroma = (cbp >> 4) & 0x03

        if transform_size_8x8_flag == 1:
            # 8x8 transform: block_cat=5, 64 coefficients per 8x8 block
            residual['luma_8x8'] = [None] * 4
            for i8x8 in range(4):
                if cbp_luma & (1 << i8x8):
                    residual['luma_8x8'][i8x8] = decode_residual_block_8x8(
                        decoder, contexts, block_cat=5,
                    )
                    if residual['luma_8x8'][i8x8] is not None:
                        has_nonzero = np.any(residual['luma_8x8'][i8x8] != 0)
                        # Mark all 4 sub-blocks as having coefficients
                        for i4x4 in range(4):
                            mb_cbf[i8x8 * 4 + i4x4] = 1 if has_nonzero else 0
        else:
            # I_4x4 or inter: block_cat=2, 16 coefficients per 4x4 block
            for i8x8 in range(4):
                if cbp_luma & (1 << i8x8):
                    for i4x4 in range(4):
                        blk_idx = i8x8 * 4 + i4x4
                        ctx_idx = _get_cbf_ctx_idx(
                            2, blk_idx, mb_cbf, left_mb_cbf, top_mb_cbf,
                            left_available, top_available, is_intra,
                        )
                        residual['luma_4x4'][blk_idx] = (
                            decode_residual_block_cabac_with_cbf(
                                decoder, contexts, max_coeff=16, block_cat=2,
                                coded_block_flag_ctx_idx=ctx_idx,
                            )
                        )
                        if residual['luma_4x4'][blk_idx] is not None:
                            mb_cbf[blk_idx] = 1 if np.any(
                                residual['luma_4x4'][blk_idx] != 0
                            ) else 0

    # Chroma DC: block_cat=3, 4 coefficients, with coded_block_flag
    if cbp_chroma >= 1:
        cb_dc_ctx = _get_cbf_ctx_idx(
            3, 0, mb_cbf, left_mb_cbf, top_mb_cbf,
            left_available, top_available, is_intra,
        )
        residual['chroma_dc_cb'] = decode_residual_block_cabac_with_cbf(
            decoder, contexts, max_coeff=4, block_cat=3,
            coded_block_flag_ctx_idx=cb_dc_ctx,
        )
        mb_cbf[25] = 1 if (residual['chroma_dc_cb'] is not None
                           and np.any(residual['chroma_dc_cb'] != 0)) else 0

        cr_dc_ctx = _get_cbf_ctx_idx(
            3, 1, mb_cbf, left_mb_cbf, top_mb_cbf,
            left_available, top_available, is_intra,
        )
        residual['chroma_dc_cr'] = decode_residual_block_cabac_with_cbf(
            decoder, contexts, max_coeff=4, block_cat=3,
            coded_block_flag_ctx_idx=cr_dc_ctx,
        )
        mb_cbf[26] = 1 if (residual['chroma_dc_cr'] is not None
                           and np.any(residual['chroma_dc_cr'] != 0)) else 0

    # Chroma AC: block_cat=4, 15 coefficients, with cbf
    if cbp_chroma == 2:
        for i in range(4):
            ctx_idx = _get_cbf_ctx_idx(
                4, i, mb_cbf, left_mb_cbf, top_mb_cbf,
                left_available, top_available, is_intra,
            )
            residual['chroma_ac_cb'][i] = decode_residual_block_cabac_with_cbf(
                decoder, contexts, max_coeff=15, block_cat=4,
                coded_block_flag_ctx_idx=ctx_idx,
            )
            if residual['chroma_ac_cb'][i] is not None:
                mb_cbf[16 + i] = 1 if np.any(
                    residual['chroma_ac_cb'][i] != 0
                ) else 0

        # For Cr AC: use Cr-specific CBF tracking (indices 20-23)
        # Within-MB neighbors stored at 16-19 in cr_cbf (same positions as Cb)
        # Cross-MB neighbors need chroma_ac_offset=4 to read from 20-23
        cr_cbf = [0] * 24
        for i in range(4):
            ctx_idx = _get_cbf_ctx_idx(
                4, i, cr_cbf, left_mb_cbf, top_mb_cbf,
                left_available, top_available, is_intra,
                chroma_ac_offset=4,
            )
            residual['chroma_ac_cr'][i] = decode_residual_block_cabac_with_cbf(
                decoder, contexts, max_coeff=15, block_cat=4,
                coded_block_flag_ctx_idx=ctx_idx,
            )
            if residual['chroma_ac_cr'][i] is not None:
                cr_cbf[16 + i] = 1 if np.any(
                    residual['chroma_ac_cr'][i] != 0
                ) else 0
                mb_cbf[20 + i] = cr_cbf[16 + i]

    # Store CBF tracking data in residual for decoder state
    residual['_mb_cbf'] = mb_cbf

    return residual


def _empty_residual() -> Dict[str, Any]:
    """Return an empty residual dict with all None/zero arrays.

    Returns:
        Dict with keys for each residual block type, initialized to
        None or lists of None.
    """
    return {
        'luma_dc': None,
        'luma_ac': [None] * 16,
        'luma_4x4': [None] * 16,
        'luma_8x8': None,
        'chroma_dc_cb': None,
        'chroma_dc_cr': None,
        'chroma_ac_cb': [None] * 4,
        'chroma_ac_cr': [None] * 4,
    }


# Helper functions

def _is_intra_mb_type(mb_type: int, slice_type: int) -> bool:
    """Check if mb_type is an intra type."""
    if slice_type == 2:  # I-slice
        return True
    elif slice_type == 0:  # P-slice
        return mb_type >= 5  # Intra in P-slice
    else:  # B-slice
        return mb_type >= 23  # Intra in B-slice


def _is_i_16x16(mb_type: int, slice_type: int) -> bool:
    """Check if mb_type is I_16x16."""
    if slice_type == 2:  # I-slice
        return 1 <= mb_type <= 24
    elif slice_type == 0:  # P-slice
        return 6 <= mb_type <= 29
    else:  # B-slice
        return 24 <= mb_type <= 47


def _is_i_pcm(mb_type: int, slice_type: int) -> bool:
    """Check if mb_type is I_PCM."""
    if slice_type == 2:  # I-slice
        return mb_type == 25
    elif slice_type == 0:  # P-slice
        return mb_type == 30
    else:  # B-slice
        return mb_type == 48


def _extract_i16x16_cbp(mb_type: int, slice_type: int) -> tuple:
    """Extract CBP from I_16x16 mb_type.

    For I_16x16: mb_type = 1 + pred_mode + 4*cbp_chroma + 12*(cbp_luma != 0)

    Args:
        mb_type: Macroblock type value
        slice_type: Slice type (0=P, 1=B, 2=I)

    Returns:
        Tuple of (cbp_luma, cbp_chroma) where:
        - cbp_luma is 0 or 15 (all blocks have AC coeffs)
        - cbp_chroma is 0, 1, or 2
    """
    # Get base I_16x16 type value (0-23)
    if slice_type == 2:  # I-slice
        base = mb_type - 1
    elif slice_type == 0:  # P-slice
        base = mb_type - 6
    else:  # B-slice
        base = mb_type - 24

    # H.264 Table 7-11 encoding:
    # mb_type = 1 + pred_mode + 4*cbp_chroma + 12*(cbp_luma != 0)
    # So base = pred_mode + 4*cbp_chroma + 12*(cbp_luma != 0)
    cbp_luma = 15 if base >= 12 else 0
    base_without_luma = base % 12
    cbp_chroma = base_without_luma // 4

    return (cbp_luma, cbp_chroma)


def _can_use_8x8_transform(mb_type: int, slice_type: int) -> bool:
    """Check if macroblock can use 8x8 transform."""
    if _is_intra_mb_type(mb_type, slice_type):
        # I_4x4 or I_8x8 can use 8x8
        if slice_type == 2:
            return mb_type == 0
        elif slice_type == 0:
            return mb_type == 5
        else:
            return mb_type == 23
    return True  # Inter MBs can use 8x8


def _get_num_partitions(mb_type: int, slice_type: int) -> int:
    """Get number of partitions for inter mb_type."""
    if slice_type == 0:  # P-slice
        if mb_type == 0:  # P_L0_16x16
            return 1
        elif mb_type in (1, 2):  # P_L0_L0_16x8, P_L0_L0_8x16
            return 2
        elif mb_type == 3:  # P_8x8
            return 4
        else:
            return 0  # Intra
    elif slice_type == 1:  # B-slice
        if mb_type == 0:  # B_Direct_16x16
            return 0
        elif mb_type <= 3:  # B_L0/L1/Bi_16x16
            return 1
        elif mb_type <= 21:  # Partitioned
            return 2
        elif mb_type == 22:  # B_8x8
            return 4
        else:
            return 0  # Intra
    return 0


# Aliases and additional functions for test compatibility


def decode_coded_block_pattern_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_type: int = 0,
    mb_info: Dict[str, Any] = None,
) -> tuple:
    """Decode coded_block_pattern returning separate luma/chroma values.

    Wrapper for decode_cbp_luma/chroma that returns (cbp_luma, cbp_chroma) tuple.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        mb_type: Macroblock type (for context selection)
        mb_info: Neighbor information with cbp values

    Returns:
        Tuple of (cbp_luma, cbp_chroma)
    """
    if mb_info is None:
        mb_info = {'left_available': False, 'top_available': False}

    # Get neighbor CBP values for context derivation
    left_available = mb_info.get('left_available', False)
    top_available = mb_info.get('top_available', False)

    left_cbp = mb_info.get('left_cbp', 0) if left_available else -1
    top_cbp = mb_info.get('top_cbp', 0) if top_available else -1

    left_cbp_luma = (left_cbp & 0x0F) if left_cbp >= 0 else -1
    top_cbp_luma = (top_cbp & 0x0F) if top_cbp >= 0 else -1
    left_cbp_chroma = ((left_cbp >> 4) & 0x03) if left_cbp >= 0 else -1
    top_cbp_chroma = ((top_cbp >> 4) & 0x03) if top_cbp >= 0 else -1

    # Decode luma CBP (4 bits) with neighbor context
    cbp_luma = decode_cbp_luma(decoder, contexts, mb_type, left_cbp_luma, top_cbp_luma)

    # Decode chroma CBP (2 bits) with neighbor context
    cbp_chroma = decode_cbp_chroma(decoder, contexts, mb_type, left_cbp_chroma, top_cbp_chroma)

    return (cbp_luma, cbp_chroma)


def is_i_pcm_macroblock(mb_type: int, slice_type: int) -> bool:
    """Check if macroblock type is I_PCM.

    Args:
        mb_type: Macroblock type value
        slice_type: Slice type (0=P, 1=B, 2=I)

    Returns:
        True if I_PCM macroblock
    """
    return _is_i_pcm(mb_type, slice_type)


def handle_i_pcm_cabac_reinit(decoder: 'CABACDecoder') -> None:
    """Reinitialize CABAC decoder after I_PCM macroblock.

    I_PCM macroblocks bypass CABAC and require reinitialization.

    Args:
        decoder: CABAC decoder to reinitialize
    """
    if hasattr(decoder, 'init_after_pcm'):
        decoder.init_after_pcm()
    elif hasattr(decoder, 'reinit'):
        decoder.reinit()


def align_to_byte_for_i_pcm(decoder: 'CABACDecoder') -> int:
    """Align bitstream to byte boundary for I_PCM data.

    Args:
        decoder: CABAC decoder

    Returns:
        Number of bits skipped for alignment
    """
    if hasattr(decoder, 'byte_align'):
        return decoder.byte_align()
    return 0


def decode_i_pcm_samples(
    reader,
    bit_depth_luma: int = 8,
    bit_depth_chroma: int = 8,
    chroma_format_idc: int = 1,
) -> tuple:
    """Decode raw PCM samples for I_PCM macroblock.

    Args:
        reader: Bit reader for raw sample reading
        bit_depth_luma: Bits per luma sample
        bit_depth_chroma: Bits per chroma sample
        chroma_format_idc: Chroma format (1=4:2:0, 2=4:2:2, 3=4:4:4)

    Returns:
        Tuple of (luma, cb, cr) numpy arrays
    """
    import numpy as np

    # Read 256 luma samples (16x16)
    luma_samples = []
    for _ in range(256):
        if hasattr(reader, 'read_bits'):
            luma_samples.append(reader.read_bits(bit_depth_luma))
        else:
            luma_samples.append(0)

    luma = np.array(luma_samples, dtype=np.uint16 if bit_depth_luma > 8 else np.uint8)
    luma = luma.reshape((16, 16))

    # Determine chroma size based on chroma_format_idc
    if chroma_format_idc == 1:  # 4:2:0
        chroma_size = (8, 8)
        num_chroma = 64
    elif chroma_format_idc == 2:  # 4:2:2
        chroma_size = (16, 8)
        num_chroma = 128
    else:  # 4:4:4
        chroma_size = (16, 16)
        num_chroma = 256

    # Read Cb samples
    cb_samples = []
    for _ in range(num_chroma):
        if hasattr(reader, 'read_bits'):
            cb_samples.append(reader.read_bits(bit_depth_chroma))
        else:
            cb_samples.append(0)

    cb = np.array(cb_samples, dtype=np.uint16 if bit_depth_chroma > 8 else np.uint8)
    cb = cb.reshape(chroma_size)

    # Read Cr samples
    cr_samples = []
    for _ in range(num_chroma):
        if hasattr(reader, 'read_bits'):
            cr_samples.append(reader.read_bits(bit_depth_chroma))
        else:
            cr_samples.append(0)

    cr = np.array(cr_samples, dtype=np.uint16 if bit_depth_chroma > 8 else np.uint8)
    cr = cr.reshape(chroma_size)

    return (luma, cb, cr)


def get_i_pcm_neighbor_nz_count(mb_info: Dict[str, Any] = None) -> int:
    """Get neighbor non-zero coefficient count for I_PCM.

    I_PCM blocks have implied nz_count of 16 (all coefficients present).

    Args:
        mb_info: Macroblock information (optional, unused)

    Returns:
        16 (I_PCM has all coefficients)
    """
    return 16


def get_mb_skip_flag_ctx_idx(
    slice_type: int,
    left_available: bool = True,
    left_skip: bool = False,
    top_available: bool = True,
    top_skip: bool = False,
) -> int:
    """Get context index for mb_skip_flag.

    Args:
        slice_type: 0=P, 1=B
        left_available: True if left MB is available
        left_skip: True if left MB is skipped
        top_available: True if top MB is available
        top_skip: True if top MB is skipped

    Returns:
        Context index for mb_skip_flag
    """
    # H.264 9.3.3.1.1.3: condTermFlagN = 1 if neighbor available AND NOT skip
    cond_term_flag_a = 1 if (left_available and not left_skip) else 0
    cond_term_flag_b = 1 if (top_available and not top_skip) else 0
    ctx_inc = cond_term_flag_a + cond_term_flag_b

    if slice_type == 1:  # B-slice
        return 24 + ctx_inc  # CTX_MB_SKIP_FLAG_B
    else:  # P-slice
        return 11 + ctx_inc  # CTX_MB_SKIP_FLAG_P


def get_mb_field_decoding_flag_ctx_idx(
    left_available: bool = True,
    left_field: int = 0,
    top_available: bool = True,
    top_field: int = 0,
) -> int:
    """Get context index for mb_field_decoding_flag.

    Args:
        left_available: True if left MB is available
        left_field: 1 if left MB is field coded
        top_available: True if top MB is available
        top_field: 1 if top MB is field coded

    Returns:
        Context index
    """
    # Unavailable neighbors are treated as frame-coded
    cond_term_flag_a = 1 if (left_available and left_field) else 0
    cond_term_flag_b = 1 if (top_available and top_field) else 0
    ctx_inc = cond_term_flag_a + cond_term_flag_b
    return CTX_MB_FIELD_DECODING_FLAG + ctx_inc


def decode_slice_data_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    slice_info: Dict[str, Any],
    slice_type: int = None,
    num_mbs: int = None,
) -> List[Dict[str, Any]]:
    """Decode all macroblocks in a slice using CABAC.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        slice_info: Slice-level information
        slice_type: Slice type (optional, extracted from slice_info)
        num_mbs: Number of macroblocks (optional, calculated from dimensions)

    Returns:
        List of decoded macroblock data
    """
    # Extract slice_type from slice_info if not provided
    if slice_type is None:
        slice_type = slice_info.get('slice_type', 2)  # Default to I-slice

    # Calculate num_mbs from dimensions if not provided
    width_in_mbs = slice_info.get('pic_width_in_mbs', slice_info.get('width_in_mbs', 1))
    height_in_mbs = slice_info.get('pic_height_in_mbs', slice_info.get('height_in_mbs', 1))

    if num_mbs is None:
        num_mbs = width_in_mbs * height_in_mbs

    macroblocks = []

    for mb_idx in range(num_mbs):
        mb_x = mb_idx % width_in_mbs
        mb_y = mb_idx // width_in_mbs

        # Build mb_info from neighbors
        mb_info = _build_mb_info(mb_x, mb_y, macroblocks, width_in_mbs, slice_info)

        # Decode macroblock
        mb_data = decode_macroblock_layer_cabac(
            decoder, contexts, slice_type, mb_info
        )
        mb_data['mb_x'] = mb_x
        mb_data['mb_y'] = mb_y
        macroblocks.append(mb_data)

        # Check for end of slice
        if decode_end_of_slice_flag_cabac(decoder, contexts) == 1:
            break

    return macroblocks


def _build_mb_info(
    mb_x: int,
    mb_y: int,
    prev_mbs: List[Dict[str, Any]],
    width_in_mbs: int,
    slice_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Build mb_info dict from neighbor macroblocks.

    Args:
        mb_x, mb_y: Current macroblock position
        prev_mbs: Previously decoded macroblocks
        width_in_mbs: Picture width in macroblocks
        slice_info: Slice-level information

    Returns:
        mb_info dict with neighbor information
    """
    mb_info = {
        'mb_x': mb_x,
        'mb_y': mb_y,
        'left_available': mb_x > 0,
        'top_available': mb_y > 0,
        'left_skip': False,
        'top_skip': False,
        'left_cbp': 0,
        'top_cbp': 0,
    }

    # Get left neighbor info
    if mb_x > 0 and len(prev_mbs) > 0:
        left_idx = len(prev_mbs) - 1
        if left_idx >= 0:
            left_mb = prev_mbs[left_idx]
            mb_info['left_skip'] = left_mb.get('mb_skip_flag', 0) == 1
            mb_info['left_cbp'] = left_mb.get('cbp', 0)
            mb_info['left_mb_type'] = left_mb.get('mb_type', 0)

    # Get top neighbor info
    if mb_y > 0:
        top_idx = len(prev_mbs) - width_in_mbs
        if 0 <= top_idx < len(prev_mbs):
            top_mb = prev_mbs[top_idx]
            mb_info['top_skip'] = top_mb.get('mb_skip_flag', 0) == 1
            mb_info['top_cbp'] = top_mb.get('cbp', 0)
            mb_info['top_mb_type'] = top_mb.get('mb_type', 0)

    # Copy slice-level info
    mb_info.update({
        'num_ref_idx_l0_active': slice_info.get('num_ref_idx_l0_active', 1),
        'num_ref_idx_l1_active': slice_info.get('num_ref_idx_l1_active', 1),
        'transform_8x8_mode_flag': slice_info.get('transform_8x8_mode_flag', False),
        'bit_depth_luma': slice_info.get('bit_depth_luma', 8),
        'bit_depth_chroma': slice_info.get('bit_depth_chroma', 8),
    })

    return mb_info


# Additional helper functions for edge cases

def derive_context_for_first_mb(
    slice_info: Dict[str, Any] = None,
    slice_type: int = None,
) -> Dict[str, Any]:
    """Derive context information for first macroblock in slice.

    First MB has no left/top neighbors within the slice.

    Args:
        slice_info: Slice-level information (optional)
        slice_type: Slice type (optional, alternative to slice_info)

    Returns:
        mb_info for first macroblock
    """
    return {
        'mb_x': 0,
        'mb_y': 0,
        'left_available': False,
        'top_available': False,
        'left_skip': False,
        'top_skip': False,
        'left_cbp': 0,
        'top_cbp': 0,
        'first_mb_in_slice': True,
        'slice_type': slice_type if slice_type is not None else (
            slice_info.get('slice_type', 2) if slice_info else 2
        ),
    }


def is_across_slice_boundary(
    curr_mb_addr: int,
    neighbor_mb_addr: int,
    first_mb_in_slice: int,
) -> bool:
    """Check if neighbor is across slice boundary.

    Args:
        curr_mb_addr: Current macroblock address
        neighbor_mb_addr: Neighbor macroblock address
        first_mb_in_slice: First MB address in current slice

    Returns:
        True if neighbor is in different slice
    """
    if neighbor_mb_addr < 0:
        return True
    return neighbor_mb_addr < first_mb_in_slice


def get_top_neighbor_for_context(
    mb_addr: int,
    width_in_mbs: int = None,
    first_mb_in_slice: int = 0,
    pic_width_in_mbs: int = None,
) -> Dict[str, Any]:
    """Get top neighbor information for context derivation.

    Args:
        mb_addr: Current macroblock address
        width_in_mbs: Picture width in MBs (deprecated, use pic_width_in_mbs)
        first_mb_in_slice: First MB in slice
        pic_width_in_mbs: Picture width in MBs

    Returns:
        Dict with top_available and top_addr info
    """
    # Support both parameter names
    w = pic_width_in_mbs if pic_width_in_mbs is not None else width_in_mbs
    if w is None:
        w = 1

    mb_y = mb_addr // w
    result = {
        'top_available': False,
        'top_addr': None,
    }

    if mb_y == 0:
        return result

    top_addr = mb_addr - w
    if top_addr < first_mb_in_slice:
        return result

    result['top_available'] = True
    result['top_addr'] = top_addr
    return result


def cabac_byte_alignment_after_slice(decoder: 'CABACDecoder') -> int:
    """Byte-align CABAC bitstream after slice data.

    Args:
        decoder: CABAC decoder

    Returns:
        Number of bits skipped for alignment
    """
    if hasattr(decoder, 'byte_align'):
        return decoder.byte_align()
    return 0


def should_decode_end_of_slice(
    mb_skip_flag: int = 0,
    is_last_mb_in_slice: bool = False,
) -> bool:
    """Check if end_of_slice_flag should be decoded.

    end_of_slice_flag is decoded after every non-skipped MB.

    Args:
        mb_skip_flag: Whether current MB was skipped
        is_last_mb_in_slice: Whether this is the last MB in slice

    Returns:
        True if end_of_slice_flag should be decoded
    """
    # Always decode after non-skip MB
    if mb_skip_flag == 0:
        return True
    # Also decode if last MB in slice
    return is_last_mb_in_slice


def mb_addr_to_xy(
    mb_addr: int,
    width_in_mbs: int = None,
    pic_width_in_mbs: int = None,
) -> tuple:
    """Convert macroblock address to (x, y) coordinates.

    Args:
        mb_addr: Macroblock raster scan address
        width_in_mbs: Picture width in MBs (deprecated)
        pic_width_in_mbs: Picture width in MBs

    Returns:
        Tuple of (mb_x, mb_y)
    """
    w = pic_width_in_mbs if pic_width_in_mbs is not None else width_in_mbs
    if w is None:
        w = 1
    mb_x = mb_addr % w
    mb_y = mb_addr // w
    return (mb_x, mb_y)


def get_left_mb_addr(
    mb_addr: int,
    width_in_mbs: int = None,
    pic_width_in_mbs: int = None,
) -> Optional[int]:
    """Get left neighbor macroblock address.

    Args:
        mb_addr: Current macroblock address
        width_in_mbs: Picture width in MBs (deprecated)
        pic_width_in_mbs: Picture width in MBs

    Returns:
        Left neighbor address or None if at left edge
    """
    w = pic_width_in_mbs if pic_width_in_mbs is not None else width_in_mbs
    if w is None:
        w = 1
    mb_x = mb_addr % w
    if mb_x == 0:
        return None
    return mb_addr - 1


def get_top_mb_addr(
    mb_addr: int,
    width_in_mbs: int = None,
    pic_width_in_mbs: int = None,
) -> Optional[int]:
    """Get top neighbor macroblock address.

    Args:
        mb_addr: Current macroblock address
        width_in_mbs: Picture width in MBs (deprecated)
        pic_width_in_mbs: Picture width in MBs

    Returns:
        Top neighbor address or None if at top edge
    """
    w = pic_width_in_mbs if pic_width_in_mbs is not None else width_in_mbs
    if w is None:
        w = 1
    mb_y = mb_addr // w
    if mb_y == 0:
        return None
    return mb_addr - w


def get_neighbor_availability(
    mb_addr: int,
    width_in_mbs: int = None,
    height_in_mbs: int = None,
    pic_width_in_mbs: int = None,
    pic_height_in_mbs: int = None,
    first_mb_in_slice: int = 0,
) -> Dict[str, bool]:
    """Get availability of all neighbor macroblocks.

    Args:
        mb_addr: Current macroblock address
        width_in_mbs: Picture width in MBs (deprecated)
        height_in_mbs: Picture height in MBs (deprecated)
        pic_width_in_mbs: Picture width in MBs
        pic_height_in_mbs: Picture height in MBs
        first_mb_in_slice: First MB address in slice

    Returns:
        Dict with left, top, top_left, top_right availability
    """
    w = pic_width_in_mbs if pic_width_in_mbs is not None else width_in_mbs
    if w is None:
        w = 1

    h = pic_height_in_mbs if pic_height_in_mbs is not None else height_in_mbs

    mb_x = mb_addr % w
    mb_y = mb_addr // w

    # Check slice boundary for neighbors
    left_addr = mb_addr - 1 if mb_x > 0 else -1
    top_addr = mb_addr - w if mb_y > 0 else -1

    return {
        'left': mb_x > 0 and left_addr >= first_mb_in_slice,
        'top': mb_y > 0 and top_addr >= first_mb_in_slice,
        'top_left': mb_x > 0 and mb_y > 0 and (mb_addr - w - 1) >= first_mb_in_slice,
        'top_right': mb_x < w - 1 and mb_y > 0 and (mb_addr - w + 1) >= first_mb_in_slice,
    }


def get_mbaff_pair_addr(
    mb_addr: int,
    pic_width_in_mbs: int = None,
) -> int:
    """Get the pair address for MBAFF mode.

    In MBAFF, MBs are coded in pairs. Returns the base address of the pair.

    Args:
        mb_addr: Macroblock address
        pic_width_in_mbs: Picture width in MBs (unused, for API compat)

    Returns:
        Base address of the MB pair (even address)
    """
    return (mb_addr // 2) * 2


def get_mbaff_neighbor(
    mb_addr: int,
    width_in_mbs: int,
    is_field_mb: bool,
    neighbor: str,
) -> Optional[int]:
    """Get neighbor address in MBAFF mode.

    MBAFF neighbor derivation is more complex than raster scan.

    Args:
        mb_addr: Current macroblock address
        width_in_mbs: Picture width in MBs
        is_field_mb: True if current MB is field coded
        neighbor: 'left', 'top', 'top_left', 'top_right'

    Returns:
        Neighbor address or None if unavailable
    """
    pair_addr = get_mbaff_pair_addr(mb_addr)
    is_top_mb = (mb_addr % 2 == 0)
    mb_x = (pair_addr // 2) % (width_in_mbs // 1)
    mb_y = (pair_addr // 2) // (width_in_mbs // 1)

    if neighbor == 'left':
        if mb_x == 0:
            return None
        return pair_addr - 2 + (0 if is_top_mb else 1)
    elif neighbor == 'top':
        if mb_y == 0 and is_top_mb:
            return None
        if is_top_mb:
            return pair_addr - width_in_mbs * 2 + 1
        return pair_addr  # Top of bottom MB is its pair

    return None


def derive_mbaff_context_info(
    mb_pair_info: Dict[str, Any] = None,
    mb_addr: int = None,
    width_in_mbs: int = None,
    mb_field_decoding_flag: int = None,
) -> Dict[str, Any]:
    """Derive MBAFF-specific context information.

    Args:
        mb_pair_info: Dict with mb_addr, is_top_mb, etc. (alternative input)
        mb_addr: Macroblock address
        width_in_mbs: Picture width in MBs
        mb_field_decoding_flag: Field/frame flag

    Returns:
        Context info dict
    """
    # Support dict-based input
    if mb_pair_info is not None:
        mb_addr = mb_pair_info.get('mb_addr', 0)
        is_top_mb = mb_pair_info.get('is_top_mb', True)
        is_field_mb = mb_pair_info.get('pair_field_flag', 0) == 1
        width_in_mbs = mb_pair_info.get('pic_width_in_mbs', 1)
    else:
        if mb_addr is None:
            mb_addr = 0
        is_top_mb = (mb_addr % 2 == 0)
        is_field_mb = (mb_field_decoding_flag == 1) if mb_field_decoding_flag is not None else False

    pair_addr = get_mbaff_pair_addr(mb_addr)
    mb_x = (pair_addr // 2) % (width_in_mbs if width_in_mbs else 1)
    mb_y = (pair_addr // 2) // (width_in_mbs if width_in_mbs else 1)

    return {
        'pair_addr': pair_addr,
        'is_top_mb': is_top_mb,
        'is_field_mb': is_field_mb,
        'partner_addr': pair_addr + (1 if is_top_mb else -1),
        'effective_top_available': not is_top_mb or mb_y > 0,
        'effective_left_available': mb_x > 0,
    }


def is_b_direct_16x16(mb_type: int, slice_type: int) -> bool:
    """Check if macroblock type is B_Direct_16x16.

    Args:
        mb_type: Macroblock type value
        slice_type: Slice type (0=P, 1=B, 2=I)

    Returns:
        True if B_Direct_16x16
    """
    if slice_type != 1:  # Not B-slice
        return False
    return mb_type == 0  # B_Direct_16x16 is mb_type 0 in B-slice


def should_decode_ref_idx(
    num_ref_idx_active: int,
    slice_type: int = None,
    mb_type: int = None,
) -> bool:
    """Check if ref_idx should be decoded.

    ref_idx is only decoded when there's more than one reference.

    Args:
        num_ref_idx_active: Number of active references
        slice_type: Slice type (optional)
        mb_type: Macroblock type (optional)

    Returns:
        True if ref_idx should be decoded
    """
    if num_ref_idx_active <= 1:
        return False

    # Skip for direct modes if slice_type and mb_type provided
    if slice_type is not None and mb_type is not None:
        if slice_type == 1 and mb_type == 0:  # B_Direct_16x16
            return False

    return True
