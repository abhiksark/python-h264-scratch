from inter.weighted_pred import WeightTable, WeightTableBSlice
from inter.weighted_pred import parse_pred_weight_table_b_slice as _parse_pred_weight_table_b_slice


def parse_pred_weight_table(reader, num_ref_idx_l0: int, chroma_format_idc: int) -> WeightTable:
    luma_log2_weight_denom = reader.read_ue()

    if chroma_format_idc != 0:
        chroma_log2_weight_denom = reader.read_ue()
    else:
        chroma_log2_weight_denom = 0

    table = WeightTable(
        luma_log2_weight_denom=luma_log2_weight_denom,
        chroma_log2_weight_denom=chroma_log2_weight_denom,
    )

    default_luma_weight = 1 << luma_log2_weight_denom
    default_chroma_weight = 1 << chroma_log2_weight_denom

    for i in range(num_ref_idx_l0):
        if reader.read_flag():
            luma_weight = reader.read_se()
            luma_offset = reader.read_se()
        else:
            luma_weight = default_luma_weight
            luma_offset = 0
        table.set_luma_weight(i, luma_weight, luma_offset)

        if chroma_format_idc == 0:
            continue

        if reader.read_flag():
            cb_weight = reader.read_se()
            cb_offset = reader.read_se()
            cr_weight = reader.read_se()
            cr_offset = reader.read_se()
        else:
            cb_weight = default_chroma_weight
            cb_offset = 0
            cr_weight = default_chroma_weight
            cr_offset = 0

        table.set_chroma_weight(i, cb_weight, cb_offset, cr_weight, cr_offset)

    return table


def parse_pred_weight_table_b_slice(
    reader,
    num_ref_idx_l0: int,
    num_ref_idx_l1: int,
    chroma_format_idc: int,
) -> WeightTableBSlice:
    table_l0, table_l1 = _parse_pred_weight_table_b_slice(
        reader,
        num_ref_idx_l0=num_ref_idx_l0,
        num_ref_idx_l1=num_ref_idx_l1,
        chroma_array_type=chroma_format_idc,
    )

    table_b = WeightTableBSlice(
        luma_log2_weight_denom=table_l0.luma_log2_weight_denom,
        chroma_log2_weight_denom=table_l0.chroma_log2_weight_denom,
    )

    for i in range(num_ref_idx_l0):
        w, o = table_l0.get_luma_weight(i)
        table_b.set_l0_luma_weight(i, w, o)

    for i in range(num_ref_idx_l1):
        w, o = table_l1.get_luma_weight(i)
        table_b.set_l1_luma_weight(i, w, o)

    return table_b
