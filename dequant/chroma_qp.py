from __future__ import annotations


def get_qpc_from_table(qpi: int) -> int:
    table = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34,
        35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
    ]

    qpi = max(0, min(51, int(qpi)))
    return table[qpi]


def calculate_chroma_qp(luma_qp: int, chroma_qp_index_offset: int) -> int:
    qpi = int(luma_qp) + int(chroma_qp_index_offset)
    qpi = max(0, min(51, qpi))
    return get_qpc_from_table(qpi)


def calculate_cb_qp(luma_qp: int, chroma_qp_index_offset: int) -> int:
    return calculate_chroma_qp(luma_qp, chroma_qp_index_offset=chroma_qp_index_offset)


def calculate_cr_qp(luma_qp: int, second_chroma_qp_index_offset: int) -> int:
    qpi = int(luma_qp) + int(second_chroma_qp_index_offset)
    qpi = max(0, min(51, qpi))
    return get_qpc_from_table(qpi)
