from __future__ import annotations

from typing import List, Optional

from parameters.pps import PPS
from parameters.sps import SPS


def _pic_size_in_map_units(sps: SPS) -> int:
    return (sps.pic_width_in_mbs_minus1 + 1) * (sps.pic_height_in_map_units_minus1 + 1)


def _pic_width_in_mbs(sps: SPS) -> int:
    return sps.pic_width_in_mbs_minus1 + 1


def generate_slice_group_map(sps: SPS, pps: PPS) -> List[int]:
    pic_width_in_mbs = _pic_width_in_mbs(sps)
    pic_size_in_map_units = _pic_size_in_map_units(sps)

    num_slice_groups = int(pps.num_slice_groups_minus1) + 1
    if num_slice_groups <= 1:
        return [0] * pic_size_in_map_units

    slice_group_map_type = int(getattr(pps, "slice_group_map_type", 0))

    if slice_group_map_type == 0:
        run_length_minus1 = list(getattr(pps, "run_length_minus1", []))
        if len(run_length_minus1) < num_slice_groups:
            run_length_minus1 = run_length_minus1 + [0] * (num_slice_groups - len(run_length_minus1))

        result: List[int] = []
        group = 0
        while len(result) < pic_size_in_map_units:
            run = int(run_length_minus1[group]) + 1
            remaining = pic_size_in_map_units - len(result)
            result.extend([group] * min(run, remaining))
            group = (group + 1) % num_slice_groups
        return result

    if slice_group_map_type == 1:
        result = [0] * pic_size_in_map_units
        for i in range(pic_size_in_map_units):
            x = i % pic_width_in_mbs
            y = i // pic_width_in_mbs
            result[i] = (x + ((y * num_slice_groups) // 2)) % num_slice_groups
        return result

    if slice_group_map_type == 2:
        top_left = list(getattr(pps, "top_left", []))
        bottom_right = list(getattr(pps, "bottom_right", []))
        background_group = num_slice_groups - 1

        result = [background_group] * pic_size_in_map_units
        for group in range(num_slice_groups - 1):
            if group >= len(top_left) or group >= len(bottom_right):
                break
            tl = int(top_left[group])
            br = int(bottom_right[group])

            tl_x = tl % pic_width_in_mbs
            tl_y = tl // pic_width_in_mbs
            br_x = br % pic_width_in_mbs
            br_y = br // pic_width_in_mbs

            for y in range(tl_y, br_y + 1):
                for x in range(tl_x, br_x + 1):
                    idx = y * pic_width_in_mbs + x
                    if 0 <= idx < pic_size_in_map_units and result[idx] == background_group:
                        result[idx] = group
        return result

    if slice_group_map_type == 3:
        change_cycle = int(getattr(pps, "slice_group_change_cycle", 0))
        if change_cycle <= 0:
            return [1] * pic_size_in_map_units

        rate = int(getattr(pps, "slice_group_change_rate_minus1", 0)) + 1
        num_in_group0 = min(pic_size_in_map_units, rate * change_cycle)
        result = [1] * pic_size_in_map_units
        for i in range(num_in_group0):
            result[i] = 0
        return result

    if slice_group_map_type == 4:
        change_cycle = int(getattr(pps, "slice_group_change_cycle", 0))
        rate = int(getattr(pps, "slice_group_change_rate_minus1", 0)) + 1
        num_in_group0 = min(pic_size_in_map_units, rate * max(0, change_cycle))
        direction = bool(getattr(pps, "slice_group_change_direction_flag", False))

        result = [1] * pic_size_in_map_units
        if direction:
            start = pic_size_in_map_units - num_in_group0
            for i in range(start, pic_size_in_map_units):
                result[i] = 0
        else:
            for i in range(num_in_group0):
                result[i] = 0
        return result

    if slice_group_map_type == 5:
        change_cycle = int(getattr(pps, "slice_group_change_cycle", 0))
        rate = int(getattr(pps, "slice_group_change_rate_minus1", 0)) + 1
        num_changed = min(pic_size_in_map_units, rate * max(0, change_cycle))
        direction = bool(getattr(pps, "slice_group_change_direction_flag", False))

        pic_height_in_map_units = sps.pic_height_in_map_units_minus1 + 1
        if pic_height_in_map_units <= 0:
            return [1] * pic_size_in_map_units

        cols_in_group0 = (num_changed + pic_height_in_map_units - 1) // pic_height_in_map_units
        cols_in_group0 = max(0, min(pic_width_in_mbs, cols_in_group0))

        result = [1] * pic_size_in_map_units
        for idx in range(pic_size_in_map_units):
            x = idx % pic_width_in_mbs
            if direction:
                in_group0 = x >= (pic_width_in_mbs - cols_in_group0)
            else:
                in_group0 = x < cols_in_group0
            if in_group0:
                result[idx] = 0
        return result

    if slice_group_map_type == 6:
        explicit = list(getattr(pps, "slice_group_id", []))
        if len(explicit) >= pic_size_in_map_units:
            return explicit[:pic_size_in_map_units]
        return explicit + [0] * (pic_size_in_map_units - len(explicit))

    return [0] * pic_size_in_map_units


def get_next_mb_address(
    current_mb_addr: int,
    sps: SPS,
    pps: PPS,
    slice_group_id: Optional[int] = None,
) -> int:
    pic_size_in_map_units = _pic_size_in_map_units(sps)
    if current_mb_addr < 0 or current_mb_addr >= pic_size_in_map_units:
        return -1

    if int(pps.num_slice_groups_minus1) <= 0:
        next_addr = current_mb_addr + 1
        return next_addr if next_addr < pic_size_in_map_units else -1

    slice_group_map = generate_slice_group_map(sps, pps)
    if slice_group_id is None:
        slice_group_id = int(slice_group_map[current_mb_addr])

    for addr in range(current_mb_addr + 1, pic_size_in_map_units):
        if int(slice_group_map[addr]) == int(slice_group_id):
            return addr

    return -1
