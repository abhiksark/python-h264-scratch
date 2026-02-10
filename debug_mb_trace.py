# h264/debug_mb_trace.py
"""Detailed MB-by-MB trace to find bit drift."""

import numpy as np
from bitstream.bit_reader import BitReader
from bitstream.nal_parser import extract_nal_units
from parameters.sps import parse_sps
from parameters.pps import parse_pps
from slice.slice_header import parse_slice_header
from entropy import cavlc


def trace_macroblocks():
    """Trace MB parsing in detail."""
    with open('/tmp/test_gradient.264', 'rb') as f:
        data = f.read()

    nals = extract_nal_units(data)

    # Parse parameters
    sps = None
    pps = None
    for nal in nals:
        if nal.nal_unit_type == 7:
            sps = parse_sps(nal.rbsp)
            print(f"SPS: profile={sps.profile_idc}, {sps.pic_width_in_mbs}x{sps.pic_height_in_map_units} MBs")
            print(f"     poc_type={sps.pic_order_cnt_type}")
        elif nal.nal_unit_type == 8:
            pps = parse_pps(nal.rbsp, is_high_profile=False)
            print(f"PPS: entropy_mode={pps.entropy_coding_mode_flag}, slice_groups={pps.num_slice_groups_minus1 + 1}")

    # Find slice
    for nal in nals:
        if nal.nal_unit_type == 5:  # IDR
            print(f"\n=== IDR Slice (NAL bytes={len(nal.rbsp)}) ===")

            # Parse slice header
            header = parse_slice_header(nal.rbsp, sps, pps, nal.nal_unit_type, nal.nal_ref_idc)
            header_end = header.header_bit_size
            reader = BitReader(nal.rbsp)
            reader.skip_bits(header_end)  # Skip past slice header
            print(f"Slice header: {header_end} bits")
            print(f"  slice_type={header.slice_type}, qp={header.slice_qp_delta + pps.pic_init_qp_minus26 + 26}")

            frame_width_mbs = sps.pic_width_in_mbs
            frame_height_mbs = sps.pic_height_in_map_units

            # Trace each MB
            total_mbs = frame_width_mbs * frame_height_mbs
            print(f"\nTracing {total_mbs} macroblocks...")

            for mb_idx in range(min(total_mbs, 20)):  # First 20 MBs
                mb_x = mb_idx % frame_width_mbs
                mb_y = mb_idx // frame_width_mbs
                mb_start = reader.position

                try:
                    # Read mb_type (ue(v))
                    mb_type = reader.read_ue()
                    type_end = reader.position

                    # Determine MB kind
                    if mb_type == 0:
                        kind = "I_4x4"
                        # Would need to decode pred modes and residual
                    elif 1 <= mb_type <= 24:
                        kind = "I_16x16"
                        # Extract prediction info
                        pred_mode = (mb_type - 1) % 4
                        cbp_chroma = ((mb_type - 1) // 4) % 3
                        cbp_luma = 15 if (mb_type - 1) >= 12 else 0
                        kind += f" pred={pred_mode} cbp_c={cbp_chroma} cbp_l={cbp_luma}"
                    elif mb_type == 25:
                        kind = "I_PCM"
                    else:
                        kind = f"INVALID({mb_type})"

                    print(f"\nMB({mb_x},{mb_y}) @ bit {mb_start}: type={mb_type} ({kind})")
                    print(f"  mb_type consumed {type_end - mb_start} bits")

                    # For I_16x16, trace DC decode
                    if 1 <= mb_type <= 24:
                        # Would need to decode intra_chroma_pred_mode
                        chroma_pred = reader.read_ue()
                        print(f"  intra_chroma_pred_mode={chroma_pred} @ {reader.position}")

                        # mb_qp_delta
                        qp_delta = reader.read_se()
                        print(f"  mb_qp_delta={qp_delta} @ {reader.position}")

                        # Decode DC block
                        dc_start = reader.position
                        dc_block = cavlc.decode_residual_block(reader, nC=0, max_coeffs=16)
                        dc_end = reader.position
                        print(f"  Luma DC: TC={dc_block.total_coeff}, bits={dc_end - dc_start} @ {dc_end}")

                        # Would need to decode chroma and AC...
                        # For now just note the position
                        print(f"  (skipping chroma/AC decode for trace)")

                        # We can't continue without full decode
                        break

                    elif mb_type == 0:
                        # I_4x4 - decode pred modes
                        print(f"  I_4x4 decoding pred modes...")
                        for i in range(16):
                            flag = reader.read_bit()
                            if flag:
                                # Use predicted mode
                                pass
                            else:
                                # Read rem_intra4x4_pred_mode
                                rem = reader.read_bits(3)
                        print(f"  After pred modes @ {reader.position}")

                        # intra_chroma_pred_mode
                        chroma_pred = reader.read_ue()
                        print(f"  intra_chroma_pred_mode={chroma_pred} @ {reader.position}")

                        # coded_block_pattern - this is a ue(v) that needs mapping
                        cbp_coded = reader.read_ue()
                        print(f"  cbp_coded={cbp_coded} @ {reader.position}")

                        if cbp_coded != 0:
                            qp_delta = reader.read_se()
                            print(f"  mb_qp_delta={qp_delta} @ {reader.position}")

                        # Can't continue without full decode
                        break

                except Exception as e:
                    print(f"  ERROR at bit {reader.position}: {e}")
                    import traceback
                    traceback.print_exc()
                    break

            break  # Only process first slice


if __name__ == '__main__':
    trace_macroblocks()
