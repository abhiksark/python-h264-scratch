# h264/debug_nc.py
"""Debug script to trace nC values during decoding."""

import numpy as np
from bitstream.bit_reader import BitReader
from decoder.decoder import H264Decoder
from reconstruct.macroblock import (
    get_luma_neighbor_nz, BLOCK_SCAN_ORDER, decode_macroblock
)
from entropy import calculate_nC

# Patch decode_residual_block to trace nC values
original_decode_residual_block = None
original_decode_macroblock = None
nc_trace = []
current_mb = None

def traced_decode_residual_block(reader, nC, max_coeffs=16):
    """Wrapper to trace nC values."""
    bit_pos = reader.position
    try:
        result = original_decode_residual_block(reader, nC, max_coeffs)
        nc_trace.append({
            'bit_pos': bit_pos,
            'end_pos': reader.position,
            'nC': nC,
            'max_coeffs': max_coeffs,
            'total_coeff': result.total_coeff,
            'mb': current_mb,
            'error': None,
        })
        return result
    except Exception as e:
        nc_trace.append({
            'bit_pos': bit_pos,
            'end_pos': reader.position,
            'nC': nC,
            'max_coeffs': max_coeffs,
            'total_coeff': '?',
            'mb': current_mb,
            'error': str(e),
        })
        raise

def traced_decode_macroblock(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs):
    """Wrapper to track MB boundaries."""
    global current_mb
    current_mb = (mb_x, mb_y)
    bit_pos = reader.position
    nc_trace.append({
        'bit_pos': bit_pos,
        'nC': 'MB_START',
        'max_coeffs': -1,
        'total_coeff': -1,
        'mb': current_mb,
    })
    return original_decode_macroblock(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs)

def main():
    global original_decode_residual_block, original_decode_macroblock

    # Import and patch BEFORE importing decoder (which imports from these modules)
    from entropy import cavlc
    original_decode_residual_block = cavlc.decode_residual_block
    cavlc.decode_residual_block = traced_decode_residual_block

    import reconstruct.macroblock as mb_module
    mb_module.decode_residual_block = traced_decode_residual_block
    original_decode_macroblock = mb_module.decode_macroblock
    mb_module.decode_macroblock = traced_decode_macroblock

    # Now patch in decoder module (which already imported decode_macroblock)
    import decoder.decoder as dec_module
    dec_module.decode_macroblock = traced_decode_macroblock

    # Decode gradient test file (create decoder after patching)
    decoder = H264Decoder()
    try:
        frames = list(decoder.decode_file('/tmp/test_gradient.264'))
        print(f"Decoded {len(frames)} frames successfully")
    except Exception as e:
        print(f"Decode failed: {e}")

    # Print trace
    print(f"\nnC trace ({len(nc_trace)} entries):")
    print("-" * 80)
    for i, entry in enumerate(nc_trace):
        if entry['nC'] == 'MB_START':
            print(f"=== MB{entry['mb']} starts at bit {entry['bit_pos']} ===")
        else:
            err = f" ERROR: {entry['error']}" if entry.get('error') else ""
            print(f"  Block {i:3d}: bit={entry['bit_pos']:5d}->{entry['end_pos']:5d}, nC={entry['nC']:2d}, "
                  f"max={entry['max_coeffs']:2d}, TC={entry['total_coeff']}{err}")

if __name__ == '__main__':
    main()
