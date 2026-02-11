"""Debug script to track bit positions at MB boundaries.

This helps identify where bit drift begins by logging positions at the start
and end of each macroblock decode.
"""
from decoder import H264Decoder
from bitstream import BitReader
import reconstruct.macroblock as mb_mod

# Track positions
positions = []

# Patch decode_macroblock to log positions
original_decode_mb = mb_mod.decode_macroblock

def logged_decode_mb(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs):
    pos_start = reader.position

    try:
        result = original_decode_mb(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs)
        pos_end = reader.position
        bits_consumed = pos_end - pos_start

        positions.append({
            'mb': (mb_x, mb_y),
            'start': pos_start,
            'end': pos_end,
            'consumed': bits_consumed,
            'status': 'OK'
        })

        print(f"MB({mb_x},{mb_y}): pos {pos_start:4d} -> {pos_end:4d} ({bits_consumed:3d} bits)")

        return result
    except Exception as e:
        pos_end = reader.position
        positions.append({
            'mb': (mb_x, mb_y),
            'start': pos_start,
            'end': pos_end,
            'consumed': pos_end - pos_start,
            'status': 'FAIL',
            'error': str(e)
        })
        print(f"MB({mb_x},{mb_y}): pos {pos_start:4d} -> {pos_end:4d} FAILED: {e}")
        raise

mb_mod.decode_macroblock = logged_decode_mb

# Decode with position tracking
decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_64x64.264'))
    print(f"\n✓ Success: {len(frames)} frames decoded")
except Exception as e:
    print(f"\n✗ Failed: {e}")

print("\n=== Position Summary ===")
for p in positions:
    status = p['status']
    mb = p['mb']
    if status == 'OK':
        print(f"MB{mb}: {p['start']:4d} -> {p['end']:4d} ({p['consumed']:3d} bits)")
    else:
        print(f"MB{mb}: {p['start']:4d} -> {p['end']:4d} ({p['consumed']:3d} bits) *** {p['error']}")
