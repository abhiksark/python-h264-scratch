# debug/test_mb_tracker.py
"""Test MB position tracker on failing video.

This script tests the MB position tracker on motion_64x64.264 to identify
where bit drift occurs during macroblock parsing.
"""
from debug.mb_position_tracker import get_tracker
import decoder.decoder as decoder_mod
from decoder import H264Decoder

# Get global tracker instance
tracker = get_tracker()

# Save original decode_macroblock function
# Note: We need to patch it in the decoder module, not in reconstruct module,
# because decoder.py imports it directly
original_decode = decoder_mod.decode_macroblock


def tracked_decode(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs):
    """Wrap decode_macroblock to track positions.

    Args:
        reader: BitReader instance
        sps: Sequence Parameter Set
        pps: Picture Parameter Set
        slice_qp: Slice QP value
        mb_x: Macroblock X coordinate
        mb_y: Macroblock Y coordinate
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments

    Returns:
        Result from original decode_macroblock
    """
    tracker.start_mb(mb_x, mb_y, reader)
    try:
        result = original_decode(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs)
        tracker.end_mb(reader, 'OK')
        return result
    except Exception as e:
        tracker.end_mb(reader, f'FAIL: {str(e)[:50]}')
        raise


# Monkey-patch decode_macroblock with tracking version
# Patch in the decoder module where it's actually used
decoder_mod.decode_macroblock = tracked_decode

# Run decoder on failing video
decoder = H264Decoder()
error_message = None
try:
    list(decoder.decode_file('test_data/motion_64x64.264'))
    print("\n✓ Success!")
except Exception as e:
    error_message = str(e)
    print(f"\n✗ Failed: {e}")

# Print summary
print("\n=== MB Position Summary ===")
num_checkpoints = len(tracker.checkpoints)
print(f"Total checkpoints recorded: {num_checkpoints}")

if num_checkpoints == 0:
    print("ERROR: No checkpoints recorded! The tracker may not be hooked correctly.")
else:
    tracker.print_summary()

    # Print detailed info for first two MBs
    print("\n=== MB(0,0) Details ===")
    mb0_summary = tracker.get_mb_summary(0, 0)
    if mb0_summary:
        print(f"  Start: {mb0_summary['start']}")
        print(f"  End: {mb0_summary['end']}")
        print(f"  Consumed: {mb0_summary['consumed']} bits")
    else:
        print("  No data recorded for MB(0,0)")

    print("\n=== MB(0,1) Details ===")
    mb1_summary = tracker.get_mb_summary(0, 1)
    if mb1_summary:
        print(f"  Start: {mb1_summary['start']}")
        print(f"  End: {mb1_summary['end']}")
        status = mb1_summary['checkpoints'][-1].get('status', 'N/A')
        print(f"  Status: {status}")
        if error_message and 'FAIL' in status:
            print(f"  Error details: {error_message}")
    else:
        print("  No data recorded for MB(0,1)")

    # Analysis
    print("\n=== Bit Drift Analysis ===")
    print("Expected flow (from plan):")
    print("  - Slice header: 28 bits (positions 0-27)")
    print("  - MB(0,0): starts at 28, expected to end around 1288")
    print("  - MB(0,1): should start around 1288")
    print()
    print("Actual flow:")
    if mb0_summary:
        print(f"  - MB(0,0): {mb0_summary['start']} -> {mb0_summary['end']} ({mb0_summary['consumed']} bits)")

    # Calculate total bits for first row
    first_row_start = mb0_summary['start'] if mb0_summary else 0
    mb3_0_summary = tracker.get_mb_summary(3, 0)
    if mb3_0_summary:
        first_row_end = mb3_0_summary['end']
        first_row_consumed = first_row_end - first_row_start
        print(f"  - First row (MB 0,0 to 3,0): {first_row_start} -> {first_row_end} ({first_row_consumed} bits)")
        print(f"  - Expected MB(0,1) start: ~1288")
        print(f"  - Actual MB(0,1) start: {mb1_summary['start'] if mb1_summary else 'N/A'}")
        if mb1_summary:
            drift = mb1_summary['start'] - 1288
            print(f"  - Bit drift: {drift} bits")
            print()
            print(f"The first row of MBs consumed {first_row_consumed} bits instead of ~1260 bits.")
            print(f"This causes MB(0,1) to start at the wrong position, leading to decode failure.")

    # Analyze individual MB consumption in first row
    print("\n=== First Row MB Breakdown ===")
    for mb_x in range(4):  # 4 MBs in first row
        mb_summary = tracker.get_mb_summary(mb_x, 0)
        if mb_summary:
            print(f"  MB({mb_x},0): {mb_summary['consumed']:3d} bits (pos {mb_summary['start']:4d} -> {mb_summary['end']:4d})")

    print("\nAverage per MB in first row: {:.1f} bits".format(first_row_consumed / 4 if mb3_0_summary else 0))
    print("Expected average per MB: ~315 bits (1260 / 4)")
    print("Excess per MB: ~{:.1f} bits".format((first_row_consumed / 4 - 315) if mb3_0_summary else 0))
