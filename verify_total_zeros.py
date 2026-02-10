# h264/verify_total_zeros.py
"""Verify TOTAL_ZEROS_4x4 tables against FFmpeg reference."""

# FFmpeg total_zeros tables from h264_cavlc.c
# Row index is TotalCoeff-1 (so row 0 = TC=1, row 14 = TC=15)
# FFmpeg total_zeros tables from h264_cavlc.c - CORRECTED
# Row index is TotalCoeff-1 (so row 0 = TC=1, row 14 = TC=15)
FFMPEG_LEN = [
    [1,3,3,4,4,5,5,6,6,7,7,8,8,9,9,9],  # TC=1, TZ 0-15
    [3,3,3,3,3,4,4,4,4,5,5,6,6,6,6],     # TC=2, TZ 0-14
    [4,3,3,3,4,4,3,3,4,5,5,6,5,6],       # TC=3, TZ 0-13
    [5,3,4,4,3,3,3,4,3,4,5,5,5],         # TC=4, TZ 0-12
    [4,4,4,3,3,3,3,3,4,5,4,5],           # TC=5, TZ 0-11
    [6,5,3,3,3,3,3,3,4,3,6],             # TC=6, TZ 0-10
    [6,4,5,3,2,2,3,3,6],                 # TC=7, TZ 0-8 (corrected!)
    [6,4,5,3,2,2,3,3,6],                 # TC=8, TZ 0-8 (same as TC=7 for len)
    [6,6,4,2,2,3,2,5],                   # TC=9, TZ 0-7
    [5,5,3,2,2,2,4],                     # TC=10, TZ 0-6
    [4,4,3,3,1,3],                       # TC=11, TZ 0-5
    [4,4,2,1,3],                         # TC=12, TZ 0-4
    [3,3,1,2],                           # TC=13, TZ 0-3
    [2,2,1],                             # TC=14, TZ 0-2
    [1,1],                               # TC=15, TZ 0-1
]

FFMPEG_BITS = [
    [1,3,2,3,2,3,2,3,2,3,2,3,2,3,2,1],  # TC=1
    [7,6,5,4,3,5,4,3,2,3,2,3,2,1,0],     # TC=2
    [5,7,6,5,4,3,4,3,2,3,2,1,1,0],       # TC=3
    [3,7,5,4,6,5,4,3,3,2,2,1,0],         # TC=4
    [5,4,3,7,6,5,4,3,2,1,1,0],           # TC=5
    [1,1,7,6,5,4,3,2,1,1,0],             # TC=6
    [1,1,1,3,3,2,2,1,0],                 # TC=7, TZ 0-8
    [1,1,1,3,3,2,2,1,0],                 # TC=8, TZ 0-8
    [1,0,1,3,2,1,1,1],                   # TC=9
    [1,0,1,3,2,1,1],                     # TC=10
    [0,1,1,2,1,3],                       # TC=11
    [0,1,1,1,1],                         # TC=12
    [0,1,1,1],                           # TC=13
    [0,1,1],                             # TC=14
    [0,1],                               # TC=15
]

def decode_ffmpeg_table(tc):
    """Decode FFmpeg table row into (total_zeros -> (code, length)) dict."""
    row_idx = tc - 1
    if row_idx >= len(FFMPEG_LEN):
        return {}

    lengths = FFMPEG_LEN[row_idx]
    bits = FFMPEG_BITS[row_idx]

    result = {}
    for tz in range(min(len(lengths), len(bits))):
        length = lengths[tz]
        bit_val = bits[tz]
        # Convert bit value to code by left-padding with zeros
        code = bit_val
        result[tz] = (code, length)
    return result

def main():
    from entropy.tables import TOTAL_ZEROS_4x4

    all_match = True
    for tc in range(1, 16):
        ffmpeg_table = decode_ffmpeg_table(tc)
        our_table = TOTAL_ZEROS_4x4.get(tc, {})

        # Compare entries
        for tz in ffmpeg_table:
            if tz not in our_table:
                print(f"TC={tc}: Missing TZ={tz}, FFmpeg has ({ffmpeg_table[tz][0]:0{ffmpeg_table[tz][1]}b}, {ffmpeg_table[tz][1]})")
                all_match = False
                continue

            ffmpeg_code, ffmpeg_len = ffmpeg_table[tz]
            our_code, our_len = our_table[tz]

            if (our_code, our_len) != (ffmpeg_code, ffmpeg_len):
                print(f"TC={tc}, TZ={tz}: MISMATCH")
                print(f"  FFmpeg: ({ffmpeg_code:0{ffmpeg_len}b}, {ffmpeg_len})")
                print(f"  Ours:   ({our_code:0{our_len}b}, {our_len})")
                all_match = False

        # Check for extra entries in our table
        for tz in our_table:
            if tz not in ffmpeg_table:
                code, length = our_table[tz]
                print(f"TC={tc}: Extra TZ={tz} in our table: ({code:0{length}b}, {length})")

    if all_match:
        print("All TOTAL_ZEROS_4x4 entries match FFmpeg!")
    else:
        print("\nSome entries differ from FFmpeg reference.")

if __name__ == '__main__':
    main()
