# h264/verify_run_before.py
"""Verify RUN_BEFORE tables against FFmpeg reference."""

# FFmpeg run_before tables from h264_cavlc.c
# Indexed by zerosLeft, values are (length, code) pairs
# Length 0 means invalid

FFMPEG_RUN_BITS = [
    [1,1],                               # zerosLeft=1
    [1,2,2],                             # zerosLeft=2
    [2,2,2,2],                           # zerosLeft=3
    [2,2,2,3,3],                         # zerosLeft=4
    [2,2,3,3,3,3],                       # zerosLeft=5
    [2,3,3,3,3,3,3],                     # zerosLeft=6
    [3,3,3,3,3,3,3,4,5,6,7,8,9,10,11],   # zerosLeft>=7
]

FFMPEG_RUN_CODE = [
    [1,0],                               # zerosLeft=1
    [1,1,0],                             # zerosLeft=2
    [3,2,1,0],                           # zerosLeft=3
    [3,2,1,1,0],                         # zerosLeft=4
    [3,2,3,2,1,0],                       # zerosLeft=5
    [3,0,1,3,2,5,4],                     # zerosLeft=6
    [7,6,5,4,3,2,1,1,1,1,1,1,1,1,1],     # zerosLeft>=7
]

def decode_ffmpeg_run_table(zl):
    """Decode FFmpeg run_before table into (run -> (code, length)) dict."""
    if zl < 1:
        return {}

    idx = min(zl - 1, 6)  # zerosLeft >= 7 all use same table
    lengths = FFMPEG_RUN_BITS[idx]
    codes = FFMPEG_RUN_CODE[idx]

    result = {}
    for run in range(min(len(lengths), len(codes))):
        length = lengths[run]
        code = codes[run]
        result[run] = (code, length)
    return result

def main():
    from entropy.tables import RUN_BEFORE

    all_match = True
    for zl in range(1, 8):
        ffmpeg_table = decode_ffmpeg_run_table(zl)
        our_table = RUN_BEFORE.get(zl, {})

        print(f"\n=== zerosLeft={zl} ===")

        # Compare entries
        for run in sorted(ffmpeg_table.keys()):
            if run not in our_table:
                ffmpeg_code, ffmpeg_len = ffmpeg_table[run]
                print(f"  MISSING run={run}: FFmpeg has ({ffmpeg_code:0{ffmpeg_len}b}, {ffmpeg_len})")
                all_match = False
                continue

            ffmpeg_code, ffmpeg_len = ffmpeg_table[run]
            our_code, our_len = our_table[run]

            if (our_code, our_len) != (ffmpeg_code, ffmpeg_len):
                print(f"  MISMATCH run={run}:")
                print(f"    FFmpeg: ({ffmpeg_code:0{ffmpeg_len}b}, {ffmpeg_len})")
                print(f"    Ours:   ({our_code:0{our_len}b}, {our_len})")
                all_match = False
            else:
                print(f"  OK run={run}: ({our_code:0{our_len}b}, {our_len})")

        # Check for extra entries in our table
        for run in our_table:
            if run not in ffmpeg_table:
                code, length = our_table[run]
                print(f"  EXTRA run={run} in our table: ({code:0{length}b}, {length})")

    if all_match:
        print("\n\nAll RUN_BEFORE entries match FFmpeg!")
    else:
        print("\n\nSome entries differ from FFmpeg reference.")

if __name__ == '__main__':
    main()
