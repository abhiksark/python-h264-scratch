# h264/trace_decode.py
"""Detailed trace of block decoding to find bit drift."""

import numpy as np
from bitstream.bit_reader import BitReader
from decoder.decoder import H264Decoder
from entropy import cavlc

# Patch to trace every decode step
original_decode_residual_block = cavlc.decode_residual_block

def traced_decode_residual_block(reader, nC, max_coeffs=16):
    """Trace every decode step."""
    start_pos = reader.position
    print(f"\n--- decode_residual_block: nC={nC}, max_coeffs={max_coeffs}, bit_pos={start_pos} ---")

    # Read coeff_token
    coeff_token_start = reader.position
    total_coeff, trailing_ones = cavlc.decode_coeff_token(reader, nC)
    coeff_token_end = reader.position
    print(f"  coeff_token: TC={total_coeff}, T1={trailing_ones} "
          f"(bits {coeff_token_start}->{coeff_token_end}, consumed {coeff_token_end - coeff_token_start})")

    if total_coeff == 0:
        print(f"  TC=0, returning early")
        return cavlc.CAVLCBlock(
            total_coeff=0,
            trailing_ones=0,
            coefficients=np.zeros(max_coeffs, dtype=np.int32)
        )

    # Read trailing ones signs
    signs_start = reader.position
    if trailing_ones > 0:
        signs = []
        for _ in range(trailing_ones):
            signs.append(-1 if reader.read_bit() else 1)
        signs.reverse()
    else:
        signs = []
    signs_end = reader.position
    print(f"  trailing_ones signs: {signs} (bits {signs_start}->{signs_end})")

    # Read levels
    levels_start = reader.position
    num_levels = total_coeff - trailing_ones
    levels = []
    suffix_length = 0
    if total_coeff > 10 and trailing_ones < 3:
        suffix_length = 1

    for i in range(num_levels):
        level_start = reader.position
        # Read level_prefix
        level_prefix = 0
        while reader.read_bit() == 0:
            level_prefix += 1
            if level_prefix > 31:
                raise ValueError("level_prefix too large")

        # Determine suffix_length
        if suffix_length == 0:
            if level_prefix < 14:
                level_suffix_size = 0
            elif level_prefix == 14:
                level_suffix_size = 4
            else:
                level_suffix_size = level_prefix - 3
        else:
            level_suffix_size = suffix_length

        level_suffix = 0
        if level_suffix_size > 0:
            level_suffix = reader.read_bits(level_suffix_size)

        # Calculate level_code
        level_code = (min(15, level_prefix) << suffix_length) + level_suffix
        if level_prefix >= 15:
            level_code += 15

        # Sign adjustment
        if i == 0 and trailing_ones < 3:
            level_code += 2
        if level_code % 2 == 0:
            level = (level_code + 2) >> 1
        else:
            level = (-level_code - 1) >> 1

        levels.append(level)
        level_end = reader.position
        print(f"    level[{i}]: prefix={level_prefix}, suffix={level_suffix}, code={level_code}, "
              f"level={level} (bits {level_start}->{level_end})")

        # Update suffix_length
        if suffix_length == 0:
            suffix_length = 1
        if abs(level) > (3 << (suffix_length - 1)):
            suffix_length += 1
        if suffix_length > 6:
            suffix_length = 6

    levels_end = reader.position
    print(f"  levels: {levels} (bits {levels_start}->{levels_end})")

    # Read total_zeros
    max_valid_zeros = max_coeffs - total_coeff
    total_zeros = 0
    if total_coeff < max_coeffs:
        tz_start = reader.position
        total_zeros = cavlc.decode_total_zeros(reader, total_coeff, max_coeffs)
        tz_end = reader.position
        print(f"  total_zeros: {total_zeros} (max_valid={max_valid_zeros}, bits {tz_start}->{tz_end})")
        if total_zeros > max_valid_zeros:
            print(f"  *** ERROR: total_zeros={total_zeros} > max_valid={max_valid_zeros} ***")
            # Show bit context
            print(f"  Bit context around position {tz_start}:")
            # This won't work without rewinding, but we can show current position

    # Read run_before
    zeros_left = total_zeros
    runs = []
    for i in range(total_coeff - 1):
        if zeros_left == 0:
            runs.append(0)
        else:
            run_start = reader.position
            run = cavlc.decode_run_before(reader, zeros_left)
            run_end = reader.position
            runs.append(run)
            print(f"    run_before[{i}]: {run} (zeros_left={zeros_left}, bits {run_start}->{run_end})")
            zeros_left -= run

    # Build coefficient array
    all_coeffs = signs + levels
    all_coeffs.reverse()

    coeffs = np.zeros(max_coeffs, dtype=np.int32)
    pos = 0
    for i, coeff in enumerate(all_coeffs):
        if i < len(runs):
            pos += runs[i]
        coeffs[pos] = coeff
        pos += 1

    end_pos = reader.position
    print(f"  Total bits: {start_pos}->{end_pos} ({end_pos - start_pos} bits)")

    return cavlc.CAVLCBlock(
        total_coeff=total_coeff,
        trailing_ones=trailing_ones,
        coefficients=coeffs
    )


def main():
    # Patch the decoder
    cavlc.decode_residual_block = traced_decode_residual_block

    # Also patch in reconstruct.macroblock
    import reconstruct.macroblock as mb_module
    mb_module.decode_residual_block = traced_decode_residual_block

    # And in decoder.decoder
    import decoder.decoder as dec_module
    dec_module.decode_residual_block = traced_decode_residual_block

    decoder = H264Decoder()
    try:
        frames = list(decoder.decode_file('/tmp/test_gradient.264'))
        print(f"\nDecoded {len(frames)} frames successfully")
    except Exception as e:
        print(f"\nDecode failed: {e}")


if __name__ == '__main__':
    main()
