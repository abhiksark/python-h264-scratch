"""Trace bit consumption with context during decode."""
import sys
from decoder import H264Decoder
from bitstream import BitReader
import logging

logging.basicConfig(level=logging.ERROR)

# Open trace file
trace_file = open('/tmp/decode_trace.txt', 'w')

# Track MB being decoded
current_mb = None

# Patch BitReader to log every read
original_read_ue = BitReader.read_ue
original_read_se = BitReader.read_se

def traced_read_ue(self):
    pos_before = self.position
    value = original_read_ue(self)
    bits_consumed = self.position - pos_before
    if current_mb is not None:
        trace_file.write(f"MB{current_mb} [{pos_before:4d}] read_ue() = {value:3d} ({bits_consumed:2d} bits) -> {self.position}\n")
        trace_file.flush()
    return value

def traced_read_se(self):
    pos_before = self.position
    value = original_read_se(self)
    bits_consumed = self.position - pos_before
    if current_mb is not None:
        trace_file.write(f"MB{current_mb} [{pos_before:4d}] read_se() = {value:3d} ({bits_consumed:2d} bits) -> {self.position}\n")
        trace_file.flush()
    return value

# Patch decode_macroblock to track current MB
import reconstruct.macroblock as mb_module
original_decode_mb = mb_module.decode_macroblock

def traced_decode_macroblock(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs):
    global current_mb
    current_mb = (mb_x, mb_y)
    trace_file.write(f"\n=== MB({mb_x}, {mb_y}) START at bit position {reader.position} ===\n")
    trace_file.flush()
    try:
        result = original_decode_mb(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs)
        trace_file.write(f"=== MB({mb_x}, {mb_y}) END at bit position {reader.position} ===\n\n")
        trace_file.flush()
        return result
    except Exception as e:
        trace_file.write(f"=== MB({mb_x}, {mb_y}) FAILED at bit position {reader.position}: {e} ===\n\n")
        trace_file.flush()
        raise

mb_module.decode_macroblock = traced_decode_macroblock

# Apply patches
BitReader.read_ue = traced_read_ue
BitReader.read_se = traced_read_se

# Decode
print("Running trace... output to /tmp/decode_trace.txt")
decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_64x64.264'))
    print(f"✓ Success: {len(frames)} frames")
except Exception as e:
    print(f"✗ Failed: {e}")
finally:
    trace_file.close()
    print("Trace written to /tmp/decode_trace.txt")
