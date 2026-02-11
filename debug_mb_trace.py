"""Trace with MB coordinates."""
from decoder import H264Decoder
from bitstream import BitReader
import logging
import reconstruct.macroblock as mb_mod

logging.basicConfig(level=logging.ERROR)

reads = []
current_mb = (None, None)

original_read_ue = BitReader.read_ue
original_read_se = BitReader.read_se
original_decode_mb = mb_mod.decode_macroblock

def tracked_read_ue(self):
    pos = self.position
    val = original_read_ue(self)
    reads.append(f"MB{current_mb} [{pos:4d}] read_ue() = {val:6d} ({self.position - pos:2d} bits)")
    return val

def tracked_read_se(self):
    pos = self.position
    val = original_read_se(self)
    reads.append(f"MB{current_mb} [{pos:4d}] read_se() = {val:6d} ({self.position - pos:2d} bits)")
    return val

def tracked_decode_mb(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs):
    global current_mb
    current_mb = (mb_x, mb_y)
    return original_decode_mb(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs)

BitReader.read_ue = tracked_read_ue
BitReader.read_se = tracked_read_se
mb_mod.decode_macroblock = tracked_decode_mb

decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_64x64.264'))
except Exception as e:
    for r in reads[-60:]:  # Last 60 reads
        print(r)
    print(f"\nError: {e}")
