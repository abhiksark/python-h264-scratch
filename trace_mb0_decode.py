"""Trace MB(0,0) decode to see exact bit consumption."""
from decoder import H264Decoder
from bitstream import BitReader, iter_nal_units
from parameters import parse_sps, parse_pps
from slice import parse_slice_header
from reconstruct import decode_macroblock

# Load and parse
with open('test_data/motion_64x64.264', 'rb') as f:
    data = f.read()

for nal in iter_nal_units(data):
    if nal.nal_unit_type == 7:
        sps = parse_sps(nal.rbsp)
    elif nal.nal_unit_type == 8:
        pps = parse_pps(nal.rbsp)
    elif nal.nal_unit_type == 5:
        slice_nal = nal

slice_header = parse_slice_header(
    slice_nal.rbsp, sps, pps,
    slice_nal.nal_unit_type, slice_nal.nal_ref_idc
)

reader = BitReader(slice_nal.rbsp)
reader.position = slice_header.header_bit_size

print(f"Starting MB(0,0) at position {reader.position}")
pos_start = reader.position

# Patch reader to log all reads
original_read_ue = reader.read_ue
original_read_se = reader.read_se
original_read_bits = reader.read_bits
original_read_bit = reader.read_bit

reads = []

def logged_read_ue():
    pos = reader.position
    val = original_read_ue()
    reads.append(f"  [{pos:4d}] read_ue() = {val:6d} ({reader.position - pos:2d} bits)")
    return val

def logged_read_se():
    pos = reader.position
    val = original_read_se()
    reads.append(f"  [{pos:4d}] read_se() = {val:6d} ({reader.position - pos:2d} bits)")
    return val

def logged_read_bits(n):
    pos = reader.position
    val = original_read_bits(n)
    reads.append(f"  [{pos:4d}] read_bits({n:2d}) = {val:6d}")
    return val

def logged_read_bit():
    pos = reader.position
    val = original_read_bit()
    if len(reads) == 0 or not reads[-1].startswith(f"  [{pos:4d}] read_bit"):
        reads.append(f"  [{pos:4d}] read_bit() = {val}")
    return val

reader.read_ue = logged_read_ue
reader.read_se = logged_read_se
reader.read_bits = logged_read_bits
reader.read_bit = logged_read_bit

# Decode MB(0,0)
try:
    mb_data = decode_macroblock(
        reader, sps, pps,
        slice_header.slice_qp,
        mb_x=0, mb_y=0
    )
    pos_end = reader.position
    print(f"MB(0,0) decoded successfully")
    print(f"  End position: {pos_end}")
    print(f"  Bits consumed: {pos_end - pos_start}")
    print(f"\nDetailed reads:")
    for r in reads[:50]:  # First 50 reads
        print(r)
    if len(reads) > 50:
        print(f"  ... ({len(reads)-50} more reads)")
except Exception as e:
    print(f"MB(0,0) decode failed: {e}")
    print(f"\nReads before failure:")
    for r in reads[-20:]:  # Last 20 reads
        print(r)
