"""Debug slice header parsing and bit positions."""
from decoder import H264Decoder
from bitstream import BitReader, iter_nal_units
from parameters import parse_sps, parse_pps
from slice import parse_slice_header

# Load file
with open('test_data/motion_64x64.264', 'rb') as f:
    data = f.read()

# Extract NAL units manually
sps_nal = None
pps_nal = None
slice_nal = None

for nal in iter_nal_units(data):
    if nal.nal_unit_type == 7:  # SPS
        sps_nal = nal
        print(f"SPS: start={nal.start_position}, RBSP size={len(nal.rbsp)}")
    elif nal.nal_unit_type == 8:  # PPS
        pps_nal = nal
        print(f"PPS: start={nal.start_position}, RBSP size={len(nal.rbsp)}")
    elif nal.nal_unit_type == 5:  # I-slice
        slice_nal = nal
        print(f"Slice: start={nal.start_position}, RBSP size={len(nal.rbsp)}")

# Parse SPS/PPS
sps = parse_sps(sps_nal.rbsp)
pps = parse_pps(pps_nal.rbsp)

print(f"\nSPS: {sps.pic_width_in_mbs}x{sps.frame_height_in_mbs} MBs")
print(f"PPS: entropy_mode={pps.entropy_coding_mode_flag}")

# Parse slice header
slice_header = parse_slice_header(
    slice_nal.rbsp, sps, pps,
    slice_nal.nal_unit_type, slice_nal.nal_ref_idc
)

print(f"\nSlice header:")
print(f"  slice_type: {slice_header.slice_type} ({slice_header.slice_type_name})")
print(f"  first_mb_in_slice: {slice_header.first_mb_in_slice}")
print(f"  slice_qp: {slice_header.slice_qp}")
print(f"  header_bit_size: {slice_header.header_bit_size} bits")

# Create BitReader and position it after header
reader = BitReader(slice_nal.rbsp)
print(f"\nBitReader created from RBSP:")
print(f"  Initial position: {reader.position}")

reader.position = slice_header.header_bit_size
print(f"  After skipping header: {reader.position}")

# Try to read first MB
print(f"\n=== Attempting to decode MB(0,0) ===")
print(f"Reader position: {reader.position}")

# Read mb_type for first MB (should be I_4x4 or I_16x16 for I-slice)
pos_before = reader.position
mb_type_ue = reader.read_ue()
print(f"  MB(0,0) mb_type: ue={mb_type_ue} (consumed {reader.position - pos_before} bits)")

# Continue until failure or success
print(f"\nContinuing decode...")
