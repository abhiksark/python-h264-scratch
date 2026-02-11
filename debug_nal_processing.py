"""Debug which NAL units are being processed for slice decode."""
from decoder import H264Decoder
import decoder.decoder as decoder_module

# Patch _process_nal to log NAL info
original_process_nal = decoder_module.H264Decoder._process_nal

def logged_process_nal(self, nal):
    print(f"\n_process_nal: type={nal.nal_unit_type} ({nal.type_name}), "
          f"size={len(nal.rbsp)} bytes, start_pos={nal.start_position}")

    result = original_process_nal(self, nal)

    if result:
        print(f"  -> Returned frame")
    else:
        print(f"  -> No frame")

    return result

decoder_module.H264Decoder._process_nal = logged_process_nal

# Patch _decode_slice to log NAL and reader info
original_decode_slice = decoder_module.H264Decoder._decode_slice

def logged_decode_slice(self, nal):
    print(f"\n_decode_slice: NAL type={nal.nal_unit_type}, RBSP size={len(nal.rbsp)} bytes")
    print(f"  NAL.start_position={nal.start_position}")
    print(f"  First 10 bytes of RBSP: {nal.rbsp[:10].hex()}")

    try:
        result = original_decode_slice(self, nal)
        print(f"  -> Slice decode SUCCESS")
        return result
    except Exception as e:
        print(f"  -> Slice decode FAILED: {e}")
        raise

decoder_module.H264Decoder._decode_slice = logged_decode_slice

# Decode
decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_64x64.264'))
    print(f"\n✓ Success: {len(frames)} frames")
except Exception as e:
    print(f"\n✗ Failed: {e}")
