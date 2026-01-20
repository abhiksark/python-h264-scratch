# H.264 Baseline Decoder

An educational H.264 Baseline profile decoder implemented in Python/NumPy.

## Overview

This project implements an H.264 video decoder from scratch for educational purposes. The goal is to understand video compression by building each component of the decoding pipeline.

**Scope:**
- Decoder only (no encoder)
- Baseline profile (CAVLC entropy coding)
- I-frames fully supported, P-frames planned
- Input: Raw Annex B bitstream (`.264`/`.h264` files)
- Output: NumPy arrays (YUV 4:2:0 and RGB)

## Project Structure

```
h264/
├── bitstream/          # NAL unit parsing, bit-level reading
├── parameters/         # SPS/PPS parsing
├── slice/              # Slice header parsing
├── entropy/            # CAVLC entropy decoding
├── dequant/            # Inverse quantization
├── transform/          # 4x4 IDCT, Hadamard transforms
├── intra/              # Intra prediction (16x16, 4x4)
├── inter/              # Inter prediction (planned)
├── reconstruct/        # Macroblock reconstruction
├── color/              # YCbCr to RGB conversion
├── decoder/            # Main decoder orchestration
├── test_utils/         # Testing utilities
├── test_data/          # Test bitstreams (not tracked)
└── docs/               # Documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/h264.git
cd h264

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from decoder import decode_h264_bytes

# Decode H.264 bitstream
with open("video.264", "rb") as f:
    bitstream = f.read()

frames = decode_h264_bytes(bitstream)

# Access decoded frame
frame = frames[0]
print(f"Frame size: {frame.width}x{frame.height}")

# Get YUV planes
y, cb, cr = frame.luma, frame.cb, frame.cr

# Convert to RGB
rgb = frame.to_rgb()
```

## Testing

```bash
# Run all tests
pytest -v

# Run specific module tests
pytest bitstream/tests/ -v
pytest decoder/tests/ -v
```

### Test Data

Test bitstreams are not included in the repository. Generate them using the JM reference software:

```bash
# Clone and build JM reference software
git clone https://github.com/shihuade/JM.git ~/JM
cd ~/JM && make

# Generate test bitstream
./bin/lencod -d encoder_baseline.cfg

# Decode reference output
./bin/ldecod -d decoder.cfg
```

See `docs/TESTING.md` for detailed test data generation instructions.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - Design decisions and module descriptions
- [Progress](docs/PROGRESS.md) - Implementation status by module
- [Spec Mapping](docs/SPEC_MAPPING.md) - H.264 specification section references
- [Testing](docs/TESTING.md) - Test strategy and JM usage

## H.264 Decoding Pipeline

```
Annex B Bitstream
       │
       ▼
┌─────────────┐
│  bitstream  │  NAL unit extraction
└─────────────┘
       │
       ▼
┌─────────────┐
│ parameters  │  SPS/PPS parsing
└─────────────┘
       │
       ▼
┌─────────────┐
│    slice    │  Slice header
└─────────────┘
       │
       ▼
┌─────────────┐
│   entropy   │  CAVLC → coefficients
└─────────────┘
       │
       ▼
┌─────────────┐
│   dequant   │  Inverse quantization
└─────────────┘
       │
       ▼
┌─────────────┐
│  transform  │  4x4 IDCT
└─────────────┘
       │
       ▼
┌─────────────┐
│    intra    │  Prediction
└─────────────┘
       │
       ▼
┌─────────────┐
│ reconstruct │  pred + residual
└─────────────┘
       │
       ▼
┌─────────────┐
│    color    │  YCbCr → RGB
└─────────────┘
       │
       ▼
   NumPy Array
```

## Dependencies

- `numpy` - Array operations and transforms
- `bitstring` - Bit-level parsing (scaffolding, to be replaced)
- `pytest` - Testing framework
- `pillow` - Image visualization (optional)

## References

- [ITU-T H.264](https://www.itu.int/rec/T-REC-H.264) - Official specification
- [JM Reference Software](https://github.com/shihuade/JM) - Reference encoder/decoder

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
