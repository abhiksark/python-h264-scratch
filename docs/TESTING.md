# h264/docs/TESTING.md
# Testing Strategy

## Overview
Every component is tested against JM reference software to ensure bit-exact compliance.

## JM Reference Software Setup

### Installation
```bash
# Clone JM reference
git clone https://github.com/shihuade/JM.git ~/JM
cd ~/JM

# Build (Linux/macOS)
make

# Binaries are in bin/
ls bin/
# lencod  ldecod  rtpdump  rtp_loss
```

### Generate Test Bitstreams

#### Minimal Test (64x64, 1 I-frame)
Create `encoder_test.cfg`:
```ini
# Basic settings
InputFile = "test_input.yuv"
OutputFile = "test.264"
ReconFile = "test_rec.yuv"
FramesToBeEncoded = 1
SourceWidth = 64
SourceHeight = 64
FrameRate = 30

# Baseline profile
ProfileIDC = 66
LevelIDC = 10

# All I-frames
IntraPeriod = 1
IDRPeriod = 1

# CAVLC (no CABAC)
SymbolMode = 0

# QP
QPISlice = 26
```

Generate input YUV (solid gray):
```bash
# Create 64x64 gray frame (Y=128, Cb=Cr=128)
python3 -c "
import numpy as np
y = np.full((64, 64), 128, dtype=np.uint8)
uv = np.full((32, 32), 128, dtype=np.uint8)
with open('test_input.yuv', 'wb') as f:
    f.write(y.tobytes())
    f.write(uv.tobytes())
    f.write(uv.tobytes())
"
```

Encode:
```bash
cd ~/JM
./bin/lencod -d encoder_test.cfg
```

### Decode with Trace Output
Create `decoder_test.cfg`:
```ini
InputFile = "test.264"
OutputFile = "test_dec.yuv"
RefFile = "test_rec.yuv"

# Enable trace (dumps all intermediate values)
Trace = 1
```

Decode:
```bash
./bin/ldecod -d decoder_test.cfg
# Creates trace_dec.txt with detailed output
```

## Test Structure

### Per-Module Tests
Each module has `tests/` folder:
```
color/
├── __init__.py
├── yuv_to_rgb.py
└── tests/
    ├── __init__.py
    ├── test_yuv_to_rgb.py
    └── fixtures/
        ├── input_yuv.npy
        └── expected_rgb.npy
```

### Test Categories

#### 1. Unit Tests
Test single functions with known inputs:
```python
def test_yuv_to_rgb_black():
    """Y=0, Cb=Cr=128 should give RGB black."""
    y = np.zeros((2, 2), dtype=np.uint8)
    cb = np.full((1, 1), 128, dtype=np.uint8)
    cr = np.full((1, 1), 128, dtype=np.uint8)

    rgb = yuv_to_rgb(y, cb, cr)

    assert np.allclose(rgb, 0, atol=1)
```

#### 2. Reference Tests
Compare with JM output:
```python
def test_dequant_matches_jm():
    """Dequantization must match JM reference exactly."""
    # Load JM trace values
    jm_coeffs = np.load("fixtures/jm_coeffs_qp26.npy")
    jm_dequant = np.load("fixtures/jm_dequant_qp26.npy")

    # Our implementation
    result = dequant_4x4(jm_coeffs, qp=26)

    np.testing.assert_array_equal(result, jm_dequant)
```

#### 3. Edge Case Tests
Test boundary conditions:
```python
def test_intra_16x16_dc_no_neighbors():
    """DC prediction with no available neighbors uses 128."""
    # Top-left macroblock has no neighbors
    pred = intra_16x16_dc(
        top_available=False,
        left_available=False,
        top_pixels=None,
        left_pixels=None
    )

    assert np.all(pred == 128)
```

## Running Tests

### All Tests
```bash
cd /home/abhik/Projects/personal/h264
pytest -v
```

### Specific Module
```bash
pytest color/tests/ -v
pytest transform/tests/ -v
```

### With Debug Logging
```bash
pytest -v --log-cli-level=DEBUG
```

### Coverage Report
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## Generating Test Fixtures

### From JM Trace
Parse `trace_dec.txt` to extract values:
```python
# scripts/parse_jm_trace.py
def extract_dequant_values(trace_path, mb_index, block_index):
    """Extract dequantized coefficients from JM trace."""
    with open(trace_path) as f:
        # Parse trace format...
        pass
    return coefficients

# Generate fixture
coeffs = extract_dequant_values("trace_dec.txt", mb=0, block=0)
np.save("fixtures/jm_dequant_mb0_block0.npy", coeffs)
```

### Synthetic Test Data
```python
# Known transform pair
input_coeffs = np.array([
    [16, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=np.int32)

# DC-only input should give flat output
expected_output = np.full((4, 4), 16, dtype=np.int32)

np.save("fixtures/idct_dc_only_input.npy", input_coeffs)
np.save("fixtures/idct_dc_only_expected.npy", expected_output)
```

## Continuous Validation

After any change:
1. Run `pytest -v`
2. Check all reference tests pass
3. Verify no regressions when replacing libs with NumPy

## Debugging Failed Tests

### Visual Diff
```python
def test_with_visual_diff():
    expected = np.load("expected.npy")
    actual = our_function(...)

    if not np.array_equal(expected, actual):
        diff = expected.astype(int) - actual.astype(int)
        print(f"Max diff: {np.abs(diff).max()}")
        print(f"Diff locations: {np.argwhere(diff != 0)}")

        # Save for visual inspection
        Image.fromarray(expected).save("expected.png")
        Image.fromarray(actual).save("actual.png")

    np.testing.assert_array_equal(expected, actual)
```
