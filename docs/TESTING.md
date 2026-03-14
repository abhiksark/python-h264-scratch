# Testing Strategy

## Overview

1,850 tests verify every pipeline stage. Pixel-perfect accuracy is validated against ffmpeg on real internet videos.

## Running Tests

```bash
# Full suite
pytest -v

# Specific module
pytest decoder/tests/ -v
pytest entropy/tests/ -v

# High Profile pixel-perfect tests (requires test videos)
pytest decoder/tests/test_high_profile.py -v

# With debug logging
pytest -v --log-cli-level=DEBUG
```

## Test Data Setup

Binary files (`.264`, `.yuv`, `.mp4`) are not tracked in git.

### Download test videos

```bash
wget "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4" \
  -O test_data/bbb_360_10s.mp4
wget "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4" \
  -O test_data/bbb_720_10s.mp4
wget "https://test-videos.co.uk/vids/jellyfish/mp4/h264/360/Jellyfish_360_10s_1MB.mp4" \
  -O test_data/jellyfish_360_10s.mp4
```

### Generate ffmpeg reference output

```bash
# No-deblock references (for pixel-perfect comparison)
ffmpeg -skip_loop_filter all -i test_data/bbb_360_10s.mp4 \
  -vframes 1 -f rawvideo -pix_fmt yuv420p test_data/bbb_frame1_ref.yuv
ffmpeg -skip_loop_filter all -i test_data/jellyfish_360_10s.mp4 \
  -vframes 1 -f rawvideo -pix_fmt yuv420p test_data/jellyfish_360_10s_ref.yuv
ffmpeg -skip_loop_filter all -i test_data/bbb_720_10s.mp4 \
  -vframes 1 -f rawvideo -pix_fmt yuv420p test_data/bbb_720_10s_ref.yuv
```

### Generate raw H.264 test streams (with JM)

```bash
git clone https://github.com/shihuade/JM.git ~/JM
cd ~/JM && make

# Or use ffmpeg
ffmpeg -f lavfi -i "testsrc2=duration=0.2:size=176x144:rate=5" \
  -c:v libx264 -profile:v high -pix_fmt yuv420p test_data/high_profile_176x144.264
```

## Test Categories

### Pixel-perfect integration tests (`decoder/tests/test_high_profile.py`)

Compare our decode output against ffmpeg reference frame-by-frame:

```python
def test_mp4_frame1_pixel_perfect(video, ref, width, height):
    d = H264Decoder(deblocking_enabled=False)
    frame = next(d.decode_file(mp4_path))
    ref_y, ref_cb, ref_cr = load_yuv420(ref_path, w, h)
    np.testing.assert_array_equal(frame.luma, ref_y)
    np.testing.assert_array_equal(frame.cb, ref_cb)
    np.testing.assert_array_equal(frame.cr, ref_cr)
```

### Unit tests (per module)

Each module has its own `tests/` directory testing individual functions:

```
entropy/tests/test_cabac_arith.py    — arithmetic decoder
intra/tests/test_intra_8x8.py       — I_8x8 prediction modes
transform/tests/test_idct_8x8.py    — 8x8 IDCT butterfly
dequant/tests/test_dequant_8x8.py   — 8x8 dequantization
```

### TDD red tests

Tests marked `xfail` define requirements for features not yet implemented (interlaced, FMO, etc). They serve as a roadmap.

## Debugging Pixel Mismatches

When a test fails with pixel differences:

1. **Find the first bad MB**: compare per-macroblock to narrow down
2. **Check dequant**: compare against JM's `rshift_rnd_sf((level * IS) << qp_per, 6)` formula
3. **Check IDCT**: verify with JM's `inverse4x4` / `inverse8x8` (rows first, then columns)
4. **Check prediction**: compute `our_pixel - our_residual` vs `ref_pixel - our_residual`
5. **Check CABAC**: trace bin decisions and compare context states with JM

The JM reference decoder (`~/JM/bin/ldecod.exe`) can be instrumented with `fprintf(stderr, ...)` in `read_comp_cabac.c` and `block.c` to dump intermediate values.
