# Pixel-Perfect I_8x8 & High Profile Expansion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve pixel-perfect I_8x8 decode on real-world High Profile video (Big Buck Bunny 640x360), then expand High Profile support to P/B-frames.

**Architecture:** Single-MB trace-and-fix methodology. Add JM trace points to dump per-MB dequant/IDCT/prediction values, build a Python debug harness that compares our pipeline stage-by-stage against JM, fix each discrepancy found. Repeat until BBB frame 1 matches ffmpeg exactly. Then wire 8x8 transform into inter frames.

**Tech Stack:** Python/NumPy decoder, JM 19.0 reference decoder (C), ffmpeg for reference YUV generation.

---

## Phase 1: Pixel-Perfect I_8x8

### Task 1: Build JM per-MB trace dumper

**Files:**
- Modify: `/home/abhik/JM/ldecod/src/read_comp_cabac.c:376` (dequant trace)
- Modify: `/home/abhik/JM/lcommon/src/transform.c:525-532` (IDCT output trace)
- Modify: `/home/abhik/JM/ldecod/src/macroblock.c` (prediction trace — find I_8x8 reconstruct)

**Step 1: Add dequant dump to JM**

In `read_comp_cabac.c`, after the dequant loop for 8x8 blocks (around line 400, after all 64 coefficients are dequantized), add:

```c
// After the coefficient loop, dump dequant output for target MB
if (currMB->mbAddrX == TARGET_MB) {
    fprintf(stderr, "JM_DEQUANT mb=%d blk=%d\n", currMB->mbAddrX, b8);
    for (int dj = 0; dj < 8; dj++) {
        for (int di = 0; di < 8; di++)
            fprintf(stderr, "%d ", tcoeffs[dj][boff_x + di]);
        fprintf(stderr, "\n");
    }
}
```

Where `TARGET_MB` is a `#define` at the top of the file (start with the worst MB index).

**Step 2: Add IDCT output dump to JM**

In `transform.c`, at the end of `inverse8x8()` (after the vertical pass, before writing to output), add:

```c
// After vertical pass normalization
if (joff/16 * mb_width + ioff/16 == TARGET_MB) { // approximate
    fprintf(stderr, "JM_IDCT\n");
    // dump the 8x8 output block
}
```

Note: The exact insertion point needs to be found by reading the JM reconstruction flow. The IDCT output is the residual that gets added to prediction.

**Step 3: Rebuild JM**

Run: `cd ~/JM && make -j4 2>&1 | tail -5`
Expected: Successful build with ldecod.exe

**Step 4: Generate JM trace for target MB**

```bash
cd ~/JM/bin
./ldecod.exe -i ~/Projects/personal/h264/test_data/bbb_frame1.264 \
  -o bbb_jm_dec.yuv 2>bbb_jm_trace.txt
grep -A9 "JM_DEQUANT" bbb_jm_trace.txt | head -40
```

**Step 5: Commit JM trace changes**

```bash
cd ~/JM && git add -A && git commit -m "debug: add I_8x8 per-MB trace for dequant/IDCT"
```

---

### Task 2: Build Python per-MB trace harness

**Files:**
- Create: `debug_i8x8_bbb.py`

**Step 1: Write the debug script**

```python
# debug_i8x8_bbb.py
"""Trace I_8x8 pipeline for a specific MB in BBB and compare with JM."""
import sys, numpy as np
sys.path.insert(0, '.')

TARGET_MB_X = 38  # Worst MB
TARGET_MB_Y = 11

import decoder.decoder as dec
from dequant.dequant import dequant_8x8
from transform.idct_8x8 import idct_8x8
from parameters.scaling import get_scaling_list_8x8
from entropy.tables import ZIGZAG_8x8 as ZZ8

# Monkey-patch to capture data at target MB
captured_blocks = []

_orig_process = dec.H264Decoder._process_cabac_macroblock
def _trace_process(self, cabac, contexts, mb_data, mb_x, mb_y, qp, sps, pps, slice_type):
    if mb_x == TARGET_MB_X and mb_y == TARGET_MB_Y:
        print(f"=== TARGET MB({mb_x},{mb_y}) ===")
        print(f"  mb_type={mb_data.get('mb_type')}")
        print(f"  t8x8={mb_data.get('transform_size_8x8_flag')}")
        print(f"  cbp=0x{mb_data.get('cbp', 0):04x}")
        print(f"  qp={qp}")

        residual = mb_data.get('residual', {})
        luma_8x8 = residual.get('luma_8x8', [None]*4)
        cbp = mb_data.get('cbp', 0)
        cbp_luma = cbp & 0x0F

        scaling_list = get_scaling_list_8x8(sps, None, index=0)

        for blk in range(4):
            raw = luma_8x8[blk] if luma_8x8 else None
            coeffs = np.zeros((8, 8), dtype=np.int32)
            if raw is not None and (cbp_luma & (1 << blk)):
                raw = np.asarray(raw, dtype=np.int32)
                for s in range(64):
                    r = int(ZZ8[s])
                    coeffs[r // 8, r % 8] = raw[s]

            dq = dequant_8x8(coeffs, qp, scaling_list)
            res = idct_8x8(dq)

            print(f"\n  Block {blk} coeffs (raster):")
            for row in coeffs:
                print(f"    {list(row)}")
            print(f"  Block {blk} dequant:")
            for row in dq:
                print(f"    {list(row)}")
            print(f"  Block {blk} IDCT residual:")
            for row in res:
                print(f"    {list(row)}")

            captured_blocks.append({
                'blk': blk, 'coeffs': coeffs.copy(),
                'dequant': dq.copy(), 'residual': res.copy()
            })

    return _orig_process(self, cabac, contexts, mb_data, mb_x, mb_y, qp, sps, pps, slice_type)

dec.H264Decoder._process_cabac_macroblock = _trace_process

# Decode
d = dec.H264Decoder(deblocking_enabled=False)
frame = next(d.decode_file('test_data/bbb_360_10s.mp4'))
w, h = frame.width, frame.height

# Load reference
with open('test_data/bbb_frame1_ref.yuv', 'rb') as f:
    ref_data = f.read()
ref_y = np.frombuffer(ref_data[:w*h], dtype=np.uint8).reshape(h, w)

# Compare target MB pixels
y0 = TARGET_MB_Y * 16
x0 = TARGET_MB_X * 16
our = frame.luma[y0:y0+16, x0:x0+16].astype(np.int32)
ref = ref_y[y0:y0+16, x0:x0+16].astype(np.int32)
diff = ref - our

print(f"\n=== PIXEL COMPARISON MB({TARGET_MB_X},{TARGET_MB_Y}) ===")
print(f"Max abs diff: {np.max(np.abs(diff))}")
for blk in range(4):
    by = (blk // 2) * 8
    bx = (blk % 2) * 8
    bd = diff[by:by+8, bx:bx+8]
    print(f"  Block {blk}: max={np.max(np.abs(bd))}")
    if np.max(np.abs(bd)) > 0:
        print(f"    Diff:")
        for row in bd:
            print(f"      {list(row)}")
```

**Step 2: Run and capture output**

Run: `python3 debug_i8x8_bbb.py 2>&1 | grep -v "^DEBUG\|^WARNING\|^INFO" > debug_bbb_mb38_11.txt`
Expected: Full pipeline dump for MB(38,11) with pixel diffs

**Step 3: Commit debug script**

```bash
git add debug_i8x8_bbb.py
git commit -m "debug: add BBB I_8x8 per-MB trace harness"
```

---

### Task 3: Compare dequant stage against JM

**Files:**
- Read: `debug_bbb_mb38_11.txt` (our trace)
- Read: `~/JM/bin/bbb_jm_trace.txt` (JM trace)

**Step 1: Compare dequant values block-by-block**

Parse both trace files, compare each 8x8 dequant matrix element-by-element. If they match, dequant is correct and the bug is downstream.

**Step 2: If dequant differs — fix `dequant/dequant.py:344-411`**

Likely causes:
- Rounding difference in `rshift_rnd_sf` vs Python `>>` for negative numbers
  - Python `>>` rounds toward -∞, C `>>` is implementation-defined (usually also toward -∞)
  - But `rshift_rnd_sf` adds rounding: `(x + (1 << (a-1))) >> a`
  - Our formula at line 399: `(coeffs * level_scale + rounding) >> (6 - qp_div_6)`
  - Must verify rounding matches JM for negative coefficients

**Step 3: Run full decode and check improvement**

```bash
python3 -c "
import sys, numpy as np; sys.path.insert(0, '.')
import decoder.decoder as dec
d = dec.H264Decoder(deblocking_enabled=False)
frame = next(d.decode_file('test_data/bbb_360_10s.mp4'))
w, h = frame.width, frame.height
with open('test_data/bbb_frame1_ref.yuv', 'rb') as f: ref = f.read()
ref_y = np.frombuffer(ref[:w*h], dtype=np.uint8).reshape(h, w)
diff = np.abs(ref_y.astype(np.int32) - frame.luma.astype(np.int32))
print(f'max_diff={diff.max()}, nonzero={np.count_nonzero(diff)}')
"
```

**Step 4: Commit any fix**

---

### Task 4: Compare IDCT stage against JM

**Step 1: If dequant matches, compare IDCT output**

The IDCT takes dequant values and produces residuals. Compare our `idct_8x8()` output against JM's `inverse8x8()` output for the same input.

**Step 2: If IDCT differs — fix `transform/idct_8x8.py:80-130`**

Known risk: the even-part formula `a2 = p6 - (p2 >> 1)` vs `(p6 >> 1) - p2`. Current code should match JM line 506: `a2 = p6 - (p2 >> 1)`. Verify the second pass (vertical, lines 535-579 in JM) uses the same butterfly.

Also verify the final normalization: `(val + 32) >> 6` matches our `idct_8x8()` rounding.

**Step 3: Write a standalone IDCT verification test**

```python
def test_idct_8x8_matches_jm_for_bbb_mb():
    """Verify IDCT output matches JM for real-world coefficients."""
    # Use captured dequant values from Task 3
    dequant_input = np.array([...])  # paste from trace
    expected_residual = np.array([...])  # paste from JM trace
    result = idct_8x8(dequant_input)
    np.testing.assert_array_equal(result, expected_residual)
```

**Step 4: Fix and commit**

---

### Task 5: Compare prediction stage against JM

**Step 1: If dequant+IDCT both match, the bug is in prediction**

Add prediction trace to the debug script — dump filtered reference samples and prediction output for each 8x8 block. Compare with JM.

**Step 2: Check these known risk areas in `decoder/i8x8.py:181-251`:**

1. **Reference sample substitution** (lines 221-232): When top is unavailable and left is available, we substitute `top = left[0]`. Check if this applies correctly for ALL blocks (not just block 0). Block 2 (bottom-left) has top available from block 0, so substitution shouldn't trigger.

2. **Lowpass filter** (`intra/intra_8x8.py:434-503`): Verify the filter handles top_right unavailability correctly for blocks 1 and 3 (right-side blocks where top_right may come from the next MB).

3. **Top-right availability** (`decoder/i8x8.py:324-331`): Block 3 (bottom-right 8x8) should NOT have top_right available (it's the diagonal neighbor, per spec). Check our logic.

**Step 3: Fix and commit**

---

### Task 6: Iterate until pixel-perfect

**Step 1: After fixing the first bad MB, re-run full decode comparison**

```bash
python3 -c "..." # same comparison script from Task 3 Step 3
```

**Step 2: If max_diff > 0, pick next worst MB, update TARGET_MB in debug script, repeat Tasks 3-5**

Each iteration should reduce the error count. Common pattern: one root-cause fix eliminates errors in many MBs simultaneously (like the HU fix eliminated all mode-8 errors at once).

**Step 3: When max_diff=0 on BBB frame 1, write a regression test**

```python
# decoder/tests/test_high_profile.py
def test_bbb_frame1_pixel_perfect():
    """BBB 640x360 High Profile I-frame: pixel-perfect luma."""
    d = H264Decoder(deblocking_enabled=False)
    frame = next(d.decode_file('test_data/bbb_360_10s.mp4'))
    ref_y = load_reference('test_data/bbb_frame1_ref.yuv', frame.width, frame.height)
    np.testing.assert_array_equal(frame.luma, ref_y)
```

**Step 4: Run full test suite, commit**

```bash
pytest -x -q
git add -A && git commit -m "feat: pixel-perfect High Profile I_8x8 on real-world video"
```

---

## Phase 2: High Profile P/B-Frame Expansion

### Task 7: Wire 8x8 transform into inter residual decode

**Files:**
- Modify: `decoder/decoder.py:3340-3350` (CABAC inter residual path)
- Test: `decoder/tests/test_high_profile.py`

**Step 1: Check current inter 8x8 residual path**

Read `decoder/decoder.py` around line 3340 where `_reconstruct_inter_mb_cabac()` handles residual. Verify it already calls `dequant_8x8` + `idct_8x8` when `transform_size_8x8_flag=1`. If not, wire it in following the same pattern as the intra path (lines 2958-2971).

**Step 2: Generate High Profile P-frame test stream**

```bash
ffmpeg -f lavfi -i "testsrc2=duration=0.5:size=176x144:rate=5" \
  -c:v libx264 -profile:v high -preset slow \
  -x264-params "bframes=0:ref=1:8x8dct=1" \
  -pix_fmt yuv420p test_data/high_profile_p_176x144.264

ffmpeg -skip_loop_filter all -i test_data/high_profile_p_176x144.264 \
  -f rawvideo -pix_fmt yuv420p test_data/reference/high_profile_p_176x144_nodeblock_ref.yuv
```

**Step 3: Write test and verify**

```python
def test_high_profile_p_frame():
    d = H264Decoder(deblocking_enabled=False)
    frames = list(d.decode_bytes(open('test_data/high_profile_p_176x144.264', 'rb').read()))
    # Compare each frame against reference
```

**Step 4: Commit**

---

### Task 8: Test High Profile B-frames

**Step 1: Generate High Profile B-frame test stream**

```bash
ffmpeg -f lavfi -i "testsrc2=duration=0.5:size=176x144:rate=10" \
  -c:v libx264 -profile:v high -preset slow \
  -x264-params "bframes=2:ref=1:8x8dct=1:b-adapt=0" \
  -pix_fmt yuv420p test_data/high_profile_b_176x144.264

ffmpeg -skip_loop_filter all -i test_data/high_profile_b_176x144.264 \
  -f rawvideo -pix_fmt yuv420p test_data/reference/high_profile_b_176x144_nodeblock_ref.yuv
```

**Step 2: Decode, compare, fix any inter-specific 8x8 bugs**

**Step 3: Write regression test and commit**

---

### Task 9: Multi-frame MP4 decode test

**Files:**
- Test: `decoder/tests/test_high_profile.py`

**Step 1: Write end-to-end MP4 decode test**

```python
def test_bbb_mp4_first_5_frames():
    """Decode first 5 frames from real MP4 without crashing."""
    d = H264Decoder(deblocking_enabled=False)
    frames = []
    for frame in d.decode_file('test_data/bbb_360_10s.mp4'):
        frames.append(frame)
        assert frame.width == 640
        assert frame.height == 360
        if len(frames) >= 5:
            break
    assert len(frames) == 5
```

**Step 2: Run and commit**

---

## Verification Checklist

After all tasks complete:
- [ ] BBB frame 1 luma: max_diff=0 vs ffmpeg (no deblock)
- [ ] BBB frame 1 chroma: max_diff=0 vs ffmpeg (no deblock)
- [ ] Small test stream (176x144): still pixel-perfect
- [ ] All existing tests pass (1843+)
- [ ] High Profile P-frame test stream passes
- [ ] High Profile B-frame test stream passes
- [ ] MP4 multi-frame decode doesn't crash
