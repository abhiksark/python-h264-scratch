# Multi-Frame B-Frame Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix remaining multi-frame B-frame decode issues: bframes >= 3 B-pyramid errors, ref > 2 reordering, and CAVLC total_zeros edge case.

**Architecture:** Three independent bugs traced to root causes. Fix each with targeted changes and regression tests.

**Tech Stack:** Python/NumPy decoder, ffmpeg for reference generation, pytest for verification.

---

## Bug Summary

| Bug | Root Cause | Fix Location |
|-----|-----------|-------------|
| bframes >= 3 B-pyramid | B_Skip MV cache trace shows MV=(16,0) but derive returns (0,0). Need to verify if the MV cache read in the previous trace was checking the wrong sub-block position. | `inter/b_reconstruct.py`, `inter/direct_mode.py` |
| ref > 2 reordering | `build_p_slice_ref_list` shift-insert-compact loop doesn't handle 3+ modifications with wrap-around correctly | `inter/reference.py:256-304` |
| CAVLC total_zeros | Wrong nC for chroma AC blocks | `decoder/decoder.py` or `reconstruct/macroblock.py` chroma AC nC derivation |

---

### Task 1: Generate Reproducible Test Streams

**Files:**
- Create: `test_data/` (temporary, gitignored)

- [ ] **Step 1: Generate bframes=3 B-pyramid test stream**

```bash
ffmpeg -y -f lavfi -i "testsrc2=duration=0.5:size=176x144:rate=10" \
  -c:v libx264 -profile:v main -preset medium \
  -x264-params "bframes=3:ref=1:b-adapt=0:b-pyramid=normal" \
  -pix_fmt yuv420p test_data/_b3_test.264

ffmpeg -y -skip_loop_filter all -i test_data/_b3_test.264 \
  -f rawvideo -pix_fmt yuv420p test_data/_b3_test_ref.yuv
```

- [ ] **Step 2: Generate ref=3 P-only test stream**

```bash
ffmpeg -y -f lavfi -i "testsrc2=duration=0.5:size=176x144:rate=10" \
  -c:v libx264 -profile:v main -preset slow \
  -x264-params "bframes=0:ref=3" \
  -pix_fmt yuv420p test_data/_ref3_test.264

ffmpeg -y -skip_loop_filter all -i test_data/_ref3_test.264 \
  -f rawvideo -pix_fmt yuv420p test_data/_ref3_test_ref.yuv
```

- [ ] **Step 3: Verify streams reproduce the bugs**

```bash
python3 -c "
import sys, numpy as np; sys.path.insert(0, '.')
import decoder.decoder as dec

for name in ['_b3_test', '_ref3_test']:
    d = dec.H264Decoder(deblocking_enabled=False)
    with open(f'test_data/{name}.264', 'rb') as f: bs = f.read()
    with open(f'test_data/{name}_ref.yuv', 'rb') as f: ref = f.read()
    w, h, fs = 176, 144, 176*144*3//2
    frames = list(d.decode_bytes(bs))
    worst = 0
    for f in frames:
        for j in range(len(ref)//fs):
            ry = np.frombuffer(ref[j*fs:j*fs+w*h], dtype=np.uint8).reshape(h,w)
            dy = int(np.max(np.abs(ry.astype(int)-f.luma.astype(int))))
            worst = max(worst, min(dy, worst+1) if worst < dy else worst)
        worst = max(worst, min(dy for j2 in range(len(ref)//fs) for dy in [int(np.max(np.abs(np.frombuffer(ref[j2*fs:j2*fs+w*h],dtype=np.uint8).reshape(h,w).astype(int)-f.luma.astype(int))))]))
    print(f'{name}: worst={worst}')
"
```

Expected: `_b3_test worst > 0`, `_ref3_test worst > 0` (bugs reproduced).

---

### Task 2: Fix bframes >= 3 B-pyramid MV Issue

**Files:**
- Modify: `inter/b_reconstruct.py:442-555`
- Modify: `inter/direct_mode.py` (if needed)

- [ ] **Step 1: Add trace to verify the actual MV values during reconstruction**

Add a temporary trace inside `reconstruct_b_direct_16x16` at line 486 (after `derive_direct_spatial_sub8x8` returns) to print the derived MVs AND the prediction pixel values. Compare with the expected output from ffmpeg reference.

The key question is: does `derive_direct_spatial_sub8x8` return the correct MVs? If yes, check if `_get_sub_prediction` fetches the correct pixels from the reference frame. If the MVs are (0,0) and the pixel fetch is wrong, the bug is in the reference frame lookup (wrong L0/L1 frame selection).

```python
# Temporary debug in reconstruct_b_direct_16x16, after line 489:
import logging
_log = logging.getLogger(__name__)
_log.debug(f"B_Direct sub{sub_idx}: MV=({mvx_l0},{mvy_l0})/({mvx_l1},{mvy_l1}) pf={pf0}/{pf1} ri={ri0}/{ri1}")
```

- [ ] **Step 2: Run with DEBUG logging on the bframes=3 stream**

```bash
python3 -c "
import sys, logging; sys.path.insert(0, '.')
logging.basicConfig(level=logging.DEBUG)
import decoder.decoder as dec
d = dec.H264Decoder(deblocking_enabled=False)
with open('test_data/_b3_test.264', 'rb') as f: bs = f.read()
for f in d.decode_bytes(bs): pass
" 2>&1 | grep "B_Direct sub" | head -20
```

Analyze: are the MVs correct? If MVs are (0,0) for all sub-blocks but the output is wrong, the issue is in the reference frame content or the pixel fetch coordinates.

- [ ] **Step 3: Compare reference frame pixels directly**

If MVs are correct but output is wrong, verify that the reference frame at L1[0] (POC=4) contains the expected pixels at the MB position. Compare `ref_buffer.get_l1_frame(0).luma[y:y+8, x:x+8]` with ffmpeg's POC=4 output.

- [ ] **Step 4: Apply fix based on findings**

Based on the trace, one of these will be the fix:
1. **Wrong reference frame**: Fix L0/L1 frame selection in `_get_sub_prediction`
2. **Wrong MV derivation**: Fix `derive_direct_spatial_sub8x8` for the 3-frame DPB case
3. **Wrong pixel coordinates**: Fix the pixel position calculation in `_get_sub_prediction`

- [ ] **Step 5: Verify fix**

```bash
python3 -c "
import sys, numpy as np; sys.path.insert(0, '.')
import decoder.decoder as dec
# Test bframes=2 (must stay pixel-perfect)
# Test bframes=3 (must improve to pixel-perfect or near)
# Test bframes=4 (should also improve)
"
```

- [ ] **Step 6: Commit**

```bash
git add inter/b_reconstruct.py inter/direct_mode.py
git commit -m "fix(b-frame): correct B_Direct spatial MV for 3+ frame DPB"
```

---

### Task 3: Fix ref > 2 Reordering

**Files:**
- Modify: `inter/reference.py:256-304`

- [ ] **Step 1: Trace the shift-insert-compact loop for ref=3**

Add a detailed trace inside `build_p_slice_ref_list` to log each modification step: the pic_num computed, the target frame found, and the ref_list state after each step.

- [ ] **Step 2: Compare against JM's ref_pic_list_reordering**

Read JM's implementation in `ldecod/src/mbuffer.c` (function `reorder_short_term` or similar). Compare the loop with ours step by step.

Key things to check:
- Is `pic_num_no_wrap` maintained correctly across iterations?
- Does the compact step correctly handle the case where the same frame appears multiple times?
- Is the final `_l0_list[:] = ref_list[:num_active]` truncation correct?

- [ ] **Step 3: Apply fix**

Based on JM comparison, fix the reorder algorithm. Common issues:
- The `pic_num_no_wrap` predictor should be maintained across modifications (it currently is, line 261)
- The compact step should compare by `frame_num`, not by identity (`is not target`)
- The `num_active` should match the slice header count

- [ ] **Step 4: Verify**

```bash
python3 -c "
# Test ref=1 (must stay pixel-perfect)
# Test ref=2 (should improve from diff=2 to diff=0)
# Test ref=3 (should improve from diff=52)
"
```

- [ ] **Step 5: Commit**

```bash
git add inter/reference.py
git commit -m "fix(reference): correct ref_pic_list_modification for ref > 2"
```

---

### Task 4: Fix CAVLC total_zeros on Ultrafast

**Files:**
- Modify: `reconstruct/macroblock.py` or `decoder/decoder.py` (chroma AC nC derivation)

- [ ] **Step 1: Trace the failing chroma AC block**

Generate the ultrafast stream and trace which MB and which chroma AC block triggers the `total_zeros=9 exceeds max_valid=8` warning. Capture the nC value used for the coeff_token lookup.

- [ ] **Step 2: Verify nC computation**

For chroma AC blocks, nC should come from neighboring chroma blocks' nz_counts. Check if:
- The chroma nz_count array is indexed correctly (4:2:0 has 8 chroma blocks, 4 Cb + 4 Cr)
- The left/top neighbor lookups use the correct block indices
- The nC = (nA + nB + 1) >> 1 formula handles unavailable neighbors correctly

- [ ] **Step 3: Apply fix**

- [ ] **Step 4: Verify**

```bash
python3 -c "
# Test baseline ultrafast (should not crash)
# Test high ultrafast (should not crash)
# Test medium preset (must stay pixel-perfect)
"
```

- [ ] **Step 5: Commit**

```bash
git commit -m "fix(cavlc): correct chroma AC nC for ultrafast preset"
```

---

### Task 5: Write Multi-Frame Regression Tests

**Files:**
- Modify: `decoder/tests/test_high_profile.py`

- [ ] **Step 1: Add multi-frame P/B decode tests**

```python
def test_bbb_multiframe_no_crash():
    """BBB 360p: decode all 300 frames without crashing."""
    d = H264Decoder(deblocking_enabled=False)
    frames = []
    for frame in d.decode_file('test_data/bbb_360_10s.mp4'):
        frames.append(frame)
    assert len(frames) == 300

def test_bbb_i_p_frames_pixel_perfect():
    """BBB 360p: I and P frames pixel-perfect."""
    # Compare first I-frame and first P-frame against reference
    pass

@pytest.mark.parametrize("bframes", [2, 3])
def test_bframes_with_pyramid(bframes):
    """B-pyramid decodes without crashing for various bframe counts."""
    # Generate stream, decode, verify no crash
    pass
```

- [ ] **Step 2: Run tests and verify**

```bash
pytest decoder/tests/test_high_profile.py -v
```

- [ ] **Step 3: Commit**

```bash
git add decoder/tests/test_high_profile.py
git commit -m "test: add multi-frame B-frame regression tests"
```

---

### Task 6: Tag v0.2 and Update Docs

- [ ] **Step 1: Update PROGRESS.md with fix status**
- [ ] **Step 2: Run full test suite**
- [ ] **Step 3: Tag v0.2.0**

```bash
git tag -a v0.2.0 -m "v0.2.0 — Multi-frame decode fixes"
```

- [ ] **Step 4: Push**

```bash
git push origin main --tags
```
