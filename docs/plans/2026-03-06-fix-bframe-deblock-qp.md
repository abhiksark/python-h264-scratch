# Fix B-Frame Deblocking QP Bug Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the B_Skip QP storage bug that causes wrong deblocking filter thresholds, achieving pixel-perfect B-frame decode for all 8 test streams.

**Architecture:** B_Skip macroblocks store a hardcoded default QP=26 (`self.state.current_mb_qp`) instead of the actual running slice QP. This causes wrong alpha/beta thresholds in the deblocking filter at edges involving B_Skip MBs. The fix passes the actual `current_qp` from the B-slice decode loop into `_process_b_skip_run`.

**Tech Stack:** Python, NumPy, H.264 spec Section 8.7

---

## Root Cause Analysis

**Bug location:** `decoder/decoder.py:3335` — `_process_b_skip_run` stores `self.state.current_mb_qp` (initialized to 26 at line 190, never updated) as the QP for B_Skip MBs.

**Comparison:** P_Skip at line 979 correctly stores `current_qp` (local variable tracking running QP).

**Impact:** For testsrc2_bf (B-slice QP=28), B_Skip MBs get QP=26 instead of 28. This changes deblocking alpha threshold from 20 to 15, causing pixels with |p0-q0| between 16-20 to be incorrectly filtered. Errors compound across vertical+horizontal edges, producing max_diff=6.

**Proof:**
- `testsrc2_bf_nodeblock2` (deblocking disabled) = PERFECT → pre-deblock output is correct
- `testsrc2_bf_nodirect` (no B_Direct/B_Skip MBs) = PERFECT → deblocking is correct for non-skip MBs
- Errors are at MB boundaries (local coords 0/15) → deblocking filter is the source
- `testsrc_bf` (QP=29) is PERFECT despite same bug — pixel values at its edges don't hit the threshold gap

---

### Task 1: Fix B_Skip QP storage

**Files:**
- Modify: `decoder/decoder.py:3293-3339` (`_process_b_skip_run`)
- Modify: `decoder/decoder.py:2023-2034` (skip run call sites)
- Modify: `decoder/decoder.py:2005-2012` (implicit skip at end of data)
- Modify: `decoder/decoder.py:2039-2048` (implicit skip after no more data)

**Step 1: Add `current_qp` parameter to `_process_b_skip_run`**

In `decoder/decoder.py`, change the method signature at line 3293:

```python
def _process_b_skip_run(
    self,
    mb_x: int,
    mb_y: int,
    use_spatial: bool,
    current_poc: int,
    weighted_bipred_idc: int = 0,
    current_qp: int = 26,  # NEW: actual running QP
) -> None:
```

**Step 2: Use `current_qp` instead of `self.state.current_mb_qp`**

At line 3335, change:
```python
# Before (bug):
self.state.mb_qps[mb_idx_skip] = self.state.current_mb_qp
# After (fix):
self.state.mb_qps[mb_idx_skip] = current_qp
```

**Step 3: Pass `current_qp` at all call sites in `_decode_b_slice_data`**

There are 4 call sites in `_decode_b_slice_data` (around lines 2005-2048):

1. Line ~2008 (implicit skip before RBSP end):
```python
self._process_b_skip_run(
    mb_x, mb_y, use_spatial, current_poc,
    weighted_bipred_idc,
    current_qp=current_qp,
)
```

2. Line ~2030 (normal skip run):
```python
self._process_b_skip_run(
    mb_x, mb_y, use_spatial, current_poc,
    weighted_bipred_idc,
    current_qp=current_qp,
)
```

3. Line ~2043 (implicit skip after data ends):
```python
self._process_b_skip_run(
    mb_x, mb_y, use_spatial, current_poc,
    weighted_bipred_idc,
    current_qp=current_qp,
)
```

4. Search for any other call sites with:
```bash
grep -n "_process_b_skip_run" decoder/decoder.py
```

**Step 4: Run tests to verify no regressions**

Run: `pytest -x -q 2>&1 | tail -5`
Expected: All existing tests pass

**Step 5: Commit**

```bash
git add decoder/decoder.py
git commit -m "fix(deblock): use actual QP for B_Skip macroblocks

B_Skip MBs were storing self.state.current_mb_qp (always 26)
instead of the actual running slice QP. This caused wrong
alpha/beta thresholds in the deblocking filter."
```

---

### Task 2: Verify pixel-perfect B-frame decode

**Step 1: Run full B-frame accuracy comparison**

```python
python3 -c "
from decoder.decoder import H264Decoder
import numpy as np

streams = [
    ('testsrc2_bf', 'test_data/testsrc2_bf.264', 'test_data/testsrc2_bf_ref.yuv'),
    ('mandel_bf', 'test_data/mandel_bf.264', 'test_data/mandel_bf_ref.yuv'),
    ('testsrc2_bf_nodeblock', 'test_data/testsrc2_bf_nodeblock.264', 'test_data/testsrc2_bf_nodeblock_ref.yuv'),
    ('smpte_bf', 'test_data/smpte_bf.264', 'test_data/smpte_bf_ref.yuv'),
    ('testsrc_bf', 'test_data/testsrc_bf.264', 'test_data/testsrc_bf_ref.yuv'),
    ('gray_motion_bframes', 'test_data/gray_motion_bframes.264', 'test_data/gray_motion_bframes_ref.yuv'),
    ('testsrc2_bf_nodirect', 'test_data/testsrc2_bf_nodirect.264', 'test_data/testsrc2_bf_nodirect_ref.yuv'),
    ('testsrc2_bf_nodeblock2', 'test_data/testsrc2_bf_nodeblock2.264', 'test_data/testsrc2_bf_nodeblock2_ref.yuv'),
]
for name, h264, yuv in streams:
    dec = H264Decoder()
    frames = sorted(dec.decode_file(h264), key=lambda f: f.poc)
    with open(yuv, 'rb') as f:
        ref_data = f.read()
    w, h = frames[0].luma.shape[1], frames[0].luma.shape[0]
    frame_size = w * h * 3 // 2
    max_diff = 0
    for i, frame in enumerate(frames):
        offset = i * frame_size
        if offset + frame_size > len(ref_data):
            break
        ref_y = np.frombuffer(ref_data[offset:offset+w*h], dtype=np.uint8).reshape(h, w)
        d = int(np.abs(frame.luma.astype(np.int16) - ref_y.astype(np.int16)).max())
        max_diff = max(max_diff, d)
    print(f'{name}: {\"PERFECT\" if max_diff == 0 else f\"max_diff={max_diff}\"}')
"
```

Expected: ALL 8 streams show PERFECT

**Step 2: Run full test suite**

Run: `pytest -x -q 2>&1 | tail -5`
Expected: All tests pass

**Step 3: If any streams still have errors, proceed to Task 3. Otherwise skip to Task 4.**

---

### Task 3: (Conditional) Investigate remaining bS errors

Only execute this task if Task 2 shows non-zero diffs after the QP fix.

**Step 1: Identify which edges have wrong bS**

Add temporary diagnostic logging to `deblock_frame_inplace` in `deblock/deblock.py` to dump bS values at the specific error locations. Compare against ffmpeg's bS values using `ffmpeg -vf showinfo` or by implementing a reference bS calculator.

**Step 2: Check reference picture identity comparison**

In `_get_bs_for_edge`, the cross-match compares `ref_p == ref_l1_q` (raw L0 index vs raw L1 index). Per H.264 8.7.2.1, this should compare by picture identity (POC), not list index. If L0[0] and L1[0] point to different frames but both have index 0, the current code incorrectly considers them matching.

Fix: Pass reference frame POC information to `_get_bs_for_edge` and compare by POC instead of index for cross-list comparisons.

**Step 3: Test and commit**

---

### Task 4: Commit final result

**Step 1: Commit with verification results**

Include in commit message which streams are now pixel-perfect.

---

## Verification Checklist

| Stream | Before | Expected After |
|--------|--------|----------------|
| smpte_bf | PERFECT | PERFECT |
| testsrc_bf | PERFECT | PERFECT |
| gray_motion_bframes | PERFECT | PERFECT |
| testsrc2_bf_nodirect | PERFECT | PERFECT |
| testsrc2_bf_nodeblock2 | PERFECT | PERFECT |
| mandel_bf | max=2 | PERFECT |
| testsrc2_bf_nodeblock | max=2 | PERFECT |
| testsrc2_bf | max=6 | PERFECT |
