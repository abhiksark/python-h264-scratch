# Fix MB(1,0) Bit Drift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix bit drift bug in MB(1,0) which consumes 684 bits instead of expected ~330 bits, causing motion video decode failures.

**Architecture:** Add detailed checkpoints within MB decode to track bit consumption at each parsing step (mb_type, CBP, qp_delta, blocks). Identify which step consumes excess bits. Fix the specific parsing logic causing drift. Validate fix on all test videos.

**Tech Stack:** Python, H.264 Baseline profile, CAVLC entropy coding, NumPy

---

## Background

### Completed Work
- **VLC Audit (6 tasks)**: ✅ Proved all CAVLC VLC functions correct (coeff_token, total_zeros, run_before, levels)
- **MB Position Tracking**: ✅ Identified MB(1,0) as source of 60-bit drift

### Known Facts
- MB(1,0) in motion_64x64.264 consumes **684 bits** (expected ~330)
- Excess: 684 - 330 = **354 bits** (or 684 - 569 = **115 bits** more than MB(0,0))
- This drift causes MB(0,1) to start at position 1348 instead of 1288
- MB(0,0): 569 bits (normal decode)
- MB(2,0), MB(3,0): 48, 19 bits (likely skipped blocks)

### Hypothesis
MB(1,0) is reading data it shouldn't, or iterating through blocks incorrectly. Possible causes:
1. **Wrong mb_type interpretation** → decodes wrong mode
2. **Wrong CBP value** → decodes wrong number of blocks
3. **mb_qp_delta read when shouldn't** → reads extra se value
4. **Block iteration bug** → decodes extra blocks not indicated by CBP
5. **I_PCM mishandling** → reads PCM samples when shouldn't

---

## Task 1: Add Internal MB Checkpoints

**Goal:** Track bit position at each major step within MB decode

**Files:**
- Modify: `debug/mb_position_tracker.py` (add internal checkpoint support)
- Modify: `reconstruct/macroblock.py` (add checkpoints)

### Step 1: Add checkpoint method to tracker

Edit `debug/mb_position_tracker.py`:

```python
def checkpoint(self, reader: BitReader, label: str, **kwargs):
    """Record current bit position with label.

    Args:
        reader: BitReader instance
        label: Description of checkpoint
        **kwargs: Additional data to store
    """
    self.checkpoints.append({
        'mb': self.current_mb,
        'position': reader.position,
        'label': label,
        **kwargs
    })
```

### Step 2: Run tests to verify no errors

```bash
PYTHONPATH=. python debug/test_mb_tracker.py 2>&1 | grep "ERROR"
```

Expected: No errors (functionality unchanged)

### Step 3: Commit

```bash
git add debug/mb_position_tracker.py
git commit -m "feat(debug): support internal MB checkpoints"
```

---

## Task 2: Instrument MB Decode with Checkpoints

**Goal:** Add checkpoints at every major parsing step in decode_macroblock

**Files:**
- Modify: `reconstruct/macroblock.py`

### Step 1: Add checkpoint after mb_type decode

Find the line where `mb_type` is decoded (likely `mb_type_ue = reader.read_ue()`).

Add immediately after:

```python
if hasattr(reader, '_checkpoint_tracker'):
    reader._checkpoint_tracker.checkpoint(
        reader, 'mb_type',
        mb_type_ue=mb_type_ue,
        bits_since_start=reader.position - mb_start_pos
    )
```

### Step 2: Add checkpoint after CBP decode

Find where CBP is decoded. Add after:

```python
if hasattr(reader, '_checkpoint_tracker'):
    reader._checkpoint_tracker.checkpoint(
        reader, 'cbp',
        cbp=cbp,
        bits_since_start=reader.position - mb_start_pos
    )
```

### Step 3: Add checkpoint after mb_qp_delta

Find where mb_qp_delta is decoded. Add after:

```python
if hasattr(reader, '_checkpoint_tracker'):
    reader._checkpoint_tracker.checkpoint(
        reader, 'mb_qp_delta',
        qp_delta=mb_qp_delta,
        bits_since_start=reader.position - mb_start_pos
    )
```

### Step 4: Add checkpoint before and after residual decode

Before residual block loop:

```python
if hasattr(reader, '_checkpoint_tracker'):
    pos_before_residual = reader.position
    reader._checkpoint_tracker.checkpoint(
        reader, 'residual_start',
        bits_since_start=reader.position - mb_start_pos
    )
```

After residual block loop:

```python
if hasattr(reader, '_checkpoint_tracker'):
    residual_bits = reader.position - pos_before_residual
    reader._checkpoint_tracker.checkpoint(
        reader, 'residual_end',
        residual_bits=residual_bits,
        bits_since_start=reader.position - mb_start_pos
    )
```

### Step 5: Test basic functionality

```bash
pytest reconstruct/tests/test_macroblock.py -v
```

Expected: All tests pass (checkpoints don't affect decode)

### Step 6: Commit

```bash
git add reconstruct/macroblock.py
git commit -m "feat(mb): add internal checkpoints for debugging"
```

---

## Task 3: Update Tracker to Store Reader Reference

**Goal:** Allow checkpoints to be called from within MB decode

**Files:**
- Modify: `debug/test_mb_tracker.py`

### Step 1: Attach tracker to reader

Edit `debug/test_mb_tracker.py`, in `tracked_decode` function:

```python
def tracked_decode(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs):
    tracker.start_mb(mb_x, mb_y, reader)

    # Attach tracker to reader for internal checkpoints
    reader._checkpoint_tracker = tracker
    mb_start_pos = reader.position

    try:
        result = original_decode(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs)
        tracker.end_mb(reader, 'OK')

        # Clean up
        if hasattr(reader, '_checkpoint_tracker'):
            delattr(reader, '_checkpoint_tracker')

        return result
    except Exception as e:
        tracker.end_mb(reader, f'FAIL: {str(e)[:50]}')
        if hasattr(reader, '_checkpoint_tracker'):
            delattr(reader, '_checkpoint_tracker')
        raise
```

### Step 2: Update print_summary to show checkpoints

Edit `debug/mb_position_tracker.py`:

```python
def print_summary(self):
    """Print summary of all MBs with internal checkpoints."""
    mbs = {}
    for cp in self.checkpoints:
        mb = cp['mb']
        if mb not in mbs:
            mbs[mb] = []
        mbs[mb].append(cp)

    for mb, cps in sorted(mbs.items()):
        if len(cps) >= 2:
            start = cps[0]['position']
            end = cps[-1]['position']
            print(f"\nMB{mb}: {start:4d} -> {end:4d} ({end-start:4d} bits)")

            # Print internal checkpoints
            for i, cp in enumerate(cps):
                label = cp['label']
                pos = cp['position']

                if label == 'MB_START':
                    print(f"  START at {pos:4d}")
                elif label == 'MB_END':
                    status = cp.get('status', 'OK')
                    print(f"  END   at {pos:4d} [{status}]")
                else:
                    # Internal checkpoint
                    bits_since_prev = 0
                    if i > 0:
                        bits_since_prev = pos - cps[i-1]['position']
                    print(f"  {label:15s} at {pos:4d} (+{bits_since_prev:3d} bits)")
```

### Step 3: Test with motion_64x64.264

```bash
PYTHONPATH=. python debug/test_mb_tracker.py 2>&1 | grep -A 30 "MB(1, 0)"
```

Expected: See internal checkpoints for MB(1,0) showing where bits are consumed

### Step 4: Commit

```bash
git add debug/test_mb_tracker.py debug/mb_position_tracker.py
git commit -m "feat(debug): show internal MB checkpoints in tracker"
```

---

## Task 4: Analyze MB(1,0) Checkpoint Data

**Goal:** Run tracker and identify which step in MB(1,0) consumes excess bits

**Files:** None (analysis only)

### Step 1: Run tracker with detailed output

```bash
PYTHONPATH=. python debug/test_mb_tracker.py 2>&1 > /tmp/mb_checkpoint_analysis.txt
```

### Step 2: Extract MB(1,0) data

```bash
grep -A 20 "^MB(1, 0)" /tmp/mb_checkpoint_analysis.txt
```

### Step 3: Compare to MB(0,0)

```bash
echo "=== MB(0,0) ===" && grep -A 20 "^MB(0, 0)" /tmp/mb_checkpoint_analysis.txt
echo ""
echo "=== MB(1,0) ===" && grep -A 20 "^MB(1, 0)" /tmp/mb_checkpoint_analysis.txt
```

### Step 4: Document findings

Create `/tmp/mb_1_0_analysis.txt`:

```bash
cat > /tmp/mb_1_0_analysis.txt << 'EOF'
# MB(1,0) Bit Consumption Analysis

## MB(0,0) Baseline
- mb_type: X bits
- cbp: X bits
- mb_qp_delta: X bits
- residual: X bits
- TOTAL: 569 bits

## MB(1,0) Actual
- mb_type: X bits
- cbp: X bits
- mb_qp_delta: X bits
- residual: X bits
- TOTAL: 684 bits

## Difference Analysis
Step with excess bits: [IDENTIFIED STEP]
Expected: X bits
Actual: Y bits
Excess: Z bits

## Root Cause Hypothesis
[Specific code path causing excess bit consumption]
EOF

cat /tmp/mb_1_0_analysis.txt
```

### Step 5: No commit (analysis only)

---

## Task 5: Add Block-Level Checkpoints

**Goal:** If residual decode consumes excess bits, add checkpoints for each block

**Files:**
- Modify: `reconstruct/macroblock.py`

### Step 1: Add checkpoint in block decode loop

Find the residual block decode loop. Add checkpoint after each block:

```python
for block_idx in range(num_blocks):
    pos_before_block = reader.position

    # [existing block decode code]

    if hasattr(reader, '_checkpoint_tracker'):
        bits_consumed = reader.position - pos_before_block
        reader._checkpoint_tracker.checkpoint(
            reader, f'block_{block_idx}',
            block_idx=block_idx,
            bits=bits_consumed
        )
```

### Step 2: Test

```bash
PYTHONPATH=. python debug/test_mb_tracker.py 2>&1 | grep -A 30 "MB(1, 0)"
```

Expected: See individual block consumption for MB(1,0)

### Step 3: Commit

```bash
git add reconstruct/macroblock.py
git commit -m "feat(mb): add per-block checkpoints in residual decode"
```

---

## Task 6: Identify Exact Bug Location

**Goal:** Use checkpoint data to pinpoint exact code causing excess bits

**Files:** None (analysis only)

### Step 1: Run updated tracker

```bash
PYTHONPATH=. python debug/test_mb_tracker.py 2>&1 > /tmp/mb_block_analysis.txt
```

### Step 2: Analyze block-by-block consumption

```bash
grep "block_" /tmp/mb_block_analysis.txt | head -40
```

### Step 3: Identify anomaly

Look for:
- Block that consumes unusually many bits
- Extra blocks being decoded (CBP=0 but blocks decoded anyway)
- Wrong block type being decoded

### Step 4: Document exact bug

Update `/tmp/mb_1_0_analysis.txt`:

```bash
cat >> /tmp/mb_1_0_analysis.txt << 'EOF'

## Exact Bug Location

File: reconstruct/macroblock.py
Function: [function name]
Line: [approximate line number]

Bug description:
[Specific issue - e.g., "Loop decodes 25 blocks instead of 16",
 "Reads mb_qp_delta when CBP=0", etc.]

Expected behavior:
[What should happen]

Actual behavior:
[What is happening]

Fix required:
[Specific code change needed]
EOF

cat /tmp/mb_1_0_analysis.txt
```

### Step 5: No commit (analysis document is temporary)

---

## Task 7: Write Failing Test

**Goal:** Create test that reproduces MB(1,0) bug in isolation

**Files:**
- Create: `tests/test_mb_1_0_bug.py`

### Step 1: Extract MB(1,0) bitstream segment

```python
# tests/test_mb_1_0_bug.py
"""Test case for MB(1,0) bit drift bug."""
import pytest
from bitstream import BitReader
from reconstruct.macroblock import decode_macroblock
from parameters import SPS, PPS

def test_mb_1_0_bit_consumption():
    """MB(1,0) should consume ~330 bits, not 684 bits.

    This test reproduces the bit drift bug found in motion_64x64.264.
    MB(1,0) starts at bit position 597 and should end around 927,
    but currently ends at 1281 (684 bits consumed).
    """
    # Load motion_64x64.264 and extract MB(1,0) segment
    with open('test_data/motion_64x64.264', 'rb') as f:
        data = f.read()

    # Extract slice NAL and position reader at MB(1,0) start
    # (Position 597 relative to slice data start)
    # This is simplified - actual test would need full NAL parsing

    # For now, just assert the bug exists
    pytest.skip("Pending: Need to extract exact bitstream segment")
```

### Step 2: Run test

```bash
pytest tests/test_mb_1_0_bug.py -v
```

Expected: Test skipped (we'll implement it after understanding the bug)

### Step 3: Commit

```bash
git add tests/test_mb_1_0_bug.py
git commit -m "test(mb): add placeholder test for MB(1,0) bug"
```

---

## Task 8: Implement Fix

**Goal:** Fix the specific code causing MB(1,0) to consume excess bits

**Files:** TBD (depends on Task 6 findings)

### Step 1: Review analysis

Read `/tmp/mb_1_0_analysis.txt` to understand exact fix needed.

### Step 2: Implement fix

Example fixes (will be customized based on actual bug):

**If bug is CBP interpretation:**
```python
# BEFORE (wrong):
if cbp > 0:
    decode_residual(...)

# AFTER (correct):
cbp_luma = cbp & 0xF
cbp_chroma = (cbp >> 4) & 0x3
if cbp_luma > 0:
    decode_luma_residual(...)
if cbp_chroma > 0:
    decode_chroma_residual(...)
```

**If bug is block count:**
```python
# BEFORE (wrong):
for i in range(16):  # Always 16 blocks
    decode_block(...)

# AFTER (correct):
num_blocks = count_blocks_from_cbp(cbp)
for i in range(num_blocks):
    decode_block(...)
```

**If bug is conditional read:**
```python
# BEFORE (wrong):
mb_qp_delta = reader.read_se()  # Always read

# AFTER (correct):
if cbp != 0:  # Only read when CBP non-zero
    mb_qp_delta = reader.read_se()
else:
    mb_qp_delta = 0
```

### Step 3: Run tests

```bash
pytest -v
```

Expected: All existing tests still pass

### Step 4: Test on motion_64x64.264

```bash
PYTHONPATH=. python debug/test_mb_tracker.py 2>&1 | grep "MB(1, 0)"
```

Expected: MB(1,0) now consumes ~330 bits instead of 684

### Step 5: Commit

```bash
git add [modified files]
git commit -m "fix(mb): correct [specific issue] causing bit drift

MB(1,0) was consuming 684 bits instead of ~330 due to [specific bug].
Fixed by [specific change].

Verified on motion_64x64.264: MB(1,0) now consumes [new value] bits."
```

---

## Task 9: Update Test with Actual Fix

**Goal:** Make test_mb_1_0_bug.py actually test the fix

**Files:**
- Modify: `tests/test_mb_1_0_bug.py`

### Step 1: Implement full test

```python
def test_mb_1_0_bit_consumption():
    """MB(1,0) should consume reasonable bits (not 684).

    Regression test for bit drift bug in motion_64x64.264.
    """
    # Use MB position tracker to verify
    from debug.mb_position_tracker import MBPositionTracker

    tracker = MBPositionTracker()

    # Run decode with tracker
    # [implementation details]

    # Assert MB(1,0) consumes reasonable bits
    mb_1_0 = tracker.get_mb_summary(1, 0)
    assert mb_1_0 is not None
    assert mb_1_0['consumed'] < 400, \
        f"MB(1,0) consumed {mb_1_0['consumed']} bits (expected <400)"
```

### Step 2: Run test

```bash
pytest tests/test_mb_1_0_bug.py -v
```

Expected: PASS

### Step 3: Commit

```bash
git add tests/test_mb_1_0_bug.py
git commit -m "test(mb): implement MB(1,0) bit consumption regression test"
```

---

## Task 10: Verify Fix on Both Failing Videos

**Goal:** Confirm motion_64x64.264 and motion_simple.264 now decode

**Files:** None (testing only)

### Step 1: Test motion_64x64.264

```bash
python -c "
from decoder import H264Decoder
decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_64x64.264'))
    print(f'✓ motion_64x64.264: {len(frames)} frames')
except Exception as e:
    print(f'✗ motion_64x64.264: {e}')
"
```

Expected: ✓ Success

### Step 2: Test motion_simple.264

```bash
python -c "
from decoder import H264Decoder
decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_simple.264'))
    print(f'✓ motion_simple.264: {len(frames)} frames')
except Exception as e:
    print(f'✗ motion_simple.264: {e}')
"
```

Expected: ✓ Success

### Step 3: Run full test suite

```bash
pytest -v
```

Expected: All tests pass, no new failures

### Step 4: Document success

```bash
echo "Both failing videos now decode successfully!" > /tmp/fix_verified.txt
```

---

## Task 11: Test All Videos in test_data/

**Goal:** Ensure fix doesn't break previously passing videos

**Files:** None (testing only)

### Step 1: Create test script

```python
# test_all_videos.py
"""Test decoder on all videos in test_data/."""
from decoder import H264Decoder
import os

test_dir = 'test_data'
decoder = H264Decoder()

passed = []
failed = []

for filename in sorted(os.listdir(test_dir)):
    if not filename.endswith('.264'):
        continue

    filepath = os.path.join(test_dir, filename)
    try:
        frames = list(decoder.decode_file(filepath))
        passed.append((filename, len(frames)))
        print(f"✓ {filename}: {len(frames)} frames")
    except Exception as e:
        failed.append((filename, str(e)[:80]))
        print(f"✗ {filename}: {str(e)[:80]}")

print(f"\n=== RESULTS ===")
print(f"Passed: {len(passed)}/{len(passed)+len(failed)}")
print(f"Failed: {len(failed)}/{len(passed)+len(failed)}")

if len(failed) > 0:
    print("\nFailed videos:")
    for name, error in failed:
        print(f"  {name}: {error}")
```

### Step 2: Run on all videos

```bash
python test_all_videos.py > /tmp/all_videos_result.txt 2>&1
cat /tmp/all_videos_result.txt
```

### Step 3: Verify no regressions

Expected results:
- motion_64x64.264: ✓ (was failing, now passes)
- motion_simple.264: ✓ (was failing, now passes)
- All previously passing videos: ✓ (still pass)
- Known failing videos (truncated, empty): ✗ (expected failures)

### Step 4: No commit (test script is temporary)

---

## Task 12: Clean Up Debug Code

**Goal:** Remove temporary debugging checkpoints from production code

**Files:**
- Modify: `reconstruct/macroblock.py`
- Keep: `debug/` directory (useful for future debugging)

### Step 1: Remove checkpoint calls from macroblock.py

Remove all lines containing `_checkpoint_tracker`:

```bash
grep -n "_checkpoint_tracker" reconstruct/macroblock.py
```

Remove those lines (or comment them out).

### Step 2: Verify tests still pass

```bash
pytest -v
```

Expected: All tests pass (checkpoints were debug-only)

### Step 3: Test motion videos again

```bash
python -c "
from decoder import H264Decoder
decoder = H264Decoder()
for video in ['motion_64x64.264', 'motion_simple.264']:
    try:
        frames = list(decoder.decode_file(f'test_data/{video}'))
        print(f'✓ {video}: {len(frames)} frames')
    except Exception as e:
        print(f'✗ {video}: {e}')
"
```

Expected: Both still pass

### Step 4: Commit

```bash
git add reconstruct/macroblock.py
git commit -m "refactor(mb): remove debug checkpoints (fix verified)"
```

---

## Task 13: Update Documentation

**Goal:** Document the bug fix and investigation process

**Files:**
- Update: `docs/SESSION_SUMMARY.md`

### Step 1: Add bug fix entry

Edit `docs/SESSION_SUMMARY.md`:

```markdown
### 7. MB(1,0) Bit Drift (motion videos fixed)

**Problem:** motion_64x64.264 and motion_simple.264 failed during decode with "Invalid coeff_token" errors.

**Root Cause:** MB(1,0) consumed 684 bits instead of ~330 bits due to [specific bug description].

**Investigation Process:**
1. VLC audit (6 tasks) proved all CAVLC VLC functions correct
2. MB position tracker identified MB(1,0) as source of 60-bit drift
3. Internal checkpoints pinpointed exact step consuming excess bits
4. Fixed [specific code issue]

**Solution:** [Specific fix description]

**Fixed Videos:**
- motion_64x64.264 ✓
- motion_simple.264 ✓

**Commit:** [commit hash] - fix(mb): correct [issue]
```

### Step 2: Update success rate

Update the success rate at top of file:
```markdown
**Success Rate:** [new %] ([new count]/29 videos)
**Improvement:** +2 videos (from 79.3%)
```

### Step 3: Commit

```bash
git add docs/SESSION_SUMMARY.md
git commit -m "docs: document MB(1,0) bit drift bug fix"
```

---

## Summary

This plan systematically identifies and fixes the bit drift bug in MB(1,0):

1. **Tasks 1-3**: Add infrastructure for detailed bit consumption tracking
2. **Tasks 4-6**: Analyze MB(1,0) to identify exact bug location
3. **Tasks 7-9**: Implement and test fix
4. **Tasks 10-11**: Verify fix on all videos
5. **Tasks 12-13**: Clean up and document

**Expected Outcome:** motion_64x64.264 and motion_simple.264 decode successfully, raising decoder success rate from 79.3% to 86.2% (25/29 videos).

**Critical Path:** The bug identification in Tasks 4-6 will determine the specific fix in Task 8. Be prepared to customize Task 8 based on actual findings.
