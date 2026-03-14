# MB-Level Parsing Bit Audit Plan

**Goal**: Identify bit consumption bugs in macroblock-level parsing that cause bit drift in motion_64x64.264 and motion_simple.264

**Context**: VLC audit (Tasks 1-6) proved all CAVLC VLC functions are correct. Bit drift must occur in MB-level parsing: mb_type, CBP, mb_qp_delta, or block iteration logic.

**Approach**: Systematic validation of bit consumption at each MB parsing step, similar to successful VLC audit.

---

## Background

### Known Facts from VLC Audit
- ✅ coeff_token: Consumes correct bits (validated with 13 tests)
- ✅ total_zeros: Consumes correct bits (validated with 29 tests)
- ✅ run_before: Consumes correct bits (validated with 45 tests)
- ✅ levels: Consumes correct bits (validated with 26 tests)

### Failure Pattern
- **motion_64x64.264**: Fails at MB(0,1), position 1288, reads ue=2431 (invalid)
- **motion_simple.264**: Fails at MB(1,0), reads 11 zero bits (invalid total_zeros)
- **Pattern**: NO validation errors before failure → bit drift from earlier MB parsing

### Hypothesis
MB(0,0) in motion_64x64.264 consumes ~1260 bits (1288 - 28 header). One of these is consuming wrong bits:
1. **mb_type decode** (ue=23 for I_16x16)
2. **I_16x16 pred_mode extraction** (2 bits from mb_type value)
3. **CBP decode** (varies by mode)
4. **mb_qp_delta** (conditional se)
5. **Residual block iteration** (16 luma + 2 chroma, or different for I_16x16)

---

## Task 1: Add MB-Level Position Tracking

**Goal**: Track bit positions at MB boundaries to identify first divergence

**Files**:
- Create: `debug/mb_position_tracker.py`

**Implementation**:

```python
# debug/mb_position_tracker.py
"""Track bit positions at MB parsing boundaries."""
from typing import Dict, List, Tuple
from bitstream import BitReader

class MBPositionTracker:
    """Track bit positions during MB decode for debugging."""

    def __init__(self):
        self.checkpoints: List[Dict] = []
        self.current_mb: Tuple[int, int] = (0, 0)

    def checkpoint(self, reader: BitReader, label: str, **kwargs):
        """Record current bit position with label."""
        self.checkpoints.append({
            'mb': self.current_mb,
            'position': reader.position,
            'label': label,
            **kwargs
        })

    def start_mb(self, mb_x: int, mb_y: int, reader: BitReader):
        """Mark start of MB decode."""
        self.current_mb = (mb_x, mb_y)
        self.checkpoint(reader, 'MB_START')

    def end_mb(self, reader: BitReader, status: str = 'OK'):
        """Mark end of MB decode."""
        self.checkpoint(reader, 'MB_END', status=status)

    def get_mb_summary(self, mb_x: int, mb_y: int) -> Dict:
        """Get summary for specific MB."""
        mb_checkpoints = [c for c in self.checkpoints if c['mb'] == (mb_x, mb_y)]
        if len(mb_checkpoints) < 2:
            return None

        start = mb_checkpoints[0]['position']
        end = mb_checkpoints[-1]['position']

        return {
            'mb': (mb_x, mb_y),
            'start': start,
            'end': end,
            'consumed': end - start,
            'checkpoints': mb_checkpoints
        }

    def print_summary(self):
        """Print summary of all MBs."""
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
                print(f"MB{mb}: {start:4d} -> {end:4d} ({end-start:4d} bits)")
                for cp in cps[1:-1]:  # Skip START and END
                    print(f"  [{cp['position']:4d}] {cp['label']}")

# Global tracker instance
_tracker = MBPositionTracker()

def get_tracker() -> MBPositionTracker:
    """Get global position tracker."""
    return _tracker
```

**Test with motion_64x64.264**:

```bash
python -c "
from debug.mb_position_tracker import get_tracker
from decoder import H264Decoder
import reconstruct.macroblock as mb_mod

tracker = get_tracker()
original_decode = mb_mod.decode_macroblock

def tracked_decode(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs):
    tracker.start_mb(mb_x, mb_y, reader)
    try:
        result = original_decode(reader, sps, pps, slice_qp, mb_x, mb_y, *args, **kwargs)
        tracker.end_mb(reader, 'OK')
        return result
    except Exception as e:
        tracker.end_mb(reader, f'FAIL: {e}')
        raise

mb_mod.decode_macroblock = tracked_decode

decoder = H264Decoder()
try:
    list(decoder.decode_file('test_data/motion_64x64.264'))
except:
    pass

tracker.print_summary()
"
```

**Commit**: `git add debug/mb_position_tracker.py && git commit -m "feat(debug): add MB position tracker"`

---

## Task 2: Add mb_type Decode Validation

**Goal**: Verify mb_type ue decode consumes correct bits and value is valid

**Files**:
- Modify: `reconstruct/macroblock.py` (add validation after mb_type decode)
- Create: `tests/test_mb_type_decode.py`

**Implementation**:

Add validation in `decode_macroblock` after reading mb_type:

```python
def decode_macroblock(reader: BitReader, sps, pps, slice_qp, mb_x, mb_y, ...):
    """Decode a single macroblock."""

    # Track position
    pos_before_mb_type = reader.position

    # Read mb_type
    mb_type_ue = reader.read_ue()

    # Validate mb_type value is in valid range
    if mb_type_ue > 25:  # Max valid I-slice mb_type is 25 (I_PCM)
        raise ValueError(
            f"Invalid mb_type={mb_type_ue} at MB({mb_x},{mb_y}), "
            f"position {pos_before_mb_type}. Valid range: 0-25 for I-slices"
        )

    # Log for debugging
    logger.debug(
        f"MB({mb_x},{mb_y}): mb_type_ue={mb_type_ue} "
        f"at position {pos_before_mb_type}"
    )

    # Continue with existing decode logic...
```

**Test**:

```python
# tests/test_mb_type_decode.py
"""Test mb_type decode validation."""
import pytest
from bitstream import BitReader
from reconstruct.macroblock import decode_macroblock

def test_mb_type_i4x4():
    """mb_type=0 (I_4x4) is valid."""
    # This would need full MB context - simplified test
    pass

def test_mb_type_invalid_large():
    """mb_type > 25 should raise ValueError."""
    # Test with constructed bitstream where ue > 25
    pass
```

**Run on failing video**:

```bash
python -c "
from decoder import H264Decoder
decoder = H264Decoder()
try:
    list(decoder.decode_file('test_data/motion_64x64.264'))
except ValueError as e:
    if 'Invalid mb_type' in str(e):
        print(f'✓ Caught invalid mb_type: {e}')
    else:
        print(f'✗ Different error: {e}')
"
```

**Commit**: `git add reconstruct/macroblock.py tests/test_mb_type_decode.py && git commit -m "feat(mb): add mb_type validation"`

---

## Task 3: Add CBP Decode Validation

**Goal**: Verify coded_block_pattern consumes correct bits for each mb_type

**Files**:
- Modify: `reconstruct/macroblock.py` (add CBP validation)

**Context**: CBP (coded_block_pattern) uses different VLC tables based on mb_type:
- I_4x4: CBP uses Table 9-4(a) (me/te encoding)
- I_16x16: CBP encoded in mb_type value (no separate decode)

**Implementation**:

```python
# After decoding mb_type, before decoding CBP

if mb_type is I_4x4:
    pos_before_cbp = reader.position
    cbp_me = reader.read_me()  # Or whatever the actual function is
    bits_consumed = reader.position - pos_before_cbp

    # Validate CBP is in valid range (0-47 for luma+chroma)
    if cbp > 47:
        raise ValueError(
            f"Invalid CBP={cbp} at MB({mb_x},{mb_y}), "
            f"position {pos_before_cbp}, consumed {bits_consumed} bits"
        )

    logger.debug(
        f"MB({mb_x},{mb_y}): CBP={cbp}, consumed {bits_consumed} bits "
        f"at position {pos_before_cbp}"
    )

elif mb_type is I_16x16:
    # CBP encoded in mb_type value
    # Extract: cbp_chroma = (mb_type_ue - 1) // 12
    #          cbp_luma = ((mb_type_ue - 1) % 12) // 4
    cbp_chroma = (mb_type_ue - 1) // 12
    cbp_luma_pattern = ((mb_type_ue - 1) % 12) // 4

    logger.debug(
        f"MB({mb_x},{mb_y}): I_16x16 CBP from mb_type: "
        f"chroma={cbp_chroma}, luma_pattern={cbp_luma_pattern}"
    )
```

**Commit**: `git add reconstruct/macroblock.py && git commit -m "feat(mb): add CBP decode validation"`

---

## Task 4: Add mb_qp_delta Validation

**Goal**: Verify mb_qp_delta is read only when CBP != 0

**Files**:
- Modify: `reconstruct/macroblock.py`

**H.264 Spec**: mb_qp_delta (se) is present only when `cbp != 0 || mb_type == I_PCM`

**Implementation**:

```python
# After decoding CBP

should_read_qp_delta = (cbp != 0) or (mb_type == I_PCM)

if should_read_qp_delta:
    pos_before_qp_delta = reader.position
    mb_qp_delta = reader.read_se()
    bits_consumed = reader.position - pos_before_qp_delta

    logger.debug(
        f"MB({mb_x},{mb_y}): mb_qp_delta={mb_qp_delta}, "
        f"consumed {bits_consumed} bits at position {pos_before_qp_delta}"
    )
else:
    mb_qp_delta = 0
    logger.debug(f"MB({mb_x},{mb_y}): mb_qp_delta=0 (not present, CBP=0)")

# Validate current QP is in valid range
current_qp = prev_qp + mb_qp_delta
if not (0 <= current_qp <= 51):
    raise ValueError(
        f"Invalid QP={current_qp} (prev={prev_qp}, delta={mb_qp_delta}) "
        f"at MB({mb_x},{mb_y})"
    )
```

**Commit**: `git add reconstruct/macroblock.py && git commit -m "feat(mb): add mb_qp_delta validation"`

---

## Task 5: Add Block Iteration Validation

**Goal**: Verify correct number of residual blocks are decoded for each mb_type

**Files**:
- Modify: `reconstruct/macroblock.py`

**Context**: Number of blocks varies by mb_type:
- I_4x4: 16 luma 4x4 + 4 chroma DC + 8 chroma AC = 28 blocks (if CBP indicates)
- I_16x16: 1 luma DC + 16 luma AC + 4 chroma DC + 8 chroma AC = 29 blocks (if CBP indicates)
- I_PCM: No residual blocks (raw samples)

**Implementation**:

```python
# Before decoding residual blocks

pos_before_residual = reader.position
blocks_decoded = 0

# Track each block decode
def decode_and_count_block(block_type, block_idx):
    nonlocal blocks_decoded
    pos_before_block = reader.position

    # Decode block (existing code)
    block_data = decode_residual_block(reader, ...)

    blocks_decoded += 1
    bits_consumed = reader.position - pos_before_block

    logger.debug(
        f"MB({mb_x},{mb_y}): Block {blocks_decoded} ({block_type}[{block_idx}]) "
        f"consumed {bits_consumed} bits at position {pos_before_block}"
    )

    return block_data

# After all residual blocks decoded

total_residual_bits = reader.position - pos_before_residual
expected_blocks = calculate_expected_blocks(mb_type, cbp)

logger.debug(
    f"MB({mb_x},{mb_y}): Decoded {blocks_decoded}/{expected_blocks} blocks, "
    f"total residual bits: {total_residual_bits}"
)

if blocks_decoded != expected_blocks:
    logger.warning(
        f"MB({mb_x},{mb_y}): Block count mismatch! "
        f"Decoded {blocks_decoded}, expected {expected_blocks}"
    )
```

**Commit**: `git add reconstruct/macroblock.py && git commit -m "feat(mb): add block iteration validation"`

---

## Task 6: Run Full MB Validation on Failing Videos

**Goal**: Identify exact MB parsing step where bit drift occurs

**Files**: None (testing only)

**Step 1**: Test motion_64x64.264 with all MB validations

```bash
python -c "
from decoder import H264Decoder
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('reconstruct.macroblock')
logger.setLevel(logging.DEBUG)

decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_64x64.264'))
    print(f'SUCCESS: Decoded {len(frames)} frames')
except ValueError as e:
    print(f'VALIDATION ERROR: {e}')
except Exception as e:
    print(f'DECODE ERROR: {e}')
" 2>&1 | tee /tmp/mb_validation_motion64.log
```

**Step 2**: Test motion_simple.264

```bash
python -c "
from decoder import H264Decoder
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('reconstruct.macroblock')
logger.setLevel(logging.DEBUG)

decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_simple.264'))
    print(f'SUCCESS: Decoded {len(frames)} frames')
except ValueError as e:
    print(f'VALIDATION ERROR: {e}')
except Exception as e:
    print(f'DECODE ERROR: {e}')
" 2>&1 | tee /tmp/mb_validation_simple.log
```

**Step 3**: Analyze logs

```bash
echo "=== motion_64x64.264 MB(0,0) ==="
grep "MB(0,0)" /tmp/mb_validation_motion64.log | head -20

echo -e "\n=== motion_64x64.264 MB(0,1) ==="
grep "MB(0,1)" /tmp/mb_validation_motion64.log | head -20

echo -e "\n=== Validation errors ==="
grep "VALIDATION ERROR\|Invalid\|mismatch" /tmp/mb_validation_motion64.log
```

**Step 4**: Document findings

Create `/tmp/mb_audit_findings.txt`:

```bash
cat > /tmp/mb_audit_findings.txt << 'EOF'
# MB-Level Parsing Audit Findings

## motion_64x64.264 Analysis

### MB(0,0)
- Start position: [X]
- mb_type: [X]
- CBP: [X]
- mb_qp_delta: [X]
- Blocks decoded: [X]
- End position: [X]
- Total bits consumed: [X]

### MB(0,1)
- Expected start position: [X]
- Actual start position: [X]
- Position drift: [X] bits

## Root Cause

[Identified MB parsing step causing bit drift]

## Fix Required

[Specific code change needed]
EOF

cat /tmp/mb_audit_findings.txt
```

---

## Task 7: Fix Identified MB Parsing Bug

**Goal**: Fix the specific MB parsing step identified in Task 6

**Files**: TBD (depends on findings)

**This task will be customized based on Task 6 findings. Possible bugs:**

1. **I_16x16 pred_mode extraction**: Wrong bit extraction from mb_type value
2. **CBP decode**: Wrong table or wrong interpretation
3. **mb_qp_delta conditional**: Reading when shouldn't or vice versa
4. **Block iteration**: Wrong block count for certain CBP patterns
5. **I_PCM handling**: Not aligning to byte boundary

**Template for fix**:

```python
# Fix will follow pattern:

# BEFORE (buggy code):
# [existing code that causes drift]

# AFTER (fixed code):
# [corrected code with proper bit consumption]

# Add test case to prevent regression
# Create test in tests/test_mb_parsing.py
```

**Verification**:

```bash
# Run full test suite
pytest -v

# Test on failing videos
python -c "
from decoder import H264Decoder
decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_64x64.264'))
    print(f'✓ motion_64x64.264: {len(frames)} frames')
except Exception as e:
    print(f'✗ motion_64x64.264: {e}')

try:
    frames = list(decoder.decode_file('test_data/motion_simple.264'))
    print(f'✓ motion_simple.264: {len(frames)} frames')
except Exception as e:
    print(f'✗ motion_simple.264: {e}')
"
```

**Commit**: `git add [files] && git commit -m "fix(mb): [description of fix]"`

---

## Task 8: Verify Fix on All Test Videos

**Goal**: Ensure fix doesn't break existing passing videos

**Files**: None (testing only)

```bash
python -c "
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
    except Exception as e:
        failed.append((filename, str(e)))

print(f'\\n=== PASSED: {len(passed)}/{len(passed)+len(failed)} ===')
for name, count in passed:
    print(f'  ✓ {name}: {count} frames')

print(f'\\n=== FAILED: {len(failed)}/{len(passed)+len(failed)} ===')
for name, error in failed:
    print(f'  ✗ {name}: {error[:80]}')
"
```

**Success criteria**:
- motion_64x64.264 ✓
- motion_simple.264 ✓
- All previously passing videos still pass
- No new failures introduced

---

## Summary

This MB-level audit systematically validates bit consumption at each macroblock parsing step, following the same proven approach as the VLC audit. Each task adds validation, creates tests, and runs on failing videos to identify the exact bug location.

**Expected outcome**: Identify and fix the specific MB parsing step causing bit drift, restoring motion video decode to working state.
