# CAVLC VLC Bit Consumption Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Find and fix the VLC decode bug causing ue=2431 (impossibly large value) by auditing bit consumption in coeff_token, total_zeros, run_before, and level decoding.

**Architecture:** Add bit position validation assertions throughout entropy/cavlc.py to verify each VLC decode consumes the correct number of bits. Compare actual vs expected bit consumption at each step to identify where drift begins.

**Tech Stack:** Python, pytest, H.264 CAVLC VLC tables

---

## Background

**Problem:** motion_64x64.264 and motion_simple.264 fail with bit drift. Investigation revealed:
- Both are I-slices (type=7), not P-slices
- `ue=2431` read at position 1288 (impossibly large for CAVLC)
- Manual audit confirmed syntax order is correct per H.264 spec
- Bug must be in VLC bit consumption, not element order

**Root Cause Hypothesis:** One of the VLC decode functions (coeff_token, total_zeros, run_before, or level) is consuming wrong number of bits, causing cumulative drift.

---

### Task 1: Add Bit Position Validation Utilities

**Files:**
- Create: `entropy/tests/test_vlc_validation.py`
- Modify: `entropy/cavlc.py` (add validation helpers)

**Step 1: Write validation helper test**

```python
# entropy/tests/test_vlc_validation.py
"""Test VLC bit consumption validation utilities."""
import pytest
from bitstream import BitReader
from entropy.cavlc import validate_vlc_bits_consumed


def test_validate_vlc_bits_consumed_correct():
    """Validation passes when bit consumption matches."""
    data = bytes([0b10110000])  # ue(2) = '011'
    reader = BitReader(data)
    pos_before = reader.position

    reader.read_bits(3)  # Read '011'

    # Should not raise
    validate_vlc_bits_consumed(
        reader, pos_before, expected_bits=3,
        context="test_ue", code_value=2
    )


def test_validate_vlc_bits_consumed_mismatch():
    """Validation fails when bit consumption doesn't match."""
    data = bytes([0b10110000])
    reader = BitReader(data)
    pos_before = reader.position

    reader.read_bits(2)  # Read only 2 bits instead of 3

    with pytest.raises(ValueError, match="VLC bit consumption mismatch"):
        validate_vlc_bits_consumed(
            reader, pos_before, expected_bits=3,
            context="test_ue", code_value=2
        )
```

**Step 2: Run test to verify it fails**

```bash
pytest entropy/tests/test_vlc_validation.py::test_validate_vlc_bits_consumed_correct -v
```

Expected: `FAIL` with "ImportError: cannot import name 'validate_vlc_bits_consumed'"

**Step 3: Implement validation helper**

Add to `entropy/cavlc.py` after imports:

```python
def validate_vlc_bits_consumed(
    reader: BitReader,
    pos_before: int,
    expected_bits: int,
    context: str,
    code_value: any
) -> None:
    """Validate VLC decode consumed expected number of bits.

    Args:
        reader: Bit reader after VLC decode
        pos_before: Bit position before VLC decode
        expected_bits: Expected number of bits consumed
        context: Description for error message (e.g., "coeff_token nC=2")
        code_value: The decoded value

    Raises:
        ValueError: If bit consumption doesn't match expected
    """
    actual_bits = reader.position - pos_before
    if actual_bits != expected_bits:
        raise ValueError(
            f"VLC bit consumption mismatch in {context}: "
            f"expected {expected_bits} bits, consumed {actual_bits} bits "
            f"(value={code_value}, pos_before={pos_before}, pos_after={reader.position})"
        )
```

**Step 4: Run tests to verify they pass**

```bash
pytest entropy/tests/test_vlc_validation.py -v
```

Expected: `2 passed`

**Step 5: Commit**

```bash
git add entropy/cavlc.py entropy/tests/test_vlc_validation.py
git commit -m "feat(cavlc): add VLC bit consumption validation helper"
```

---

### Task 2: Audit coeff_token Bit Consumption

**Files:**
- Modify: `entropy/cavlc.py` (decode_coeff_token function)
- Create: `entropy/tests/test_coeff_token_bits.py`

**Step 1: Write test for known coeff_token codes**

```python
# entropy/tests/test_coeff_token_bits.py
"""Test coeff_token VLC decoding consumes correct number of bits."""
import pytest
from bitstream import BitReader
from entropy.cavlc import decode_coeff_token


def test_coeff_token_nC0_code_1():
    """coeff_token with nC=0, code '1' -> TC=0, T1=0, 1 bit."""
    data = bytes([0b10000000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=0)

    assert tc == 0
    assert t1 == 0
    assert reader.position == 1  # Consumed exactly 1 bit


def test_coeff_token_nC0_code_000101():
    """coeff_token with nC=0, code '000101' -> TC=1, T1=1, 6 bits."""
    data = bytes([0b00010100])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=0)

    assert tc == 1
    assert t1 == 1
    assert reader.position == 6  # Consumed exactly 6 bits


def test_coeff_token_nC2_code_01():
    """coeff_token with nC=2, code '01' -> TC=0, T1=0, 2 bits."""
    data = bytes([0b01000000])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=2)

    assert tc == 0
    assert t1 == 0
    assert reader.position == 2


def test_coeff_token_nC5_fixed_6bits():
    """coeff_token with nC>=8 uses fixed 6-bit code."""
    # TC=4, T1=1 -> fixed code 010001
    data = bytes([0b01000100])
    reader = BitReader(data)

    tc, t1 = decode_coeff_token(reader, nC=8)

    assert tc == 4
    assert t1 == 1
    assert reader.position == 6
```

**Step 2: Run tests**

```bash
pytest entropy/tests/test_coeff_token_bits.py -v
```

Expected: Tests should PASS (if coeff_token is correct) or FAIL (revealing bit consumption bug)

**Step 3: Add validation to decode_coeff_token**

Find the `decode_coeff_token` function in `entropy/cavlc.py` and add validation:

```python
def decode_coeff_token(reader: BitReader, nC: int) -> Tuple[int, int]:
    """Decode coeff_token to get total_coeff and trailing_ones.

    [existing docstring...]
    """
    pos_before = reader.position  # Track starting position

    # [existing code to decode...]
    # total_coeff, trailing_ones = ...

    # Validate bit consumption
    expected_bits = len(code)  # VLC code length from table lookup
    validate_vlc_bits_consumed(
        reader, pos_before, expected_bits,
        context=f"coeff_token nC={nC}",
        code_value=(total_coeff, trailing_ones)
    )

    return total_coeff, trailing_ones
```

**Step 4: Run motion_64x64.264 decode**

```bash
python -c "
from decoder import H264Decoder
decoder = H264Decoder()
try:
    list(decoder.decode_file('test_data/motion_64x64.264'))
except ValueError as e:
    print(f'Caught validation error: {e}')
" 2>&1 | grep -E "(Caught|VLC bit)"
```

Expected: If coeff_token has bug, will show "VLC bit consumption mismatch" with details

**Step 5: Commit**

```bash
git add entropy/cavlc.py entropy/tests/test_coeff_token_bits.py
git commit -m "test(cavlc): add coeff_token bit consumption validation"
```

---

### Task 3: Audit total_zeros Bit Consumption

**Files:**
- Modify: `entropy/cavlc.py` (decode_total_zeros function)
- Create: `entropy/tests/test_total_zeros_bits.py`

**Step 1: Write test for known total_zeros codes**

```python
# entropy/tests/test_total_zeros_bits.py
"""Test total_zeros VLC decoding consumes correct number of bits."""
import pytest
from bitstream import BitReader
from entropy.cavlc import decode_total_zeros


def test_total_zeros_TC1_code_1():
    """total_zeros with TC=1, code '1' -> TZ=0, 1 bit."""
    data = bytes([0b10000000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=1, max_coeffs=16)

    assert tz == 0
    assert reader.position == 1


def test_total_zeros_TC4_code_0001():
    """total_zeros with TC=4, code '0001' -> TZ=11, 4 bits."""
    data = bytes([0b00010000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=4, max_coeffs=16)

    assert tz == 11
    assert reader.position == 4


def test_total_zeros_TC7_code_111():
    """total_zeros with TC=7, code '111' -> TZ=7, 3 bits."""
    data = bytes([0b11100000])
    reader = BitReader(data)

    tz = decode_total_zeros(reader, total_coeff=7, max_coeffs=16)

    assert tz == 7
    assert reader.position == 3
```

**Step 2: Run tests**

```bash
pytest entropy/tests/test_total_zeros_bits.py -v
```

**Step 3: Add validation to decode_total_zeros**

```python
def decode_total_zeros(reader: BitReader, total_coeff: int, max_coeffs: int) -> int:
    """Decode total_zeros value.

    [existing docstring...]
    """
    pos_before = reader.position

    # [existing decode logic...]
    # total_zeros = ...

    # Validate bit consumption
    expected_bits = len(code)  # From table lookup
    validate_vlc_bits_consumed(
        reader, pos_before, expected_bits,
        context=f"total_zeros TC={total_coeff}",
        code_value=total_zeros
    )

    return total_zeros
```

**Step 4: Run motion_64x64.264 decode**

```bash
python -c "
from decoder import H264Decoder
decoder = H264Decoder()
try:
    list(decoder.decode_file('test_data/motion_64x64.264'))
except ValueError as e:
    print(f'Caught validation error: {e}')
" 2>&1 | grep -E "(Caught|VLC bit)"
```

**Step 5: Commit**

```bash
git add entropy/cavlc.py entropy/tests/test_total_zeros_bits.py
git commit -m "test(cavlc): add total_zeros bit consumption validation"
```

---

### Task 4: Audit run_before Bit Consumption

**Files:**
- Modify: `entropy/cavlc.py` (decode_run_before function)
- Create: `entropy/tests/test_run_before_bits.py`

**Step 1: Write test for known run_before codes**

```python
# entropy/tests/test_run_before_bits.py
"""Test run_before VLC decoding consumes correct number of bits."""
import pytest
from bitstream import BitReader
from entropy.cavlc import decode_run_before


def test_run_before_zeros_left_1_code_1():
    """run_before with zeros_left=1, code '1' -> run=0, 1 bit."""
    data = bytes([0b10000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=1)

    assert run == 0
    assert reader.position == 1


def test_run_before_zeros_left_3_code_00():
    """run_before with zeros_left=3, code '00' -> run=3, 2 bits."""
    data = bytes([0b00000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=3)

    assert run == 3
    assert reader.position == 2


def test_run_before_zeros_left_6_code_000():
    """run_before with zeros_left=6, code '000' -> run=6, 3 bits."""
    data = bytes([0b00000000])
    reader = BitReader(data)

    run = decode_run_before(reader, zeros_left=6)

    assert run == 6
    assert reader.position == 3
```

**Step 2: Run tests**

```bash
pytest entropy/tests/test_run_before_bits.py -v
```

**Step 3: Add validation to decode_run_before**

```python
def decode_run_before(reader: BitReader, zeros_left: int) -> int:
    """Decode run_before value.

    [existing docstring...]
    """
    pos_before = reader.position

    # [existing decode logic...]
    # run = ...

    # Validate bit consumption
    expected_bits = len(code)  # From table lookup
    validate_vlc_bits_consumed(
        reader, pos_before, expected_bits,
        context=f"run_before zeros_left={zeros_left}",
        code_value=run
    )

    return run
```

**Step 4: Run motion_64x64.264 decode**

```bash
python -c "
from decoder import H264Decoder
decoder = H264Decoder()
try:
    list(decoder.decode_file('test_data/motion_64x64.264'))
except ValueError as e:
    print(f'Caught validation error: {e}')
" 2>&1 | grep -E "(Caught|VLC bit)"
```

**Step 5: Commit**

```bash
git add entropy/cavlc.py entropy/tests/test_run_before_bits.py
git commit -m "test(cavlc): add run_before bit consumption validation"
```

---

### Task 5: Audit Level Decoding Bit Consumption

**Files:**
- Modify: `entropy/cavlc.py` (decode_levels function)
- Create: `entropy/tests/test_levels_bits.py`

**Step 1: Write test for known level codes**

```python
# entropy/tests/test_levels_bits.py
"""Test level VLC decoding consumes correct number of bits."""
import pytest
from bitstream import BitReader
from entropy.cavlc import decode_levels


def test_levels_simple_prefix_0():
    """Level with prefix=0, suffix_length=0 -> level=1, 1 bit."""
    # Code: '1' (prefix=0, no suffix)
    data = bytes([0b10000000])
    reader = BitReader(data)

    levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

    assert levels == [1]
    assert reader.position == 1


def test_levels_prefix_3_suffix_2():
    """Level with prefix=3, suffix_length=2 -> consumes 3+1+2=6 bits."""
    # Code: '00010' (prefix=3) + 'XX' (2-bit suffix)
    data = bytes([0b00010110])  # prefix '00010' + suffix '11'
    reader = BitReader(data)

    levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

    # Exact level value depends on suffix_length state and suffix bits
    # Main concern: total bits consumed should be 6 (3 zeros + 1 terminator + 2 suffix)
    assert reader.position == 6


def test_levels_extended_escape_prefix_15():
    """Level with prefix>=15 uses extended escape (prefix-3 bits)."""
    # prefix=16 means 16 zeros + 1 terminator (17 bits) + (16-3=13) suffix bits
    # Total: 17 + 13 = 30 bits when suffix_length=0
    data = bytes([
        0b00000000,  # 8 zeros
        0b00000000,  # 8 more zeros
        0b10000000,  # terminator + start of suffix
        0b00000000   # rest of suffix
    ])
    reader = BitReader(data)

    # This will depend on suffix_length state
    # Just verify it doesn't crash and consumes reasonable bits
    levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

    assert len(levels) == 1
    assert reader.position >= 17  # At least prefix bits
```

**Step 2: Run tests**

```bash
pytest entropy/tests/test_levels_bits.py -v
```

**Step 3: Add validation to decode_levels**

In `entropy/cavlc.py`, find the level decode loop and add validation:

```python
def decode_levels(
    reader: BitReader,
    total_coeff: int,
    trailing_ones: int
) -> List[int]:
    """Decode level values for non-trailing coefficients.

    [existing docstring...]
    """
    num_levels = total_coeff - trailing_ones
    if num_levels == 0:
        return []

    levels = []

    # [existing suffix_length initialization...]

    for i in range(num_levels):
        pos_before = reader.position

        # Decode level_prefix (number of leading zeros)
        level_prefix = 0
        while reader.read_bit() == 0:
            level_prefix += 1
            if level_prefix > 31:
                raise ValueError("level_prefix too large")

        # Determine suffix size
        if level_prefix < 14:
            suffix_bits = suffix_length
        elif level_prefix == 14:
            suffix_bits = 4 if suffix_length == 0 else suffix_length
        else:  # level_prefix >= 15
            suffix_bits = (level_prefix - 3) if suffix_length == 0 else suffix_length

        # Read suffix
        if suffix_bits > 0:
            level_suffix = reader.read_bits(suffix_bits)
        else:
            level_suffix = 0

        # Compute level_code
        level_code = _compute_level_code(level_prefix, suffix_length, level_suffix)

        # [existing level conversion and suffix_length update...]

        # Validate total bits consumed
        expected_bits = level_prefix + 1 + suffix_bits  # prefix zeros + terminator + suffix
        actual_bits = reader.position - pos_before
        if actual_bits != expected_bits:
            raise ValueError(
                f"Level decode bit mismatch: expected {expected_bits} bits "
                f"(prefix={level_prefix}, suffix_bits={suffix_bits}), "
                f"but consumed {actual_bits} bits (level={level}, pos={pos_before})"
            )

        levels.append(level)

        # [existing suffix_length update...]

    return levels
```

**Step 4: Run motion_64x64.264 decode**

```bash
python -c "
from decoder import H264Decoder
decoder = H264Decoder()
try:
    list(decoder.decode_file('test_data/motion_64x64.264'))
except ValueError as e:
    print(f'Caught validation error: {e}')
" 2>&1 | head -20
```

Expected: If level decode has bug, validation will catch it with detailed position info

**Step 5: Commit**

```bash
git add entropy/cavlc.py entropy/tests/test_levels_bits.py
git commit -m "test(cavlc): add level decode bit consumption validation"
```

---

### Task 6: Run Full Validation on Failing Videos

**Files:**
- None (testing only)

**Step 1: Test motion_64x64.264 with all validations**

```bash
python -c "
from decoder import H264Decoder
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('entropy.cavlc')
logger.setLevel(logging.DEBUG)

decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_64x64.264'))
    print(f'SUCCESS: Decoded {len(frames)} frames')
except ValueError as e:
    print(f'VALIDATION ERROR: {e}')
except Exception as e:
    print(f'DECODE ERROR: {e}')
" 2>&1 | tee /tmp/motion_64x64_validation.log
```

Expected: Should catch the first VLC decode that consumes wrong number of bits

**Step 2: Test motion_simple.264**

```bash
python -c "
from decoder import H264Decoder
import logging

logging.basicConfig(level=logging.DEBUG)

decoder = H264Decoder()
try:
    frames = list(decoder.decode_file('test_data/motion_simple.264'))
    print(f'SUCCESS: Decoded {len(frames)} frames')
except ValueError as e:
    print(f'VALIDATION ERROR: {e}')
except Exception as e:
    print(f'DECODE ERROR: {e}')
" 2>&1 | tee /tmp/motion_simple_validation.log
```

**Step 3: Analyze validation errors**

```bash
echo "=== motion_64x64.264 errors ==="
grep "VLC bit\|VALIDATION ERROR" /tmp/motion_64x64_validation.log | head -5

echo -e "\n=== motion_simple.264 errors ==="
grep "VLC bit\|VALIDATION ERROR" /tmp/motion_simple_validation.log | head -5
```

Expected: Error messages will show exact VLC decode function and parameters where bit consumption is wrong

**Step 4: Document findings**

Create `/tmp/vlc_audit_findings.txt`:

```bash
cat > /tmp/vlc_audit_findings.txt << 'EOF'
# VLC Bit Consumption Audit Findings

## Test Results

motion_64x64.264: [PASS/FAIL - which function?]
motion_simple.264: [PASS/FAIL - which function?]

## Root Cause

Function: [coeff_token/total_zeros/run_before/levels]
Condition: [e.g., "when nC>=8 and TC>10"]
Expected bits: [X]
Actual bits: [Y]
Code location: entropy/cavlc.py:[line number]

## Reproduction

[Minimal code to reproduce the bug]

## Next Steps

1. [Fix the specific VLC table or bit counting logic]
2. [Verify fix with failing videos]
3. [Run full test suite]
EOF

cat /tmp/vlc_audit_findings.txt
```

**Step 5: No commit** (findings document is temporary analysis)

---

### Task 7: Fix Identified VLC Bug (Template)

**Note:** This task will be customized based on Task 6 findings. Below is a template.

**Files:**
- Modify: `entropy/cavlc.py` (the buggy function)
- Modify: `entropy/tests/test_[function]_bits.py` (add regression test)

**Step 1: Write regression test for the bug**

```python
# Add to appropriate test file
def test_[function]_bug_reproduction():
    """Regression test for bit consumption bug causing ue=2431.

    [Description of bug condition]
    """
    data = bytes([...])  # Exact bitstream that triggered bug
    reader = BitReader(data)

    result = [buggy_function](reader, ...)

    assert result == [expected_value]
    assert reader.position == [expected_bits_consumed]
```

**Step 2: Run test to verify it fails**

```bash
pytest entropy/tests/test_[function]_bits.py::test_[function]_bug_reproduction -v
```

Expected: FAIL showing wrong bit consumption

**Step 3: Fix the VLC decode logic**

```python
# In entropy/cavlc.py, fix the buggy code
# [Specific fix will depend on bug found]
```

**Step 4: Run test to verify fix**

```bash
pytest entropy/tests/test_[function]_bits.py::test_[function]_bug_reproduction -v
```

Expected: PASS

**Step 5: Run full CAVLC test suite**

```bash
pytest entropy/tests/test_cavlc.py -v
```

Expected: All tests pass

**Step 6: Test on failing videos**

```bash
python -c "
from decoder import H264Decoder

for video in ['motion_64x64.264', 'motion_simple.264']:
    try:
        decoder = H264Decoder()
        frames = list(decoder.decode_file(f'test_data/{video}'))
        print(f'{video}: SUCCESS - {len(frames)} frames')
    except Exception as e:
        print(f'{video}: FAILED - {e}')
"
```

Expected: Both videos decode successfully

**Step 7: Commit**

```bash
git add entropy/cavlc.py entropy/tests/test_[function]_bits.py
git commit -m "fix(cavlc): correct bit consumption in [function]

Fixes bit drift causing ue=2431 error in motion videos.

[Description of what was wrong and how it's fixed]

Closes #[issue-number]
"
```

---

### Task 8: Clean Up Validation Code (Optional)

**Files:**
- Modify: `entropy/cavlc.py`

**Decision Point:** Keep validation code as assertions (can be disabled with -O) or remove after bug is fixed?

**Option A: Keep as assertions**

```python
# Change validate_vlc_bits_consumed calls to:
assert reader.position - pos_before == expected_bits, \
    f"VLC bit consumption mismatch in {context}: ..."
```

**Option B: Remove validation**

Comment out or remove all `validate_vlc_bits_consumed` calls.

**Recommendation:** Keep as assertions - minimal runtime cost, catches future bugs

**Step 1: Convert to assertions (if Option A)**

```bash
# Search and replace in entropy/cavlc.py
sed -i 's/validate_vlc_bits_consumed(/assert reader.position - pos_before == expected_bits, /' entropy/cavlc.py
```

**Step 2: Test performance**

```bash
time python -c "
from decoder import H264Decoder
decoder = H264Decoder()
for i in range(10):
    list(decoder.decode_file('test_data/blue_64x64.264'))
"
```

**Step 3: Commit if keeping assertions**

```bash
git add entropy/cavlc.py
git commit -m "refactor(cavlc): convert VLC validation to assertions"
```

---

## Success Criteria

- [ ] All VLC decode functions have bit consumption validation
- [ ] motion_64x64.264 decodes successfully (or validation pinpoints exact bug location)
- [ ] motion_simple.264 decodes successfully
- [ ] Root cause of ue=2431 is identified and documented
- [ ] Bug is fixed with regression test
- [ ] All existing tests still pass
- [ ] Success rate improves from 79.3% (23/29) to 86.2% (25/29)

## Notes

- The ue=2431 value appeared at position 1288, so the bug occurs before that point
- Most likely culprits (in order of probability):
  1. **Level decode suffix_bits**: Complex logic with prefix>=15 cases
  2. **coeff_token with nC>=8**: Uses fixed 6-bit codes instead of VLC tables
  3. **total_zeros edge cases**: Different tables based on total_coeff value
- If validation doesn't catch bug, it means the VLC functions are correct but being called at wrong times or with wrong parameters

## References

- H.264 Spec Section 9.2: CAVLC decoding process
- H.264 Spec Section 9.2.1: coeff_token VLC tables
- H.264 Spec Section 9.2.2: Level decoding (suffix size rules)
- Investigation summary: docs/SESSION_SUMMARY.md
