# Fix CAVLC Level Code Formula Bug

## Problem Summary

**Issue:** 2/29 test videos fail with CAVLC run_before errors:
- `test_data/motion_64x64.264`: "Invalid run_before: no match found"
- `test_data/motion_cavlc.264`: "Invalid run_before: 8 exceeds zeros_left 7"

**Root Cause:** Bit drift from incorrect level_code calculation in CAVLC level decoding. When `level_prefix >= 15` AND `suffix_length > 0`, the code uses `level_prefix` but H.264 spec requires capping at `15`.

**Impact:**
- ✅ Simple content works (22/29 videos decode successfully)
- ❌ Complex content fails (motion/detail triggers large coefficients)
- ❌ Bit drift causes subsequent run_before decode to read from wrong position

---

## Root Cause Analysis

### CAVLC Decode Order
1. coeff_token → TotalCoeff, TrailingOnes
2. trailing_ones_sign_flag
3. **level values** ← BUG IS HERE
4. total_zeros
5. run_before ← Fails due to bit drift from step 3

### The Bug

**Location:** `entropy/cavlc.py` lines 269-272

**Current Code (WRONG):**
```python
if suffix_length == 0:
    level_code = level_prefix + level_suffix
else:
    level_code = (level_prefix << suffix_length) + level_suffix  # BUG!
```

**H.264 Spec Section 9.2.2:**
```
if( level_prefix < 15 )
    level_code = (level_prefix << suffixLength) + level_suffix
else
    if( suffixLength == 0 )
        level_code = level_prefix + level_suffix
    else
        level_code = (15 << suffixLength) + level_suffix  // CAP at 15
```

**Example of Bug:**
- `level_prefix = 16, suffix_length = 2, level_suffix = 3`
- Current (WRONG): `(16 << 2) + 3 = 64 + 3 = 67`
- Correct (SPEC): `(15 << 2) + 3 = 60 + 3 = 63`
- Difference: 4 units → incorrect coefficient magnitude → affects suffix_length update → cumulative bit drift

---

## The Fix

### Step 1: Correct the level_code Formula

**File:** `entropy/cavlc.py` lines 265-272

**Before:**
```python
# H.264 Spec 9.2.2: Compute level_code
# Simplified formula per FFmpeg reference:
# - suffix_length == 0: level_code = prefix + suffix
# - suffix_length > 0: level_code = (prefix << suffix_length) + suffix
if suffix_length == 0:
    level_code = level_prefix + level_suffix
else:
    level_code = (level_prefix << suffix_length) + level_suffix
```

**After:**
```python
# H.264 Spec 9.2.2 Table 9-7: Compute level_code
# Three cases based on level_prefix and suffix_length:
if level_prefix < 15:
    # Standard case: use level_prefix directly
    level_code = (level_prefix << suffix_length) + level_suffix
elif suffix_length == 0:
    # Extended escape without suffix
    level_code = level_prefix + level_suffix
else:
    # Extended escape with suffix: CAP level_prefix at 15
    level_code = (15 << suffix_length) + level_suffix
```

**Why the Change:**
- Current code simplifies by branching on suffix_length only
- Spec requires checking BOTH level_prefix AND suffix_length
- For level_prefix >= 15 with suffix > 0, must cap at 15 to prevent overflow
- This prevents incorrect level magnitudes that cause bit drift

---

## Testing Strategy

### Test File: `entropy/tests/test_cavlc_level_code.py` (NEW)

**Test 1: Standard Case (level_prefix < 15)**
```python
def test_level_code_standard_case():
    """Test standard case: level_prefix < 15."""
    # Test various suffix_length values
    test_cases = [
        # (level_prefix, suffix_length, level_suffix, expected_level_code)
        (5, 0, 0, 5),          # No suffix
        (5, 2, 3, 23),         # (5 << 2) + 3 = 20 + 3 = 23
        (10, 1, 1, 21),        # (10 << 1) + 1 = 20 + 1 = 21
        (14, 3, 7, 119),       # (14 << 3) + 7 = 112 + 7 = 119
    ]
    for prefix, suffix_len, suffix, expected in test_cases:
        result = _compute_level_code(prefix, suffix_len, suffix)
        assert result == expected
```

**Test 2: Escape Without Suffix (level_prefix >= 15, suffix_length == 0)**
```python
def test_level_code_escape_no_suffix():
    """Test escape code without suffix: level_prefix >= 15, suffix_length == 0."""
    test_cases = [
        (15, 0, 0, 15),        # 15 + 0 = 15
        (16, 0, 5, 21),        # 16 + 5 = 21
        (20, 0, 10, 30),       # 20 + 10 = 30
    ]
    for prefix, suffix_len, suffix, expected in test_cases:
        result = _compute_level_code(prefix, suffix_len, suffix)
        assert result == expected
```

**Test 3: Escape With Suffix - THE BUG CASE**
```python
def test_level_code_escape_with_suffix():
    """Test escape code WITH suffix: level_prefix >= 15, suffix_length > 0.

    This is the bug case - must cap level_prefix at 15.
    """
    test_cases = [
        # (level_prefix, suffix_length, level_suffix, expected_level_code)
        (15, 1, 0, 30),        # (15 << 1) + 0 = 30
        (15, 2, 3, 63),        # (15 << 2) + 3 = 60 + 3 = 63
        (16, 1, 0, 30),        # (15 << 1) + 0 = 30 (CAPPED!)
        (16, 2, 3, 63),        # (15 << 2) + 3 = 63 (CAPPED!)
        (20, 3, 7, 127),       # (15 << 3) + 7 = 120 + 7 = 127 (CAPPED!)
    ]
    for prefix, suffix_len, suffix, expected in test_cases:
        result = _compute_level_code(prefix, suffix_len, suffix)
        assert result == expected

        # Verify current code would give WRONG answer for prefix >= 15
        if prefix >= 15 and suffix_len > 0:
            wrong_result = (prefix << suffix_len) + suffix
            assert result != wrong_result, \
                f"Bug: level_prefix={prefix} not capped at 15"
```

**Test 4: Boundary Cases**
```python
def test_level_code_boundary_at_15():
    """Test boundary between standard and escape formulas."""
    # level_prefix = 14 uses standard formula
    result_14 = _compute_level_code(14, 2, 3)  # (14 << 2) + 3 = 59
    assert result_14 == 59

    # level_prefix = 15 uses capped formula
    result_15 = _compute_level_code(15, 2, 3)  # (15 << 2) + 3 = 63
    assert result_15 == 63

    # Verify they're different
    assert result_15 != result_14
```

**Test 5: Integration Test**
```python
def test_decode_motion_video_after_fix():
    """Verify motion videos decode successfully after fix."""
    from decoder.decoder import H264Decoder

    # These videos currently fail with run_before errors
    test_videos = [
        'test_data/motion_64x64.264',
        'test_data/motion_cavlc.264',
    ]

    for video_path in test_videos:
        decoder = H264Decoder()
        frames = list(decoder.decode_file(video_path))
        assert len(frames) > 0, f"Failed to decode {video_path}"
```

### Test Helper Function

Extract level_code calculation into testable function:

```python
def _compute_level_code(level_prefix: int, suffix_length: int, level_suffix: int) -> int:
    """Compute level_code per H.264 spec.

    Args:
        level_prefix: Number of leading zeros in level VLC
        suffix_length: Current suffix length state
        level_suffix: Decoded suffix bits

    Returns:
        level_code value per H.264 Table 9-7
    """
    if level_prefix < 15:
        return (level_prefix << suffix_length) + level_suffix
    elif suffix_length == 0:
        return level_prefix + level_suffix
    else:
        return (15 << suffix_length) + level_suffix
```

---

## Implementation Steps

### Step 1: Create Test File
- **File:** `entropy/tests/test_cavlc_level_code.py`
- Write all 5 test cases
- Run tests → should FAIL on test 3 (escape with suffix)

### Step 2: Extract level_code Calculation
- **File:** `entropy/cavlc.py`
- Create `_compute_level_code()` helper function
- Replace inline calculation with function call
- Run tests → should still FAIL

### Step 3: Fix the Formula
- **File:** `entropy/cavlc.py` in `_compute_level_code()`
- Change from 2-way branch to 3-way branch
- Implement correct H.264 spec logic
- Run tests → should PASS

### Step 4: Verify Integration
- Run full test suite: `pytest -v`
- Test motion videos manually
- Check that 2 previously failing videos now decode

### Step 5: Commit
```bash
git add entropy/cavlc.py entropy/tests/test_cavlc_level_code.py
git commit -m "fix(cavlc): correct level_code formula for level_prefix >= 15

Per H.264 spec Table 9-7, when level_prefix >= 15 and suffix_length > 0,
level_code must cap level_prefix at 15: (15 << suffix_length) + level_suffix.

Previous code used level_prefix directly, causing incorrect coefficient
magnitudes in complex content (motion/detail), leading to bit drift that
manifested as run_before decode errors.

Fixes:
- test_data/motion_64x64.264 (was: Invalid run_before: no match found)
- test_data/motion_cavlc.264 (was: Invalid run_before: 8 exceeds zeros_left 7)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Verification Criteria

### Before Fix
- ❌ `pytest entropy/tests/test_cavlc_level_code.py::test_level_code_escape_with_suffix` FAILS
- ❌ `test_data/motion_64x64.264` fails with "Invalid run_before: no match found"
- ❌ `test_data/motion_cavlc.264` fails with "Invalid run_before: 8 exceeds zeros_left 7"
- ✅ 22/29 test videos decode successfully (76%)

### After Fix
- ✅ All tests in `test_cavlc_level_code.py` PASS
- ✅ `test_data/motion_64x64.264` decodes successfully
- ✅ `test_data/motion_cavlc.264` decodes successfully
- ✅ 24/29 test videos decode successfully (83%) ← +2 videos fixed
- ✅ No regressions in previously working videos

---

## H.264 Specification Reference

**Section 9.2.2 - Parsing process for level information**

```c
// Determine suffix size
if( level_prefix < 14 )
    levelSuffixSize = suffixLength
else if( level_prefix == 14 )
    levelSuffixSize = ( suffixLength == 0 ) ? 4 : suffixLength
else  // level_prefix >= 15
    levelSuffixSize = ( suffixLength == 0 ) ? level_prefix - 3 : suffixLength

// Read suffix bits
if( levelSuffixSize > 0 )
    level_suffix = u(levelSuffixSize)
else
    level_suffix = 0

// Compute level_code (Table 9-7)
if( level_prefix < 15 )
    level_code = ( level_prefix << suffixLength ) + level_suffix
else
    if( suffixLength == 0 )
        level_code = level_prefix + level_suffix
    else
        level_code = ( 15 << suffixLength ) + level_suffix  // ← CAP AT 15
```

**Key Insight:** The cap at 15 prevents level_code from growing unbounded when large level_prefix values (escape codes) are used with non-zero suffix_length. This ensures coefficient magnitudes stay within expected ranges.

---

## Related Issues

This fix addresses the bit drift issue tracked in task #3:
- Task #3: "Fix bit drift issue causing failures on 48x48+ videos"

After this fix, the bit drift should be resolved for CAVLC decoding, allowing larger and more complex videos to decode successfully.

---

## Future Enhancements

After fixing this bug, consider:

1. **Verify against FFmpeg:** Cross-check level decoding with FFmpeg's CAVLC implementation
2. **Add fuzzing:** Generate test cases with random coefficient patterns to catch edge cases
3. **Performance profiling:** Ensure the 3-way branch doesn't impact decode performance
4. **Remaining failures:** Investigate the 5 other failing videos (MB type parsing, end-of-stream handling)
