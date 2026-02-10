# h264/docs/REAL_VIDEO_PLAN.md
# Real-World Video Decoding Implementation Plan

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| I-frames (simple) | Working | Bit-perfect with JM reference |
| P-frames (simple) | Working | Basic motion compensation |
| B-frames | Partial | CABAC issues |
| CAVLC (complex content) | Broken | Desync after ~14 blocks |
| CABAC | Partial | CBP context fixed, residual issues |

**Root Cause of Failure**: CAVLC coefficient decoding loses sync on complex content (ffmpeg testsrc pattern). After 14 AC blocks, bitstream position is wrong.

---

## Phase 1: Fix CAVLC Coefficient Decoding (Critical)

**Priority**: HIGHEST - This blocks all real-world video decoding

### 1.1 Debug Neighbor Context (nC) Derivation
**File**: `entropy/cavlc.py`, `reconstruct/macroblock.py`

Currently `calculate_nC(None, None)` always returns 0, which selects `COEFF_TOKEN_DECODE_0` table. Real videos need proper neighbor-based nC calculation.

**H.264 Spec Reference**: Section 9.2.1

```python
# Current (broken)
ac_nC = calculate_nC(None, None)

# Required (per spec)
def calculate_nC(left_nz, top_nz, available_left, available_top):
    if available_left and available_top:
        nC = (left_nz + top_nz + 1) >> 1
    elif available_left:
        nC = left_nz
    elif available_top:
        nC = top_nz
    else:
        nC = 0
    return nC
```

**Implementation Steps**:
1. Store non-zero counts per 4x4 block in `DecoderState.nz_counts`
2. Pass neighbor availability flags to `decode_residual_block`
3. Calculate nC based on left/top block's non-zero count
4. Use correct coeff_token table based on nC

### 1.2 Verify CAVLC Level Decoding
**File**: `entropy/cavlc.py` (lines 200-274)

The level decoding may consume wrong number of bits for certain patterns. Compare against H.264 spec Table 9-7.

**Test**: Create bitstream with known level values, verify bit-exact decoding.

### 1.3 Verify run_before Decoding
**File**: `entropy/cavlc.py` (lines 331-370)

Check RUN_BEFORE_DECODE tables match H.264 spec Table 9-10.

### 1.4 Add CAVLC Conformance Tests
Create test cases using JM encoder output with:
- Complex I-frames (testsrc pattern)
- Varying QP values (5, 15, 26, 40)
- Edge case coefficient patterns

---

## Phase 2: Complete P-Frame Support

### 2.1 Sub-macroblock Partitions
**Files**: `inter/p_macroblock.py`, `inter/p_reconstruct.py`

Currently P_8x8 sub-partitions (8x4, 4x8, 4x4) may not be fully implemented.

**Test**: P-frames with complex motion (sports, action scenes)

### 2.2 Motion Vector Prediction Edge Cases
**File**: `inter/mv_prediction.py`

- Verify median prediction for unavailable neighbors
- Check MV range clipping per level

### 2.3 Reference Frame Management
**File**: `inter/reference.py`

- Multiple reference frames (max_num_ref_frames > 1)
- Reference picture list reordering (rarely used but required)

---

## Phase 3: B-Frame Support

### 3.1 B-Slice Reference List Building
**Files**: `decoder/decoder.py`, `inter/b_macroblock.py`

- L0 list: forward references
- L1 list: backward references
- Combined list for bi-prediction

### 3.2 Direct Mode
**File**: `inter/direct_mode.py`

- Temporal direct mode (most common)
- Spatial direct mode

### 3.3 Bi-Prediction Weighting
- Default weights (1, 1)
- Implicit weights based on POC distance

---

## Phase 4: CABAC Completion

### 4.1 Full mb_type/sub_mb_type Parsing
**File**: `entropy/cabac_macroblock.py`

Currently 49 xfailed tests for CABAC macroblock syntax.

### 4.2 Residual Block Decoding
**File**: `entropy/cabac_residual.py`

- significance_map decoding
- coeff_abs_level_minus1 decoding
- coeff_sign_flag decoding

### 4.3 Context Model Verification
**File**: `entropy/cabac_context.py`

Verify all 460 context models match H.264 spec Table 9-11 through 9-23.

---

## Phase 5: Robustness & Compatibility

### 5.1 Error Resilience
**File**: `decoder/error_concealment.py`

- Skip corrupted MBs
- Copy from previous frame or neighbors
- Continue decoding after errors

### 5.2 Multiple PPS Support
**File**: `decoder/decoder.py`

Currently assumes PPS ID 0. Real streams may have multiple.

```python
# Current
pps = self.state.get_pps(0)

# Required
pps = self.state.get_pps(slice_header.pic_parameter_set_id)
```

### 5.3 SEI Message Handling
**File**: `parameters/sei.py`

- Parse SEI NAL units (currently ignored)
- Handle buffering_period, pic_timing for A/V sync

---

## Phase 6: Performance Optimization (Optional)

### 6.1 Vectorized Transform
Use NumPy broadcasting for batch IDCT.

### 6.2 Frame Caching
Avoid redundant allocations.

### 6.3 Parallel MB Decoding
Process independent MBs concurrently (limited by dependencies).

---

## Implementation Order (Recommended)

```
Week 1-2: Phase 1 (CAVLC fix) - CRITICAL PATH
Week 3:   Phase 2 (P-frame completion)
Week 4:   Phase 3 (B-frames)
Week 5-6: Phase 4 (CABAC)
Week 7:   Phase 5 (Robustness)
```

---

## Testing Strategy

### Unit Tests
- Each CAVLC syntax element
- Each CABAC context model
- Each prediction mode

### Integration Tests
- JM encoder reference bitstreams
- ffmpeg baseline profile output
- Real-world samples (Big Buck Bunny, Sintel)

### Conformance Tests
- ITU-T H.264 conformance streams (if available)
- Compare frame PSNR against reference decoder

---

## Quick Wins (Can Do Now)

1. **Fix nC calculation** - 2-3 hours, high impact
2. **Add multiple PPS support** - 1 hour, prevents crashes
3. **Enable error resilience** - Already have `error_concealment.py`, just wire it up
4. **Remove xfail markers** - 294 tests now passing but still marked xfail

---

## Files to Modify (Priority Order)

| File | Changes | Impact |
|------|---------|--------|
| `entropy/cavlc.py` | Fix nC, verify tables | Critical |
| `reconstruct/macroblock.py` | Pass neighbor info | Critical |
| `decoder/decoder.py` | nz_counts tracking, multi-PPS | High |
| `inter/mv_prediction.py` | Edge cases | Medium |
| `entropy/cabac_residual.py` | Complete residual | Medium |
| `decoder/error_concealment.py` | Wire up | Medium |

---

## Validation Checkpoints

### Checkpoint 1: CAVLC Fixed
- [ ] ffmpeg testsrc 64x64 decodes without errors
- [ ] All existing tests still pass
- [ ] nC values match spec for known patterns

### Checkpoint 2: P-Frames Complete
- [ ] Motion sequences decode correctly
- [ ] PSNR > 40dB vs reference

### Checkpoint 3: B-Frames Working
- [ ] GOP with B-frames decodes
- [ ] POC ordering correct

### Checkpoint 4: Real Video
- [ ] Big Buck Bunny 360p baseline decodes
- [ ] YouTube-dl sample decodes
- [ ] Mobile phone recording decodes
