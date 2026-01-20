# Implementation Progress

## Current Status

**Working:** I-frame only decoder for Baseline profile (I_16x16 macroblocks)
**Tests:** 423 passing

## What Works

| Feature | Status | Notes |
|---------|--------|-------|
| NAL unit parsing | ✅ | Start codes, emulation prevention |
| SPS/PPS parsing | ✅ | Baseline through High profile syntax |
| Slice header parsing | ✅ | I-slices fully supported |
| CAVLC entropy decoding | ✅ | All VLC tables |
| Inverse quantization | ✅ | QP 0-51, DC/AC scaling |
| 4x4 Integer IDCT | ✅ | Spec-compliant |
| Hadamard transforms | ✅ | 4x4 (luma DC), 2x2 (chroma DC) |
| I_16x16 prediction | ✅ | All 4 modes (V, H, DC, Plane) |
| Chroma prediction | ✅ | All 4 modes |
| YCbCr → RGB | ✅ | BT.601 and BT.709 |
| Multi-MB frames | ✅ | Neighbor pixel handling |

## What Does NOT Work

| Feature | Status | Notes |
|---------|--------|-------|
| I_4x4 macroblocks | ❌ | 9 prediction modes not implemented |
| I_8x8 macroblocks | ❌ | High profile only |
| P-frames | ❌ | Motion compensation not implemented |
| B-frames | ❌ | Bi-directional prediction not implemented |
| CABAC entropy | ❌ | Main/High profile only |
| Deblocking filter | ❌ | Post-processing not implemented |
| Multiple slices | ⚠️ | Untested |
| Multiple frames | ⚠️ | Only first frame decoded |
| Interlaced video | ❌ | Frame-only |

## Known Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| Gradient DC coefficients | Medium | Complex horizontal gradients decode at ~33% accuracy. DC coefficient scan order issue for I_16x16 blocks with internal variation. Simple uniform blocks work 100%. |

## Module Status

| Module | Status | Key Functions |
|--------|--------|---------------|
| bitstream/ | ✅ Complete | `extract_nal_units`, `BitReader`, `NumpyBitReader` |
| parameters/ | ✅ Complete | `parse_sps`, `parse_pps` |
| slice/ | ✅ Complete | `parse_slice_header` |
| entropy/ | ✅ Complete | `decode_residual_block`, `decode_coeff_token` |
| dequant/ | ✅ Complete | `dequant_4x4`, `dequant_dc_4x4` |
| transform/ | ✅ Complete | `idct_4x4`, `hadamard_4x4`, `hadamard_2x2` |
| intra/ | 🟡 Partial | `predict_intra_16x16` (I_4x4 missing) |
| inter/ | ❌ Not Started | Motion compensation |
| reconstruct/ | 🟡 Partial | `reconstruct_i16x16_luma` (I_4x4 missing) |
| color/ | ✅ Complete | `ycbcr_to_rgb`, `upsample_420` |
| decoder/ | 🟡 Partial | `decode_h264_bytes` (I-frames only) |
| test_utils/ | ✅ Complete | `compare_with_jm`, `load_yuv_420` |

## Decoder Accuracy

Tested against JM reference decoder:

| Test Video | Resolution | Match | Notes |
|------------|------------|-------|-------|
| quadrant_32x32 | 32×32 | ✅ 100% | 4 uniform regions |
| double_gray_32x16 | 32×16 | ✅ 100% | 2 uniform regions |
| gradient_128x128 | 128×128 | ❌ 33% | Complex gradient |

## Implementation Phases

### ✅ Phase 1: Core Components
- [x] NAL parsing
- [x] SPS/PPS parsing
- [x] Slice header parsing
- [x] CAVLC decoding
- [x] Dequantization
- [x] IDCT transforms
- [x] I_16x16 prediction
- [x] Color conversion

### 🟡 Phase 2: Complete I-frame Support
- [ ] I_4x4 prediction (9 modes)
- [ ] Fix gradient DC coefficient issue
- [ ] I_PCM macroblocks

### ❌ Phase 3: P-frame Support
- [ ] Motion vector prediction
- [ ] Motion compensation
- [ ] Reference frame buffer
- [ ] Inter prediction modes

### ❌ Phase 4: Polish
- [ ] Deblocking filter
- [ ] Multiple slice support
- [ ] Streaming decoder API
- [ ] Error resilience

## Test Breakdown

```
bitstream/     106 tests
parameters/     45 tests
slice/          28 tests
entropy/        35 tests
dequant/        25 tests
transform/      26 tests
intra/          33 tests
reconstruct/    34 tests
color/          15 tests
decoder/        27 tests (includes JM comparison)
test_utils/      9 tests
─────────────────────────
Total:         423 tests
```
