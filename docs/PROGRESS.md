# Implementation Progress

## Current Status — v0.2

**Working:** Pixel-perfect I/P/B-frame decoder with CAVLC, CABAC, High Profile I_8x8, B-pyramid, and CABAC weighted prediction.
**Tests:** 1,833 passing
**Verified on:** 4 real internet videos + generated multi-frame streams (bframes=3, ref=3)

### v0.2 Changes (from v0.1)
- **fix(b-frame):** Correct collocated MV storage for L1-only partitions — fixes B-pyramid with bframes>=3
- **fix(cabac):** Add weighted prediction for CABAC P-slices — fixes ref>2 decode errors
- **test:** Multi-frame regression tests (B-pyramid, P-only, BBB 300-frame)

## What Works

| Feature | Status | Notes |
|---------|--------|-------|
| NAL unit parsing | Done | Start codes, emulation prevention |
| SPS/PPS parsing | Done | Baseline through High profile |
| Slice header parsing | Done | I, P, B slices |
| CAVLC entropy decoding | Done | All VLC tables |
| CABAC entropy decoding | Done | Context models, arithmetic decoder, all syntax elements |
| Inverse quantization | Done | QP 0-51, DC/AC scaling, 4x4 and 8x8, scaling lists |
| 4x4 Integer IDCT | Done | Spec-compliant |
| 8x8 Integer IDCT | Done | Matches JM reference exactly |
| Hadamard transforms | Done | 4x4 (luma DC), 2x2 (chroma DC) |
| I_16x16 prediction | Done | All 4 modes |
| I_4x4 prediction | Done | All 9 modes |
| I_8x8 prediction | Done | All 9 modes + lowpass filtering |
| Chroma prediction | Done | All 4 modes |
| YCbCr to RGB | Done | BT.601 and BT.709 |
| Reference frame buffer | Done | DPB, L0/L1 lists, POC-based ordering |
| Motion compensation | Done | 6-tap luma + bilinear chroma interpolation |
| MV prediction | Done | Spatial median, all partition sizes |
| P-frame (all types) | Done | Skip, 16x16, 16x8, 8x16, 8x8, sub-partitions |
| B-frame (all types) | Done | L0, L1, Bi, Direct (spatial + temporal) |
| Weighted prediction | Done | Implicit weights, explicit P/B-frame weights (CAVLC + CABAC) |
| Deblocking filter | Done | Boundary strength, adaptive filtering |
| Scaling lists | Done | 4x4/8x8, flat + custom |
| MP4 demuxer | Done | avcC SPS/PPS extraction, auto-detection |

## Known Gaps

| Feature | Status | Notes |
|---------|--------|-------|
| High Profile inter 8x8 | Done | P and B frames pixel-perfect with 8x8 transform |
| CABAC at QP < 6 | Edge case | Sintel at QP=5 triggers arithmetic desync |
| weightp=2 with ref>2 | Edge case | x264 duplicate-ref weighted prediction mode |
| I_PCM macroblocks | Partial | Rare in practice |
| Multiple slices (FMO/ASO) | Partial | Untested |
| Interlaced (MBAFF) | No | Frame-only |

## Module Status

| Module | Files | Description |
|--------|-------|-------------|
| bitstream/ | 3 | NAL parsing, bit-level I/O |
| parameters/ | 3 | SPS, PPS, scaling lists |
| slice/ | 4 | Slice header, weight tables, FMO, ASO |
| entropy/ | 7 | CAVLC + CABAC (contexts, arithmetic, binarization, syntax, residual) |
| dequant/ | 2 | Inverse quantization, chroma QP mapping |
| transform/ | 4 | 4x4 IDCT, 8x8 IDCT, Hadamard, transform size selection |
| intra/ | 4 | I_16x16, I_4x4, I_8x8, chroma prediction |
| inter/ | 9 | Motion comp, MV prediction, bipred, direct mode, weighted pred |
| deblock/ | 4 | Boundary strength, thresholds, filter |
| reconstruct/ | 2 | Macroblock reconstruction, MB state |
| color/ | 2 | YCbCr to RGB, chroma format handling |
| container/ | 1 | MP4 demuxer |
| decoder/ | 6 | Main decoder, I_8x8, POC, MMCO, error concealment, frame output |

## Test Breakdown

```
Total:       1,833 passing
Skipped:        96 (missing test data)
xfailed:       317 (TDD red tests for future features)
xpassed:       444 (features completed ahead of tests)
```
