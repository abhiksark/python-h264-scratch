# Fix: Deblocking Filter bS=1 MV/Ref Check

## Problem
`deblock/deblock.py:_get_bs_for_edge()` returns bS=0 for all inter-inter boundaries
without coded coefficients. Per H.264 Section 8.7.2.1, it should return bS=1 when
adjacent blocks have different MVs or different reference indices.

## Evidence
- Pre-deblock output is pixel-perfect (verified against ffmpeg -skip_loop_filter all)
- ffmpeg deblocking changes 2282-11317 pixels per frame
- Our deblocking works for intra edges (bS=3/4) but missing bS=1 for inter edges
- First error appears at frame 2 (first frame with inter MBs), max_diff=2, then drifts

## Implementation Plan

### Task 1: Add MV cache parameter to deblock_frame_inplace
- Add `mv_cache` parameter to `deblock_frame_inplace()` signature
- Pass it through from `decoder.py:_deblock_frame()`
- No behavior change yet

### Task 2: Implement bS=1 MV/ref check in _get_bs_for_edge
- Add MV and ref_idx parameters to `_get_bs_for_edge()`
- For inter-inter boundaries without coefficients:
  - bS=1 if |mvx_p - mvx_q| >= 4 OR |mvy_p - mvy_q| >= 4 OR different ref_idx
  - bS=0 otherwise
- Update all call sites in `deblock_frame_inplace()` to pass MV/ref data
- For inter blocks: look up MVs from mv_cache using (mb_x, mb_y, block_x, block_y)
- For intra blocks: no change (already handled)

### Task 3: Verify with test streams
- Run all unit tests (expect 1822 pass)
- Verify all 8 test streams (smpte, testsrc10, mandel, mptest, cellauto, sierpinski, mandel320, life)
- All should be pixel-perfect or significantly improved

## bS Decision Table (H.264 Section 8.7.2.1)
| Condition | bS |
|-----------|-----|
| p or q intra, MB boundary | 4 |
| p or q intra, not MB boundary | 3 |
| p or q has coded coefficients | 2 |
| Different ref frames OR |mvx diff|>=4 OR |mvy diff|>=4 | 1 |
| Same ref, similar MVs | 0 |
