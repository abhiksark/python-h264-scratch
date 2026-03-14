# Deblocking Filter Implementation Design

## Context

The decoder achieves pixel-perfect I-slice decoding when compared against ffmpeg
with `-skip_loop_filter all`. The remaining differences vs standard reference
output come from the unimplemented H.264 in-loop deblocking filter (Section 8.7).

Test results before deblocking:
- vs ffmpeg no-loop: max_diff=0 across all sizes (32-128px)
- vs ffmpeg with-loop: max_diff=3-54 depending on content/size

## Architecture

### Module Structure

```
deblock/
  __init__.py          (exists, empty)
  filter.py            (exists - low-level filter math, needs update)
  thresholds.py        (NEW - alpha/beta/tc0 tables from spec Table 8-16)
  boundary.py          (NEW - bS calculation per 4x4 block pair)
  deblock.py           (NEW - per-MB edge iteration + frame-level driver)
```

### Pipeline Integration

Deblocking runs after all MBs in a slice are reconstructed, before copying
to reference buffer:

```
decode MBs -> deblock frame (in-place) -> copy to DecodedFrame + ref buffer
```

This is "in-loop" because the reference buffer stores deblocked pixels,
affecting P/B-frame prediction.

## Block Numbering

Deblocking uses **spatial raster scan** (not coding scan order):

```
Spatial (deblocking):     Coding scan (entropy/coeffs):
 0  1  2  3                0  1  4  5
 4  5  6  7                2  3  6  7
 8  9 10 11                8  9 12 13
12 13 14 15               10 11 14 15
```

A mapping from spatial position to coding scan index is needed to look up
coefficient flags and intra modes from `DecoderState.mb_coeffs`.

## Edge Iteration (Section 8.7.1)

For each MB in raster scan order:

1. **4 vertical edges** (x=0,4,8,12), left-to-right
2. **4 horizontal edges** (y=0,4,8,12), top-to-bottom

Each edge has 4 "segments" (groups of 4 rows/columns), each with its own bS.

### Vertical Edge Block Pairs (spatial indices)

| Edge | x pos | Segment 0 | Segment 1 | Segment 2 | Segment 3 |
|------|-------|-----------|-----------|-----------|-----------|
| 0    | 0     | left_MB → 0 | left_MB → 4 | left_MB → 8 | left_MB → 12 |
| 1    | 4     | 0 → 1   | 4 → 5   | 8 → 9   | 12 → 13  |
| 2    | 8     | 1 → 2   | 5 → 6   | 9 → 10  | 13 → 14  |
| 3    | 12    | 2 → 3   | 6 → 7   | 10 → 11 | 14 → 15  |

### Horizontal Edge Block Pairs (spatial indices)

| Edge | y pos | Segment 0 | Segment 1 | Segment 2 | Segment 3 |
|------|-------|-----------|-----------|-----------|-----------|
| 0    | 0     | top_MB → 0 | top_MB → 1 | top_MB → 2 | top_MB → 3 |
| 1    | 4     | 0 → 4   | 1 → 5   | 2 → 6   | 3 → 7   |
| 2    | 8     | 4 → 8   | 5 → 9   | 6 → 10  | 7 → 11  |
| 3    | 12    | 8 → 12  | 9 → 13  | 10 → 14 | 11 → 15 |

## Boundary Strength (Section 8.7.2.2)

For each block pair (p, q) across an edge:

| Priority | Condition | bS |
|----------|-----------|-----|
| 1        | Either block is intra AND edge is MB boundary | 4 |
| 2        | Either block is intra (internal edge) | 3 |
| 3        | Either block has non-zero coded coefficients | 2 |
| 4        | Different reference pictures OR \|mv_x diff\| >= 4 OR \|mv_y diff\| >= 4 | 1 |
| 5        | None of the above | 0 |

For I-slices (all MBs are intra): bS is always 4 for MB boundaries, 3 for
internal edges. This simplifies the initial implementation.

## Threshold Tables (from JM ~/JM/ldecod/inc/loop_filter.h)

### Index Calculation

```python
qp_avg = (qp_p + qp_q + 1) >> 1
index_a = clip(0, 51, qp_avg + alpha_offset)
index_b = clip(0, 51, qp_avg + beta_offset)
alpha = ALPHA_TABLE[index_a]
beta  = BETA_TABLE[index_b]
tc0   = TC0_TABLE[index_a][bS - 1]   # bS 1-3 only
```

### ALPHA_TABLE[52]

```
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
4,4,5,6,7,8,9,10,12,13,15,17,
20,22,25,28,32,36,40,45,50,56,63,71,
80,90,101,113,127,144,162,182,203,226,255,255
```

### BETA_TABLE[52]

```
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
2,2,2,3,3,3,3,4,4,4,6,6,
7,7,8,8,9,9,10,10,11,11,12,12,
13,13,14,14,15,15,16,16,17,17,18,18
```

### TC0_TABLE[52][3] (indexed by [indexA][bS-1])

```
[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,1],
[0,0,1],[0,0,1],[0,0,1],[0,1,1],[0,1,1],[1,1,1],
[1,1,1],[1,1,1],[1,1,1],[1,1,2],[1,1,2],[1,1,2],
[1,1,2],[1,2,3],[1,2,3],[2,2,3],[2,2,4],[2,3,4],
[2,3,4],[3,3,5],[3,4,6],[3,4,6],[4,5,7],[4,5,8],
[4,6,9],[5,7,10],[6,8,11],[6,8,13],[7,10,14],
[8,11,16],[9,12,18],[10,13,20],[11,15,23],[13,17,25]
```

### QPC_TABLE[52] (luma QP -> chroma QP)

```
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
20,21,22,23,24,25,26,27,28,29,
29,30,31,32,32,33,34,34,35,35,36,36,37,37,37,38,
38,38,39,39,39,39
```

## Filter Formulas

### Filter Decision (per sample row/column)

Filter only if ALL of:
- bS > 0
- |p0 - q0| < alpha
- |p1 - p0| < beta
- |q1 - q0| < beta

### Normal Filter (bS = 1, 2, 3)

```
ap = (|p2 - p0| < beta)
aq = (|q2 - q0| < beta)
tc0 = TC0_TABLE[indexA][bS - 1]
tc = tc0 + ap + aq

delta = clip(-tc, tc, ((4*(q0-p0) + (p1-q1) + 4) >> 3))
p0' = clip(0, 255, p0 + delta)
q0' = clip(0, 255, q0 - delta)

if ap and tc0 > 0:
    p1' = p1 + clip(-tc0, tc0, (p2 + ((p0+q0+1)>>1) - 2*p1) >> 1)
if aq and tc0 > 0:
    r1' = q1 + clip(-tc0, tc0, (q2 + ((p0+q0+1)>>1) - 2*q1) >> 1)
```

### Strong Filter (bS = 4)

First check: `|p0 - q0| < (alpha >> 2) + 2`

**If true (strong mode):**

```
if |p2 - p0| < beta:   # filter 3 pixels on p side
    p0' = (p2 + 2*p1 + 2*p0 + 2*q0 + q1 + 4) >> 3
    p1' = (p2 + p1 + p0 + q0 + 2) >> 2
    p2' = (2*p3 + 3*p2 + p1 + p0 + q0 + 4) >> 3
else:                   # filter 1 pixel on p side
    p0' = (2*p1 + p0 + q1 + 2) >> 2

if |q2 - q0| < beta:   # filter 3 pixels on q side
    q0' = (p1 + 2*p0 + 2*q0 + 2*q1 + q2 + 4) >> 3
    q1' = (p0 + q0 + q1 + q2 + 2) >> 2
    q2' = (2*q3 + 3*q2 + q1 + q0 + p0 + 4) >> 3
else:                   # filter 1 pixel on q side
    q0' = (2*q1 + q0 + p1 + 2) >> 2
```

**If false (weak strong mode):**

```
p0' = (2*p1 + p0 + q1 + 2) >> 2
q0' = (2*q1 + q0 + p1 + 2) >> 2
```

### Chroma Filter

Same as luma normal filter but:
- Uses QPC (from QPC_TABLE) instead of luma QP for threshold lookup
- tc = tc0 + 1 (always, no ap/aq adjustment)
- Only p0 and q0 are modified (never p1/q1)
- For bS=4: same formula as normal filter (no strong mode for chroma)
- 2 vertical + 2 horizontal edges per 8x8 chroma MB (4:2:0)

## DecoderState Changes

Add `mb_qps` array to track per-MB QP:

```python
self.mb_qps = np.zeros(mb_count, dtype=np.int32)
```

Store in decode loop alongside `mb_types` and `mb_coeffs`.

## Implementation Steps

1. Create `deblock/thresholds.py` with tables and lookup functions
2. Create `deblock/boundary.py` with bS calculation
3. Rewrite `deblock/filter.py` with correct strong/normal/chroma formulas
4. Create `deblock/deblock.py` with per-MB edge iteration
5. Add `mb_qps` to DecoderState, store per-MB QP in all decode paths
6. Integrate `_deblock_frame` into decoder pipeline
7. Verify pixel-perfect match vs ffmpeg with loop filter enabled

## Verification

1. `pytest -v` - all existing tests pass (no regression)
2. All 6 test sizes (32-128px) match ffmpeg WITH loop filter (max_diff=0)
3. smptebars 128x128 matches ffmpeg WITH loop filter
