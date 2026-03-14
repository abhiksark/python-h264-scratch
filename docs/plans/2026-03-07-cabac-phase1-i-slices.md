# CABAC Phase 1: I-Slice Decoding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decode I-slices using CABAC entropy coding, achieving pixel-perfect output against ffmpeg for Main profile streams.

**Architecture:** Replace CAVLC bitstream reading with CABAC decoding while reusing the existing reconstruction pipeline. The existing CABAC infrastructure (arithmetic decoder, binarization, syntax elements, residual blocks) is complete but untested end-to-end. The main work is: (1) fixing the context initialization tables, (2) refactoring `reconstruct_i16x16_luma` to accept pre-parsed coefficients, and (3) wiring the CABAC decode path into `decoder.py`.

**Tech Stack:** Python, NumPy, H.264 Spec Sections 7.3.5, 9.3

---

## Background

The decoder has 6 CABAC modules in `entropy/`:
- `cabac_arith.py` — Binary arithmetic decoder (complete, correct per spec)
- `cabac_binarize.py` — Binarization schemes (unary, TU, UEGk, FL)
- `cabac_syntax.py` — Syntax element decoding (mb_type, cbp, mvd, ref_idx, qp_delta)
- `cabac_residual.py` — Residual block decoding (significance map + levels)
- `cabac_macroblock.py` — MB-level decode returning structured dict
- `cabac_context.py` — Context init (460 models, but only 64 have real values)

The integration stubs in `decoder/decoder.py`:
- `_decode_slice_data_cabac` (line 3696) — Loops over MBs, decodes skip + mb_type only
- `_decode_macroblock_cabac` (line 3656) — Decodes mb_type only, has TODO for everything else

## Critical Issue: Context Init Tables

`cabac_context.py` has real (m,n) init values for contexts 0-63 only. Contexts 64-459 use `_DEFAULT_MN = (0, 64)` which gives `pStateIdx=0, valMPS=1` — essentially random. This affects:
- ctxIdx 64-72: intra_chroma_pred_mode, prev_intra_pred_flag
- ctxIdx 73-84: coded_block_pattern
- ctxIdx 85-104: coded_block_flag (per-block coded flag)
- ctxIdx 105-165: significant_coeff_flag (residual significance map)
- ctxIdx 166-226: last_significant_coeff_flag (residual last position)
- ctxIdx 227-275: coeff_abs_level_minus1 (coefficient magnitudes)

Without correct init values, EVERY CABAC decode after mb_type will produce wrong results.

---

### Task 1: Generate CABAC I-only test stream

We need a Main profile I-only CABAC stream with reference YUV for pixel-perfect comparison.

**Step 1: Generate test stream**

Run:
```bash
ffmpeg -f lavfi -i testsrc2=size=176x144:n=5 \
  -c:v libx264 -profile:v main -x264-params "bframes=0:no-cabac=0" \
  -intra -qp 28 -pix_fmt yuv420p \
  test_data/testsrc2_cabac_intra.264

ffmpeg -i test_data/testsrc2_cabac_intra.264 \
  -f rawvideo -pix_fmt yuv420p \
  test_data/testsrc2_cabac_intra_ref.yuv
```

**Step 2: Verify it's actually CABAC**

Run:
```bash
ffprobe -show_entries stream=profile test_data/testsrc2_cabac_intra.264 2>/dev/null
```
Expected: `profile=Main` (Baseline = CAVLC only, Main = CABAC)

**Step 3: Verify it's I-only**

Run:
```bash
ffprobe -show_frames -select_streams v -show_entries frame=pict_type test_data/testsrc2_cabac_intra.264 2>/dev/null | grep pict_type | sort | uniq -c
```
Expected: All frames are `pict_type=I`

**Step 4: Commit**

```bash
git add test_data/testsrc2_cabac_intra.264 test_data/testsrc2_cabac_intra_ref.yuv
git commit -m "test: add CABAC I-only Main profile test stream"
```

---

### Task 2: Extract full CABAC context init tables from ffmpeg

**Files:**
- Modify: `entropy/cabac_context.py`

The ffmpeg source `libavcodec/h264_cabac.c` contains the complete (m,n) tables for all 460 contexts. These are in arrays `ff_h264_cabac_context_init_I` and `ff_h264_cabac_context_init_PB[3]`.

**Step 1: Fetch the tables from ffmpeg source**

Use web search or fetch to get the complete tables from:
- `https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavcodec/h264_cabac.c`
- Or the H.264 spec Tables 9-12 through 9-23

The tables have format `{ m, n }` for each ctxIdx (0-459 for I, 0-459×3 for P/B with cabac_init_idc 0/1/2).

**Step 2: Replace placeholder tables in `cabac_context.py`**

Replace `_INIT_I`, `_INIT_P`, `_INIT_B` with complete 460-entry lists. Also add the 3 `cabac_init_idc` variants for P and B slices.

Current code at line 95:
```python
_INIT_I = [
    # Only 14 real entries, rest is _DEFAULT_MN
    ...
] + [_DEFAULT_MN] * (NUM_CONTEXTS - 14)
```

Replace with:
```python
_INIT_I = [
    # ctxIdx 0-2: mb_type SI (Table 9-12)
    (20, -15), (2, 54), (3, 74),
    # ctxIdx 3-10: mb_type I (Table 9-12)
    (20, -15), (2, 54), (3, 74), (-28, 127), (-23, 104),
    (-6, 53), (-1, 54), (7, 51),
    # ... ALL 460 entries from spec/ffmpeg ...
]
```

Do the same for `_INIT_P` and `_INIT_B`, and add `_INIT_PB_IDC0`, `_INIT_PB_IDC1`, `_INIT_PB_IDC2` if the tables differ per `cabac_init_idc`.

**Step 3: Fix `_get_init_params_with_idc` to use real tables**

Current code (line 245) applies fake modifications (`m - 2`, `m + 2`). Replace with actual table selection:
```python
def _get_init_params_with_idc(slice_type, cabac_init_idc):
    if slice_type == 2:  # I-slice
        return _INIT_I
    # Select from 3 real tables based on cabac_init_idc
    tables = [_INIT_PB_IDC0, _INIT_PB_IDC1, _INIT_PB_IDC2]
    return tables[cabac_init_idc]
```

**Step 4: Write a test to verify context initialization**

Create `entropy/tests/test_cabac_context_init.py`:
```python
def test_context_init_i_slice_qp26():
    """Verify context initialization produces correct pStateIdx/valMPS for known QP."""
    from entropy.cabac_context import init_context_models
    contexts = init_context_models(slice_type=2, slice_qp=26)
    assert len(contexts) == 460
    # Spot-check: context 0 (mb_type SI) with m=20, n=-15
    # preCtxState = Clip3(1, 126, (20*26)>>4 + (-15)) = Clip3(1,126, 32-15) = 17
    # valMPS = 0 (17 < 64), pStateIdx = 63 - 17 = 46
    assert contexts[0].pStateIdx == 46
    assert contexts[0].valMPS == 0

def test_all_460_contexts_initialized():
    """Ensure no context uses default (0, 64) placeholder."""
    from entropy.cabac_context import _INIT_I
    for i, (m, n) in enumerate(_INIT_I):
        assert not (m == 0 and n == 64), f"Context {i} still has default (0,64)"
```

**Step 5: Run tests**

Run: `pytest entropy/tests/test_cabac_context_init.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add entropy/cabac_context.py entropy/tests/test_cabac_context_init.py
git commit -m "feat(cabac): add complete context init tables from H.264 spec"
```

---

### Task 3: Refactor `reconstruct_i16x16_luma` to separate parse from reconstruct

**Files:**
- Modify: `reconstruct/macroblock.py:1338-1477`

Currently `reconstruct_i16x16_luma` takes a `BitReader` and reads residual coefficients inline. We need to extract the reconstruction logic so it can accept pre-parsed coefficients from either CAVLC or CABAC.

**Step 1: Create `reconstruct_i16x16_from_coeffs` function**

Add this new function right before the existing `reconstruct_i16x16_luma` (before line 1338):

```python
def reconstruct_i16x16_from_coeffs(
    pred_mode: int,
    cbp_luma: int,
    qp: int,
    dc_coeffs: np.ndarray,
    ac_coeffs_list: List[Optional[np.ndarray]],
    neighbors_top: Optional[np.ndarray],
    neighbors_left: Optional[np.ndarray],
    neighbor_top_left: Optional[int],
    nz_counts: np.ndarray,
) -> np.ndarray:
    """Reconstruct I_16x16 luma from pre-parsed coefficients.

    This is the entropy-coding-independent reconstruction core.

    Args:
        pred_mode: Intra 16x16 prediction mode (0-3)
        cbp_luma: Luma CBP (0 or 15)
        qp: Quantization parameter
        dc_coeffs: 16-element array of DC coefficients (scan order)
        ac_coeffs_list: List of 16 arrays, each 15-element AC coefficients
                        (None for uncoded blocks, i.e., 8x8 region not in CBP)
        neighbors_top: Top neighbor pixels (16,) or None
        neighbors_left: Left neighbor pixels (16,) or None
        neighbor_top_left: Top-left corner pixel or None
        nz_counts: Array to update with non-zero counts per block

    Returns:
        Reconstructed 16x16 luma block
    """
    # Step 1: Generate prediction
    top_available = neighbors_top is not None
    left_available = neighbors_left is not None

    prediction = predict_intra_16x16(
        pred_mode, neighbors_top, neighbors_left, neighbor_top_left,
        top_available=top_available, left_available=left_available
    )

    # Step 2: Process DC coefficients
    residual = np.zeros((16, 16), dtype=np.int32)

    dc_4x4 = np.zeros((4, 4), dtype=np.int32)
    for scan_idx in range(16):
        raster_pos = ZIGZAG_4x4[scan_idx]
        row, col = raster_pos // 4, raster_pos % 4
        dc_4x4[row, col] = dc_coeffs[scan_idx]

    dc_transformed = hadamard_4x4(dc_4x4)
    dc_dequant = dequant_dc_4x4(dc_transformed, qp)

    # Step 3: Process AC coefficients and build residual
    if cbp_luma != 0:
        for block_idx in range(16):
            row, col = _get_4x4_block_position(block_idx)
            block_8x8_idx = block_idx // 4
            if not (cbp_luma & (1 << block_8x8_idx)):
                continue

            ac_coeffs = ac_coeffs_list[block_idx]
            if ac_coeffs is None:
                continue

            nz_counts[block_idx] = int(np.count_nonzero(ac_coeffs))

            coeffs = np.zeros((4, 4), dtype=np.int32)
            dc_row, dc_col = row // 4, col // 4
            coeffs[0, 0] = dc_dequant[dc_row, dc_col]

            for i, pos in enumerate(ZIGZAG_4x4[1:]):
                if i < len(ac_coeffs):
                    r, c = pos // 4, pos % 4
                    coeffs[r, c] = ac_coeffs[i]

            coeffs_dequant = dequant_4x4(coeffs, qp)
            coeffs_dequant[0, 0] = coeffs[0, 0]  # DC already dequantized

            block_residual = idct_4x4(coeffs_dequant)
            residual[row:row+4, col:col+4] = block_residual

    # Step 4: DC-only residual when cbp_luma=0
    if cbp_luma == 0:
        for block_idx in range(16):
            row, col = _get_4x4_block_position(block_idx)
            dc_row, dc_col = row // 4, col // 4
            if dc_dequant[dc_row, dc_col] != 0:
                coeffs = np.zeros((4, 4), dtype=np.int32)
                coeffs[0, 0] = dc_dequant[dc_row, dc_col]
                block_residual = idct_4x4(coeffs)
                residual[row:row+4, col:col+4] = block_residual

    # Step 5: Add residual to prediction and clip
    reconstructed = prediction.astype(np.int32) + residual
    return _clip_block(reconstructed)
```

**Step 2: Refactor existing `reconstruct_i16x16_luma` to use the new function**

Change the body of `reconstruct_i16x16_luma` (lines 1371-1477) to:

```python
def reconstruct_i16x16_luma(
    reader, pred_mode, cbp_luma, qp,
    neighbors_top, neighbors_left, neighbor_top_left,
    nz_counts, mb_x=0, mb_y=0, frame_nz_counts=None, frame_width_mbs=0,
):
    """Reconstruct I_16x16 luma macroblock (CAVLC path).

    Reads residual coefficients from BitReader, then delegates to
    reconstruct_i16x16_from_coeffs for the actual reconstruction.
    """
    # Parse DC block via CAVLC
    dc_nA, dc_nB = get_luma_neighbor_nz(
        0, mb_x, mb_y, nz_counts, frame_nz_counts, frame_width_mbs
    )
    dc_nC = calculate_nC(dc_nA, dc_nB)
    dc_block = decode_residual_block(reader, nC=dc_nC, max_coeffs=16)
    dc_coeffs = dc_block.coefficients

    # Parse AC blocks via CAVLC
    ac_coeffs_list = [None] * 16
    if cbp_luma != 0:
        for block_idx in range(16):
            block_8x8_idx = block_idx // 4
            if not (cbp_luma & (1 << block_8x8_idx)):
                continue
            ac_nA, ac_nB = get_luma_neighbor_nz(
                block_idx, mb_x, mb_y, nz_counts,
                frame_nz_counts, frame_width_mbs or 0
            )
            ac_nC = calculate_nC(ac_nA, ac_nB)
            ac_block = decode_residual_block(reader, ac_nC, max_coeffs=15)
            nz_counts[block_idx] = ac_block.total_coeff
            ac_coeffs_list[block_idx] = ac_block.coefficients

    # Delegate to entropy-independent reconstruction
    return reconstruct_i16x16_from_coeffs(
        pred_mode, cbp_luma, qp, dc_coeffs, ac_coeffs_list,
        neighbors_top, neighbors_left, neighbor_top_left, nz_counts,
    )
```

**Step 3: Run ALL existing tests to verify no regression**

Run: `pytest -x -q 2>&1 | tail -10`
Expected: All 1822+ tests pass (no behavior change, just refactoring)

**Step 4: Commit**

```bash
git add reconstruct/macroblock.py
git commit -m "refactor: extract reconstruct_i16x16_from_coeffs from CAVLC-specific function"
```

---

### Task 4: Add chroma DC reconstruction from pre-parsed coefficients

**Files:**
- Modify: `reconstruct/macroblock.py` (near line 1480)

`decode_chroma_dc` reads from BitReader. We need a version that takes pre-parsed coefficients.

**Step 1: Create `process_chroma_dc_coeffs` function**

Add before `decode_chroma_dc` (before line 1480):

```python
def process_chroma_dc_coeffs(
    dc_coeffs: np.ndarray,
    qp_chroma: int,
) -> np.ndarray:
    """Process pre-parsed chroma DC coefficients.

    Applies inverse Hadamard and dequantization to pre-parsed DC coefficients.

    Args:
        dc_coeffs: 4-element coefficient array (scan order)
        qp_chroma: Chroma QP

    Returns:
        Dequantized 2x2 DC block
    """
    dc_2x2 = np.zeros((2, 2), dtype=np.int32)
    for i in range(min(4, len(dc_coeffs))):
        dc_2x2[i // 2, i % 2] = dc_coeffs[i]

    dc_transformed = hadamard_2x2(dc_2x2)
    dc_dequant = dequant_dc_2x2(dc_transformed, qp_chroma)
    return dc_dequant
```

**Step 2: Refactor `decode_chroma_dc` to use it**

```python
def decode_chroma_dc(reader, cbp_chroma, qp_chroma):
    if cbp_chroma < 1:
        return None
    dc_block = decode_residual_block(reader, nC=-1, max_coeffs=4)
    return process_chroma_dc_coeffs(dc_block.coefficients, qp_chroma)
```

**Step 3: Run tests**

Run: `pytest -x -q 2>&1 | tail -10`
Expected: All tests pass

**Step 4: Commit**

```bash
git add reconstruct/macroblock.py
git commit -m "refactor: extract process_chroma_dc_coeffs from CAVLC-specific function"
```

---

### Task 5: Add `coded_block_flag` to CABAC residual decoding

**Files:**
- Modify: `entropy/cabac_residual.py`

In CABAC, each residual block is preceded by a `coded_block_flag` that indicates whether the block has any non-zero coefficients. The existing `decode_residual_block_cabac` skips this flag.

H.264 Section 9.3.3.1.1.9: The coded_block_flag context depends on block category and neighbor coded_block_flags.

**Step 1: Add `decode_residual_block_cabac_with_cbf` function**

Add to `entropy/cabac_residual.py` after `decode_residual_block_cabac` (after line 261):

```python
def decode_residual_block_cabac_with_cbf(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    max_coeff: int,
    block_cat: int,
    coded_block_flag_ctx_idx: int,
) -> np.ndarray:
    """Decode residual block with coded_block_flag check.

    For CABAC, each block has a coded_block_flag decoded before the
    significance map. If coded_block_flag=0, the block is all zeros.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        max_coeff: Maximum coefficients (4, 15, or 16)
        block_cat: Block category (0-4)
        coded_block_flag_ctx_idx: Context index for coded_block_flag

    Returns:
        Array of coefficients (all zeros if coded_block_flag=0)
    """
    # Decode coded_block_flag
    cbf = decoder.decode_decision(contexts[coded_block_flag_ctx_idx])

    if cbf == 0:
        return np.zeros(max_coeff, dtype=np.int32)

    # Block is coded - decode significance map and levels
    return decode_residual_block_cabac(decoder, contexts, max_coeff, block_cat)
```

**Step 2: Write test**

Add to `entropy/tests/test_cabac_residual.py`:

```python
def test_coded_block_flag_zero_returns_empty():
    """When coded_block_flag=0, block should be all zeros."""
    from unittest.mock import MagicMock
    from entropy.cabac_residual import decode_residual_block_cabac_with_cbf
    from entropy.cabac_context import CABACContext
    import numpy as np

    decoder = MagicMock()
    decoder.decode_decision.return_value = 0  # coded_block_flag = 0
    contexts = [CABACContext(pStateIdx=0, valMPS=0)] * 460
    result = decode_residual_block_cabac_with_cbf(decoder, contexts, 16, 2, 85)
    assert np.all(result == 0)
    assert result.shape == (16,)
```

**Step 3: Run test**

Run: `pytest entropy/tests/test_cabac_residual.py::test_coded_block_flag_zero_returns_empty -v`
Expected: PASS

**Step 4: Commit**

```bash
git add entropy/cabac_residual.py entropy/tests/test_cabac_residual.py
git commit -m "feat(cabac): add coded_block_flag check to residual block decode"
```

---

### Task 6: Add residual decoding to CABAC macroblock layer

**Files:**
- Modify: `entropy/cabac_macroblock.py`

`decode_macroblock_layer_cabac` currently parses mb_type, CBP, mb_pred, and qp_delta, but does NOT decode residual coefficient blocks. We need to add residual decoding for I_4x4, I_16x16, and chroma.

**Step 1: Add residual decoding to `decode_macroblock_layer_cabac`**

After the mb_qp_delta decode (line 366), add:

```python
    # Decode residual blocks
    if has_coded_blocks:
        result['residual'] = _decode_residual_cabac(
            decoder, contexts, result['mb_type'], slice_type, result['cbp'],
            result.get('transform_size_8x8_flag', 0),
        )
    else:
        result['residual'] = {
            'luma_dc': None,
            'luma_ac': [None] * 16,
            'luma_4x4': [None] * 16,
            'chroma_dc_cb': None,
            'chroma_dc_cr': None,
            'chroma_ac_cb': [None] * 4,
            'chroma_ac_cr': [None] * 4,
        }
```

**Step 2: Implement `_decode_residual_cabac` helper**

Add this function at module level in `entropy/cabac_macroblock.py`:

```python
from entropy.cabac_residual import (
    decode_residual_block_cabac,
    decode_residual_block_cabac_with_cbf,
)

def _decode_residual_cabac(
    decoder: 'CABACDecoder',
    contexts: List['CABACContext'],
    mb_type: int,
    slice_type: int,
    cbp: int,
    transform_size_8x8_flag: int,
) -> Dict[str, Any]:
    """Decode all residual blocks for a macroblock using CABAC.

    H.264 Spec Section 7.3.5.3: residual() syntax structure.

    Args:
        decoder: CABAC decoder
        contexts: Context models
        mb_type: Macroblock type
        slice_type: Slice type
        cbp: Coded block pattern (bits 0-3: luma, bits 4-5: chroma)
        transform_size_8x8_flag: 0 for 4x4, 1 for 8x8

    Returns:
        Dict with residual coefficient arrays
    """
    from entropy.cabac_context import CTX_CODED_BLOCK_FLAG_START

    cbp_luma = cbp & 0x0F
    cbp_chroma = (cbp >> 4) & 0x03

    result = {
        'luma_dc': None,
        'luma_ac': [None] * 16,
        'luma_4x4': [None] * 16,
        'chroma_dc_cb': None,
        'chroma_dc_cr': None,
        'chroma_ac_cb': [None] * 4,
        'chroma_ac_cr': [None] * 4,
    }

    is_i16x16 = _is_i_16x16(mb_type, slice_type)

    if is_i16x16:
        # I_16x16: DC block is always coded (block_cat=0)
        result['luma_dc'] = decode_residual_block_cabac(
            decoder, contexts, max_coeff=16, block_cat=0
        )

        # AC blocks: only in 8x8 regions with cbp_luma set (block_cat=1)
        if cbp_luma != 0:
            for block_idx in range(16):
                block_8x8 = block_idx // 4
                if cbp_luma & (1 << block_8x8):
                    # coded_block_flag context for luma AC (block_cat=1)
                    # CTX_CODED_BLOCK_FLAG_START + cat_offset + ctx_inc
                    # For simplicity, use base context (ctx_inc=0)
                    cbf_ctx = CTX_CODED_BLOCK_FLAG_START + 4  # luma AC base
                    result['luma_ac'][block_idx] = decode_residual_block_cabac_with_cbf(
                        decoder, contexts, max_coeff=15, block_cat=1,
                        coded_block_flag_ctx_idx=cbf_ctx,
                    )
    else:
        # I_4x4 / I_NxN: each 4x4 block coded independently (block_cat=2)
        if cbp_luma != 0:
            for block_idx in range(16):
                block_8x8 = block_idx // 4
                if cbp_luma & (1 << block_8x8):
                    cbf_ctx = CTX_CODED_BLOCK_FLAG_START + 4  # luma 4x4 base
                    result['luma_4x4'][block_idx] = decode_residual_block_cabac_with_cbf(
                        decoder, contexts, max_coeff=16, block_cat=2,
                        coded_block_flag_ctx_idx=cbf_ctx,
                    )

    # Chroma DC (block_cat=3)
    if cbp_chroma >= 1:
        result['chroma_dc_cb'] = decode_residual_block_cabac(
            decoder, contexts, max_coeff=4, block_cat=3
        )
        result['chroma_dc_cr'] = decode_residual_block_cabac(
            decoder, contexts, max_coeff=4, block_cat=3
        )

    # Chroma AC (block_cat=4)
    if cbp_chroma >= 2:
        for i in range(4):
            cbf_ctx = CTX_CODED_BLOCK_FLAG_START + 8  # chroma AC base
            result['chroma_ac_cb'][i] = decode_residual_block_cabac_with_cbf(
                decoder, contexts, max_coeff=15, block_cat=4,
                coded_block_flag_ctx_idx=cbf_ctx,
            )
        for i in range(4):
            cbf_ctx = CTX_CODED_BLOCK_FLAG_START + 8
            result['chroma_ac_cr'][i] = decode_residual_block_cabac_with_cbf(
                decoder, contexts, max_coeff=15, block_cat=4,
                coded_block_flag_ctx_idx=cbf_ctx,
            )

    return result
```

**Step 3: Run existing CABAC tests**

Run: `pytest entropy/tests/test_cabac_macroblock.py -v`
Expected: Existing tests still pass

**Step 4: Commit**

```bash
git add entropy/cabac_macroblock.py
git commit -m "feat(cabac): add residual block decoding to macroblock layer"
```

---

### Task 7: Wire CABAC I-slice decode path in decoder.py

**Files:**
- Modify: `decoder/decoder.py:3656-3753` (`_decode_macroblock_cabac` and `_decode_slice_data_cabac`)

This is the core integration task. We need `_decode_macroblock_cabac` to:
1. Call `decode_macroblock_layer_cabac` (which now returns full syntax + residual)
2. Feed the parsed data into the reconstruction functions
3. Write results to frame buffers and update state

**Step 1: Rewrite `_decode_macroblock_cabac`**

Replace the stub (lines 3656-3694) with:

```python
    def _decode_macroblock_cabac(
        self,
        cabac_decoder: 'CABACDecoder',
        slice_type: int,
        mb_x: int,
        mb_y: int,
        current_qp: int,
    ) -> int:
        """Decode and reconstruct a macroblock using CABAC.

        Args:
            cabac_decoder: CABAC decoder
            slice_type: Slice type (0=P, 1=B, 2=I)
            mb_x, mb_y: Macroblock position
            current_qp: Current running QP

        Returns:
            Updated QP after this macroblock
        """
        from entropy.cabac_macroblock import decode_macroblock_layer_cabac
        from reconstruct.macroblock import (
            reconstruct_i16x16_from_coeffs,
            reconstruct_i4x4_block,
            process_chroma_dc_coeffs,
            reconstruct_chroma_plane,
            decode_i16x16_mb_type,
            _get_4x4_block_position,
            _clip_block,
        )
        from entropy.cavlc import CAVLCBlock
        from intra.intra_4x4 import predict_intra_4x4
        from dequant.dequant import dequant_4x4, get_chroma_qp
        from transform.idct_4x4 import idct_4x4

        sps = self.state.current_sps
        pps = self.state.current_pps
        mb_width = sps.pic_width_in_mbs

        mb_idx = mb_y * mb_width + mb_x

        # Build mb_info for CABAC context
        mb_info = self._build_cabac_mb_info(mb_x, mb_y, slice_type)

        # Parse macroblock via CABAC
        mb_data = decode_macroblock_layer_cabac(
            cabac_decoder, self.state.cabac_contexts, slice_type, mb_info
        )

        mb_type = mb_data['mb_type']
        cbp = mb_data['cbp']
        qp_delta = mb_data.get('mb_qp_delta', 0)

        # Update QP
        qp = (current_qp + qp_delta + 52) % 52
        cbp_luma = cbp & 0x0F
        cbp_chroma = (cbp >> 4) & 0x03

        logger.debug(f"CABAC MB ({mb_x},{mb_y}): type={mb_type}, cbp={cbp}, qp_delta={qp_delta}")

        # Pixel positions
        luma_x, luma_y = mb_x * 16, mb_y * 16
        chroma_x, chroma_y = mb_x * 8, mb_y * 8

        # Get neighbor pixels for prediction
        neighbors_top = self.state.frame_luma[luma_y - 1, luma_x:luma_x + 16] if mb_y > 0 else None
        neighbors_left = self.state.frame_luma[luma_y:luma_y + 16, luma_x - 1] if mb_x > 0 else None
        neighbor_top_left = int(self.state.frame_luma[luma_y - 1, luma_x - 1]) if (mb_x > 0 and mb_y > 0) else None

        # Initialize nz_counts for this MB
        nz_counts = np.zeros(24, dtype=np.int32)

        residual = mb_data.get('residual', {})

        # Determine if I_16x16 or I_4x4
        is_i_slice = (slice_type % 5 == 2)
        if is_i_slice:
            if 1 <= mb_type <= 24:
                # I_16x16
                pred_mode, i16_cbp_luma, i16_cbp_chroma = decode_i16x16_mb_type(mb_type)

                dc_coeffs = residual.get('luma_dc')
                if dc_coeffs is None:
                    dc_coeffs = np.zeros(16, dtype=np.int32)
                ac_list = residual.get('luma_ac', [None] * 16)

                luma = reconstruct_i16x16_from_coeffs(
                    pred_mode, i16_cbp_luma, qp, dc_coeffs, ac_list,
                    neighbors_top, neighbors_left, neighbor_top_left,
                    nz_counts,
                )
                self.state.frame_luma[luma_y:luma_y+16, luma_x:luma_x+16] = luma

                # Use I_16x16 embedded CBP for chroma
                cbp_chroma = i16_cbp_chroma

            elif mb_type == 0:
                # I_4x4 (I_NxN)
                mb_pred = mb_data.get('mb_pred', {})
                pred_modes = mb_pred.get('intra_pred_modes', [2] * 16)  # Default DC

                luma_4x4_coeffs = residual.get('luma_4x4', [None] * 16)

                # Reconstruct each 4x4 block in raster scan order
                for block_idx in range(16):
                    row, col = _get_4x4_block_position(block_idx)
                    bx, by = luma_x + col, luma_y + row

                    # Get prediction mode
                    if block_idx < len(pred_modes):
                        mode = pred_modes[block_idx]
                        if mode == -1:
                            mode = 2  # Use DC as default for predicted mode
                    else:
                        mode = 2

                    # Get neighbor pixels from frame
                    top = self.state.frame_luma[by - 1, bx:bx + 4] if by > 0 else None
                    left = self.state.frame_luma[by:by + 4, bx - 1] if bx > 0 else None
                    top_left = int(self.state.frame_luma[by - 1, bx - 1]) if (bx > 0 and by > 0) else None

                    # Top-right neighbor
                    top_right = None
                    top_right_available = False
                    if by > 0 and bx + 4 <= luma_x + 16:
                        if bx + 4 < self.state.frame_luma.shape[1]:
                            top_right = self.state.frame_luma[by - 1, bx + 4:bx + 8]
                            top_right_available = True

                    # Get residual
                    coeffs = luma_4x4_coeffs[block_idx]
                    if coeffs is not None:
                        nz_counts[block_idx] = int(np.count_nonzero(coeffs))
                        # Dequant and IDCT
                        coeff_block = np.zeros((4, 4), dtype=np.int32)
                        from reconstruct.macroblock import ZIGZAG_4x4
                        for i, pos in enumerate(ZIGZAG_4x4):
                            if i < len(coeffs):
                                r, c = pos // 4, pos % 4
                                coeff_block[r, c] = coeffs[i]
                        coeff_dequant = dequant_4x4(coeff_block, qp)
                        residual_block = idct_4x4(coeff_dequant)
                    else:
                        residual_block = np.zeros((4, 4), dtype=np.int32)

                    # Predict + add residual
                    pred = predict_intra_4x4(
                        mode, top, left, top_left, top_right,
                        top_available=(by > 0),
                        left_available=(bx > 0),
                        top_right_available=top_right_available,
                    )
                    block = _clip_block(pred.astype(np.int32) + residual_block)
                    self.state.frame_luma[by:by+4, bx:bx+4] = block

            elif mb_type == 25:
                # I_PCM - handle separately
                logger.warning("I_PCM in CABAC not yet implemented")

        # Chroma reconstruction (same for I_4x4 and I_16x16)
        chroma_pred_mode = 0
        mb_pred = mb_data.get('mb_pred', {})
        if 'intra_chroma_pred_mode' in mb_pred:
            chroma_pred_mode = mb_pred['intra_chroma_pred_mode']

        qp_chroma = get_chroma_qp(qp)

        for plane, frame, dc_key, ac_key in [
            ('cb', self.state.frame_cb, 'chroma_dc_cb', 'chroma_ac_cb'),
            ('cr', self.state.frame_cr, 'chroma_dc_cr', 'chroma_ac_cr'),
        ]:
            top_c = frame[chroma_y - 1, chroma_x:chroma_x + 8] if mb_y > 0 else None
            left_c = frame[chroma_y:chroma_y + 8, chroma_x - 1] if mb_x > 0 else None
            tl_c = int(frame[chroma_y - 1, chroma_x - 1]) if (mb_x > 0 and mb_y > 0) else None

            dc_dequant = None
            dc_coeffs = residual.get(dc_key)
            if dc_coeffs is not None and cbp_chroma >= 1:
                dc_dequant = process_chroma_dc_coeffs(dc_coeffs, qp_chroma)

            ac_blocks = []
            ac_coeffs = residual.get(ac_key, [None] * 4)
            if cbp_chroma >= 2:
                for i in range(4):
                    coeffs = ac_coeffs[i] if i < len(ac_coeffs) else None
                    if coeffs is not None:
                        ac_blocks.append(CAVLCBlock(
                            total_coeff=int(np.count_nonzero(coeffs)),
                            trailing_ones=0,
                            coefficients=coeffs,
                        ))
                    else:
                        ac_blocks.append(CAVLCBlock(
                            total_coeff=0, trailing_ones=0,
                            coefficients=np.zeros(15, dtype=np.int32),
                        ))

            chroma = reconstruct_chroma_plane(
                dc_dequant, ac_blocks, cbp_chroma, qp_chroma,
                chroma_pred_mode, top_c, left_c, tl_c,
            )
            frame[chroma_y:chroma_y+8, chroma_x:chroma_x+8] = chroma

        # Store metadata for deblocking
        self.state.mb_types[mb_idx] = mb_type
        self.state.mb_qps[mb_idx] = qp
        self.state.nz_counts[mb_idx] = nz_counts
        self.state.mb_coeffs[mb_idx, :16] = (nz_counts[:16] > 0)

        # Update intra modes cache
        if hasattr(self.state, 'intra_modes') and self.state.intra_modes is not None:
            if mb_type == 0:  # I_4x4
                mb_pred = mb_data.get('mb_pred', {})
                modes = mb_pred.get('intra_pred_modes', [2] * 16)
                self.state.intra_modes[mb_idx, :] = modes[:16]

        return qp
```

**Step 2: Add `_build_cabac_mb_info` helper**

Add this helper method to the H264Decoder class (near the other CABAC helpers around line 3598):

```python
    def _build_cabac_mb_info(self, mb_x: int, mb_y: int, slice_type: int) -> dict:
        """Build mb_info dict for CABAC context derivation.

        Args:
            mb_x, mb_y: Macroblock position
            slice_type: Slice type

        Returns:
            Dict with neighbor information for CABAC
        """
        mb_width = self.state.current_sps.pic_width_in_mbs
        mb_idx = mb_y * mb_width + mb_x
        pps = self.state.current_pps

        mb_info = {
            'mb_x': mb_x,
            'mb_y': mb_y,
            'left_available': mb_x > 0,
            'top_available': mb_y > 0,
            'left_skip': False,
            'top_skip': False,
            'left_cbp': 0,
            'top_cbp': 0,
            'num_ref_idx_l0_active': 1,
            'num_ref_idx_l1_active': 1,
            'transform_8x8_mode_flag': getattr(pps, 'transform_8x8_mode_flag', False),
        }

        if mb_x > 0:
            left_idx = mb_idx - 1
            mb_info['left_skip'] = (self.state.mb_types[left_idx] == -1)
            mb_info['left_mb_type'] = int(self.state.mb_types[left_idx])

        if mb_y > 0:
            top_idx = mb_idx - mb_width
            mb_info['top_skip'] = (self.state.mb_types[top_idx] == -1)
            mb_info['top_mb_type'] = int(self.state.mb_types[top_idx])

        return mb_info
```

**Step 3: Update `_decode_slice_data_cabac` to use updated method**

Replace the slice loop (lines 3696-3753) to properly pass and track QP:

```python
    def _decode_slice_data_cabac(
        self, reader, slice_header, sps, pps, slice_qp,
    ):
        """Decode slice data using CABAC entropy coding."""
        mb_width = sps.pic_width_in_mbs
        mb_height = sps.frame_height_in_mbs
        total_mbs = mb_width * mb_height

        slice_type = slice_header.slice_type % 5
        self._init_cabac_contexts(slice_type, slice_qp)
        cabac_decoder = self._create_cabac_decoder(reader)

        logger.debug(f"Decoding CABAC slice: {total_mbs} MBs, type={slice_type}")

        current_qp = slice_qp
        mb_idx = slice_header.first_mb_in_slice

        while mb_idx < total_mbs:
            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Check for mb_skip_flag (P/B slices only)
            if slice_type in (0, 1):
                skip_flag = self._decode_mb_skip_flag_cabac(
                    cabac_decoder, slice_type, mb_x, mb_y
                )
                if skip_flag == 1:
                    self.state.mb_types[mb_idx] = -1
                    mb_idx += 1
                    continue

            # Decode and reconstruct macroblock
            current_qp = self._decode_macroblock_cabac(
                cabac_decoder, slice_type, mb_x, mb_y, current_qp
            )

            # Check for end_of_slice
            if cabac_decoder.decode_terminate() == 1:
                break

            mb_idx += 1
```

**Step 4: Run tests**

Run: `pytest -x -q 2>&1 | tail -10`
Expected: All existing tests pass (CABAC path not triggered by existing test data)

**Step 5: Commit**

```bash
git add decoder/decoder.py
git commit -m "feat(cabac): wire CABAC I-slice decode with full reconstruction"
```

---

### Task 8: End-to-end CABAC I-slice test

**Step 1: Run pixel-perfect comparison**

```python
python3 -c "
from decoder.decoder import H264Decoder
import numpy as np

dec = H264Decoder()
frames = sorted(dec.decode_file('test_data/testsrc2_cabac_intra.264'), key=lambda f: f.poc)
with open('test_data/testsrc2_cabac_intra_ref.yuv', 'rb') as f:
    ref_data = f.read()
w, h = frames[0].luma.shape[1], frames[0].luma.shape[0]
frame_size = w * h * 3 // 2
for i, frame in enumerate(frames):
    offset = i * frame_size
    if offset + frame_size > len(ref_data):
        break
    ref_y = np.frombuffer(ref_data[offset:offset+w*h], dtype=np.uint8).reshape(h, w)
    d = int(np.abs(frame.luma.astype(np.int16) - ref_y.astype(np.int16)).max())
    print(f'Frame {i}: {\"PERFECT\" if d == 0 else f\"max_diff={d}\"}')
"
```

Expected: All frames show PERFECT

**Step 2: If errors exist, debug**

If max_diff > 0, diagnose by:
1. Check which MB has the first error: compare frame pixel by pixel
2. Check if context init tables are correct: dump pStateIdx/valMPS for key contexts
3. Check mb_type decode: print decoded mb_type vs expected
4. Check residual coefficients: compare against ffmpeg's CABAC decode trace

Common issues:
- Wrong context init values → wrong pStateIdx → wrong decode decisions → garbage
- Wrong binarization tree → wrong mb_type → wrong syntax parse → crash or wrong data
- Missing coded_block_flag check → reading coefficients for empty blocks → bit desync

**Step 3: Run full test suite**

Run: `pytest -x -q 2>&1 | tail -10`
Expected: All tests pass (existing CAVLC tests unaffected, new CABAC test passes)

**Step 4: Commit with results**

```bash
git add -A
git commit -m "feat(cabac): pixel-perfect I-slice decode via CABAC

Phase 1 of CABAC integration. Decodes I-slices (I_4x4, I_16x16)
using CABAC entropy coding with full 460-context initialization
tables from H.264 spec. Reuses existing reconstruction pipeline
via refactored reconstruct_i16x16_from_coeffs."
```

---

## Verification Checklist

| Item | Expected |
|------|----------|
| testsrc2_cabac_intra (CABAC I-only) | PERFECT |
| All existing CAVLC test streams | PERFECT (no regression) |
| All 1822+ unit tests | PASS |

## Files Modified

| File | Changes |
|------|---------|
| `entropy/cabac_context.py` | Complete 460-entry init tables |
| `entropy/cabac_residual.py` | Add `decode_residual_block_cabac_with_cbf` |
| `entropy/cabac_macroblock.py` | Add residual decoding to MB layer |
| `reconstruct/macroblock.py` | Extract `reconstruct_i16x16_from_coeffs`, `process_chroma_dc_coeffs` |
| `decoder/decoder.py` | Complete `_decode_macroblock_cabac`, update `_decode_slice_data_cabac` |
| `entropy/tests/test_cabac_context_init.py` | Context init verification |
| `entropy/tests/test_cabac_residual.py` | coded_block_flag test |
| `test_data/testsrc2_cabac_intra.*` | Test stream + reference |

## Known Risks

1. **Context init tables** — Must be exact. Even one wrong (m,n) pair can cascade through the arithmetic decoder.
2. **Binarization bugs** — `decode_mb_type_i` and other syntax decoders haven't been tested against real bitstreams. Bugs will show up as crashes or wrong mb_type values.
3. **coded_block_flag context** — The context index depends on neighbor coded_block_flags. The simplified version (using base context) may need neighbor tracking for full correctness.
4. **I_4x4 prediction mode derivation** — The CABAC path returns raw modes from `prev_intra4x4_pred_mode_flag` / `rem_intra4x4_pred_mode`, but the actual mode depends on neighbor MPM calculation (same as CAVLC). This derivation needs to happen in the decoder, not the entropy layer.
