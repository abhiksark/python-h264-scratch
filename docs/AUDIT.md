# H.264 Decoder Codebase Audit

Date: 2026-01-22

## Scope
This audit focused on:
- Repository structure, entry points, and build/dependency setup
- Core decoding pipeline logic (Annex B NAL parsing → SPS/PPS → slice headers → entropy → reconstruction → inter/intra → deblock → color)
- Robustness and safety considerations for malformed/untrusted bitstreams
- Test suite health, CI/tooling, and coverage gaps

## Repository map (high-level)
- **Entry points**
  - `decoder/decoder.py`: `H264Decoder`, `decode_h264_file`, `decode_h264_bytes`
  - `decoder/__init__.py`: exports public API

- **Pipeline modules**
  - `bitstream/`: `nal_parser.py`, `bit_reader.py`, `numpy_bit_reader.py`
  - `parameters/`: `sps.py`, `pps.py`, `scaling.py`
  - `slice/`: `slice_header.py`
  - `entropy/`: `cavlc.py`, `cabac_*`, `tables.py`
  - `reconstruct/`: `macroblock.py`
  - `intra/`: intra prediction + I_PCM parsing
  - `inter/`: motion compensation, MV prediction, P/B reconstruction, reference buffer
  - `deblock/`: deblocking filter implementation
  - `color/`: YCbCr → RGB conversion

- **Testing**
  - Root `conftest.py` defines shared fixtures and JM comparison helpers.
  - Per-module tests exist under `*/tests/`.
  - Many tests are explicitly marked `skip`/`xfail` as “RED tests”.

## Key findings (prioritized)

### 1) BitReader selection / dependency mismatch (Critical)
**Where**:
- `bitstream/bit_reader.py`, `bitstream/__init__.py`
- Broadly: any module doing `from bitstream import BitReader` (e.g. `parameters/sps.py`, `parameters/pps.py`, `slice/slice_header.py`, `decoder/decoder.py`).

**What**:
- `bitstream.BitReader` is the **bitstring-based** reader and raises `ImportError` if `bitstring` is not installed.
- `USE_NUMPY_READER = True` exists and there is a factory `create_bit_reader()`, but core parsing code generally **does not use** it.

**Impact**:
- The “NumPy reader is default” intent is not reflected in actual imports.
- A missing `bitstring` dependency breaks core parsing.

**Recommendation**:
- Choose one:
  - **Option A (explicit)**: update callsites to use `create_bit_reader(data)` instead of `BitReader(data)`.
  - **Option B (alias)**: change exports so `bitstream.BitReader` resolves to NumPy implementation when `USE_NUMPY_READER=True`.
- Adjust tests that blanket-skip on `BITSTRING_AVAILABLE` if NumPy-only operation is desired.

### 2) Decoder assumes PPS 0 and “one slice == one frame” (Critical)
**Where**:
- `decoder/decoder.py` in `_decode_slice()`

**What**:
- PPS is selected via `self.state.get_pps(0)` with the comment “For now, assume PPS 0”.
- Frame emission assumes each slice completes a frame.

**Impact**:
- Streams with multiple PPS IDs decode incorrectly.
- Multi-slice pictures are handled incorrectly (potential over-read / incorrect frame boundaries).

**Recommendation**:
- Parse the first three ue(v) fields of the slice header (`first_mb_in_slice`, `slice_type`, `pic_parameter_set_id`) with a temporary reader to select the correct PPS before full parsing.
- Only emit a `DecodedFrame` at picture boundary / access unit boundary (e.g., AU delimiting via AUD/IDR boundaries or a simple slice aggregation strategy).

### 3) Inter residual parsing is incomplete / not wired (Major)
**Where**:
- `decoder/decoder.py`: `_decode_p_macroblock()` has multiple TODOs and passes residuals as `None`.
- `inter/p_macroblock.py`: `parse_p_residual()` is TODO and currently returns empty residual.
- `decoder/tests/test_p_residual_integration.py` documents this as a known gap.

**Impact**:
- P-slice decoding may become “prediction-only” in several paths, producing incorrect output for many streams.

**Recommendation**:
- Implement `parse_p_residual()` (CBP-driven coefficient parsing + dequant + IDCT) and wire it into `_decode_p_macroblock()`.
- Ensure `mb_qp_delta` is parsed and the per-MB QP is updated when needed.

### 4) I_4x4 residual context (`nC`) is incorrect (Major)
**Where**:
- `reconstruct/macroblock.py` uses `calculate_nC(None, None)` with a TODO.

**Impact**:
- Wrong VLC table selection for coeff_token in many blocks → non-bit-exact results.

**Recommendation**:
- Derive `nA`/`nB` from actual left/top neighbor `nz_counts` per spec (9.2.1).

### 5) Deblocking integration is incomplete / potentially wrong when enabled (Major)
**Where**:
- `decoder/decoder.py` defines `_deblock_frame()` but it does not appear to be invoked from the normal decode path.
- `_deblock_frame()` passes `mb_info=None` (TODO).
- `deblock/deblock.py` uses “assume intra” behavior when `mb_info` is missing.

**Impact**:
- Deblocking may not run at all, or may run with incorrect strength decisions.

**Recommendation**:
- Call deblocking at the correct stage (end of picture or end of slice depending on `disable_deblocking_filter_idc`).
- Provide per-MB info: intra/inter, coded-block flags, MVs, reference indices, and respect slice boundaries (`disable_deblocking_filter_idc==2`).

### 6) SPS-driven allocations can be abused (Robustness / DoS)
**Where**:
- `DecoderState.allocate_frame_buffers()` allocates `np.zeros((height, width))` directly from SPS dimensions.

**Impact**:
- Malicious bitstreams can request huge dimensions → memory exhaustion.

**Recommendation**:
- Add hard caps on:
  - Maximum frame dimensions / MB count
  - Maximum DPB size
- Validate level constraints (Annex A tables) before allocating.

### 7) CABAC context initialization is explicitly simplified (Correctness risk)
**Where**:
- `entropy/cabac_context.py`

**What**:
- Only a subset of the 460 `(m,n)` entries appear defined; the rest default.
- `cabac_init_idc` handling is a lightweight adjustment rather than proper table selection.

**Impact**:
- CABAC decoding is unlikely to be bit-exact beyond narrow coverage.

**Recommendation**:
- Implement full spec tables (9-12 .. 9-23) for all 460 contexts and proper init-idc selection.

### 8) Documentation drift / inconsistency
**Where**:
- `README.md` vs `docs/PROGRESS.md` vs code.

**What**:
- README states limited support (e.g., only I_16x16, no P/B, no deblocking), while the code and `docs/PROGRESS.md` indicate broader support.

**Impact**:
- Users and future maintainers will have incorrect expectations.

**Recommendation**:
- Reconcile README with actual supported features.

## Tests and tooling

### Current state
- Extensive unit tests exist per module.
- Many tests are explicitly marked `skip`/`xfail` as RED tests.
- Integration tests compare output to JM reference where available.

### Gaps
- No CI configuration detected (no `.github/workflows/*`).

### Recommendations
- Add a minimal CI workflow:
  - Install dependencies
  - Run `pytest -q`
  - Publish coverage (`pytest-cov`)
- Keep default test runs “green” and ensure RED tests are clearly separated or gated.

## Suggested remediation roadmap

### Phase 1 (Stability)
- Fix BitReader selection so NumPy-reader truly works without `bitstring`.
- Remove PPS=0 assumption; parse `pic_parameter_set_id` to select correct PPS.
- Add SPS dimension caps to prevent memory DoS.

### Phase 2 (Correctness)
- Implement and wire P/B residual parsing (CBP + `mb_qp_delta` + residual blocks).
- Fix `nC` derivation for I_4x4.
- Integrate deblocking with correct MB metadata.

### Phase 3 (Compliance)
- Full CABAC init tables and correct `cabac_init_idc` behavior.
- Profile/level validation module (to match existing RED tests).

## Notes
This codebase is an educational implementation and has a clear, modular structure. The primary risks are integration consistency (BitReader selection, PPS selection, frame/slice boundary handling) and correctness gaps hidden behind TODO placeholders.
