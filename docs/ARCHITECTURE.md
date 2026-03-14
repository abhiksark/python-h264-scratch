# Architecture

## Design Philosophy

### One folder per pipeline stage
Each decoding stage is isolated in its own directory with source files, tests, and a public API via `__init__.py`. This maps 1:1 to the H.264 spec sections and makes it easy to test each stage independently.

### Correctness first, performance never
This is an educational decoder. Every function prioritizes readability and spec compliance over speed. Pure Python with NumPy — no SIMD, no C extensions, no threading.

### Verified against reference decoders
Every pipeline stage is tested against JM (the H.264 reference software) and ffmpeg. Pixel-perfect accuracy is verified on real internet videos.

## Data Flow

```
Input (MP4 or Annex B .264)
    │
    ▼
container/       MP4 box parsing, avcC extraction (or Annex B start code scan)
    │
    ▼
bitstream/       NAL unit extraction, emulation prevention byte removal
    │
    ├──▶ parameters/   SPS (NAL type 7), PPS (NAL type 8)
    │
    ▼
slice/           Slice header parsing (type, QP, reference lists, weight tables)
    │
    ▼
entropy/         CAVLC or CABAC → quantized coefficients + MB syntax
    │
    ▼
dequant/         Inverse quantization (QP-dependent scaling, scaling lists)
    │
    ▼
transform/       4x4 or 8x8 inverse integer transform → spatial residual
    │
    ▼
intra/           Intra prediction from neighboring pixels (I-frames)
inter/           Motion compensation from reference frames (P/B-frames)
    │
    ▼
reconstruct/     prediction + residual → reconstructed macroblock
    │
    ▼
deblock/         In-loop deblocking filter (boundary strength, adaptive)
    │
    ▼
decoder/         Frame buffer management, DPB, output ordering
    │
    ▼
color/           YCbCr 4:2:0 → RGB (optional)
```

## Key Data Structures

### DecodedFrame
```python
@dataclass
class DecodedFrame:
    luma: np.ndarray      # (H, W) uint8
    cb: np.ndarray        # (H/2, W/2) uint8
    cr: np.ndarray        # (H/2, W/2) uint8
    width: int
    height: int
    poc: int              # Picture order count (display order)
```

### DecoderState
The main decoder maintains per-frame state:
- `frame_luma`, `frame_cb`, `frame_cr` — current frame being reconstructed
- `mb_types`, `mb_qps`, `mb_cbps` — per-macroblock metadata
- `nz_counts`, `intra_modes` — neighbor context for entropy coding
- `ref_buffer` — decoded picture buffer (DPB) for inter prediction
- `mv_cache` — motion vector storage for MV prediction

## Module Responsibilities

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `decoder/decoder.py` | ~5300 | Main orchestration — ties all modules together |
| `entropy/cabac_macroblock.py` | ~2200 | CABAC MB-level syntax (mb_type, MVD, CBP, coefficients) |
| `entropy/cabac_context.py` | ~2100 | 460 CABAC context models with init tables from spec |
| `reconstruct/macroblock.py` | ~2300 | I_16x16, I_4x4 reconstruction pipelines |
| `deblock/deblock.py` | ~1700 | Boundary strength + luma/chroma filtering |
| `inter/p_reconstruct.py` | ~1450 | P-frame partition reconstruction |
| `decoder/i8x8.py` | ~1070 | I_8x8 reconstruction with reference sample filtering |

## Why These Choices?

### Why Python/NumPy?
- The math is visible — `(a + 2*b + c + 2) >> 2` reads like the spec
- Easy to step through with a debugger
- No build system, runs anywhere

### Why not start with Baseline?
We did. The decoder grew from Baseline (I-only) → Baseline (I+P) → Main (B-frames, CABAC) → High (8x8 transform) over several months.

### Why MP4 support?
So you can download any video from the internet and decode it. Raw Annex B files are also supported.

### Why pixel-perfect?
An H.264 decoder is either correct or wrong — there's no "close enough". Pixel-perfect verification against ffmpeg proves the implementation matches the spec exactly.
