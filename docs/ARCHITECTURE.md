# h264/docs/ARCHITECTURE.md
# Architecture Decisions

## Design Philosophy

### 1. One Folder Per Pipeline Stage
Each decoding stage is isolated in its own folder with:
- `__init__.py` - Public API exports
- Core implementation files
- `tests/` subfolder

**Rationale**:
- Easy to test each stage independently
- Clear ownership of functionality
- Can swap implementations (e.g., bitstring → NumPy) without affecting other stages

### 2. Lib First, NumPy Later
Start with libraries (bitstring) for complex bit manipulation, then replace with NumPy.

**Rationale**:
- Get working decoder faster
- Validate correctness before optimizing
- Learn the algorithm before implementing low-level details

### 3. Heavy Testing with JM Reference
Every function is tested against JM reference software output.

**Rationale**:
- H.264 is a solved problem - we know correct outputs
- Bit-exact compliance is required
- Catch regressions when replacing libs with NumPy

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Annex B Bitstream                         │
│                    (bytes from .264 file)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ bitstream/                                                       │
│   Input: bytes                                                   │
│   Output: List[NALUnit] (type, rbsp_bytes)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│ parameters/              │   │ slice/                   │
│   Input: NAL type 7,8    │   │   Input: NAL type 1,5    │
│   Output: SPS, PPS       │   │   Output: SliceHeader    │
└──────────────────────────┘   └──────────────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ entropy/                                                         │
│   Input: slice data bits, SPS, PPS                              │
│   Output: coefficients[mb][block] (4x4 arrays)                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ dequant/                                                         │
│   Input: coefficients, QP                                        │
│   Output: dequantized coefficients                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ transform/                                                       │
│   Input: dequantized coefficients                                │
│   Output: residual pixels (4x4 blocks)                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ intra/ (or inter/)                                               │
│   Input: neighboring pixels, mode                                │
│   Output: prediction pixels (16x16 or 4x4)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ reconstruct/                                                     │
│   Input: prediction + residual                                   │
│   Output: reconstructed Y, Cb, Cr planes                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ color/                                                           │
│   Input: Y, Cb, Cr planes (4:2:0)                               │
│   Output: RGB numpy array (H, W, 3), uint8                      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Data Structures

### NALUnit
```python
@dataclass
class NALUnit:
    nal_ref_idc: int      # 2 bits
    nal_unit_type: int    # 5 bits
    rbsp: bytes           # Raw byte sequence payload
```

### SPS (Sequence Parameter Set)
```python
@dataclass
class SPS:
    profile_idc: int
    level_idc: int
    seq_parameter_set_id: int
    pic_width_in_mbs: int
    pic_height_in_map_units: int
    frame_mbs_only_flag: bool
    # ... more fields
```

### Macroblock
```python
@dataclass
class Macroblock:
    mb_type: int
    intra_16x16_pred_mode: int  # For I_16x16
    coded_block_pattern: int
    qp_delta: int
    coefficients: List[np.ndarray]  # 16 luma + 8 chroma 4x4 blocks
```

## Why These Choices?

### Why Baseline Profile?
- Simplest H.264 profile
- No B-frames (simpler reference management)
- No CABAC (CAVLC is easier to understand)
- Widely supported

### Why Annex B Format?
- Self-contained (no container parsing)
- Start codes make NAL boundaries obvious
- Easier to debug

### Why NumPy?
- Educational - see the math clearly
- Vectorized operations match spec notation
- Easy to visualize intermediate results

### Why Not FFmpeg?
- FFmpeg is heavily optimized, hard to learn from
- Thousands of edge cases obscure core algorithm
- JM reference is cleaner for learning
