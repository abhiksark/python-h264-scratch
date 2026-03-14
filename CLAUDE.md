# Development Guide

## Conventions

### Code Style
- Google Python Style Guide
- Type hints on all public functions
- Docstrings with H.264 spec section references

### File Headers
```python
# h264/<module>/<file>.py
"""Brief description.

H.264 Spec Reference: Section X.Y.Z
"""
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Processing: {value}")
```

### NumPy Patterns
- Use `dtype=np.int32` during computation (avoid overflow)
- Clip to `[0, 255]` and cast to `uint8` only at final output
- Prefer vectorized operations over loops

## Quick Commands

```bash
# Run all tests
pytest -v

# Run specific module tests
pytest decoder/tests/ -v
pytest entropy/tests/ -v

# Run with logging
pytest -v --log-cli-level=DEBUG

# Run High Profile pixel-perfect tests
pytest decoder/tests/test_high_profile.py -v
```

## Module Dependencies (Pipeline Order)

```
bitstream → parameters → slice → entropy → dequant → transform → intra/inter → reconstruct → deblock → color
```

## Key Files

- `decoder/decoder.py` — main decoder orchestration (~5000 lines)
- `entropy/cabac_macroblock.py` — CABAC MB-level syntax parsing
- `entropy/cabac_residual.py` — CABAC coefficient decoding
- `intra/intra_8x8.py` — I_8x8 prediction modes
- `decoder/i8x8.py` — I_8x8 reconstruction with reference sample filtering
- `inter/reference.py` — DPB and reference frame management
- `reconstruct/macroblock.py` — macroblock reconstruction pipeline
