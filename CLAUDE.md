# h264/CLAUDE.md
# H.264 Baseline Decoder - Project Memory

## Project Overview
Educational H.264 Baseline profile decoder in NumPy. Goal: understand video compression by implementing from scratch.

## Current Phase
**Track A + B (Parallel)**: Building NumPy components AND bitstream parsing simultaneously.

## Key Paths
- **JM Reference**: `~/JM/` (clone from https://github.com/shihuade/JM)
- **Test Data**: `test_data/reference/` (JM ldecod outputs)
- **Plan File**: `~/.claude/plans/abstract-napping-stardust.md`

## Conventions

### Code Style
- Google Python Style Guide
- Type hints on all public functions
- Docstrings with H.264 spec section references

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

### File Headers
```python
# h264/<module>/<file>.py
"""Brief description.

H.264 Spec Reference: Section X.Y.Z
"""
```

## Quick Commands
```bash
# Run all tests
pytest -v

# Run specific module tests
pytest color/tests/ -v

# Run with logging
pytest -v --log-cli-level=DEBUG

# Generate test bitstream (JM)
cd ~/JM && ./bin/lencod -d encoder_baseline.cfg
```

## Module Dependencies (Pipeline Order)
```
bitstream → parameters → slice → entropy → dequant → transform → intra → reconstruct → color
```

## Status
See `docs/PROGRESS.md` for detailed per-module status.
