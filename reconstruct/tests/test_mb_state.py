# reconstruct/tests/test_mb_state.py
"""Tests for macroblock state machine."""

import pytest
from reconstruct.mb_state import MBState, MBDecodeContext, MBStateValidationError


def test_mb_state_enum_has_all_states():
    """Verify all required states exist."""
    assert hasattr(MBState, 'START_MB')
    assert hasattr(MBState, 'LUMA_RESIDUAL')
    assert hasattr(MBState, 'CHROMA_DC')
    assert hasattr(MBState, 'CHROMA_AC')
    assert hasattr(MBState, 'MB_COMPLETE')


def test_mb_decode_context_creation():
    """Test creating decode context."""
    ctx = MBDecodeContext(
        mb_x=0,
        mb_y=0,
        mb_type=0,
        cbp_luma=15,
        cbp_chroma=2,
        qp=26,
        bit_position_start=100
    )
    assert ctx.mb_x == 0
    assert ctx.mb_y == 0
    assert ctx.current_state == MBState.START_MB


def test_validation_error_contains_context():
    """Test validation error includes debug context."""
    ctx = MBDecodeContext(
        mb_x=1,
        mb_y=2,
        mb_type=0,
        cbp_luma=7,
        cbp_chroma=2,
        qp=26,
        bit_position_start=500
    )

    error = MBStateValidationError(
        message="Expected 8 luma blocks, got 6",
        context=ctx,
        expected_state=MBState.CHROMA_DC,
        actual_state=MBState.LUMA_RESIDUAL
    )

    error_str = str(error)
    assert "MB(1,2)" in error_str
    assert "bit_pos=500" in error_str or "500" in error_str
    assert "Expected 8 luma blocks" in error_str
    assert "LUMA_RESIDUAL" in error_str
