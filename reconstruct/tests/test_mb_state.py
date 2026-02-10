# reconstruct/tests/test_mb_state.py
"""Tests for macroblock state machine."""

import pytest
from reconstruct.mb_state import MBState, MBDecodeContext


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
