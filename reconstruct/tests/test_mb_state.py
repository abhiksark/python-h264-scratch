# reconstruct/tests/test_mb_state.py
"""Tests for macroblock state machine."""

import pytest
import numpy as np
from reconstruct.mb_state import MBState, MBDecodeContext, MBStateValidationError, MacroblockDecoder
from bitstream.bit_reader import BitReader


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


def test_mb_decoder_initialization():
    """Test decoder initializes with correct state."""
    reader = BitReader(b'\x00\x00\x00\x00')

    decoder = MacroblockDecoder(
        reader=reader,
        mb_x=0,
        mb_y=0,
        mb_type=0,
        cbp_luma=15,
        cbp_chroma=2,
        qp=26
    )

    assert decoder.context.current_state == MBState.START_MB
    assert decoder.context.mb_x == 0
    assert decoder.context.mb_y == 0


def test_mb_decoder_validates_state_transition():
    """Test decoder validates illegal state transitions."""
    reader = BitReader(b'\x00\x00\x00\x00')
    decoder = MacroblockDecoder(reader, 0, 0, 0, 15, 2, 26)

    # Try to skip from START_MB to CHROMA_AC (illegal)
    with pytest.raises(MBStateValidationError) as exc_info:
        decoder._validate_transition(MBState.CHROMA_AC)

    assert "Invalid state transition" in str(exc_info.value)
    assert "START_MB" in str(exc_info.value)
    assert "CHROMA_AC" in str(exc_info.value)


def test_decode_luma_residual_validates_block_count():
    """Test luma residual validates correct number of blocks decoded."""
    # Create test bitstream with valid coeff_token for empty blocks
    # Empty block: TC=0, T1=0, for nC=0 uses code '1' (1 bit)
    # We need 16 blocks (CBP=15 means all 4 quadrants, each with 4 blocks)
    bits = '1' * 16  # 16 empty blocks
    data = int(bits, 2).to_bytes((len(bits) + 7) // 8, 'big')
    reader = BitReader(data)

    decoder = MacroblockDecoder(reader, 0, 0, 0, cbp_luma=15, cbp_chroma=0, qp=26)

    # Mock the luma blocks array and nz_counts
    decoder.luma_blocks = [np.zeros((4, 4), dtype=np.int32) for _ in range(16)]
    decoder.nz_counts = np.zeros(24, dtype=np.int32)

    # Decode luma - should validate 12 blocks
    decoder.decode_luma_residual()

    assert decoder.context.luma_blocks_decoded == 16  # CBP=15 means all 16 blocks
    assert decoder.context.current_state == MBState.CHROMA_DC
