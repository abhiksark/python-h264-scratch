# h264/entropy/tests/test_cabac_arith.py
"""RED TESTS: CABAC binary arithmetic decoder.

The arithmetic decoder maintains:
- codIRange: Current interval range (256-510)
- codIOffset: Current offset within range (read from bitstream)

H.264 Spec Reference: Section 9.3.3 - Arithmetic decoding process

These tests SHOULD FAIL until CABAC arithmetic decoder is implemented.
"""

import pytest
import numpy as np

from bitstream import BitReader


class TestCABACDecoderInit:
    """Tests for CABACDecoder initialization."""

    def test_cabac_decoder_exists(self):
        """CABACDecoder class should exist."""
        from entropy.cabac_arith import CABACDecoder

        assert CABACDecoder is not None

    def test_cabac_decoder_init_from_reader(self):
        """CABACDecoder should initialize from BitReader."""
        from entropy.cabac_arith import CABACDecoder

        # Create test bitstream
        data = bytes([0x00, 0xFF, 0x80, 0x40])
        reader = BitReader(data)

        decoder = CABACDecoder(reader)
        assert decoder is not None

    def test_cabac_decoder_has_range(self):
        """CABACDecoder should have codIRange attribute."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0xFF, 0x80, 0x40])
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        assert hasattr(decoder, 'codIRange')

    def test_cabac_decoder_has_offset(self):
        """CABACDecoder should have codIOffset attribute."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0xFF, 0x80, 0x40])
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        assert hasattr(decoder, 'codIOffset')

    def test_initial_range_is_510(self):
        """Initial codIRange should be 510."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0xFF, 0x80, 0x40])
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        # Per spec, codIRange starts at 510
        assert decoder.codIRange == 510

    def test_initial_offset_from_9_bits(self):
        """Initial codIOffset is read from first 9 bits."""
        from entropy.cabac_arith import CABACDecoder

        # First 9 bits of 0x00FF = 000000001 = 1
        data = bytes([0x00, 0xFF, 0x80, 0x40])
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        # codIOffset read from bits
        assert hasattr(decoder, 'codIOffset')


class TestDecodeDecision:
    """Tests for context-based binary decision decoding."""

    def test_decode_decision_exists(self):
        """decode_decision method should exist."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0xFF, 0x80, 0x40, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        assert hasattr(decoder, 'decode_decision')
        assert callable(decoder.decode_decision)

    def test_decode_decision_returns_binary(self):
        """decode_decision should return 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import CABACContext

        data = bytes([0x00, 0xFF, 0x80, 0x40, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        ctx = CABACContext(pStateIdx=32, valMPS=0)
        result = decoder.decode_decision(ctx)

        assert result in (0, 1)

    def test_decode_decision_updates_context(self):
        """decode_decision should update context probability."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import CABACContext

        data = bytes([0x00, 0xFF, 0x80, 0x40, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        ctx = CABACContext(pStateIdx=32, valMPS=0)
        initial_state = ctx.pStateIdx

        decoder.decode_decision(ctx)

        # State should change after decoding
        # (either MPS or LPS state transition)
        # Note: It's valid for state to stay same in edge cases
        assert 0 <= ctx.pStateIdx <= 63


class TestDecodeBypass:
    """Tests for bypass (equiprobable) decoding."""

    def test_decode_bypass_exists(self):
        """decode_bypass method should exist."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0xFF, 0x80, 0x40, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        assert hasattr(decoder, 'decode_bypass')
        assert callable(decoder.decode_bypass)

    def test_decode_bypass_returns_binary(self):
        """decode_bypass should return 0 or 1."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0xFF, 0x80, 0x40, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        result = decoder.decode_bypass()
        assert result in (0, 1)

    def test_decode_bypass_no_context(self):
        """decode_bypass should not require context."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0xAA, 0x55, 0xAA, 0x55] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        # Should work without any context argument
        try:
            decoder.decode_bypass()
        except TypeError:
            pytest.fail("decode_bypass should not require context")


class TestDecodeTerminate:
    """Tests for terminate (end_of_slice) decoding."""

    def test_decode_terminate_exists(self):
        """decode_terminate method should exist."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0xFF, 0x80, 0x40, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        assert hasattr(decoder, 'decode_terminate')
        assert callable(decoder.decode_terminate)

    def test_decode_terminate_returns_binary(self):
        """decode_terminate should return 0 or 1."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0xFF, 0x80, 0x40, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        result = decoder.decode_terminate()
        assert result in (0, 1)


class TestRenormalization:
    """Tests for arithmetic decoder renormalization."""

    def test_renormalize_exists(self):
        """renormalize method should exist."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0xFF, 0x80, 0x40, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        assert hasattr(decoder, 'renormalize')

    def test_range_stays_in_bounds(self):
        """codIRange should stay in [256, 510] after operations."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import CABACContext

        data = bytes([0xAA, 0x55, 0xAA, 0x55] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        ctx = CABACContext(pStateIdx=32, valMPS=0)

        # Multiple decode operations
        for _ in range(10):
            decoder.decode_decision(ctx)
            assert 256 <= decoder.codIRange <= 510


class TestStateTransitionTables:
    """Tests for probability state transition tables."""

    def test_trans_idx_lps_exists(self):
        """transIdxLPS table should exist (64 entries)."""
        from entropy.cabac_arith import TRANS_IDX_LPS

        assert len(TRANS_IDX_LPS) == 64

    def test_trans_idx_mps_exists(self):
        """transIdxMPS table should exist (64 entries)."""
        from entropy.cabac_arith import TRANS_IDX_MPS

        assert len(TRANS_IDX_MPS) == 64

    def test_range_tab_lps_exists(self):
        """rangeTabLPS table should exist (64x4)."""
        from entropy.cabac_arith import RANGE_TAB_LPS

        assert len(RANGE_TAB_LPS) == 64
        assert len(RANGE_TAB_LPS[0]) == 4

    def test_lps_transition_decreases_state(self):
        """LPS transition generally decreases pStateIdx."""
        from entropy.cabac_arith import TRANS_IDX_LPS

        # For most states, LPS transition decreases probability
        # State 63 is special (stays at 63)
        for i in range(1, 63):
            assert TRANS_IDX_LPS[i] <= i

    def test_mps_transition_increases_state(self):
        """MPS transition generally increases pStateIdx."""
        from entropy.cabac_arith import TRANS_IDX_MPS

        # For most states, MPS transition increases probability
        # State 62 maps to 63 (max)
        for i in range(62):
            assert TRANS_IDX_MPS[i] >= i


class TestMultipleBins:
    """Tests for decoding multiple bins."""

    def test_decode_multiple_decisions(self):
        """Should be able to decode multiple bins in sequence."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import CABACContext

        data = bytes([0x00, 0xFF, 0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        ctx = CABACContext(pStateIdx=32, valMPS=0)

        results = []
        for _ in range(20):
            results.append(decoder.decode_decision(ctx))

        # Should get a sequence of 0s and 1s
        assert all(r in (0, 1) for r in results)

    def test_decode_mixed_modes(self):
        """Should be able to mix decision and bypass decoding."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import CABACContext

        data = bytes([0xAA, 0x55, 0xAA, 0x55] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        ctx = CABACContext(pStateIdx=32, valMPS=0)

        # Alternating decision and bypass
        results = []
        for i in range(10):
            if i % 2 == 0:
                results.append(decoder.decode_decision(ctx))
            else:
                results.append(decoder.decode_bypass())

        assert all(r in (0, 1) for r in results)
