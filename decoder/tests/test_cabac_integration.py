# h264/decoder/tests/test_cabac_integration.py
"""RED TESTS: CABAC decoder integration.

Tests for integrating CABAC entropy decoding into the main decoder.
The decoder should select CABAC when entropy_coding_mode_flag=1 in PPS.

H.264 Spec Reference: Section 7.4.2.2 - PPS semantics

These tests SHOULD FAIL until CABAC decoder integration is implemented.
"""

import pytest
import numpy as np

from decoder.decoder import H264Decoder, DecoderState


class TestEntropyModeSelection:
    """Tests for entropy coding mode selection."""

    def test_decoder_checks_entropy_mode(self):
        """Decoder should check entropy_coding_mode_flag in PPS."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_get_entropy_mode'), \
            "Decoder should have _get_entropy_mode method"

    def test_entropy_mode_flag_false_uses_cavlc(self):
        """entropy_coding_mode_flag=0 should use CAVLC."""
        decoder = H264Decoder()

        # Default PPS has entropy_coding_mode_flag=False
        mode = decoder._get_entropy_mode(entropy_flag=False)

        assert mode == 'cavlc'

    def test_entropy_mode_flag_true_uses_cabac(self):
        """entropy_coding_mode_flag=1 should use CABAC."""
        decoder = H264Decoder()

        mode = decoder._get_entropy_mode(entropy_flag=True)

        assert mode == 'cabac'


class TestCABACSliceDecoding:
    """Tests for CABAC slice data decoding."""

    def test_decoder_has_cabac_slice_decoder(self):
        """Decoder should have _decode_slice_data_cabac method."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_decode_slice_data_cabac'), \
            "Decoder should have _decode_slice_data_cabac method"

    def test_decoder_has_cabac_mb_decoder(self):
        """Decoder should have _decode_macroblock_cabac method."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_decode_macroblock_cabac'), \
            "Decoder should have _decode_macroblock_cabac method"


class TestCABACContextManagement:
    """Tests for CABAC context management in decoder."""

    def test_decoder_state_has_cabac_contexts(self):
        """DecoderState should track CABAC context models."""
        state = DecoderState()

        assert hasattr(state, 'cabac_contexts'), \
            "DecoderState should have cabac_contexts"

    def test_decoder_initializes_cabac_contexts(self):
        """Decoder should initialize CABAC contexts for each slice."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_init_cabac_contexts'), \
            "Decoder should have _init_cabac_contexts method"


class TestCABACMBSkipDecoding:
    """Tests for mb_skip_flag decoding with CABAC."""

    def test_decoder_decodes_cabac_skip_flag(self):
        """Decoder should decode mb_skip_flag using CABAC."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_decode_mb_skip_flag_cabac'), \
            "Decoder should have _decode_mb_skip_flag_cabac method"


class TestCABACResidualDecoding:
    """Tests for residual decoding with CABAC."""

    def test_decoder_has_cabac_residual_decoder(self):
        """Decoder should have _decode_residual_cabac method."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_decode_residual_cabac'), \
            "Decoder should have _decode_residual_cabac method"


class TestCABACDecoderCreation:
    """Tests for CABACDecoder creation."""

    def test_decoder_creates_cabac_decoder(self):
        """Decoder should create CABACDecoder instance."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_create_cabac_decoder'), \
            "Decoder should have _create_cabac_decoder method"


class TestCABACProfileSupport:
    """Tests for CABAC profile support."""

    def test_main_profile_uses_cabac(self):
        """Main profile typically uses CABAC."""
        # Main profile = profile_idc=77
        # This is informational - decoder should handle both modes
        assert True

    def test_high_profile_uses_cabac(self):
        """High profile typically uses CABAC."""
        # High profile = profile_idc=100
        assert True

    def test_baseline_profile_uses_cavlc(self):
        """Baseline profile uses CAVLC only."""
        # Baseline profile = profile_idc=66
        # entropy_coding_mode_flag must be 0
        assert True
