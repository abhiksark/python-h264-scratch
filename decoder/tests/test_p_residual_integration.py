# h264/decoder/tests/test_p_residual_integration.py
"""RED TESTS: P-macroblock residual decoding integration.

These tests verify that the decoder properly parses and applies
residual data for P-macroblocks. Currently the decoder has TODO
comments and ignores residuals.

These tests SHOULD FAIL until residual parsing is implemented.
"""

import pytest
import numpy as np

from bitstream import BitWriter
from decoder.decoder import H264Decoder


class TestPMBResidualDecoding:
    """Tests for P-MB residual parsing in decoder."""

    def test_decoder_parses_cbp_for_p_mb(self):
        """Decoder should parse coded_block_pattern for P-MBs."""
        from decoder.decoder import H264Decoder

        # This test verifies that _decode_p_macroblock reads CBP
        # Currently it doesn't - it just skips to reconstruction
        decoder = H264Decoder()

        # Check that decoder has method to parse P-MB with residual
        assert hasattr(decoder, '_parse_p_mb_residual'), \
            "Decoder should have _parse_p_mb_residual method"

    def test_decoder_applies_residual_to_prediction(self):
        """Decoder should add residual to inter prediction."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Decoder should have method that combines prediction + residual
        assert hasattr(decoder, '_apply_p_residual'), \
            "Decoder should have _apply_p_residual method"

    def test_p_16x16_with_nonzero_cbp(self):
        """P_L0_16x16 with CBP!=0 should decode residual blocks."""
        # This would require constructing a valid bitstream with:
        # - P_L0_16x16 mb_type
        # - non-zero CBP
        # - Residual coefficient data
        # Then verifying the output differs from prediction-only

        # For now, test that the infrastructure exists
        from inter.p_macroblock import decode_inter_cbp, PResidual

        # These exist, but decoder doesn't use them yet
        assert callable(decode_inter_cbp)
        assert PResidual is not None

        # The actual integration test would be:
        # 1. Create reference frame with known values
        # 2. Create P-frame bitstream with residual
        # 3. Decode and verify residual was applied
        pytest.skip("Full integration test requires bitstream construction")


class TestResidualBlockDecoding:
    """Tests for 4x4 residual block decoding in P-MBs."""

    def test_decode_luma_residual_blocks(self):
        """Decoder should decode luma 4x4 blocks based on CBP."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Should have method to decode luma residual for P-MB
        assert hasattr(decoder, '_decode_p_luma_residual'), \
            "Decoder should have _decode_p_luma_residual method"

    def test_decode_chroma_residual_blocks(self):
        """Decoder should decode chroma DC and AC based on CBP."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Should have method to decode chroma residual for P-MB
        assert hasattr(decoder, '_decode_p_chroma_residual'), \
            "Decoder should have _decode_p_chroma_residual method"

    def test_qp_delta_updates_qp(self):
        """mb_qp_delta should update the slice QP."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Decoder state should track current QP that gets updated
        # per macroblock when CBP != 0
        assert hasattr(decoder.state, 'current_mb_qp'), \
            "DecoderState should track current_mb_qp"
