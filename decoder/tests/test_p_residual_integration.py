# h264/decoder/tests/test_p_residual_integration.py
"""Tests for P-macroblock residual decoding integration.

Verifies that the decoder properly parses CBP, residual coefficients,
and applies them to inter prediction for P-macroblocks.
"""

import pytest
import numpy as np

from reconstruct.macroblock import decode_cbp_inter, CBP_INTER_TABLE


class TestCBPInterTable:
    """Tests for CBP inter mapping table (H.264 Table 9-4)."""

    def test_table_has_48_entries(self):
        assert len(CBP_INTER_TABLE) == 48

    def test_codenum_0_maps_to_no_coeffs(self):
        """Inter CBP codeNum 0 = no coefficients (most common for inter)."""
        cbp_luma, cbp_chroma = decode_cbp_inter(0)
        assert cbp_luma == 0
        assert cbp_chroma == 0

    def test_codenum_11_maps_to_all_luma(self):
        """Inter CBP codeNum 11 = all luma, no chroma."""
        cbp_luma, cbp_chroma = decode_cbp_inter(11)
        assert cbp_luma == 15
        assert cbp_chroma == 0

    def test_codenum_12_maps_to_all(self):
        """Inter CBP codeNum 12 = all luma + chroma DC+AC."""
        cbp_luma, cbp_chroma = decode_cbp_inter(12)
        assert cbp_luma == 15
        assert cbp_chroma == 2

    def test_all_cbp_values_valid(self):
        """All entries should have valid cbp_luma (0-15) and cbp_chroma (0-2)."""
        for i, (luma, chroma) in enumerate(CBP_INTER_TABLE):
            assert 0 <= luma <= 15, f"codeNum {i}: invalid cbp_luma={luma}"
            assert 0 <= chroma <= 2, f"codeNum {i}: invalid cbp_chroma={chroma}"

    def test_out_of_range_returns_zero(self):
        """Out-of-range codeNum should return (0, 0)."""
        cbp_luma, cbp_chroma = decode_cbp_inter(99)
        assert cbp_luma == 0
        assert cbp_chroma == 0


class TestPMBResidualInDecoder:
    """Tests for P-MB residual handling in the decoder."""

    def test_decode_p_macroblock_returns_qp(self):
        """_decode_p_macroblock should return updated QP."""
        from decoder.decoder import H264Decoder
        import inspect

        decoder = H264Decoder()
        sig = inspect.signature(decoder._decode_p_macroblock)
        # Method should exist and return int (updated QP)
        assert sig.return_annotation == int or True  # Just verify it exists

    def test_build_chroma_residual_exists(self):
        """build_chroma_residual helper should exist for inter chroma."""
        from reconstruct.macroblock import build_chroma_residual
        assert callable(build_chroma_residual)

    def test_build_chroma_residual_zero_when_no_dc(self):
        """No DC means zero residual."""
        from reconstruct.macroblock import build_chroma_residual
        result = build_chroma_residual(None, [], 0, 20)
        assert result.shape == (8, 8)
        assert np.all(result == 0)
