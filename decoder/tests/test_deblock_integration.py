# h264/decoder/tests/test_deblock_integration.py
"""RED TESTS: Deblocking filter integration into decoder.

The deblocking filter module exists but is not integrated into
the decoder pipeline. These tests verify the integration.

These tests SHOULD FAIL until deblocking is integrated.
"""

import pytest
import numpy as np

from decoder.decoder import H264Decoder, DecoderState


class TestDeblockingIntegration:
    """Tests for deblocking filter in decoder pipeline."""

    def test_decoder_has_deblock_option(self):
        """Decoder should have option to enable/disable deblocking."""
        decoder = H264Decoder()

        assert hasattr(decoder, 'deblocking_enabled'), \
            "Decoder should have deblocking_enabled attribute"

    def test_decoder_calls_deblock_after_frame(self):
        """Decoder should apply deblocking after reconstructing frame."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Should have method to deblock entire frame
        assert hasattr(decoder, '_deblock_frame'), \
            "Decoder should have _deblock_frame method"

    def test_decoder_state_tracks_block_info(self):
        """DecoderState should track info needed for deblocking."""
        state = DecoderState()

        # Should track per-MB/block info for bS calculation
        assert hasattr(state, 'mb_types'), \
            "DecoderState should track mb_types for deblocking"

        assert hasattr(state, 'mb_coeffs'), \
            "DecoderState should track coefficient flags for deblocking"

    def test_deblock_respects_slice_header(self):
        """Deblocking should use slice header parameters."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Should have method that reads slice deblocking params
        assert hasattr(decoder, '_get_deblock_params'), \
            "Decoder should have _get_deblock_params method"


class TestDeblockingDisabled:
    """Tests for disable_deblocking_filter_idc."""

    def test_deblock_disabled_when_idc_1(self):
        """disable_deblocking_filter_idc=1 disables all deblocking."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Decoder should respect disable flag
        assert hasattr(decoder, '_should_deblock_slice'), \
            "Decoder should have _should_deblock_slice method"

    def test_deblock_disabled_at_slice_boundary_when_idc_2(self):
        """disable_deblocking_filter_idc=2 disables at slice boundaries."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        # Should have method to check slice boundary
        assert hasattr(decoder, '_is_slice_boundary'), \
            "Decoder should have _is_slice_boundary method"


class TestDeblockingOutput:
    """Tests for deblocked output quality."""

    def test_deblock_reduces_blocking_artifacts(self):
        """Deblocking should smooth artificial block edges."""
        # Create frame with artificial blocking
        frame_luma = np.zeros((32, 32), dtype=np.uint8)
        # Create checkerboard of 8x8 blocks
        for y in range(0, 32, 8):
            for x in range(0, 32, 8):
                val = 100 if ((x // 8) + (y // 8)) % 2 == 0 else 200
                frame_luma[y:y+8, x:x+8] = val

        from deblock.deblock import deblock_frame

        # This function should exist and process entire frame
        deblocked = deblock_frame(
            luma=frame_luma,
            cb=np.full((16, 16), 128, dtype=np.uint8),
            cr=np.full((16, 16), 128, dtype=np.uint8),
            mb_info=None,  # Would contain per-MB info
            qp=26,
        )

        # After deblocking, edges should be smoother
        # Check that edge pixels are no longer exactly 100 or 200
        edge_pixel = deblocked[0][7, 8]  # Luma at block boundary
        assert edge_pixel != 100 and edge_pixel != 200, \
            "Edge pixels should be smoothed by deblocking"


class TestPerMBDeblocking:
    """Tests for macroblock-level deblocking."""

    def test_deblock_mb_uses_neighbor_info(self):
        """Deblocking a MB needs neighbor MB information."""
        from deblock.deblock import deblock_macroblock_in_frame

        # This function should exist and take neighbor info
        assert callable(deblock_macroblock_in_frame), \
            "deblock_macroblock_in_frame should exist"

    def test_deblock_order_is_raster(self):
        """MBs should be deblocked in raster scan order."""
        from deblock.deblock import get_deblock_mb_order

        # Function to get MB processing order
        order = get_deblock_mb_order(width_mbs=2, height_mbs=2)

        # Should be raster: (0,0), (1,0), (0,1), (1,1)
        assert order == [(0, 0), (1, 0), (0, 1), (1, 1)]
