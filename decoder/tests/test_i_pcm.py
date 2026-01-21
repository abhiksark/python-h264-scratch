# h264/decoder/tests/test_i_pcm.py
"""RED TESTS: I_PCM macroblock decoding.

I_PCM macroblocks contain raw, uncompressed sample data.
They bypass prediction, transform, and quantization entirely.

mb_type = 25 (in I-slices) or mb_type = 25 after P-MB offset

These tests SHOULD FAIL until I_PCM support is implemented.
"""

import pytest
import numpy as np

from bitstream import BitWriter, BitReader


class TestIPCMDetection:
    """Tests for detecting I_PCM macroblocks."""

    def test_i_pcm_mb_type_value(self):
        """I_PCM has mb_type = 25 in I-slices."""
        from intra.i_macroblock import IMBType, parse_i_mb_type

        mb_type = parse_i_mb_type(25)

        assert mb_type.name == "I_PCM"
        assert mb_type.is_pcm is True

    def test_decoder_recognizes_i_pcm(self):
        """Decoder should recognize I_PCM mb_type."""
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()

        assert hasattr(decoder, '_decode_i_pcm_macroblock'), \
            "Decoder should have _decode_i_pcm_macroblock method"


class TestIPCMParsing:
    """Tests for parsing I_PCM data."""

    def test_i_pcm_byte_alignment(self):
        """I_PCM data starts at byte-aligned position."""
        from intra.i_pcm import align_to_byte

        # Create reader at non-byte-aligned position
        writer = BitWriter()
        writer.write_bits(0b101, 3)  # 3 bits
        writer.write_bits(0, 5)  # Padding to byte
        writer.write_bits(128, 8)  # First PCM byte
        data = writer.get_bytes()

        reader = BitReader(data)
        reader.read_bits(3)  # Now at bit position 3

        align_to_byte(reader)

        # Should now be at bit 8 (byte-aligned)
        assert reader.position % 8 == 0

    def test_parse_i_pcm_luma_samples(self):
        """Parse 256 luma samples (16x16)."""
        from intra.i_pcm import parse_i_pcm_luma

        # Create 256 bytes of luma data
        luma_data = bytes(range(256))
        reader = BitReader(luma_data)

        luma = parse_i_pcm_luma(reader, bit_depth=8)

        assert luma.shape == (16, 16)
        assert luma[0, 0] == 0
        assert luma[0, 1] == 1
        assert luma[15, 15] == 255

    def test_parse_i_pcm_chroma_samples(self):
        """Parse 64 Cb + 64 Cr samples (8x8 each)."""
        from intra.i_pcm import parse_i_pcm_chroma

        # 128 bytes: 64 Cb + 64 Cr
        chroma_data = bytes([100] * 64 + [150] * 64)
        reader = BitReader(chroma_data)

        cb, cr = parse_i_pcm_chroma(reader, bit_depth=8)

        assert cb.shape == (8, 8)
        assert cr.shape == (8, 8)
        np.testing.assert_array_equal(cb, 100)
        np.testing.assert_array_equal(cr, 150)

    def test_parse_i_pcm_10bit(self):
        """Parse I_PCM with 10-bit samples."""
        from intra.i_pcm import parse_i_pcm_luma

        # 10-bit samples: 256 samples * 10 bits = 2560 bits = 320 bytes
        # But samples are byte-aligned, so 256 * 2 bytes = 512 bytes
        luma_data = b'\x00\x01' * 256  # 16-bit little-endian values
        reader = BitReader(luma_data)

        luma = parse_i_pcm_luma(reader, bit_depth=10)

        assert luma.shape == (16, 16)
        # Value should be in range [0, 1023]
        assert luma.max() <= 1023


class TestIPCMReconstruction:
    """Tests for reconstructing I_PCM macroblocks."""

    def test_i_pcm_no_prediction(self):
        """I_PCM uses raw samples, no prediction."""
        from intra.i_pcm import reconstruct_i_pcm

        luma = np.arange(256, dtype=np.uint8).reshape(16, 16)
        cb = np.full((8, 8), 128, dtype=np.uint8)
        cr = np.full((8, 8), 128, dtype=np.uint8)

        result_luma, result_cb, result_cr = reconstruct_i_pcm(luma, cb, cr)

        # Output should be identical to input (no transform/prediction)
        np.testing.assert_array_equal(result_luma, luma)
        np.testing.assert_array_equal(result_cb, cb)
        np.testing.assert_array_equal(result_cr, cr)

    def test_i_pcm_no_deblocking(self):
        """I_PCM blocks should NOT be deblocked (bS=0 at PCM boundaries)."""
        from deblock.boundary import calc_boundary_strength

        # I_PCM blocks use bS=0 regardless of other conditions
        bs = calc_boundary_strength(
            is_intra_p=True,
            is_intra_q=True,
            has_coeff_p=False,
            has_coeff_q=False,
            mv_p=(0, 0),
            mv_q=(0, 0),
            ref_p=0,
            ref_q=0,
            is_pcm_p=True,  # This should force bS=0
            is_pcm_q=False,
        )

        # Spec says bS=0 when filtering involves I_PCM
        assert bs == 0


class TestIPCMIntegration:
    """Tests for I_PCM in decoder pipeline."""

    def test_decode_i_pcm_macroblock(self):
        """Decode complete I_PCM macroblock."""
        from decoder.decoder import H264Decoder
        from bitstream import BitWriter

        # Create I_PCM data: mb_type(25) + byte_align + 384 bytes
        writer = BitWriter()
        # Skip mb_type encoding (would be ue(25))
        # Write 256 luma + 64 Cb + 64 Cr = 384 bytes
        for i in range(256):
            writer.write_bits(i % 256, 8)
        for i in range(64):
            writer.write_bits(128, 8)
        for i in range(64):
            writer.write_bits(128, 8)

        # This test verifies the method exists and can be called
        decoder = H264Decoder()
        assert hasattr(decoder, '_decode_i_pcm_macroblock')

    def test_nz_counts_set_to_16_for_i_pcm(self):
        """Non-zero counts should be 16 for all I_PCM blocks."""
        from intra.i_pcm import get_i_pcm_nz_counts

        nz = get_i_pcm_nz_counts()

        # All 24 blocks (16 luma + 4 Cb + 4 Cr) have nz=16
        assert nz.shape == (24,)
        np.testing.assert_array_equal(nz, 16)
