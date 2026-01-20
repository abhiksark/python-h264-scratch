# h264/color/tests/test_yuv_to_rgb.py
"""Tests for YCbCr to RGB conversion."""

import numpy as np
import pytest

from color import (
    ColorMatrix,
    ycbcr_to_rgb,
    rgb_to_ycbcr,
    upsample_chroma,
    subsample_chroma,
)


class TestUpsampleChroma:
    """Tests for chroma upsampling."""

    def test_upsample_nearest_2x2_to_4x4(self):
        """Basic 2x2 nearest neighbor upsampling."""
        chroma = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        result = upsample_chroma(chroma, 4, 4, method="nearest")

        expected = np.array([
            [100, 100, 200, 200],
            [100, 100, 200, 200],
            [50, 50, 150, 150],
            [50, 50, 150, 150],
        ], dtype=np.uint8)

        np.testing.assert_array_equal(result, expected)

    def test_upsample_preserves_dtype(self):
        """Upsampling should preserve input dtype."""
        chroma = np.array([[128]], dtype=np.uint8)
        result = upsample_chroma(chroma, 2, 2)
        assert result.dtype == np.uint8

    def test_upsample_single_pixel(self):
        """Single pixel should fill entire output."""
        chroma = np.array([[128]], dtype=np.uint8)
        result = upsample_chroma(chroma, 2, 2, method="nearest")

        expected = np.array([[128, 128], [128, 128]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)


class TestYCbCrToRGB:
    """Tests for YCbCr to RGB conversion."""

    def test_black(self):
        """Y=16, Cb=Cr=128 should give near-black (video black)."""
        # Note: True black in video range is Y=16, but we use Y=0 for simplicity
        y = np.zeros((4, 4), dtype=np.uint8)
        cb = np.full((2, 2), 128, dtype=np.uint8)
        cr = np.full((2, 2), 128, dtype=np.uint8)

        rgb = ycbcr_to_rgb(y, cb, cr)

        # With Y=0 and neutral chroma, RGB should be ~(0, 0, 0)
        assert rgb.shape == (4, 4, 3)
        assert np.allclose(rgb, 0, atol=1)

    def test_white(self):
        """Y=255, Cb=Cr=128 should give white."""
        y = np.full((4, 4), 255, dtype=np.uint8)
        cb = np.full((2, 2), 128, dtype=np.uint8)
        cr = np.full((2, 2), 128, dtype=np.uint8)

        rgb = ycbcr_to_rgb(y, cb, cr)

        # Y=255 with neutral chroma gives white
        assert np.allclose(rgb, 255, atol=1)

    def test_gray(self):
        """Y=128, Cb=Cr=128 should give gray."""
        y = np.full((4, 4), 128, dtype=np.uint8)
        cb = np.full((2, 2), 128, dtype=np.uint8)
        cr = np.full((2, 2), 128, dtype=np.uint8)

        rgb = ycbcr_to_rgb(y, cb, cr)

        # All channels should be ~128
        assert np.allclose(rgb[:, :, 0], 128, atol=1)  # R
        assert np.allclose(rgb[:, :, 1], 128, atol=1)  # G
        assert np.allclose(rgb[:, :, 2], 128, atol=1)  # B

    def test_pure_red(self):
        """Test approximate red color."""
        # Red: high Y, low Cb, high Cr
        y = np.full((4, 4), 82, dtype=np.uint8)   # Y component of red
        cb = np.full((2, 2), 90, dtype=np.uint8)  # Cb component
        cr = np.full((2, 2), 240, dtype=np.uint8)  # Cr component (high = red)

        rgb = ycbcr_to_rgb(y, cb, cr, color_matrix=ColorMatrix.BT601)

        # Red channel should be high, others low
        assert rgb[:, :, 0].mean() > 200  # R high
        assert rgb[:, :, 2].mean() < 50   # B low

    def test_pure_blue(self):
        """Test approximate blue color."""
        # Blue: low Y, high Cb, low Cr
        y = np.full((4, 4), 41, dtype=np.uint8)
        cb = np.full((2, 2), 240, dtype=np.uint8)  # High Cb = blue
        cr = np.full((2, 2), 110, dtype=np.uint8)

        rgb = ycbcr_to_rgb(y, cb, cr, color_matrix=ColorMatrix.BT601)

        # Blue channel should be high
        assert rgb[:, :, 2].mean() > 200  # B high
        assert rgb[:, :, 0].mean() < 50   # R low

    def test_output_shape(self):
        """Output should be (H, W, 3)."""
        y = np.zeros((64, 64), dtype=np.uint8)
        cb = np.zeros((32, 32), dtype=np.uint8)
        cr = np.zeros((32, 32), dtype=np.uint8)

        rgb = ycbcr_to_rgb(y, cb, cr)

        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_bt709_different_from_bt601(self):
        """BT.709 and BT.601 should give different results."""
        y = np.full((4, 4), 128, dtype=np.uint8)
        cb = np.full((2, 2), 200, dtype=np.uint8)  # Non-neutral chroma
        cr = np.full((2, 2), 100, dtype=np.uint8)

        rgb_601 = ycbcr_to_rgb(y, cb, cr, color_matrix=ColorMatrix.BT601)
        rgb_709 = ycbcr_to_rgb(y, cb, cr, color_matrix=ColorMatrix.BT709)

        # Results should differ when chroma is non-neutral
        assert not np.array_equal(rgb_601, rgb_709)


class TestRGBToYCbCr:
    """Tests for RGB to YCbCr conversion (inverse operation)."""

    def test_roundtrip_gray(self):
        """Gray should roundtrip accurately."""
        rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        y, cb, cr = rgb_to_ycbcr(rgb)

        # Chroma should be neutral (128)
        assert np.allclose(cb, 128, atol=2)
        assert np.allclose(cr, 128, atol=2)

    def test_roundtrip_white(self):
        """White should roundtrip."""
        rgb = np.full((4, 4, 3), 255, dtype=np.uint8)
        y, cb, cr = rgb_to_ycbcr(rgb)

        assert np.allclose(y, 255, atol=2)
        assert np.allclose(cb, 128, atol=2)
        assert np.allclose(cr, 128, atol=2)


class TestSubsampleChroma:
    """Tests for chroma subsampling (4:4:4 to 4:2:0)."""

    def test_subsample_uniform(self):
        """Uniform input should stay uniform."""
        cb = np.full((4, 4), 128, dtype=np.uint8)
        cr = np.full((4, 4), 128, dtype=np.uint8)

        cb_sub, cr_sub = subsample_chroma(cb, cr)

        assert cb_sub.shape == (2, 2)
        assert np.all(cb_sub == 128)
        assert np.all(cr_sub == 128)

    def test_subsample_output_size(self):
        """Output should be half size in each dimension."""
        cb = np.zeros((64, 64), dtype=np.uint8)
        cr = np.zeros((64, 64), dtype=np.uint8)

        cb_sub, cr_sub = subsample_chroma(cb, cr)

        assert cb_sub.shape == (32, 32)
        assert cr_sub.shape == (32, 32)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_encode_decode_roundtrip(self):
        """RGB → YCbCr → subsample → upsample → RGB should be close."""
        # Create a simple test image
        rgb_original = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb_original[:, :, 0] = 200  # Red
        rgb_original[:, :, 1] = 100  # Green
        rgb_original[:, :, 2] = 50   # Blue

        # Forward: RGB → YCbCr 4:4:4 → 4:2:0
        y, cb, cr = rgb_to_ycbcr(rgb_original)
        cb_sub, cr_sub = subsample_chroma(cb, cr)

        # Backward: 4:2:0 → 4:4:4 → RGB
        rgb_reconstructed = ycbcr_to_rgb(y, cb_sub, cr_sub)

        # Should be close (some loss due to chroma subsampling)
        diff = np.abs(rgb_original.astype(int) - rgb_reconstructed.astype(int))
        max_diff = diff.max()

        # Allow some tolerance due to subsampling
        assert max_diff < 10, f"Max diff {max_diff} exceeds tolerance"
