# h264/transform/tests/test_idct.py
"""Tests for 4x4 inverse transform and Hadamard transforms."""

import numpy as np
import pytest

from transform import (
    idct_4x4,
    idct_1d,
    hadamard_4x4,
    hadamard_2x2,
    inverse_hadamard_4x4,
    inverse_hadamard_2x2,
    forward_4x4,
)


class TestIDCT1D:
    """Tests for 1D inverse transform."""

    def test_idct_1d_zero(self):
        """Zero input should give zero output."""
        x = np.array([0, 0, 0, 0], dtype=np.int32)
        result = idct_1d(x)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_idct_1d_dc_only(self):
        """DC-only input should give constant output."""
        x = np.array([4, 0, 0, 0], dtype=np.int32)
        result = idct_1d(x)
        # All outputs should be equal for DC-only
        assert result[0] == result[1] == result[2] == result[3]

    def test_idct_1d_shape(self):
        """Output should have same shape as input."""
        x = np.array([1, 2, 3, 4], dtype=np.int32)
        result = idct_1d(x)
        assert result.shape == (4,)


class TestIDCT4x4:
    """Tests for 4x4 inverse transform."""

    def test_idct_4x4_zero(self):
        """Zero coefficients should give zero output."""
        coeffs = np.zeros((4, 4), dtype=np.int32)
        result = idct_4x4(coeffs)
        np.testing.assert_array_equal(result, np.zeros((4, 4)))

    def test_idct_4x4_dc_only(self):
        """DC-only input should give flat output."""
        coeffs = np.zeros((4, 4), dtype=np.int32)
        coeffs[0, 0] = 64  # DC coefficient

        result = idct_4x4(coeffs)

        # All values should be approximately equal for DC-only
        # Due to normalization (>>6), DC=64 should give ~1 everywhere
        unique_values = np.unique(result)
        assert len(unique_values) <= 2  # Allow for rounding differences

    def test_idct_4x4_shape(self):
        """Output should be 4x4."""
        coeffs = np.ones((4, 4), dtype=np.int32)
        result = idct_4x4(coeffs)
        assert result.shape == (4, 4)

    def test_idct_4x4_dtype(self):
        """Output should be int32."""
        coeffs = np.ones((4, 4), dtype=np.int32)
        result = idct_4x4(coeffs)
        assert result.dtype == np.int32

    def test_idct_4x4_invalid_shape(self):
        """Non-4x4 input should raise."""
        coeffs = np.ones((3, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            idct_4x4(coeffs)

    def test_idct_4x4_large_dc(self):
        """Large DC should produce reasonable output."""
        coeffs = np.zeros((4, 4), dtype=np.int32)
        coeffs[0, 0] = 1024  # Large DC

        result = idct_4x4(coeffs)

        # Output should be positive and uniform-ish
        assert result.min() >= 0
        assert result.max() <= 1024  # Bounded


class TestHadamard4x4:
    """Tests for 4x4 Hadamard transform."""

    def test_hadamard_4x4_zero(self):
        """Zero input gives zero output."""
        dc = np.zeros((4, 4), dtype=np.int32)
        result = hadamard_4x4(dc)
        np.testing.assert_array_equal(result, np.zeros((4, 4)))

    def test_hadamard_4x4_shape(self):
        """Output should be 4x4."""
        dc = np.ones((4, 4), dtype=np.int32)
        result = hadamard_4x4(dc)
        assert result.shape == (4, 4)

    def test_hadamard_4x4_self_inverse(self):
        """Hadamard applied twice should return scaled original."""
        dc = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ], dtype=np.int32)

        # Apply twice
        once = hadamard_4x4(dc)
        twice = hadamard_4x4(once)

        # Should be original scaled by 16 (4x4)
        np.testing.assert_array_equal(twice, dc * 16)

    def test_hadamard_4x4_invalid_shape(self):
        """Non-4x4 input should raise."""
        dc = np.ones((3, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            hadamard_4x4(dc)

    def test_hadamard_4x4_uniform(self):
        """Uniform input should concentrate in DC."""
        dc = np.ones((4, 4), dtype=np.int32) * 4
        result = hadamard_4x4(dc)

        # DC (0,0) should have the energy
        assert result[0, 0] == 64  # 4 * 16
        # Other positions should be zero
        result_no_dc = result.copy()
        result_no_dc[0, 0] = 0
        assert np.all(result_no_dc == 0)


class TestHadamard2x2:
    """Tests for 2x2 Hadamard transform (chroma)."""

    def test_hadamard_2x2_zero(self):
        """Zero input gives zero output."""
        dc = np.zeros((2, 2), dtype=np.int32)
        result = hadamard_2x2(dc)
        np.testing.assert_array_equal(result, np.zeros((2, 2)))

    def test_hadamard_2x2_shape(self):
        """Output should be 2x2."""
        dc = np.ones((2, 2), dtype=np.int32)
        result = hadamard_2x2(dc)
        assert result.shape == (2, 2)

    def test_hadamard_2x2_self_inverse(self):
        """Hadamard applied twice should return scaled original."""
        dc = np.array([[1, 2], [3, 4]], dtype=np.int32)

        once = hadamard_2x2(dc)
        twice = hadamard_2x2(once)

        # Should be original scaled by 4 (2x2)
        np.testing.assert_array_equal(twice, dc * 4)

    def test_hadamard_2x2_invalid_shape(self):
        """Non-2x2 input should raise."""
        dc = np.ones((3, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            hadamard_2x2(dc)


class TestInverseHadamard:
    """Tests for inverse Hadamard transforms."""

    def test_inverse_hadamard_4x4_basic(self):
        """Inverse Hadamard should work."""
        dc = np.array([
            [16, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.int32)

        result = inverse_hadamard_4x4(dc)

        # DC-only input should give uniform output
        expected = np.ones((4, 4), dtype=np.int32) * 16
        np.testing.assert_array_equal(result, expected)

    def test_inverse_hadamard_2x2_basic(self):
        """Inverse Hadamard 2x2 should work."""
        dc = np.array([[4, 0], [0, 0]], dtype=np.int32)
        result = inverse_hadamard_2x2(dc)

        # DC-only should give uniform
        expected = np.ones((2, 2), dtype=np.int32) * 4
        np.testing.assert_array_equal(result, expected)


class TestForwardTransform:
    """Tests for forward transform (for validation)."""

    def test_forward_4x4_zero(self):
        """Zero input gives zero output."""
        spatial = np.zeros((4, 4), dtype=np.int32)
        result = forward_4x4(spatial)
        np.testing.assert_array_equal(result, np.zeros((4, 4)))

    def test_forward_4x4_shape(self):
        """Output should be 4x4."""
        spatial = np.ones((4, 4), dtype=np.int32)
        result = forward_4x4(spatial)
        assert result.shape == (4, 4)

    def test_forward_4x4_invalid_shape(self):
        """Non-4x4 should raise."""
        spatial = np.ones((3, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            forward_4x4(spatial)


class TestIntegration:
    """Integration tests for transform pipeline."""

    def test_dc_energy_preservation(self):
        """DC coefficient should capture average energy."""
        # Flat block with value 10
        coeffs = np.zeros((4, 4), dtype=np.int32)
        coeffs[0, 0] = 640  # 10 * 64 for normalization

        result = idct_4x4(coeffs)

        # Should give approximately 10 everywhere
        assert np.allclose(result, 10, atol=1)

    def test_high_frequency_variation(self):
        """High frequency coefficients should create variation."""
        coeffs = np.zeros((4, 4), dtype=np.int32)
        coeffs[0, 0] = 640  # DC
        coeffs[0, 1] = 128   # AC

        result = idct_4x4(coeffs)

        # Should have variation across columns
        assert result.std() > 0

    def test_typical_block(self):
        """Test with typical I-frame coefficient pattern."""
        # Large DC, decreasing AC
        coeffs = np.array([
            [512, -64, 32, 0],
            [-32, 16, 0, 0],
            [16, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.int32)

        result = idct_4x4(coeffs)

        # Result should have reasonable values
        assert result.min() > -256
        assert result.max() < 256

        # DC should dominate, so values should be mostly positive
        assert result.mean() > 0
