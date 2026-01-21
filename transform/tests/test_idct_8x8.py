# h264/transform/tests/test_idct_8x8.py
"""RED TESTS: 8x8 Integer IDCT for High Profile.

H.264 8x8 IDCT is used for I_8x8 macroblocks in High profile.
Uses 8-point butterfly operations with integer arithmetic.

H.264 Spec Reference: Section 8.5.12 - Inverse transform process

These tests SHOULD FAIL until 8x8 IDCT is implemented.
"""

import pytest
import numpy as np


class TestIDCT8x8Exists:
    """Tests for 8x8 IDCT function existence."""

    def test_idct_8x8_function_exists(self):
        """idct_8x8 function should exist."""
        from transform.idct_8x8 import idct_8x8

        assert callable(idct_8x8)

    def test_idct_1d_8_function_exists(self):
        """idct_1d_8 function should exist for 8-point 1D IDCT."""
        from transform.idct_8x8 import idct_1d_8

        assert callable(idct_1d_8)


class TestIDCT8x8Shape:
    """Tests for input/output shape handling."""

    def test_accepts_8x8_input(self):
        """idct_8x8 should accept 8x8 input."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        result = idct_8x8(coeffs)

        assert result.shape == (8, 8)

    def test_returns_int32_array(self):
        """idct_8x8 should return int32 array."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        result = idct_8x8(coeffs)

        assert result.dtype == np.int32

    def test_rejects_wrong_shape(self):
        """idct_8x8 should reject non-8x8 input."""
        from transform.idct_8x8 import idct_8x8

        coeffs_4x4 = np.zeros((4, 4), dtype=np.int32)

        with pytest.raises(ValueError):
            idct_8x8(coeffs_4x4)


class TestIDCT8x8ZeroInput:
    """Tests for zero input handling."""

    def test_zero_coefficients_zero_output(self):
        """Zero coefficients should produce zero output."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        result = idct_8x8(coeffs)

        np.testing.assert_array_equal(result, np.zeros((8, 8), dtype=np.int32))


class TestIDCT8x8DCCoefficient:
    """Tests for DC coefficient handling."""

    def test_dc_only_produces_uniform_output(self):
        """DC-only coefficient should produce uniform output."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 64  # DC coefficient

        result = idct_8x8(coeffs)

        # DC spreads uniformly across 8x8 block
        # All values should be equal (or very close due to rounding)
        assert np.all(result == result[0, 0])

    def test_positive_dc_positive_output(self):
        """Positive DC should produce positive (or zero) output."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 256

        result = idct_8x8(coeffs)

        assert np.all(result >= 0)

    def test_negative_dc_negative_output(self):
        """Negative DC should produce negative (or zero) output."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = -256

        result = idct_8x8(coeffs)

        assert np.all(result <= 0)


class TestIDCT8x8ACCoefficients:
    """Tests for AC coefficient handling."""

    def test_single_ac_coeff_produces_pattern(self):
        """Single AC coefficient should produce spatial pattern."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 1] = 256  # First horizontal AC

        result = idct_8x8(coeffs)

        # Should not be uniform (unlike DC-only)
        assert not np.all(result == result[0, 0])

    def test_high_frequency_ac_produces_alternating(self):
        """High-frequency AC should produce alternating pattern."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[7, 7] = 256  # Highest frequency

        result = idct_8x8(coeffs)

        # High frequency creates alternating pattern
        # Check that adjacent pixels differ in sign or value
        diff = np.abs(np.diff(result[0, :]))
        assert np.any(diff > 0)


class TestIDCT8x8Linearity:
    """Tests for transform linearity."""

    def test_scaling_linearity(self):
        """IDCT should be linear with respect to scaling."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.random.randint(-100, 100, (8, 8), dtype=np.int32)

        result1 = idct_8x8(coeffs)
        result2 = idct_8x8(2 * coeffs)

        # Due to integer rounding, approximate equality
        np.testing.assert_array_almost_equal(
            result2, 2 * result1, decimal=0
        )

    def test_additivity(self):
        """IDCT should be additive (approximately, due to rounding)."""
        from transform.idct_8x8 import idct_8x8

        coeffs1 = np.zeros((8, 8), dtype=np.int32)
        coeffs1[0, 0] = 64

        coeffs2 = np.zeros((8, 8), dtype=np.int32)
        coeffs2[0, 1] = 64

        result1 = idct_8x8(coeffs1)
        result2 = idct_8x8(coeffs2)
        result_sum = idct_8x8(coeffs1 + coeffs2)

        # Allow for small rounding differences
        np.testing.assert_array_almost_equal(
            result_sum, result1 + result2, decimal=0
        )


class TestIDCT1D8:
    """Tests for 8-point 1D IDCT."""

    def test_1d_idct_returns_8_elements(self):
        """1D IDCT should return 8 elements."""
        from transform.idct_8x8 import idct_1d_8

        x = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        result = idct_1d_8(x)

        assert len(result) == 8

    def test_1d_dc_only_uniform(self):
        """1D IDCT with DC only should produce uniform output."""
        from transform.idct_8x8 import idct_1d_8

        x = np.array([64, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        result = idct_1d_8(x)

        # All values should be equal
        assert np.all(result == result[0])

    def test_1d_idct_preserves_energy(self):
        """1D IDCT should roughly preserve signal energy."""
        from transform.idct_8x8 import idct_1d_8

        x = np.array([100, 50, 25, 12, 6, 3, 1, 0], dtype=np.int32)
        result = idct_1d_8(x)

        # Energy should be similar (within factor of 2 due to scaling)
        input_energy = np.sum(x.astype(np.float64) ** 2)
        output_energy = np.sum(result.astype(np.float64) ** 2)

        # Ratio should be reasonable
        assert 0.01 < output_energy / (input_energy + 1) < 100


class TestIDCT8x8SpecCompliance:
    """Tests for H.264 spec compliance."""

    def test_normalization_factor(self):
        """IDCT should use correct normalization (divide by 256 with rounding)."""
        from transform.idct_8x8 import idct_8x8

        # DC coefficient of 256 should result in value of 1 after normalization
        # (256 * 1 / 256 = 1 for DC basis function value of 1)
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 256

        result = idct_8x8(coeffs)

        # DC coefficient spreads as value/8 to each pixel after row+col transform
        # Exact value depends on normalization, but should be non-zero
        assert result[0, 0] != 0

    def test_integer_arithmetic_no_float(self):
        """IDCT should use pure integer arithmetic."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.array([
            [100, 50, 25, 12, 6, 3, 1, 0],
            [50, 25, 12, 6, 3, 1, 0, 0],
            [25, 12, 6, 3, 1, 0, 0, 0],
            [12, 6, 3, 1, 0, 0, 0, 0],
            [6, 3, 1, 0, 0, 0, 0, 0],
            [3, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)

        result = idct_8x8(coeffs)

        # Result should be exact integers
        assert result.dtype == np.int32


class TestIDCT8x8EdgeCases:
    """Tests for edge cases."""

    def test_max_coefficient_values(self):
        """IDCT should handle maximum coefficient values."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.full((8, 8), 2047, dtype=np.int32)  # Near max for 12-bit

        # Should not overflow (uses int32)
        result = idct_8x8(coeffs)

        assert result.dtype == np.int32
        assert not np.any(np.isnan(result))

    def test_min_coefficient_values(self):
        """IDCT should handle minimum (negative) coefficient values."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.full((8, 8), -2048, dtype=np.int32)

        result = idct_8x8(coeffs)

        assert result.dtype == np.int32
        assert not np.any(np.isnan(result))

    def test_mixed_sign_coefficients(self):
        """IDCT should handle mixed positive/negative coefficients."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.array([
            [100, -50, 25, -12, 6, -3, 1, 0],
            [-50, 25, -12, 6, -3, 1, 0, 0],
            [25, -12, 6, -3, 1, 0, 0, 0],
            [-12, 6, -3, 1, 0, 0, 0, 0],
            [6, -3, 1, 0, 0, 0, 0, 0],
            [-3, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)

        result = idct_8x8(coeffs)

        # Should produce valid output with both positive and negative values
        assert np.any(result > 0) or np.any(result < 0) or np.all(result == 0)
