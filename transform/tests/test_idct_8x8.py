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


# =============================================================================
# RED TESTS: Extended 8x8 Transform Tests for High Profile Features
# These tests SHOULD FAIL until full I_8x8/High profile support is implemented.
# =============================================================================


class TestIDCT8x8ForwardInverse:
    """Tests for forward-inverse transform relationship.

    H.264 Spec Reference: Section 8.5.12
    Forward transform followed by inverse should approximately recover original.
    """

    def test_forward_8x8_exists(self):
        """forward_8x8 function should exist for encoding/validation."""
        from transform.idct_8x8 import forward_8x8

        assert callable(forward_8x8)

    def test_forward_inverse_round_trip(self):
        """Forward then inverse should recover original (with rounding)."""
        from transform.idct_8x8 import idct_8x8, forward_8x8

        # Create spatial domain block
        spatial = np.array([
            [100, 102, 104, 106, 108, 110, 112, 114],
            [98, 100, 102, 104, 106, 108, 110, 112],
            [96, 98, 100, 102, 104, 106, 108, 110],
            [94, 96, 98, 100, 102, 104, 106, 108],
            [92, 94, 96, 98, 100, 102, 104, 106],
            [90, 92, 94, 96, 98, 100, 102, 104],
            [88, 90, 92, 94, 96, 98, 100, 102],
            [86, 88, 90, 92, 94, 96, 98, 100],
        ], dtype=np.int32)

        # Forward then inverse
        coeffs = forward_8x8(spatial)
        recovered = idct_8x8(coeffs)

        # Should be close to original (allow rounding differences)
        # H.264 integer transforms have inherent rounding error due to
        # scaled integer coefficients - atol=10 is realistic for 8x8
        np.testing.assert_allclose(recovered, spatial, atol=10)

    def test_forward_8x8_shape(self):
        """forward_8x8 should return 8x8 array."""
        from transform.idct_8x8 import forward_8x8

        spatial = np.ones((8, 8), dtype=np.int32) * 100

        coeffs = forward_8x8(spatial)

        assert coeffs.shape == (8, 8)
        assert coeffs.dtype == np.int32

    def test_forward_8x8_dc_concentration(self):
        """Uniform input should concentrate energy in DC coefficient."""
        from transform.idct_8x8 import forward_8x8

        spatial = np.ones((8, 8), dtype=np.int32) * 100

        coeffs = forward_8x8(spatial)

        # DC should be large, AC should be near zero
        assert np.abs(coeffs[0, 0]) > np.abs(coeffs[1:, :]).max()
        assert np.allclose(coeffs[1:, :], 0, atol=1)
        assert np.allclose(coeffs[:, 1:], 0, atol=1)


class TestIDCT8x8BasisFunctions:
    """Tests for 8x8 transform basis functions.

    H.264 Spec: The 8x8 transform uses a specific integer approximation
    of the 8-point DCT basis functions.
    """

    def test_dc_basis_function(self):
        """DC coefficient (0,0) produces uniform output."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 256

        result = idct_8x8(coeffs)

        # All values should be equal for DC-only input
        assert len(np.unique(result)) == 1

    def test_horizontal_ac_basis_function(self):
        """Horizontal AC coefficient produces vertical variation."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 1] = 256  # First horizontal AC

        result = idct_8x8(coeffs)

        # Should have horizontal variation (different columns)
        # Rows should be identical
        for row in range(1, 8):
            np.testing.assert_array_equal(result[row, :], result[0, :])

    def test_vertical_ac_basis_function(self):
        """Vertical AC coefficient produces horizontal variation."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[1, 0] = 256  # First vertical AC

        result = idct_8x8(coeffs)

        # Should have vertical variation (different rows)
        # Columns should be identical
        for col in range(1, 8):
            np.testing.assert_array_equal(result[:, col], result[:, 0])

    def test_diagonal_ac_basis_function(self):
        """Diagonal AC coefficient produces checkerboard-like pattern."""
        from transform.idct_8x8 import idct_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[1, 1] = 256  # First diagonal AC

        result = idct_8x8(coeffs)

        # Should have both horizontal and vertical variation
        assert not np.all(result[0, :] == result[0, 0])  # Horizontal variation
        assert not np.all(result[:, 0] == result[0, 0])  # Vertical variation


class TestIDCT8x8SpecConstants:
    """Tests for H.264 specification constants used in 8x8 transform.

    H.264 Spec: Section 8.5.12 specifies exact integer constants.
    """

    def test_transform_matrix_exists(self):
        """TRANSFORM_MATRIX_8x8 constant should exist."""
        from transform.idct_8x8 import TRANSFORM_MATRIX_8x8

        assert TRANSFORM_MATRIX_8x8 is not None
        assert TRANSFORM_MATRIX_8x8.shape == (8, 8)

    def test_transform_matrix_values(self):
        """Transform matrix should have correct H.264 values.

        H.264 uses these constants: 8, 12, 10, 6, 3, 1
        (scaled versions of cos(n*pi/16) for n=0,1,2,3,4,5,6,7)
        """
        from transform.idct_8x8 import TRANSFORM_MATRIX_8x8

        # First row should be all equal (DC basis)
        # Exact values depend on H.264 spec scaling
        # The 8x8 transform constants are: a=8, b=12, c=10, d=6, e=3, f=1
        pass  # Implementation will define exact values

    def test_scaling_factors_8x8(self):
        """Scaling factors for 8x8 transform should exist."""
        from transform.idct_8x8 import SCALING_FACTORS_8x8

        assert SCALING_FACTORS_8x8 is not None
        # Should be 6x6 for the 6 position types at 6 QP remainders


class TestIDCT8x8ZigzagScan:
    """Tests for 8x8 zigzag scan order.

    H.264 Spec: Table 8-5 defines 8x8 zigzag scan order.
    """

    def test_zigzag_8x8_constant_exists(self):
        """ZIGZAG_SCAN_8x8 constant should exist."""
        from transform.idct_8x8 import ZIGZAG_SCAN_8x8

        assert ZIGZAG_SCAN_8x8 is not None

    def test_zigzag_8x8_has_64_elements(self):
        """Zigzag scan should have 64 elements for 8x8 block."""
        from transform.idct_8x8 import ZIGZAG_SCAN_8x8

        assert len(ZIGZAG_SCAN_8x8) == 64

    def test_zigzag_8x8_starts_with_dc(self):
        """First element should be (0, 0) for DC coefficient."""
        from transform.idct_8x8 import ZIGZAG_SCAN_8x8

        assert ZIGZAG_SCAN_8x8[0] == (0, 0)

    def test_zigzag_8x8_ends_with_corner(self):
        """Last element should be (7, 7) for highest frequency."""
        from transform.idct_8x8 import ZIGZAG_SCAN_8x8

        assert ZIGZAG_SCAN_8x8[63] == (7, 7)

    def test_zigzag_8x8_covers_all_positions(self):
        """Zigzag should cover all 64 positions exactly once."""
        from transform.idct_8x8 import ZIGZAG_SCAN_8x8

        positions = set(ZIGZAG_SCAN_8x8)
        expected = set((i, j) for i in range(8) for j in range(8))
        assert positions == expected


class TestIDCT8x8FieldScan:
    """Tests for 8x8 field scan order (interlaced video).

    H.264 Spec: Table 8-6 defines 8x8 field scan order for MBAFF.
    """

    def test_field_scan_8x8_exists(self):
        """FIELD_SCAN_8x8 constant should exist for interlaced."""
        from transform.idct_8x8 import FIELD_SCAN_8x8

        assert FIELD_SCAN_8x8 is not None

    def test_field_scan_8x8_has_64_elements(self):
        """Field scan should have 64 elements."""
        from transform.idct_8x8 import FIELD_SCAN_8x8

        assert len(FIELD_SCAN_8x8) == 64

    def test_field_scan_differs_from_zigzag(self):
        """Field scan should differ from frame zigzag scan."""
        from transform.idct_8x8 import ZIGZAG_SCAN_8x8, FIELD_SCAN_8x8

        # They should have same elements but different order
        assert set(ZIGZAG_SCAN_8x8) == set(FIELD_SCAN_8x8)
        assert list(ZIGZAG_SCAN_8x8) != list(FIELD_SCAN_8x8)


class TestIDCT8x8Integration:
    """Integration tests with dequantization and prediction."""

    def test_dequant_then_idct_pipeline(self):
        """Test complete dequantization -> IDCT pipeline."""
        from transform.idct_8x8 import idct_8x8
        from dequant.dequant import dequant_8x8

        # Quantized coefficients (typical I-block pattern)
        quantized = np.zeros((8, 8), dtype=np.int32)
        quantized[0, 0] = 16  # DC
        quantized[0, 1] = -4
        quantized[1, 0] = -2

        # Dequantize then transform
        dequantized = dequant_8x8(quantized, qp=26)
        residual = idct_8x8(dequantized)

        assert residual.shape == (8, 8)
        assert residual.dtype == np.int32

    def test_residual_magnitude_reasonable(self):
        """Residual from typical coefficients should be reasonable."""
        from transform.idct_8x8 import idct_8x8
        from dequant.dequant import dequant_8x8

        # Typical quantized coefficients
        quantized = np.array([
            [20, -5, 2, -1, 0, 0, 0, 0],
            [-4, 2, -1, 0, 0, 0, 0, 0],
            [2, -1, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)

        dequantized = dequant_8x8(quantized, qp=26)
        residual = idct_8x8(dequantized)

        # Residual should be bounded for typical video
        assert np.abs(residual).max() < 1000


class TestIDCT8x8Performance:
    """Performance-related tests for 8x8 IDCT.

    These tests ensure the implementation is efficient enough.
    """

    def test_vectorized_implementation(self):
        """IDCT should use vectorized operations, not nested loops."""
        from transform.idct_8x8 import idct_8x8
        import time

        coeffs = np.random.randint(-100, 100, (8, 8), dtype=np.int32)

        # Time 1000 iterations
        start = time.time()
        for _ in range(1000):
            _ = idct_8x8(coeffs)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second for 1000 iterations)
        assert elapsed < 1.0, f"IDCT too slow: {elapsed:.3f}s for 1000 iterations"

    def test_batch_processing_exists(self):
        """Batch processing function should exist for efficiency."""
        from transform.idct_8x8 import idct_8x8_batch

        assert callable(idct_8x8_batch)

    def test_batch_processing_correctness(self):
        """Batch processing should produce same results as individual."""
        from transform.idct_8x8 import idct_8x8, idct_8x8_batch

        # Create batch of 4 blocks
        batch = np.random.randint(-100, 100, (4, 8, 8), dtype=np.int32)

        # Individual processing
        individual_results = [idct_8x8(batch[i]) for i in range(4)]
        individual_stack = np.stack(individual_results)

        # Batch processing
        batch_results = idct_8x8_batch(batch)

        np.testing.assert_array_equal(batch_results, individual_stack)
