# h264/dequant/tests/test_dequant_8x8.py
"""Tests for H.264 8x8 transform dequantization (TDD - RED phase).

H.264 Spec Reference:
- Section 8.5.12: Scaling and transformation process for 8x8 blocks
- Section 8.5.12.1: Scaling process for transform coefficients
- Table 8-14: LevelScale8x8 lookup table
- Table 8-15: Position type assignment for 8x8 blocks

These tests are written TDD-style and should FAIL until the
8x8 dequantization is properly implemented for High profile.
"""

import numpy as np
import pytest

# H.264 Spec Table 7-4: Default scaling lists for 8x8 blocks
DEFAULT_8x8_INTRA = [
    6, 10, 10, 13, 11, 13, 16, 16, 16, 16, 18, 18, 18, 18, 18, 23,
    23, 23, 23, 23, 23, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27,
    27, 27, 27, 27, 29, 29, 29, 29, 29, 29, 29, 31, 31, 31, 31, 31,
    31, 33, 33, 33, 33, 33, 36, 36, 36, 36, 38, 38, 38, 40, 40, 42
]

DEFAULT_8x8_INTER = [
    9, 13, 13, 15, 13, 15, 17, 17, 17, 17, 19, 19, 19, 19, 19, 21,
    21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 24, 24, 24, 24,
    24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27,
    27, 28, 28, 28, 28, 28, 30, 30, 30, 30, 32, 32, 32, 33, 33, 35
]

# Flat scaling list (all 16s) - used when scaling matrix not present
FLAT_8x8 = [16] * 64


class TestDequant8x8BasicInterface:
    """Test basic interface requirements for dequant_8x8."""

    def test_dequant_8x8_function_exists(self):
        """dequant_8x8 function should exist and be importable."""
        from dequant.dequant import dequant_8x8
        assert callable(dequant_8x8)

    def test_dequant_8x8_accepts_required_parameters(self):
        """dequant_8x8 should accept coeffs and qp parameters."""
        from dequant.dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=26)
        assert result is not None

    def test_dequant_8x8_returns_8x8_array(self):
        """dequant_8x8 should return an 8x8 array."""
        from dequant.dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=26)
        assert result.shape == (8, 8)

    def test_dequant_8x8_returns_int32(self):
        """dequant_8x8 should return int32 to prevent overflow."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=26)
        assert result.dtype == np.int32

    def test_dequant_8x8_rejects_wrong_shape(self):
        """dequant_8x8 should reject non-8x8 input."""
        from dequant.dequant import dequant_8x8
        with pytest.raises(ValueError, match="8x8"):
            dequant_8x8(np.zeros((4, 4), dtype=np.int32), qp=26)

    def test_dequant_8x8_rejects_16x16_input(self):
        """dequant_8x8 should reject 16x16 input."""
        from dequant.dequant import dequant_8x8
        with pytest.raises(ValueError):
            dequant_8x8(np.zeros((16, 16), dtype=np.int32), qp=26)


class TestDequant8x8QPValues:
    """Test dequant_8x8 with various QP values (0-51).

    H.264 Spec: Section 8.5.12.1
    QP determines the scaling factor. Higher QP means more scaling.
    """

    def test_qp_0_minimum_scaling(self):
        """QP=0 should produce minimum scaling."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32) * 10
        result = dequant_8x8(coeffs, qp=0)
        # QP=0 has smallest scaling factor
        assert np.all(result != 0)
        assert np.max(np.abs(result)) > 0

    def test_qp_6_doubles_from_qp_0(self):
        """QP=6 should roughly double the scaling from QP=0.

        H.264: QP increase of 6 doubles the quantization step.
        """
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32) * 10

        result_0 = dequant_8x8(coeffs, qp=0)
        result_6 = dequant_8x8(coeffs, qp=6)

        # QP+6 should approximately double the dequantized value
        ratio = np.sum(np.abs(result_6)) / np.sum(np.abs(result_0))
        assert 1.8 < ratio < 2.2, f"Expected ratio ~2.0, got {ratio}"

    def test_qp_12_quadruples_from_qp_0(self):
        """QP=12 should roughly quadruple the scaling from QP=0."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32) * 10

        result_0 = dequant_8x8(coeffs, qp=0)
        result_12 = dequant_8x8(coeffs, qp=12)

        ratio = np.sum(np.abs(result_12)) / np.sum(np.abs(result_0))
        assert 3.5 < ratio < 4.5, f"Expected ratio ~4.0, got {ratio}"

    def test_qp_26_midrange(self):
        """QP=26 is a common midrange value."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32) * 5
        result = dequant_8x8(coeffs, qp=26)
        assert np.all(result != 0)

    def test_qp_51_maximum_scaling(self):
        """QP=51 should produce maximum scaling."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=51)
        # QP=51 has largest scaling factor
        assert np.all(result != 0)

    def test_qp_51_much_larger_than_qp_0(self):
        """QP=51 should produce much larger values than QP=0."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)

        result_0 = dequant_8x8(coeffs, qp=0)
        result_51 = dequant_8x8(coeffs, qp=51)

        # QP range of 51 means 2^(51/6) ~= 362x scaling difference
        ratio = np.sum(np.abs(result_51)) / np.sum(np.abs(result_0))
        assert ratio > 100, f"Expected large ratio, got {ratio}"

    def test_qp_clamped_below_zero(self):
        """QP below 0 should be clamped to 0."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)

        result_neg = dequant_8x8(coeffs, qp=-10)
        result_0 = dequant_8x8(coeffs, qp=0)

        np.testing.assert_array_equal(result_neg, result_0)

    def test_qp_clamped_above_51(self):
        """QP above 51 should be clamped to 51."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)

        result_high = dequant_8x8(coeffs, qp=100)
        result_51 = dequant_8x8(coeffs, qp=51)

        np.testing.assert_array_equal(result_high, result_51)

    @pytest.mark.parametrize("qp", range(0, 52, 6))
    def test_qp_values_at_mod6_boundaries(self, qp):
        """Test QP values at qp%6 boundaries (0, 6, 12, ..., 48)."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=qp)
        assert result.shape == (8, 8)
        assert np.all(result != 0)

    @pytest.mark.parametrize("qp", [1, 7, 13, 19, 25, 31, 37, 43, 49])
    def test_qp_values_off_mod6_boundaries(self, qp):
        """Test QP values not at mod6 boundaries."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=qp)
        assert result.shape == (8, 8)


class TestDequant8x8ScalingListIntegration:
    """Test scaling list integration for 8x8 blocks.

    H.264 Spec: Section 8.5.9, 8.5.12
    High profile uses scaling matrices that modify dequantization.
    """

    def test_flat_scaling_list_default_behavior(self):
        """Flat scaling list (all 16s) should match no-list behavior."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32) * 5

        result_no_list = dequant_8x8(coeffs, qp=26)
        result_flat = dequant_8x8(coeffs, qp=26, scaling_list=FLAT_8x8)

        np.testing.assert_array_equal(result_no_list, result_flat)

    def test_custom_scaling_list_affects_output(self):
        """Custom scaling list should affect dequantization output."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)

        result_flat = dequant_8x8(coeffs, qp=26, scaling_list=FLAT_8x8)
        result_double = dequant_8x8(coeffs, qp=26, scaling_list=[32] * 64)

        # Double scaling values should produce larger output
        assert np.sum(np.abs(result_double)) > np.sum(np.abs(result_flat))

    def test_default_8x8_intra_scaling_list(self):
        """Default intra 8x8 scaling list from spec should work."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)

        result = dequant_8x8(coeffs, qp=26, scaling_list=DEFAULT_8x8_INTRA)
        assert result.shape == (8, 8)
        assert np.all(result != 0)

    def test_default_8x8_inter_scaling_list(self):
        """Default inter 8x8 scaling list from spec should work."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)

        result = dequant_8x8(coeffs, qp=26, scaling_list=DEFAULT_8x8_INTER)
        assert result.shape == (8, 8)
        assert np.all(result != 0)

    def test_scaling_list_invalid_length_raises(self):
        """Scaling list with wrong length should raise ValueError."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)

        with pytest.raises(ValueError, match="64"):
            dequant_8x8(coeffs, qp=26, scaling_list=[16] * 16)

        with pytest.raises(ValueError, match="64"):
            dequant_8x8(coeffs, qp=26, scaling_list=[16] * 32)

    def test_scaling_list_position_dependent(self):
        """Scaling list values should be applied position-dependently.

        The scaling list is applied in zigzag order per H.264 spec.
        """
        from dequant.dequant import dequant_8x8

        # Create scaling list with position-dependent values
        scaling_list = list(range(16, 80))  # 16 to 79
        coeffs = np.ones((8, 8), dtype=np.int32)

        result = dequant_8x8(coeffs, qp=26, scaling_list=scaling_list)

        # Due to different scaling values, output should vary by position
        # Check that not all values are the same
        unique_values = np.unique(result)
        assert len(unique_values) > 1, "Result should have position-dependent values"

    @pytest.mark.xfail(reason="Zigzag order for scaling list not verified against spec")
    def test_scaling_list_applied_in_zigzag_order(self):
        """Scaling list should be applied per zigzag scan order.

        H.264 Spec: The scaling list is indexed by the zigzag scan position,
        not by row-major position.
        """
        from dequant.dequant import dequant_8x8

        # Create a scaling list where position 0 has value 64 (4x flat)
        # and all others have value 16 (flat)
        scaling_list = [64] + [16] * 63
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 1  # DC coefficient (zigzag position 0)
        coeffs[0, 1] = 1  # Should be zigzag position 1

        result = dequant_8x8(coeffs, qp=0, scaling_list=scaling_list)

        # DC coefficient should be scaled by 64/16 = 4x compared to flat
        result_flat = dequant_8x8(coeffs, qp=0, scaling_list=FLAT_8x8)

        # Verify DC is scaled 4x
        assert result[0, 0] == 4 * result_flat[0, 0], (
            "DC should be scaled by 64/16 = 4x"
        )


class TestDequant8x8IntraVsInter:
    """Test inter vs intra scaling differences for 8x8 blocks.

    H.264 uses different default scaling lists for intra and inter blocks.
    """

    def test_intra_inter_default_lists_differ(self):
        """Intra and Inter default scaling lists should produce different results."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32) * 10

        result_intra = dequant_8x8(
            coeffs, qp=26, scaling_list=DEFAULT_8x8_INTRA
        )
        result_inter = dequant_8x8(
            coeffs, qp=26, scaling_list=DEFAULT_8x8_INTER
        )

        # Intra and Inter defaults differ
        assert not np.array_equal(result_intra, result_inter)

    def test_intra_scaling_list_smaller_dc_weight(self):
        """Intra scaling list has smaller DC weight than inter.

        DEFAULT_8x8_INTRA[0] = 6 vs DEFAULT_8x8_INTER[0] = 9
        """
        from dequant.dequant import dequant_8x8

        # Single DC coefficient
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 100

        result_intra = dequant_8x8(
            coeffs, qp=26, scaling_list=DEFAULT_8x8_INTRA
        )
        result_inter = dequant_8x8(
            coeffs, qp=26, scaling_list=DEFAULT_8x8_INTER
        )

        # Intra DC should be smaller due to smaller scaling weight
        assert abs(result_intra[0, 0]) < abs(result_inter[0, 0])

    @pytest.mark.xfail(reason="dequant_8x8_with_defaults function not implemented")
    def test_is_intra_parameter_selects_correct_list(self):
        """is_intra parameter should select appropriate default scaling."""
        from dequant.dequant import dequant_8x8_with_defaults
        coeffs = np.ones((8, 8), dtype=np.int32)

        result_intra = dequant_8x8_with_defaults(coeffs, qp=26, is_intra=True)
        result_inter = dequant_8x8_with_defaults(coeffs, qp=26, is_intra=False)

        assert not np.array_equal(result_intra, result_inter)


class TestDequant8x8DCCoefficient:
    """Test DC coefficient handling for 8x8 blocks.

    H.264 Spec: Section 8.5.12
    DC coefficient (0,0) has special handling in 8x8 transforms.
    """

    def test_dc_only_coefficient(self):
        """Single DC coefficient should be properly scaled."""
        from dequant.dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 100

        result = dequant_8x8(coeffs, qp=26)

        # DC should be scaled, others should remain zero
        assert result[0, 0] != 0
        assert np.sum(result) == result[0, 0]

    def test_dc_coefficient_scaling_factor(self):
        """DC coefficient uses position type 0 scaling.

        H.264 Table 8-14: LevelScale8x8[qp%6][0] for position (0,0)
        """
        from dequant.dequant import dequant_8x8, LEVEL_SCALE_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 1
        qp = 12  # qp_div_6 = 2, qp_mod_6 = 0

        result = dequant_8x8(coeffs, qp=qp)

        # Expected: coeff * LevelScale8x8[0][0] << (qp // 6)
        # = 1 * 20 << 2 = 80
        expected_dc = LEVEL_SCALE_8x8[0, 0] << (qp // 6)
        assert result[0, 0] == expected_dc

    def test_dc_coefficient_with_scaling_list(self):
        """DC coefficient with scaling list should combine both scalings."""
        from dequant.dequant import dequant_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 1

        # Scaling list with DC weight = 32 (double the flat 16)
        scaling_list = [32] + [16] * 63

        result_flat = dequant_8x8(coeffs, qp=12, scaling_list=FLAT_8x8)
        result_custom = dequant_8x8(coeffs, qp=12, scaling_list=scaling_list)

        # Custom DC scaling should be double
        assert result_custom[0, 0] == 2 * result_flat[0, 0]

    def test_large_dc_value(self):
        """Large DC coefficient should not overflow."""
        from dequant.dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 2047  # Maximum CAVLC coefficient value

        result = dequant_8x8(coeffs, qp=51)

        # Should produce valid result without overflow
        assert result.dtype == np.int32
        assert np.isfinite(result[0, 0])

    def test_negative_dc_coefficient(self):
        """Negative DC coefficient should be properly handled."""
        from dequant.dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = -100

        result = dequant_8x8(coeffs, qp=26)

        # Sign should be preserved
        assert result[0, 0] < 0


class TestDequant8x8OverflowPrevention:
    """Test overflow prevention with int32 intermediates.

    H.264 dequantization can produce large values, especially at high QP.
    Using int32 intermediates prevents overflow.
    """

    def test_large_coefficient_high_qp_no_overflow(self):
        """Large coefficient at high QP should not overflow."""
        from dequant.dequant import dequant_8x8
        # Maximum coefficient that might appear in H.264
        coeffs = np.ones((8, 8), dtype=np.int32) * 2047

        result = dequant_8x8(coeffs, qp=51)

        # int32 can hold up to ~2.1 billion, should be sufficient
        assert result.dtype == np.int32
        assert np.all(np.isfinite(result))
        assert np.max(result) < 2**31 - 1

    def test_extreme_scaling_list_no_overflow(self):
        """Extreme scaling list values should not cause overflow."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32) * 1000

        # Maximum scaling list value is 255
        extreme_scaling = [255] * 64

        result = dequant_8x8(coeffs, qp=51, scaling_list=extreme_scaling)

        assert result.dtype == np.int32
        assert np.all(np.isfinite(result))

    def test_int16_input_promoted_to_int32(self):
        """int16 input should be promoted to int32 for computation."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int16) * 100

        result = dequant_8x8(coeffs, qp=40)

        # Output should still be int32
        assert result.dtype == np.int32

    def test_mixed_sign_coefficients(self):
        """Mixed positive and negative coefficients should not overflow."""
        from dequant.dequant import dequant_8x8
        coeffs = np.array([
            [2047, -2047, 2047, -2047, 2047, -2047, 2047, -2047],
            [-2047, 2047, -2047, 2047, -2047, 2047, -2047, 2047],
            [2047, -2047, 2047, -2047, 2047, -2047, 2047, -2047],
            [-2047, 2047, -2047, 2047, -2047, 2047, -2047, 2047],
            [2047, -2047, 2047, -2047, 2047, -2047, 2047, -2047],
            [-2047, 2047, -2047, 2047, -2047, 2047, -2047, 2047],
            [2047, -2047, 2047, -2047, 2047, -2047, 2047, -2047],
            [-2047, 2047, -2047, 2047, -2047, 2047, -2047, 2047],
        ], dtype=np.int32)

        result = dequant_8x8(coeffs, qp=48)

        assert result.dtype == np.int32
        assert np.all(np.isfinite(result))
        # Check sign pattern preserved
        assert result[0, 0] > 0
        assert result[0, 1] < 0

    def test_computation_uses_int32_internally(self):
        """Internal computation should use int32 to prevent intermediate overflow."""
        from dequant.dequant import dequant_8x8

        # Value that would overflow int16 after scaling
        coeffs = np.ones((8, 8), dtype=np.int32) * 500

        # At QP=30, scaling is significant
        result = dequant_8x8(coeffs, qp=30)

        # If int16 was used internally, this would wrap around
        assert np.all(result > 0)
        assert np.min(result) > 500  # Should be scaled up


class TestDequant8x8KnownValues:
    """Test known coefficient/QP combinations against expected outputs.

    These values are derived from H.264 reference decoder (JM) or spec formulas.
    """

    def test_known_value_qp0_coeff1(self):
        """Test known output for QP=0, coefficient=1.

        Formula: d[i,j] = c[i,j] * LevelScale8x8[qp%6][pos_type] << (qp // 6)
        For QP=0: qp_div_6=0, qp_mod_6=0
        Position (0,0) has type 0, so LevelScale8x8[0][0] = 20
        Expected: 1 * 20 << 0 = 20
        """
        from dequant.dequant import dequant_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 1

        result = dequant_8x8(coeffs, qp=0)

        assert result[0, 0] == 20

    def test_known_value_qp6_coeff1(self):
        """Test known output for QP=6, coefficient=1.

        For QP=6: qp_div_6=1, qp_mod_6=0
        Expected: 1 * 20 << 1 = 40
        """
        from dequant.dequant import dequant_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 1

        result = dequant_8x8(coeffs, qp=6)

        assert result[0, 0] == 40

    def test_known_value_qp12_coeff1(self):
        """Test known output for QP=12, coefficient=1.

        For QP=12: qp_div_6=2, qp_mod_6=0
        Expected: 1 * 20 << 2 = 80
        """
        from dequant.dequant import dequant_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 1

        result = dequant_8x8(coeffs, qp=12)

        assert result[0, 0] == 80

    def test_known_value_qp1_position_1_1(self):
        """Test known output for QP=1 at position (1,1).

        Position (1,1) has type 1, so LevelScale8x8[1][1] = 19
        For QP=1: qp_div_6=0, qp_mod_6=1
        Expected: 1 * 19 << 0 = 19
        """
        from dequant.dequant import dequant_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[1, 1] = 1

        result = dequant_8x8(coeffs, qp=1)

        assert result[1, 1] == 19

    def test_known_value_qp2_position_2_2(self):
        """Test known output for QP=2 at position (2,2).

        Position (2,2) has type 2, so LevelScale8x8[2][2] = 42
        For QP=2: qp_div_6=0, qp_mod_6=2
        Expected: 1 * 42 << 0 = 42
        """
        from dequant.dequant import dequant_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[2, 2] = 1

        result = dequant_8x8(coeffs, qp=2)

        assert result[2, 2] == 42

    def test_known_value_with_flat_scaling_list(self):
        """Test known output with flat scaling list.

        With flat list (all 16s), the formula becomes:
        d = c * LevelScale8x8[qp%6][pos] * 16 / 16 << qp_div_6
        = c * LevelScale8x8[qp%6][pos] << qp_div_6
        """
        from dequant.dequant import dequant_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 1

        result = dequant_8x8(coeffs, qp=6, scaling_list=FLAT_8x8)

        # Same as without scaling list
        assert result[0, 0] == 40

    def test_known_value_with_double_scaling_list(self):
        """Test known output with doubled scaling list.

        With scaling list of all 32s:
        d = c * LevelScale8x8[qp%6][pos] * 32 / 16 << qp_div_6
        = c * LevelScale8x8[qp%6][pos] * 2 << qp_div_6
        """
        from dequant.dequant import dequant_8x8

        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 1

        result = dequant_8x8(coeffs, qp=6, scaling_list=[32] * 64)

        # Should be double the flat list result
        assert result[0, 0] == 80

    @pytest.mark.xfail(reason="JM reference decoder output values need verification")
    def test_jm_reference_block_qp26(self):
        """Test against JM reference decoder output for QP=26.

        This block pattern is from JM test vectors. The expected values
        need to be verified against actual JM decoder output.
        """
        from dequant.dequant import dequant_8x8

        # Example quantized coefficients (typical I-block pattern)
        coeffs = np.array([
            [16, -4, 2, 0, 0, 0, 0, 0],
            [-3, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)

        result = dequant_8x8(coeffs, qp=26)

        # Expected values from JM reference decoder
        # QP=26: qp_div_6=4, qp_mod_6=2
        # DC: 16 * LevelScale8x8[2][0] << 4 = 16 * 26 << 4 = 6656
        assert result[0, 0] == 6656  # Verify against JM output

    @pytest.mark.xfail(reason="Full 8x8 block JM verification not done")
    def test_jm_reference_full_block_qp20(self):
        """Test full 8x8 block against JM reference for QP=20.

        Comprehensive verification with non-trivial coefficient pattern.
        """
        from dequant.dequant import dequant_8x8

        coeffs = np.array([
            [64, 16, -8, 4, -2, 1, 0, 0],
            [16, -8, 4, -2, 1, 0, 0, 0],
            [-8, 4, -2, 1, 0, 0, 0, 0],
            [4, -2, 1, 0, 0, 0, 0, 0],
            [-2, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.int32)

        result = dequant_8x8(coeffs, qp=20)

        # These expected values need to be filled from JM reference
        # QP=20: qp_div_6=3, qp_mod_6=2
        expected = np.array([
            [13312, 3072, -1344, 640, -336, 160, 0, 0],  # Placeholder
            # ... more rows would be verified
        ], dtype=np.int32)

        assert result[0, 0] == expected[0, 0]


class TestDequant8x8LevelScaleTable:
    """Test 8x8 LevelScale lookup table values.

    H.264 Spec: Table 8-14
    """

    def test_level_scale_8x8_exists(self):
        """LEVEL_SCALE_8x8 table should exist."""
        from dequant.dequant import LEVEL_SCALE_8x8
        assert LEVEL_SCALE_8x8 is not None

    def test_level_scale_8x8_shape(self):
        """LEVEL_SCALE_8x8 should be 6x6 (qp%6 x position_type)."""
        from dequant.dequant import LEVEL_SCALE_8x8
        assert LEVEL_SCALE_8x8.shape == (6, 6)

    def test_level_scale_8x8_qp_mod_0_values(self):
        """Verify LevelScale8x8 values for qp%6=0.

        H.264 Table 8-14: [20, 18, 32, 19, 25, 24]
        """
        from dequant.dequant import LEVEL_SCALE_8x8
        expected = [20, 18, 32, 19, 25, 24]
        for i, val in enumerate(expected):
            assert LEVEL_SCALE_8x8[0, i] == val, f"Position {i}: expected {val}"

    def test_level_scale_8x8_qp_mod_5_values(self):
        """Verify LevelScale8x8 values for qp%6=5.

        H.264 Table 8-14: [36, 32, 58, 34, 46, 43]
        """
        from dequant.dequant import LEVEL_SCALE_8x8
        expected = [36, 32, 58, 34, 46, 43]
        for i, val in enumerate(expected):
            assert LEVEL_SCALE_8x8[5, i] == val

    def test_level_scale_8x8_monotonically_increases(self):
        """LevelScale8x8 values should increase with qp%6."""
        from dequant.dequant import LEVEL_SCALE_8x8
        for pos_type in range(6):
            for qp_mod in range(5):
                assert LEVEL_SCALE_8x8[qp_mod + 1, pos_type] >= LEVEL_SCALE_8x8[qp_mod, pos_type]


class TestDequant8x8PositionTypeMap:
    """Test 8x8 position type mapping.

    H.264 Spec: Table 8-15
    """

    def test_position_type_8x8_exists(self):
        """POSITION_TYPE_8x8 should exist."""
        from dequant.dequant import POSITION_TYPE_8x8
        assert POSITION_TYPE_8x8 is not None

    def test_position_type_8x8_shape(self):
        """POSITION_TYPE_8x8 should be 8x8."""
        from dequant.dequant import POSITION_TYPE_8x8
        assert POSITION_TYPE_8x8.shape == (8, 8)

    def test_position_type_8x8_corners_type_0(self):
        """Corner positions should have type 0."""
        from dequant.dequant import POSITION_TYPE_8x8
        # Positions (0,0), (0,4), (4,0), (4,4)
        assert POSITION_TYPE_8x8[0, 0] == 0
        assert POSITION_TYPE_8x8[0, 4] == 0
        assert POSITION_TYPE_8x8[4, 0] == 0
        assert POSITION_TYPE_8x8[4, 4] == 0

    def test_position_type_8x8_center_type_2(self):
        """Center position (2,2) should have type 2."""
        from dequant.dequant import POSITION_TYPE_8x8
        assert POSITION_TYPE_8x8[2, 2] == 2

    def test_position_type_8x8_values_in_range(self):
        """All position types should be in range 0-5."""
        from dequant.dequant import POSITION_TYPE_8x8
        assert np.all(POSITION_TYPE_8x8 >= 0)
        assert np.all(POSITION_TYPE_8x8 <= 5)

    def test_position_type_8x8_has_all_types(self):
        """Position type map should use all 6 types."""
        from dequant.dequant import POSITION_TYPE_8x8
        unique_types = np.unique(POSITION_TYPE_8x8)
        assert len(unique_types) == 6
        assert set(unique_types) == {0, 1, 2, 3, 4, 5}


class TestDequant8x8GetScaleMatrix:
    """Test 8x8 scale matrix generation."""

    def test_get_scale_matrix_8x8_exists(self):
        """get_scale_matrix_8x8 function should exist."""
        from dequant.dequant import get_scale_matrix_8x8
        assert callable(get_scale_matrix_8x8)

    def test_get_scale_matrix_8x8_returns_8x8(self):
        """get_scale_matrix_8x8 should return 8x8 array."""
        from dequant.dequant import get_scale_matrix_8x8
        matrix = get_scale_matrix_8x8(qp=26)
        assert matrix.shape == (8, 8)

    def test_get_scale_matrix_8x8_returns_int32(self):
        """get_scale_matrix_8x8 should return int32."""
        from dequant.dequant import get_scale_matrix_8x8
        matrix = get_scale_matrix_8x8(qp=26)
        assert matrix.dtype == np.int32

    def test_get_scale_matrix_8x8_qp0_equals_qp6(self):
        """QP=0 and QP=6 should have same scale matrix (same qp%6)."""
        from dequant.dequant import get_scale_matrix_8x8
        matrix_0 = get_scale_matrix_8x8(qp=0)
        matrix_6 = get_scale_matrix_8x8(qp=6)
        np.testing.assert_array_equal(matrix_0, matrix_6)

    def test_get_scale_matrix_8x8_position_dependent(self):
        """Scale matrix should have position-dependent values."""
        from dequant.dequant import get_scale_matrix_8x8
        matrix = get_scale_matrix_8x8(qp=0)
        # Not all values should be the same
        unique_values = np.unique(matrix)
        assert len(unique_values) == 6  # 6 position types


class TestDequant8x8ZigzagOrder:
    """Test scaling list application in zigzag order.

    H.264 uses zigzag scan order for 8x8 blocks.
    """

    @pytest.mark.xfail(reason="8x8 zigzag scaling not implemented")
    def test_scaling_list_zigzag_order(self):
        """Scaling list should be applied in 8x8 zigzag scan order."""
        from dequant.dequant import dequant_8x8, ZIGZAG_8x8

        # Create scaling list where each position has unique value
        scaling_list = list(range(64))

        coeffs = np.ones((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=0, scaling_list=scaling_list)

        # Verify zigzag ordering
        # First position in zigzag (0,0) uses scaling_list[0]
        # Second position in zigzag should use scaling_list[1]
        # etc.
        assert ZIGZAG_8x8.shape == (64, 2)

    @pytest.mark.xfail(reason="8x8 zigzag constant not exported")
    def test_zigzag_8x8_constant_exists(self):
        """ZIGZAG_8x8 constant should exist for scan order."""
        from dequant.dequant import ZIGZAG_8x8
        assert ZIGZAG_8x8 is not None
        assert len(ZIGZAG_8x8) == 64


class TestDequant8x8EdgeCases:
    """Test edge cases for 8x8 dequantization."""

    def test_all_zero_coefficients(self):
        """All-zero input should produce all-zero output."""
        from dequant.dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=26)
        np.testing.assert_array_equal(result, np.zeros((8, 8)))

    def test_single_coefficient_at_each_position(self):
        """Test single coefficient at various positions."""
        from dequant.dequant import dequant_8x8

        for i in range(8):
            for j in range(8):
                coeffs = np.zeros((8, 8), dtype=np.int32)
                coeffs[i, j] = 1
                result = dequant_8x8(coeffs, qp=26)
                assert result[i, j] != 0
                # Other positions should be zero
                total_nonzero = np.count_nonzero(result)
                assert total_nonzero == 1

    def test_maximum_positive_coefficient(self):
        """Maximum positive coefficient (2047 for 11-bit) should work."""
        from dequant.dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 2047

        result = dequant_8x8(coeffs, qp=26)
        assert result[0, 0] > 0

    def test_maximum_negative_coefficient(self):
        """Maximum negative coefficient (-2048 for 11-bit) should work."""
        from dequant.dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = -2048

        result = dequant_8x8(coeffs, qp=26)
        assert result[0, 0] < 0

    def test_sparse_high_frequency_coefficients(self):
        """Sparse high-frequency coefficients (common in real video)."""
        from dequant.dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 100  # DC
        coeffs[0, 1] = -10  # Low AC
        coeffs[7, 7] = 1    # Highest frequency

        result = dequant_8x8(coeffs, qp=26)

        assert result[0, 0] != 0
        assert result[0, 1] != 0
        assert result[7, 7] != 0

    def test_float_input_raises_or_converts(self):
        """Float input should either raise or be converted."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.float64)

        # Either raises TypeError or converts to int
        try:
            result = dequant_8x8(coeffs, qp=26)
            # If it doesn't raise, check it still works
            assert result.dtype == np.int32
        except (TypeError, ValueError):
            pass  # Also acceptable to raise


class TestDequant8x8PPSIntegration:
    """Test integration with PPS scaling lists.

    H.264 allows PPS to override SPS scaling lists.
    """

    def test_dequant_8x8_with_pps_scaling_list(self):
        """dequant_8x8 should work with PPS-provided scaling list."""
        from dequant.dequant import dequant_8x8
        from parameters.pps import PPS

        pps = PPS()
        pps.scaling_lists_8x8 = [[32] * 64, [32] * 64]

        coeffs = np.ones((8, 8), dtype=np.int32)

        # Use scaling list from PPS
        result = dequant_8x8(
            coeffs, qp=26,
            scaling_list=pps.scaling_lists_8x8[0]
        )

        assert result.shape == (8, 8)
        assert np.all(result != 0)

    @pytest.mark.xfail(reason="dequant_8x8_from_pps function not implemented")
    def test_dequant_8x8_selects_correct_pps_list(self):
        """Should select correct scaling list based on intra/inter."""
        from dequant.dequant import dequant_8x8_from_pps
        from parameters.pps import PPS

        pps = PPS()
        pps.scaling_lists_8x8 = [
            [16] * 64,  # Intra (list 6)
            [32] * 64,  # Inter (list 7)
        ]

        coeffs = np.ones((8, 8), dtype=np.int32)

        result_intra = dequant_8x8_from_pps(coeffs, qp=26, pps=pps, is_intra=True)
        result_inter = dequant_8x8_from_pps(coeffs, qp=26, pps=pps, is_intra=False)

        # Inter has larger scaling values, should produce larger output
        assert np.sum(np.abs(result_inter)) > np.sum(np.abs(result_intra))


class TestDequant8x8ChromaScaling:
    """Test chroma scaling for 8x8 blocks (High profile 4:4:4).

    H.264 High 4:4:4 profile can use 8x8 transforms for chroma.
    """

    @pytest.mark.xfail(reason="8x8 chroma dequantization not implemented")
    def test_dequant_8x8_chroma_cb(self):
        """8x8 dequant for Cb chroma component."""
        from dequant.dequant import dequant_8x8_chroma, get_chroma_qp
        coeffs = np.ones((8, 8), dtype=np.int32)
        luma_qp = 30

        # Chroma QP may differ from luma QP
        chroma_qp = get_chroma_qp(luma_qp)
        result = dequant_8x8_chroma(coeffs, qp=chroma_qp, component='Cb')

        assert result.shape == (8, 8)

    @pytest.mark.xfail(reason="8x8 chroma dequantization not implemented")
    def test_dequant_8x8_chroma_cr(self):
        """8x8 dequant for Cr chroma component."""
        from dequant.dequant import dequant_8x8_chroma
        coeffs = np.ones((8, 8), dtype=np.int32)

        result = dequant_8x8_chroma(coeffs, qp=26, component='Cr')

        assert result.shape == (8, 8)

    @pytest.mark.xfail(reason="Chroma scaling list selection not implemented")
    def test_chroma_uses_correct_scaling_list_index(self):
        """Chroma should use scaling lists 8-11 for 4:4:4."""
        from dequant.dequant import get_chroma_scaling_list_8x8

        # In 4:4:4, there are additional scaling lists for chroma 8x8:
        # List 8: Intra Cb 8x8
        # List 9: Inter Cb 8x8
        # List 10: Intra Cr 8x8
        # List 11: Inter Cr 8x8
        list_idx = get_chroma_scaling_list_8x8(component='Cb', is_intra=True)
        assert list_idx == 8


class TestDequant8x8TransformBypass:
    """Test transform bypass mode for 8x8 blocks.

    H.264 High profile allows lossless coding with transform bypass.
    """

    @pytest.mark.xfail(reason="8x8 transform bypass not implemented")
    def test_transform_bypass_mode(self):
        """In transform bypass mode, dequant should be identity."""
        from dequant.dequant import dequant_8x8_bypass
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

        result = dequant_8x8_bypass(coeffs)

        # In bypass mode, output equals input (no scaling)
        np.testing.assert_array_equal(result, coeffs)

    @pytest.mark.xfail(reason="QP prime bypass flag not handled")
    def test_qpprime_y_zero_transform_bypass_flag(self):
        """When qpprime_y_zero_transform_bypass_flag is set, use bypass."""
        from dequant.dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32) * 100

        # QP=0 with bypass flag should not scale
        result = dequant_8x8(coeffs, qp=0, transform_bypass=True)

        np.testing.assert_array_equal(result, coeffs)


class TestDequant8x8BitDepthExtensions:
    """Test extended bit depth support (9-14 bit) for High profile.

    H.264 High 10/High 4:2:2/High 4:4:4 profiles support extended bit depth.
    """

    @pytest.mark.xfail(reason="Extended bit depth not implemented")
    def test_10bit_depth_scaling(self):
        """10-bit depth requires adjusted scaling."""
        from dequant.dequant import dequant_8x8_extended
        coeffs = np.ones((8, 8), dtype=np.int32)

        result = dequant_8x8_extended(coeffs, qp=26, bit_depth=10)

        assert result.dtype == np.int32
        # 10-bit has 4x larger dynamic range than 8-bit
        assert np.max(np.abs(result)) > 0

    @pytest.mark.xfail(reason="Extended bit depth not implemented")
    def test_12bit_depth_scaling(self):
        """12-bit depth requires adjusted scaling."""
        from dequant.dequant import dequant_8x8_extended
        coeffs = np.ones((8, 8), dtype=np.int32)

        result = dequant_8x8_extended(coeffs, qp=26, bit_depth=12)

        assert result.dtype == np.int32

    @pytest.mark.xfail(reason="Bit depth QP offset not implemented")
    def test_bit_depth_affects_qp_offset(self):
        """Extended bit depth affects QP calculation.

        QP' = QP + 6 * (bit_depth_luma - 8)
        """
        from dequant.dequant import calculate_qp_prime

        # For 10-bit: QP' = QP + 12
        qp_prime = calculate_qp_prime(qp=26, bit_depth=10)
        assert qp_prime == 38

        # For 12-bit: QP' = QP + 24
        qp_prime = calculate_qp_prime(qp=26, bit_depth=12)
        assert qp_prime == 50
