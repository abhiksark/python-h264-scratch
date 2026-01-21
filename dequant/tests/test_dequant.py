# h264/dequant/tests/test_dequant.py
"""Tests for dequantization module."""

import numpy as np
import pytest

from dequant import (
    LEVEL_SCALE,
    POSITION_TYPE,
    get_scale_matrix,
    dequant_4x4,
    dequant_4x4_simple,
    dequant_dc_4x4,
    dequant_dc_2x2,
    get_chroma_qp,
    qp_to_qstep,
)


class TestLevelScale:
    """Tests for the LevelScale lookup table."""

    def test_level_scale_shape(self):
        """LevelScale should be 6x3."""
        assert LEVEL_SCALE.shape == (6, 3)

    def test_level_scale_values(self):
        """Verify some known LevelScale values from spec."""
        # qp%6=0: [10, 16, 13]
        assert LEVEL_SCALE[0, 0] == 10
        assert LEVEL_SCALE[0, 1] == 16
        assert LEVEL_SCALE[0, 2] == 13

        # qp%6=5: [18, 29, 23]
        assert LEVEL_SCALE[5, 0] == 18
        assert LEVEL_SCALE[5, 1] == 29
        assert LEVEL_SCALE[5, 2] == 23

    def test_level_scale_increasing(self):
        """Values should generally increase with qp%6."""
        for col in range(3):
            for row in range(5):
                assert LEVEL_SCALE[row + 1, col] >= LEVEL_SCALE[row, col]


class TestPositionType:
    """Tests for position type mapping."""

    def test_position_type_shape(self):
        """Position type map should be 4x4."""
        assert POSITION_TYPE.shape == (4, 4)

    def test_corner_positions(self):
        """Corners should be type 0."""
        assert POSITION_TYPE[0, 0] == 0
        assert POSITION_TYPE[0, 2] == 0
        assert POSITION_TYPE[2, 0] == 0
        assert POSITION_TYPE[2, 2] == 0

    def test_center_positions(self):
        """Center diagonal should be type 1."""
        assert POSITION_TYPE[1, 1] == 1
        assert POSITION_TYPE[1, 3] == 1
        assert POSITION_TYPE[3, 1] == 1
        assert POSITION_TYPE[3, 3] == 1


class TestGetScaleMatrix:
    """Tests for scale matrix generation."""

    def test_scale_matrix_shape(self):
        """Scale matrix should be 4x4."""
        matrix = get_scale_matrix(qp=26)
        assert matrix.shape == (4, 4)

    def test_scale_matrix_qp0(self):
        """Test scale matrix for QP=0."""
        matrix = get_scale_matrix(qp=0)
        # qp%6=0, so scales are [10, 16, 13]
        assert matrix[0, 0] == 10  # corner
        assert matrix[1, 1] == 16  # center
        assert matrix[0, 1] == 13  # other

    def test_scale_matrix_qp6(self):
        """QP=6 should have same scale values as QP=0 (qp%6=0)."""
        matrix_0 = get_scale_matrix(qp=0)
        matrix_6 = get_scale_matrix(qp=6)
        np.testing.assert_array_equal(matrix_0, matrix_6)


class TestDequant4x4:
    """Tests for 4x4 block dequantization."""

    def test_zero_coefficients(self):
        """Zero coefficients should stay zero."""
        coeffs = np.zeros((4, 4), dtype=np.int32)
        result = dequant_4x4(coeffs, qp=26)
        np.testing.assert_array_equal(result, np.zeros((4, 4)))

    def test_dc_only(self):
        """Single DC coefficient."""
        coeffs = np.zeros((4, 4), dtype=np.int32)
        coeffs[0, 0] = 10
        result = dequant_4x4(coeffs, qp=26)

        # DC should be scaled
        assert result[0, 0] != 0
        assert result[0, 0] == 10 * get_scale_matrix(26)[0, 0] << (26 // 6 - 1)

    def test_output_dtype(self):
        """Output should be int32."""
        coeffs = np.ones((4, 4), dtype=np.int32)
        result = dequant_4x4(coeffs, qp=20)
        assert result.dtype == np.int32

    def test_qp_clamping(self):
        """QP should be clamped to valid range."""
        coeffs = np.ones((4, 4), dtype=np.int32)

        # Should not raise for out-of-range QP
        result_low = dequant_4x4(coeffs, qp=-5)  # Clamped to 0
        result_high = dequant_4x4(coeffs, qp=100)  # Clamped to 51

        assert result_low is not None
        assert result_high is not None

    def test_invalid_shape_raises(self):
        """Non-4x4 input should raise."""
        coeffs = np.ones((3, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            dequant_4x4(coeffs, qp=26)

    def test_higher_qp_gives_larger_values(self):
        """Higher QP should give larger dequantized values (more compression)."""
        coeffs = np.array([[10, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]], dtype=np.int32)

        result_low = dequant_4x4_simple(coeffs, qp=10)
        result_high = dequant_4x4_simple(coeffs, qp=30)

        # Higher QP means larger step size, so same coefficient
        # becomes larger after dequant
        assert result_high[0, 0] > result_low[0, 0]


class TestDequantDC:
    """Tests for DC coefficient dequantization."""

    def test_dequant_dc_4x4_zero(self):
        """Zero DC coefficients should stay zero."""
        dc = np.zeros((4, 4), dtype=np.int32)
        result = dequant_dc_4x4(dc, qp=26)
        np.testing.assert_array_equal(result, np.zeros((4, 4)))

    def test_dequant_dc_2x2_shape(self):
        """2x2 DC should maintain shape."""
        dc = np.ones((2, 2), dtype=np.int32)
        result = dequant_dc_2x2(dc, qp=26)
        assert result.shape == (2, 2)

    def test_dequant_dc_2x2_invalid_shape(self):
        """Non-2x2 should raise."""
        dc = np.ones((3, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            dequant_dc_2x2(dc, qp=26)


class TestChromaQP:
    """Tests for chroma QP mapping."""

    def test_low_qp_unchanged(self):
        """Low QP values should be unchanged."""
        for qp in range(30):
            assert get_chroma_qp(qp) == qp

    def test_high_qp_reduced(self):
        """High QP values should be reduced."""
        # At QP=51, chroma QP should be 39
        assert get_chroma_qp(51) == 39

        # At QP=40, chroma QP should be less
        assert get_chroma_qp(40) < 40

    def test_qp_clamping(self):
        """Out of range QP should be clamped."""
        assert get_chroma_qp(-10) == get_chroma_qp(0)
        assert get_chroma_qp(100) == get_chroma_qp(51)


class TestQPToQStep:
    """Tests for QP to quantization step conversion."""

    def test_qp0_step(self):
        """QP=0 should give smallest step."""
        step = qp_to_qstep(0)
        assert 0.5 < step < 1.0

    def test_qp_doubles_every_6(self):
        """Step size should double for every 6 QP increase."""
        step_0 = qp_to_qstep(0)
        step_6 = qp_to_qstep(6)
        step_12 = qp_to_qstep(12)

        assert abs(step_6 / step_0 - 2.0) < 0.01
        assert abs(step_12 / step_6 - 2.0) < 0.01

    def test_step_increases_with_qp(self):
        """Step should monotonically increase with QP."""
        prev_step = 0
        for qp in range(52):
            step = qp_to_qstep(qp)
            assert step > prev_step
            prev_step = step


class TestIntegration:
    """Integration tests combining dequant functions."""

    def test_typical_coefficient_pattern(self):
        """Test with typical coefficient pattern (DC + few AC)."""
        # Typical I-block: large DC, small AC
        coeffs = np.array([
            [64, -8, 4, 0],
            [-4, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.int32)

        result = dequant_4x4(coeffs, qp=26)

        # DC should be largest magnitude
        assert abs(result[0, 0]) > abs(result).sum() / 2

        # Non-zero inputs should give non-zero outputs
        assert result[0, 0] != 0
        assert result[0, 1] != 0
        assert result[1, 0] != 0


class TestDequant8x8:
    """Tests for 8x8 dequantization (High profile)."""

    def test_dequant_8x8_exists(self):
        """dequant_8x8 function should exist."""
        from dequant import dequant_8x8
        assert callable(dequant_8x8)

    def test_dequant_8x8_accepts_8x8_input(self):
        """dequant_8x8 should accept 8x8 input."""
        from dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=26)
        assert result.shape == (8, 8)

    def test_dequant_8x8_returns_int32(self):
        """dequant_8x8 should return int32 array."""
        from dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=26)
        assert result.dtype == np.int32

    def test_dequant_8x8_rejects_wrong_shape(self):
        """dequant_8x8 should reject non-8x8 input."""
        from dequant import dequant_8x8
        coeffs = np.zeros((4, 4), dtype=np.int32)
        with pytest.raises(ValueError):
            dequant_8x8(coeffs, qp=26)

    def test_dequant_8x8_zero_coeffs_zero_output(self):
        """Zero coefficients should produce zero output."""
        from dequant import dequant_8x8
        coeffs = np.zeros((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=26)
        np.testing.assert_array_equal(result, np.zeros((8, 8)))

    def test_dequant_8x8_nonzero_coeffs_nonzero_output(self):
        """Non-zero coefficients should produce non-zero output."""
        from dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)
        result = dequant_8x8(coeffs, qp=26)
        assert np.all(result != 0)

    def test_dequant_8x8_scales_with_qp(self):
        """Higher QP should produce larger dequantized values."""
        from dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)

        result_low_qp = dequant_8x8(coeffs, qp=12)
        result_high_qp = dequant_8x8(coeffs, qp=24)

        # Higher QP means larger scaling
        assert np.sum(np.abs(result_high_qp)) > np.sum(np.abs(result_low_qp))

    def test_dequant_8x8_with_scaling_list(self):
        """dequant_8x8 should accept optional scaling list."""
        from dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)
        scaling_list = [16] * 64  # Flat scaling list
        result = dequant_8x8(coeffs, qp=26, scaling_list=scaling_list)
        assert result.shape == (8, 8)

    def test_dequant_8x8_custom_scaling_affects_output(self):
        """Custom scaling list should affect dequantization output."""
        from dequant import dequant_8x8
        coeffs = np.ones((8, 8), dtype=np.int32)

        # Default (no scaling list)
        result_default = dequant_8x8(coeffs, qp=26)

        # Custom scaling list (double all values)
        scaling_list = [32] * 64
        result_scaled = dequant_8x8(coeffs, qp=26, scaling_list=scaling_list)

        # Scaled result should be larger
        assert np.sum(np.abs(result_scaled)) > np.sum(np.abs(result_default))

    def test_dequant_8x8_qp_0(self):
        """dequant_8x8 should work with QP=0."""
        from dequant import dequant_8x8
        coeffs = np.array([[10] * 8] * 8, dtype=np.int32)
        result = dequant_8x8(coeffs, qp=0)
        assert result.shape == (8, 8)
        assert np.all(result != 0)

    def test_dequant_8x8_qp_51(self):
        """dequant_8x8 should work with QP=51 (max)."""
        from dequant import dequant_8x8
        coeffs = np.array([[1] * 8] * 8, dtype=np.int32)
        result = dequant_8x8(coeffs, qp=51)
        assert result.shape == (8, 8)


class TestLevelScale8x8:
    """Tests for 8x8 level scale table."""

    def test_level_scale_8x8_exists(self):
        """LEVEL_SCALE_8x8 table should exist."""
        from dequant import LEVEL_SCALE_8x8
        assert LEVEL_SCALE_8x8 is not None

    def test_level_scale_8x8_shape(self):
        """LEVEL_SCALE_8x8 should be 6x6 (qp%6 x position_type)."""
        from dequant import LEVEL_SCALE_8x8
        assert LEVEL_SCALE_8x8.shape == (6, 6)
