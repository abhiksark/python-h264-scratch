# h264/intra/tests/test_intra_8x8.py
"""RED TESTS: Intra 8x8 prediction modes for High profile.

H.264 Spec Reference: Section 8.3.1.3 - Intra_8x8 prediction

These tests SHOULD FAIL until 8x8 intra prediction is implemented.
"""

import pytest
import numpy as np


class TestIntra8x8ModeEnum:
    """Tests for Intra8x8Mode enum."""

    def test_intra_8x8_mode_exists(self):
        """Intra8x8Mode enum should exist."""
        from intra.intra_8x8 import Intra8x8Mode
        assert Intra8x8Mode is not None

    def test_intra_8x8_mode_values(self):
        """Intra8x8Mode should have 9 modes (0-8)."""
        from intra.intra_8x8 import Intra8x8Mode

        assert Intra8x8Mode.VERTICAL == 0
        assert Intra8x8Mode.HORIZONTAL == 1
        assert Intra8x8Mode.DC == 2
        assert Intra8x8Mode.DIAGONAL_DOWN_LEFT == 3
        assert Intra8x8Mode.DIAGONAL_DOWN_RIGHT == 4
        assert Intra8x8Mode.VERTICAL_RIGHT == 5
        assert Intra8x8Mode.HORIZONTAL_DOWN == 6
        assert Intra8x8Mode.VERTICAL_LEFT == 7
        assert Intra8x8Mode.HORIZONTAL_UP == 8


class TestIntra8x8Vertical:
    """Tests for mode 0: Vertical prediction."""

    def test_vertical_exists(self):
        """intra_8x8_vertical function should exist."""
        from intra.intra_8x8 import intra_8x8_vertical
        assert callable(intra_8x8_vertical)

    def test_vertical_returns_8x8(self):
        """Vertical prediction should return 8x8 block."""
        from intra.intra_8x8 import intra_8x8_vertical

        top = np.array([100, 110, 120, 130, 140, 150, 160, 170], dtype=np.uint8)
        result = intra_8x8_vertical(top)

        assert result.shape == (8, 8)

    def test_vertical_copies_top_row(self):
        """Vertical mode should copy top pixels to all rows."""
        from intra.intra_8x8 import intra_8x8_vertical

        top = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint8)
        result = intra_8x8_vertical(top)

        for row in range(8):
            np.testing.assert_array_equal(result[row, :], top)


class TestIntra8x8Horizontal:
    """Tests for mode 1: Horizontal prediction."""

    def test_horizontal_exists(self):
        """intra_8x8_horizontal function should exist."""
        from intra.intra_8x8 import intra_8x8_horizontal
        assert callable(intra_8x8_horizontal)

    def test_horizontal_returns_8x8(self):
        """Horizontal prediction should return 8x8 block."""
        from intra.intra_8x8 import intra_8x8_horizontal

        left = np.array([100, 110, 120, 130, 140, 150, 160, 170], dtype=np.uint8)
        result = intra_8x8_horizontal(left)

        assert result.shape == (8, 8)

    def test_horizontal_copies_left_column(self):
        """Horizontal mode should copy left pixels to all columns."""
        from intra.intra_8x8 import intra_8x8_horizontal

        left = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint8)
        result = intra_8x8_horizontal(left)

        for col in range(8):
            np.testing.assert_array_equal(result[:, col], left)


class TestIntra8x8DC:
    """Tests for mode 2: DC prediction."""

    def test_dc_exists(self):
        """intra_8x8_dc function should exist."""
        from intra.intra_8x8 import intra_8x8_dc
        assert callable(intra_8x8_dc)

    def test_dc_returns_8x8(self):
        """DC prediction should return 8x8 block."""
        from intra.intra_8x8 import intra_8x8_dc

        top = np.array([100] * 8, dtype=np.uint8)
        left = np.array([100] * 8, dtype=np.uint8)
        result = intra_8x8_dc(top, left)

        assert result.shape == (8, 8)

    def test_dc_uniform_input(self):
        """DC with uniform neighbors should produce uniform output."""
        from intra.intra_8x8 import intra_8x8_dc

        top = np.array([100] * 8, dtype=np.uint8)
        left = np.array([100] * 8, dtype=np.uint8)
        result = intra_8x8_dc(top, left)

        # All values should be 100 (or very close)
        assert np.all(result == 100)

    def test_dc_average(self):
        """DC should average top and left when both available."""
        from intra.intra_8x8 import intra_8x8_dc

        top = np.array([0] * 8, dtype=np.uint8)
        left = np.array([200] * 8, dtype=np.uint8)
        result = intra_8x8_dc(top, left)

        # Average of 0 and 200 = 100
        expected = 100
        assert np.all(result == expected)


class TestIntra8x8DiagonalDownLeft:
    """Tests for mode 3: Diagonal Down-Left prediction."""

    def test_ddl_exists(self):
        """intra_8x8_diagonal_down_left function should exist."""
        from intra.intra_8x8 import intra_8x8_diagonal_down_left
        assert callable(intra_8x8_diagonal_down_left)

    def test_ddl_returns_8x8(self):
        """DDL prediction should return 8x8 block."""
        from intra.intra_8x8 import intra_8x8_diagonal_down_left

        top = np.arange(8, dtype=np.uint8)
        top_right = np.arange(8, 16, dtype=np.uint8)
        result = intra_8x8_diagonal_down_left(top, top_right)

        assert result.shape == (8, 8)


class TestIntra8x8DiagonalDownRight:
    """Tests for mode 4: Diagonal Down-Right prediction."""

    def test_ddr_exists(self):
        """intra_8x8_diagonal_down_right function should exist."""
        from intra.intra_8x8 import intra_8x8_diagonal_down_right
        assert callable(intra_8x8_diagonal_down_right)

    def test_ddr_returns_8x8(self):
        """DDR prediction should return 8x8 block."""
        from intra.intra_8x8 import intra_8x8_diagonal_down_right

        top = np.arange(8, dtype=np.uint8)
        left = np.arange(8, dtype=np.uint8)
        top_left = 0
        result = intra_8x8_diagonal_down_right(top, left, top_left)

        assert result.shape == (8, 8)


class TestIntra8x8VerticalRight:
    """Tests for mode 5: Vertical-Right prediction."""

    def test_vr_exists(self):
        """intra_8x8_vertical_right function should exist."""
        from intra.intra_8x8 import intra_8x8_vertical_right
        assert callable(intra_8x8_vertical_right)

    def test_vr_returns_8x8(self):
        """VR prediction should return 8x8 block."""
        from intra.intra_8x8 import intra_8x8_vertical_right

        top = np.arange(8, dtype=np.uint8)
        left = np.arange(8, dtype=np.uint8)
        top_left = 0
        result = intra_8x8_vertical_right(top, left, top_left)

        assert result.shape == (8, 8)


class TestIntra8x8HorizontalDown:
    """Tests for mode 6: Horizontal-Down prediction."""

    def test_hd_exists(self):
        """intra_8x8_horizontal_down function should exist."""
        from intra.intra_8x8 import intra_8x8_horizontal_down
        assert callable(intra_8x8_horizontal_down)

    def test_hd_returns_8x8(self):
        """HD prediction should return 8x8 block."""
        from intra.intra_8x8 import intra_8x8_horizontal_down

        top = np.arange(8, dtype=np.uint8)
        left = np.arange(8, dtype=np.uint8)
        top_left = 0
        result = intra_8x8_horizontal_down(top, left, top_left)

        assert result.shape == (8, 8)


class TestIntra8x8VerticalLeft:
    """Tests for mode 7: Vertical-Left prediction."""

    def test_vl_exists(self):
        """intra_8x8_vertical_left function should exist."""
        from intra.intra_8x8 import intra_8x8_vertical_left
        assert callable(intra_8x8_vertical_left)

    def test_vl_returns_8x8(self):
        """VL prediction should return 8x8 block."""
        from intra.intra_8x8 import intra_8x8_vertical_left

        top = np.arange(8, dtype=np.uint8)
        top_right = np.arange(8, 16, dtype=np.uint8)
        result = intra_8x8_vertical_left(top, top_right)

        assert result.shape == (8, 8)


class TestIntra8x8HorizontalUp:
    """Tests for mode 8: Horizontal-Up prediction."""

    def test_hu_exists(self):
        """intra_8x8_horizontal_up function should exist."""
        from intra.intra_8x8 import intra_8x8_horizontal_up
        assert callable(intra_8x8_horizontal_up)

    def test_hu_returns_8x8(self):
        """HU prediction should return 8x8 block."""
        from intra.intra_8x8 import intra_8x8_horizontal_up

        left = np.arange(8, dtype=np.uint8)
        result = intra_8x8_horizontal_up(left)

        assert result.shape == (8, 8)


class TestPredictIntra8x8Dispatcher:
    """Tests for the dispatcher function."""

    def test_predict_intra_8x8_exists(self):
        """predict_intra_8x8 dispatcher function should exist."""
        from intra.intra_8x8 import predict_intra_8x8
        assert callable(predict_intra_8x8)

    def test_predict_intra_8x8_dispatches_vertical(self):
        """Dispatcher should handle mode 0 (vertical)."""
        from intra.intra_8x8 import predict_intra_8x8, Intra8x8Mode

        top = np.array([100] * 8, dtype=np.uint8)
        left = np.array([50] * 8, dtype=np.uint8)
        top_left = 75
        top_right = np.array([100] * 8, dtype=np.uint8)

        result = predict_intra_8x8(
            Intra8x8Mode.VERTICAL,
            top, left, top_left, top_right
        )

        assert result.shape == (8, 8)
        # Vertical mode uses top
        assert np.all(result[0, :] == top)

    def test_predict_intra_8x8_dispatches_horizontal(self):
        """Dispatcher should handle mode 1 (horizontal)."""
        from intra.intra_8x8 import predict_intra_8x8, Intra8x8Mode

        top = np.array([100] * 8, dtype=np.uint8)
        left = np.array([50] * 8, dtype=np.uint8)
        top_left = 75
        top_right = np.array([100] * 8, dtype=np.uint8)

        result = predict_intra_8x8(
            Intra8x8Mode.HORIZONTAL,
            top, left, top_left, top_right
        )

        assert result.shape == (8, 8)
        # Horizontal mode uses left
        assert np.all(result[:, 0] == left)


# =============================================================================
# RED TESTS: Extended 8x8 Prediction Tests for High Profile Features
# These tests SHOULD FAIL until full I_8x8 support is implemented.
# =============================================================================


class TestIntra8x8LowpassFilter:
    """Tests for 8x8 lowpass reference sample filtering.

    H.264 Spec Reference: Section 8.3.4.2.2 - Reference sample filtering
    for Intra_8x8 samples.

    I_8x8 uses a 3-tap lowpass filter [1, 2, 1]/4 on reference samples
    before prediction, unlike I_4x4 which uses unfiltered samples.
    """

    def test_lowpass_filter_function_exists(self):
        """lowpass_filter_8x8 function should exist."""
        from intra.intra_8x8 import lowpass_filter_8x8

        assert callable(lowpass_filter_8x8)

    def test_lowpass_filter_returns_correct_shape(self):
        """Filtered samples should have same shape as input."""
        from intra.intra_8x8 import lowpass_filter_8x8

        # 8 top + 8 top-right + 8 left + 1 top-left = 25 samples
        top = np.full(8, 100, dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        top_left = 100

        filtered = lowpass_filter_8x8(top, left, top_left, top_right)

        assert len(filtered['top']) == 8
        assert len(filtered['left']) == 8

    def test_lowpass_uniform_samples_unchanged(self):
        """Uniform samples should remain unchanged after filtering."""
        from intra.intra_8x8 import lowpass_filter_8x8

        top = np.full(8, 100, dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        top_left = 100

        filtered = lowpass_filter_8x8(top, left, top_left, top_right)

        np.testing.assert_array_equal(filtered['top'], top)
        np.testing.assert_array_equal(filtered['left'], left)
        assert filtered['top_left'] == top_left

    def test_lowpass_smooths_edge_transition(self):
        """Lowpass filter should smooth edge transitions."""
        from intra.intra_8x8 import lowpass_filter_8x8

        # Abrupt transition from 0 to 100
        top = np.array([0, 0, 0, 0, 100, 100, 100, 100], dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 50, dtype=np.uint8)
        top_left = 0

        filtered = lowpass_filter_8x8(top, left, top_left, top_right)

        # After filtering, the transition should be smoothed
        # Position 4 (first 100) should be affected by position 3 (last 0)
        # Filter: [1*0 + 2*100 + 1*100]/4 = 75
        assert filtered['top'][4] < 100
        # Position 3 (last 0) should be affected by position 4 (first 100)
        assert filtered['top'][3] > 0

    def test_lowpass_corner_sample_handling(self):
        """Corner sample (top-left) uses special formula.

        H.264 Spec: p'[-1,-1] = (p[-1,0] + 2*p[-1,-1] + p[0,-1] + 2) >> 2
        """
        from intra.intra_8x8 import lowpass_filter_8x8

        top = np.full(8, 100, dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 200, dtype=np.uint8)
        top_left = 150

        filtered = lowpass_filter_8x8(top, left, top_left, top_right)

        # p'[-1,-1] = (left[0] + 2*top_left + top[0] + 2) >> 2
        # = (200 + 2*150 + 100 + 2) >> 2 = 602 >> 2 = 150
        expected_corner = (200 + 2 * 150 + 100 + 2) >> 2
        assert filtered['top_left'] == expected_corner


class TestIntra8x8DCPredictionModes:
    """Extended tests for DC prediction with neighbor availability.

    H.264 Spec Reference: Section 8.3.3.3 - DC prediction for 8x8 blocks
    """

    def test_dc_only_top_available(self):
        """DC with only top neighbors available."""
        from intra.intra_8x8 import intra_8x8_dc

        top = np.full(8, 80, dtype=np.uint8)

        result = intra_8x8_dc(
            top=top,
            left=None,
            top_available=True,
            left_available=False
        )

        # DC = (sum(top) + 4) >> 3 = (640 + 4) >> 3 = 80
        assert np.all(result == 80)

    def test_dc_only_left_available(self):
        """DC with only left neighbors available."""
        from intra.intra_8x8 import intra_8x8_dc

        left = np.full(8, 120, dtype=np.uint8)

        result = intra_8x8_dc(
            top=None,
            left=left,
            top_available=False,
            left_available=True
        )

        # DC = (sum(left) + 4) >> 3 = (960 + 4) >> 3 = 120
        assert np.all(result == 120)

    def test_dc_neither_available(self):
        """DC with no neighbors available uses default value 128."""
        from intra.intra_8x8 import intra_8x8_dc

        result = intra_8x8_dc(
            top=None,
            left=None,
            top_available=False,
            left_available=False
        )

        assert np.all(result == 128)


class TestIntra8x8DiagonalModeFiltering:
    """Tests for diagonal modes with filtered reference samples.

    H.264 Spec: 8x8 diagonal modes use filtered reference samples
    """

    def test_ddl_uses_filtered_samples(self):
        """DDL mode should use lowpass filtered reference samples."""
        from intra.intra_8x8 import intra_8x8_diagonal_down_left_filtered

        top = np.array([0, 0, 0, 0, 100, 100, 100, 100], dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)

        # Should apply filtering before prediction
        result = intra_8x8_diagonal_down_left_filtered(top, top_right)

        assert result.shape == (8, 8)
        # With filtered samples, transitions should be smoother

    def test_ddr_uses_filtered_samples(self):
        """DDR mode should use lowpass filtered reference samples."""
        from intra.intra_8x8 import intra_8x8_diagonal_down_right_filtered

        top = np.arange(8, dtype=np.uint8) * 20
        left = np.arange(8, dtype=np.uint8) * 20
        top_left = 0

        result = intra_8x8_diagonal_down_right_filtered(top, left, top_left)

        assert result.shape == (8, 8)


class TestIntra8x8SpecFormulas:
    """Tests for exact H.264 specification formulas.

    H.264 Spec Reference: Section 8.3.3.1-8.3.3.9
    These tests verify pixel-level accuracy against spec formulas.
    """

    def test_vertical_exact_formula(self):
        """Vertical: pred8x8[y,x] = p'[x,-1] (Section 8.3.3.1)."""
        from intra.intra_8x8 import intra_8x8_vertical

        top = np.array([10, 30, 50, 70, 90, 110, 130, 150], dtype=np.uint8)

        result = intra_8x8_vertical(top)

        # Each column j should have value p'[j,-1] = top[j]
        for y in range(8):
            for x in range(8):
                assert result[y, x] == top[x], f"Mismatch at ({y},{x})"

    def test_horizontal_exact_formula(self):
        """Horizontal: pred8x8[y,x] = p'[-1,y] (Section 8.3.3.2)."""
        from intra.intra_8x8 import intra_8x8_horizontal

        left = np.array([10, 30, 50, 70, 90, 110, 130, 150], dtype=np.uint8)

        result = intra_8x8_horizontal(left)

        # Each row i should have value p'[-1,i] = left[i]
        for y in range(8):
            for x in range(8):
                assert result[y, x] == left[y], f"Mismatch at ({y},{x})"

    def test_dc_both_available_exact_formula(self):
        """DC with both: (sum(p'[x,-1]) + sum(p'[-1,y]) + 8) >> 4.

        Section 8.3.3.3 case when both top and left available.
        """
        from intra.intra_8x8 import intra_8x8_dc

        # Use values that test rounding: sum = 800 + 1600 + 8 = 2408
        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 200, dtype=np.uint8)

        result = intra_8x8_dc(top, left)

        # (8*100 + 8*200 + 8) >> 4 = (800 + 1600 + 8) >> 4 = 2408 >> 4 = 150
        expected = (800 + 1600 + 8) >> 4
        assert np.all(result == expected)

    def test_ddl_bottom_right_corner_formula(self):
        """DDL corner case: pred8x8[7,7] uses special formula.

        Section 8.3.3.4: pred8x8[7,7] = (p'[14,-1] + 3*p'[15,-1] + 2) >> 2
        """
        from intra.intra_8x8 import intra_8x8_diagonal_down_left

        top = np.arange(8, dtype=np.uint8) * 10  # 0, 10, 20, ..., 70
        top_right = np.arange(8, 16, dtype=np.uint8) * 10  # 80, 90, ..., 150

        result = intra_8x8_diagonal_down_left(top, top_right)

        # pred[7,7] = (p'[14,-1] + 3*p'[15,-1] + 2) >> 2
        # p'[14,-1] = top_right[6] = 140, p'[15,-1] = top_right[7] = 150
        expected_corner = (140 + 3 * 150 + 2) >> 2
        assert result[7, 7] == expected_corner, (
            f"Expected {expected_corner}, got {result[7, 7]}"
        )


class TestIntra8x8AllModesParametrized:
    """Parametrized tests across all 9 prediction modes."""

    @pytest.mark.parametrize("mode", range(9))
    def test_mode_returns_uint8(self, mode):
        """All modes should return uint8 dtype."""
        from intra.intra_8x8 import predict_intra_8x8

        top = np.arange(8, dtype=np.uint8) * 20 + 50
        left = np.arange(8, dtype=np.uint8) * 15 + 60
        top_right = np.arange(8, 16, dtype=np.uint8) * 10
        top_left = 40

        result = predict_intra_8x8(
            mode, top, left, top_left, top_right
        )

        assert result.dtype == np.uint8

    @pytest.mark.parametrize("mode", range(9))
    def test_mode_values_in_valid_range(self, mode):
        """All output values should be in [0, 255]."""
        from intra.intra_8x8 import predict_intra_8x8

        top = np.full(8, 255, dtype=np.uint8)
        left = np.full(8, 0, dtype=np.uint8)
        top_right = np.full(8, 128, dtype=np.uint8)
        top_left = 200

        result = predict_intra_8x8(
            mode, top, left, top_left, top_right
        )

        assert result.min() >= 0
        assert result.max() <= 255

    @pytest.mark.parametrize("mode", range(9))
    def test_mode_uniform_input_reasonable_output(self, mode):
        """Uniform input should produce reasonable (near-uniform) output."""
        from intra.intra_8x8 import predict_intra_8x8

        val = 100
        top = np.full(8, val, dtype=np.uint8)
        left = np.full(8, val, dtype=np.uint8)
        top_right = np.full(8, val, dtype=np.uint8)
        top_left = val

        result = predict_intra_8x8(
            mode, top, left, top_left, top_right
        )

        # With uniform input, output should be close to input value
        assert result.mean() > 80 and result.mean() < 120


class TestIntra8x8EdgeCases:
    """Edge case tests for 8x8 prediction modes."""

    def test_invalid_mode_raises_error(self):
        """Invalid mode (9 or higher) should raise ValueError."""
        from intra.intra_8x8 import predict_intra_8x8

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)

        with pytest.raises(ValueError, match="[Mm]ode|[Ii]nvalid"):
            predict_intra_8x8(9, top, left, 100, top_right)

    def test_negative_mode_raises_error(self):
        """Negative mode should raise ValueError."""
        from intra.intra_8x8 import predict_intra_8x8

        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)

        with pytest.raises(ValueError):
            predict_intra_8x8(-1, top, left, 100, top_right)

    def test_top_right_none_handled(self):
        """Modes requiring top-right should handle None gracefully."""
        from intra.intra_8x8 import intra_8x8_diagonal_down_left

        top = np.full(8, 100, dtype=np.uint8)

        # Should either raise clear error or replicate last top pixel
        try:
            result = intra_8x8_diagonal_down_left(top, top_right=None)
            assert result.shape == (8, 8)
        except (ValueError, TypeError) as e:
            assert "top_right" in str(e).lower() or "neighbor" in str(e).lower()

    def test_all_zero_neighbors(self):
        """All-zero neighbors should produce valid output."""
        from intra.intra_8x8 import predict_intra_8x8

        top = np.zeros(8, dtype=np.uint8)
        left = np.zeros(8, dtype=np.uint8)
        top_right = np.zeros(8, dtype=np.uint8)

        for mode in range(9):
            result = predict_intra_8x8(mode, top, left, 0, top_right)
            assert result.shape == (8, 8)
            assert result.dtype == np.uint8

    def test_all_max_neighbors(self):
        """All-255 neighbors should produce valid output."""
        from intra.intra_8x8 import predict_intra_8x8

        top = np.full(8, 255, dtype=np.uint8)
        left = np.full(8, 255, dtype=np.uint8)
        top_right = np.full(8, 255, dtype=np.uint8)

        for mode in range(9):
            result = predict_intra_8x8(mode, top, left, 255, top_right)
            assert result.shape == (8, 8)
            assert result.max() <= 255


class TestIntra8x8NeighborRequirements:
    """Tests for neighbor availability requirements per mode.

    H.264 Spec: Different modes have different neighbor requirements.
    - Mode 0 (V): top required
    - Mode 1 (H): left required
    - Mode 2 (DC): handles all availability cases
    - Mode 3 (DDL): top required, top-right optional
    - Mode 4 (DDR): top, left, top-left required
    - Mode 5 (VR): top, left, top-left required
    - Mode 6 (HD): top, left, top-left required
    - Mode 7 (VL): top required, top-right optional
    - Mode 8 (HU): left required
    """

    def test_vertical_requires_top(self):
        """Vertical mode should fail or use default without top."""
        from intra.intra_8x8 import intra_8x8_vertical_safe

        # With top unavailable, should return default or raise
        result = intra_8x8_vertical_safe(
            top=None,
            top_available=False
        )

        # Either returns 8x8 with default value (128) or raises
        if result is not None:
            assert result.shape == (8, 8)
            assert np.all(result == 128)

    def test_horizontal_requires_left(self):
        """Horizontal mode should fail or use default without left."""
        from intra.intra_8x8 import intra_8x8_horizontal_safe

        result = intra_8x8_horizontal_safe(
            left=None,
            left_available=False
        )

        if result is not None:
            assert result.shape == (8, 8)
            assert np.all(result == 128)

    def test_ddr_requires_all_three_neighbors(self):
        """DDR requires top, left, and top-left."""
        from intra.intra_8x8 import intra_8x8_diagonal_down_right_safe

        # Missing top-left should fail or use approximation
        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)

        result = intra_8x8_diagonal_down_right_safe(
            top=top,
            left=left,
            top_left=None,
            top_available=True,
            left_available=True,
            top_left_available=False
        )

        # Should either produce valid output or raise
        if result is not None:
            assert result.shape == (8, 8)
