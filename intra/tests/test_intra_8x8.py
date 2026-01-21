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
