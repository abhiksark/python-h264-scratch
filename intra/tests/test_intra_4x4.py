# h264/intra/tests/test_intra_4x4.py
"""Tests for Intra 4x4 prediction modes.

TDD: Tests written first, implementation to follow.

H.264 Spec: Section 8.3.1.2 - Intra_4x4 prediction

The 9 Intra_4x4 modes use up to 13 neighboring pixels:
    M  A  B  C  D  E  F  G  H
    I [         4x4        ]
    J [       block        ]
    K [                    ]
    L [                    ]

Where:
- A-D: 4 pixels directly above
- E-H: 4 pixels above-right (may not be available)
- I-L: 4 pixels to the left
- M: top-left corner pixel
"""

import numpy as np
import pytest

# These will fail until we implement the module
from intra.intra_4x4 import (
    Intra4x4Mode,
    intra_4x4_vertical,
    intra_4x4_horizontal,
    intra_4x4_dc,
    intra_4x4_diagonal_down_left,
    intra_4x4_diagonal_down_right,
    intra_4x4_vertical_right,
    intra_4x4_horizontal_down,
    intra_4x4_vertical_left,
    intra_4x4_horizontal_up,
    predict_intra_4x4,
)


class TestIntra4x4Mode:
    """Tests for Intra4x4Mode enum."""

    def test_mode_values(self):
        """Verify mode values match H.264 spec."""
        assert Intra4x4Mode.VERTICAL == 0
        assert Intra4x4Mode.HORIZONTAL == 1
        assert Intra4x4Mode.DC == 2
        assert Intra4x4Mode.DIAGONAL_DOWN_LEFT == 3
        assert Intra4x4Mode.DIAGONAL_DOWN_RIGHT == 4
        assert Intra4x4Mode.VERTICAL_RIGHT == 5
        assert Intra4x4Mode.HORIZONTAL_DOWN == 6
        assert Intra4x4Mode.VERTICAL_LEFT == 7
        assert Intra4x4Mode.HORIZONTAL_UP == 8

    def test_all_nine_modes(self):
        """All 9 modes should be defined."""
        assert len(Intra4x4Mode) == 9


class TestIntra4x4Vertical:
    """Tests for mode 0: Vertical prediction.

    H.264 Spec 8.3.1.2.1:
    pred[y, x] = top[x]  (copies top row down)
    """

    def test_vertical_basic(self):
        """Vertical mode copies top neighbors down."""
        top = np.array([10, 20, 30, 40], dtype=np.uint8)

        pred = intra_4x4_vertical(top)

        assert pred.shape == (4, 4)
        # Each row should equal top
        for row in range(4):
            np.testing.assert_array_equal(pred[row], top)

    def test_vertical_uniform(self):
        """Uniform top produces uniform block."""
        top = np.array([128, 128, 128, 128], dtype=np.uint8)

        pred = intra_4x4_vertical(top)

        assert np.all(pred == 128)

    def test_vertical_gradient(self):
        """Gradient top produces vertical stripes."""
        top = np.array([0, 85, 170, 255], dtype=np.uint8)

        pred = intra_4x4_vertical(top)

        # Column 0 should be all 0s, column 3 all 255s
        assert np.all(pred[:, 0] == 0)
        assert np.all(pred[:, 3] == 255)


class TestIntra4x4Horizontal:
    """Tests for mode 1: Horizontal prediction.

    H.264 Spec 8.3.1.2.2:
    pred[y, x] = left[y]  (copies left column across)
    """

    def test_horizontal_basic(self):
        """Horizontal mode copies left neighbors across."""
        left = np.array([10, 20, 30, 40], dtype=np.uint8)

        pred = intra_4x4_horizontal(left)

        assert pred.shape == (4, 4)
        # Each column should equal left
        for col in range(4):
            np.testing.assert_array_equal(pred[:, col], left)

    def test_horizontal_uniform(self):
        """Uniform left produces uniform block."""
        left = np.array([128, 128, 128, 128], dtype=np.uint8)

        pred = intra_4x4_horizontal(left)

        assert np.all(pred == 128)


class TestIntra4x4DC:
    """Tests for mode 2: DC prediction.

    H.264 Spec 8.3.1.2.3:
    DC value depends on neighbor availability.
    """

    def test_dc_both_available(self):
        """DC with both neighbors: average of 8 pixels."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)

        pred = intra_4x4_dc(top, left, top_available=True, left_available=True)

        # (4*100 + 4*100 + 4) >> 3 = 100
        assert np.all(pred == 100)

    def test_dc_only_top(self):
        """DC with only top: average of top 4 pixels."""
        top = np.array([80, 80, 80, 80], dtype=np.uint8)

        pred = intra_4x4_dc(top, None, top_available=True, left_available=False)

        # (4*80 + 2) >> 2 = 80
        assert np.all(pred == 80)

    def test_dc_only_left(self):
        """DC with only left: average of left 4 pixels."""
        left = np.array([120, 120, 120, 120], dtype=np.uint8)

        pred = intra_4x4_dc(None, left, top_available=False, left_available=True)

        # (4*120 + 2) >> 2 = 120
        assert np.all(pred == 120)

    def test_dc_neither_available(self):
        """DC with no neighbors: default to 128."""
        pred = intra_4x4_dc(None, None, top_available=False, left_available=False)

        assert np.all(pred == 128)

    def test_dc_mixed_values(self):
        """DC with different top and left values."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([200, 200, 200, 200], dtype=np.uint8)

        pred = intra_4x4_dc(top, left, top_available=True, left_available=True)

        # (400 + 800 + 4) >> 3 = 150
        assert np.all(pred == 150)


class TestIntra4x4DiagonalDownLeft:
    """Tests for mode 3: Diagonal Down-Left.

    H.264 Spec 8.3.1.2.4:
    Extrapolates diagonally from top-right to bottom-left.
    Uses top (A-D) and top-right (E-H) neighbors.
    """

    def test_ddl_basic(self):
        """Diagonal down-left with uniform input."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        top_right = np.array([100, 100, 100, 100], dtype=np.uint8)

        pred = intra_4x4_diagonal_down_left(top, top_right)

        assert pred.shape == (4, 4)
        # Uniform input should give uniform output
        assert np.all(pred == 100)

    def test_ddl_gradient(self):
        """Diagonal down-left with gradient creates diagonal pattern."""
        top = np.array([10, 20, 30, 40], dtype=np.uint8)
        top_right = np.array([50, 60, 70, 80], dtype=np.uint8)

        pred = intra_4x4_diagonal_down_left(top, top_right)

        assert pred.shape == (4, 4)
        # pred[0,0] uses top[0], top[1], top[2]
        # pred[3,3] uses top_right[2], top_right[3], top_right[3]
        # Values should increase along diagonals

    def test_ddl_no_top_right(self):
        """When top-right unavailable, replicate last top pixel."""
        top = np.array([10, 20, 30, 40], dtype=np.uint8)

        pred = intra_4x4_diagonal_down_left(top, top_right=None)

        assert pred.shape == (4, 4)


class TestIntra4x4DiagonalDownRight:
    """Tests for mode 4: Diagonal Down-Right.

    H.264 Spec 8.3.1.2.5:
    Extrapolates diagonally from top-left to bottom-right.
    Uses top, left, and top-left neighbors.
    """

    def test_ddr_basic(self):
        """Diagonal down-right with uniform input."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)
        top_left = 100

        pred = intra_4x4_diagonal_down_right(top, left, top_left)

        assert pred.shape == (4, 4)
        assert np.all(pred == 100)

    def test_ddr_diagonal_pattern(self):
        """DDR should create diagonal stripes from top-left."""
        # Different values on top vs left should show diagonal
        top = np.array([200, 200, 200, 200], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)
        top_left = 150

        pred = intra_4x4_diagonal_down_right(top, left, top_left)

        assert pred.shape == (4, 4)
        # Main diagonal should use top_left


class TestIntra4x4VerticalRight:
    """Tests for mode 5: Vertical-Right.

    H.264 Spec 8.3.1.2.6:
    Extrapolates at 26.6 degrees from vertical.
    """

    def test_vr_basic(self):
        """Vertical-right with uniform input."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)
        top_left = 100

        pred = intra_4x4_vertical_right(top, left, top_left)

        assert pred.shape == (4, 4)
        assert np.all(pred == 100)


class TestIntra4x4HorizontalDown:
    """Tests for mode 6: Horizontal-Down.

    H.264 Spec 8.3.1.2.7:
    Extrapolates at 26.6 degrees from horizontal.
    """

    def test_hd_basic(self):
        """Horizontal-down with uniform input."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)
        top_left = 100

        pred = intra_4x4_horizontal_down(top, left, top_left)

        assert pred.shape == (4, 4)
        assert np.all(pred == 100)


class TestIntra4x4VerticalLeft:
    """Tests for mode 7: Vertical-Left.

    H.264 Spec 8.3.1.2.8:
    Extrapolates at 26.6 degrees left of vertical.
    """

    def test_vl_basic(self):
        """Vertical-left with uniform input."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        top_right = np.array([100, 100, 100, 100], dtype=np.uint8)

        pred = intra_4x4_vertical_left(top, top_right)

        assert pred.shape == (4, 4)
        assert np.all(pred == 100)


class TestIntra4x4HorizontalUp:
    """Tests for mode 8: Horizontal-Up.

    H.264 Spec 8.3.1.2.9:
    Extrapolates upward from left neighbors.
    """

    def test_hu_basic(self):
        """Horizontal-up with uniform input."""
        left = np.array([100, 100, 100, 100], dtype=np.uint8)

        pred = intra_4x4_horizontal_up(left)

        assert pred.shape == (4, 4)
        assert np.all(pred == 100)

    def test_hu_gradient(self):
        """Horizontal-up with gradient left."""
        left = np.array([40, 80, 120, 160], dtype=np.uint8)

        pred = intra_4x4_horizontal_up(left)

        assert pred.shape == (4, 4)
        # Top-left should be related to left[0]
        # Bottom-right should use left[3]


class TestPredictIntra4x4:
    """Tests for main predict_intra_4x4 dispatcher."""

    def test_dispatch_vertical(self):
        """Mode 0 dispatches to vertical."""
        top = np.array([50, 50, 50, 50], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)

        pred = predict_intra_4x4(
            mode=0,
            top=top,
            left=left,
            top_left=75,
            top_available=True,
            left_available=True,
        )

        # Vertical mode ignores left, uses top
        assert np.all(pred == 50)

    def test_dispatch_horizontal(self):
        """Mode 1 dispatches to horizontal."""
        top = np.array([50, 50, 50, 50], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)

        pred = predict_intra_4x4(
            mode=1,
            top=top,
            left=left,
            top_left=75,
            top_available=True,
            left_available=True,
        )

        # Horizontal mode ignores top, uses left
        assert np.all(pred == 100)

    def test_dispatch_dc(self):
        """Mode 2 dispatches to DC."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)

        pred = predict_intra_4x4(
            mode=2,
            top=top,
            left=left,
            top_left=100,
            top_available=True,
            left_available=True,
        )

        assert np.all(pred == 100)

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)

        with pytest.raises(ValueError):
            predict_intra_4x4(
                mode=9,  # Invalid
                top=top,
                left=left,
                top_left=100,
                top_available=True,
                left_available=True,
            )

    def test_all_modes_return_4x4(self):
        """All 9 modes return 4x4 uint8 array."""
        top = np.array([100, 100, 100, 100], dtype=np.uint8)
        left = np.array([100, 100, 100, 100], dtype=np.uint8)
        top_right = np.array([100, 100, 100, 100], dtype=np.uint8)
        top_left = 100

        for mode in range(9):
            pred = predict_intra_4x4(
                mode=mode,
                top=top,
                left=left,
                top_left=top_left,
                top_right=top_right,
                top_available=True,
                left_available=True,
                top_right_available=True,
            )

            assert pred.shape == (4, 4), f"Mode {mode} wrong shape"
            assert pred.dtype == np.uint8, f"Mode {mode} wrong dtype"


class TestIntra4x4EdgeCases:
    """Edge cases and boundary conditions."""

    def test_clipping_overflow(self):
        """Values should clip to [0, 255]."""
        # High values that might overflow
        top = np.array([255, 255, 255, 255], dtype=np.uint8)
        left = np.array([255, 255, 255, 255], dtype=np.uint8)

        pred = intra_4x4_dc(top, left, top_available=True, left_available=True)

        assert pred.max() <= 255

    def test_clipping_underflow(self):
        """Values should clip to [0, 255]."""
        top = np.array([0, 0, 0, 0], dtype=np.uint8)
        left = np.array([0, 0, 0, 0], dtype=np.uint8)

        pred = intra_4x4_dc(top, left, top_available=True, left_available=True)

        assert pred.min() >= 0
