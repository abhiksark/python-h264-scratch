# h264/intra/tests/test_intra_16x16.py
"""Tests for Intra 16x16 prediction."""

import numpy as np
import pytest

from intra import (
    Intra16x16Mode,
    DEFAULT_PIXEL_VALUE,
    intra_16x16_vertical,
    intra_16x16_horizontal,
    intra_16x16_dc,
    intra_16x16_plane,
    predict_intra_16x16,
    get_neighbors_for_macroblock,
    validate_prediction_mode,
)


class TestIntra16x16Vertical:
    """Tests for vertical prediction mode."""

    def test_vertical_basic(self):
        """Basic vertical prediction."""
        top = np.arange(16, dtype=np.uint8)  # [0, 1, 2, ..., 15]
        result = intra_16x16_vertical(top)

        # Each column should have the same value as top
        for col in range(16):
            assert np.all(result[:, col] == top[col])

    def test_vertical_uniform(self):
        """Uniform top should give uniform block."""
        top = np.full(16, 100, dtype=np.uint8)
        result = intra_16x16_vertical(top)

        assert np.all(result == 100)

    def test_vertical_shape(self):
        """Output should be 16x16."""
        top = np.zeros(16, dtype=np.uint8)
        result = intra_16x16_vertical(top)
        assert result.shape == (16, 16)

    def test_vertical_unavailable(self):
        """When top unavailable, use default value."""
        top = np.zeros(16, dtype=np.uint8)
        result = intra_16x16_vertical(top, top_available=False)

        assert np.all(result == DEFAULT_PIXEL_VALUE)

    def test_vertical_dtype(self):
        """Output should be uint8."""
        top = np.zeros(16, dtype=np.uint8)
        result = intra_16x16_vertical(top)
        assert result.dtype == np.uint8


class TestIntra16x16Horizontal:
    """Tests for horizontal prediction mode."""

    def test_horizontal_basic(self):
        """Basic horizontal prediction."""
        left = np.arange(16, dtype=np.uint8)  # [0, 1, 2, ..., 15]
        result = intra_16x16_horizontal(left)

        # Each row should have the same value as left
        for row in range(16):
            assert np.all(result[row, :] == left[row])

    def test_horizontal_uniform(self):
        """Uniform left should give uniform block."""
        left = np.full(16, 100, dtype=np.uint8)
        result = intra_16x16_horizontal(left)

        assert np.all(result == 100)

    def test_horizontal_shape(self):
        """Output should be 16x16."""
        left = np.zeros(16, dtype=np.uint8)
        result = intra_16x16_horizontal(left)
        assert result.shape == (16, 16)

    def test_horizontal_unavailable(self):
        """When left unavailable, use default value."""
        left = np.zeros(16, dtype=np.uint8)
        result = intra_16x16_horizontal(left, left_available=False)

        assert np.all(result == DEFAULT_PIXEL_VALUE)


class TestIntra16x16DC:
    """Tests for DC prediction mode."""

    def test_dc_both_available(self):
        """DC with both neighbors available."""
        top = np.full(16, 100, dtype=np.uint8)
        left = np.full(16, 100, dtype=np.uint8)
        result = intra_16x16_dc(top, left, True, True)

        # Average of 100 should be 100
        assert np.all(result == 100)

    def test_dc_only_top(self):
        """DC with only top available."""
        top = np.full(16, 100, dtype=np.uint8)
        result = intra_16x16_dc(top, None, top_available=True, left_available=False)

        assert np.all(result == 100)

    def test_dc_only_left(self):
        """DC with only left available."""
        left = np.full(16, 100, dtype=np.uint8)
        result = intra_16x16_dc(None, left, top_available=False, left_available=True)

        assert np.all(result == 100)

    def test_dc_neither_available(self):
        """DC with no neighbors uses default."""
        result = intra_16x16_dc(None, None, False, False)

        assert np.all(result == DEFAULT_PIXEL_VALUE)

    def test_dc_mixed_values(self):
        """DC with different top and left values."""
        top = np.full(16, 100, dtype=np.uint8)
        left = np.full(16, 200, dtype=np.uint8)
        result = intra_16x16_dc(top, left, True, True)

        # Average of 100 and 200 = 150
        # (16*100 + 16*200 + 16) >> 5 = (4816) >> 5 = 150
        expected = (16 * 100 + 16 * 200 + 16) >> 5
        assert np.all(result == expected)

    def test_dc_shape(self):
        """DC output should be 16x16."""
        result = intra_16x16_dc(None, None, False, False)
        assert result.shape == (16, 16)


class TestIntra16x16Plane:
    """Tests for plane prediction mode."""

    def test_plane_uniform_neighbors(self):
        """Uniform neighbors should give uniform result."""
        top = np.full(16, 100, dtype=np.uint8)
        left = np.full(16, 100, dtype=np.uint8)
        top_left = 100

        result = intra_16x16_plane(top, left, top_left)

        # With uniform neighbors, H=V=0, so a=3200, b=c=0
        # pred = (3200 + 0 + 0 + 16) >> 5 = 100.5 -> 100
        assert np.allclose(result, 100, atol=1)

    def test_plane_gradient_horizontal(self):
        """Horizontal gradient in neighbors."""
        top = np.arange(16, dtype=np.uint8)  # [0, 1, ..., 15]
        left = np.full(16, 0, dtype=np.uint8)
        top_left = 0

        result = intra_16x16_plane(top, left, top_left)

        # Should have horizontal gradient (values increase left to right)
        for row in range(16):
            assert result[row, 15] >= result[row, 0]

    def test_plane_shape(self):
        """Plane output should be 16x16."""
        top = np.zeros(16, dtype=np.uint8)
        left = np.zeros(16, dtype=np.uint8)
        result = intra_16x16_plane(top, left, 0)
        assert result.shape == (16, 16)

    def test_plane_fallback_when_unavailable(self):
        """Plane falls back to DC when neighbors unavailable."""
        top = np.zeros(16, dtype=np.uint8)
        left = np.zeros(16, dtype=np.uint8)
        result = intra_16x16_plane(top, left, 0, top_available=False)

        # Should fall back to DC prediction
        assert result.shape == (16, 16)


class TestPredictIntra16x16:
    """Tests for the main prediction function."""

    def test_predict_mode_vertical(self):
        """Mode 0 should use vertical."""
        top = np.full(16, 50, dtype=np.uint8)
        result = predict_intra_16x16(
            mode=Intra16x16Mode.VERTICAL,
            top=top,
            left=None
        )
        assert np.all(result == 50)

    def test_predict_mode_horizontal(self):
        """Mode 1 should use horizontal."""
        left = np.full(16, 60, dtype=np.uint8)
        result = predict_intra_16x16(
            mode=Intra16x16Mode.HORIZONTAL,
            top=None,
            left=left
        )
        assert np.all(result == 60)

    def test_predict_mode_dc(self):
        """Mode 2 should use DC."""
        top = np.full(16, 70, dtype=np.uint8)
        left = np.full(16, 70, dtype=np.uint8)
        result = predict_intra_16x16(
            mode=Intra16x16Mode.DC,
            top=top,
            left=left
        )
        assert np.all(result == 70)

    def test_predict_mode_plane(self):
        """Mode 3 should use plane."""
        top = np.full(16, 80, dtype=np.uint8)
        left = np.full(16, 80, dtype=np.uint8)
        result = predict_intra_16x16(
            mode=Intra16x16Mode.PLANE,
            top=top,
            left=left,
            top_left=80
        )
        assert np.allclose(result, 80, atol=1)

    def test_predict_invalid_mode(self):
        """Invalid mode should raise."""
        with pytest.raises(ValueError):
            predict_intra_16x16(mode=5, top=None, left=None)


class TestGetNeighbors:
    """Tests for neighbor extraction."""

    def test_get_neighbors_top_left_mb(self):
        """Top-left macroblock has no neighbors."""
        frame = np.zeros((64, 64), dtype=np.uint8)
        top, left, top_left, top_avail, left_avail = get_neighbors_for_macroblock(
            frame, mb_x=0, mb_y=0
        )

        assert not top_avail
        assert not left_avail
        assert top is None
        assert left is None

    def test_get_neighbors_second_mb(self):
        """Second MB (0,1) has left neighbor."""
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[:, 15] = 100  # Left edge of second MB

        top, left, top_left, top_avail, left_avail = get_neighbors_for_macroblock(
            frame, mb_x=1, mb_y=0
        )

        assert not top_avail
        assert left_avail
        assert np.all(left == 100)

    def test_get_neighbors_below_first_row(self):
        """MB below first row has top neighbor."""
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[15, :16] = 50  # Bottom of first MB

        top, left, top_left, top_avail, left_avail = get_neighbors_for_macroblock(
            frame, mb_x=0, mb_y=1
        )

        assert top_avail
        assert not left_avail
        assert np.all(top == 50)


class TestValidatePredictionMode:
    """Tests for mode validation."""

    def test_vertical_needs_top(self):
        """Vertical mode requires top."""
        assert validate_prediction_mode(0, top_available=True, left_available=False)
        assert not validate_prediction_mode(0, top_available=False, left_available=True)

    def test_horizontal_needs_left(self):
        """Horizontal mode requires left."""
        assert validate_prediction_mode(1, top_available=False, left_available=True)
        assert not validate_prediction_mode(1, top_available=True, left_available=False)

    def test_dc_always_valid(self):
        """DC mode is always valid."""
        assert validate_prediction_mode(2, top_available=False, left_available=False)
        assert validate_prediction_mode(2, top_available=True, left_available=True)

    def test_plane_needs_both(self):
        """Plane mode requires both neighbors."""
        assert validate_prediction_mode(3, top_available=True, left_available=True)
        assert not validate_prediction_mode(3, top_available=True, left_available=False)
        assert not validate_prediction_mode(3, top_available=False, left_available=True)


class TestIntegration:
    """Integration tests."""

    def test_all_modes_produce_valid_output(self):
        """All modes should produce valid uint8 16x16 blocks."""
        top = np.random.randint(0, 256, 16, dtype=np.uint8)
        left = np.random.randint(0, 256, 16, dtype=np.uint8)
        top_left = np.random.randint(0, 256)

        for mode in range(4):
            result = predict_intra_16x16(
                mode=mode,
                top=top,
                left=left,
                top_left=top_left,
                top_available=True,
                left_available=True
            )
            assert result.shape == (16, 16)
            assert result.dtype == np.uint8
            assert result.min() >= 0
            assert result.max() <= 255

    def test_prediction_plus_residual(self):
        """Verify prediction + residual workflow."""
        # Simulate: prediction + residual = reconstruction
        top = np.full(16, 100, dtype=np.uint8)
        left = np.full(16, 100, dtype=np.uint8)

        prediction = predict_intra_16x16(
            mode=Intra16x16Mode.DC,
            top=top,
            left=left
        )

        # Small residual
        residual = np.full((16, 16), 5, dtype=np.int32)

        # Reconstruction
        reconstruction = np.clip(
            prediction.astype(np.int32) + residual,
            0, 255
        ).astype(np.uint8)

        assert reconstruction.shape == (16, 16)
        assert np.all(reconstruction == 105)
