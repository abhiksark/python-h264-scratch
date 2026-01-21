# h264/inter/tests/test_chroma_mc.py
"""Tests for chroma motion compensation.

H.264 Spec Reference: Section 8.4.2.2 - Chroma sample interpolation

Chroma uses bilinear interpolation at 1/8-pixel precision.
"""

import pytest
import numpy as np

from inter.motion_comp import (
    get_block_integer,
    get_chroma_block_fractional,
)


class TestChromaIntegerMC:
    """Tests for chroma at integer positions."""

    def test_integer_position_copy(self):
        """Integer position copies block directly."""
        ref = np.arange(64, dtype=np.uint8).reshape(8, 8)

        result = get_chroma_block_fractional(ref, 0, 0, 0, 0, 4, 4)

        np.testing.assert_array_equal(result, ref[0:4, 0:4])

    def test_integer_with_offset(self):
        """Integer position with offset."""
        ref = np.arange(64, dtype=np.uint8).reshape(8, 8)

        result = get_chroma_block_fractional(ref, 2, 2, 0, 0, 4, 4)

        np.testing.assert_array_equal(result, ref[2:6, 2:6])


class TestChromaFractionalMC:
    """Tests for chroma fractional interpolation."""

    @pytest.fixture
    def simple_ref(self):
        """4x4 reference with known values."""
        return np.array([
            [0, 8, 16, 24],
            [8, 16, 24, 32],
            [16, 24, 32, 40],
            [24, 32, 40, 48],
        ], dtype=np.uint8)

    def test_half_horizontal(self, simple_ref):
        """1/2-pixel horizontal (dx=4, dy=0).

        At half-pixel: avg(A, B)
        Between (0,0)=0 and (1,0)=8: result = 4
        """
        result = get_chroma_block_fractional(simple_ref, 0, 0, 4, 0, 1, 1)

        # (0 + 8) / 2 = 4
        assert result[0, 0] == 4

    def test_half_vertical(self, simple_ref):
        """1/2-pixel vertical (dx=0, dy=4).

        Between (0,0)=0 and (0,1)=8: result = 4
        """
        result = get_chroma_block_fractional(simple_ref, 0, 0, 0, 4, 1, 1)

        # (0 + 8) / 2 = 4
        assert result[0, 0] == 4

    def test_half_diagonal(self, simple_ref):
        """1/2-pixel diagonal (dx=4, dy=4).

        Bilinear interpolation of 4 corners:
        A(0,0)=0, B(1,0)=8, C(0,1)=8, D(1,1)=16
        Result = (0 + 8 + 8 + 16) / 4 = 8
        """
        result = get_chroma_block_fractional(simple_ref, 0, 0, 4, 4, 1, 1)

        # avg of 4 corners
        assert result[0, 0] == 8

    def test_quarter_horizontal(self, simple_ref):
        """1/4-pixel horizontal (dx=2, dy=0).

        Weighted: (6*A + 2*B + 4) >> 3
        A=0, B=8: (6*0 + 2*8 + 4) >> 3 = 20 >> 3 = 2
        """
        result = get_chroma_block_fractional(simple_ref, 0, 0, 2, 0, 1, 1)

        # 6/8 * 0 + 2/8 * 8 = 2
        assert result[0, 0] == 2

    def test_three_quarter_horizontal(self, simple_ref):
        """3/4-pixel horizontal (dx=6, dy=0).

        Weighted: (2*A + 6*B + 4) >> 3
        A=0, B=8: (2*0 + 6*8 + 4) >> 3 = 52 >> 3 = 6
        """
        result = get_chroma_block_fractional(simple_ref, 0, 0, 6, 0, 1, 1)

        # 2/8 * 0 + 6/8 * 8 = 6
        assert result[0, 0] == 6

    def test_eighth_pixel(self, simple_ref):
        """1/8-pixel positions (finest granularity)."""
        result = get_chroma_block_fractional(simple_ref, 0, 0, 1, 0, 1, 1)

        # (7*0 + 1*8 + 4) >> 3 = 12 >> 3 = 1
        assert result[0, 0] == 1

    def test_block_extraction(self, simple_ref):
        """Extract larger block with fractional position."""
        # Extend reference for larger extraction
        ref = np.zeros((8, 8), dtype=np.uint8)
        ref[0:4, 0:4] = simple_ref

        result = get_chroma_block_fractional(ref, 0, 0, 4, 0, 2, 2)

        assert result.shape == (2, 2)
        # Each pixel is horizontal half-pixel interpolation
        assert result[0, 0] == 4   # avg(0, 8)
        assert result[0, 1] == 12  # avg(8, 16)


class TestChromaMVMapping:
    """Tests for luma MV to chroma MV conversion."""

    def test_luma_to_chroma_mv(self):
        """Chroma MV is half of luma MV.

        Luma MVx = 8 (2 full pixels in quarter-pel)
        Chroma MVx = 4 (1 full pixel in eighth-pel for 4:2:0)
        """
        luma_mv = 8  # Quarter-pixel units
        chroma_mv = luma_mv >> 1  # Eighth-pixel units (actually half the quarter-pel)

        # Integer part
        chroma_int = chroma_mv >> 2
        chroma_frac = chroma_mv & 3

        assert chroma_int == 1
        assert chroma_frac == 0

    def test_odd_luma_mv(self):
        """Odd luma MV creates fractional chroma position.

        Luma MVx = 5 (1.25 pixels)
        Chroma: 5 >> 1 = 2 (0.5 chroma pixel = 4 eighth-pels)
        """
        luma_mv = 5
        chroma_mv = luma_mv >> 1

        chroma_int = chroma_mv >> 2
        chroma_frac = chroma_mv & 3

        assert chroma_int == 0
        assert chroma_frac == 2


class TestChromaBoundaryHandling:
    """Tests for chroma interpolation at frame boundaries."""

    def test_boundary_clipping(self):
        """Interpolation near boundary clips to valid samples."""
        ref = np.full((8, 8), 100, dtype=np.uint8)
        ref[7, :] = 200  # Bottom row different

        # Extract at bottom edge
        result = get_chroma_block_fractional(ref, 0, 6, 0, 0, 4, 2)

        assert result.shape == (2, 4)
        np.testing.assert_array_equal(result[0, :], 100)
        np.testing.assert_array_equal(result[1, :], 200)

    def test_fractional_at_boundary(self):
        """Fractional position near boundary uses valid neighbors."""
        ref = np.full((4, 4), 100, dtype=np.uint8)
        ref[3, :] = 200

        # Half-pixel vertical at second-to-last row
        result = get_chroma_block_fractional(ref, 0, 2, 0, 4, 1, 1)

        # Between 100 and 200: avg = 150
        assert result[0, 0] == 150
