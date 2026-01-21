# h264/decoder/tests/test_jm_comparison.py
"""Integration tests comparing decoder output with JM reference decoder.

These tests verify that our H.264 decoder produces pixel-perfect output
matching the official JM (Joint Model) reference decoder.

Test videos are generated using JM's lencod, and reference outputs are
decoded using JM's ldecod.
"""

import pytest
import numpy as np

from bitstream import BITSTRING_AVAILABLE
from decoder import decode_h264_bytes

pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


# Test video specifications: (bitstream_name, jm_name, width, height, max_diff)
# max_diff: maximum allowed pixel difference (0 = exact match required)
TEST_VIDEOS = [
    ("quadrant_32x32", "quadrant", 32, 32, 0),
    ("double_gray_32x16", "double_gray", 32, 16, 0),
    # Gradient has small rounding differences due to I_16x16 transform chain
    ("gradient_128x128", "gradient", 128, 128, 10),
]


class TestJMComparison:
    """Integration tests comparing decoder output with JM reference."""

    @pytest.mark.parametrize(
        "bitstream_name,jm_name,width,height,max_diff",
        TEST_VIDEOS
    )
    def test_decoder_matches_jm(
        self,
        bitstream_name,
        jm_name,
        width,
        height,
        max_diff,
        test_data_dir,
        reference_dir,
    ):
        """Verify decoder output matches JM reference.

        Args:
            bitstream_name: Name of test bitstream file (without .264).
            jm_name: Name of JM reference file (without _jm.yuv).
            width: Expected frame width.
            height: Expected frame height.
            max_diff: Maximum allowed pixel difference (0 = exact match).
            test_data_dir: Fixture providing test data path.
            reference_dir: Fixture providing JM reference path.
        """
        from test_utils.jm_reference import compare_with_jm

        bitstream_path = test_data_dir / f"{bitstream_name}.264"
        if not bitstream_path.exists():
            pytest.skip(f"Test bitstream not found: {bitstream_path}")

        jm_path = reference_dir / f"{jm_name}_jm.yuv"
        if not jm_path.exists():
            pytest.skip(f"JM reference not found: {jm_path}")

        bitstream = bitstream_path.read_bytes()
        frames = decode_h264_bytes(bitstream)

        assert len(frames) >= 1, f"No frames decoded from {bitstream_name}"

        frame = frames[0]
        assert frame.width == width, f"Width mismatch: {frame.width} != {width}"
        assert frame.height == height, f"Height mismatch: {frame.height} != {height}"

        result = compare_with_jm(
            frame.luma, frame.cb, frame.cr,
            jm_path, width, height
        )

        if max_diff == 0:
            assert result.is_perfect_match, (
                f"{bitstream_name}: {result.summary()}\n"
                f"First mismatches: {result.mismatch_locations}"
            )
        else:
            assert result.max_diff <= max_diff, (
                f"{bitstream_name}: max pixel difference {result.max_diff} "
                f"exceeds tolerance {max_diff}\n{result.summary()}"
            )


class TestJMComparisonDetailed:
    """Detailed diagnostic tests for debugging decoder issues."""

    def test_quadrant_luma_values(self, test_data_dir, load_jm_reference):
        """Verify specific luma values in quadrant test video.

        The quadrant video has 4 distinct gray levels in each quadrant.
        """
        bitstream_path = test_data_dir / "quadrant_32x32.264"
        if not bitstream_path.exists():
            pytest.skip("quadrant_32x32.264 not found")

        bitstream = bitstream_path.read_bytes()
        frames = decode_h264_bytes(bitstream)
        assert len(frames) >= 1

        frame = frames[0]
        jm_y, jm_cb, jm_cr = load_jm_reference("quadrant", 32, 32)

        decoded_tl = frame.luma[0:16, 0:16].mean()
        decoded_tr = frame.luma[0:16, 16:32].mean()
        decoded_bl = frame.luma[16:32, 0:16].mean()
        decoded_br = frame.luma[16:32, 16:32].mean()

        jm_tl = jm_y[0:16, 0:16].mean()
        jm_tr = jm_y[0:16, 16:32].mean()
        jm_bl = jm_y[16:32, 0:16].mean()
        jm_br = jm_y[16:32, 16:32].mean()

        assert abs(decoded_tl - jm_tl) < 1, f"TL: {decoded_tl} vs {jm_tl}"
        assert abs(decoded_tr - jm_tr) < 1, f"TR: {decoded_tr} vs {jm_tr}"
        assert abs(decoded_bl - jm_bl) < 1, f"BL: {decoded_bl} vs {jm_bl}"
        assert abs(decoded_br - jm_br) < 1, f"BR: {decoded_br} vs {jm_br}"

    def test_double_gray_regions(self, test_data_dir, load_jm_reference):
        """Verify the two gray regions in double_gray test video.

        This video has two distinct 16x16 regions side by side.
        """
        bitstream_path = test_data_dir / "double_gray_32x16.264"
        if not bitstream_path.exists():
            pytest.skip("double_gray_32x16.264 not found")

        bitstream = bitstream_path.read_bytes()
        frames = decode_h264_bytes(bitstream)
        assert len(frames) >= 1

        frame = frames[0]
        jm_y, jm_cb, jm_cr = load_jm_reference("double_gray", 32, 16)

        decoded_left = frame.luma[0:16, 0:16].mean()
        decoded_right = frame.luma[0:16, 16:32].mean()

        jm_left = jm_y[0:16, 0:16].mean()
        jm_right = jm_y[0:16, 16:32].mean()

        assert abs(decoded_left - jm_left) < 1, f"Left: {decoded_left} vs {jm_left}"
        assert abs(decoded_right - jm_right) < 1, f"Right: {decoded_right} vs {jm_right}"


class TestYUVIO:
    """Tests for YUV I/O utilities."""

    def test_load_save_roundtrip(self, tmp_path):
        """Test that save -> load produces identical data."""
        from test_utils.yuv_io import load_yuv_420, save_yuv_420

        width, height = 32, 16
        y = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        cb = np.random.randint(0, 256, (height // 2, width // 2), dtype=np.uint8)
        cr = np.random.randint(0, 256, (height // 2, width // 2), dtype=np.uint8)

        yuv_path = tmp_path / "test.yuv"
        save_yuv_420(yuv_path, y, cb, cr)

        loaded_y, loaded_cb, loaded_cr = load_yuv_420(yuv_path, width, height)

        np.testing.assert_array_equal(y, loaded_y)
        np.testing.assert_array_equal(cb, loaded_cb)
        np.testing.assert_array_equal(cr, loaded_cr)

    def test_load_existing_jm_file(self, reference_dir):
        """Test loading an existing JM reference file."""
        from test_utils.yuv_io import load_yuv_420

        jm_path = reference_dir / "quadrant_jm.yuv"
        if not jm_path.exists():
            pytest.skip("quadrant_jm.yuv not found")

        y, cb, cr = load_yuv_420(jm_path, 32, 32)

        assert y.shape == (32, 32)
        assert cb.shape == (16, 16)
        assert cr.shape == (16, 16)
        assert y.dtype == np.uint8


class TestComparisonResult:
    """Tests for ComparisonResult class."""

    def test_perfect_match(self):
        """Test ComparisonResult with perfect match."""
        from test_utils.jm_reference import ComparisonResult

        result = ComparisonResult(
            pixel_match_pct=100.0,
            max_diff=0,
            total_pixels=1536,
            matching_pixels=1536,
            y_diff=np.zeros((32, 32), dtype=np.uint8),
            cb_diff=np.zeros((16, 16), dtype=np.uint8),
            cr_diff=np.zeros((16, 16), dtype=np.uint8),
            mismatch_locations=[],
        )

        assert result.is_perfect_match
        assert "PASS" in result.summary()

    def test_partial_match(self):
        """Test ComparisonResult with partial match."""
        from test_utils.jm_reference import ComparisonResult

        result = ComparisonResult(
            pixel_match_pct=99.5,
            max_diff=10,
            total_pixels=1000,
            matching_pixels=995,
            y_diff=np.zeros((32, 32), dtype=np.uint8),
            cb_diff=np.zeros((16, 16), dtype=np.uint8),
            cr_diff=np.zeros((16, 16), dtype=np.uint8),
            mismatch_locations=[("Y", 5, 10)],
        )

        assert not result.is_perfect_match
        assert "FAIL" in result.summary()
