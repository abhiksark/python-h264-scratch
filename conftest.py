# h264/conftest.py
"""Pytest configuration and shared fixtures."""

import logging
import numpy as np
import pytest
from pathlib import Path

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent


@pytest.fixture
def test_data_dir(project_root):
    """Return test data directory."""
    return project_root / "test_data"


@pytest.fixture
def reference_dir(test_data_dir):
    """Return JM reference outputs directory."""
    return test_data_dir / "reference"


@pytest.fixture
def sample_yuv_frame():
    """Generate a simple 64x64 YUV 4:2:0 test frame.

    Returns:
        tuple: (Y, Cb, Cr) numpy arrays
            Y: (64, 64) uint8 - luma
            Cb: (32, 32) uint8 - chroma blue
            Cr: (32, 32) uint8 - chroma red
    """
    # Gray frame (Y=128, neutral chroma)
    y = np.full((64, 64), 128, dtype=np.uint8)
    cb = np.full((32, 32), 128, dtype=np.uint8)
    cr = np.full((32, 32), 128, dtype=np.uint8)
    return y, cb, cr


@pytest.fixture
def sample_coefficients_4x4():
    """Generate sample 4x4 coefficient block for testing.

    Returns:
        np.ndarray: (4, 4) int32 coefficient block
    """
    # Typical coefficient pattern (DC + some AC)
    return np.array([
        [64, -8, 4, 0],
        [-4, 2, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.int32)


@pytest.fixture
def sample_prediction_block():
    """Generate sample 16x16 prediction block.

    Returns:
        np.ndarray: (16, 16) int32 prediction values
    """
    return np.full((16, 16), 128, dtype=np.int32)


def assert_pixels_equal(actual, expected, tolerance=0):
    """Assert two pixel arrays are equal within tolerance.

    Args:
        actual: Computed pixel array
        expected: Reference pixel array
        tolerance: Maximum allowed difference per pixel
    """
    diff = np.abs(actual.astype(int) - expected.astype(int))
    max_diff = diff.max()

    if max_diff > tolerance:
        diff_locations = np.argwhere(diff > tolerance)
        pytest.fail(
            f"Pixel mismatch: max_diff={max_diff}, "
            f"locations={diff_locations[:5]}..."  # Show first 5
        )


# Register custom assertion
pytest.assert_pixels_equal = assert_pixels_equal


# JM Reference Comparison Fixtures

@pytest.fixture
def load_jm_reference(reference_dir):
    """Fixture to load JM reference YUV for a test video.

    Usage:
        y, cb, cr = load_jm_reference("quadrant", 32, 32)
    """
    from test_utils.yuv_io import load_yuv_420

    def _load(name: str, width: int, height: int):
        jm_path = reference_dir / f"{name}_jm.yuv"
        if not jm_path.exists():
            pytest.skip(f"JM reference file not found: {jm_path}")
        return load_yuv_420(jm_path, width, height)
    return _load


@pytest.fixture
def compare_decoder_output(reference_dir):
    """Fixture to compare decoder output with JM reference.

    Usage:
        result = compare_decoder_output(decoded_frame, "quadrant")
        assert result.is_perfect_match
    """
    from test_utils.jm_reference import compare_frame_with_jm

    def _compare(decoded_frame, video_name: str):
        jm_path = reference_dir / f"{video_name}_jm.yuv"
        if not jm_path.exists():
            pytest.skip(f"JM reference file not found: {jm_path}")
        return compare_frame_with_jm(decoded_frame, jm_path)
    return _compare


@pytest.fixture
def compare_yuv_arrays(reference_dir):
    """Fixture to compare raw YUV arrays with JM reference.

    Usage:
        result = compare_yuv_arrays(y, cb, cr, "quadrant", 32, 32)
        assert result.is_perfect_match
    """
    from test_utils.jm_reference import compare_with_jm

    def _compare(y, cb, cr, video_name: str, width: int, height: int):
        jm_path = reference_dir / f"{video_name}_jm.yuv"
        if not jm_path.exists():
            pytest.skip(f"JM reference file not found: {jm_path}")
        return compare_with_jm(y, cb, cr, jm_path, width, height)
    return _compare
