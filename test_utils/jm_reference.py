# h264/test_utils/jm_reference.py
"""JM reference decoder comparison utilities.

Compares our decoder output against JM (Joint Model) reference decoder
to verify correctness.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from test_utils.yuv_io import load_yuv_420


@dataclass
class ComparisonResult:
    """Result of comparing decoded frame with JM reference.

    Attributes:
        pixel_match_pct: Percentage of pixels that match exactly (0-100).
        max_diff: Maximum absolute difference across all pixels.
        total_pixels: Total number of pixels compared.
        matching_pixels: Number of exactly matching pixels.
        y_diff: Per-pixel difference for luma (abs(decoded - reference)).
        cb_diff: Per-pixel difference for Cb chroma.
        cr_diff: Per-pixel difference for Cr chroma.
        mismatch_locations: List of (plane, row, col) for first mismatches.
    """
    pixel_match_pct: float
    max_diff: int
    total_pixels: int
    matching_pixels: int
    y_diff: np.ndarray
    cb_diff: np.ndarray
    cr_diff: np.ndarray
    mismatch_locations: list[tuple[str, int, int]]

    @property
    def is_perfect_match(self) -> bool:
        """Returns True if all pixels match exactly."""
        return self.pixel_match_pct == 100.0

    def summary(self) -> str:
        """Returns human-readable summary string."""
        status = "PASS" if self.is_perfect_match else "FAIL"
        return (
            f"[{status}] pixel_match={self.pixel_match_pct:.1f}%, "
            f"max_diff={self.max_diff}, "
            f"matching={self.matching_pixels}/{self.total_pixels}"
        )


def compare_with_jm(
    decoded_y: np.ndarray,
    decoded_cb: np.ndarray,
    decoded_cr: np.ndarray,
    jm_yuv_path: Union[str, Path],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> ComparisonResult:
    """Compare decoded frame with JM reference.

    Args:
        decoded_y: Decoded luma plane (height, width) uint8.
        decoded_cb: Decoded Cb chroma plane.
        decoded_cr: Decoded Cr chroma plane.
        jm_yuv_path: Path to JM reference .yuv file.
        width: Frame width (inferred from decoded_y if not provided).
        height: Frame height (inferred from decoded_y if not provided).

    Returns:
        ComparisonResult with detailed comparison metrics.
    """
    if width is None:
        height, width = decoded_y.shape

    jm_y, jm_cb, jm_cr = load_yuv_420(jm_yuv_path, width, height)

    y_diff = np.abs(decoded_y.astype(np.int16) - jm_y.astype(np.int16))
    cb_diff = np.abs(decoded_cb.astype(np.int16) - jm_cb.astype(np.int16))
    cr_diff = np.abs(decoded_cr.astype(np.int16) - jm_cr.astype(np.int16))

    y_matches = np.sum(y_diff == 0)
    cb_matches = np.sum(cb_diff == 0)
    cr_matches = np.sum(cr_diff == 0)

    total_pixels = decoded_y.size + decoded_cb.size + decoded_cr.size
    matching_pixels = y_matches + cb_matches + cr_matches

    pixel_match_pct = 100.0 * matching_pixels / total_pixels

    max_diff = max(y_diff.max(), cb_diff.max(), cr_diff.max())

    mismatch_locations = []
    if y_diff.max() > 0:
        rows, cols = np.where(y_diff > 0)
        for i in range(min(3, len(rows))):
            mismatch_locations.append(("Y", int(rows[i]), int(cols[i])))
    if cb_diff.max() > 0:
        rows, cols = np.where(cb_diff > 0)
        for i in range(min(2, len(rows))):
            mismatch_locations.append(("Cb", int(rows[i]), int(cols[i])))
    if cr_diff.max() > 0:
        rows, cols = np.where(cr_diff > 0)
        for i in range(min(2, len(rows))):
            mismatch_locations.append(("Cr", int(rows[i]), int(cols[i])))

    return ComparisonResult(
        pixel_match_pct=pixel_match_pct,
        max_diff=int(max_diff),
        total_pixels=total_pixels,
        matching_pixels=int(matching_pixels),
        y_diff=y_diff.astype(np.uint8),
        cb_diff=cb_diff.astype(np.uint8),
        cr_diff=cr_diff.astype(np.uint8),
        mismatch_locations=mismatch_locations,
    )


def compare_frame_with_jm(
    decoded_frame,
    jm_yuv_path: Union[str, Path],
) -> ComparisonResult:
    """Compare DecodedFrame object with JM reference.

    Args:
        decoded_frame: DecodedFrame from our decoder.
        jm_yuv_path: Path to JM reference .yuv file.

    Returns:
        ComparisonResult with detailed comparison metrics.
    """
    return compare_with_jm(
        decoded_y=decoded_frame.luma,
        decoded_cb=decoded_frame.cb,
        decoded_cr=decoded_frame.cr,
        jm_yuv_path=jm_yuv_path,
        width=decoded_frame.width,
        height=decoded_frame.height,
    )


def print_diff_map(diff: np.ndarray, max_width: int = 80) -> str:
    """Generate ASCII diff map for visual inspection.

    Args:
        diff: Difference array from ComparisonResult.
        max_width: Maximum width for output.

    Returns:
        String showing diff intensity using ASCII characters.
    """
    height, width = diff.shape

    step_x = max(1, width // max_width)
    step_y = max(1, height // (max_width // 2))

    lines = []
    for y in range(0, height, step_y):
        row = []
        for x in range(0, width, step_x):
            val = diff[y, x]
            if val == 0:
                row.append(".")
            elif val < 10:
                row.append(str(val))
            elif val < 50:
                row.append("+")
            elif val < 100:
                row.append("*")
            else:
                row.append("#")
        lines.append("".join(row))

    return "\n".join(lines)
