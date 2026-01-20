# h264/test_utils/yuv_io.py
"""YUV file I/O utilities for testing.

YUV 4:2:0 planar format:
- Y plane: width x height bytes
- Cb plane: (width/2) x (height/2) bytes
- Cr plane: (width/2) x (height/2) bytes
"""

from pathlib import Path
from typing import Union

import numpy as np


def load_yuv_420(
    path: Union[str, Path],
    width: int,
    height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load planar YUV 4:2:0 file.

    Args:
        path: Path to .yuv file.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        Tuple of (Y, Cb, Cr) numpy arrays:
        - Y: (height, width) uint8
        - Cb: (height/2, width/2) uint8
        - Cr: (height/2, width/2) uint8

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file size doesn't match expected frame size.
    """
    path = Path(path)

    y_size = width * height
    chroma_width = width // 2
    chroma_height = height // 2
    chroma_size = chroma_width * chroma_height
    frame_size = y_size + 2 * chroma_size

    data = path.read_bytes()

    if len(data) < frame_size:
        raise ValueError(
            f"File size {len(data)} bytes is smaller than expected "
            f"frame size {frame_size} bytes for {width}x{height} YUV 4:2:0"
        )

    y_data = np.frombuffer(data[:y_size], dtype=np.uint8).reshape(height, width)
    cb_data = np.frombuffer(
        data[y_size:y_size + chroma_size],
        dtype=np.uint8
    ).reshape(chroma_height, chroma_width)
    cr_data = np.frombuffer(
        data[y_size + chroma_size:y_size + 2 * chroma_size],
        dtype=np.uint8
    ).reshape(chroma_height, chroma_width)

    return y_data.copy(), cb_data.copy(), cr_data.copy()


def save_yuv_420(
    path: Union[str, Path],
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray
) -> None:
    """Save YUV 4:2:0 to file.

    Args:
        path: Output path for .yuv file.
        y: Luma plane (height, width) uint8.
        cb: Cb chroma plane (height/2, width/2) uint8.
        cr: Cr chroma plane (height/2, width/2) uint8.

    Raises:
        ValueError: If array dimensions don't match 4:2:0 format.
    """
    path = Path(path)

    if y.dtype != np.uint8:
        y = np.clip(y, 0, 255).astype(np.uint8)
    if cb.dtype != np.uint8:
        cb = np.clip(cb, 0, 255).astype(np.uint8)
    if cr.dtype != np.uint8:
        cr = np.clip(cr, 0, 255).astype(np.uint8)

    height, width = y.shape
    expected_chroma_shape = (height // 2, width // 2)

    if cb.shape != expected_chroma_shape:
        raise ValueError(
            f"Cb shape {cb.shape} doesn't match expected {expected_chroma_shape} "
            f"for luma shape {y.shape}"
        )
    if cr.shape != expected_chroma_shape:
        raise ValueError(
            f"Cr shape {cr.shape} doesn't match expected {expected_chroma_shape} "
            f"for luma shape {y.shape}"
        )

    data = y.tobytes() + cb.tobytes() + cr.tobytes()
    path.write_bytes(data)


def get_frame_size(width: int, height: int) -> int:
    """Calculate total bytes for one YUV 4:2:0 frame.

    Args:
        width: Frame width.
        height: Frame height.

    Returns:
        Total bytes (Y + Cb + Cr planes).
    """
    y_size = width * height
    chroma_size = (width // 2) * (height // 2)
    return y_size + 2 * chroma_size
