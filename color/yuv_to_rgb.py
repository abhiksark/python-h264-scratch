# h264/color/yuv_to_rgb.py
"""YCbCr to RGB color space conversion.

H.264 uses YCbCr color space (often called YUV colloquially).
This module handles:
1. Chroma upsampling (4:2:0 → 4:4:4)
2. Color matrix conversion (YCbCr → RGB)

H.264 Spec Reference: Annex E (Video Usability Information)
Color matrix coefficients defined in E.2.1

Supported standards:
- BT.601: SD video (480i/576i)
- BT.709: HD video (720p/1080p and above)
"""

import logging
from enum import Enum
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ColorMatrix(Enum):
    """Color matrix standards for YCbCr to RGB conversion."""
    BT601 = "bt601"   # SD video
    BT709 = "bt709"   # HD video


# Color conversion matrices
# Formula: R = Y + Cr * Kr
#          G = Y - Cb * Kb_g - Cr * Kr_g
#          B = Y + Cb * Kb
#
# These are derived from the standard color primaries

COLOR_MATRICES = {
    ColorMatrix.BT601: {
        # ITU-R BT.601 (SD video)
        # Kr = 0.299, Kb = 0.114
        "cr_to_r": 1.402,
        "cb_to_b": 1.772,
        "cb_to_g": -0.344136,
        "cr_to_g": -0.714136,
    },
    ColorMatrix.BT709: {
        # ITU-R BT.709 (HD video)
        # Kr = 0.2126, Kb = 0.0722
        "cr_to_r": 1.5748,
        "cb_to_b": 1.8556,
        "cb_to_g": -0.187324,
        "cr_to_g": -0.468124,
    },
}


def upsample_chroma(
    chroma: np.ndarray,
    target_height: int,
    target_width: int,
    method: str = "nearest"
) -> np.ndarray:
    """Upsample chroma plane from 4:2:0 to 4:4:4.

    In 4:2:0, chroma planes are half the resolution of luma in both dimensions.
    This function upsamples them to full resolution.

    Args:
        chroma: Input chroma plane (H/2, W/2)
        target_height: Target height (H)
        target_width: Target width (W)
        method: Upsampling method - "nearest" or "bilinear"

    Returns:
        Upsampled chroma plane (H, W)

    Note:
        Nearest neighbor is simpler and matches many hardware decoders.
        Bilinear gives smoother results but may differ from reference.
    """
    logger.debug(
        f"Upsampling chroma from {chroma.shape} to ({target_height}, {target_width})"
    )

    if method == "nearest":
        # Nearest neighbor: repeat each pixel 2x in both dimensions
        upsampled = np.repeat(np.repeat(chroma, 2, axis=0), 2, axis=1)
    elif method == "bilinear":
        # Simple bilinear interpolation
        # For each output pixel, interpolate from 4 nearest input pixels
        h_in, w_in = chroma.shape
        h_out, w_out = target_height, target_width

        # Create coordinate grids
        y_out = np.arange(h_out)
        x_out = np.arange(w_out)

        # Map to input coordinates (scale by 0.5 since input is half size)
        y_in = y_out * (h_in - 1) / (h_out - 1) if h_out > 1 else np.zeros(h_out)
        x_in = x_out * (w_in - 1) / (w_out - 1) if w_out > 1 else np.zeros(w_out)

        # Get integer and fractional parts
        y0 = np.floor(y_in).astype(int)
        x0 = np.floor(x_in).astype(int)
        y1 = np.minimum(y0 + 1, h_in - 1)
        x1 = np.minimum(x0 + 1, w_in - 1)

        fy = (y_in - y0).reshape(-1, 1)
        fx = (x_in - x0).reshape(1, -1)

        # Bilinear interpolation
        upsampled = (
            chroma[y0][:, x0] * (1 - fy) * (1 - fx) +
            chroma[y0][:, x1] * (1 - fy) * fx +
            chroma[y1][:, x0] * fy * (1 - fx) +
            chroma[y1][:, x1] * fy * fx
        )
    else:
        raise ValueError(f"Unknown upsampling method: {method}")

    # Ensure exact target size (handle odd dimensions)
    upsampled = upsampled[:target_height, :target_width]

    return upsampled.astype(chroma.dtype)


def ycbcr_to_rgb(
    y: np.ndarray,
    cb: np.ndarray,
    cr: np.ndarray,
    color_matrix: ColorMatrix = ColorMatrix.BT601,
    upsample_method: str = "nearest"
) -> np.ndarray:
    """Convert YCbCr 4:2:0 to RGB.

    Args:
        y: Luma plane (H, W), uint8, range [0, 255]
        cb: Chroma blue plane (H/2, W/2), uint8, range [0, 255]
        cr: Chroma red plane (H/2, W/2), uint8, range [0, 255]
        color_matrix: Color standard to use (BT.601 or BT.709)
        upsample_method: Chroma upsampling method

    Returns:
        RGB image (H, W, 3), uint8, range [0, 255]

    Note:
        Input Cb/Cr are in [0, 255] with 128 as neutral.
        We shift to [-128, 127] for the conversion formula.
    """
    height, width = y.shape
    logger.debug(f"Converting YCbCr to RGB: {width}x{height}, matrix={color_matrix.value}")

    # Validate input shapes
    expected_chroma_h = (height + 1) // 2
    expected_chroma_w = (width + 1) // 2
    if cb.shape != (expected_chroma_h, expected_chroma_w):
        logger.warning(
            f"Unexpected Cb shape: {cb.shape}, expected ({expected_chroma_h}, {expected_chroma_w})"
        )
    if cr.shape != (expected_chroma_h, expected_chroma_w):
        logger.warning(
            f"Unexpected Cr shape: {cr.shape}, expected ({expected_chroma_h}, {expected_chroma_w})"
        )

    # Upsample chroma to full resolution
    cb_up = upsample_chroma(cb, height, width, method=upsample_method)
    cr_up = upsample_chroma(cr, height, width, method=upsample_method)

    # Convert to float for computation
    y_f = y.astype(np.float32)
    cb_f = cb_up.astype(np.float32) - 128.0  # Shift to [-128, 127]
    cr_f = cr_up.astype(np.float32) - 128.0

    # Get color matrix coefficients
    matrix = COLOR_MATRICES[color_matrix]

    # Apply color conversion
    # R = Y + Cr * cr_to_r
    # G = Y + Cb * cb_to_g + Cr * cr_to_g
    # B = Y + Cb * cb_to_b
    r = y_f + cr_f * matrix["cr_to_r"]
    g = y_f + cb_f * matrix["cb_to_g"] + cr_f * matrix["cr_to_g"]
    b = y_f + cb_f * matrix["cb_to_b"]

    # Clip to valid range and convert to uint8
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    # Stack into RGB image
    rgb = np.stack([r, g, b], axis=-1)

    logger.debug(f"RGB output shape: {rgb.shape}, dtype: {rgb.dtype}")
    return rgb


def rgb_to_ycbcr(
    rgb: np.ndarray,
    color_matrix: ColorMatrix = ColorMatrix.BT601
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB to YCbCr 4:4:4.

    This is the inverse operation, useful for testing.

    Args:
        rgb: RGB image (H, W, 3), uint8
        color_matrix: Color standard to use

    Returns:
        Tuple of (Y, Cb, Cr) planes, each (H, W), uint8
    """
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    if color_matrix == ColorMatrix.BT601:
        # BT.601 coefficients
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    elif color_matrix == ColorMatrix.BT709:
        # BT.709 coefficients
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        cb = -0.114572 * r - 0.385428 * g + 0.5 * b + 128
        cr = 0.5 * r - 0.454153 * g - 0.045847 * b + 128
    else:
        raise ValueError(f"Unknown color matrix: {color_matrix}")

    y = np.clip(y, 0, 255).astype(np.uint8)
    cb = np.clip(cb, 0, 255).astype(np.uint8)
    cr = np.clip(cr, 0, 255).astype(np.uint8)

    return y, cb, cr


def subsample_chroma(
    cb: np.ndarray,
    cr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample chroma from 4:4:4 to 4:2:0.

    Takes every other pixel in both dimensions (simple box filter).

    Args:
        cb: Chroma blue plane (H, W)
        cr: Chroma red plane (H, W)

    Returns:
        Tuple of (Cb, Cr) subsampled to (H/2, W/2)
    """
    # Simple 2x2 averaging
    h, w = cb.shape
    h2, w2 = h // 2, w // 2

    cb_sub = cb[:h2*2:2, :w2*2:2].astype(np.int32)
    cb_sub = cb_sub + cb[1:h2*2:2, :w2*2:2]
    cb_sub = cb_sub + cb[:h2*2:2, 1:w2*2:2]
    cb_sub = cb_sub + cb[1:h2*2:2, 1:w2*2:2]
    cb_sub = (cb_sub + 2) // 4  # Round

    cr_sub = cr[:h2*2:2, :w2*2:2].astype(np.int32)
    cr_sub = cr_sub + cr[1:h2*2:2, :w2*2:2]
    cr_sub = cr_sub + cr[:h2*2:2, 1:w2*2:2]
    cr_sub = cr_sub + cr[1:h2*2:2, 1:w2*2:2]
    cr_sub = (cr_sub + 2) // 4

    return cb_sub.astype(np.uint8), cr_sub.astype(np.uint8)
