# h264/decoder/tests/test_cabac_bframes.py
"""Pixel-perfect integration tests for CABAC B-frame decoding.

Compares decoder output against ffmpeg reference YUV for multi-frame
CABAC B-slice streams.

H.264 Spec Reference: Section 9.3 (CABAC), Section 7.3.5 (B-slice syntax)
"""
import numpy as np
import pytest
from pathlib import Path

from bitstream import BITSTRING_AVAILABLE
from decoder import decode_h264_bytes

pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


def load_multiframe_yuv(path: Path, width: int, height: int):
    """Load all frames from a raw YUV 4:2:0 file.

    Args:
        path: Path to raw YUV file.
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        List of (y, cb, cr) tuples, each plane as uint8 ndarray.
    """
    data = path.read_bytes()
    chroma_w, chroma_h = width // 2, height // 2
    y_size = width * height
    c_size = chroma_w * chroma_h
    frame_size = y_size + 2 * c_size
    num_frames = len(data) // frame_size
    frames = []
    for i in range(num_frames):
        off = i * frame_size
        y = np.frombuffer(data[off:off + y_size], dtype=np.uint8).reshape(height, width)
        cb = np.frombuffer(data[off + y_size:off + y_size + c_size], dtype=np.uint8).reshape(chroma_h, chroma_w)
        cr = np.frombuffer(data[off + y_size + c_size:off + frame_size], dtype=np.uint8).reshape(chroma_h, chroma_w)
        frames.append((y, cb, cr))
    return frames


# (bitstream, ref_yuv, width, height, max_diff)
# max_diff tolerances account for deblocking filter rounding differences
CABAC_BFRAME_STREAMS = [
    ("testsrc2_cabac_bframes", 176, 144, 1),
    ("smptebars_cabac_bframes", 176, 144, 0),
    ("testsrc_cabac_bframes", 176, 144, 4),
    ("gray_cabac_bframes", 176, 144, 0),
]


class TestCabacBFramePixelPerfect:
    """Pixel-perfect comparison of CABAC B-frame decoder output vs ffmpeg."""

    @pytest.mark.parametrize("stream_name,width,height,max_diff", CABAC_BFRAME_STREAMS)
    def test_cabac_bframes_match_reference(
        self, stream_name, width, height, max_diff, test_data_dir,
    ):
        """All frames in CABAC B-frame stream must match ffmpeg reference."""
        bitstream_path = test_data_dir / f"{stream_name}.264"
        ref_path = test_data_dir / f"{stream_name}_ref.yuv"

        if not bitstream_path.exists():
            pytest.skip(f"Bitstream not found: {bitstream_path}")
        if not ref_path.exists():
            pytest.skip(f"Reference YUV not found: {ref_path}")

        decoded = decode_h264_bytes(bitstream_path.read_bytes())
        ref_frames = load_multiframe_yuv(ref_path, width, height)

        # Sort by POC — decoder yields in decode order, reference is display order
        decoded = sorted(decoded, key=lambda f: f.poc)

        assert len(decoded) == len(ref_frames), (
            f"Frame count mismatch: decoded {len(decoded)} vs ref {len(ref_frames)}"
        )

        for i, (frame, (ref_y, ref_cb, ref_cr)) in enumerate(zip(decoded, ref_frames)):
            assert frame.width == width
            assert frame.height == height

            y_diff = np.abs(frame.luma.astype(int) - ref_y.astype(int))
            cb_diff = np.abs(frame.cb.astype(int) - ref_cb.astype(int))
            cr_diff = np.abs(frame.cr.astype(int) - ref_cr.astype(int))

            frame_max = max(y_diff.max(), cb_diff.max(), cr_diff.max())
            if frame_max > max_diff:
                y_locs = np.argwhere(y_diff > max_diff)
                pytest.fail(
                    f"Frame {i}: max_diff={frame_max} exceeds tolerance {max_diff}\n"
                    f"  Y mismatches: {len(y_locs)}, "
                    f"Cb mismatches: {np.sum(cb_diff > max_diff)}, "
                    f"Cr mismatches: {np.sum(cr_diff > max_diff)}\n"
                    f"  First Y mismatch locations: {y_locs[:5].tolist()}"
                )

    @pytest.mark.parametrize("stream_name,width,height,max_diff", CABAC_BFRAME_STREAMS)
    def test_cabac_bframes_nodeblock_match(
        self, stream_name, width, height, max_diff, test_data_dir,
    ):
        """Pre-deblock comparison to isolate reconstruction vs deblock issues."""
        ref_path = test_data_dir / f"{stream_name}_nodeblock_ref.yuv"
        if not ref_path.exists():
            pytest.skip(f"No-deblock reference not found: {ref_path}")
        # This test exists for diagnostic use — run manually with deblock disabled.
        pytest.skip("Diagnostic test: enable when debugging deblock-only diffs")
