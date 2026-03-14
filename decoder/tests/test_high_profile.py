# h264/decoder/tests/test_high_profile.py
"""High Profile pixel-perfect decode tests.

Verifies that our decoder produces identical output to ffmpeg for real-world
High Profile H.264 videos downloaded from the internet. Tests cover:
- I_8x8 macroblock prediction (all 9 intra modes)
- 8x8 transform and dequantization with flat scaling lists
- Lowpass reference sample filtering
- Mixed I_4x4 / I_8x8 / I_16x16 macroblocks
- Variable QP (17-31)
- Resolutions from 176x144 to 1280x720

H.264 Spec Reference: Sections 8.3.2.2, 8.5.12
"""

import os
import pytest
import numpy as np

from decoder.decoder import H264Decoder


# All test data paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_yuv420(path: str, width: int, height: int):
    """Load a raw YUV 4:2:0 file and return Y, Cb, Cr arrays."""
    with open(path, 'rb') as f:
        data = f.read()
    y_size = width * height
    cb_size = (width // 2) * (height // 2)
    y = np.frombuffer(data[:y_size], dtype=np.uint8).reshape(height, width)
    cb = np.frombuffer(data[y_size:y_size + cb_size], dtype=np.uint8).reshape(
        height // 2, width // 2
    )
    cr = np.frombuffer(data[y_size + cb_size:y_size + 2 * cb_size], dtype=np.uint8).reshape(
        height // 2, width // 2
    )
    return y, cb, cr


def _assert_pixel_perfect(frame, ref_path: str, label: str):
    """Assert decoded frame matches reference YUV exactly."""
    w, h = frame.width, frame.height
    ref_y, ref_cb, ref_cr = _load_yuv420(ref_path, w, h)

    diff_y = np.max(np.abs(ref_y.astype(np.int32) - frame.luma.astype(np.int32)))
    diff_cb = np.max(np.abs(ref_cb.astype(np.int32) - frame.cb.astype(np.int32)))
    diff_cr = np.max(np.abs(ref_cr.astype(np.int32) - frame.cr.astype(np.int32)))

    assert diff_y == 0, f"{label}: Y max_diff={diff_y}"
    assert diff_cb == 0, f"{label}: Cb max_diff={diff_cb}"
    assert diff_cr == 0, f"{label}: Cr max_diff={diff_cr}"


class TestHighProfileSmallStream:
    """Tests with the small 176x144 High Profile test stream."""

    @pytest.fixture
    def stream_path(self):
        return os.path.join(PROJECT_ROOT, 'test_data', 'high_profile_176x144.264')

    @pytest.fixture
    def ref_path(self):
        return os.path.join(
            PROJECT_ROOT, 'test_data', 'reference',
            'high_profile_176x144_nodeblock_ref.yuv',
        )

    def test_pixel_perfect_luma(self, stream_path, ref_path):
        """176x144 High Profile I-frame: luma pixel-perfect."""
        if not os.path.exists(stream_path):
            pytest.skip("Test stream not available")
        d = H264Decoder(deblocking_enabled=False)
        frame = next(d.decode_bytes(open(stream_path, 'rb').read()))
        ref_y, _, _ = _load_yuv420(ref_path, frame.width, frame.height)
        np.testing.assert_array_equal(frame.luma, ref_y)

    def test_pixel_perfect_chroma(self, stream_path, ref_path):
        """176x144 High Profile I-frame: chroma pixel-perfect."""
        if not os.path.exists(stream_path):
            pytest.skip("Test stream not available")
        d = H264Decoder(deblocking_enabled=False)
        frame = next(d.decode_bytes(open(stream_path, 'rb').read()))
        _, ref_cb, ref_cr = _load_yuv420(ref_path, frame.width, frame.height)
        np.testing.assert_array_equal(frame.cb, ref_cb)
        np.testing.assert_array_equal(frame.cr, ref_cr)


class TestHighProfileMP4:
    """Pixel-perfect tests with real MP4 files from the internet."""

    @pytest.mark.parametrize("video,ref,width,height", [
        ("bbb_360_10s.mp4", "bbb_frame1_ref.yuv", 640, 360),
        ("jellyfish_360_10s.mp4", "jellyfish_360_10s_ref.yuv", 640, 360),
        ("bbb_720_10s.mp4", "bbb_720_10s_ref.yuv", 1280, 720),
        ("sintel_360_10s_crf23.mp4", "sintel_crf23_ref.yuv", 640, 360),
    ])
    def test_mp4_frame1_pixel_perfect(self, video, ref, width, height):
        """MP4 frame 1 decode: Y/Cb/Cr all pixel-perfect vs ffmpeg."""
        mp4_path = os.path.join(PROJECT_ROOT, 'test_data', video)
        ref_path = os.path.join(PROJECT_ROOT, 'test_data', ref)
        if not os.path.exists(mp4_path) or not os.path.exists(ref_path):
            pytest.skip(f"Test data not available: {video}")

        d = H264Decoder(deblocking_enabled=False)
        frame = next(d.decode_file(mp4_path))
        assert frame.width == width
        assert frame.height == height
        _assert_pixel_perfect(frame, ref_path, video)

    def test_bbb_multi_frame_decode(self):
        """BBB 360p: decode 5 frames without crashing."""
        mp4_path = os.path.join(PROJECT_ROOT, 'test_data', 'bbb_360_10s.mp4')
        if not os.path.exists(mp4_path):
            pytest.skip("Test data not available")

        d = H264Decoder(deblocking_enabled=False)
        frames = []
        for frame in d.decode_file(mp4_path):
            frames.append(frame)
            assert frame.width == 640
            assert frame.height == 360
            if len(frames) >= 5:
                break
        assert len(frames) == 5
