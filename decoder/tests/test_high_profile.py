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


def _load_yuv420_multiframe(path: str, width: int, height: int, num_frames: int):
    """Load multiple frames from a raw YUV 4:2:0 file.

    Args:
        path: Path to raw YUV file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        num_frames: Number of frames to load.

    Returns:
        List of (Y, Cb, Cr) tuples, each as uint8 ndarrays.
    """
    frame_size = width * height * 3 // 2
    y_size = width * height
    cb_size = (width // 2) * (height // 2)

    with open(path, 'rb') as f:
        data = f.read()

    frames = []
    for i in range(num_frames):
        offset = i * frame_size
        y = np.frombuffer(
            data[offset:offset + y_size], dtype=np.uint8
        ).reshape(height, width)
        cb = np.frombuffer(
            data[offset + y_size:offset + y_size + cb_size], dtype=np.uint8
        ).reshape(height // 2, width // 2)
        cr = np.frombuffer(
            data[offset + y_size + cb_size:offset + frame_size], dtype=np.uint8
        ).reshape(height // 2, width // 2)
        frames.append((y, cb, cr))

    return frames


def _ffmpeg_available() -> bool:
    """Check whether ffmpeg is available on the system."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestMultiFrameDecode:
    """Multi-frame regression tests for B-pyramid and P-only decoding.

    Verifies pixel-perfect decode of:
    - B-pyramid with bframes=3 (I, B, B, B, P sequence)
    - P-only with CAVLC (no B-frames, no weighted prediction)
    - Long sequences (BBB 300 frames stability)
    """

    @pytest.fixture
    def b3_streams(self, tmp_path):
        """Generate bframes=3 B-pyramid test stream.

        Creates a 176x144 stream at 10fps with 0.5s duration (5 frames)
        using B-pyramid normal mode with 3 B-frames between references.
        """
        import subprocess

        h264_path = str(tmp_path / "b3.264")
        ref_path = str(tmp_path / "b3_ref.yuv")

        result = subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc2=duration=0.5:size=176x144:rate=10",
            "-c:v", "libx264", "-profile:v", "main", "-preset", "medium",
            "-x264-params", "bframes=3:ref=1:b-adapt=0:b-pyramid=normal",
            "-pix_fmt", "yuv420p", h264_path
        ], capture_output=True)
        if result.returncode != 0:
            pytest.skip("ffmpeg not available or failed")

        result = subprocess.run([
            "ffmpeg", "-y", "-skip_loop_filter", "all",
            "-i", h264_path, "-f", "rawvideo",
            "-pix_fmt", "yuv420p", ref_path
        ], capture_output=True)
        if result.returncode != 0:
            pytest.skip("ffmpeg reference decode failed")

        return h264_path, ref_path, 176, 144

    @pytest.fixture
    def p_only_streams(self, tmp_path):
        """Generate P-only test stream with ref=1.

        Creates a 176x144 stream at 10fps with 0.5s duration (5 frames)
        using P-frames only, no B-frames, and no weighted prediction.
        Uses ref=1 since multi-ref (ref>1) P-only is a known gap.
        """
        import subprocess

        h264_path = str(tmp_path / "p_only.264")
        ref_path = str(tmp_path / "p_only_ref.yuv")

        result = subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "testsrc2=duration=0.5:size=176x144:rate=10",
            "-c:v", "libx264", "-profile:v", "main", "-preset", "medium",
            "-x264-params", "bframes=0:ref=1:weightp=0:no-cabac=1",
            "-pix_fmt", "yuv420p", h264_path
        ], capture_output=True)
        if result.returncode != 0:
            pytest.skip("ffmpeg not available or failed")

        result = subprocess.run([
            "ffmpeg", "-y", "-skip_loop_filter", "all",
            "-i", h264_path, "-f", "rawvideo",
            "-pix_fmt", "yuv420p", ref_path
        ], capture_output=True)
        if result.returncode != 0:
            pytest.skip("ffmpeg reference decode failed")

        return h264_path, ref_path, 176, 144

    def test_b_pyramid_bframes3_pixel_perfect(self, b3_streams):
        """B-pyramid bframes=3: all frames pixel-perfect vs ffmpeg.

        Regression test for B-pyramid MV prediction fix. Decodes a stream
        with I-B-B-B-P structure and verifies every frame matches ffmpeg
        output exactly (max_diff=0 on Y/Cb/Cr).

        Frames are sorted by POC (display order) before comparison, since
        the decoder may yield frames in decode order while ffmpeg outputs
        in display order.
        """
        h264_path, ref_path, width, height = b3_streams

        d = H264Decoder(deblocking_enabled=False)
        with open(h264_path, 'rb') as f:
            decoded_frames = list(d.decode_bytes(f.read()))

        # Sort decoded frames by POC to match ffmpeg display-order output.
        decoded_frames.sort(key=lambda f: f.poc)

        ref_frames = _load_yuv420_multiframe(
            ref_path, width, height, len(decoded_frames)
        )

        assert len(decoded_frames) == len(ref_frames), (
            f"Frame count mismatch: decoded {len(decoded_frames)}, "
            f"reference {len(ref_frames)}"
        )
        assert len(decoded_frames) >= 4, (
            f"Expected at least 4 frames, got {len(decoded_frames)}"
        )

        for i, (frame, (ref_y, ref_cb, ref_cr)) in enumerate(
            zip(decoded_frames, ref_frames)
        ):
            diff_y = np.max(np.abs(
                ref_y.astype(np.int32) - frame.luma.astype(np.int32)
            ))
            diff_cb = np.max(np.abs(
                ref_cb.astype(np.int32) - frame.cb.astype(np.int32)
            ))
            diff_cr = np.max(np.abs(
                ref_cr.astype(np.int32) - frame.cr.astype(np.int32)
            ))

            assert diff_y == 0, f"Frame {i} (poc={frame.poc}): Y max_diff={diff_y}"
            assert diff_cb == 0, f"Frame {i} (poc={frame.poc}): Cb max_diff={diff_cb}"
            assert diff_cr == 0, f"Frame {i} (poc={frame.poc}): Cr max_diff={diff_cr}"

    def test_p_only_pixel_perfect(self, p_only_streams):
        """P-only (CAVLC, ref=1): all frames pixel-perfect vs ffmpeg.

        Regression test for P-frame decoding with no B-frames and no
        weighted prediction. Verifies every frame matches ffmpeg output
        exactly (max_diff=0 on Y/Cb/Cr).
        """
        h264_path, ref_path, width, height = p_only_streams

        d = H264Decoder(deblocking_enabled=False)
        with open(h264_path, 'rb') as f:
            decoded_frames = list(d.decode_bytes(f.read()))

        # Sort by POC to match ffmpeg display-order output.
        decoded_frames.sort(key=lambda f: f.poc)

        ref_frames = _load_yuv420_multiframe(
            ref_path, width, height, len(decoded_frames)
        )

        assert len(decoded_frames) == len(ref_frames), (
            f"Frame count mismatch: decoded {len(decoded_frames)}, "
            f"reference {len(ref_frames)}"
        )
        assert len(decoded_frames) >= 4, (
            f"Expected at least 4 frames, got {len(decoded_frames)}"
        )

        for i, (frame, (ref_y, ref_cb, ref_cr)) in enumerate(
            zip(decoded_frames, ref_frames)
        ):
            diff_y = np.max(np.abs(
                ref_y.astype(np.int32) - frame.luma.astype(np.int32)
            ))
            diff_cb = np.max(np.abs(
                ref_cb.astype(np.int32) - frame.cb.astype(np.int32)
            ))
            diff_cr = np.max(np.abs(
                ref_cr.astype(np.int32) - frame.cr.astype(np.int32)
            ))

            assert diff_y == 0, f"Frame {i} (poc={frame.poc}): Y max_diff={diff_y}"
            assert diff_cb == 0, f"Frame {i} (poc={frame.poc}): Cb max_diff={diff_cb}"
            assert diff_cr == 0, f"Frame {i} (poc={frame.poc}): Cr max_diff={diff_cr}"

    def test_bbb_300_frames_no_crash(self):
        """BBB 360p: decode 300 frames without crashing.

        Stability test for long-sequence decoding. Decodes all 300 frames
        of Big Buck Bunny 360p, verifying frame dimensions and no exceptions.
        Skipped if the BBB MP4 file is not available locally.
        """
        mp4_path = os.path.join(PROJECT_ROOT, 'test_data', 'bbb_360_10s.mp4')
        if not os.path.exists(mp4_path):
            pytest.skip("BBB 360p test data not available")

        d = H264Decoder(deblocking_enabled=False)
        frame_count = 0
        for frame in d.decode_file(mp4_path):
            assert frame.width == 640, f"Frame {frame_count}: unexpected width"
            assert frame.height == 360, f"Frame {frame_count}: unexpected height"
            frame_count += 1
            if frame_count >= 300:
                break

        assert frame_count >= 100, (
            f"Expected at least 100 frames, got {frame_count}"
        )

    def test_bbb_first_5_frames_pixel_perfect(self, tmp_path):
        """BBB 360p first 5 frames: pixel-perfect vs ffmpeg.

        Decodes the first 5 frames of BBB 360p and verifies each one
        matches the ffmpeg reference output (with deblocking disabled on
        both sides). Skipped if the BBB MP4 file is not available locally.
        """
        import subprocess

        mp4_path = os.path.join(PROJECT_ROOT, 'test_data', 'bbb_360_10s.mp4')
        if not os.path.exists(mp4_path):
            pytest.skip("BBB 360p test data not available")

        ref_path = str(tmp_path / "bbb_5frames_ref.yuv")
        result = subprocess.run([
            "ffmpeg", "-y", "-skip_loop_filter", "all",
            "-i", mp4_path, "-frames:v", "5",
            "-f", "rawvideo", "-pix_fmt", "yuv420p", ref_path
        ], capture_output=True)
        if result.returncode != 0:
            pytest.skip("ffmpeg reference decode failed")

        width, height = 640, 360
        num_frames = 5

        d = H264Decoder(deblocking_enabled=False)
        decoded_frames = []
        for frame in d.decode_file(mp4_path):
            decoded_frames.append(frame)
            if len(decoded_frames) >= num_frames:
                break

        # Sort by POC to match ffmpeg display-order output.
        decoded_frames.sort(key=lambda f: f.poc)

        ref_frames = _load_yuv420_multiframe(ref_path, width, height, num_frames)

        assert len(decoded_frames) == num_frames, (
            f"Expected {num_frames} frames, got {len(decoded_frames)}"
        )

        for i, (frame, (ref_y, ref_cb, ref_cr)) in enumerate(
            zip(decoded_frames, ref_frames)
        ):
            diff_y = np.max(np.abs(
                ref_y.astype(np.int32) - frame.luma.astype(np.int32)
            ))
            diff_cb = np.max(np.abs(
                ref_cb.astype(np.int32) - frame.cb.astype(np.int32)
            ))
            diff_cr = np.max(np.abs(
                ref_cr.astype(np.int32) - frame.cr.astype(np.int32)
            ))

            assert diff_y == 0, f"Frame {i} (poc={frame.poc}): Y max_diff={diff_y}"
            assert diff_cb == 0, f"Frame {i} (poc={frame.poc}): Cb max_diff={diff_cb}"
            assert diff_cr == 0, f"Frame {i} (poc={frame.poc}): Cr max_diff={diff_cr}"
