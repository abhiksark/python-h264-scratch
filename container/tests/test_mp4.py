# h264/container/tests/test_mp4.py
"""Tests for MP4 container parser.

ISO 14496-12 (ISOBMFF), ISO 14496-15 (AVCC)
"""
import struct
import pytest


class TestBoxParsing:
    """Test ISO Base Media File Format box parsing."""

    def test_parse_single_box(self):
        """Parse a minimal MP4 box."""
        from container.mp4 import parse_box

        data = struct.pack('>I', 12) + b'ftyp' + b'isom'
        box = parse_box(data, 0)
        assert box.type == 'ftyp'
        assert box.size == 12
        assert box.data == b'isom'

    def test_parse_multiple_boxes(self):
        """Parse consecutive boxes."""
        from container.mp4 import parse_boxes

        box1 = struct.pack('>I', 12) + b'ftyp' + b'isom'
        box2 = struct.pack('>I', 10) + b'free' + b'\x00\x00'
        boxes = parse_boxes(box1 + box2)
        assert len(boxes) == 2
        assert boxes[0].type == 'ftyp'
        assert boxes[1].type == 'free'

    def test_parse_nested_moov(self):
        """Parse moov with nested trak box."""
        from container.mp4 import parse_boxes

        trak_data = b'\x00' * 4
        trak_box = struct.pack('>I', 8 + len(trak_data)) + b'trak' + trak_data
        moov_box = struct.pack('>I', 8 + len(trak_box)) + b'moov' + trak_box
        boxes = parse_boxes(moov_box)
        assert boxes[0].type == 'moov'
        assert len(boxes[0].children) == 1
        assert boxes[0].children[0].type == 'trak'

    def test_parse_64bit_box(self):
        """Parse box with 64-bit extended size."""
        from container.mp4 import parse_box

        payload = b'\xab' * 10
        # size=1 signals 64-bit extended size follows
        data = struct.pack('>I', 1) + b'mdat' + struct.pack('>Q', 26) + payload
        box = parse_box(data, 0)
        assert box.type == 'mdat'
        assert box.size == 26
        assert box.data == payload

    def test_parse_box_extends_to_eof(self):
        """Parse box with size=0 (extends to end of data)."""
        from container.mp4 import parse_box

        payload = b'\x01\x02\x03'
        data = struct.pack('>I', 0) + b'mdat' + payload
        box = parse_box(data, 0)
        assert box.type == 'mdat'
        assert box.size == len(data)
        assert box.data == payload

    def test_find_box(self):
        """Find a box by type in a list."""
        from container.mp4 import parse_boxes, find_box

        box1 = struct.pack('>I', 12) + b'ftyp' + b'isom'
        box2 = struct.pack('>I', 12) + b'moov' + b'\x00' * 4
        boxes = parse_boxes(box1 + box2)
        moov = find_box(boxes, 'moov')
        assert moov is not None
        assert moov.type == 'moov'
        assert find_box(boxes, 'xxxx') is None


class TestAvccParsing:
    """Test AVC Decoder Configuration Record parsing."""

    def test_parse_avcc_extracts_sps_pps(self):
        """Parse avcC box and extract SPS/PPS NAL units."""
        from container.mp4 import parse_avcc

        sps = b'\x67\x64\x00\x1f'
        pps = b'\x68\xee\x38'
        avcc = (
            b'\x01'
            b'\x64\x00\x1f'
            b'\xff'
            + bytes([0xe0 | 1])
            + struct.pack('>H', len(sps)) + sps
            + bytes([1])
            + struct.pack('>H', len(pps)) + pps
        )
        result = parse_avcc(avcc)
        assert result.nal_length_size == 4
        assert len(result.sps_list) == 1
        assert result.sps_list[0] == sps
        assert len(result.pps_list) == 1
        assert result.pps_list[0] == pps

    def test_parse_avcc_nal_length_2(self):
        """Parse avcC with 2-byte NAL length prefix."""
        from container.mp4 import parse_avcc

        sps = b'\x67\x42\x00\x0a'
        pps = b'\x68\xce\x38\x80'
        avcc = (
            b'\x01'
            b'\x42\x00\x0a'
            + bytes([0xfc | 1])  # lengthSizeMinusOne = 1 → 2 bytes
            + bytes([0xe0 | 1])
            + struct.pack('>H', len(sps)) + sps
            + bytes([1])
            + struct.pack('>H', len(pps)) + pps
        )
        result = parse_avcc(avcc)
        assert result.nal_length_size == 2

    def test_parse_avcc_multiple_sps(self):
        """Parse avcC with multiple SPS entries."""
        from container.mp4 import parse_avcc

        sps1 = b'\x67\x64\x00\x1f'
        sps2 = b'\x67\x42\x00\x0a'
        pps = b'\x68\xee\x38'
        avcc = (
            b'\x01'
            b'\x64\x00\x1f'
            b'\xff'
            + bytes([0xe0 | 2])  # 2 SPS
            + struct.pack('>H', len(sps1)) + sps1
            + struct.pack('>H', len(sps2)) + sps2
            + bytes([1])
            + struct.pack('>H', len(pps)) + pps
        )
        result = parse_avcc(avcc)
        assert len(result.sps_list) == 2
        assert result.sps_list[0] == sps1
        assert result.sps_list[1] == sps2


class TestNalExtraction:
    """Test length-prefixed NAL to Annex B conversion."""

    def test_length_prefixed_to_annex_b_4byte(self):
        """Convert 4-byte length-prefixed NALs to Annex B."""
        from container.mp4 import length_prefixed_to_annex_b

        data = (
            struct.pack('>I', 5) + b'\x65\x01\x02\x03\x04'
            + struct.pack('>I', 3) + b'\x41\x01\x02'
        )
        result = length_prefixed_to_annex_b(data, nal_length_size=4)
        expected = (
            b'\x00\x00\x00\x01\x65\x01\x02\x03\x04'
            + b'\x00\x00\x00\x01\x41\x01\x02'
        )
        assert result == expected

    def test_length_prefixed_to_annex_b_2byte(self):
        """Convert 2-byte length-prefixed NALs to Annex B."""
        from container.mp4 import length_prefixed_to_annex_b

        data = struct.pack('>H', 3) + b'\x67\x42\x00'
        result = length_prefixed_to_annex_b(data, nal_length_size=2)
        assert result == b'\x00\x00\x00\x01\x67\x42\x00'

    def test_empty_data(self):
        """Empty data returns empty bytes."""
        from container.mp4 import length_prefixed_to_annex_b

        assert length_prefixed_to_annex_b(b'', nal_length_size=4) == b''


class TestSampleTableParsing:
    """Test MP4 sample table (stbl) parsing."""

    def test_parse_stsz_fixed_size(self):
        """Parse stsz with fixed sample size."""
        from container.mp4 import parse_stsz

        # version=0, flags=0, sample_size=100, sample_count=5
        data = (
            b'\x00\x00\x00\x00'  # version + flags
            + struct.pack('>I', 100)  # sample_size
            + struct.pack('>I', 5)    # sample_count
        )
        sizes = parse_stsz(data)
        assert sizes == [100, 100, 100, 100, 100]

    def test_parse_stsz_variable_sizes(self):
        """Parse stsz with per-sample sizes."""
        from container.mp4 import parse_stsz

        data = (
            b'\x00\x00\x00\x00'
            + struct.pack('>I', 0)     # sample_size=0 → per-sample
            + struct.pack('>I', 3)     # sample_count=3
            + struct.pack('>I', 200)
            + struct.pack('>I', 300)
            + struct.pack('>I', 150)
        )
        sizes = parse_stsz(data)
        assert sizes == [200, 300, 150]

    def test_parse_stco(self):
        """Parse stco (32-bit chunk offsets)."""
        from container.mp4 import parse_stco

        data = (
            b'\x00\x00\x00\x00'
            + struct.pack('>I', 3)
            + struct.pack('>I', 1000)
            + struct.pack('>I', 2000)
            + struct.pack('>I', 3000)
        )
        offsets = parse_stco(data)
        assert offsets == [1000, 2000, 3000]

    def test_parse_stsc(self):
        """Parse stsc (sample-to-chunk mapping)."""
        from container.mp4 import parse_stsc

        data = (
            b'\x00\x00\x00\x00'
            + struct.pack('>I', 2)  # entry_count
            + struct.pack('>III', 1, 3, 1)  # first_chunk=1, samples_per_chunk=3
            + struct.pack('>III', 3, 2, 1)  # first_chunk=3, samples_per_chunk=2
        )
        entries = parse_stsc(data)
        assert entries == [(1, 3, 1), (3, 2, 1)]


class TestMp4Extraction:
    """Test full MP4-to-Annex-B extraction pipeline."""

    def test_extract_h264_from_mp4(self):
        """Full pipeline: MP4 bytes -> Annex B bitstream."""
        from container.mp4 import extract_h264_from_mp4
        import os

        mp4_path = os.path.join(
            os.path.dirname(__file__), '../../test_data/tiny_test.mp4'
        )
        if not os.path.exists(mp4_path):
            pytest.skip("tiny_test.mp4 not generated yet")

        with open(mp4_path, 'rb') as f:
            mp4_data = f.read()

        annex_b = extract_h264_from_mp4(mp4_data)
        # Must start with start code
        assert annex_b[:4] == b'\x00\x00\x00\x01'
        # First NAL should be SPS (type 7)
        nal_type = annex_b[4] & 0x1F
        assert nal_type == 7, f"First NAL should be SPS (7), got {nal_type}"
