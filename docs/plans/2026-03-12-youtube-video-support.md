# YouTube Video Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decode real-world YouTube H.264 videos (Main & High Profile) by adding MP4 demuxing, completing CABAC B-frames, and integrating 8x8 transform into the decoder pipeline.

**Architecture:** Three independent phases: (1) CABAC B-frame bug fix + validation — nearest to working, (2) MP4 container demuxer — new module extracting NAL units from .mp4/.mkv, (3) High Profile 8x8 transform pipeline integration — threading existing 8x8 modules into the CABAC decoder. Each phase is independently testable and valuable.

**Tech Stack:** Python 3, NumPy, pytest. No external dependencies for container parsing (pure Python struct-based MP4 parser).

---

## Phase 1: CABAC B-Frame Completion

### Task 1: Fix B_8x8 sub-partition MV prediction bug

The same bug that was fixed for P_8x8 exists in `_cabac_b_8x8_mc()`: only j==0 sub-partition gets MV prediction; j>0 sub-partitions get MVD applied to (0,0) instead of the predicted MV.

**Files:**
- Modify: `decoder/decoder.py:3443-3549` (`_cabac_b_8x8_mc`)

**Step 1: Write the failing test**

Create a unit test that constructs a B_8x8 with sub_type=4 (B_L0_8x4, 2 parts) and verifies both sub-partitions get correct MV prediction, not just the first.

Create file: `decoder/tests/test_cabac_b_8x8_sub.py`

```python
# decoder/tests/test_cabac_b_8x8_sub.py
"""Test CABAC B_8x8 sub-partition MV prediction.

Verifies fix for bug where only j==0 sub-partition gets MV prediction.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from inter.mv_prediction import MVCache


class TestCabacB8x8SubPartitionMV:
    """Verify each sub-partition in B_8x8 gets its own MV prediction."""

    def test_l0_8x4_both_parts_get_mv_prediction(self):
        """B_L0_8x4 (sub_type=4) has 2 parts: both need MV prediction."""
        # sub_type=4 is B_L0_8x4 → 2 sub-partitions, both use L0
        # With MVDs of (2,3) and (4,5), and mock predictor returning (10,20),
        # final MVs should be (12,23) and (14,25) — NOT (12,23) and (4,5).
        from decoder.decoder import H264Decoder

        decoder = H264Decoder()
        # Set up minimal state
        decoder.state.ref_buffer = MagicMock()
        ref_frame = MagicMock()
        ref_frame.luma = np.zeros((32, 32), dtype=np.uint8)
        ref_frame.cb = np.zeros((16, 16), dtype=np.uint8)
        ref_frame.cr = np.zeros((16, 16), dtype=np.uint8)
        decoder.state.ref_buffer.get_frame.return_value = ref_frame
        decoder.state.ref_buffer.get_l0_frame.return_value = ref_frame
        decoder.state.ref_buffer.get_l1_frame.return_value = ref_frame
        ref_frame.poc = 0

        decoder.state.mv_cache = MVCache(2, 2)
        decoder.state.mv_cache_l1 = MVCache(2, 2)

        # sub_type=4 (B_L0_8x4): 2 parts, L0 only
        sub_types = [4, 0, 0, 0]  # Only sub_idx=0 matters
        ref_idx_l0 = [0, 0, 0, 0]
        ref_idx_l1 = [0, 0, 0, 0]
        mvd_l0 = [(2, 3), (4, 5)]  # Two MVDs for two parts
        mvd_l1 = []

        # Track what MVs get stored in cache
        stored_mvs = {}
        original_set_mv = decoder.state.mv_cache.set_mv

        def tracking_set_mv(mbx, mby, bx, by, mvx, mvy, ref_idx=0):
            stored_mvs[(bx, by)] = (mvx, mvy)
            original_set_mv(mbx, mby, bx, by, mvx, mvy, ref_idx=ref_idx)

        with patch.object(decoder.state.mv_cache, 'set_mv', side_effect=tracking_set_mv):
            with patch('decoder.decoder.predict_mv_partition', return_value=(10, 20)):
                decoder._cabac_b_8x8_mc(
                    sub_types, ref_idx_l0, ref_idx_l1,
                    mvd_l0, mvd_l1, 0, 0,
                )

        # Part 0 (top 8x4): blocks (0,0) and (1,0) should have MV = (10+2, 20+3) = (12, 23)
        # Part 1 (bottom 8x4): blocks (0,1) and (1,1) should have MV = (10+4, 20+5) = (14, 25)
        # Bug: part 1 would get (0+4, 0+5) = (4, 5) without fix
        assert stored_mvs.get((0, 0)) == (12, 23), f"Part 0 MV wrong: {stored_mvs.get((0, 0))}"
        assert stored_mvs.get((0, 1)) == (14, 25), f"Part 1 MV wrong: {stored_mvs.get((0, 1))}"
```

**Step 2: Run test to verify it fails**

Run: `pytest decoder/tests/test_cabac_b_8x8_sub.py -v`
Expected: FAIL — part 1 gets (4, 5) instead of (14, 25)

**Step 3: Implement the fix**

Rewrite `_cabac_b_8x8_mc()` in `decoder/decoder.py:3443-3549` to use per-sub-partition MV prediction, following the same pattern as the P_8x8 fix at line 3142-3224.

Key changes:
- Import `predict_mv_partition` (already imported at top of file)
- For each sub-partition j, compute block position `(sub_bx + dx, sub_by + dy)`
- Call `predict_mv_partition()` for each j, not just j==0
- Store per-block MVs in cache (not one MV for entire sub-MB)
- Use sub-partition-aware chroma reconstruction

Replace the L0/L1 loops (lines 3488-3510) with:

```python
# Sub-partition layout tables (same as P_8x8)
_b_sub_sp = {
    # sub_shape → [(block_offset, part_width_4x4, coverage_blocks)]
    '8x8': [((0, 0), 2, [(0,0),(1,0),(0,1),(1,1)])],
    '8x4': [((0, 0), 2, [(0,0),(1,0)]),
            ((0, 1), 2, [(0,1),(1,1)])],
    '4x8': [((0, 0), 1, [(0,0),(0,1)]),
            ((1, 0), 1, [(1,0),(1,1)])],
    '4x4': [((0, 0), 1, [(0,0)]), ((1, 0), 1, [(1,0)]),
            ((0, 1), 1, [(0,1)]), ((1, 1), 1, [(1,1)])],
}
_b_sub_shape = {
    1: '8x8', 2: '8x8', 3: '8x8',
    4: '8x4', 6: '8x4', 8: '8x4',
    5: '4x8', 7: '4x8', 9: '4x8',
    10: '4x4', 11: '4x4', 12: '4x4',
}

# For each sub-partition j, predict MV at correct block position
shape = _b_sub_shape.get(sub_type, '8x8')
parts = _b_sub_sp[shape]

sub_mvs_l0 = []
sub_mvs_l1 = []

if _b_sub_uses_l0(sub_type):
    for j, ((dx, dy), pw, coverage) in enumerate(parts):
        mvd = mvd_l0[l0_mvd_idx] if l0_mvd_idx < len(mvd_l0) else (0, 0)
        l0_mvd_idx += 1
        mvp_x, mvp_y = predict_mv_partition(
            self.state.mv_cache, mb_x, mb_y,
            block_x=sub_bx + dx, block_y=sub_by + dy,
            part_width_blocks=pw, target_ref=refs_l0[sub_idx],
        )
        final_mvx = mvp_x + mvd[0]
        final_mvy = mvp_y + mvd[1]
        sub_mvs_l0.append((final_mvx, final_mvy))
        for (bx_off, by_off) in coverage:
            self.state.mv_cache.set_mv(
                mb_x, mb_y, sub_bx + bx_off, sub_by + by_off,
                final_mvx, final_mvy, ref_idx=refs_l0[sub_idx],
            )

# Same pattern for L1 using mv_cache_l1
```

**Step 4: Run test to verify it passes**

Run: `pytest decoder/tests/test_cabac_b_8x8_sub.py -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `pytest -x -q`
Expected: All existing tests pass

**Step 6: Commit**

```bash
git add decoder/decoder.py decoder/tests/test_cabac_b_8x8_sub.py
git commit -m "fix(cabac): per-sub-partition MV prediction for B_8x8 modes"
```

---

### Task 2: Generate CABAC B-frame reference data

We have `test_data/testsrc2_cabac_bframes.264` but no reference YUV. Generate it.

**Files:**
- Create: `test_data/testsrc2_cabac_bframes_ref.yuv` (via ffmpeg)

**Step 1: Inspect the test stream**

Run: `ffprobe test_data/testsrc2_cabac_bframes.264`
Note the resolution and frame count.

**Step 2: Generate reference YUV with ffmpeg**

Run: `ffmpeg -i test_data/testsrc2_cabac_bframes.264 -f rawvideo -pix_fmt yuv420p test_data/testsrc2_cabac_bframes_ref.yuv`

**Step 3: Generate pre-deblock reference (for diagnosing deblock-only diffs)**

Run: `ffmpeg -skip_loop_filter all -i test_data/testsrc2_cabac_bframes.264 -f rawvideo -pix_fmt yuv420p test_data/testsrc2_cabac_bframes_nodeblock_ref.yuv`

**Step 4: Commit test data**

```bash
git add test_data/testsrc2_cabac_bframes_ref.yuv test_data/testsrc2_cabac_bframes_nodeblock_ref.yuv
git commit -m "test: add CABAC B-frame reference YUV files"
```

---

### Task 3: Write CABAC B-frame pixel-perfect integration test

**Files:**
- Modify: `decoder/tests/test_decoder_cabac.py` (add B-frame test case)

**Step 1: Write the integration test**

Follow existing pattern in test_decoder_cabac.py — decode the bitstream, compare Y/Cb/Cr against reference YUV frame by frame.

```python
def test_cabac_bframes_pixel_perfect(self):
    """CABAC B-frame stream must match ffmpeg reference."""
    from decoder import decode_h264_file
    from test_utils.yuv_io import load_yuv_420

    bitstream = "test_data/testsrc2_cabac_bframes.264"
    ref_yuv = "test_data/testsrc2_cabac_bframes_ref.yuv"

    frames = list(decode_h264_file(bitstream))
    assert len(frames) > 0

    width, height = frames[0].width, frames[0].height
    # Load all reference frames
    with open(ref_yuv, "rb") as f:
        ref_data = f.read()
    frame_size = width * height + 2 * (width // 2) * (height // 2)
    num_ref = len(ref_data) // frame_size

    assert len(frames) == num_ref, f"Frame count mismatch: {len(frames)} vs {num_ref}"

    for i, frame in enumerate(frames):
        offset = i * frame_size
        ref_y = np.frombuffer(ref_data[offset:offset + width * height], dtype=np.uint8).reshape(height, width)
        chroma_size = (width // 2) * (height // 2)
        ref_cb = np.frombuffer(ref_data[offset + width * height:offset + width * height + chroma_size], dtype=np.uint8).reshape(height // 2, width // 2)
        ref_cr = np.frombuffer(ref_data[offset + width * height + chroma_size:offset + frame_size], dtype=np.uint8).reshape(height // 2, width // 2)

        np.testing.assert_array_equal(frame.luma, ref_y, err_msg=f"Frame {i} luma mismatch")
        np.testing.assert_array_equal(frame.cb, ref_cb, err_msg=f"Frame {i} Cb mismatch")
        np.testing.assert_array_equal(frame.cr, ref_cr, err_msg=f"Frame {i} Cr mismatch")
```

**Step 2: Run test**

Run: `pytest decoder/tests/test_decoder_cabac.py::TestCabacBFrames -v`
Expected: May fail initially — debug and fix until pixel-perfect.

**Step 3: Debug any mismatches**

Use the pre-deblock reference to isolate deblock vs reconstruction issues. Compare frame-by-frame, MB-by-MB if needed. Fix bugs as found (likely in B_8x8 sub-partition handling, direct mode, or bipred).

**Step 4: Commit when passing**

```bash
git add decoder/tests/test_decoder_cabac.py
git commit -m "test: add CABAC B-frame pixel-perfect integration test"
```

---

### Task 4: Generate additional CABAC B-frame test streams

More diverse streams improve coverage. Generate streams with different content, multiple references, and varied sub-partition types.

**Step 1: Generate test streams with ffmpeg**

```bash
# Main profile CABAC B-frames with 2 refs
ffmpeg -f lavfi -i testsrc2=duration=1:size=176x144:rate=15 \
  -c:v libx264 -profile:v main \
  -x264-params "bframes=2:ref=2:b-adapt=0" \
  -pix_fmt yuv420p test_data/cabac_bframes_2ref.264

ffmpeg -i test_data/cabac_bframes_2ref.264 \
  -f rawvideo -pix_fmt yuv420p test_data/cabac_bframes_2ref_ref.yuv
```

**Step 2: Add parametrized test covering all CABAC B-frame streams**

Extend the test in Task 3 to be parametrized over multiple streams.

**Step 3: Commit**

```bash
git add test_data/cabac_bframes_* decoder/tests/test_decoder_cabac.py
git commit -m "test: add additional CABAC B-frame test coverage"
```

---

## Phase 2: MP4 Container Demuxer

### Task 5: Write MP4 box parser

Parse the ISO Base Media File Format (ISO 14496-12) box structure.

**Files:**
- Create: `container/__init__.py`
- Create: `container/mp4.py`
- Create: `container/tests/__init__.py`
- Create: `container/tests/test_mp4.py`

**Step 1: Write failing test for box parsing**

```python
# container/tests/test_mp4.py
"""Tests for MP4 container parser."""
import struct
import pytest
from container.mp4 import parse_box, parse_boxes


class TestBoxParsing:
    def test_parse_single_box(self):
        """Parse a minimal MP4 box."""
        # Box: size=12, type='ftyp', data=b'isom'
        data = struct.pack('>I', 12) + b'ftyp' + b'isom'
        box = parse_box(data, 0)
        assert box.type == 'ftyp'
        assert box.size == 12
        assert box.data == b'isom'

    def test_parse_nested_moov(self):
        """Parse moov with nested trak box."""
        trak_data = b'\x00' * 4
        trak_box = struct.pack('>I', 8 + len(trak_data)) + b'trak' + trak_data
        moov_box = struct.pack('>I', 8 + len(trak_box)) + b'moov' + trak_box
        boxes = parse_boxes(moov_box)
        assert boxes[0].type == 'moov'
```

**Step 2: Run test to verify it fails**

Run: `pytest container/tests/test_mp4.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement box parser**

```python
# container/mp4.py
"""MP4 (ISO 14496-12) container parser for H.264 NAL unit extraction.

Parses MP4 box structure and extracts H.264 NAL units from video tracks.
Converts length-prefixed NALs (AVCC format) to Annex B format for the decoder.

H.264 Spec Reference: ISO 14496-15 (AVCC), ISO 14496-12 (ISOBMFF)
"""
import struct
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Box:
    """ISO Base Media File Format box."""
    type: str
    size: int
    data: bytes
    offset: int = 0
    children: List['Box'] = field(default_factory=list)


# Container boxes that hold child boxes
CONTAINER_BOXES = {'moov', 'trak', 'mdia', 'minf', 'stbl', 'edts', 'dinf'}


def parse_box(data: bytes, offset: int) -> Optional[Box]:
    """Parse a single MP4 box at the given offset."""
    if offset + 8 > len(data):
        return None

    size = struct.unpack('>I', data[offset:offset + 4])[0]
    box_type = data[offset + 4:offset + 8].decode('ascii', errors='replace')

    header_size = 8
    if size == 1:  # 64-bit extended size
        if offset + 16 > len(data):
            return None
        size = struct.unpack('>Q', data[offset + 8:offset + 16])[0]
        header_size = 16
    elif size == 0:  # Box extends to end of file
        size = len(data) - offset

    box_data = data[offset + header_size:offset + size]
    box = Box(type=box_type, size=size, data=box_data, offset=offset)

    if box_type in CONTAINER_BOXES:
        box.children = parse_boxes(box_data)

    return box


def parse_boxes(data: bytes) -> List[Box]:
    """Parse all top-level boxes in data."""
    boxes = []
    offset = 0
    while offset < len(data):
        box = parse_box(data, offset)
        if box is None or box.size < 8:
            break
        boxes.append(box)
        offset += box.size
    return boxes
```

**Step 4: Run test to verify it passes**

Run: `pytest container/tests/test_mp4.py::TestBoxParsing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add container/__init__.py container/mp4.py container/tests/__init__.py container/tests/test_mp4.py
git commit -m "feat(container): add MP4 box parser"
```

---

### Task 6: Parse AVCC (avcC) box for SPS/PPS extraction

The avcC box inside stsd/avc1 contains SPS and PPS in length-prefixed format.

**Files:**
- Modify: `container/mp4.py`
- Modify: `container/tests/test_mp4.py`

**Step 1: Write failing test**

```python
class TestAvccParsing:
    def test_parse_avcc_extracts_sps_pps(self):
        """Parse avcC box and extract SPS/PPS NAL units."""
        from container.mp4 import parse_avcc

        # Minimal avcC: version=1, profile=100, compat=0, level=31
        # nal_length_size=4 (stored as 0xFF = length_size_minus_one=3)
        # 1 SPS: 4 bytes; 1 PPS: 3 bytes
        sps = b'\x67\x64\x00\x1f'  # Fake SPS NAL (type 7)
        pps = b'\x68\xee\x38'       # Fake PPS NAL (type 8)
        avcc = (
            b'\x01'                  # configurationVersion
            b'\x64\x00\x1f'         # profile, compat, level
            b'\xff'                  # lengthSizeMinusOne = 3 (4 bytes)
            + bytes([0xe0 | 1])     # numOfSequenceParameterSets = 1
            + struct.pack('>H', len(sps)) + sps
            + bytes([1])            # numOfPictureParameterSets = 1
            + struct.pack('>H', len(pps)) + pps
        )
        result = parse_avcc(avcc)
        assert result.nal_length_size == 4
        assert len(result.sps_list) == 1
        assert result.sps_list[0] == sps
        assert len(result.pps_list) == 1
        assert result.pps_list[0] == pps
```

**Step 2: Run test to verify it fails**

Run: `pytest container/tests/test_mp4.py::TestAvccParsing -v`
Expected: FAIL

**Step 3: Implement avcC parser**

Add to `container/mp4.py`:

```python
@dataclass
class AVCConfig:
    """AVC Decoder Configuration Record (ISO 14496-15 Section 5.2.4.1)."""
    profile_idc: int
    level_idc: int
    nal_length_size: int  # 1, 2, or 4
    sps_list: List[bytes] = field(default_factory=list)
    pps_list: List[bytes] = field(default_factory=list)


def parse_avcc(data: bytes) -> AVCConfig:
    """Parse AVCDecoderConfigurationRecord from avcC box data."""
    if len(data) < 7:
        raise ValueError(f"avcC too short: {len(data)} bytes")

    version = data[0]
    profile_idc = data[1]
    level_idc = data[3]
    nal_length_size = (data[4] & 0x03) + 1

    config = AVCConfig(
        profile_idc=profile_idc,
        level_idc=level_idc,
        nal_length_size=nal_length_size,
    )

    offset = 5
    num_sps = data[offset] & 0x1F
    offset += 1

    for _ in range(num_sps):
        sps_len = struct.unpack('>H', data[offset:offset + 2])[0]
        offset += 2
        config.sps_list.append(data[offset:offset + sps_len])
        offset += sps_len

    num_pps = data[offset]
    offset += 1

    for _ in range(num_pps):
        pps_len = struct.unpack('>H', data[offset:offset + 2])[0]
        offset += 2
        config.pps_list.append(data[offset:offset + pps_len])
        offset += pps_len

    return config
```

**Step 4: Run test to verify it passes**

Run: `pytest container/tests/test_mp4.py::TestAvccParsing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add container/mp4.py container/tests/test_mp4.py
git commit -m "feat(container): parse avcC box for SPS/PPS extraction"
```

---

### Task 7: Extract NAL units from MP4 samples

Parse stbl (sample table) to find sample offsets/sizes, then extract length-prefixed NALs from mdat.

**Files:**
- Modify: `container/mp4.py`
- Modify: `container/tests/test_mp4.py`

**Step 1: Write failing test**

```python
class TestNalExtraction:
    def test_length_prefixed_to_annex_b(self):
        """Convert length-prefixed NALs to Annex B format."""
        from container.mp4 import length_prefixed_to_annex_b

        # Two NALs: [0x00000005][65 01 02 03 04] [0x00000003][41 01 02]
        data = (
            struct.pack('>I', 5) + b'\x65\x01\x02\x03\x04'
            + struct.pack('>I', 3) + b'\x41\x01\x02'
        )
        annex_b = length_prefixed_to_annex_b(data, nal_length_size=4)
        # Expected: start_code + nal1 + start_code + nal2
        expected = b'\x00\x00\x00\x01\x65\x01\x02\x03\x04\x00\x00\x00\x01\x41\x01\x02'
        assert annex_b == expected
```

**Step 2: Run test to verify it fails**

Run: `pytest container/tests/test_mp4.py::TestNalExtraction -v`
Expected: FAIL

**Step 3: Implement NAL extraction**

```python
def length_prefixed_to_annex_b(data: bytes, nal_length_size: int = 4) -> bytes:
    """Convert length-prefixed NAL units to Annex B format.

    Args:
        data: Length-prefixed NAL unit data from MP4 sample.
        nal_length_size: Bytes per NAL length prefix (1, 2, or 4).

    Returns:
        Annex B formatted bytes with start codes.
    """
    START_CODE = b'\x00\x00\x00\x01'
    fmt = {1: '>B', 2: '>H', 4: '>I'}[nal_length_size]
    result = bytearray()
    offset = 0

    while offset + nal_length_size <= len(data):
        nal_len = struct.unpack(fmt, data[offset:offset + nal_length_size])[0]
        offset += nal_length_size
        if offset + nal_len > len(data):
            logger.warning(f"NAL length {nal_len} exceeds data at offset {offset}")
            break
        result.extend(START_CODE)
        result.extend(data[offset:offset + nal_len])
        offset += nal_len

    return bytes(result)
```

**Step 4: Run test to verify it passes**

Run: `pytest container/tests/test_mp4.py::TestNalExtraction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add container/mp4.py container/tests/test_mp4.py
git commit -m "feat(container): convert length-prefixed NALs to Annex B"
```

---

### Task 8: Full MP4-to-Annex-B extraction pipeline

Tie together box parsing, avcC, sample table, and NAL extraction into one `extract_h264_from_mp4()` function.

**Files:**
- Modify: `container/mp4.py`
- Modify: `container/tests/test_mp4.py`

**Step 1: Write failing test**

```python
class TestMp4Extraction:
    def test_extract_h264_from_mp4(self):
        """Full pipeline: MP4 bytes → Annex B bitstream."""
        from container.mp4 import extract_h264_from_mp4

        # Use a real tiny MP4 file for this test
        # Generate with: ffmpeg -f lavfi -i color=black:s=16x16:d=0.1:r=5 -c:v libx264 -profile:v baseline tiny.mp4
        import os
        mp4_path = os.path.join(os.path.dirname(__file__), '../../test_data/tiny_test.mp4')
        if not os.path.exists(mp4_path):
            pytest.skip("tiny_test.mp4 not generated yet")

        with open(mp4_path, 'rb') as f:
            mp4_data = f.read()

        annex_b = extract_h264_from_mp4(mp4_data)
        # Must start with SPS (NAL type 7) after start code
        assert annex_b[:4] == b'\x00\x00\x00\x01'
        nal_type = annex_b[4] & 0x1F
        assert nal_type == 7, f"First NAL should be SPS (7), got {nal_type}"
```

**Step 2: Generate test MP4**

Run:
```bash
ffmpeg -f lavfi -i "color=black:s=16x16:d=0.2:r=5" -c:v libx264 -profile:v baseline -pix_fmt yuv420p test_data/tiny_test.mp4
```

**Step 3: Implement full extraction**

Add `extract_h264_from_mp4()` to `container/mp4.py` that:
1. Parses all boxes to find moov/trak/stbl
2. Finds avcC in stsd for SPS/PPS
3. Parses stco/stsz/stsc for sample offsets and sizes
4. Reads samples from mdat
5. Prepends SPS/PPS as Annex B NALs
6. Converts all sample NALs to Annex B format
7. Returns complete Annex B bitstream bytes

**Step 4: Run test to verify it passes**

Run: `pytest container/tests/test_mp4.py::TestMp4Extraction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add container/mp4.py container/tests/test_mp4.py test_data/tiny_test.mp4
git commit -m "feat(container): full MP4-to-Annex-B extraction pipeline"
```

---

### Task 9: Wire MP4 support into decoder entry point

Add `decode_mp4_file()` and auto-detection in `decode_file()`.

**Files:**
- Modify: `decoder/decoder.py:436-448` (`decode_file`)
- Modify: `decoder/__init__.py` (export new function)

**Step 1: Write failing test**

```python
def test_decode_mp4_file(self):
    """Decoder should accept MP4 files directly."""
    from decoder import decode_h264_file
    frames = list(decode_h264_file("test_data/tiny_test.mp4"))
    assert len(frames) > 0
    assert frames[0].width == 16
    assert frames[0].height == 16
```

**Step 2: Implement auto-detection in decode_file**

In `decoder/decoder.py:436`, modify `decode_file()`:

```python
def decode_file(self, path: str):
    """Decode H.264 file and yield frames.

    Supports both raw Annex B (.264, .h264) and MP4 container files.
    """
    with open(path, "rb") as f:
        data = f.read()

    # Auto-detect MP4: starts with ftyp box or has ftyp within first 8 bytes
    if len(data) >= 8 and data[4:8] == b'ftyp':
        from container.mp4 import extract_h264_from_mp4
        data = extract_h264_from_mp4(data)

    yield from self.decode_bytes(data)
```

**Step 3: Run test**

Run: `pytest decoder/tests/test_decoder.py::test_decode_mp4_file -v`
Expected: PASS

**Step 4: Run full test suite**

Run: `pytest -x -q`
Expected: All tests pass (MP4 detection doesn't affect .264 files)

**Step 5: Commit**

```bash
git add decoder/decoder.py decoder/__init__.py
git commit -m "feat: auto-detect MP4 containers in decode_file()"
```

---

## Phase 3: High Profile 8x8 Transform Integration

### Task 10: Add CABAC 8x8 residual block category (block_cat=5)

The CABAC residual decoder needs context offsets for 8x8 luma blocks.

**Files:**
- Modify: `entropy/cabac_residual.py` (add block_cat=5 contexts)
- Create: `entropy/tests/test_cabac_8x8_residual.py`

**Step 1: Research context offsets**

Read H.264 Table 9-39 for block_cat=5 (8x8 luma):
- `ctxIdxBlockCatOffset` for `significant_coeff_flag`: 402
- `ctxIdxBlockCatOffset` for `last_significant_coeff_flag`: 417
- `ctxIdxBlockCatOffset` for `coeff_abs_level_minus1`: 426
- `maxNumCoeff`: 64
- Zigzag scan: 8x8 diagonal scan (H.264 Table 8-11)

**Step 2: Write failing test for 8x8 context offsets**

**Step 3: Add block_cat=5 to BLOCK_CAT_CTX_OFFSETS and implement 8x8 residual decoding**

**Step 4: Run test**

**Step 5: Commit**

```bash
git commit -m "feat(cabac): add block_cat=5 for 8x8 luma residual"
```

---

### Task 11: Add I_8x8 dispatch in CABAC intra reconstruction

`_reconstruct_intra_mb_cabac()` currently only handles I_4x4. Add I_8x8 path.

**Files:**
- Modify: `decoder/decoder.py:2548-2960` (`_reconstruct_intra_mb_cabac`)

**Step 1: Research what's needed**

When `mb_data['transform_size_8x8_flag'] == 1`:
- 4 intra 8x8 prediction modes (vs 16 for 4x4)
- 8x8 residual blocks (block_cat=5)
- Use `dequant_8x8()` + `idct_8x8()` per block
- Use 8x8 intra prediction modes from `intra/intra_8x8.py`

**Step 2: Write test for I_8x8 CABAC dispatch**

Generate a High Profile CABAC I-frame test stream:
```bash
ffmpeg -f lavfi -i testsrc2=duration=0.1:size=176x144:rate=10 \
  -c:v libx264 -profile:v high -x264-params "bframes=0:ref=1" \
  -pix_fmt yuv420p test_data/cabac_high_intra.264

ffmpeg -i test_data/cabac_high_intra.264 -f rawvideo -pix_fmt yuv420p test_data/cabac_high_intra_ref.yuv
```

**Step 3: Implement I_8x8 branch in `_reconstruct_intra_mb_cabac()`**

Check `mb_data.get('transform_size_8x8_flag')`. If set:
- Derive 4 intra 8x8 modes (from prev_intra_pred_mode + rem_intra_pred_mode in mb_data)
- For each 8x8 block: predict, dequant 8x8, IDCT 8x8, reconstruct
- Reuse logic from `reconstruct/macroblock.py:decode_and_reconstruct_i8x8_luma()`

**Step 4: Run pixel-perfect test**

**Step 5: Commit**

```bash
git commit -m "feat(cabac): I_8x8 intra reconstruction for High Profile"
```

---

### Task 12: Add 8x8 transform for inter residual in CABAC

High Profile inter MBs can use 8x8 transform for residual when `transform_size_8x8_flag=1`.

**Files:**
- Modify: `decoder/decoder.py` (`_reconstruct_inter_mb_cabac` residual path)

**Step 1: Write test**

Generate High Profile P/B-frame test stream and reference.

**Step 2: Implement 8x8 inter residual path**

In `_reconstruct_inter_mb_cabac()`, after motion compensation:
- If `transform_size_8x8_flag`: dequant + IDCT residual as 8x8 blocks
- Else: existing 4x4 path

**Step 3: Run pixel-perfect test**

**Step 4: Commit**

```bash
git commit -m "feat(cabac): 8x8 inter residual for High Profile"
```

---

### Task 13: Wire scaling lists into dequantization

Pass SPS/PPS scaling lists to `dequant_4x4()` and `dequant_8x8()`.

**Files:**
- Modify: `decoder/decoder.py` (pass scaling lists from SPS/PPS to dequant calls)
- Modify: `reconstruct/macroblock.py` (pass scaling lists in CAVLC path)

**Step 1: Research scaling list flow**

SPS has `scaling_lists_4x4` (6 lists) and `scaling_lists_8x8` (2+ lists).
PPS can override with its own lists.
Active list = PPS if present, else SPS, else flat-16 default.

**Step 2: Write test with custom scaling list stream**

**Step 3: Thread scaling lists through dequant calls**

**Step 4: Run test**

**Step 5: Commit**

```bash
git commit -m "feat: pass scaling lists to dequant for High Profile"
```

---

## Phase 4: Integration Testing with Real Video

### Task 14: Test with Main Profile CABAC video

**Step 1: Encode a Main Profile test video**

```bash
ffmpeg -f lavfi -i testsrc2=duration=2:size=352x288:rate=15 \
  -c:v libx264 -profile:v main \
  -x264-params "bframes=2:ref=2" \
  -pix_fmt yuv420p test_data/main_profile_352x288.264

ffmpeg -i test_data/main_profile_352x288.264 -f rawvideo -pix_fmt yuv420p test_data/main_profile_352x288_ref.yuv
```

**Step 2: Write pixel-perfect test**

**Step 3: Debug until passing**

**Step 4: Commit**

---

### Task 15: Test with High Profile video

**Step 1: Encode a High Profile test video**

```bash
ffmpeg -f lavfi -i testsrc2=duration=2:size=352x288:rate=15 \
  -c:v libx264 -profile:v high \
  -x264-params "bframes=2:ref=2" \
  -pix_fmt yuv420p test_data/high_profile_352x288.264

ffmpeg -i test_data/high_profile_352x288.264 -f rawvideo -pix_fmt yuv420p test_data/high_profile_352x288_ref.yuv
```

**Step 2: Write pixel-perfect test**

**Step 3: Debug until passing**

**Step 4: Commit**

---

### Task 16: Test with real MP4 file

**Step 1: Download a small YouTube-compatible MP4**

```bash
ffmpeg -f lavfi -i testsrc2=duration=3:size=640x480:rate=30 \
  -c:v libx264 -profile:v high -preset medium \
  -pix_fmt yuv420p test_data/real_video_640x480.mp4

ffmpeg -i test_data/real_video_640x480.mp4 -f rawvideo -pix_fmt yuv420p test_data/real_video_640x480_ref.yuv
```

**Step 2: Write end-to-end test**

```python
def test_decode_mp4_end_to_end(self):
    """Decode MP4 container → extract NALs → decode H.264 → pixel-perfect."""
    from decoder import decode_h264_file
    frames = list(decode_h264_file("test_data/real_video_640x480.mp4"))
    # Compare against reference
    ...
```

**Step 3: Debug until passing**

**Step 4: Commit**

```bash
git commit -m "test: end-to-end MP4 decode passes pixel-perfect"
```
