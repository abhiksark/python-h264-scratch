# h264/container/mp4.py
"""MP4 (ISO 14496-12) container parser for H.264 NAL unit extraction.

Parses MP4 box structure and extracts H.264 NAL units from video tracks.
Converts length-prefixed NALs (AVCC format) to Annex B format for the decoder.

ISO 14496-12 (ISOBMFF), ISO 14496-15 (AVCC)
"""
import struct
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Box parsing (ISO 14496-12)
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """ISO Base Media File Format box."""

    type: str
    size: int
    data: bytes
    offset: int = 0
    children: List['Box'] = field(default_factory=list)


CONTAINER_BOXES = {
    'moov', 'trak', 'mdia', 'minf', 'stbl', 'edts', 'dinf', 'udta',
}


def parse_box(data: bytes, offset: int) -> Optional[Box]:
    """Parse a single MP4 box at the given offset.

    Args:
        data: Raw file bytes.
        offset: Byte offset of the box header.

    Returns:
        Parsed Box, or None if data is too short.
    """
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
    elif size == 0:  # Box extends to end of data
        size = len(data) - offset

    box_data = data[offset + header_size:offset + size]
    box = Box(type=box_type, size=size, data=box_data, offset=offset)

    if box_type in CONTAINER_BOXES:
        box.children = parse_boxes(box_data)

    return box


def parse_boxes(data: bytes) -> List[Box]:
    """Parse all top-level boxes in data.

    Args:
        data: Raw bytes containing consecutive boxes.

    Returns:
        List of parsed Box objects.
    """
    boxes = []
    offset = 0
    while offset < len(data):
        box = parse_box(data, offset)
        if box is None or box.size < 8:
            break
        boxes.append(box)
        offset += box.size
    return boxes


def find_box(boxes: List[Box], box_type: str) -> Optional[Box]:
    """Find first box of given type in a list.

    Args:
        boxes: List of boxes to search.
        box_type: Four-character box type code.

    Returns:
        Matching Box, or None.
    """
    for box in boxes:
        if box.type == box_type:
            return box
    return None


def find_box_recursive(boxes: List[Box], box_type: str) -> Optional[Box]:
    """Recursively find first box of given type.

    Args:
        boxes: List of boxes to search (depth-first).
        box_type: Four-character box type code.

    Returns:
        Matching Box, or None.
    """
    for box in boxes:
        if box.type == box_type:
            return box
        result = find_box_recursive(box.children, box_type)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# AVCC parsing (ISO 14496-15 Section 5.2.4.1)
# ---------------------------------------------------------------------------

@dataclass
class AVCConfig:
    """AVC Decoder Configuration Record."""

    profile_idc: int
    level_idc: int
    nal_length_size: int  # 1, 2, or 4
    sps_list: List[bytes] = field(default_factory=list)
    pps_list: List[bytes] = field(default_factory=list)


def parse_avcc(data: bytes) -> AVCConfig:
    """Parse AVCDecoderConfigurationRecord from avcC box data.

    Args:
        data: Raw avcC box payload.

    Returns:
        Parsed AVCConfig with SPS/PPS NAL units.

    Raises:
        ValueError: If data is too short.
    """
    if len(data) < 7:
        raise ValueError(f"avcC too short: {len(data)} bytes")

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


# ---------------------------------------------------------------------------
# NAL unit conversion
# ---------------------------------------------------------------------------

_START_CODE = b'\x00\x00\x00\x01'


def length_prefixed_to_annex_b(data: bytes, nal_length_size: int = 4) -> bytes:
    """Convert length-prefixed NAL units to Annex B format.

    Args:
        data: Length-prefixed NAL unit data from MP4 sample.
        nal_length_size: Bytes per NAL length prefix (1, 2, or 4).

    Returns:
        Annex B formatted bytes with start codes.
    """
    fmt = {1: '>B', 2: '>H', 4: '>I'}[nal_length_size]
    result = bytearray()
    offset = 0

    while offset + nal_length_size <= len(data):
        nal_len = struct.unpack(fmt, data[offset:offset + nal_length_size])[0]
        offset += nal_length_size
        if offset + nal_len > len(data):
            logger.warning(
                f"NAL length {nal_len} exceeds data at offset {offset}"
            )
            break
        result.extend(_START_CODE)
        result.extend(data[offset:offset + nal_len])
        offset += nal_len

    return bytes(result)


# ---------------------------------------------------------------------------
# Sample table parsing (ISO 14496-12)
# ---------------------------------------------------------------------------

def parse_stsz(data: bytes) -> List[int]:
    """Parse stsz (sample size) box data.

    Args:
        data: Raw stsz box payload (after box header).

    Returns:
        List of sample sizes in bytes.
    """
    # version(1) + flags(3) + sample_size(4) + sample_count(4)
    sample_size = struct.unpack('>I', data[4:8])[0]
    sample_count = struct.unpack('>I', data[8:12])[0]

    if sample_size != 0:
        return [sample_size] * sample_count

    sizes = []
    for i in range(sample_count):
        off = 12 + i * 4
        sizes.append(struct.unpack('>I', data[off:off + 4])[0])
    return sizes


def parse_stco(data: bytes) -> List[int]:
    """Parse stco (chunk offset, 32-bit) box data.

    Args:
        data: Raw stco box payload.

    Returns:
        List of chunk byte offsets into the file.
    """
    entry_count = struct.unpack('>I', data[4:8])[0]
    offsets = []
    for i in range(entry_count):
        off = 8 + i * 4
        offsets.append(struct.unpack('>I', data[off:off + 4])[0])
    return offsets


def parse_co64(data: bytes) -> List[int]:
    """Parse co64 (chunk offset, 64-bit) box data.

    Args:
        data: Raw co64 box payload.

    Returns:
        List of chunk byte offsets into the file.
    """
    entry_count = struct.unpack('>I', data[4:8])[0]
    offsets = []
    for i in range(entry_count):
        off = 8 + i * 8
        offsets.append(struct.unpack('>Q', data[off:off + 8])[0])
    return offsets


def parse_stsc(data: bytes) -> List[Tuple[int, int, int]]:
    """Parse stsc (sample-to-chunk) box data.

    Args:
        data: Raw stsc box payload.

    Returns:
        List of (first_chunk, samples_per_chunk, sample_description_index).
    """
    entry_count = struct.unpack('>I', data[4:8])[0]
    entries = []
    for i in range(entry_count):
        off = 8 + i * 12
        first_chunk = struct.unpack('>I', data[off:off + 4])[0]
        spc = struct.unpack('>I', data[off + 4:off + 8])[0]
        sdi = struct.unpack('>I', data[off + 8:off + 12])[0]
        entries.append((first_chunk, spc, sdi))
    return entries


def _expand_stsc(
    stsc_entries: List[Tuple[int, int, int]],
    num_chunks: int,
) -> List[int]:
    """Expand stsc entries to per-chunk sample counts.

    Args:
        stsc_entries: Parsed stsc entries.
        num_chunks: Total number of chunks.

    Returns:
        List of samples_per_chunk for each chunk (0-indexed).
    """
    result = [0] * num_chunks
    for i, (first_chunk, spc, _) in enumerate(stsc_entries):
        # first_chunk is 1-indexed
        end = stsc_entries[i + 1][0] - 1 if i + 1 < len(stsc_entries) else num_chunks
        for c in range(first_chunk - 1, end):
            result[c] = spc
    return result


def _compute_sample_offsets(
    chunk_offsets: List[int],
    sample_sizes: List[int],
    stsc_entries: List[Tuple[int, int, int]],
) -> List[int]:
    """Compute absolute byte offset for each sample.

    Args:
        chunk_offsets: Per-chunk file offsets.
        sample_sizes: Per-sample sizes.
        stsc_entries: Sample-to-chunk mapping.

    Returns:
        List of absolute byte offsets, one per sample.
    """
    samples_per_chunk = _expand_stsc(stsc_entries, len(chunk_offsets))
    offsets = []
    sample_idx = 0

    for chunk_idx, chunk_offset in enumerate(chunk_offsets):
        spc = samples_per_chunk[chunk_idx]
        offset = chunk_offset
        for _ in range(spc):
            if sample_idx >= len(sample_sizes):
                break
            offsets.append(offset)
            offset += sample_sizes[sample_idx]
            sample_idx += 1

    return offsets


# ---------------------------------------------------------------------------
# stsd (sample description) parsing
# ---------------------------------------------------------------------------

def _find_avcc_in_stsd(stsd_data: bytes) -> Optional[bytes]:
    """Extract avcC payload from stsd box data.

    The stsd box contains sample entries. For H.264, the entry type is
    'avc1' or 'avc3', which contains an avcC box at a known offset.

    Args:
        stsd_data: Raw stsd box payload.

    Returns:
        avcC box payload bytes, or None if not found.
    """
    # version(1) + flags(3) + entry_count(4)
    if len(stsd_data) < 8:
        return None

    # Skip to first sample entry (offset 8)
    entry_offset = 8
    if entry_offset + 8 > len(stsd_data):
        return None

    entry_size = struct.unpack('>I', stsd_data[entry_offset:entry_offset + 4])[0]
    entry_type = stsd_data[entry_offset + 4:entry_offset + 8].decode(
        'ascii', errors='replace'
    )

    if entry_type not in ('avc1', 'avc3'):
        logger.debug(f"stsd entry type '{entry_type}' is not H.264")
        return None

    # avc1 sample entry: 8 (box header) + 6 (reserved) + 2 (data_ref_idx)
    # + 16 (pre_defined + reserved) + 2 (width) + 2 (height)
    # + 4 (horiz_res) + 4 (vert_res) + 4 (reserved) + 2 (frame_count)
    # + 32 (compressor_name) + 2 (depth) + 2 (pre_defined)
    # = 86 bytes before extension boxes
    ext_offset = entry_offset + 86

    # Parse extension boxes inside the sample entry
    entry_end = entry_offset + entry_size
    while ext_offset + 8 <= entry_end:
        ext_size = struct.unpack('>I', stsd_data[ext_offset:ext_offset + 4])[0]
        ext_type = stsd_data[ext_offset + 4:ext_offset + 8].decode(
            'ascii', errors='replace'
        )
        if ext_size < 8:
            break
        if ext_type == 'avcC':
            return stsd_data[ext_offset + 8:ext_offset + ext_size]
        ext_offset += ext_size

    return None


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def extract_h264_from_mp4(mp4_data: bytes) -> bytes:
    """Extract H.264 Annex B bitstream from MP4 container.

    Parses MP4 box structure, finds the H.264 video track, extracts SPS/PPS
    from avcC, and converts all samples to Annex B format.

    Args:
        mp4_data: Complete MP4 file bytes.

    Returns:
        Annex B formatted H.264 bitstream bytes.

    Raises:
        ValueError: If no H.264 track is found.
    """
    boxes = parse_boxes(mp4_data)
    moov = find_box(boxes, 'moov')
    if moov is None:
        raise ValueError("No moov box found in MP4 data")

    # Find the video track with H.264 content
    avcc_data = None
    stbl = None

    for trak in [c for c in moov.children if c.type == 'trak']:
        mdia = find_box(trak.children, 'mdia')
        if mdia is None:
            continue
        # Check handler type
        hdlr = find_box(mdia.children, 'hdlr')
        if hdlr is not None:
            # hdlr: version(1) + flags(3) + pre_defined(4) + handler_type(4)
            if len(hdlr.data) >= 12:
                handler = hdlr.data[8:12].decode('ascii', errors='replace')
                if handler != 'vide':
                    continue

        minf = find_box(mdia.children, 'minf')
        if minf is None:
            continue
        stbl_box = find_box(minf.children, 'stbl')
        if stbl_box is None:
            continue

        # Look for avcC inside stsd
        stsd = find_box(stbl_box.children, 'stsd')
        if stsd is not None:
            avcc_payload = _find_avcc_in_stsd(stsd.data)
            if avcc_payload is not None:
                avcc_data = avcc_payload
                stbl = stbl_box
                break

    if avcc_data is None or stbl is None:
        raise ValueError("No H.264 video track found in MP4")

    avc_config = parse_avcc(avcc_data)
    logger.info(
        f"H.264 track: profile={avc_config.profile_idc}, "
        f"level={avc_config.level_idc}, "
        f"nal_length_size={avc_config.nal_length_size}"
    )

    # Build Annex B output: SPS + PPS first
    result = bytearray()
    for sps in avc_config.sps_list:
        result.extend(_START_CODE)
        result.extend(sps)
    for pps in avc_config.pps_list:
        result.extend(_START_CODE)
        result.extend(pps)

    # Parse sample table
    stsz_box = find_box(stbl.children, 'stsz')
    stco_box = find_box(stbl.children, 'stco')
    co64_box = find_box(stbl.children, 'co64')
    stsc_box = find_box(stbl.children, 'stsc')

    if stsz_box is None:
        raise ValueError("Missing stsz box")
    if stco_box is None and co64_box is None:
        raise ValueError("Missing stco/co64 box")
    if stsc_box is None:
        raise ValueError("Missing stsc box")

    sample_sizes = parse_stsz(stsz_box.data)
    chunk_offsets = (
        parse_stco(stco_box.data) if stco_box is not None
        else parse_co64(co64_box.data)
    )
    stsc_entries = parse_stsc(stsc_box.data)

    sample_offsets = _compute_sample_offsets(
        chunk_offsets, sample_sizes, stsc_entries
    )

    logger.info(
        f"Extracting {len(sample_sizes)} samples from "
        f"{len(chunk_offsets)} chunks"
    )

    # Convert each sample's length-prefixed NALs to Annex B
    for i, (offset, size) in enumerate(zip(sample_offsets, sample_sizes)):
        sample_data = mp4_data[offset:offset + size]
        annex_b_sample = length_prefixed_to_annex_b(
            sample_data, avc_config.nal_length_size
        )
        result.extend(annex_b_sample)

    return bytes(result)
