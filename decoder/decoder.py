# h264/decoder/decoder.py
"""H.264 Baseline profile decoder.

Main decoder orchestration that ties together:
- NAL unit parsing (bitstream/)
- SPS/PPS parsing (parameters/)
- Slice header parsing (slice/)
- Macroblock reconstruction (reconstruct/)
- Color conversion (color/)

H.264 Spec Reference:
- Section 7: Syntax and semantics
- Section 8: Decoding process

This decoder supports:
- Baseline profile (no CABAC, no B-frames)
- I-slices only (no P/B inter prediction yet)
- 4:2:0 chroma format
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bitstream import (
    NALUnit,
    NALUnitType,
    BitReader,
    extract_nal_units,
    iter_nal_units,
)
from parameters import SPS, PPS, parse_sps, parse_pps
from slice import SliceHeader, SliceType, parse_slice_header
from reconstruct import decode_macroblock, MacroblockData
from color import ycbcr_to_rgb, ColorMatrix

logger = logging.getLogger(__name__)


@dataclass
class DecodedFrame:
    """Represents a decoded video frame.

    Attributes:
        frame_num: H.264 frame number
        poc: Picture order count
        luma: Y plane (H x W), uint8
        cb: Cb plane (H/2 x W/2), uint8
        cr: Cr plane (H/2 x W/2), uint8
        width: Frame width in pixels
        height: Frame height in pixels
    """

    frame_num: int
    poc: int
    luma: np.ndarray
    cb: np.ndarray
    cr: np.ndarray
    width: int
    height: int

    @property
    def shape(self) -> tuple:
        """Return (height, width) of luma plane."""
        return (self.height, self.width)

    def to_rgb(self, color_matrix: ColorMatrix = ColorMatrix.BT601) -> np.ndarray:
        """Convert YUV frame to RGB.

        Args:
            color_matrix: Color standard (BT601 or BT709)

        Returns:
            RGB array (H x W x 3), uint8
        """
        return ycbcr_to_rgb(self.luma, self.cb, self.cr, color_matrix=color_matrix)


@dataclass
class DecoderState:
    """Internal decoder state.

    Tracks SPS/PPS parameter sets and current decoding context.
    """

    sps_dict: dict = field(default_factory=dict)  # id -> SPS
    pps_dict: dict = field(default_factory=dict)  # id -> PPS
    current_sps: Optional[SPS] = None
    current_pps: Optional[PPS] = None

    # Frame buffers for reconstruction
    frame_luma: Optional[np.ndarray] = None
    frame_cb: Optional[np.ndarray] = None
    frame_cr: Optional[np.ndarray] = None

    # Non-zero coefficient counts for CAVLC context
    nz_counts: Optional[np.ndarray] = None

    def get_sps(self, sps_id: int) -> SPS:
        """Get SPS by ID."""
        if sps_id not in self.sps_dict:
            raise ValueError(f"SPS {sps_id} not found")
        return self.sps_dict[sps_id]

    def get_pps(self, pps_id: int) -> PPS:
        """Get PPS by ID."""
        if pps_id not in self.pps_dict:
            raise ValueError(f"PPS {pps_id} not found")
        return self.pps_dict[pps_id]

    def allocate_frame_buffers(self, sps: SPS) -> None:
        """Allocate frame buffers based on SPS dimensions."""
        height = sps.frame_height_in_mbs * 16
        width = sps.pic_width_in_mbs * 16

        self.frame_luma = np.zeros((height, width), dtype=np.uint8)
        self.frame_cb = np.zeros((height // 2, width // 2), dtype=np.uint8)
        self.frame_cr = np.zeros((height // 2, width // 2), dtype=np.uint8)

        # nz_counts: 24 per MB (16 luma + 4 Cb + 4 Cr)
        # Stored per MB for context calculation
        mb_count = sps.frame_height_in_mbs * sps.pic_width_in_mbs
        self.nz_counts = np.zeros((mb_count, 24), dtype=np.int32)

        logger.debug(
            f"Allocated frame buffers: {width}x{height} "
            f"({sps.pic_width_in_mbs}x{sps.frame_height_in_mbs} MBs)"
        )


class H264Decoder:
    """H.264 Baseline profile decoder.

    Decodes I-slices from Annex B bitstreams.

    Example:
        decoder = H264Decoder()
        for frame in decoder.decode_file("video.264"):
            rgb = frame.to_rgb()
            # Display or save rgb array
    """

    def __init__(self):
        """Initialize decoder."""
        self.state = DecoderState()

    def decode_file(self, path: str):
        """Decode H.264 file and yield frames.

        Args:
            path: Path to Annex B bitstream file (.264, .h264)

        Yields:
            DecodedFrame objects
        """
        with open(path, "rb") as f:
            data = f.read()

        yield from self.decode_bytes(data)

    def decode_bytes(self, data: bytes):
        """Decode H.264 bitstream bytes and yield frames.

        Args:
            data: Annex B bitstream bytes

        Yields:
            DecodedFrame objects
        """
        for nal in iter_nal_units(data):
            frame = self._process_nal(nal)
            if frame is not None:
                yield frame

    def _process_nal(self, nal: NALUnit) -> Optional[DecodedFrame]:
        """Process a single NAL unit.

        Args:
            nal: NAL unit to process

        Returns:
            DecodedFrame if this NAL completed a frame, None otherwise
        """
        logger.debug(f"Processing NAL: type={nal.nal_unit_type}, size={len(nal.rbsp)}")

        if nal.nal_unit_type == NALUnitType.SPS:
            self._process_sps(nal)
            return None

        elif nal.nal_unit_type == NALUnitType.PPS:
            self._process_pps(nal)
            return None

        elif nal.nal_unit_type in (NALUnitType.SLICE_IDR, NALUnitType.SLICE_NON_IDR):
            return self._decode_slice(nal)

        else:
            logger.debug(f"Skipping NAL type {nal.nal_unit_type}")
            return None

    def _process_sps(self, nal: NALUnit) -> None:
        """Parse and store SPS."""
        sps = parse_sps(nal.rbsp)  # parse_sps expects raw bytes
        self.state.sps_dict[sps.seq_parameter_set_id] = sps
        logger.info(
            f"Parsed SPS {sps.seq_parameter_set_id}: "
            f"{sps.pic_width_in_mbs * 16}x{sps.frame_height_in_mbs * 16}"
        )

    def _process_pps(self, nal: NALUnit) -> None:
        """Parse and store PPS."""
        pps = parse_pps(nal.rbsp)  # parse_pps expects raw bytes
        self.state.pps_dict[pps.pic_parameter_set_id] = pps
        logger.info(
            f"Parsed PPS {pps.pic_parameter_set_id}: "
            f"entropy_mode={pps.entropy_coding_mode_flag}"
        )

    def _decode_slice(self, nal: NALUnit) -> Optional[DecodedFrame]:
        """Decode a slice NAL unit.

        Args:
            nal: Slice NAL unit

        Returns:
            DecodedFrame if slice completes a frame
        """
        # Need SPS/PPS to parse slice header
        if not self.state.sps_dict or not self.state.pps_dict:
            logger.warning("No SPS/PPS available, skipping slice")
            return None

        # Peek at first_mb_in_slice to determine PPS
        # For now, assume PPS 0
        pps = self.state.get_pps(0)
        sps = self.state.get_sps(pps.seq_parameter_set_id)

        # Parse slice header (expects raw bytes)
        slice_header = parse_slice_header(
            nal.rbsp, sps, pps, nal.nal_unit_type, nal.nal_ref_idc
        )
        logger.debug(
            f"Slice header: type={slice_header.slice_type}, "
            f"frame_num={slice_header.frame_num}"
        )

        # Check if this is an I-slice
        if not slice_header.is_i_slice:
            logger.warning(f"Skipping non-I slice: {slice_header.slice_type_name}")
            return None

        # Update current parameter sets
        self.state.current_sps = sps
        self.state.current_pps = pps

        # Allocate frame buffers if needed
        if self.state.frame_luma is None:
            self.state.allocate_frame_buffers(sps)

        # Calculate slice QP
        slice_qp = 26 + pps.pic_init_qp_minus26 + slice_header.slice_qp_delta
        logger.debug(f"Slice QP: {slice_qp}")

        # Create BitReader for slice data (positioned after slice header)
        reader = BitReader(nal.rbsp)
        reader.position = slice_header.header_bit_size

        # Decode macroblocks
        self._decode_slice_data(reader, slice_header, sps, pps, slice_qp)

        # Return completed frame
        # For simplicity, assume each slice is a complete frame
        frame = DecodedFrame(
            frame_num=slice_header.frame_num,
            poc=getattr(slice_header, "pic_order_cnt_lsb", 0),
            luma=self.state.frame_luma.copy(),
            cb=self.state.frame_cb.copy(),
            cr=self.state.frame_cr.copy(),
            width=sps.pic_width_in_mbs * 16,
            height=sps.frame_height_in_mbs * 16,
        )

        return frame

    def _decode_slice_data(
        self,
        reader: BitReader,
        slice_header: SliceHeader,
        sps: SPS,
        pps: PPS,
        slice_qp: int,
    ) -> None:
        """Decode slice data (macroblocks).

        Processes macroblocks in raster scan order.

        Args:
            reader: BitReader positioned at slice data
            slice_header: Parsed slice header
            sps: Active SPS
            pps: Active PPS
            slice_qp: Slice-level QP
        """
        mb_width = sps.pic_width_in_mbs
        mb_height = sps.frame_height_in_mbs
        total_mbs = mb_width * mb_height

        logger.debug(f"Decoding {total_mbs} macroblocks ({mb_width}x{mb_height})")

        current_qp = slice_qp

        for mb_idx in range(slice_header.first_mb_in_slice, total_mbs):
            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Get neighbor information for prediction
            neighbors = self._get_mb_neighbors(mb_x, mb_y, mb_width)

            try:
                # Decode single macroblock
                mb_data = decode_macroblock(
                    reader=reader,
                    sps=sps,
                    pps=pps,
                    slice_qp=current_qp,
                    mb_x=mb_x,
                    mb_y=mb_y,
                    frame_luma=self.state.frame_luma,
                    frame_cb=self.state.frame_cb,
                    frame_cr=self.state.frame_cr,
                    is_i_slice=True,
                )

                # Store non-zero counts for CAVLC context
                self.state.nz_counts[mb_idx] = mb_data.nz_counts

                logger.debug(f"Decoded MB ({mb_x}, {mb_y}): type={mb_data.mb_type}")

            except Exception as e:
                logger.error(f"Error decoding MB ({mb_x}, {mb_y}): {e}")
                raise

    def _get_mb_neighbors(
        self, mb_x: int, mb_y: int, mb_width: int
    ) -> dict:
        """Get neighbor macroblock data for prediction.

        Args:
            mb_x: Macroblock X position
            mb_y: Macroblock Y position
            mb_width: Frame width in macroblocks

        Returns:
            Dictionary with neighbor information
        """
        neighbors = {
            "top_available": mb_y > 0,
            "left_available": mb_x > 0,
            "top_left_available": mb_x > 0 and mb_y > 0,
            "top_right_available": mb_y > 0 and mb_x < mb_width - 1,
        }

        # Extract neighbor pixels from frame buffers
        if neighbors["top_available"] and self.state.frame_luma is not None:
            top_row = mb_y * 16 - 1
            left_col = mb_x * 16
            neighbors["top_luma"] = self.state.frame_luma[
                top_row, left_col : left_col + 16
            ]
        else:
            neighbors["top_luma"] = None

        if neighbors["left_available"] and self.state.frame_luma is not None:
            top_row = mb_y * 16
            left_col = mb_x * 16 - 1
            neighbors["left_luma"] = self.state.frame_luma[
                top_row : top_row + 16, left_col
            ]
        else:
            neighbors["left_luma"] = None

        return neighbors


def decode_h264_file(path: str) -> list:
    """Convenience function to decode an H.264 file.

    Args:
        path: Path to Annex B bitstream file

    Returns:
        List of DecodedFrame objects
    """
    decoder = H264Decoder()
    return list(decoder.decode_file(path))


def decode_h264_bytes(data: bytes) -> list:
    """Convenience function to decode H.264 bytes.

    Args:
        data: Annex B bitstream bytes

    Returns:
        List of DecodedFrame objects
    """
    decoder = H264Decoder()
    return list(decoder.decode_bytes(data))
