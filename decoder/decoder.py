# h264/decoder/decoder.py
"""H.264 Baseline profile decoder.

Main decoder orchestration that ties together:
- NAL unit parsing (bitstream/)
- SPS/PPS parsing (parameters/)
- Slice header parsing (slice/)
- Macroblock reconstruction (reconstruct/)
- Inter prediction (inter/)
- Color conversion (color/)

H.264 Spec Reference:
- Section 7: Syntax and semantics
- Section 8: Decoding process

This decoder supports:
- Baseline profile (no CABAC, no B-frames)
- I-slices and P-slices
- 4:2:0 chroma format
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

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
from inter.reference import ReferenceFrame, ReferenceFrameBuffer
from inter.mv_prediction import (
    MVCache,
    predict_mv_16x16,
    predict_mv_16x8,
    predict_mv_8x16,
    predict_mv_8x8,
)
from inter.p_macroblock import parse_p_mb_type, parse_sub_mb_type, PMacroblockInfo
from inter.p_reconstruct import (
    reconstruct_p_skip,
    reconstruct_p_16x16,
    reconstruct_p_16x8,
    reconstruct_p_8x16,
    reconstruct_p_8x8,
)

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

    # Reference frame buffer for inter prediction
    ref_buffer: Optional[ReferenceFrameBuffer] = None

    # MV cache for current frame
    mv_cache: Optional[MVCache] = None

    # Current macroblock QP (updated by mb_qp_delta)
    current_mb_qp: int = 26

    # Per-MB info for deblocking filter
    mb_types: Optional[np.ndarray] = None  # MB type per MB
    mb_coeffs: Optional[np.ndarray] = None  # Has non-zero coeffs per 4x4 block

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

        # Initialize reference frame buffer
        max_refs = sps.max_num_ref_frames if hasattr(sps, 'max_num_ref_frames') else 4
        self.ref_buffer = ReferenceFrameBuffer(max_frames=max(1, max_refs))

        # Per-MB info for deblocking
        self.mb_types = np.zeros(mb_count, dtype=np.int32)
        # 16 luma + 8 chroma 4x4 blocks per MB
        self.mb_coeffs = np.zeros((mb_count, 24), dtype=bool)

        logger.debug(
            f"Allocated frame buffers: {width}x{height} "
            f"({sps.pic_width_in_mbs}x{sps.frame_height_in_mbs} MBs)"
        )

    def init_mv_cache(self, sps: SPS) -> None:
        """Initialize MV cache for a new frame."""
        self.mv_cache = MVCache(
            width_in_mbs=sps.pic_width_in_mbs,
            height_in_mbs=sps.frame_height_in_mbs
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

    def __init__(self, deblocking_enabled: bool = True):
        """Initialize decoder.

        Args:
            deblocking_enabled: Whether to apply deblocking filter (default True)
        """
        self.state = DecoderState()
        self.deblocking_enabled = deblocking_enabled

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

        # Check slice type
        is_i_slice = slice_header.is_i_slice
        is_p_slice = slice_header.is_p_slice if hasattr(slice_header, 'is_p_slice') else False

        if not is_i_slice and not is_p_slice:
            logger.warning(f"Skipping unsupported slice: {slice_header.slice_type_name}")
            return None

        # P-slices need reference frames
        if is_p_slice:
            if self.state.ref_buffer is None or len(self.state.ref_buffer) == 0:
                logger.warning("P-slice but no reference frames available")
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

        # Initialize MV cache for P-slices
        if is_p_slice:
            self.state.init_mv_cache(sps)

        # Create BitReader for slice data (positioned after slice header)
        reader = BitReader(nal.rbsp)
        reader.position = slice_header.header_bit_size

        # Decode macroblocks
        if is_i_slice:
            self._decode_slice_data(reader, slice_header, sps, pps, slice_qp)
        else:
            self._decode_p_slice_data(reader, slice_header, sps, pps, slice_qp)

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

        # Add to reference buffer if this is a reference frame
        if nal.nal_ref_idc > 0:
            ref_frame = ReferenceFrame(
                luma=self.state.frame_luma.copy(),
                cb=self.state.frame_cb.copy(),
                cr=self.state.frame_cr.copy(),
                frame_num=slice_header.frame_num,
            )
            self.state.ref_buffer.add_frame(ref_frame)
            logger.debug(f"Added frame {slice_header.frame_num} to reference buffer")

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

    def _decode_p_slice_data(
        self,
        reader: BitReader,
        slice_header: SliceHeader,
        sps: SPS,
        pps: PPS,
        slice_qp: int,
    ) -> None:
        """Decode P-slice data (macroblocks with inter prediction).

        P-slices can contain:
        - P_Skip macroblocks (mb_skip_run)
        - P_L0_16x16, P_L0_L0_16x8, P_L0_L0_8x16, P_8x8 macroblocks
        - I-macroblocks (intra in P-slice)

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

        logger.debug(f"Decoding P-slice: {total_mbs} macroblocks")

        current_qp = slice_qp
        mb_idx = slice_header.first_mb_in_slice

        while mb_idx < total_mbs:
            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Read mb_skip_run for P-slices
            mb_skip_run = reader.read_ue()
            logger.debug(f"mb_skip_run={mb_skip_run} at MB {mb_idx}")

            # Process skipped macroblocks
            for _ in range(mb_skip_run):
                if mb_idx >= total_mbs:
                    break

                mb_x = mb_idx % mb_width
                mb_y = mb_idx // mb_width

                # Reconstruct P_Skip macroblock
                luma, cb, cr = reconstruct_p_skip(
                    ref_buffer=self.state.ref_buffer,
                    mv_cache=self.state.mv_cache,
                    mb_x=mb_x,
                    mb_y=mb_y,
                )

                # Write to frame buffers
                ly, lx = mb_y * 16, mb_x * 16
                cy, cx = mb_y * 8, mb_x * 8
                self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
                self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
                self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

                logger.debug(f"P_Skip MB ({mb_x}, {mb_y})")
                mb_idx += 1

            # After skip run, decode regular macroblock if not at end
            if mb_idx >= total_mbs:
                break

            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Read mb_type
            mb_type_code = reader.read_ue()
            mb_type = parse_p_mb_type(mb_type_code)

            logger.debug(f"MB ({mb_x}, {mb_y}): type={mb_type.name}")

            if mb_type.is_intra:
                # I-macroblock in P-slice - use I-MB decoder
                # Need to parse using I-MB syntax
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
                    is_i_slice=False,
                )
                # Clear MV cache for intra MB
                self.state.mv_cache.set_mv_16x16(mb_x, mb_y, 0, 0)

            else:
                # P-macroblock
                self._decode_p_macroblock(
                    reader, mb_type, mb_x, mb_y, current_qp, sps, pps
                )

            mb_idx += 1

    def _decode_p_macroblock(
        self,
        reader: BitReader,
        mb_type,
        mb_x: int,
        mb_y: int,
        qp: int,
        sps: SPS,
        pps: PPS,
    ) -> None:
        """Decode a single P-macroblock.

        Args:
            reader: BitReader
            mb_type: Parsed PMBType
            mb_x, mb_y: Macroblock position
            qp: Current QP
            sps, pps: Parameter sets
        """
        # For P_L0_16x16 (simplest case)
        if mb_type.name == "P_L0_16x16":
            # Read ref_idx if more than one reference
            ref_idx = 0
            if self.state.ref_buffer is not None and len(self.state.ref_buffer) > 1:
                ref_idx = reader.read_te(len(self.state.ref_buffer) - 1)

            # Read MVD
            mvd_x = reader.read_se()
            mvd_y = reader.read_se()

            # Get predicted MV
            mvp_x, mvp_y = predict_mv_16x16(self.state.mv_cache, mb_x, mb_y)

            # Final MV = prediction + difference
            mvx = mvp_x + mvd_x
            mvy = mvp_y + mvd_y

            # Update MV cache
            self.state.mv_cache.set_mv_16x16(mb_x, mb_y, mvx, mvy)

            # TODO: Parse and decode residual if present
            # For now, no residual
            luma, cb, cr = reconstruct_p_16x16(
                ref_buffer=self.state.ref_buffer,
                ref_idx=ref_idx,
                mvx=mvx,
                mvy=mvy,
                residual_luma=None,
                residual_cb=None,
                residual_cr=None,
                mb_x=mb_x,
                mb_y=mb_y,
            )

            # Write to frame buffers
            ly, lx = mb_y * 16, mb_x * 16
            cy, cx = mb_y * 8, mb_x * 8
            self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
            self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
            self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

            logger.debug(f"P_16x16 MB ({mb_x}, {mb_y}): MV=({mvx}, {mvy})")

        elif mb_type.name == "P_L0_L0_16x8":
            # Two 16x8 partitions (top and bottom)
            ref_idx = [0, 0]
            mvx = [0, 0]
            mvy = [0, 0]

            # Read ref_idx for each partition if multiple references
            num_refs = len(self.state.ref_buffer) if self.state.ref_buffer else 1
            if num_refs > 1:
                ref_idx[0] = reader.read_te(num_refs - 1)
                ref_idx[1] = reader.read_te(num_refs - 1)

            # Read MVD for each partition
            for part in range(2):
                mvd_x = reader.read_se()
                mvd_y = reader.read_se()

                # Get predicted MV for this partition
                mvp_x, mvp_y = predict_mv_16x8(self.state.mv_cache, mb_x, mb_y, part)

                mvx[part] = mvp_x + mvd_x
                mvy[part] = mvp_y + mvd_y

            # TODO: Parse residual

            luma, cb, cr = reconstruct_p_16x8(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                ref_idx=ref_idx,
                mvx=mvx,
                mvy=mvy,
                residual_luma=None,
                residual_cb=None,
                residual_cr=None,
                mb_x=mb_x,
                mb_y=mb_y,
            )

            ly, lx = mb_y * 16, mb_x * 16
            cy, cx = mb_y * 8, mb_x * 8
            self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
            self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
            self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

            logger.debug(f"P_16x8 MB ({mb_x}, {mb_y}): MVs={list(zip(mvx, mvy))}")

        elif mb_type.name == "P_L0_L0_8x16":
            # Two 8x16 partitions (left and right)
            ref_idx = [0, 0]
            mvx = [0, 0]
            mvy = [0, 0]

            num_refs = len(self.state.ref_buffer) if self.state.ref_buffer else 1
            if num_refs > 1:
                ref_idx[0] = reader.read_te(num_refs - 1)
                ref_idx[1] = reader.read_te(num_refs - 1)

            for part in range(2):
                mvd_x = reader.read_se()
                mvd_y = reader.read_se()

                mvp_x, mvp_y = predict_mv_8x16(self.state.mv_cache, mb_x, mb_y, part)

                mvx[part] = mvp_x + mvd_x
                mvy[part] = mvp_y + mvd_y

            luma, cb, cr = reconstruct_p_8x16(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                ref_idx=ref_idx,
                mvx=mvx,
                mvy=mvy,
                residual_luma=None,
                residual_cb=None,
                residual_cr=None,
                mb_x=mb_x,
                mb_y=mb_y,
            )

            ly, lx = mb_y * 16, mb_x * 16
            cy, cx = mb_y * 8, mb_x * 8
            self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
            self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
            self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

            logger.debug(f"P_8x16 MB ({mb_x}, {mb_y}): MVs={list(zip(mvx, mvy))}")

        elif mb_type.name in ("P_8x8", "P_8x8ref0"):
            # Four 8x8 sub-macroblocks
            sub_mb_types = []
            ref_idx = [0, 0, 0, 0]
            mvx = [0, 0, 0, 0]
            mvy = [0, 0, 0, 0]

            # Read sub_mb_type for each 8x8 block
            for i in range(4):
                sub_type_code = reader.read_ue()
                sub_mb_types.append(sub_type_code)

            # Read ref_idx for each sub-MB (unless P_8x8ref0)
            num_refs = len(self.state.ref_buffer) if self.state.ref_buffer else 1
            if mb_type.name == "P_8x8" and num_refs > 1:
                for i in range(4):
                    ref_idx[i] = reader.read_te(num_refs - 1)

            # Read MVD for each sub-MB (assuming 8x8 sub-type for simplicity)
            for sub_idx in range(4):
                mvd_x = reader.read_se()
                mvd_y = reader.read_se()

                mvp_x, mvp_y = predict_mv_8x8(self.state.mv_cache, mb_x, mb_y, sub_idx)

                mvx[sub_idx] = mvp_x + mvd_x
                mvy[sub_idx] = mvp_y + mvd_y

            luma, cb, cr = reconstruct_p_8x8(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                ref_idx=ref_idx,
                mvx=mvx,
                mvy=mvy,
                sub_mb_types=sub_mb_types,
                residual_luma=None,
                residual_cb=None,
                residual_cr=None,
                mb_x=mb_x,
                mb_y=mb_y,
            )

            ly, lx = mb_y * 16, mb_x * 16
            cy, cx = mb_y * 8, mb_x * 8
            self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
            self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
            self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

            logger.debug(f"P_8x8 MB ({mb_x}, {mb_y}): MVs={list(zip(mvx, mvy))}")

        else:
            # Unknown type - fall back to skip
            logger.warning(f"Unsupported P-MB type: {mb_type.name}, treating as skip")
            luma, cb, cr = reconstruct_p_skip(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                mb_x=mb_x,
                mb_y=mb_y,
            )
            ly, lx = mb_y * 16, mb_x * 16
            cy, cx = mb_y * 8, mb_x * 8
            self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
            self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
            self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

    def _parse_p_mb_residual(
        self,
        reader: 'BitReader',
        cbp: int,
        qp: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse residual data for a P-macroblock.

        Args:
            reader: BitReader positioned at residual data
            cbp: Coded block pattern
            qp: Quantization parameter

        Returns:
            Tuple of (luma, cb, cr) residual arrays or None if no residual
        """
        from inter.p_macroblock import parse_p_residual
        return parse_p_residual(reader, cbp, qp)

    def _apply_p_residual(
        self,
        prediction: np.ndarray,
        residual: Optional[np.ndarray],
    ) -> np.ndarray:
        """Apply residual to prediction to get reconstructed block.

        Args:
            prediction: Predicted block from motion compensation
            residual: Residual block or None

        Returns:
            Reconstructed block (clipped to [0, 255])
        """
        if residual is None:
            return prediction
        return np.clip(
            prediction.astype(np.int32) + residual, 0, 255
        ).astype(np.uint8)

    def _decode_p_luma_residual(
        self,
        reader: 'BitReader',
        cbp_luma: int,
        qp: int,
    ) -> Optional[np.ndarray]:
        """Decode luma residual blocks for P-macroblock.

        Args:
            reader: BitReader
            cbp_luma: Lower 4 bits of CBP indicating which 8x8 blocks have residual
            qp: Quantization parameter

        Returns:
            16x16 luma residual or None if CBP indicates no luma residual
        """
        if cbp_luma == 0:
            return None

        from entropy import decode_residual_block
        from dequant import dequant_4x4
        from transform import idct_4x4

        residual = np.zeros((16, 16), dtype=np.int32)

        # Process each 8x8 block
        for i8x8 in range(4):
            if not (cbp_luma & (1 << i8x8)):
                continue

            # 8x8 block position
            bx8 = (i8x8 % 2) * 8
            by8 = (i8x8 // 2) * 8

            # Process four 4x4 blocks within this 8x8
            for i4x4 in range(4):
                bx4 = bx8 + (i4x4 % 2) * 4
                by4 = by8 + (i4x4 // 2) * 4

                # Decode coefficients
                coeffs = decode_residual_block(reader, 16, 'luma_ac')
                if coeffs is not None:
                    dequant = dequant_4x4(coeffs, qp)
                    block = idct_4x4(dequant)
                    residual[by4:by4+4, bx4:bx4+4] = block

        return residual

    def _decode_p_chroma_residual(
        self,
        reader: 'BitReader',
        cbp_chroma: int,
        qp: int,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Decode chroma residual blocks for P-macroblock.

        Args:
            reader: BitReader
            cbp_chroma: Upper bits of CBP for chroma (0=none, 1=DC only, 2=DC+AC)
            qp: Quantization parameter

        Returns:
            Tuple of (cb, cr) 8x8 residual arrays or None
        """
        if cbp_chroma == 0:
            return None, None

        from entropy import decode_residual_block
        from dequant import dequant_4x4, dequant_dc_2x2
        from transform import idct_4x4, hadamard_2x2_inverse

        cb_residual = np.zeros((8, 8), dtype=np.int32)
        cr_residual = np.zeros((8, 8), dtype=np.int32)

        qp_c = min(qp, 51)  # Chroma QP

        for chroma_idx, residual in enumerate([cb_residual, cr_residual]):
            # Decode DC coefficients (2x2)
            dc_coeffs = decode_residual_block(reader, 4, 'chroma_dc')
            if dc_coeffs is not None:
                dc_dequant = dequant_dc_2x2(dc_coeffs, qp_c)
                dc_transform = hadamard_2x2_inverse(dc_dequant)

            # Decode AC coefficients if cbp_chroma == 2
            if cbp_chroma >= 2:
                for i4x4 in range(4):
                    bx = (i4x4 % 2) * 4
                    by = (i4x4 // 2) * 4

                    ac_coeffs = decode_residual_block(reader, 15, 'chroma_ac')
                    block = np.zeros((4, 4), dtype=np.int32)
                    if dc_coeffs is not None:
                        block[0, 0] = dc_transform[i4x4 // 2, i4x4 % 2]
                    if ac_coeffs is not None:
                        # Fill in AC coefficients
                        pass  # Simplified for now
                    dequant = dequant_4x4(block, qp_c)
                    residual[by:by+4, bx:bx+4] = idct_4x4(dequant)

        return cb_residual, cr_residual

    def _deblock_frame(self) -> None:
        """Apply deblocking filter to the entire frame."""
        if not self.deblocking_enabled:
            return

        from deblock.deblock import deblock_frame

        self.state.frame_luma, self.state.frame_cb, self.state.frame_cr = deblock_frame(
            luma=self.state.frame_luma,
            cb=self.state.frame_cb,
            cr=self.state.frame_cr,
            mb_info=None,  # TODO: Pass per-MB info
            qp=self.state.current_mb_qp,
        )

    def _get_deblock_params(self, slice_header: 'SliceHeader') -> dict:
        """Get deblocking filter parameters from slice header.

        Args:
            slice_header: Parsed slice header

        Returns:
            Dictionary with deblocking parameters
        """
        return {
            'disable_deblocking_filter_idc': getattr(
                slice_header, 'disable_deblocking_filter_idc', 0
            ),
            'slice_alpha_c0_offset_div2': getattr(
                slice_header, 'slice_alpha_c0_offset_div2', 0
            ),
            'slice_beta_offset_div2': getattr(
                slice_header, 'slice_beta_offset_div2', 0
            ),
        }

    def _should_deblock_slice(self, slice_header: 'SliceHeader') -> bool:
        """Check if deblocking should be applied for this slice.

        Args:
            slice_header: Parsed slice header

        Returns:
            True if deblocking should be applied
        """
        if not self.deblocking_enabled:
            return False

        disable_idc = getattr(slice_header, 'disable_deblocking_filter_idc', 0)
        # 0 = deblock all edges
        # 1 = disable all deblocking
        # 2 = disable at slice boundaries
        return disable_idc != 1

    def _is_slice_boundary(
        self,
        mb_x: int,
        mb_y: int,
        neighbor_mb_x: int,
        neighbor_mb_y: int,
    ) -> bool:
        """Check if an edge is at a slice boundary.

        Args:
            mb_x, mb_y: Current macroblock position
            neighbor_mb_x, neighbor_mb_y: Neighbor macroblock position

        Returns:
            True if the edge between these MBs is a slice boundary
        """
        # For single-slice frames, no boundaries
        # Full implementation would track which slice each MB belongs to
        return False


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
