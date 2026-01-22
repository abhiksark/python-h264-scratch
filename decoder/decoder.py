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
from decoder.poc import POCCalculator
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
from inter.b_macroblock import parse_b_mb_type, get_b_skip_info
from inter.b_reconstruct import (
    reconstruct_b_skip,
    reconstruct_b_l0_16x16,
    reconstruct_b_l1_16x16,
    reconstruct_b_bi_16x16,
    reconstruct_b_direct_16x16,
)
from entropy.cabac_arith import CABACDecoder
from entropy.cabac_context import init_context_models
from entropy.cabac_macroblock import (
    decode_macroblock_layer_cabac,
    decode_end_of_slice_flag_cabac,
)
from entropy.cabac_residual import decode_residual_block_cabac

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
    mb_cbps: Optional[np.ndarray] = None  # CBP per MB for CABAC context

    # Slice tracking for multiple slice support
    mb_slice_ids: Optional[np.ndarray] = None  # Which slice each MB belongs to
    current_slice_id: int = 0  # Current slice being decoded

    # B-frame reference lists
    l0_list: list = field(default_factory=list)  # L0 reference list (past frames)
    l1_list: list = field(default_factory=list)  # L1 reference list (future frames)

    # POC tracking
    prev_poc_msb: int = 0
    prev_poc_lsb: int = 0

    # CABAC context models (None = not initialized)
    cabac_contexts: Optional[list] = None

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
        # CBP per MB for CABAC neighbor context
        self.mb_cbps = np.zeros(mb_count, dtype=np.int32)

        # Slice tracking
        self.mb_slice_ids = np.zeros(mb_count, dtype=np.int32)
        self.current_slice_id = 0

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
        self.poc_calculator = POCCalculator()

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
        is_b_slice = slice_header.is_b_slice if hasattr(slice_header, 'is_b_slice') else False

        if not is_i_slice and not is_p_slice and not is_b_slice:
            logger.warning(f"Skipping unsupported slice: {slice_header.slice_type_name}")
            return None

        # P/B-slices need reference frames
        if is_p_slice or is_b_slice:
            if self.state.ref_buffer is None or len(self.state.ref_buffer) == 0:
                logger.warning(f"{'P' if is_p_slice else 'B'}-slice but no reference frames available")
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

        # Initialize MV cache for P/B-slices
        if is_p_slice or is_b_slice:
            self.state.init_mv_cache(sps)

        # Create BitReader for slice data (positioned after slice header)
        reader = BitReader(nal.rbsp)
        reader.position = slice_header.header_bit_size

        # Check entropy coding mode
        use_cabac = getattr(pps, 'entropy_coding_mode_flag', 0) == 1

        # For CABAC, align to byte boundary (skip cabac_alignment_one_bit + zeros)
        if use_cabac:
            reader.byte_align()

        # Decode macroblocks
        if use_cabac:
            # CABAC entropy coding
            self._decode_slice_cabac(
                reader, slice_header, sps, pps, slice_qp,
                is_i_slice, is_p_slice, is_b_slice
            )
        elif is_i_slice:
            self._decode_slice_data(reader, slice_header, sps, pps, slice_qp)
        elif is_p_slice:
            self._decode_p_slice_data(reader, slice_header, sps, pps, slice_qp)
        else:  # B-slice
            self._decode_b_slice_data(reader, slice_header, sps, pps, slice_qp)

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

    def _decode_i_pcm_macroblock(
        self,
        reader: 'BitReader',
        mb_x: int,
        mb_y: int,
    ) -> None:
        """Decode I_PCM macroblock.

        I_PCM contains raw, uncompressed samples.

        Args:
            reader: BitReader positioned after mb_type
            mb_x, mb_y: Macroblock position
        """
        from intra.i_pcm import (
            align_to_byte,
            parse_i_pcm_luma,
            parse_i_pcm_chroma,
            get_i_pcm_nz_counts,
        )

        # Align to byte boundary
        align_to_byte(reader)

        # Parse raw samples
        bit_depth = 8  # Baseline profile uses 8-bit
        luma = parse_i_pcm_luma(reader, bit_depth)
        cb, cr = parse_i_pcm_chroma(reader, bit_depth)

        # Write to frame buffers
        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma.astype(np.uint8)
        self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb.astype(np.uint8)
        self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr.astype(np.uint8)

        # Set nz_counts (all 16 for I_PCM)
        mb_idx = mb_y * self.state.current_sps.pic_width_in_mbs + mb_x
        self.state.nz_counts[mb_idx] = get_i_pcm_nz_counts()

        logger.debug(f"I_PCM MB ({mb_x}, {mb_y})")

    def _use_weighted_prediction(
        self,
        slice_header: 'SliceHeader',
    ) -> bool:
        """Check if weighted prediction should be used for this slice.

        Args:
            slice_header: Parsed slice header

        Returns:
            True if weighted prediction is enabled
        """
        if self.state.current_pps is None:
            return False

        pps = self.state.current_pps

        # P-slices: check weighted_pred_flag
        if slice_header.is_p_slice:
            return getattr(pps, 'weighted_pred_flag', False)

        # B-slices: check weighted_bipred_idc
        if slice_header.is_b_slice:
            weighted_bipred_idc = getattr(pps, 'weighted_bipred_idc', 0)
            return weighted_bipred_idc > 0

        return False

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
        if self.state.mb_slice_ids is None:
            return False

        sps = self.state.current_sps
        if sps is None:
            return False

        mb_width = sps.pic_width_in_mbs
        current_idx = mb_y * mb_width + mb_x
        neighbor_idx = neighbor_mb_y * mb_width + neighbor_mb_x

        return self.state.mb_slice_ids[current_idx] != self.state.mb_slice_ids[neighbor_idx]

    def _continue_slice(
        self,
        reader: 'BitReader',
        slice_header: 'SliceHeader',
        sps: 'SPS',
        pps: 'PPS',
        slice_qp: int,
    ) -> None:
        """Continue decoding a slice from first_mb_in_slice.

        Used when decoding multiple slices in a single frame.

        Args:
            reader: BitReader positioned at slice data
            slice_header: Parsed slice header
            sps: Active SPS
            pps: Active PPS
            slice_qp: Slice-level QP
        """
        # Increment slice ID for this new slice
        self.state.current_slice_id += 1

        # Decode from first_mb_in_slice
        if slice_header.is_i_slice:
            self._decode_slice_data(reader, slice_header, sps, pps, slice_qp)
        elif slice_header.is_p_slice:
            self._decode_p_slice_data(reader, slice_header, sps, pps, slice_qp)

    def _get_slice_qp(self, slice_header: 'SliceHeader') -> int:
        """Get the QP for a slice.

        Args:
            slice_header: Parsed slice header

        Returns:
            Slice QP value
        """
        pps = self.state.current_pps
        if pps is None:
            return 26 + slice_header.slice_qp_delta

        return 26 + pps.pic_init_qp_minus26 + slice_header.slice_qp_delta

    def _get_slice_group(self, mb_x: int, mb_y: int) -> int:
        """Get the slice group for a macroblock.

        Args:
            mb_x, mb_y: Macroblock position

        Returns:
            Slice group ID
        """
        pps = self.state.current_pps
        if pps is None or getattr(pps, 'num_slice_groups_minus1', 0) == 0:
            return 0

        sps = self.state.current_sps
        if sps is None:
            return 0

        map_type = getattr(pps, 'slice_group_map_type', 0)

        if map_type == 0:
            return self._calc_interleaved_map(mb_x, mb_y)
        elif map_type == 1:
            return self._calc_dispersed_map(mb_x, mb_y)
        else:
            # Types 2-6 not implemented
            return 0

    def _calc_interleaved_map(self, mb_x: int, mb_y: int) -> int:
        """Calculate slice group for interleaved pattern (type 0).

        In interleaved mode, MBs are assigned to groups in runs.

        Args:
            mb_x, mb_y: Macroblock position

        Returns:
            Slice group ID
        """
        pps = self.state.current_pps
        sps = self.state.current_sps

        if pps is None or sps is None:
            return 0

        num_groups = getattr(pps, 'num_slice_groups_minus1', 0) + 1
        run_length = getattr(pps, 'run_length_minus1', [0])

        if num_groups == 1:
            return 0

        mb_width = sps.pic_width_in_mbs
        mb_idx = mb_y * mb_width + mb_x

        # Simple interleaving based on run_length
        group = 0
        remaining = mb_idx

        while remaining >= 0 and group < num_groups:
            run = run_length[group] + 1 if group < len(run_length) else 1
            if remaining < run:
                return group
            remaining -= run
            group = (group + 1) % num_groups

        return 0

    def _calc_dispersed_map(self, mb_x: int, mb_y: int) -> int:
        """Calculate slice group for dispersed pattern (type 1).

        In dispersed mode, MBs are assigned in a checkered pattern.

        Args:
            mb_x, mb_y: Macroblock position

        Returns:
            Slice group ID
        """
        pps = self.state.current_pps
        sps = self.state.current_sps

        if pps is None or sps is None:
            return 0

        num_groups = getattr(pps, 'num_slice_groups_minus1', 0) + 1

        if num_groups == 1:
            return 0

        mb_width = sps.pic_width_in_mbs
        mb_idx = mb_y * mb_width + mb_x

        # Dispersed: cycle through groups based on MB position
        return (mb_idx + mb_y * (num_groups // 2)) % num_groups

    # -------------------------------------------------------------------------
    # B-frame support methods
    # -------------------------------------------------------------------------

    def _decode_b_slice_data(
        self,
        reader: 'BitReader',
        slice_header: 'SliceHeader',
        sps: 'SPS',
        pps: 'PPS',
        slice_qp: int,
    ) -> None:
        """Decode B-slice data (macroblocks with bi-directional prediction).

        B-slices can contain:
        - B_Skip macroblocks (mb_skip_run)
        - B_Direct_16x16, B_L0_16x16, B_L1_16x16, B_Bi_16x16
        - Partition modes (16x8, 8x16, 8x8)
        - I-macroblocks (intra in B-slice)

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

        logger.debug(f"Decoding B-slice: {total_mbs} macroblocks")

        current_qp = slice_qp
        mb_idx = slice_header.first_mb_in_slice

        # Get direct mode flag
        use_spatial = getattr(slice_header, 'direct_spatial_mv_pred_flag', True)
        current_poc = getattr(slice_header, 'pic_order_cnt_lsb', 0)

        # Build L0/L1 reference lists before decoding B-slice
        self._build_b_slice_ref_lists(slice_header)

        while mb_idx < total_mbs:
            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Read mb_skip_run for B-slices
            mb_skip_run = reader.read_ue()
            logger.debug(f"mb_skip_run={mb_skip_run} at MB {mb_idx}")

            # Process B_Skip macroblocks
            for _ in range(mb_skip_run):
                if mb_idx >= total_mbs:
                    break

                mb_x = mb_idx % mb_width
                mb_y = mb_idx // mb_width

                self._process_b_skip_run(
                    mb_x, mb_y, use_spatial, current_poc
                )
                mb_idx += 1

            # After skip run, decode regular macroblock
            if mb_idx >= total_mbs:
                break

            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Read mb_type
            mb_type_code = reader.read_ue()
            mb_info = parse_b_mb_type(mb_type_code)

            logger.debug(f"MB ({mb_x}, {mb_y}): type={mb_info.name}")

            if mb_info.is_intra:
                # I-macroblock in B-slice
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
            else:
                # B-macroblock
                self._decode_b_macroblock(
                    reader, mb_info, mb_x, mb_y, current_qp,
                    sps, pps, use_spatial, current_poc
                )

            mb_idx += 1

    def _decode_slice_cabac(
        self,
        reader: 'BitReader',
        slice_header: 'SliceHeader',
        sps: 'SPS',
        pps: 'PPS',
        slice_qp: int,
        is_i_slice: bool,
        is_p_slice: bool,
        is_b_slice: bool,
    ) -> None:
        """Decode slice data using CABAC entropy coding.

        Handles I, P, and B slices with CABAC-encoded syntax elements.

        Args:
            reader: BitReader positioned at slice data
            slice_header: Parsed slice header
            sps: Active SPS
            pps: Active PPS
            slice_qp: Slice-level QP
            is_i_slice, is_p_slice, is_b_slice: Slice type flags
        """
        mb_width = sps.pic_width_in_mbs
        mb_height = sps.frame_height_in_mbs
        total_mbs = mb_width * mb_height

        # Determine slice type for context initialization
        if is_i_slice:
            slice_type = 2  # I
        elif is_p_slice:
            slice_type = 0  # P
        else:
            slice_type = 1  # B

        # Initialize CABAC contexts
        contexts = init_context_models(slice_type, slice_qp)

        # Create CABAC decoder
        cabac = CABACDecoder(reader)

        logger.debug(f"Decoding CABAC slice: {total_mbs} MBs, type={slice_type}")

        # Build reference lists for P/B slices
        if is_b_slice:
            self._build_b_slice_ref_lists(slice_header)

        current_qp = slice_qp
        mb_idx = slice_header.first_mb_in_slice

        while mb_idx < total_mbs:
            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Build neighbor info for context derivation
            mb_info = self._build_cabac_mb_info(mb_x, mb_y, mb_width, sps, pps)

            # Decode macroblock syntax using CABAC
            mb_data = decode_macroblock_layer_cabac(
                cabac, contexts, slice_type, mb_info
            )

            logger.debug(
                f"CABAC MB ({mb_x}, {mb_y}): skip={mb_data.get('mb_skip_flag', 0)}, "
                f"type={mb_data.get('mb_type', 0)}, cbp={mb_data.get('cbp', 0):#x}, "
                f"left_cbp={mb_info.get('left_cbp', -1)}, top_cbp={mb_info.get('top_cbp', -1)}"
            )

            # Process the decoded macroblock
            self._process_cabac_macroblock(
                cabac, contexts, mb_data, mb_x, mb_y,
                current_qp, sps, pps, slice_type
            )

            # Update QP
            current_qp = (current_qp + mb_data.get('mb_qp_delta', 0) + 52) % 52

            # Check for end of slice
            if decode_end_of_slice_flag_cabac(cabac, contexts) == 1:
                break

            mb_idx += 1

    def _build_cabac_mb_info(
        self,
        mb_x: int,
        mb_y: int,
        mb_width: int,
        sps: 'SPS',
        pps: 'PPS',
    ) -> dict:
        """Build neighbor information for CABAC context derivation.

        Args:
            mb_x, mb_y: Macroblock position
            mb_width: Frame width in macroblocks
            sps, pps: Parameter sets

        Returns:
            Dict with neighbor availability and types
        """
        mb_idx = mb_y * mb_width + mb_x

        # Get neighbor MB types and CBPs
        left_mb_type = None
        top_mb_type = None
        left_cbp = -1  # -1 = unavailable
        top_cbp = -1

        if mb_x > 0 and self.state.mb_types is not None:
            left_idx = mb_idx - 1
            if left_idx >= 0:
                left_mb_type = self.state.mb_types[left_idx]
                if self.state.mb_cbps is not None:
                    left_cbp = int(self.state.mb_cbps[left_idx])

        if mb_y > 0 and self.state.mb_types is not None:
            top_idx = mb_idx - mb_width
            if top_idx >= 0:
                top_mb_type = self.state.mb_types[top_idx]
                if self.state.mb_cbps is not None:
                    top_cbp = int(self.state.mb_cbps[top_idx])

        return {
            'mb_x': mb_x,
            'mb_y': mb_y,
            'mb_idx': mb_idx,
            'left_available': mb_x > 0,
            'top_available': mb_y > 0,
            'left_mb_type': left_mb_type,
            'top_mb_type': top_mb_type,
            'left_cbp': left_cbp,
            'top_cbp': top_cbp,
            'transform_8x8_mode_flag': getattr(pps, 'transform_8x8_mode_flag', 0),
        }

    def _process_cabac_macroblock(
        self,
        cabac: 'CABACDecoder',
        contexts: list,
        mb_data: dict,
        mb_x: int,
        mb_y: int,
        qp: int,
        sps: 'SPS',
        pps: 'PPS',
        slice_type: int,
    ) -> None:
        """Process a CABAC-decoded macroblock and reconstruct pixels.

        Args:
            cabac: CABAC decoder for residual
            contexts: Context models
            mb_data: Decoded MB syntax from CABAC
            mb_x, mb_y: Macroblock position
            qp: Current QP
            sps, pps: Parameter sets
            slice_type: Slice type (0=P, 1=B, 2=I)
        """
        mb_width = sps.pic_width_in_mbs
        mb_idx = mb_y * mb_width + mb_x

        # Store MB type and CBP for neighbor reference
        if self.state.mb_types is not None:
            self.state.mb_types[mb_idx] = mb_data.get('mb_type', 0)
        if self.state.mb_cbps is not None:
            self.state.mb_cbps[mb_idx] = mb_data.get('cbp', 0)

        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        # Handle skip macroblock
        if mb_data.get('mb_skip_flag', 0) == 1:
            self._reconstruct_skip_mb_cabac(mb_x, mb_y, slice_type)
            return

        mb_type = mb_data.get('mb_type', 0)

        # Handle I-macroblock in any slice
        if self._is_intra_mb_cabac(mb_type, slice_type):
            self._reconstruct_intra_mb_cabac(
                cabac, contexts, mb_data, mb_x, mb_y, qp, sps, pps, slice_type
            )
        else:
            # Inter macroblock (P or B prediction)
            self._reconstruct_inter_mb_cabac(
                cabac, contexts, mb_data, mb_x, mb_y, qp, sps, pps, slice_type
            )

    def _is_intra_mb_cabac(self, mb_type: int, slice_type: int) -> bool:
        """Check if macroblock type is intra.

        Args:
            mb_type: Macroblock type code
            slice_type: Slice type

        Returns:
            True if intra macroblock
        """
        if slice_type == 2:  # I-slice
            return True
        elif slice_type == 0:  # P-slice
            # In P-slice, mb_type >= 5 means intra
            return mb_type >= 5
        else:  # B-slice
            # In B-slice, mb_type >= 23 means intra
            return mb_type >= 23

    def _reconstruct_skip_mb_cabac(
        self,
        mb_x: int,
        mb_y: int,
        slice_type: int,
    ) -> None:
        """Reconstruct a skipped macroblock.

        Args:
            mb_x, mb_y: Macroblock position
            slice_type: Slice type (0=P, 1=B)
        """
        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        if slice_type == 0:  # P-skip
            luma, cb, cr = reconstruct_p_skip(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                mb_x=mb_x,
                mb_y=mb_y,
            )
        else:  # B-skip
            use_spatial = True
            current_poc = 0
            luma, cb, cr = reconstruct_b_skip(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                mb_x=mb_x,
                mb_y=mb_y,
                use_spatial=use_spatial,
                current_poc=current_poc,
            )

        # Copy to frame buffers
        self.state.frame_luma[ly:ly+16, lx:lx+16] = luma
        self.state.frame_cb[cy:cy+8, cx:cx+8] = cb
        self.state.frame_cr[cy:cy+8, cx:cx+8] = cr

    def _reconstruct_intra_mb_cabac(
        self,
        cabac: 'CABACDecoder',
        contexts: list,
        mb_data: dict,
        mb_x: int,
        mb_y: int,
        qp: int,
        sps: 'SPS',
        pps: 'PPS',
        slice_type: int,
    ) -> None:
        """Reconstruct an intra macroblock decoded with CABAC.

        Args:
            cabac: CABAC decoder
            contexts: Context models
            mb_data: Decoded MB syntax
            mb_x, mb_y: Macroblock position
            qp: Current QP
            sps, pps: Parameter sets
            slice_type: Slice type
        """
        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        # Get MB type relative to I-slice
        mb_type = mb_data.get('mb_type', 0)
        if slice_type == 0:  # P-slice
            mb_type = mb_type - 5
        elif slice_type == 1:  # B-slice
            mb_type = mb_type - 23

        # Determine prediction type
        if mb_type == 0:
            # I_4x4
            self._reconstruct_i4x4_cabac(
                cabac, contexts, mb_data, mb_x, mb_y, qp, sps
            )
        elif 1 <= mb_type <= 24:
            # I_16x16
            self._reconstruct_i16x16_cabac(
                cabac, contexts, mb_data, mb_x, mb_y, qp, sps, mb_type
            )
        else:
            # I_PCM or other - fill with mid-gray for now
            self.state.frame_luma[ly:ly+16, lx:lx+16] = 128
            self.state.frame_cb[cy:cy+8, cx:cx+8] = 128
            self.state.frame_cr[cy:cy+8, cx:cx+8] = 128

    def _reconstruct_i16x16_cabac(
        self,
        cabac: 'CABACDecoder',
        contexts: list,
        mb_data: dict,
        mb_x: int,
        mb_y: int,
        qp: int,
        sps: 'SPS',
        mb_type: int,
    ) -> None:
        """Reconstruct I_16x16 macroblock with CABAC residual.

        Args:
            cabac: CABAC decoder
            contexts: Context models
            mb_data: Decoded MB syntax
            mb_x, mb_y: Macroblock position
            qp: Current QP
            sps: SPS
            mb_type: I_16x16 sub-type (1-24)
        """
        from intra import predict_intra_16x16
        from transform import idct_4x4, inverse_hadamard_4x4
        from dequant import dequant_4x4

        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        # Decode I_16x16 mode from mb_type
        # mb_type 1-24 maps to mode, cbp_luma, cbp_chroma
        mode = (mb_type - 1) % 4
        cbp_chroma = ((mb_type - 1) // 4) % 3
        cbp_luma = 15 if (mb_type - 1) >= 12 else 0

        # Get prediction with neighbor availability
        top_available = mb_y > 0
        left_available = mb_x > 0
        top = self.state.frame_luma[ly-1, lx:lx+16] if top_available else None
        left = self.state.frame_luma[ly:ly+16, lx-1] if left_available else None
        top_left = self.state.frame_luma[ly-1, lx-1] if (left_available and top_available) else None

        pred = predict_intra_16x16(
            mode, top, left, top_left,
            top_available=top_available,
            left_available=left_available
        )

        # Decode luma DC coefficients (16 values via Hadamard)
        # Must check coded_block_flag first per H.264 spec 9.3.3.1.3
        import numpy as np
        from entropy.cabac_syntax import decode_coded_block_flag
        dc_cbf = decode_coded_block_flag(cabac, contexts, 0, 0)  # cat=0 for Luma DC
        if dc_cbf == 1:
            dc_coeffs = decode_residual_block_cabac(cabac, contexts, 16, 0)  # block_cat=0
        else:
            dc_coeffs = np.zeros(16, dtype=np.int32)

        # Dequant and inverse Hadamard for DC
        dc_block = np.array(dc_coeffs[:16]).reshape(4, 4)
        dc_block = inverse_hadamard_4x4(dc_block)

        # Scale DC coefficients
        qp_per = qp // 6
        qp_rem = qp % 6
        dc_scale = 1 << max(0, qp_per - 2)
        dc_block = dc_block * dc_scale

        # Decode AC coefficients for each 4x4 block
        luma = pred.copy().astype(np.int32)

        for blk_idx in range(16):
            by = (blk_idx // 4) * 4
            bx = (blk_idx % 4) * 4

            # Get DC from Hadamard result
            dc_val = dc_block[blk_idx // 4, blk_idx % 4]

            coeffs = np.zeros(16, dtype=np.int32)
            coeffs[0] = dc_val

            if cbp_luma & (1 << (blk_idx // 4)):
                # Check coded_block_flag before decoding AC
                cbf = decode_coded_block_flag(cabac, contexts, 1, 0)  # cat=1 for I16x16 AC
                if cbf == 1:
                    # Decode AC coefficients
                    ac_coeffs = decode_residual_block_cabac(cabac, contexts, 15, 1)
                    coeffs[1:] = ac_coeffs[:15]

            # Dequant and IDCT
            block = coeffs.reshape(4, 4)
            block = dequant_4x4(block, qp, is_intra=True)
            residual = idct_4x4(block)

            luma[by:by+4, bx:bx+4] += residual

        # Clip and store
        self.state.frame_luma[ly:ly+16, lx:lx+16] = np.clip(luma, 0, 255).astype(np.uint8)

        # Decode chroma if needed
        if cbp_chroma > 0:
            self._decode_chroma_cabac(cabac, contexts, mb_x, mb_y, qp, cbp_chroma)
        else:
            self.state.frame_cb[cy:cy+8, cx:cx+8] = 128
            self.state.frame_cr[cy:cy+8, cx:cx+8] = 128

    def _reconstruct_i4x4_cabac(
        self,
        cabac: 'CABACDecoder',
        contexts: list,
        mb_data: dict,
        mb_x: int,
        mb_y: int,
        qp: int,
        sps: 'SPS',
    ) -> None:
        """Reconstruct I_4x4 macroblock with CABAC.

        Decodes residual coefficients for luma and chroma blocks.
        """
        from entropy.cabac_residual import decode_residual_block_cabac
        from entropy.cabac_syntax import decode_coded_block_flag
        from intra import predict_intra_4x4
        from transform import idct_4x4
        from dequant import dequant_4x4
        import numpy as np

        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        # Get CBP from decoded mb_data
        cbp = mb_data.get('cbp', 0)
        cbp_luma = cbp & 0x0F
        cbp_chroma = (cbp >> 4) & 0x03

        # Get intra prediction modes
        pred_modes = mb_data.get('mb_pred', {}).get('intra_pred_modes', [])

        # Map 4x4 block index to 8x8 quadrant for CBP check
        # Blocks 0-3 -> quadrant 0, blocks 4-7 -> quadrant 1, etc.
        block_to_8x8 = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]

        # Decode luma 4x4 blocks
        luma = np.zeros((16, 16), dtype=np.int32)

        for blk_idx in range(16):
            # 4x4 block position within MB
            by = ((blk_idx // 4) % 2) * 8 + ((blk_idx % 4) // 2) * 4
            bx = ((blk_idx // 4) // 2) * 8 + ((blk_idx % 4) % 2) * 4

            # Get prediction mode for this block (-1 means use predicted)
            mode = pred_modes[blk_idx] if blk_idx < len(pred_modes) else 0
            if mode == -1:
                mode = 2  # Default to DC if no predicted mode available

            # Absolute position in frame
            abs_y = ly + by
            abs_x = lx + bx

            # Check neighbor availability based on absolute position
            top_available = abs_y > 0
            left_available = abs_x > 0

            # Get neighbor samples
            if top_available:
                top = self.state.frame_luma[abs_y - 1, abs_x:abs_x + 4]
            else:
                top = None

            if left_available:
                left = self.state.frame_luma[abs_y:abs_y + 4, abs_x - 1]
            else:
                left = None

            # Get top-left if both neighbors available
            top_left = None
            if top_available and left_available:
                top_left = int(self.state.frame_luma[abs_y - 1, abs_x - 1])

            # Predict block with availability flags
            pred = predict_intra_4x4(
                mode, top, left, top_left,
                top_available=top_available,
                left_available=left_available
            )

            # Check if this 8x8 quadrant has coefficients
            quad = block_to_8x8[blk_idx]
            if cbp_luma & (1 << quad):
                # Decode coded_block_flag for this 4x4 block
                cbf = decode_coded_block_flag(cabac, contexts, 2, 0)  # cat=2 for I_4x4 luma

                if cbf == 1:
                    # Decode residual coefficients
                    coeffs = decode_residual_block_cabac(cabac, contexts, 16, 2)
                    block = coeffs.reshape(4, 4)
                    block = dequant_4x4(block, qp, is_intra=True)
                    residual = idct_4x4(block)
                    luma[by:by + 4, bx:bx + 4] = pred + residual
                else:
                    luma[by:by + 4, bx:bx + 4] = pred
            else:
                luma[by:by + 4, bx:bx + 4] = pred

            # Store to frame buffer for neighbor prediction
            self.state.frame_luma[abs_y:abs_y + 4, abs_x:abs_x + 4] = np.clip(
                luma[by:by + 4, bx:bx + 4], 0, 255
            ).astype(np.uint8)

        # Decode chroma if needed
        if cbp_chroma > 0:
            # Decode chroma DC and AC
            self._decode_chroma_cabac(cabac, contexts, mb_x, mb_y, qp, cbp_chroma)
        else:
            # Fill with mid-gray
            self.state.frame_cb[cy:cy + 8, cx:cx + 8] = 128
            self.state.frame_cr[cy:cy + 8, cx:cx + 8] = 128

    def _decode_chroma_cabac(
        self,
        cabac: 'CABACDecoder',
        contexts: list,
        mb_x: int,
        mb_y: int,
        qp: int,
        cbp_chroma: int,
    ) -> None:
        """Decode chroma coefficients using CABAC.

        Args:
            cabac: CABAC decoder
            contexts: Context models
            mb_x, mb_y: Macroblock position
            qp: Current QP
            cbp_chroma: Chroma CBP (0=none, 1=DC only, 2=DC+AC)
        """
        from entropy.cabac_residual import decode_residual_block_cabac
        from entropy.cabac_syntax import decode_coded_block_flag
        from transform import inverse_hadamard_2x2, idct_4x4
        from dequant import dequant_4x4
        import numpy as np

        cy, cx = mb_y * 8, mb_x * 8

        # Chroma QP adjustment
        qp_c = min(51, qp)

        for plane in range(2):  # Cb, Cr
            frame = self.state.frame_cb if plane == 0 else self.state.frame_cr

            # Decode DC coefficients (4 values via 2x2 Hadamard)
            # Must check coded_block_flag first per H.264 spec 9.3.3.1.3
            dc_cbf = decode_coded_block_flag(cabac, contexts, 3, 0)  # cat=3 for chroma DC
            if dc_cbf == 1:
                dc_coeffs = decode_residual_block_cabac(cabac, contexts, 4, 3)  # block_cat=3 for chroma DC
                dc_block = np.array(dc_coeffs[:4]).reshape(2, 2)
                dc_block = inverse_hadamard_2x2(dc_block)
            else:
                dc_block = np.zeros((2, 2), dtype=np.int32)

            # Initialize chroma plane
            chroma = np.full((8, 8), 128, dtype=np.int32)

            # Decode AC coefficients for each 4x4 block if cbp_chroma == 2
            for blk_idx in range(4):
                by = (blk_idx // 2) * 4
                bx = (blk_idx % 2) * 4

                # Get DC value
                dc_val = dc_block[blk_idx // 2, blk_idx % 2]

                coeffs = np.zeros(16, dtype=np.int32)
                coeffs[0] = dc_val

                if cbp_chroma == 2:
                    # Check coded_block_flag for AC
                    cbf = decode_coded_block_flag(cabac, contexts, 4, 0)  # cat=4 for chroma AC
                    if cbf == 1:
                        ac_coeffs = decode_residual_block_cabac(cabac, contexts, 15, 4)
                        coeffs[1:] = ac_coeffs[:15]

                # Dequant and IDCT
                block = coeffs.reshape(4, 4)
                block = dequant_4x4(block, qp_c, is_intra=True)
                residual = idct_4x4(block)

                chroma[by:by + 4, bx:bx + 4] = 128 + residual

            # Clip and store
            frame[cy:cy + 8, cx:cx + 8] = np.clip(chroma, 0, 255).astype(np.uint8)

    def _reconstruct_inter_mb_cabac(
        self,
        cabac: 'CABACDecoder',
        contexts: list,
        mb_data: dict,
        mb_x: int,
        mb_y: int,
        qp: int,
        sps: 'SPS',
        pps: 'PPS',
        slice_type: int,
    ) -> None:
        """Reconstruct an inter macroblock decoded with CABAC.

        For now, uses skip reconstruction as placeholder.
        """
        # Use skip reconstruction as fallback
        self._reconstruct_skip_mb_cabac(mb_x, mb_y, slice_type)

    def _build_b_slice_ref_lists(
        self,
        slice_header: 'SliceHeader',
    ) -> None:
        """Build L0 and L1 reference lists for B-slice.

        L0 contains past frames, L1 contains future frames,
        both ordered by POC distance.

        Args:
            slice_header: Parsed slice header
        """
        if self.state.ref_buffer is None:
            self.state.l0_list = []
            self.state.l1_list = []
            return

        current_poc = getattr(slice_header, 'pic_order_cnt_lsb', 0)
        self.state.ref_buffer.build_ref_lists(current_poc)
        self.state.l0_list = self.state.ref_buffer.get_l0_list()
        self.state.l1_list = self.state.ref_buffer.get_l1_list()

        logger.debug(
            f"Built ref lists: L0={len(self.state.l0_list)}, "
            f"L1={len(self.state.l1_list)}"
        )

    def _decode_b_macroblock(
        self,
        reader: 'BitReader',
        mb_info,
        mb_x: int,
        mb_y: int,
        qp: int,
        sps: 'SPS',
        pps: 'PPS',
        use_spatial: bool,
        current_poc: int,
    ) -> None:
        """Decode a single B-macroblock.

        Args:
            reader: BitReader
            mb_info: Parsed BMacroblockInfo
            mb_x, mb_y: Macroblock position
            qp: Current QP
            sps, pps: Parameter sets
            use_spatial: True for spatial direct mode
            current_poc: Picture order count
        """
        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        if mb_info.is_direct:
            # B_Direct_16x16
            luma, cb, cr = reconstruct_b_direct_16x16(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                residual_luma=None,
                residual_cb=None,
                residual_cr=None,
                mb_x=mb_x,
                mb_y=mb_y,
                use_spatial=use_spatial,
                current_poc=current_poc,
            )
        elif mb_info.pred_mode == "L0":
            # B_L0_16x16
            ref_idx = 0
            if len(self.state.l0_list) > 1:
                ref_idx = reader.read_te(len(self.state.l0_list) - 1)

            mvd_x = reader.read_se()
            mvd_y = reader.read_se()

            mvp_x, mvp_y = predict_mv_16x16(self.state.mv_cache, mb_x, mb_y)
            mvx = mvp_x + mvd_x
            mvy = mvp_y + mvd_y

            self.state.mv_cache.set_mv_16x16(mb_x, mb_y, mvx, mvy)

            luma, cb, cr = reconstruct_b_l0_16x16(
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
        elif mb_info.pred_mode == "L1":
            # B_L1_16x16
            ref_idx = 0
            if len(self.state.l1_list) > 1:
                ref_idx = reader.read_te(len(self.state.l1_list) - 1)

            mvd_x = reader.read_se()
            mvd_y = reader.read_se()

            # Use L1 MV prediction (simplified: same as L0 for now)
            mvp_x, mvp_y = predict_mv_16x16(self.state.mv_cache, mb_x, mb_y)
            mvx = mvp_x + mvd_x
            mvy = mvp_y + mvd_y

            luma, cb, cr = reconstruct_b_l1_16x16(
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
        elif mb_info.pred_mode == "Bi":
            # B_Bi_16x16
            ref_idx_l0 = 0
            ref_idx_l1 = 0
            if len(self.state.l0_list) > 1:
                ref_idx_l0 = reader.read_te(len(self.state.l0_list) - 1)
            if len(self.state.l1_list) > 1:
                ref_idx_l1 = reader.read_te(len(self.state.l1_list) - 1)

            # L0 MVD
            mvd_x_l0 = reader.read_se()
            mvd_y_l0 = reader.read_se()
            # L1 MVD
            mvd_x_l1 = reader.read_se()
            mvd_y_l1 = reader.read_se()

            mvp_x, mvp_y = predict_mv_16x16(self.state.mv_cache, mb_x, mb_y)
            mvx_l0 = mvp_x + mvd_x_l0
            mvy_l0 = mvp_y + mvd_y_l0
            mvx_l1 = mvp_x + mvd_x_l1
            mvy_l1 = mvp_y + mvd_y_l1

            luma, cb, cr = reconstruct_b_bi_16x16(
                ref_buffer=self.state.ref_buffer,
                ref_idx_l0=ref_idx_l0,
                mvx_l0=mvx_l0,
                mvy_l0=mvy_l0,
                ref_idx_l1=ref_idx_l1,
                mvx_l1=mvx_l1,
                mvy_l1=mvy_l1,
                residual_luma=None,
                residual_cb=None,
                residual_cr=None,
                mb_x=mb_x,
                mb_y=mb_y,
            )
        else:
            # Fallback to direct mode
            logger.warning(f"Unsupported B-MB pred_mode: {mb_info.pred_mode}")
            luma, cb, cr = reconstruct_b_skip(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                mb_x=mb_x,
                mb_y=mb_y,
                use_spatial=use_spatial,
                current_poc=current_poc,
            )

        self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
        self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
        self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

    def _process_b_skip_run(
        self,
        mb_x: int,
        mb_y: int,
        use_spatial: bool,
        current_poc: int,
    ) -> None:
        """Process a B_Skip macroblock.

        B_Skip uses direct mode MV derivation with no residual.

        Args:
            mb_x, mb_y: Macroblock position
            use_spatial: True for spatial direct mode
            current_poc: Picture order count
        """
        luma, cb, cr = reconstruct_b_skip(
            ref_buffer=self.state.ref_buffer,
            mv_cache=self.state.mv_cache,
            mb_x=mb_x,
            mb_y=mb_y,
            use_spatial=use_spatial,
            current_poc=current_poc,
        )

        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8
        self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
        self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
        self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

        logger.debug(f"B_Skip MB ({mb_x}, {mb_y})")

    def _calculate_poc(
        self,
        slice_header: 'SliceHeader',
        sps: 'SPS',
    ) -> int:
        """Calculate Picture Order Count for current picture.

        Args:
            slice_header: Parsed slice header
            sps: Active SPS

        Returns:
            Calculated POC value
        """
        poc_type = getattr(sps, 'pic_order_cnt_type', 0)

        if poc_type == 0:
            return self._calc_poc_type_0(slice_header, sps)
        elif poc_type == 2:
            return self._calc_poc_type_2(slice_header, sps)
        else:
            # Type 1 not implemented
            return slice_header.frame_num * 2

    def _calc_poc_type_0(
        self,
        slice_header: 'SliceHeader',
        sps: 'SPS',
    ) -> int:
        """Calculate POC using type 0 (pic_order_cnt_lsb).

        POC type 0 uses explicit LSB in slice header with MSB derivation.

        Args:
            slice_header: Parsed slice header
            sps: Active SPS

        Returns:
            Calculated POC value
        """
        max_poc_lsb = 1 << (getattr(sps, 'log2_max_pic_order_cnt_lsb_minus4', 0) + 4)
        poc_lsb = getattr(slice_header, 'pic_order_cnt_lsb', 0)

        # Derive POC MSB
        if poc_lsb < self.state.prev_poc_lsb and \
           (self.state.prev_poc_lsb - poc_lsb) >= (max_poc_lsb // 2):
            poc_msb = self.state.prev_poc_msb + max_poc_lsb
        elif poc_lsb > self.state.prev_poc_lsb and \
             (poc_lsb - self.state.prev_poc_lsb) > (max_poc_lsb // 2):
            poc_msb = self.state.prev_poc_msb - max_poc_lsb
        else:
            poc_msb = self.state.prev_poc_msb

        poc = poc_msb + poc_lsb

        # Update state for next picture
        self.state.prev_poc_msb = poc_msb
        self.state.prev_poc_lsb = poc_lsb

        return poc

    def _calc_poc_type_2(
        self,
        slice_header: 'SliceHeader',
        sps: 'SPS',
    ) -> int:
        """Calculate POC using type 2 (derived from frame_num).

        POC type 2 derives POC directly from frame_num.

        Args:
            slice_header: Parsed slice header
            sps: Active SPS

        Returns:
            Calculated POC value
        """
        # For non-reference pictures, POC = 2*frame_num - 1
        # For reference pictures, POC = 2*frame_num
        frame_num = slice_header.frame_num
        return frame_num * 2

    def _handle_frame_reordering(
        self,
        frame: 'DecodedFrame',
    ) -> Optional['DecodedFrame']:
        """Handle frame reordering for B-frames.

        B-frames may be decoded out of display order.
        This method manages the DPB to output frames in POC order.

        Args:
            frame: Decoded frame

        Returns:
            Frame to output (may be different from input due to reordering)
        """
        # For now, simple pass-through
        # Full implementation would use DPB and output in POC order
        return frame

    def _use_weighted_bipred(
        self,
        slice_header: 'SliceHeader',
    ) -> bool:
        """Check if weighted bi-prediction should be used.

        Args:
            slice_header: Parsed slice header

        Returns:
            True if weighted bi-prediction is enabled
        """
        if self.state.current_pps is None:
            return False

        pps = self.state.current_pps
        weighted_bipred_idc = getattr(pps, 'weighted_bipred_idc', 0)

        # 0 = default (no weighted)
        # 1 = explicit weighted
        # 2 = implicit weighted
        return weighted_bipred_idc > 0

    def get_weighted_pred_mode(self) -> int:
        """Get weighted prediction mode from current PPS.

        Returns:
            0 = no weighted prediction (default)
            1 = explicit weighted prediction (P-slices)
            2 = implicit weighted prediction (B-slices only)

        For P-slices: weighted_pred_flag (0 or 1)
        For B-slices: weighted_bipred_idc (0, 1, or 2)
        """
        if self.state.current_pps is None:
            return 0

        pps = self.state.current_pps

        # weighted_pred_flag controls P-slice weighted prediction
        weighted_pred_flag = getattr(pps, 'weighted_pred_flag', False)
        if weighted_pred_flag:
            return 1

        # weighted_bipred_idc controls B-slice weighted prediction
        weighted_bipred_idc = getattr(pps, 'weighted_bipred_idc', 0)
        return weighted_bipred_idc

    def _store_mvs_in_ref(
        self,
        ref_frame: 'ReferenceFrame',
    ) -> None:
        """Store MVs in reference frame for temporal direct mode.

        Args:
            ref_frame: Reference frame to store MVs in
        """
        if self.state.mv_cache is None:
            return

        sps = self.state.current_sps
        if sps is None:
            return

        for mb_y in range(sps.frame_height_in_mbs):
            for mb_x in range(sps.pic_width_in_mbs):
                mv = self.state.mv_cache.get_mv(mb_x, mb_y, 0, 0)
                if mv != (0, 0):
                    ref_frame.store_mv(mb_x, mb_y, mv[0], mv[1])

    def _combine_partitions(
        self,
        partition_a: bytes,
        partition_b: bytes,
        partition_c: bytes,
    ) -> bytes:
        """Combine data partitions A, B, C into complete slice.

        Data partitioning splits slice data:
        - Partition A: Slice header, MB type, motion data
        - Partition B: Intra residual data
        - Partition C: Inter residual data

        Args:
            partition_a: Partition A NAL data
            partition_b: Partition B NAL data
            partition_c: Partition C NAL data

        Returns:
            Combined slice data
        """
        # Simple concatenation for basic support
        # Full implementation would properly interleave based on MB syntax
        return partition_a + partition_b + partition_c

    # -------------------------------------------------------------------------
    # CABAC entropy decoding support
    # -------------------------------------------------------------------------

    def _get_entropy_mode(self, entropy_flag: bool) -> str:
        """Get entropy coding mode based on PPS flag.

        Args:
            entropy_flag: entropy_coding_mode_flag from PPS

        Returns:
            'cavlc' or 'cabac'
        """
        return 'cabac' if entropy_flag else 'cavlc'

    def _init_cabac_contexts(
        self,
        slice_type: int,
        slice_qp: int,
    ) -> None:
        """Initialize CABAC context models for a slice.

        Args:
            slice_type: Slice type (0=P, 1=B, 2=I)
            slice_qp: Slice QP value
        """
        from entropy.cabac_context import init_context_models

        self.state.cabac_contexts = init_context_models(slice_type, slice_qp)
        logger.debug(f"Initialized CABAC contexts for slice_type={slice_type}, QP={slice_qp}")

    def _create_cabac_decoder(
        self,
        reader: 'BitReader',
    ) -> 'CABACDecoder':
        """Create CABAC arithmetic decoder.

        Args:
            reader: BitReader positioned at CABAC data

        Returns:
            CABACDecoder instance
        """
        from entropy.cabac_arith import CABACDecoder

        return CABACDecoder(reader)

    def _decode_mb_skip_flag_cabac(
        self,
        cabac_decoder: 'CABACDecoder',
        slice_type: int,
        mb_x: int,
        mb_y: int,
    ) -> int:
        """Decode mb_skip_flag using CABAC.

        Args:
            cabac_decoder: CABAC decoder
            slice_type: Slice type
            mb_x, mb_y: Macroblock position

        Returns:
            mb_skip_flag value (0 or 1)
        """
        from entropy.cabac_syntax import decode_mb_skip_flag

        # Get neighbor skip status
        mb_width = self.state.current_sps.pic_width_in_mbs
        left_skip = False
        top_skip = False

        if mb_x > 0:
            left_idx = mb_y * mb_width + (mb_x - 1)
            left_skip = self.state.mb_types[left_idx] == -1  # -1 = skip
        if mb_y > 0:
            top_idx = (mb_y - 1) * mb_width + mb_x
            top_skip = self.state.mb_types[top_idx] == -1

        return decode_mb_skip_flag(
            cabac_decoder, self.state.cabac_contexts,
            slice_type, mb_x, mb_y, left_skip, top_skip
        )

    def _decode_residual_cabac(
        self,
        cabac_decoder: 'CABACDecoder',
        max_coeff: int,
        block_cat: int,
    ) -> 'np.ndarray':
        """Decode residual block using CABAC.

        Args:
            cabac_decoder: CABAC decoder
            max_coeff: Maximum coefficients (4, 15, or 16)
            block_cat: Block category (0-4)

        Returns:
            Array of coefficients
        """
        from entropy.cabac_residual import decode_residual_block_cabac

        return decode_residual_block_cabac(
            cabac_decoder, self.state.cabac_contexts, max_coeff, block_cat
        )

    def _decode_macroblock_cabac(
        self,
        cabac_decoder: 'CABACDecoder',
        slice_type: int,
        mb_x: int,
        mb_y: int,
        qp: int,
    ) -> None:
        """Decode a macroblock using CABAC.

        Args:
            cabac_decoder: CABAC decoder
            slice_type: Slice type
            mb_x, mb_y: Macroblock position
            qp: Current QP
        """
        from entropy.cabac_syntax import (
            decode_mb_type_i,
            decode_mb_type_p,
            decode_mb_type_b,
        )

        # Decode mb_type based on slice type
        if slice_type == 2 or slice_type == 7:  # I-slice
            mb_type = decode_mb_type_i(cabac_decoder, self.state.cabac_contexts)
        elif slice_type == 0 or slice_type == 5:  # P-slice
            mb_type = decode_mb_type_p(cabac_decoder, self.state.cabac_contexts)
        else:  # B-slice
            mb_type = decode_mb_type_b(cabac_decoder, self.state.cabac_contexts)

        logger.debug(f"CABAC MB ({mb_x}, {mb_y}): type={mb_type}")

        # Store mb_type for neighbor reference
        mb_width = self.state.current_sps.pic_width_in_mbs
        mb_idx = mb_y * mb_width + mb_x
        self.state.mb_types[mb_idx] = mb_type

        # TODO: Decode prediction modes, MVD, CBP, residual using CABAC
        # For now, just placeholder

    def _decode_slice_data_cabac(
        self,
        reader: 'BitReader',
        slice_header: 'SliceHeader',
        sps: 'SPS',
        pps: 'PPS',
        slice_qp: int,
    ) -> None:
        """Decode slice data using CABAC entropy coding.

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

        # Initialize CABAC contexts
        slice_type = slice_header.slice_type % 5
        self._init_cabac_contexts(slice_type, slice_qp)

        # Create CABAC decoder
        cabac_decoder = self._create_cabac_decoder(reader)

        logger.debug(f"Decoding CABAC slice: {total_mbs} macroblocks")

        mb_idx = slice_header.first_mb_in_slice

        while mb_idx < total_mbs:
            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Check for mb_skip_flag (P/B slices only)
            if slice_type in (0, 1):  # P or B slice
                skip_flag = self._decode_mb_skip_flag_cabac(
                    cabac_decoder, slice_type, mb_x, mb_y
                )

                if skip_flag == 1:
                    # Skip this macroblock
                    self.state.mb_types[mb_idx] = -1  # Mark as skipped
                    mb_idx += 1
                    continue

            # Decode macroblock
            self._decode_macroblock_cabac(
                cabac_decoder, slice_type, mb_x, mb_y, slice_qp
            )

            # Check for end_of_slice
            if cabac_decoder.decode_terminate() == 1:
                break

            mb_idx += 1


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
