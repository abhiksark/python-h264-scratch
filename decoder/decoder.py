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
import os
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
from reconstruct.macroblock import (
    decode_cbp_inter,
    decode_chroma_dc,
    decode_chroma_ac,
    build_chroma_residual,
    get_luma_neighbor_nz,
    BLOCK_SCAN_ORDER,
)
from entropy import decode_residual_block, calculate_nC, ZIGZAG_4x4
from dequant import dequant_4x4, get_chroma_qp
from transform import idct_4x4
from color import ycbcr_to_rgb, ColorMatrix
from color.chroma_format import monochrome_to_rgb, ycbcr_422_to_rgb, ycbcr_444_to_rgb
from inter.reference import ReferenceFrame, ReferenceFrameBuffer
from decoder.poc import POCCalculator
from decoder.mmco import MMCOProcessor
from inter.mv_prediction import (
    MVCache,
    predict_mv_16x16,
    predict_mv_16x8,
    predict_mv_8x16,
    predict_mv_8x8,
    predict_mv_partition,
)
from inter.p_macroblock import parse_p_mb_type, parse_sub_mb_type, PMacroblockInfo
from inter.p_reconstruct import (
    reconstruct_p_skip,
    reconstruct_p_skip_weighted,
    reconstruct_p_16x16,
    reconstruct_p_16x16_weighted,
    reconstruct_p_16x8,
    reconstruct_p_16x8_weighted,
    reconstruct_p_8x16,
    reconstruct_p_8x16_weighted,
    reconstruct_p_8x8,
    reconstruct_p_8x8_weighted,
    reconstruct_p_8x8_sub,
    apply_chroma_prediction,
    apply_inter_prediction,
)
from inter.b_macroblock import (
    parse_b_mb_type,
    get_b_skip_info,
    get_partition_pred_modes,
    parse_b_sub_mb_type,
)
from inter.b_reconstruct import (
    reconstruct_b_skip,
    reconstruct_b_l0_16x16,
    reconstruct_b_l1_16x16,
    reconstruct_b_bi_16x16,
    reconstruct_b_direct_16x16,
    reconstruct_b_16x8,
    reconstruct_b_8x16,
    reconstruct_b_8x8,
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
class DecoderStats:
    concealed_mb_count: int = 0


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
        chroma_format_idc: Chroma format ID (0 = monochrome, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4)
    """

    frame_num: int
    poc: int
    luma: np.ndarray
    cb: Optional[np.ndarray]
    cr: Optional[np.ndarray]
    width: int
    height: int
    chroma_format_idc: int = 1

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
        if self.chroma_format_idc == 0 or self.cb is None or self.cr is None:
            return monochrome_to_rgb(self.luma)
        if self.chroma_format_idc == 1:
            return ycbcr_to_rgb(self.luma, self.cb, self.cr, color_matrix=color_matrix)
        if self.chroma_format_idc == 2:
            return ycbcr_422_to_rgb(self.luma, self.cb, self.cr, color_matrix=color_matrix)
        if self.chroma_format_idc == 3:
            return ycbcr_444_to_rgb(self.luma, self.cb, self.cr, color_matrix=color_matrix)
        raise ValueError(f"Invalid chroma_format_idc: {self.chroma_format_idc}")


@dataclass
class DecoderState:
    """Internal decoder state.

    Tracks SPS/PPS parameter sets and current decoding context.
    """

    sps_dict: dict = field(default_factory=dict)  # id -> SPS
    pps_dict: dict = field(default_factory=dict)  # id -> PPS
    current_sps: Optional[SPS] = None
    current_pps: Optional[PPS] = None

    chroma_format_idc: int = 1

    # Frame buffers for reconstruction
    frame_luma: Optional[np.ndarray] = None
    frame_cb: Optional[np.ndarray] = None
    frame_cr: Optional[np.ndarray] = None

    # Non-zero coefficient counts for CAVLC context
    nz_counts: Optional[np.ndarray] = None

    # Intra 4x4 prediction modes per MB for cross-MB MPM computation
    # Shape: (num_mbs, 16) - 16 modes per MB, -1 for non-I4x4 MBs
    intra_modes: Optional[np.ndarray] = None

    # Reference frame buffer for inter prediction
    ref_buffer: Optional[ReferenceFrameBuffer] = None

    # MV cache for current frame (L0 and L1 for B-frames)
    mv_cache: Optional[MVCache] = None
    mv_cache_l1: Optional[MVCache] = None

    # Current macroblock QP (updated by mb_qp_delta)
    current_mb_qp: int = 26

    # Per-MB info for deblocking filter
    mb_types: Optional[np.ndarray] = None  # MB type per MB
    mb_coeffs: Optional[np.ndarray] = None  # Has non-zero coeffs per 4x4 block
    mb_cbps: Optional[np.ndarray] = None  # CBP per MB for CABAC context
    mb_qps: Optional[np.ndarray] = None  # Per-MB QP values
    mb_chroma_modes: Optional[np.ndarray] = None  # Chroma pred mode per MB for CABAC
    mb_qp_deltas: Optional[np.ndarray] = None  # QP delta per MB for CABAC context

    # Per-MB MVD values per 4x4 block for CABAC MVD context (H.264 9.3.3.1.1.7)
    mb_mvds_l0: Optional[np.ndarray] = None  # Shape: (mb_count, 4, 4, 2)
    mb_mvds_l1: Optional[np.ndarray] = None  # Shape: (mb_count, 4, 4, 2)

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

    def apply_sps(self, sps: SPS) -> None:
        self.sps_dict[sps.seq_parameter_set_id] = sps
        self.current_sps = sps
        self.chroma_format_idc = getattr(sps, "chroma_format_idc", 1)

    def allocate_frame_buffers(self, sps: SPS) -> None:
        """Allocate frame buffers based on SPS dimensions."""
        height = sps.frame_height_in_mbs * 16
        width = sps.pic_width_in_mbs * 16

        self.frame_luma = np.zeros((height, width), dtype=np.uint8)

        chroma_format_idc = getattr(sps, "chroma_format_idc", 1)
        self.chroma_format_idc = chroma_format_idc
        if chroma_format_idc == 0:
            cb_shape = (0, 0)
        elif chroma_format_idc == 1:
            cb_shape = (height // 2, width // 2)
        elif chroma_format_idc == 2:
            cb_shape = (height, width // 2)
        elif chroma_format_idc == 3:
            cb_shape = (height, width)
        else:
            raise ValueError(f"Invalid chroma_format_idc: {chroma_format_idc}")

        self.frame_cb = np.zeros(cb_shape, dtype=np.uint8)
        self.frame_cr = np.zeros(cb_shape, dtype=np.uint8)

        # Stored per MB for context calculation
        mb_count = sps.frame_height_in_mbs * sps.pic_width_in_mbs
        if chroma_format_idc == 0:
            blocks_per_mb = 16
        elif chroma_format_idc in (1, 2):
            blocks_per_mb = 24
        else:  # chroma_format_idc == 3
            blocks_per_mb = 48
        self.nz_counts = np.zeros((mb_count, blocks_per_mb), dtype=np.int32)

        # Intra 4x4 prediction modes: initialized to DC (2) like JM reference.
        # I_16x16 and non-intra MBs keep this default value.
        # Only genuinely unavailable neighbors (outside picture) use -1.
        self.intra_modes = np.full((mb_count, 16), 2, dtype=np.int32)

        # Initialize reference frame buffer
        max_refs = sps.max_num_ref_frames if hasattr(sps, 'max_num_ref_frames') else 4
        self.ref_buffer = ReferenceFrameBuffer(max_frames=max(1, max_refs))

        # Per-MB info for deblocking
        self.mb_types = np.zeros(mb_count, dtype=np.int32)
        self.mb_coeffs = np.zeros((mb_count, blocks_per_mb), dtype=bool)
        self.mb_qps = np.zeros(mb_count, dtype=np.int32)
        # CBP per MB for CABAC neighbor context
        self.mb_cbps = np.zeros(mb_count, dtype=np.int32)
        # Chroma pred mode per MB for CABAC neighbor context
        self.mb_chroma_modes = np.zeros(mb_count, dtype=np.int32)
        # QP delta per MB for CABAC context
        self.mb_qp_deltas = np.zeros(mb_count, dtype=np.int32)
        # DC coded_block_flag per MB for CABAC context
        # [0]=luma DC, [1]=Cb DC, [2]=Cr DC
        self.mb_dc_cbf = np.zeros((mb_count, 3), dtype=np.int32)
        # MB skip flags for CABAC skip context derivation
        self.mb_skip_flags = np.zeros(mb_count, dtype=np.int32)
        # Raw CABAC mb_type per MB (preserved for CABAC context derivation).
        # Unlike mb_types (overwritten to 99 for deblocking), this keeps the
        # original decoded value so neighbor lookups get correct condTermFlags.
        self.cabac_mb_types = np.zeros(mb_count, dtype=np.int32)

        # Per-MB MVD values per 4x4 block for CABAC MVD context derivation
        # Shape: (mb_count, 4, 4, 2) → [mb_idx, blk_row, blk_col, comp]
        self.mb_mvds_l0 = np.zeros((mb_count, 4, 4, 2), dtype=np.int32)
        self.mb_mvds_l1 = np.zeros((mb_count, 4, 4, 2), dtype=np.int32)

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
        self.mv_cache_l1 = MVCache(
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

    def __init__(self, deblocking_enabled: bool = True, aso_enabled: bool = False):
        """Initialize decoder.

        Args:
            deblocking_enabled: Whether to apply deblocking filter (default True)
        """
        self.state = DecoderState()
        self.deblocking_enabled = deblocking_enabled
        self.aso_enabled = aso_enabled
        self.chroma_format_idc = 1
        self.poc_calculator = POCCalculator()
        self.error_resilience = False
        self.on_concealment = None
        self.stats = DecoderStats()
        self.mmco_processor = MMCOProcessor()
        self._mmco = self.mmco_processor

        self._trace_mb = None
        self._trace_pixel = None
        self._trace_pre_mb = None
        trace_mb = os.environ.get("H264_TRACE_MB")
        if trace_mb:
            try:
                parts = [p.strip() for p in trace_mb.split(",")]
                if len(parts) == 3:
                    self._trace_mb = (int(parts[0]), int(parts[1]), int(parts[2]))
            except Exception:
                self._trace_mb = None

        trace_pixel = os.environ.get("H264_TRACE_PIXEL")
        if trace_pixel:
            try:
                parts = [p.strip() for p in trace_pixel.split(",")]
                if len(parts) == 2:
                    self._trace_pixel = (int(parts[0]), int(parts[1]))
            except Exception:
                self._trace_pixel = None

    def _process_dec_ref_pic_marking(self, slice_header: SliceHeader, nal: NALUnit) -> None:
        marking = getattr(slice_header, "dec_ref_pic_marking", None)
        if nal.nal_ref_idc == 0 or marking is None:
            return
        if nal.nal_unit_type == NALUnitType.SLICE_IDR:
            self.mmco_processor.process_idr(marking)

    def _apply_dec_ref_pic_marking(
        self,
        slice_header: SliceHeader,
        nal: NALUnit,
        ref_frame: Optional[ReferenceFrame] = None,
    ) -> None:
        marking = getattr(slice_header, "dec_ref_pic_marking", None)

        if nal.nal_ref_idc == 0:
            self.mmco_processor.process_for_non_reference()
            return
        if ref_frame is None or marking is None:
            return

        if nal.nal_unit_type == NALUnitType.SLICE_IDR:
            self.mmco_processor.process_idr(marking, idr_frame=ref_frame)
            return

        sps = getattr(self.state, "current_sps", None)
        max_frame_num = getattr(sps, "max_frame_num", None)
        self.mmco_processor.process_non_idr(
            marking,
            current_frame=ref_frame,
            current_frame_num=slice_header.frame_num,
            max_frame_num=max_frame_num,
            current_poc=getattr(slice_header, "pic_order_cnt_lsb", 0),
        )

    def conceal_macroblock(self, *args, **kwargs):
        from decoder.error_concealment import conceal_macroblock as _conceal_macroblock

        result = _conceal_macroblock(*args, **kwargs)
        try:
            self.stats.concealed_mb_count += 1
        except Exception:
            pass

        cb = getattr(self, "on_concealment", None)
        if callable(cb):
            try:
                cb(result)
            except Exception:
                pass
        return result

    def _detect_aso_mode(self) -> bool:
        return False

    def configure_chroma_format(self, chroma_format_idc: int) -> None:
        chroma_format_idc = int(chroma_format_idc)
        if chroma_format_idc not in (0, 1, 2, 3):
            raise ValueError(f"Invalid chroma_format_idc: {chroma_format_idc}")
        self.chroma_format_idc = chroma_format_idc
        self.state.chroma_format_idc = chroma_format_idc

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
            try:
                frame = self._process_nal(nal)
            except Exception as e:
                if not getattr(self, "error_resilience", False):
                    raise
                logger.warning(
                    f"Error processing NAL type={getattr(nal, 'nal_unit_type', None)} "
                    f"at pos={getattr(nal, 'start_position', None)}: {e}"
                )
                continue
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
        try:
            sps = parse_sps(nal.rbsp)  # parse_sps expects raw bytes
        except Exception as e:
            logger.warning(f"Failed to parse SPS (truncated or malformed): {e}")
            return
        self.state.sps_dict[sps.seq_parameter_set_id] = sps
        logger.info(
            f"Parsed SPS {sps.seq_parameter_set_id}: "
            f"{sps.pic_width_in_mbs * 16}x{sps.frame_height_in_mbs * 16}"
        )

    def _process_pps(self, nal: NALUnit) -> None:
        """Parse and store PPS."""
        try:
            pps = parse_pps(nal.rbsp)  # parse_pps expects raw bytes
        except Exception as e:
            logger.warning(f"Failed to parse PPS (truncated or malformed): {e}")
            return
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

        # Peek at first_mb_in_slice / slice_type / pic_parameter_set_id
        # to select the correct PPS for this slice.
        try:
            probe_reader = BitReader(nal.rbsp)
            probe_reader.read_ue()  # first_mb_in_slice
            probe_reader.read_ue()  # slice_type
            pps_id = probe_reader.read_ue()
        except Exception as exc:
            logger.warning(f"Failed to probe slice header PPS id: {exc}")
            return None

        try:
            pps = self.state.get_pps(pps_id)
        except Exception as exc:
            logger.warning(f"PPS {pps_id} not found for slice: {exc}")
            return None

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

        # Clear reference buffer on IDR (H.264 8.2.5.1)
        if nal.nal_unit_type == NALUnitType.SLICE_IDR:
            if self.state.ref_buffer is not None:
                self.state.ref_buffer.clear()
                logger.debug("Cleared reference buffer on IDR")

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

        # Apply deblocking filter (in-place on frame buffers)
        if self.deblocking_enabled and slice_header.deblocking_enabled:
            self._deblock_frame(slice_header, sps, pps)

        trace_pre_mb = getattr(self, "_trace_pre_mb", None)
        if trace_pre_mb is not None:
            try:
                t_frame_num, t_mb_x, t_mb_y, pre_luma = trace_pre_mb
                if t_frame_num == slice_header.frame_num:
                    ly, lx = t_mb_y * 16, t_mb_x * 16
                    post_luma = self.state.frame_luma[ly:ly + 16, lx:lx + 16]
                    d = np.abs(post_luma.astype(np.int16) - pre_luma.astype(np.int16))
                    logger.warning(
                        "TRACE MB post-deblock frame_num=%s MB=(%s,%s) max_delta=%s mean_delta=%.6f",
                        t_frame_num,
                        t_mb_x,
                        t_mb_y,
                        int(d.max()),
                        float(d.mean()),
                    )
                    trace_pixel = getattr(self, "_trace_pixel", None)
                    if trace_pixel is not None:
                        px, py = trace_pixel
                        if lx <= px < lx + 16 and ly <= py < ly + 16:
                            rx, ry = px - lx, py - ly
                            logger.warning(
                                "TRACE pixel post-deblock frame_num=%s (x,y)=(%s,%s) rel=(%s,%s) pre=%s post=%s",
                                t_frame_num,
                                px,
                                py,
                                rx,
                                ry,
                                int(pre_luma[ry, rx]),
                                int(post_luma[ry, rx]),
                            )
            except Exception:
                pass

        # Return completed frame
        # For simplicity, assume each slice is a complete frame
        out_luma = self.state.frame_luma
        out_cb = self.state.frame_cb
        out_cr = self.state.frame_cr
        out_width = sps.pic_width_in_mbs * 16
        out_height = sps.frame_height_in_mbs * 16

        if getattr(sps, "frame_cropping_flag", False):
            chroma_format_idc = getattr(sps, "chroma_format_idc", 1)
            separate_colour_plane_flag = getattr(
                sps, "separate_colour_plane_flag", False
            )

            if separate_colour_plane_flag or chroma_format_idc == 0:
                sub_w, sub_h = 1, 1
            elif chroma_format_idc == 1:
                sub_w, sub_h = 2, 2
            elif chroma_format_idc == 2:
                sub_w, sub_h = 2, 1
            else:  # chroma_format_idc == 3
                sub_w, sub_h = 1, 1

            if separate_colour_plane_flag or chroma_format_idc == 0:
                crop_unit_x = 1
                crop_unit_y = 2 - (1 if sps.frame_mbs_only_flag else 0)
            else:
                crop_unit_x = sub_w
                crop_unit_y = sub_h * (2 - (1 if sps.frame_mbs_only_flag else 0))

            crop_left = int(sps.frame_crop_left_offset * crop_unit_x)
            crop_right = int(sps.frame_crop_right_offset * crop_unit_x)
            crop_top = int(sps.frame_crop_top_offset * crop_unit_y)
            crop_bottom = int(sps.frame_crop_bottom_offset * crop_unit_y)

            out_luma = out_luma[
                crop_top : out_height - crop_bottom,
                crop_left : out_width - crop_right,
            ]
            out_height, out_width = out_luma.shape

            if chroma_format_idc != 0 and out_cb is not None and out_cr is not None:
                crop_left_c = crop_left // sub_w
                crop_right_c = crop_right // sub_w
                crop_top_c = crop_top // sub_h
                crop_bottom_c = crop_bottom // sub_h

                out_cb = out_cb[
                    crop_top_c : out_cb.shape[0] - crop_bottom_c,
                    crop_left_c : out_cb.shape[1] - crop_right_c,
                ]
                out_cr = out_cr[
                    crop_top_c : out_cr.shape[0] - crop_bottom_c,
                    crop_left_c : out_cr.shape[1] - crop_right_c,
                ]

        frame = DecodedFrame(
            frame_num=slice_header.frame_num,
            poc=getattr(slice_header, "pic_order_cnt_lsb", 0),
            luma=out_luma.copy(),
            cb=out_cb.copy() if out_cb is not None else None,
            cr=out_cr.copy() if out_cr is not None else None,
            width=out_width,
            height=out_height,
            chroma_format_idc=getattr(sps, "chroma_format_idc", 1),
        )

        # Add to reference buffer if this is a reference frame
        if nal.nal_ref_idc > 0:
            ref_frame = ReferenceFrame(
                luma=self.state.frame_luma.copy(),
                cb=self.state.frame_cb.copy(),
                cr=self.state.frame_cr.copy(),
                frame_num=slice_header.frame_num,
                poc=getattr(slice_header, "pic_order_cnt_lsb", 0),
            )
            self._store_mvs_in_ref(ref_frame)
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

        if slice_header.first_mb_in_slice == 0 and self.state.intra_modes is not None:
            # Reset intra 4x4 prediction modes at the start of a new frame.
            # Prevents stale modes from previous frames influencing MPM.
            self.state.intra_modes[:] = 2

        current_qp = slice_qp

        for mb_idx in range(slice_header.first_mb_in_slice, total_mbs):
            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Check for RBSP trailing bits (H.264 spec 7.2)
            if not reader.more_rbsp_data():
                logger.warning(f"Reached RBSP trailing bits after MB ({mb_x-1 if mb_x > 0 else mb_width-1}, "
                             f"{mb_y-1 if mb_x == 0 else mb_y}), stopping slice decode "
                             f"({mb_idx}/{total_mbs} MBs decoded)")
                break

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
                    frame_nz_counts=self.state.nz_counts,
                    frame_width_mbs=mb_width,
                    frame_intra_modes=self.state.intra_modes,
                )

                # Store non-zero counts for CAVLC context
                self.state.nz_counts[mb_idx] = mb_data.nz_counts

                # Store MB type for neighbor reference and diagnostics
                if self.state.mb_types is not None:
                    self.state.mb_types[mb_idx] = mb_data.mb_type

                # Update running QP for next MB (H.264: QP is cumulative)
                current_qp = (current_qp + mb_data.mb_qp_delta + 52) % 52

                # Store per-MB QP for deblocking filter
                if self.state.mb_qps is not None:
                    self.state.mb_qps[mb_idx] = current_qp

                # Store per-block coefficient flags for deblocking
                if self.state.mb_coeffs is not None:
                    self.state.mb_coeffs[mb_idx, :16] = (
                        mb_data.nz_counts[:16] > 0
                    )

                logger.debug(f"Decoded MB ({mb_x}, {mb_y}): type={mb_data.mb_type}")

            except ValueError as e:
                # Check if it's an invalid mb_type error (likely RBSP trailing bits)
                if "Invalid I_16x16 mb_type" in str(e) or "Invalid I_4x4 mb_type" in str(e):
                    logger.warning(f"Invalid mb_type at MB ({mb_x}, {mb_y}), likely RBSP trailing bits. "
                                 f"Stopping slice decode ({mb_idx}/{total_mbs} MBs decoded)")
                    break
                else:
                    logger.error(f"Error decoding MB ({mb_x}, {mb_y}): {e}")
                    raise
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

        # Apply reference list reordering for P-slices (H.264 8.2.4)
        max_frame_num = getattr(sps, 'max_frame_num', 256)
        mod_l0 = getattr(slice_header, 'ref_pic_list_modification_l0', None)
        self.state.ref_buffer.build_p_slice_ref_list(
            slice_header.frame_num, max_frame_num, mod_l0
        )

        current_qp = slice_qp
        mb_idx = slice_header.first_mb_in_slice

        use_weighted_pred = self._use_weighted_prediction(slice_header)
        weight_table = slice_header.weighted_pred_table if use_weighted_pred else None
        if use_weighted_pred and weight_table is None:
            logger.debug(
                "Weighted prediction enabled but no weight table found; "
                "falling back to unweighted prediction."
            )
            use_weighted_pred = False

        skip_weight_params = None
        if use_weighted_pred and weight_table is not None:
            w_luma, o_luma = weight_table.get_luma_weight(0)
            w_cb, o_cb, w_cr, o_cr = weight_table.get_chroma_weight(0)
            skip_weight_params = {
                "weight_luma": w_luma,
                "offset_luma": o_luma,
                "log2_denom_luma": weight_table.luma_log2_weight_denom,
                "weight_cb": w_cb,
                "offset_cb": o_cb,
                "weight_cr": w_cr,
                "offset_cr": o_cr,
                "log2_denom_chroma": weight_table.chroma_log2_weight_denom,
            }

        # Reset intra modes to DC (2) for this frame.
        # H.264 8.3.1.1: non-I_NxN MBs use intraMxMPredModeN = 2 (DC).
        # Without reset, stale I-frame modes leak through I_16x16 and P-MB positions.
        self.state.intra_modes[:] = 2

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
                if skip_weight_params is not None:
                    luma, cb, cr = reconstruct_p_skip_weighted(
                        ref_buffer=self.state.ref_buffer,
                        mv_cache=self.state.mv_cache,
                        mb_x=mb_x,
                        mb_y=mb_y,
                        **skip_weight_params,
                    )
                else:
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

                # Clear nz_counts for CAVLC context (skip has no residual)
                self.state.nz_counts[mb_idx] = 0

                # Store deblocking metadata for P_Skip
                if self.state.mb_types is not None:
                    self.state.mb_types[mb_idx] = 99  # Non-intra sentinel
                if self.state.mb_qps is not None:
                    self.state.mb_qps[mb_idx] = current_qp
                if self.state.mb_coeffs is not None:
                    self.state.mb_coeffs[mb_idx] = 0

                logger.debug(f"P_Skip MB ({mb_x}, {mb_y})")
                mb_idx += 1

            # After skip run, check for end of slice (H.264 7.3.4)
            if mb_idx >= total_mbs:
                break
            if not reader.more_rbsp_data():
                # Remaining MBs are implicitly skipped
                while mb_idx < total_mbs:
                    mb_x = mb_idx % mb_width
                    mb_y = mb_idx // mb_width
                    if skip_weight_params is not None:
                        luma, cb, cr = reconstruct_p_skip_weighted(
                            ref_buffer=self.state.ref_buffer,
                            mv_cache=self.state.mv_cache,
                            mb_x=mb_x,
                            mb_y=mb_y,
                            **skip_weight_params,
                        )
                    else:
                        luma, cb, cr = reconstruct_p_skip(
                            ref_buffer=self.state.ref_buffer,
                            mv_cache=self.state.mv_cache,
                            mb_x=mb_x, mb_y=mb_y,
                        )
                    ly, lx = mb_y * 16, mb_x * 16
                    cy, cx = mb_y * 8, mb_x * 8
                    self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
                    self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
                    self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr
                    self.state.nz_counts[mb_idx] = 0
                    if self.state.mb_types is not None:
                        self.state.mb_types[mb_idx] = 99
                    if self.state.mb_qps is not None:
                        self.state.mb_qps[mb_idx] = current_qp
                    if self.state.mb_coeffs is not None:
                        self.state.mb_coeffs[mb_idx] = 0
                    logger.debug(f"P_Skip MB ({mb_x}, {mb_y}) [implicit]")
                    mb_idx += 1
                break

            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Read mb_type
            mb_type_code = reader.read_ue()
            mb_type = parse_p_mb_type(mb_type_code)

            logger.debug(f"MB ({mb_x}, {mb_y}): type={mb_type.name}")

            if mb_type.is_intra:
                # I-macroblock in P-slice - use I-MB decoder
                # mb_type already consumed above; pass intra_type to skip re-read
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
                    frame_nz_counts=self.state.nz_counts,
                    frame_width_mbs=mb_width,
                    frame_intra_modes=self.state.intra_modes,
                    pre_parsed_mb_type=mb_type.intra_type,
                )
                # Mark intra MB: MV=(0,0) with ref_idx=-1
                # Prevents P_Skip zero-MV shortcut for intra neighbors
                self.state.mv_cache.mark_intra(mb_x, mb_y)
                # Update running QP for next MB
                current_qp = (current_qp + mb_data.mb_qp_delta + 52) % 52

                # Store nz_counts for CAVLC context of neighboring MBs
                mb_idx_p = mb_y * mb_width + mb_x
                self.state.nz_counts[mb_idx_p] = mb_data.nz_counts

                # Store deblocking metadata for intra-in-P
                if self.state.mb_types is not None:
                    self.state.mb_types[mb_idx_p] = mb_data.mb_type
                if self.state.mb_qps is not None:
                    self.state.mb_qps[mb_idx_p] = current_qp
                if self.state.mb_coeffs is not None:
                    self.state.mb_coeffs[mb_idx_p, :16] = (
                        mb_data.nz_counts[:16] > 0
                    )

            else:
                # P-macroblock
                current_qp = self._decode_p_macroblock(
                    reader,
                    mb_type,
                    mb_x,
                    mb_y,
                    current_qp,
                    sps,
                    pps,
                    frame_num=slice_header.frame_num,
                    use_weighted_pred=use_weighted_pred,
                    weight_table=weight_table,
                    num_ref_idx_active=getattr(
                        slice_header,
                        'num_ref_idx_l0_active_minus1', 0
                    ) + 1,
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
        frame_num: Optional[int] = None,
        use_weighted_pred: bool = False,
        weight_table: Optional['WeightTable'] = None,
        num_ref_idx_active: Optional[int] = None,
    ) -> int:
        """Decode a single P-macroblock.

        H.264 Spec Reference: Section 7.3.5 (macroblock_layer syntax)

        Args:
            reader: BitReader
            mb_type: Parsed PMBType
            mb_x, mb_y: Macroblock position
            qp: Current QP
            sps, pps: Parameter sets

        Returns:
            Updated QP after mb_qp_delta
        """
        mb_width = sps.pic_width_in_mbs
        mb_idx = mb_y * mb_width + mb_x

        trace_mb = getattr(self, "_trace_mb", None)
        trace = (
            trace_mb is not None
            and frame_num is not None
            and trace_mb == (frame_num, mb_x, mb_y)
        )
        if trace:
            logger.warning(
                "TRACE MB start frame_num=%s MB=(%s,%s) type=%s",
                frame_num,
                mb_x,
                mb_y,
                getattr(mb_type, "name", None),
            )

        use_weighted = use_weighted_pred and weight_table is not None
        if use_weighted:
            luma_log2_denom = weight_table.luma_log2_weight_denom
            chroma_log2_denom = weight_table.chroma_log2_weight_denom

            def _weights_for_ref(ref_index: int):
                w_luma, o_luma = weight_table.get_luma_weight(ref_index)
                w_cb, o_cb, w_cr, o_cr = weight_table.get_chroma_weight(ref_index)
                return w_luma, o_luma, w_cb, o_cb, w_cr, o_cr

        # ── Step 1: Parse MVs and do motion compensation (partition-specific) ──

        # Number of active L0 references for syntax parsing
        if num_ref_idx_active is None:
            num_refs = len(self.state.ref_buffer) if self.state.ref_buffer else 1
        else:
            num_refs = num_ref_idx_active

        if mb_type.name == "P_L0_16x16":
            ref_idx = 0
            if num_refs > 1:
                ref_idx = reader.read_te(num_refs - 1)

            mvd_x = reader.read_se()
            mvd_y = reader.read_se()
            mvp_x, mvp_y = predict_mv_16x16(
                self.state.mv_cache, mb_x, mb_y, target_ref=ref_idx,
            )
            mvx = mvp_x + mvd_x
            mvy = mvp_y + mvd_y

            if trace:
                logger.warning(
                    "TRACE P_16x16 ref_idx=%s mvd=(%s,%s) mvp=(%s,%s) mv=(%s,%s)",
                    ref_idx,
                    mvd_x,
                    mvd_y,
                    mvp_x,
                    mvp_y,
                    mvx,
                    mvy,
                )

            # Store MV for this MB so neighbors can predict correctly.
            if self.state.mv_cache is not None:
                self.state.mv_cache.set_mv_16x16(
                    mb_x, mb_y, mvx, mvy, ref_idx=ref_idx,
                )

            if use_weighted:
                w_luma, o_luma, w_cb, o_cb, w_cr, o_cr = _weights_for_ref(ref_idx)
                luma_pred, cb_pred, cr_pred = reconstruct_p_16x16_weighted(
                    ref_buffer=self.state.ref_buffer,
                    ref_idx=ref_idx,
                    mvx=mvx,
                    mvy=mvy,
                    weight_luma=w_luma,
                    offset_luma=o_luma,
                    log2_denom_luma=luma_log2_denom,
                    weight_cb=w_cb,
                    offset_cb=o_cb,
                    weight_cr=w_cr,
                    offset_cr=o_cr,
                    log2_denom_chroma=chroma_log2_denom,
                    residual_luma=None,
                    residual_cb=None,
                    residual_cr=None,
                    mb_x=mb_x,
                    mb_y=mb_y,
                )
            else:
                luma_pred, cb_pred, cr_pred = reconstruct_p_16x16(
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
            logger.debug(f"P_16x16 MB ({mb_x}, {mb_y}): MV=({mvx}, {mvy})")

        elif mb_type.name == "P_L0_L0_16x8":
            ref_idx = [0, 0]
            mvx = [0, 0]
            mvy = [0, 0]
            if num_refs > 1:
                ref_idx[0] = reader.read_te(num_refs - 1)
                ref_idx[1] = reader.read_te(num_refs - 1)
            for part in range(2):
                mvd_x = reader.read_se()
                mvd_y = reader.read_se()
                mvp_x, mvp_y = predict_mv_16x8(
                    self.state.mv_cache, mb_x, mb_y, part,
                    target_ref=ref_idx[part],
                )
                mvx[part] = mvp_x + mvd_x
                mvy[part] = mvp_y + mvd_y
                if trace:
                    logger.warning(
                        "TRACE P_16x8 part=%s ref_idx=%s mvd=(%s,%s) mvp=(%s,%s) mv=(%s,%s)",
                        part,
                        ref_idx[part],
                        mvd_x,
                        mvd_y,
                        mvp_x,
                        mvp_y,
                        mvx[part],
                        mvy[part],
                    )
                # Store partition MV immediately so next partition can
                # use it as a neighbor (H.264 8.4.1.3.1)
                by_start = part * 2
                for bx in range(4):
                    for by in range(by_start, by_start + 2):
                        self.state.mv_cache.set_mv(
                            mb_x, mb_y, bx, by,
                            mvx[part], mvy[part],
                            ref_idx=ref_idx[part],
                        )

            if use_weighted:
                weights_luma = []
                offsets_luma = []
                weights_cb = []
                offsets_cb = []
                weights_cr = []
                offsets_cr = []
                for idx in ref_idx:
                    w_luma, o_luma, w_cb, o_cb, w_cr, o_cr = _weights_for_ref(idx)
                    weights_luma.append(w_luma)
                    offsets_luma.append(o_luma)
                    weights_cb.append(w_cb)
                    offsets_cb.append(o_cb)
                    weights_cr.append(w_cr)
                    offsets_cr.append(o_cr)

                luma_pred, cb_pred, cr_pred = reconstruct_p_16x8_weighted(
                    ref_buffer=self.state.ref_buffer,
                    mv_cache=self.state.mv_cache,
                    ref_idx=ref_idx,
                    mvx=mvx,
                    mvy=mvy,
                    weights_luma=weights_luma,
                    offsets_luma=offsets_luma,
                    log2_denom_luma=luma_log2_denom,
                    weights_cb=weights_cb,
                    offsets_cb=offsets_cb,
                    weights_cr=weights_cr,
                    offsets_cr=offsets_cr,
                    log2_denom_chroma=chroma_log2_denom,
                    residual_luma=None,
                    residual_cb=None,
                    residual_cr=None,
                    mb_x=mb_x,
                    mb_y=mb_y,
                )
            else:
                luma_pred, cb_pred, cr_pred = reconstruct_p_16x8(
                    ref_buffer=self.state.ref_buffer, mv_cache=self.state.mv_cache,
                    ref_idx=ref_idx, mvx=mvx, mvy=mvy,
                    residual_luma=None, residual_cb=None, residual_cr=None,
                    mb_x=mb_x, mb_y=mb_y,
                )
            logger.debug(f"P_16x8 MB ({mb_x}, {mb_y}): MVs={list(zip(mvx, mvy))}")

        elif mb_type.name == "P_L0_L0_8x16":
            ref_idx = [0, 0]
            mvx = [0, 0]
            mvy = [0, 0]
            if num_refs > 1:
                ref_idx[0] = reader.read_te(num_refs - 1)
                ref_idx[1] = reader.read_te(num_refs - 1)
            for part in range(2):
                mvd_x = reader.read_se()
                mvd_y = reader.read_se()
                mvp_x, mvp_y = predict_mv_8x16(
                    self.state.mv_cache, mb_x, mb_y, part,
                    target_ref=ref_idx[part],
                )
                mvx[part] = mvp_x + mvd_x
                mvy[part] = mvp_y + mvd_y
                if trace:
                    logger.warning(
                        "TRACE P_8x16 part=%s ref_idx=%s mvd=(%s,%s) mvp=(%s,%s) mv=(%s,%s)",
                        part,
                        ref_idx[part],
                        mvd_x,
                        mvd_y,
                        mvp_x,
                        mvp_y,
                        mvx[part],
                        mvy[part],
                    )
                # Store partition MV immediately so next partition can
                # use it as a neighbor (H.264 8.4.1.3.1)
                bx_start = part * 2
                for by in range(4):
                    for bx in range(bx_start, bx_start + 2):
                        self.state.mv_cache.set_mv(
                            mb_x, mb_y, bx, by,
                            mvx[part], mvy[part],
                            ref_idx=ref_idx[part],
                        )

            if use_weighted:
                weights_luma = []
                offsets_luma = []
                weights_cb = []
                offsets_cb = []
                weights_cr = []
                offsets_cr = []
                for idx in ref_idx:
                    w_luma, o_luma, w_cb, o_cb, w_cr, o_cr = _weights_for_ref(idx)
                    weights_luma.append(w_luma)
                    offsets_luma.append(o_luma)
                    weights_cb.append(w_cb)
                    offsets_cb.append(o_cb)
                    weights_cr.append(w_cr)
                    offsets_cr.append(o_cr)

                luma_pred, cb_pred, cr_pred = reconstruct_p_8x16_weighted(
                    ref_buffer=self.state.ref_buffer,
                    mv_cache=self.state.mv_cache,
                    ref_idx=ref_idx,
                    mvx=mvx,
                    mvy=mvy,
                    weights_luma=weights_luma,
                    offsets_luma=offsets_luma,
                    log2_denom_luma=luma_log2_denom,
                    weights_cb=weights_cb,
                    offsets_cb=offsets_cb,
                    weights_cr=weights_cr,
                    offsets_cr=offsets_cr,
                    log2_denom_chroma=chroma_log2_denom,
                    residual_luma=None,
                    residual_cb=None,
                    residual_cr=None,
                    mb_x=mb_x,
                    mb_y=mb_y,
                )
            else:
                luma_pred, cb_pred, cr_pred = reconstruct_p_8x16(
                    ref_buffer=self.state.ref_buffer, mv_cache=self.state.mv_cache,
                    ref_idx=ref_idx, mvx=mvx, mvy=mvy,
                    residual_luma=None, residual_cb=None, residual_cr=None,
                    mb_x=mb_x, mb_y=mb_y,
                )
            logger.debug(f"P_8x16 MB ({mb_x}, {mb_y}): MVs={list(zip(mvx, mvy))}")

        elif mb_type.name in ("P_8x8", "P_8x8ref0"):
            sub_mb_types = []
            sub_mb_num_parts = []
            ref_idx = [0, 0, 0, 0]
            mvx = [0, 0, 0, 0]
            mvy = [0, 0, 0, 0]
            for i in range(4):
                raw_sub_type = reader.read_ue()
                sub_mb_types.append(raw_sub_type)
                # P sub-MB: 0=8x8(1 part), 1=8x4(2), 2=4x8(2), 3=4x4(4)
                num_parts = [1, 2, 2, 4][min(raw_sub_type, 3)]
                sub_mb_num_parts.append(num_parts)
            if mb_type.name == "P_8x8" and num_refs > 1:
                for i in range(4):
                    ref_idx[i] = reader.read_te(num_refs - 1)
            for sub_idx in range(4):
                num_sub_parts = sub_mb_num_parts[sub_idx]
                for j in range(num_sub_parts):
                    mvd_x = reader.read_se()
                    mvd_y = reader.read_se()
                    if j == 0:
                        mvp_x, mvp_y = predict_mv_8x8(
                            self.state.mv_cache, mb_x, mb_y, sub_idx,
                            target_ref=ref_idx[sub_idx],
                        )
                        mvx[sub_idx] = mvp_x + mvd_x
                        mvy[sub_idx] = mvp_y + mvd_y
                        if trace:
                            logger.warning(
                                "TRACE P_8x8 sub=%s ref_idx=%s mvd=(%s,%s) mvp=(%s,%s) mv=(%s,%s)",
                                sub_idx,
                                ref_idx[sub_idx],
                                mvd_x,
                                mvd_y,
                                mvp_x,
                                mvp_y,
                                mvx[sub_idx],
                                mvy[sub_idx],
                            )
                # Store MV in cache immediately so subsequent sub-MBs
                # can use it as a neighbor (H.264 8.4.1.3.1)
                bx_start = (sub_idx % 2) * 2
                by_start = (sub_idx // 2) * 2
                for by in range(by_start, by_start + 2):
                    for bx in range(bx_start, bx_start + 2):
                        self.state.mv_cache.set_mv(
                            mb_x, mb_y, bx, by,
                            mvx[sub_idx], mvy[sub_idx],
                            ref_idx=ref_idx[sub_idx],
                        )

            if use_weighted:
                weights_luma = []
                offsets_luma = []
                weights_cb = []
                offsets_cb = []
                weights_cr = []
                offsets_cr = []
                for idx in ref_idx:
                    w_luma, o_luma, w_cb, o_cb, w_cr, o_cr = _weights_for_ref(idx)
                    weights_luma.append(w_luma)
                    offsets_luma.append(o_luma)
                    weights_cb.append(w_cb)
                    offsets_cb.append(o_cb)
                    weights_cr.append(w_cr)
                    offsets_cr.append(o_cr)

                luma_pred, cb_pred, cr_pred = reconstruct_p_8x8_weighted(
                    ref_buffer=self.state.ref_buffer,
                    mv_cache=self.state.mv_cache,
                    ref_idx=ref_idx,
                    mvx=mvx,
                    mvy=mvy,
                    sub_mb_types=sub_mb_types,
                    weights_luma=weights_luma,
                    offsets_luma=offsets_luma,
                    log2_denom_luma=luma_log2_denom,
                    weights_cb=weights_cb,
                    offsets_cb=offsets_cb,
                    weights_cr=weights_cr,
                    offsets_cr=offsets_cr,
                    log2_denom_chroma=chroma_log2_denom,
                    residual_luma=None,
                    residual_cb=None,
                    residual_cr=None,
                    mb_x=mb_x,
                    mb_y=mb_y,
                )
            else:
                luma_pred, cb_pred, cr_pred = reconstruct_p_8x8(
                    ref_buffer=self.state.ref_buffer, mv_cache=self.state.mv_cache,
                    ref_idx=ref_idx, mvx=mvx, mvy=mvy,
                    sub_mb_types=sub_mb_types,
                    residual_luma=None, residual_cb=None, residual_cr=None,
                    mb_x=mb_x, mb_y=mb_y,
                )
            logger.debug(f"P_8x8 MB ({mb_x}, {mb_y}): MVs={list(zip(mvx, mvy))}")

        else:
            logger.warning(f"Unsupported P-MB type: {mb_type.name}, treating as skip")
            if use_weighted:
                w_luma, o_luma, w_cb, o_cb, w_cr, o_cr = _weights_for_ref(0)
                luma_pred, cb_pred, cr_pred = reconstruct_p_skip_weighted(
                    ref_buffer=self.state.ref_buffer,
                    mv_cache=self.state.mv_cache,
                    mb_x=mb_x,
                    mb_y=mb_y,
                    weight_luma=w_luma,
                    offset_luma=o_luma,
                    log2_denom_luma=luma_log2_denom,
                    weight_cb=w_cb,
                    offset_cb=o_cb,
                    weight_cr=w_cr,
                    offset_cr=o_cr,
                    log2_denom_chroma=chroma_log2_denom,
                )
            else:
                luma_pred, cb_pred, cr_pred = reconstruct_p_skip(
                    ref_buffer=self.state.ref_buffer,
                    mv_cache=self.state.mv_cache,
                    mb_x=mb_x, mb_y=mb_y,
                )

        # ── Step 2: Parse coded_block_pattern (H.264 Table 9-4, inter column) ──

        cbp_coded = reader.read_ue()
        cbp_luma, cbp_chroma = decode_cbp_inter(cbp_coded)

        # Parse mb_qp_delta if any coefficients are coded
        mb_qp_delta = 0
        if cbp_luma > 0 or cbp_chroma > 0:
            mb_qp_delta = reader.read_se()
        current_qp = (qp + mb_qp_delta + 52) % 52

        if trace:
            logger.warning(
                "TRACE CBP cbp_coded=%s luma=%s chroma=%s mb_qp_delta=%s qp=%s",
                cbp_coded,
                cbp_luma,
                cbp_chroma,
                mb_qp_delta,
                current_qp,
            )

        logger.debug(
            f"  CBP: luma={cbp_luma:#06b}, chroma={cbp_chroma}, "
            f"qp_delta={mb_qp_delta}, qp={current_qp}"
        )

        # ── Step 3: Parse luma residual (16 4x4 blocks via CAVLC) ──

        nz_counts = np.zeros(24, dtype=np.int32)
        luma_residual = np.zeros((16, 16), dtype=np.int32)

        for block_idx in range(16):
            block_8x8_idx = block_idx // 4
            if not (cbp_luma & (1 << block_8x8_idx)):
                nz_counts[block_idx] = 0
                continue

            nA, nB = get_luma_neighbor_nz(
                block_idx, mb_x, mb_y, nz_counts,
                self.state.nz_counts, mb_width
            )
            nC = calculate_nC(nA, nB)
            block = decode_residual_block(reader, nC, max_coeffs=16)
            nz_counts[block_idx] = block.total_coeff

            if trace:
                logger.warning(
                    "TRACE block=%d nA=%s nB=%s nC=%d tc=%d coeffs=%s",
                    block_idx, nA, nB, nC, block.total_coeff,
                    [block.coefficients[j] for j in range(16) if block.coefficients[j] != 0][:8]
                )

            if block.total_coeff > 0:
                coeffs_2d = np.zeros((4, 4), dtype=np.int32)
                for scan_idx in range(16):
                    raster_pos = ZIGZAG_4x4[scan_idx]
                    r, c = raster_pos // 4, raster_pos % 4
                    coeffs_2d[r, c] = block.coefficients[scan_idx]
                dequant = dequant_4x4(coeffs_2d, current_qp)
                residual_block = idct_4x4(dequant)
                row, col = BLOCK_SCAN_ORDER[block_idx]
                luma_residual[row:row+4, col:col+4] = residual_block

        # ── Step 4: Parse chroma residual ──

        chroma_qp_offset = getattr(pps, 'chroma_qp_index_offset', 0)
        qpi = max(0, min(51, current_qp + chroma_qp_offset))
        qp_chroma = get_chroma_qp(qpi)

        # Chroma DC: Cb then Cr (H.264 7.3.5.3)
        cb_dc = decode_chroma_dc(reader, cbp_chroma, qp_chroma)
        cr_dc = decode_chroma_dc(reader, cbp_chroma, qp_chroma)

        # Chroma AC: Cb then Cr
        cb_ac = decode_chroma_ac(
            reader, cbp_chroma, nz_counts, 16,
            mb_x, mb_y, self.state.nz_counts, mb_width
        )
        cr_ac = decode_chroma_ac(
            reader, cbp_chroma, nz_counts, 20,
            mb_x, mb_y, self.state.nz_counts, mb_width
        )

        # Build chroma residual arrays
        cb_residual = build_chroma_residual(cb_dc, cb_ac, cbp_chroma, qp_chroma)
        cr_residual = build_chroma_residual(cr_dc, cr_ac, cbp_chroma, qp_chroma)

        # ── Step 5: Combine prediction + residual, write to frame ──
        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        luma = np.clip(
            luma_pred.astype(np.int32) + luma_residual, 0, 255
        ).astype(np.uint8)
        cb = np.clip(
            cb_pred.astype(np.int32) + cb_residual, 0, 255
        ).astype(np.uint8)
        cr = np.clip(
            cr_pred.astype(np.int32) + cr_residual, 0, 255
        ).astype(np.uint8)

        self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
        self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
        self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

        if trace:
            trace_pixel = getattr(self, "_trace_pixel", None)
            if trace_pixel is not None:
                px, py = trace_pixel
                if lx <= px < lx + 16 and ly <= py < ly + 16:
                    rx, ry = px - lx, py - ly
                    logger.warning(
                        "TRACE pixel pre-deblock frame_num=%s (x,y)=(%s,%s) rel=(%s,%s) pred=%s resid=%s out=%s",
                        frame_num,
                        px,
                        py,
                        rx,
                        ry,
                        int(luma_pred[ry, rx]),
                        int(luma_residual[ry, rx]),
                        int(luma[ry, rx]),
                    )
            self._trace_pre_mb = (frame_num, mb_x, mb_y, luma.copy())

        # Store nz_counts for CAVLC context of neighboring MBs
        self.state.nz_counts[mb_idx] = nz_counts

        # Store deblocking metadata for inter P-MB
        if self.state.mb_types is not None:
            self.state.mb_types[mb_idx] = 99  # Non-intra sentinel
        if self.state.mb_qps is not None:
            self.state.mb_qps[mb_idx] = current_qp
        if self.state.mb_coeffs is not None:
            self.state.mb_coeffs[mb_idx, :16] = (nz_counts[:16] > 0)

        return current_qp

    def _deblock_frame(self, slice_header: 'SliceHeader', sps: 'SPS',
                       pps: 'PPS' = None) -> None:
        """Apply deblocking filter to the entire frame in-place.

        Section 8.7: Deblocking filter modifies frame buffers in-place
        before they are copied to the reference buffer.
        """
        from deblock.deblock import deblock_frame_inplace

        mb_width = sps.pic_width_in_mbs
        mb_height = sps.frame_height_in_mbs
        chroma_qp_offset = pps.chroma_qp_index_offset if pps else 0

        deblock_frame_inplace(
            frame_luma=self.state.frame_luma,
            frame_cb=self.state.frame_cb,
            frame_cr=self.state.frame_cr,
            mb_width=mb_width,
            mb_height=mb_height,
            mb_qps=self.state.mb_qps,
            mb_types=self.state.mb_types,
            mb_coeffs=self.state.mb_coeffs,
            alpha_offset=slice_header.alpha_offset,
            beta_offset=slice_header.beta_offset,
            disable_deblocking_filter_idc=slice_header.disable_deblocking_filter_idc,
            mb_slice_ids=self.state.mb_slice_ids,
            chroma_qp_index_offset=chroma_qp_offset,
            mv_cache=self.state.mv_cache,
            mv_cache_l1=self.state.mv_cache_l1,
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
        weighted_bipred_idc = getattr(pps, 'weighted_bipred_idc', 0)

        # Build L0/L1 reference lists before decoding B-slice
        self._build_b_slice_ref_lists(slice_header)

        while mb_idx < total_mbs:
            # Check for end of data before reading mb_skip_run
            if not reader.more_rbsp_data():
                while mb_idx < total_mbs:
                    mb_x = mb_idx % mb_width
                    mb_y = mb_idx // mb_width
                    self._process_b_skip_run(
                        mb_x, mb_y, use_spatial, current_poc,
                        weighted_bipred_idc,
                        current_qp=current_qp,
                    )
                    mb_idx += 1
                break

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
                    mb_x, mb_y, use_spatial, current_poc,
                    weighted_bipred_idc,
                    current_qp=current_qp,
                )
                mb_idx += 1

            # After skip run, check for end of slice (H.264 7.3.4)
            if mb_idx >= total_mbs:
                break
            if not reader.more_rbsp_data():
                while mb_idx < total_mbs:
                    mb_x = mb_idx % mb_width
                    mb_y = mb_idx // mb_width
                    self._process_b_skip_run(
                        mb_x, mb_y, use_spatial, current_poc,
                        weighted_bipred_idc,
                        current_qp=current_qp,
                    )
                    mb_idx += 1
                break

            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Read mb_type
            mb_type_code = reader.read_ue()
            mb_info = parse_b_mb_type(mb_type_code)

            logger.debug(f"MB ({mb_x}, {mb_y}): type={mb_info.name}")

            if mb_info.is_intra:
                # I-macroblock in B-slice
                # mb_type_code >= 23 means intra; intra type = mb_type_code - 23
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
                    frame_nz_counts=self.state.nz_counts,
                    frame_width_mbs=mb_width,
                    frame_intra_modes=self.state.intra_modes,
                    pre_parsed_mb_type=mb_type_code - 23,
                )
                # Update running QP for next MB
                current_qp = (current_qp + mb_data.mb_qp_delta + 52) % 52

                # Store nz_counts for CAVLC context of neighboring MBs
                mb_idx_b = mb_y * mb_width + mb_x
                self.state.nz_counts[mb_idx_b] = mb_data.nz_counts

                # Mark intra MB in MV caches
                if self.state.mv_cache is not None:
                    self.state.mv_cache.mark_intra(mb_x, mb_y)
                if self.state.mv_cache_l1 is not None:
                    self.state.mv_cache_l1.mark_intra(mb_x, mb_y)

                # Store deblocking metadata
                if self.state.mb_types is not None:
                    self.state.mb_types[mb_idx_b] = mb_data.mb_type
                if self.state.mb_qps is not None:
                    self.state.mb_qps[mb_idx_b] = current_qp
                if self.state.mb_coeffs is not None:
                    self.state.mb_coeffs[mb_idx_b, :16] = (
                        mb_data.nz_counts[:16] > 0
                    )
            else:
                # B-macroblock
                current_qp = self._decode_b_macroblock(
                    reader, mb_info, mb_x, mb_y, current_qp,
                    sps, pps, use_spatial, current_poc,
                    slice_header,
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

        # Store slice header for B-frame motion compensation access
        self.state.current_slice_header = slice_header

        # Build reference lists for P/B slices
        if is_p_slice:
            max_frame_num = getattr(sps, 'max_frame_num', 256)
            mod_l0 = getattr(slice_header, 'ref_pic_list_modification_l0', None)
            self.state.ref_buffer.build_p_slice_ref_list(
                slice_header.frame_num, max_frame_num, mod_l0
            )
        elif is_b_slice:
            self._build_b_slice_ref_lists(slice_header)

        current_qp = slice_qp
        mb_idx = slice_header.first_mb_in_slice

        # Reset intra prediction modes at start of new frame
        # Prevents stale modes from previous frames influencing MPM
        if mb_idx == 0 and self.state.intra_modes is not None:
            self.state.intra_modes[:] = 2

        while mb_idx < total_mbs:
            mb_x = mb_idx % mb_width
            mb_y = mb_idx // mb_width

            # Build neighbor info for context derivation
            mb_info = self._build_cabac_mb_info(mb_x, mb_y, mb_width, sps, pps)

            # Add reference count info for P/B slices
            if is_p_slice or is_b_slice:
                mb_info['num_ref_idx_l0_active'] = getattr(
                    slice_header, 'num_ref_idx_l0_active_minus1', 0
                ) + 1
            if is_b_slice:
                mb_info['num_ref_idx_l1_active'] = getattr(
                    slice_header, 'num_ref_idx_l1_active_minus1', 0
                ) + 1

            # Decode macroblock syntax using CABAC
            try:
                mb_data = decode_macroblock_layer_cabac(
                    cabac, contexts, slice_type, mb_info
                )
            except Exception as exc:
                msg = str(exc)
                if "Cannot read" in msg or "Needed a length" in msg:
                    logger.warning(
                        "CABAC bitstream ended early while decoding MB "
                        f"({mb_x}, {mb_y}); stopping slice decode."
                    )
                    break
                raise

            logger.debug(
                f"CABAC MB ({mb_x}, {mb_y}): skip={mb_data.get('mb_skip_flag', 0)}, "
                f"type={mb_data.get('mb_type', 0)}, cbp={mb_data.get('cbp', 0):#x}, "
                f"left_cbp={mb_info.get('left_cbp', -1)}, top_cbp={mb_info.get('top_cbp', -1)}"
            )

            # Process the decoded macroblock
            try:
                self._process_cabac_macroblock(
                    cabac, contexts, mb_data, mb_x, mb_y,
                    current_qp, sps, pps, slice_type
                )
            except Exception as exc:
                msg = str(exc)
                if "Cannot read" in msg or "Needed a length" in msg:
                    logger.warning(
                        "CABAC macroblock decode failed at MB "
                        f"({mb_x}, {mb_y}); stopping slice decode."
                    )
                    break
                raise

            # Update QP
            current_qp = (current_qp + mb_data.get('mb_qp_delta', 0) + 52) % 52

            # Check for end of slice
            try:
                if decode_end_of_slice_flag_cabac(cabac, contexts) == 1:
                    break
            except Exception as exc:
                msg = str(exc)
                if "Cannot read" in msg or "Needed a length" in msg:
                    logger.warning(
                        "CABAC end_of_slice_flag read failed at MB "
                        f"({mb_x}, {mb_y}); stopping slice decode."
                    )
                    break
                raise

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

        # Get neighbor MB types, CBPs, chroma modes, qp deltas
        left_mb_type = None
        top_mb_type = None
        left_cbp = -1  # -1 = unavailable
        top_cbp = -1
        left_chroma_mode = 0
        top_chroma_mode = 0
        left_qp_delta = 0
        top_qp_delta = 0

        # Use cabac_mb_types for CABAC context derivation (raw types,
        # not overwritten to 99 by inter reconstruction).
        type_array = (self.state.cabac_mb_types
                      if self.state.cabac_mb_types is not None
                      else self.state.mb_types)

        if mb_x > 0 and type_array is not None:
            left_idx = mb_idx - 1
            if left_idx >= 0:
                left_mb_type = int(type_array[left_idx])
                if self.state.mb_cbps is not None:
                    left_cbp = int(self.state.mb_cbps[left_idx])
                if self.state.mb_chroma_modes is not None:
                    left_chroma_mode = int(self.state.mb_chroma_modes[left_idx])
                if self.state.mb_qp_deltas is not None:
                    left_qp_delta = int(self.state.mb_qp_deltas[left_idx])

        if mb_y > 0 and type_array is not None:
            top_idx = mb_idx - mb_width
            if top_idx >= 0:
                top_mb_type = int(type_array[top_idx])
                if self.state.mb_cbps is not None:
                    top_cbp = int(self.state.mb_cbps[top_idx])
                if self.state.mb_chroma_modes is not None:
                    top_chroma_mode = int(self.state.mb_chroma_modes[top_idx])
                if self.state.mb_qp_deltas is not None:
                    top_qp_delta = int(self.state.mb_qp_deltas[top_idx])

        # Determine prev_mb_qp_delta for context derivation (H.264 9.3.3.1.1.10)
        # Must be the previously decoded MB in scan order (mb_idx - 1), not just left neighbor.
        # For first MB of slice, prevMbQpDelta = 0.
        prev_mb_qp_delta = 0
        if mb_idx > 0 and self.state.mb_qp_deltas is not None:
            prev_mb_qp_delta = int(self.state.mb_qp_deltas[mb_idx - 1])

        # Build per-block CBF lists for coded_block_flag context
        # CBF = 1 if nz_count > 0, else 0
        # Format: list[27] = [16 luma, 4 Cb AC, 4 Cr AC, luma DC, Cb DC, Cr DC]
        left_mb_cbf = None
        top_mb_cbf = None

        if mb_x > 0 and self.state.nz_counts is not None:
            left_idx = mb_idx - 1
            if left_idx >= 0:
                nz = self.state.nz_counts[left_idx]
                left_mb_cbf = [1 if nz[i] > 0 else 0 for i in range(24)]
                dc = self.state.mb_dc_cbf[left_idx]
                left_mb_cbf.extend([int(dc[0]), int(dc[1]), int(dc[2])])

        if mb_y > 0 and self.state.nz_counts is not None:
            top_idx = mb_idx - mb_width
            if top_idx >= 0:
                nz = self.state.nz_counts[top_idx]
                top_mb_cbf = [1 if nz[i] > 0 else 0 for i in range(24)]
                dc = self.state.mb_dc_cbf[top_idx]
                top_mb_cbf.extend([int(dc[0]), int(dc[1]), int(dc[2])])

        # Get neighbor skip flags for skip context derivation
        left_skip = False
        top_skip = False
        if mb_x > 0 and self.state.mb_skip_flags is not None:
            left_idx = mb_idx - 1
            if left_idx >= 0:
                left_skip = bool(self.state.mb_skip_flags[left_idx])
        if mb_y > 0 and self.state.mb_skip_flags is not None:
            top_idx = mb_idx - mb_width
            if top_idx >= 0:
                top_skip = bool(self.state.mb_skip_flags[top_idx])

        # Get neighbor MVD arrays for MVD context derivation (H.264 9.3.3.1.1.7)
        left_mvds_l0 = None
        top_mvds_l0 = None
        left_mvds_l1 = None
        top_mvds_l1 = None
        if mb_x > 0 and self.state.mb_mvds_l0 is not None:
            left_idx = mb_idx - 1
            left_mvds_l0 = self.state.mb_mvds_l0[left_idx]
            if self.state.mb_mvds_l1 is not None:
                left_mvds_l1 = self.state.mb_mvds_l1[left_idx]
        if mb_y > 0 and self.state.mb_mvds_l0 is not None:
            top_idx = mb_idx - mb_width
            top_mvds_l0 = self.state.mb_mvds_l0[top_idx]
            if self.state.mb_mvds_l1 is not None:
                top_mvds_l1 = self.state.mb_mvds_l1[top_idx]

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
            'left_chroma_mode': left_chroma_mode,
            'top_chroma_mode': top_chroma_mode,
            'prev_mb_qp_delta': prev_mb_qp_delta,
            'transform_8x8_mode_flag': getattr(pps, 'transform_8x8_mode_flag', 0),
            'left_mb_cbf': left_mb_cbf,
            'top_mb_cbf': top_mb_cbf,
            'left_skip': left_skip,
            'top_skip': top_skip,
            'left_mb_mvds_l0': left_mvds_l0,
            'top_mb_mvds_l0': top_mvds_l0,
            'left_mb_mvds_l1': left_mvds_l1,
            'top_mb_mvds_l1': top_mvds_l1,
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

        # Store MB type, CBP, chroma mode, qp delta, and skip flag
        if self.state.mb_types is not None:
            self.state.mb_types[mb_idx] = mb_data.get('mb_type', 0)
        # Preserve raw CABAC mb_type for context derivation (not overwritten)
        if self.state.cabac_mb_types is not None:
            self.state.cabac_mb_types[mb_idx] = mb_data.get('mb_type', 0)
        if self.state.mb_cbps is not None:
            self.state.mb_cbps[mb_idx] = mb_data.get('cbp', 0)
        if self.state.mb_chroma_modes is not None:
            pred = mb_data.get('mb_pred', {})
            self.state.mb_chroma_modes[mb_idx] = pred.get('intra_chroma_pred_mode', 0) if pred else 0
        if self.state.mb_qp_deltas is not None:
            self.state.mb_qp_deltas[mb_idx] = mb_data.get('mb_qp_delta', 0)
        if self.state.mb_skip_flags is not None:
            self.state.mb_skip_flags[mb_idx] = mb_data.get('mb_skip_flag', 0)

        # Store DC coded_block_flag for CABAC neighbor context
        # Always clear first to prevent stale values from previous frame
        if self.state.mb_dc_cbf is not None:
            self.state.mb_dc_cbf[mb_idx] = 0
        residual = mb_data.get('residual', {})
        mb_cbf = residual.get('_mb_cbf', None) if residual else None
        if mb_cbf is not None and self.state.mb_dc_cbf is not None and len(mb_cbf) >= 27:
            self.state.mb_dc_cbf[mb_idx, 0] = mb_cbf[24]  # luma DC
            self.state.mb_dc_cbf[mb_idx, 1] = mb_cbf[25]  # Cb DC
            self.state.mb_dc_cbf[mb_idx, 2] = mb_cbf[26]  # Cr DC

        # Store MVD grids for CABAC neighbor context (H.264 9.3.3.1.1.7)
        pred = mb_data.get('mb_pred', {})
        if pred:
            if self.state.mb_mvds_l0 is not None:
                mvd_grid = pred.get('mvd_grid_l0')
                if mvd_grid is not None:
                    self.state.mb_mvds_l0[mb_idx] = mvd_grid
                else:
                    self.state.mb_mvds_l0[mb_idx] = 0
            if self.state.mb_mvds_l1 is not None:
                mvd_grid = pred.get('mvd_grid_l1')
                if mvd_grid is not None:
                    self.state.mb_mvds_l1[mb_idx] = mvd_grid
                else:
                    self.state.mb_mvds_l1[mb_idx] = 0
        else:
            if self.state.mb_mvds_l0 is not None:
                self.state.mb_mvds_l0[mb_idx] = 0
            if self.state.mb_mvds_l1 is not None:
                self.state.mb_mvds_l1[mb_idx] = 0

        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        # Handle skip macroblock
        if mb_data.get('mb_skip_flag', 0) == 1:
            if self.state.mb_types is not None:
                self.state.mb_types[mb_idx] = 99  # inter sentinel for deblocking
            if self.state.mb_qps is not None:
                self.state.mb_qps[mb_idx] = qp  # skip uses current QP
            # Clear nz_counts, dc_cbf, and coeffs for skip MBs (no residual)
            self.state.nz_counts[mb_idx] = 0
            if self.state.mb_dc_cbf is not None:
                self.state.mb_dc_cbf[mb_idx] = 0
            if self.state.mb_coeffs is not None:
                self.state.mb_coeffs[mb_idx] = 0
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
            sh = getattr(self.state, 'current_slice_header', None)
            use_spatial = getattr(sh, 'direct_spatial_mv_pred_flag', True)
            current_poc = getattr(sh, 'pic_order_cnt_lsb', 0)
            luma, cb, cr = reconstruct_b_skip(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                mv_cache_l1=getattr(self.state, 'mv_cache_l1', None),
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
        """Reconstruct an intra macroblock from CABAC-parsed syntax.

        Uses pre-parsed coefficients from decode_macroblock_layer_cabac and
        feeds them into the shared reconstruction pipeline (same as CAVLC).

        H.264 Spec Reference: Section 7.3.5, 8.3, 8.5

        Args:
            cabac: CABAC decoder (not used - syntax already parsed)
            contexts: Context models (not used - syntax already parsed)
            mb_data: Decoded MB syntax from decode_macroblock_layer_cabac
            mb_x, mb_y: Macroblock position
            qp: Current QP (before mb_qp_delta)
            sps, pps: Parameter sets
            slice_type: Slice type (0=P, 1=B, 2=I)
        """
        from reconstruct.macroblock import (
            decode_i16x16_mb_type,
            reconstruct_i16x16_from_coeffs,
            process_chroma_dc_coeffs,
            reconstruct_chroma_plane,
            _get_4x4_block_position,
            _get_i4x4_neighbors,
            BLOCK_SCAN_ORDER,
            BLOCK_IDX_FROM_POS,
            _find_block_at_position,
        )
        from entropy.cavlc import CAVLCBlock
        from intra.intra_4x4 import predict_intra_4x4
        from dequant.dequant import get_chroma_qp

        mb_width = sps.pic_width_in_mbs
        mb_idx = mb_y * mb_width + mb_x
        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        # Get MB type relative to I-slice
        mb_type = mb_data.get('mb_type', 0)
        if slice_type == 0:  # P-slice
            mb_type = mb_type - 5
        elif slice_type == 1:  # B-slice
            mb_type = mb_type - 23

        # Apply QP delta
        mb_qp_delta = mb_data.get('mb_qp_delta', 0)
        current_qp = (qp + mb_qp_delta + 52) % 52

        # Chroma QP
        chroma_qp_offset = getattr(pps, 'chroma_qp_index_offset', 0)
        qpi = max(0, min(51, current_qp + chroma_qp_offset))
        qp_chroma = get_chroma_qp(qpi)

        # Get residual from pre-parsed data
        residual = mb_data.get('residual', {})
        mb_pred = mb_data.get('mb_pred', {})
        chroma_pred_mode = mb_pred.get('intra_chroma_pred_mode', 0)

        # Initialize nz_counts for this MB
        nz_counts = np.zeros(24, dtype=np.int32)

        # Handle I_PCM
        if mb_type == 25:
            i_pcm = mb_data.get('i_pcm', {})
            luma_samples = i_pcm.get('luma_samples', [128] * 256)
            cb_samples = i_pcm.get('cb_samples', [128] * 64)
            cr_samples = i_pcm.get('cr_samples', [128] * 64)
            self.state.frame_luma[ly:ly+16, lx:lx+16] = np.array(
                luma_samples, dtype=np.uint8
            ).reshape(16, 16)
            self.state.frame_cb[cy:cy+8, cx:cx+8] = np.array(
                cb_samples, dtype=np.uint8
            ).reshape(8, 8)
            self.state.frame_cr[cy:cy+8, cx:cx+8] = np.array(
                cr_samples, dtype=np.uint8
            ).reshape(8, 8)
            # I_PCM: all nz_counts = 16 (forces nC = 16 for neighbors)
            nz_counts[:] = 16
            self.state.nz_counts[mb_idx] = nz_counts
            if self.state.mb_qps is not None:
                self.state.mb_qps[mb_idx] = current_qp
            if self.state.mb_coeffs is not None:
                self.state.mb_coeffs[mb_idx, :16] = True
            return

        # Get neighbor pixels for prediction
        neighbors_top_luma = (
            self.state.frame_luma[ly - 1, lx:lx + 16] if mb_y > 0 else None
        )
        neighbors_left_luma = (
            self.state.frame_luma[ly:ly + 16, lx - 1] if mb_x > 0 else None
        )
        neighbor_tl_luma = (
            self.state.frame_luma[ly - 1, lx - 1]
            if mb_y > 0 and mb_x > 0 else None
        )

        if 1 <= mb_type <= 24:
            # --- I_16x16 ---
            pred_mode, cbp_luma, cbp_chroma = decode_i16x16_mb_type(mb_type)

            # Get DC and AC coefficients from pre-parsed residual
            dc_coeffs_raw = residual.get('luma_dc')
            if dc_coeffs_raw is None:
                dc_coeffs_raw = np.zeros(16, dtype=np.int32)
            else:
                dc_coeffs_raw = np.asarray(dc_coeffs_raw, dtype=np.int32)

            ac_list = []
            for i in range(16):
                ac_raw = residual.get('luma_ac', [None] * 16)[i]
                if ac_raw is not None:
                    ac_list.append(np.asarray(ac_raw, dtype=np.int32))
                else:
                    ac_list.append(None)

            # Reconstruct luma using shared pipeline
            luma_block = reconstruct_i16x16_from_coeffs(
                pred_mode, cbp_luma, current_qp,
                dc_coeffs_raw, ac_list,
                neighbors_top_luma, neighbors_left_luma, neighbor_tl_luma,
            )

            # Write luma to frame
            self.state.frame_luma[ly:ly+16, lx:lx+16] = luma_block

            # Update nz_counts for luma AC blocks
            for i in range(16):
                if ac_list[i] is not None:
                    nz_counts[i] = int(np.count_nonzero(ac_list[i]))

            # Store intra_modes as DC default (I_16x16 is not I_4x4)
            if self.state.intra_modes is not None:
                self.state.intra_modes[mb_idx, :] = 2  # DC default

        elif mb_type == 0:
            # --- I_4x4 ---
            cbp = mb_data.get('cbp', 0)
            cbp_luma = cbp & 0x0F
            cbp_chroma = (cbp >> 4) & 0x03

            # Derive actual prediction modes from CABAC-parsed flags
            # mb_pred['intra_pred_modes'] has -1 for "use predicted" or
            # rem_intra4x4_pred_mode value
            raw_modes = mb_pred.get('intra_pred_modes', [])
            actual_modes = []
            decoded_modes = {}

            for block_idx in range(16):
                row, col = BLOCK_SCAN_ORDER[block_idx]

                # Find left neighbor mode (H.264 Section 8.3.1.1)
                if col > 0:
                    left_idx = _find_block_at_position(row, col - 4)
                    mode_a = decoded_modes.get(left_idx, -1)
                elif mb_x > 0 and self.state.intra_modes is not None:
                    left_mb_idx = mb_y * mb_width + (mb_x - 1)
                    left_block_idx = BLOCK_IDX_FROM_POS.get((row // 4, 3))
                    if left_block_idx is not None:
                        mode_a = int(
                            self.state.intra_modes[left_mb_idx, left_block_idx]
                        )
                    else:
                        mode_a = -1
                else:
                    mode_a = -1

                # Find top neighbor mode
                if row > 0:
                    top_idx = _find_block_at_position(row - 4, col)
                    mode_b = decoded_modes.get(top_idx, -1)
                elif mb_y > 0 and self.state.intra_modes is not None:
                    top_mb_idx = (mb_y - 1) * mb_width + mb_x
                    top_block_idx = BLOCK_IDX_FROM_POS.get((3, col // 4))
                    if top_block_idx is not None:
                        mode_b = int(
                            self.state.intra_modes[top_mb_idx, top_block_idx]
                        )
                    else:
                        mode_b = -1
                else:
                    mode_b = -1

                # MPM derivation
                if mode_a < 0 or mode_b < 0:
                    predicted_mode = 2  # DC
                else:
                    predicted_mode = min(mode_a, mode_b)

                # Apply CABAC-parsed flag/rem
                cabac_mode = raw_modes[block_idx] if block_idx < len(raw_modes) else 0
                if cabac_mode == -1:
                    # prev_intra4x4_pred_mode_flag == 1 -> use MPM
                    mode = predicted_mode
                else:
                    # prev_intra4x4_pred_mode_flag == 0 -> rem value
                    rem = cabac_mode
                    if rem < predicted_mode:
                        mode = rem
                    else:
                        mode = rem + 1

                actual_modes.append(mode)
                decoded_modes[block_idx] = mode

            # Reconstruct each 4x4 block in scan order
            mb_luma = np.zeros((16, 16), dtype=np.uint8)

            for block_idx in range(16):
                row, col = BLOCK_SCAN_ORDER[block_idx]
                mode = actual_modes[block_idx]

                # Get neighbors from frame buffer + partially reconstructed MB
                top, left, top_left, top_right = _get_i4x4_neighbors(
                    self.state.frame_luma, mb_luma,
                    mb_x, mb_y, row, col, block_idx
                )

                # Generate prediction
                pred = predict_intra_4x4(
                    mode, top, left, top_left, top_right,
                    top_available=(top is not None),
                    left_available=(left is not None),
                    top_right_available=(top_right is not None),
                )

                # Get residual from pre-parsed coefficients
                block_8x8_idx = block_idx // 4
                luma_4x4_raw = residual.get('luma_4x4', [None] * 16)
                coeffs_raw = luma_4x4_raw[block_idx]

                if (cbp_luma & (1 << block_8x8_idx)) and coeffs_raw is not None:
                    coeffs_arr = np.asarray(coeffs_raw, dtype=np.int32)
                    nz_counts[block_idx] = int(np.count_nonzero(coeffs_arr))

                    # Zigzag to raster, dequant, IDCT
                    coeffs_2d = np.zeros((4, 4), dtype=np.int32)
                    for scan_idx in range(16):
                        raster_pos = ZIGZAG_4x4[scan_idx]
                        r, c = raster_pos // 4, raster_pos % 4
                        coeffs_2d[r, c] = coeffs_arr[scan_idx]
                    dequant = dequant_4x4(coeffs_2d, current_qp)
                    block_residual = idct_4x4(dequant)
                else:
                    block_residual = np.zeros((4, 4), dtype=np.int32)
                    nz_counts[block_idx] = 0

                # Add prediction + residual, clip, store
                result = pred.astype(np.int32) + block_residual
                result = np.clip(result, 0, 255).astype(np.uint8)
                mb_luma[row:row+4, col:col+4] = result

                # Write to frame immediately for neighbor prediction
                abs_y = ly + row
                abs_x = lx + col
                self.state.frame_luma[abs_y:abs_y+4, abs_x:abs_x+4] = result

            # Store intra modes for cross-MB MPM
            if self.state.intra_modes is not None:
                for block_idx in range(16):
                    self.state.intra_modes[mb_idx, block_idx] = actual_modes[block_idx]

        else:
            # Unexpected type - fill with mid-gray
            logger.warning(
                f"Unexpected intra mb_type={mb_type} at ({mb_x}, {mb_y})"
            )
            self.state.frame_luma[ly:ly+16, lx:lx+16] = 128
            self.state.frame_cb[cy:cy+8, cx:cx+8] = 128
            self.state.frame_cr[cy:cy+8, cx:cx+8] = 128
            self.state.nz_counts[mb_idx] = nz_counts
            if self.state.mb_qps is not None:
                self.state.mb_qps[mb_idx] = current_qp
            return

        # --- Chroma reconstruction (shared for I_16x16 and I_4x4) ---
        neighbors_top_cb = (
            self.state.frame_cb[cy - 1, cx:cx + 8] if mb_y > 0 else None
        )
        neighbors_left_cb = (
            self.state.frame_cb[cy:cy + 8, cx - 1] if mb_x > 0 else None
        )
        neighbor_tl_cb = (
            self.state.frame_cb[cy - 1, cx - 1]
            if mb_y > 0 and mb_x > 0 else None
        )
        neighbors_top_cr = (
            self.state.frame_cr[cy - 1, cx:cx + 8] if mb_y > 0 else None
        )
        neighbors_left_cr = (
            self.state.frame_cr[cy:cy + 8, cx - 1] if mb_x > 0 else None
        )
        neighbor_tl_cr = (
            self.state.frame_cr[cy - 1, cx - 1]
            if mb_y > 0 and mb_x > 0 else None
        )

        # Process chroma DC from pre-parsed coefficients
        cb_dc_raw = residual.get('chroma_dc_cb')
        cr_dc_raw = residual.get('chroma_dc_cr')

        cb_dc_dequant = None
        cr_dc_dequant = None
        if cb_dc_raw is not None:
            cb_dc_dequant = process_chroma_dc_coeffs(
                np.asarray(cb_dc_raw, dtype=np.int32), qp_chroma
            )
        if cr_dc_raw is not None:
            cr_dc_dequant = process_chroma_dc_coeffs(
                np.asarray(cr_dc_raw, dtype=np.int32), qp_chroma
            )

        # Wrap chroma AC coefficients in CAVLCBlock adapter objects
        cb_ac_blocks = []
        cr_ac_blocks = []
        chroma_ac_cb = residual.get('chroma_ac_cb', [None] * 4)
        chroma_ac_cr = residual.get('chroma_ac_cr', [None] * 4)

        for i in range(4):
            cb_raw = chroma_ac_cb[i]
            if cb_raw is not None:
                cb_arr = np.asarray(cb_raw, dtype=np.int32)
                tc = int(np.count_nonzero(cb_arr))
                cb_ac_blocks.append(
                    CAVLCBlock(
                        total_coeff=tc,
                        trailing_ones=0,
                        coefficients=cb_arr,
                    )
                )
                nz_counts[16 + i] = tc
            else:
                cb_ac_blocks.append(
                    CAVLCBlock(
                        total_coeff=0,
                        trailing_ones=0,
                        coefficients=np.zeros(15, dtype=np.int32),
                    )
                )

            cr_raw = chroma_ac_cr[i]
            if cr_raw is not None:
                cr_arr = np.asarray(cr_raw, dtype=np.int32)
                tc = int(np.count_nonzero(cr_arr))
                cr_ac_blocks.append(
                    CAVLCBlock(
                        total_coeff=tc,
                        trailing_ones=0,
                        coefficients=cr_arr,
                    )
                )
                nz_counts[20 + i] = tc
            else:
                cr_ac_blocks.append(
                    CAVLCBlock(
                        total_coeff=0,
                        trailing_ones=0,
                        coefficients=np.zeros(15, dtype=np.int32),
                    )
                )

        # Determine effective cbp_chroma for reconstruction
        if mb_type >= 1 and mb_type <= 24:
            # I_16x16: cbp_chroma already extracted above
            pass
        else:
            # I_4x4: from parsed CBP
            pass

        # Only pass ac_blocks when cbp_chroma == 2 (DC+AC)
        cb_ac_for_recon = cb_ac_blocks if cbp_chroma == 2 else []
        cr_ac_for_recon = cr_ac_blocks if cbp_chroma == 2 else []

        # Reconstruct Cb
        self.state.frame_cb[cy:cy+8, cx:cx+8] = reconstruct_chroma_plane(
            cb_dc_dequant, cb_ac_for_recon, cbp_chroma, qp_chroma,
            chroma_pred_mode,
            neighbors_top_cb, neighbors_left_cb, neighbor_tl_cb,
        )

        # Reconstruct Cr
        self.state.frame_cr[cy:cy+8, cx:cx+8] = reconstruct_chroma_plane(
            cr_dc_dequant, cr_ac_for_recon, cbp_chroma, qp_chroma,
            chroma_pred_mode,
            neighbors_top_cr, neighbors_left_cr, neighbor_tl_cr,
        )

        # --- Store metadata ---
        self.state.nz_counts[mb_idx] = nz_counts

        if self.state.mb_qps is not None:
            self.state.mb_qps[mb_idx] = current_qp

        if self.state.mb_coeffs is not None:
            self.state.mb_coeffs[mb_idx, :16] = nz_counts[:16] > 0

        # Mark intra in MV cache so neighbors see ref=-1 (not unavailable)
        if self.state.mv_cache is not None:
            self.state.mv_cache.mark_intra(mb_x, mb_y)
        if self.state.mv_cache_l1 is not None:
            self.state.mv_cache_l1.mark_intra(mb_x, mb_y)

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

        Uses pre-parsed syntax from decode_macroblock_layer_cabac:
        mb_data contains mb_type, mb_pred (ref_idx, MVD), cbp, residual.

        Args:
            cabac: CABAC decoder (not used - syntax already parsed)
            contexts: Context models (not used)
            mb_data: Pre-parsed MB syntax
            mb_x, mb_y: Macroblock position
            qp: Current QP (before mb_qp_delta)
            sps, pps: Parameter sets
            slice_type: 0=P, 1=B
        """
        from dequant.dequant import get_chroma_qp

        mb_width = sps.pic_width_in_mbs
        mb_idx = mb_y * mb_width + mb_x
        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        mb_type = mb_data['mb_type']
        mb_pred = mb_data.get('mb_pred', {})
        residual = mb_data.get('residual', {})
        cbp = mb_data.get('cbp', 0)
        cbp_luma = cbp & 0x0F
        cbp_chroma = (cbp >> 4) & 0x03

        # Apply QP delta
        mb_qp_delta = mb_data.get('mb_qp_delta', 0)
        current_qp = (qp + mb_qp_delta + 52) % 52

        # --- Motion compensation ---
        if slice_type == 0:
            luma_pred, cb_pred, cr_pred = self._cabac_p_motion_comp(
                mb_type, mb_pred, mb_x, mb_y
            )
        else:
            luma_pred, cb_pred, cr_pred = self._cabac_b_motion_comp(
                mb_type, mb_pred, mb_x, mb_y
            )

        # --- Luma residual ---
        nz_counts = np.zeros(24, dtype=np.int32)
        luma_residual = np.zeros((16, 16), dtype=np.int32)

        luma_4x4_raw = residual.get('luma_4x4', [None] * 16)
        for block_idx in range(16):
            block_8x8_idx = block_idx // 4
            if not (cbp_luma & (1 << block_8x8_idx)):
                continue
            coeffs_raw = luma_4x4_raw[block_idx]
            if coeffs_raw is None:
                continue
            coeffs_arr = np.asarray(coeffs_raw, dtype=np.int32)
            nz_counts[block_idx] = int(np.count_nonzero(coeffs_arr))
            if nz_counts[block_idx] > 0:
                coeffs_2d = np.zeros((4, 4), dtype=np.int32)
                for scan_idx in range(16):
                    raster_pos = ZIGZAG_4x4[scan_idx]
                    r, c = raster_pos // 4, raster_pos % 4
                    coeffs_2d[r, c] = coeffs_arr[scan_idx]
                dequant = dequant_4x4(coeffs_2d, current_qp)
                block_residual = idct_4x4(dequant)
                row, col = BLOCK_SCAN_ORDER[block_idx]
                luma_residual[row:row+4, col:col+4] = block_residual

        # --- Chroma residual ---
        chroma_qp_offset = getattr(pps, 'chroma_qp_index_offset', 0)
        qpi = max(0, min(51, current_qp + chroma_qp_offset))
        qp_chroma = get_chroma_qp(qpi)

        cb_residual = self._cabac_chroma_residual(
            residual, 'cb', cbp_chroma, qp_chroma, nz_counts, 16
        )
        cr_residual = self._cabac_chroma_residual(
            residual, 'cr', cbp_chroma, qp_chroma, nz_counts, 20
        )

        # --- Combine prediction + residual ---
        luma = np.clip(
            luma_pred.astype(np.int32) + luma_residual, 0, 255
        ).astype(np.uint8)
        cb = np.clip(
            cb_pred.astype(np.int32) + cb_residual, 0, 255
        ).astype(np.uint8)
        cr = np.clip(
            cr_pred.astype(np.int32) + cr_residual, 0, 255
        ).astype(np.uint8)

        self.state.frame_luma[ly:ly+16, lx:lx+16] = luma
        self.state.frame_cb[cy:cy+8, cx:cx+8] = cb
        self.state.frame_cr[cy:cy+8, cx:cx+8] = cr

        # --- Store metadata ---
        self.state.nz_counts[mb_idx] = nz_counts
        if self.state.mb_types is not None:
            self.state.mb_types[mb_idx] = 99  # Non-intra sentinel
        if self.state.mb_qps is not None:
            self.state.mb_qps[mb_idx] = current_qp
        if self.state.mb_coeffs is not None:
            self.state.mb_coeffs[mb_idx, :16] = (nz_counts[:16] > 0)

    def _cabac_p_motion_comp(
        self, mb_type: int, mb_pred: dict, mb_x: int, mb_y: int,
    ) -> tuple:
        """P-slice motion compensation from pre-parsed CABAC syntax."""
        ref_idx_l0 = mb_pred.get('ref_idx_l0', [])
        mvd_l0 = mb_pred.get('mvd_l0', [])
        sub_types = mb_pred.get('sub_mb_types', [])

        if mb_type == 0:  # P_L0_16x16
            ref = ref_idx_l0[0] if ref_idx_l0 else 0
            mvd = mvd_l0[0] if mvd_l0 else (0, 0)
            mvp_x, mvp_y = predict_mv_16x16(
                self.state.mv_cache, mb_x, mb_y, target_ref=ref,
            )
            mvx, mvy = mvp_x + mvd[0], mvp_y + mvd[1]
            if self.state.mv_cache is not None:
                self.state.mv_cache.set_mv_16x16(mb_x, mb_y, mvx, mvy, ref_idx=ref)
            return reconstruct_p_16x16(
                ref_buffer=self.state.ref_buffer, ref_idx=ref,
                mvx=mvx, mvy=mvy,
                residual_luma=None, residual_cb=None, residual_cr=None,
                mb_x=mb_x, mb_y=mb_y,
            )

        elif mb_type == 1:  # P_L0_L0_16x8
            refs = [ref_idx_l0[i] if i < len(ref_idx_l0) else 0 for i in range(2)]
            mvds = [mvd_l0[i] if i < len(mvd_l0) else (0, 0) for i in range(2)]
            mvxs, mvys = [], []
            for part in range(2):
                mvp_x, mvp_y = predict_mv_16x8(
                    self.state.mv_cache, mb_x, mb_y, part, target_ref=refs[part],
                )
                mvxs.append(mvp_x + mvds[part][0])
                mvys.append(mvp_y + mvds[part][1])
                self.state.mv_cache.set_mv_16x8(
                    mb_x, mb_y, part, mvxs[part], mvys[part], ref_idx=refs[part],
                )
            return reconstruct_p_16x8(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                ref_idx=refs, mvx=mvxs, mvy=mvys,
                residual_luma=None, residual_cb=None, residual_cr=None,
                mb_x=mb_x, mb_y=mb_y,
            )

        elif mb_type == 2:  # P_L0_L0_8x16
            refs = [ref_idx_l0[i] if i < len(ref_idx_l0) else 0 for i in range(2)]
            mvds = [mvd_l0[i] if i < len(mvd_l0) else (0, 0) for i in range(2)]
            mvxs, mvys = [], []
            for part in range(2):
                mvp_x, mvp_y = predict_mv_8x16(
                    self.state.mv_cache, mb_x, mb_y, part, target_ref=refs[part],
                )
                mvxs.append(mvp_x + mvds[part][0])
                mvys.append(mvp_y + mvds[part][1])
                self.state.mv_cache.set_mv_8x16(
                    mb_x, mb_y, part, mvxs[part], mvys[part], ref_idx=refs[part],
                )
            return reconstruct_p_8x16(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                ref_idx=refs, mvx=mvxs, mvy=mvys,
                residual_luma=None, residual_cb=None, residual_cr=None,
                mb_x=mb_x, mb_y=mb_y,
            )

        elif mb_type in (3, 4):  # P_8x8, P_8x8ref0
            refs = [ref_idx_l0[i] if i < len(ref_idx_l0) else 0 for i in range(4)]

            # Sub-partition definitions: (block_offset, part_width_4x4, coverage)
            _sp = {
                0: [((0, 0), 2, [(0,0),(1,0),(0,1),(1,1)])],
                1: [((0, 0), 2, [(0,0),(1,0)]),
                    ((0, 1), 2, [(0,1),(1,1)])],
                2: [((0, 0), 1, [(0,0),(0,1)]),
                    ((1, 0), 1, [(1,0),(1,1)])],
                3: [((0, 0), 1, [(0,0)]), ((1, 0), 1, [(1,0)]),
                    ((0, 1), 1, [(0,1)]), ((1, 1), 1, [(1,1)])],
            }
            # Chroma sub-partition geometry: (cx_off, cy_off, cw, ch)
            _csp = {
                0: [(0, 0, 4, 4)],
                1: [(0, 0, 4, 2), (0, 2, 4, 2)],
                2: [(0, 0, 2, 4), (2, 0, 2, 4)],
                3: [(0, 0, 2, 2), (2, 0, 2, 2), (0, 2, 2, 2), (2, 2, 2, 2)],
            }

            luma = np.zeros((16, 16), dtype=np.uint8)
            pred_cb = np.zeros((8, 8), dtype=np.uint8)
            pred_cr = np.zeros((8, 8), dtype=np.uint8)
            sub_mb_offsets = [(0, 0), (8, 0), (0, 8), (8, 8)]
            chroma_offsets = [(0, 0), (4, 0), (0, 4), (4, 4)]

            mvd_idx = 0
            for sub_idx in range(4):
                sub_type = sub_types[sub_idx] if sub_idx < len(sub_types) else 0
                sub_bx = (sub_idx % 2) * 2
                sub_by = (sub_idx // 2) * 2
                parts = _sp[min(sub_type, 3)]
                chroma_parts = _csp[min(sub_type, 3)]

                sub_mvs = []
                for j, ((dx, dy), pw, coverage) in enumerate(parts):
                    mvd = mvd_l0[mvd_idx] if mvd_idx < len(mvd_l0) else (0, 0)
                    mvd_idx += 1

                    mvp_x, mvp_y = predict_mv_partition(
                        self.state.mv_cache, mb_x, mb_y,
                        block_x=sub_bx + dx, block_y=sub_by + dy,
                        part_width_blocks=pw, target_ref=refs[sub_idx],
                    )
                    final_mvx = mvp_x + mvd[0]
                    final_mvy = mvp_y + mvd[1]
                    sub_mvs.append((final_mvx, final_mvy))

                    for (bx_off, by_off) in coverage:
                        self.state.mv_cache.set_mv(
                            mb_x, mb_y, sub_bx + bx_off, sub_by + by_off,
                            final_mvx, final_mvy, ref_idx=refs[sub_idx],
                        )

                # Luma
                sub_luma = reconstruct_p_8x8_sub(
                    ref_buffer=self.state.ref_buffer,
                    mv_cache=self.state.mv_cache,
                    ref_idx=refs[sub_idx], sub_mb_type=sub_type,
                    mvs=sub_mvs, mb_x=mb_x, mb_y=mb_y,
                    sub_idx=sub_idx, residual=None,
                )
                sx, sy = sub_mb_offsets[sub_idx]
                luma[sy:sy+8, sx:sx+8] = sub_luma

                # Chroma per sub-partition
                ref_frame = self.state.ref_buffer.get_frame(refs[sub_idx])
                cx_base, cy_base = chroma_offsets[sub_idx]
                for j, (cx_off, cy_off, cw, ch) in enumerate(chroma_parts):
                    cmvx, cmvy = sub_mvs[j]
                    cx = mb_x * 8 + cx_base + cx_off + (cmvx >> 3)
                    cy = mb_y * 8 + cy_base + cy_off + (cmvy >> 3)
                    fx, fy = cmvx & 7, cmvy & 7
                    dst_x, dst_y = cx_base + cx_off, cy_base + cy_off
                    pred_cb[dst_y:dst_y+ch, dst_x:dst_x+cw] = (
                        apply_chroma_prediction(ref_frame.cb, cx, cy, fx, fy, cw, ch)
                    )
                    pred_cr[dst_y:dst_y+ch, dst_x:dst_x+cw] = (
                        apply_chroma_prediction(ref_frame.cr, cx, cy, fx, fy, cw, ch)
                    )

            return luma, pred_cb, pred_cr

        else:
            # Unexpected P type — use skip
            logger.warning(f"Unexpected P mb_type={mb_type} at ({mb_x},{mb_y})")
            return reconstruct_p_skip(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                mb_x=mb_x, mb_y=mb_y,
            )

    def _cabac_b_motion_comp(
        self, mb_type: int, mb_pred: dict, mb_x: int, mb_y: int,
    ) -> tuple:
        """B-slice motion compensation from pre-parsed CABAC syntax.

        Handles all 23 B-MB types (0-22).
        """
        from inter.direct_mode import (
            derive_direct_spatial_sub8x8,
            derive_direct_temporal,
        )

        ref_idx_l0 = mb_pred.get('ref_idx_l0', [])
        ref_idx_l1 = mb_pred.get('ref_idx_l1', [])
        mvd_l0 = mb_pred.get('mvd_l0', [])
        mvd_l1 = mb_pred.get('mvd_l1', [])
        sub_types = mb_pred.get('sub_mb_types', [])

        sh = getattr(self.state, 'current_slice_header', None)
        use_spatial = getattr(sh, 'direct_spatial_mv_pred_flag', True)
        current_poc = getattr(sh, 'pic_order_cnt_lsb', 0)
        pps = self.state.current_pps
        weighted_bipred_idc = getattr(pps, 'weighted_bipred_idc', 0) if pps else 0

        if mb_type == 0:  # B_Direct_16x16
            luma, cb, cr = reconstruct_b_direct_16x16(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                mv_cache_l1=getattr(self.state, 'mv_cache_l1', None),
                mb_x=mb_x, mb_y=mb_y,
                use_spatial=use_spatial,
                current_poc=current_poc,
                residual_luma=None, residual_cb=None, residual_cr=None,
                weighted_bipred_idc=weighted_bipred_idc,
            )
            return luma, cb, cr

        # For types 1-21: extract partition info
        if 1 <= mb_type <= 21:
            return self._cabac_b_partitioned_mc(
                mb_type, ref_idx_l0, ref_idx_l1, mvd_l0, mvd_l1,
                mb_x, mb_y,
            )

        if mb_type == 22:  # B_8x8
            return self._cabac_b_8x8_mc(
                sub_types, ref_idx_l0, ref_idx_l1, mvd_l0, mvd_l1,
                mb_x, mb_y,
            )

        logger.warning(f"Unexpected B mb_type={mb_type} at ({mb_x},{mb_y})")
        return reconstruct_b_skip(
            ref_buffer=self.state.ref_buffer,
            mv_cache=self.state.mv_cache,
            mv_cache_l1=getattr(self.state, 'mv_cache_l1', None),
            mb_x=mb_x, mb_y=mb_y,
            use_spatial=use_spatial, current_poc=current_poc,
        )

    def _cabac_b_partitioned_mc(
        self, mb_type: int,
        ref_idx_l0: list, ref_idx_l1: list,
        mvd_l0: list, mvd_l1: list,
        mb_x: int, mb_y: int,
    ) -> tuple:
        """B-slice partitioned MC (types 1-21)."""
        from entropy.cabac_macroblock import _get_b_pred_flags

        sh = getattr(self.state, 'current_slice_header', None)
        weighted_bipred_idc = getattr(self.state.current_pps, 'weighted_bipred_idc', 0) if self.state.current_pps else 0
        current_poc = getattr(sh, 'pic_order_cnt_lsb', 0)

        pred_flags = _get_b_pred_flags(mb_type)
        is_16x16 = mb_type in (1, 2, 3)
        is_16x8 = mb_type in (4, 6, 8, 10, 12, 14, 16, 18, 20)

        num_parts = len(pred_flags)
        refs_l0 = [0] * num_parts
        refs_l1 = [0] * num_parts
        mvx_l0 = [0] * num_parts
        mvy_l0 = [0] * num_parts
        mvx_l1 = [0] * num_parts
        mvy_l1 = [0] * num_parts

        l0_ref_idx = 0
        l1_ref_idx = 0
        l0_mvd_idx = 0
        l1_mvd_idx = 0

        # Resolve ref indices
        for i in range(num_parts):
            if pred_flags[i][0]:
                refs_l0[i] = ref_idx_l0[l0_ref_idx] if l0_ref_idx < len(ref_idx_l0) else 0
                l0_ref_idx += 1
            else:
                refs_l0[i] = -1
        for i in range(num_parts):
            if pred_flags[i][1]:
                refs_l1[i] = ref_idx_l1[l1_ref_idx] if l1_ref_idx < len(ref_idx_l1) else 0
                l1_ref_idx += 1
            else:
                refs_l1[i] = -1

        # Resolve MVs and store per-partition immediately so partition 1
        # can see partition 0's MVs in the cache during prediction.
        mv_cache_l1 = getattr(self.state, 'mv_cache_l1', None)
        for i in range(num_parts):
            # L0 MV prediction + store
            if pred_flags[i][0]:
                mvd = mvd_l0[l0_mvd_idx] if l0_mvd_idx < len(mvd_l0) else (0, 0)
                l0_mvd_idx += 1
                if is_16x16:
                    mvp_x, mvp_y = predict_mv_16x16(
                        self.state.mv_cache, mb_x, mb_y, target_ref=refs_l0[i],
                    )
                elif is_16x8:
                    mvp_x, mvp_y = predict_mv_16x8(
                        self.state.mv_cache, mb_x, mb_y, i, target_ref=refs_l0[i],
                    )
                else:
                    mvp_x, mvp_y = predict_mv_8x16(
                        self.state.mv_cache, mb_x, mb_y, i, target_ref=refs_l0[i],
                    )
                mvx_l0[i] = mvp_x + mvd[0]
                mvy_l0[i] = mvp_y + mvd[1]

            # Store L0 MV (or mark unavailable for this list)
            if is_16x16:
                if pred_flags[i][0]:
                    self.state.mv_cache.set_mv_16x16(
                        mb_x, mb_y, mvx_l0[i], mvy_l0[i], ref_idx=refs_l0[i],
                    )
                else:
                    self.state.mv_cache.set_mv_16x16(mb_x, mb_y, 0, 0, ref_idx=-1)
            elif is_16x8:
                if pred_flags[i][0]:
                    self.state.mv_cache.set_mv_16x8(
                        mb_x, mb_y, i, mvx_l0[i], mvy_l0[i], ref_idx=refs_l0[i],
                    )
                else:
                    self.state.mv_cache.set_mv_16x8(
                        mb_x, mb_y, i, 0, 0, ref_idx=-1,
                    )
            else:  # 8x16
                if pred_flags[i][0]:
                    self.state.mv_cache.set_mv_8x16(
                        mb_x, mb_y, i, mvx_l0[i], mvy_l0[i], ref_idx=refs_l0[i],
                    )
                else:
                    self.state.mv_cache.set_mv_8x16(
                        mb_x, mb_y, i, 0, 0, ref_idx=-1,
                    )

            # L1 MV prediction + store
            if pred_flags[i][1]:
                mvd = mvd_l1[l1_mvd_idx] if l1_mvd_idx < len(mvd_l1) else (0, 0)
                l1_mvd_idx += 1
                if mv_cache_l1:
                    if is_16x16:
                        mvp_x, mvp_y = predict_mv_16x16(
                            mv_cache_l1, mb_x, mb_y, target_ref=refs_l1[i],
                        )
                    elif is_16x8:
                        mvp_x, mvp_y = predict_mv_16x8(
                            mv_cache_l1, mb_x, mb_y, i, target_ref=refs_l1[i],
                        )
                    else:
                        mvp_x, mvp_y = predict_mv_8x16(
                            mv_cache_l1, mb_x, mb_y, i, target_ref=refs_l1[i],
                        )
                else:
                    mvp_x, mvp_y = 0, 0
                mvx_l1[i] = mvp_x + mvd[0]
                mvy_l1[i] = mvp_y + mvd[1]

            # Store L1 MV (or mark unavailable)
            if mv_cache_l1:
                if is_16x16:
                    if pred_flags[i][1]:
                        mv_cache_l1.set_mv_16x16(
                            mb_x, mb_y, mvx_l1[i], mvy_l1[i], ref_idx=refs_l1[i],
                        )
                    else:
                        mv_cache_l1.set_mv_16x16(mb_x, mb_y, 0, 0, ref_idx=-1)
                elif is_16x8:
                    if pred_flags[i][1]:
                        mv_cache_l1.set_mv_16x8(
                            mb_x, mb_y, i, mvx_l1[i], mvy_l1[i], ref_idx=refs_l1[i],
                        )
                    else:
                        mv_cache_l1.set_mv_16x8(
                            mb_x, mb_y, i, 0, 0, ref_idx=-1,
                        )
                else:  # 8x16
                    if pred_flags[i][1]:
                        mv_cache_l1.set_mv_8x16(
                            mb_x, mb_y, i, mvx_l1[i], mvy_l1[i], ref_idx=refs_l1[i],
                        )
                    else:
                        mv_cache_l1.set_mv_8x16(
                            mb_x, mb_y, i, 0, 0, ref_idx=-1,
                        )

        # Dispatch to appropriate reconstruction
        if mb_type == 1:  # B_L0_16x16
            return reconstruct_b_l0_16x16(
                ref_buffer=self.state.ref_buffer,
                ref_idx=refs_l0[0], mvx=mvx_l0[0], mvy=mvy_l0[0],
                residual_luma=None, residual_cb=None, residual_cr=None,
                mb_x=mb_x, mb_y=mb_y,
            )
        elif mb_type == 2:  # B_L1_16x16
            return reconstruct_b_l1_16x16(
                ref_buffer=self.state.ref_buffer,
                ref_idx=refs_l1[0], mvx=mvx_l1[0], mvy=mvy_l1[0],
                residual_luma=None, residual_cb=None, residual_cr=None,
                mb_x=mb_x, mb_y=mb_y,
            )
        elif mb_type == 3:  # B_Bi_16x16
            return reconstruct_b_bi_16x16(
                ref_buffer=self.state.ref_buffer,
                ref_idx_l0=refs_l0[0], mvx_l0=mvx_l0[0], mvy_l0=mvy_l0[0],
                ref_idx_l1=refs_l1[0], mvx_l1=mvx_l1[0], mvy_l1=mvy_l1[0],
                residual_luma=None, residual_cb=None, residual_cr=None,
                mb_x=mb_x, mb_y=mb_y,
                weighted_bipred_idc=weighted_bipred_idc,
                current_poc=current_poc,
            )
        # Convert pred_flags to pred_modes strings
        pred_modes = []
        for l0, l1 in pred_flags:
            if l0 and l1:
                pred_modes.append("Bi")
            elif l0:
                pred_modes.append("L0")
            else:
                pred_modes.append("L1")

        if is_16x8:
            return reconstruct_b_16x8(
                ref_buffer=self.state.ref_buffer,
                pred_modes=pred_modes,
                ref_idx_l0=refs_l0, ref_idx_l1=refs_l1,
                mvx_l0=mvx_l0, mvy_l0=mvy_l0,
                mvx_l1=mvx_l1, mvy_l1=mvy_l1,
                residual_luma=None, residual_cb=None, residual_cr=None,
                mb_x=mb_x, mb_y=mb_y,
                weighted_bipred_idc=weighted_bipred_idc,
                current_poc=current_poc,
            )
        else:  # 8x16
            return reconstruct_b_8x16(
                ref_buffer=self.state.ref_buffer,
                pred_modes=pred_modes,
                ref_idx_l0=refs_l0, ref_idx_l1=refs_l1,
                mvx_l0=mvx_l0, mvy_l0=mvy_l0,
                mvx_l1=mvx_l1, mvy_l1=mvy_l1,
                residual_luma=None, residual_cb=None, residual_cr=None,
                mb_x=mb_x, mb_y=mb_y,
                weighted_bipred_idc=weighted_bipred_idc,
                current_poc=current_poc,
            )

    def _cabac_b_8x8_mc(
        self, sub_types: list,
        ref_idx_l0: list, ref_idx_l1: list,
        mvd_l0: list, mvd_l1: list,
        mb_x: int, mb_y: int,
    ) -> tuple:
        """B_8x8 motion compensation with sub-partitions.

        Handles all B sub_mb_types (0-12) including sub-partition shapes
        8x8, 8x4, 4x8, and 4x4 with per-sub-partition MV prediction.

        H.264 Spec Reference: Table 7-18, Section 8.4
        """
        from inter.direct_mode import derive_direct_spatial_sub8x8
        from inter.bipred import bipred_average, implicit_bipred
        from entropy.cabac_macroblock import (
            _b_sub_uses_l0, _b_sub_uses_l1, _b_sub_num_parts,
        )

        mv_cache_l1 = getattr(self.state, 'mv_cache_l1', None)
        sh = getattr(self.state, 'current_slice_header', None)
        weighted_bipred_idc = getattr(self.state.current_pps, 'weighted_bipred_idc', 0) if self.state.current_pps else 0
        current_poc = getattr(sh, 'pic_order_cnt_lsb', 0)

        refs_l0 = [ref_idx_l0[i] if i < len(ref_idx_l0) else 0 for i in range(4)]
        refs_l1 = [ref_idx_l1[i] if i < len(ref_idx_l1) else 0 for i in range(4)]

        def _do_bipred(pred_l0, pred_l1, ri_l0, ri_l1):
            """Apply bipred with correct weighting per weighted_bipred_idc."""
            if weighted_bipred_idc == 2:
                l0_poc = self.state.ref_buffer.get_l0_frame(max(ri_l0, 0)).poc
                l1_poc = self.state.ref_buffer.get_l1_frame(max(ri_l1, 0)).poc
                return implicit_bipred(pred_l0, pred_l1, current_poc, l0_poc, l1_poc)
            return bipred_average(pred_l0, pred_l1)

        # B sub_mb_type → shape index: 0=8x8, 1=8x4, 2=4x8, 3=4x4
        _b_sub_shape = {
            0: 0, 1: 0, 2: 0, 3: 0,   # Direct/L0/L1/Bi 8x8
            4: 1, 6: 1, 8: 1,          # L0/L1/Bi 8x4
            5: 2, 7: 2, 9: 2,          # L0/L1/Bi 4x8
            10: 3, 11: 3, 12: 3,       # L0/L1/Bi 4x4
        }

        # Sub-partition definitions: (block_offset, part_width_4x4, coverage)
        _sp = {
            0: [((0, 0), 2, [(0, 0), (1, 0), (0, 1), (1, 1)])],
            1: [((0, 0), 2, [(0, 0), (1, 0)]),
                ((0, 1), 2, [(0, 1), (1, 1)])],
            2: [((0, 0), 1, [(0, 0), (0, 1)]),
                ((1, 0), 1, [(1, 0), (1, 1)])],
            3: [((0, 0), 1, [(0, 0)]), ((1, 0), 1, [(1, 0)]),
                ((0, 1), 1, [(0, 1)]), ((1, 1), 1, [(1, 1)])],
        }

        # Chroma sub-partition geometry: (cx_off, cy_off, cw, ch)
        _csp = {
            0: [(0, 0, 4, 4)],
            1: [(0, 0, 4, 2), (0, 2, 4, 2)],
            2: [(0, 0, 2, 4), (2, 0, 2, 4)],
            3: [(0, 0, 2, 2), (2, 0, 2, 2), (0, 2, 2, 2), (2, 2, 2, 2)],
        }

        # Luma sub-partition pixel geometry: (px_off, py_off, pw, ph)
        _lsp = {
            0: [(0, 0, 8, 8)],
            1: [(0, 0, 8, 4), (0, 4, 8, 4)],
            2: [(0, 0, 4, 8), (4, 0, 4, 8)],
            3: [(0, 0, 4, 4), (4, 0, 4, 4), (0, 4, 4, 4), (4, 4, 4, 4)],
        }

        # B sub_mb_type → pred mode string
        _b_sub_pred_mode = {
            0: "Direct",
            1: "L0", 4: "L0", 5: "L0", 10: "L0",
            2: "L1", 6: "L1", 7: "L1", 11: "L1",
            3: "Bi", 8: "Bi", 9: "Bi", 12: "Bi",
        }

        luma = np.zeros((16, 16), dtype=np.uint8)
        pred_cb = np.zeros((8, 8), dtype=np.uint8)
        pred_cr = np.zeros((8, 8), dtype=np.uint8)
        sub_mb_offsets = [(0, 0), (8, 0), (0, 8), (8, 8)]
        chroma_offsets = [(0, 0), (4, 0), (0, 4), (4, 4)]

        l0_mvd_idx = 0
        l1_mvd_idx = 0

        for sub_idx in range(4):
            sub_type = sub_types[sub_idx] if sub_idx < len(sub_types) else 0
            sub_bx = (sub_idx % 2) * 2
            sub_by = (sub_idx // 2) * 2
            sx, sy = sub_mb_offsets[sub_idx]
            cx_base, cy_base = chroma_offsets[sub_idx]

            if sub_type == 0:  # B_Direct_8x8
                direct_result = derive_direct_spatial_sub8x8(
                    self.state.mv_cache,
                    mv_cache_l1,
                    mb_x, mb_y, sub_idx,
                    self.state.ref_buffer,
                )
                if direct_result:
                    (d_mvx_l0, d_mvy_l0, d_mvx_l1, d_mvy_l1,
                     pf_l0, pf_l1, d_ref_l0, d_ref_l1) = direct_result

                    # Store derived MVs in caches for neighbor lookups
                    for by in range(sub_by, sub_by + 2):
                        for bx in range(sub_bx, sub_bx + 2):
                            self.state.mv_cache.set_mv(
                                mb_x, mb_y, bx, by,
                                d_mvx_l0, d_mvy_l0, ref_idx=d_ref_l0,
                            )
                            if mv_cache_l1:
                                mv_cache_l1.set_mv(
                                    mb_x, mb_y, bx, by,
                                    d_mvx_l1, d_mvy_l1, ref_idx=d_ref_l1,
                                )

                    # Determine prediction mode from pred flags
                    if pf_l0 and pf_l1:
                        ref_l0 = self.state.ref_buffer.get_l0_frame(d_ref_l0)
                        ref_l1 = self.state.ref_buffer.get_l1_frame(d_ref_l1)
                        l0_luma = apply_inter_prediction(
                            ref_l0.luma,
                            mb_x * 16 + sx + (d_mvx_l0 >> 2),
                            mb_y * 16 + sy + (d_mvy_l0 >> 2),
                            d_mvx_l0 & 3, d_mvy_l0 & 3, 8, 8,
                        )
                        l1_luma = apply_inter_prediction(
                            ref_l1.luma,
                            mb_x * 16 + sx + (d_mvx_l1 >> 2),
                            mb_y * 16 + sy + (d_mvy_l1 >> 2),
                            d_mvx_l1 & 3, d_mvy_l1 & 3, 8, 8,
                        )
                        luma[sy:sy + 8, sx:sx + 8] = _do_bipred(l0_luma, l1_luma, d_ref_l0, d_ref_l1)

                        l0_cb = apply_chroma_prediction(
                            ref_l0.cb,
                            mb_x * 8 + cx_base + (d_mvx_l0 >> 3),
                            mb_y * 8 + cy_base + (d_mvy_l0 >> 3),
                            d_mvx_l0 & 7, d_mvy_l0 & 7, 4, 4,
                        )
                        l0_cr = apply_chroma_prediction(
                            ref_l0.cr,
                            mb_x * 8 + cx_base + (d_mvx_l0 >> 3),
                            mb_y * 8 + cy_base + (d_mvy_l0 >> 3),
                            d_mvx_l0 & 7, d_mvy_l0 & 7, 4, 4,
                        )
                        l1_cb = apply_chroma_prediction(
                            ref_l1.cb,
                            mb_x * 8 + cx_base + (d_mvx_l1 >> 3),
                            mb_y * 8 + cy_base + (d_mvy_l1 >> 3),
                            d_mvx_l1 & 7, d_mvy_l1 & 7, 4, 4,
                        )
                        l1_cr = apply_chroma_prediction(
                            ref_l1.cr,
                            mb_x * 8 + cx_base + (d_mvx_l1 >> 3),
                            mb_y * 8 + cy_base + (d_mvy_l1 >> 3),
                            d_mvx_l1 & 7, d_mvy_l1 & 7, 4, 4,
                        )
                        pred_cb[cy_base:cy_base + 4, cx_base:cx_base + 4] = (
                            _do_bipred(l0_cb, l1_cb, d_ref_l0, d_ref_l1)
                        )
                        pred_cr[cy_base:cy_base + 4, cx_base:cx_base + 4] = (
                            _do_bipred(l0_cr, l1_cr, d_ref_l0, d_ref_l1)
                        )
                    elif pf_l0:
                        ref_l0 = self.state.ref_buffer.get_l0_frame(max(d_ref_l0, 0))
                        luma[sy:sy + 8, sx:sx + 8] = apply_inter_prediction(
                            ref_l0.luma,
                            mb_x * 16 + sx + (d_mvx_l0 >> 2),
                            mb_y * 16 + sy + (d_mvy_l0 >> 2),
                            d_mvx_l0 & 3, d_mvy_l0 & 3, 8, 8,
                        )
                        pred_cb[cy_base:cy_base + 4, cx_base:cx_base + 4] = (
                            apply_chroma_prediction(
                                ref_l0.cb,
                                mb_x * 8 + cx_base + (d_mvx_l0 >> 3),
                                mb_y * 8 + cy_base + (d_mvy_l0 >> 3),
                                d_mvx_l0 & 7, d_mvy_l0 & 7, 4, 4,
                            )
                        )
                        pred_cr[cy_base:cy_base + 4, cx_base:cx_base + 4] = (
                            apply_chroma_prediction(
                                ref_l0.cr,
                                mb_x * 8 + cx_base + (d_mvx_l0 >> 3),
                                mb_y * 8 + cy_base + (d_mvy_l0 >> 3),
                                d_mvx_l0 & 7, d_mvy_l0 & 7, 4, 4,
                            )
                        )
                    else:
                        ref_l1 = self.state.ref_buffer.get_l1_frame(max(d_ref_l1, 0))
                        luma[sy:sy + 8, sx:sx + 8] = apply_inter_prediction(
                            ref_l1.luma,
                            mb_x * 16 + sx + (d_mvx_l1 >> 2),
                            mb_y * 16 + sy + (d_mvy_l1 >> 2),
                            d_mvx_l1 & 3, d_mvy_l1 & 3, 8, 8,
                        )
                        pred_cb[cy_base:cy_base + 4, cx_base:cx_base + 4] = (
                            apply_chroma_prediction(
                                ref_l1.cb,
                                mb_x * 8 + cx_base + (d_mvx_l1 >> 3),
                                mb_y * 8 + cy_base + (d_mvy_l1 >> 3),
                                d_mvx_l1 & 7, d_mvy_l1 & 7, 4, 4,
                            )
                        )
                        pred_cr[cy_base:cy_base + 4, cx_base:cx_base + 4] = (
                            apply_chroma_prediction(
                                ref_l1.cr,
                                mb_x * 8 + cx_base + (d_mvx_l1 >> 3),
                                mb_y * 8 + cy_base + (d_mvy_l1 >> 3),
                                d_mvx_l1 & 7, d_mvy_l1 & 7, 4, 4,
                            )
                        )
                continue

            shape = _b_sub_shape.get(sub_type, 0)
            parts = _sp[shape]
            chroma_parts = _csp[shape]
            luma_parts = _lsp[shape]
            pred_mode = _b_sub_pred_mode.get(sub_type, "L0")
            uses_l0 = _b_sub_uses_l0(sub_type)
            uses_l1 = _b_sub_uses_l1(sub_type)

            sub_mvs_l0 = []
            sub_mvs_l1 = []

            # L0 MV prediction per sub-partition
            if uses_l0:
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
            else:
                # No L0 — mark all blocks as unavailable in L0 cache
                for (bx_off, by_off) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                    self.state.mv_cache.set_mv(
                        mb_x, mb_y, sub_bx + bx_off, sub_by + by_off,
                        0, 0, ref_idx=-1,
                    )
                sub_mvs_l0 = [(0, 0)] * len(parts)

            # L1 MV prediction per sub-partition
            if uses_l1 and mv_cache_l1:
                for j, ((dx, dy), pw, coverage) in enumerate(parts):
                    mvd = mvd_l1[l1_mvd_idx] if l1_mvd_idx < len(mvd_l1) else (0, 0)
                    l1_mvd_idx += 1

                    mvp_x, mvp_y = predict_mv_partition(
                        mv_cache_l1, mb_x, mb_y,
                        block_x=sub_bx + dx, block_y=sub_by + dy,
                        part_width_blocks=pw, target_ref=refs_l1[sub_idx],
                    )
                    final_mvx = mvp_x + mvd[0]
                    final_mvy = mvp_y + mvd[1]
                    sub_mvs_l1.append((final_mvx, final_mvy))

                    for (bx_off, by_off) in coverage:
                        mv_cache_l1.set_mv(
                            mb_x, mb_y, sub_bx + bx_off, sub_by + by_off,
                            final_mvx, final_mvy, ref_idx=refs_l1[sub_idx],
                        )
            else:
                # No L1 — mark all blocks as unavailable in L1 cache
                if mv_cache_l1:
                    for (bx_off, by_off) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                        mv_cache_l1.set_mv(
                            mb_x, mb_y, sub_bx + bx_off, sub_by + by_off,
                            0, 0, ref_idx=-1,
                        )
                sub_mvs_l1 = [(0, 0)] * len(parts)

            # Reconstruct luma per sub-partition
            for j, (px_off, py_off, pw, ph) in enumerate(luma_parts):
                if pred_mode == "L0":
                    mvx, mvy = sub_mvs_l0[j]
                    ref_frame = self.state.ref_buffer.get_l0_frame(refs_l0[sub_idx])
                    part_luma = apply_inter_prediction(
                        ref_frame.luma,
                        mb_x * 16 + sx + px_off + (mvx >> 2),
                        mb_y * 16 + sy + py_off + (mvy >> 2),
                        mvx & 3, mvy & 3, pw, ph,
                    )
                elif pred_mode == "L1":
                    mvx, mvy = sub_mvs_l1[j]
                    ref_frame = self.state.ref_buffer.get_l1_frame(refs_l1[sub_idx])
                    part_luma = apply_inter_prediction(
                        ref_frame.luma,
                        mb_x * 16 + sx + px_off + (mvx >> 2),
                        mb_y * 16 + sy + py_off + (mvy >> 2),
                        mvx & 3, mvy & 3, pw, ph,
                    )
                else:  # Bi
                    mvx0, mvy0 = sub_mvs_l0[j]
                    mvx1, mvy1 = sub_mvs_l1[j]
                    ref_l0 = self.state.ref_buffer.get_l0_frame(refs_l0[sub_idx])
                    ref_l1 = self.state.ref_buffer.get_l1_frame(refs_l1[sub_idx])
                    l0_part = apply_inter_prediction(
                        ref_l0.luma,
                        mb_x * 16 + sx + px_off + (mvx0 >> 2),
                        mb_y * 16 + sy + py_off + (mvy0 >> 2),
                        mvx0 & 3, mvy0 & 3, pw, ph,
                    )
                    l1_part = apply_inter_prediction(
                        ref_l1.luma,
                        mb_x * 16 + sx + px_off + (mvx1 >> 2),
                        mb_y * 16 + sy + py_off + (mvy1 >> 2),
                        mvx1 & 3, mvy1 & 3, pw, ph,
                    )
                    part_luma = _do_bipred(l0_part, l1_part, refs_l0[sub_idx], refs_l1[sub_idx])

                luma[sy + py_off:sy + py_off + ph,
                     sx + px_off:sx + px_off + pw] = part_luma

            # Reconstruct chroma per sub-partition
            for j, (cx_off, cy_off, cw, ch) in enumerate(chroma_parts):
                if pred_mode == "L0":
                    mvx, mvy = sub_mvs_l0[j]
                    ref_frame = self.state.ref_buffer.get_l0_frame(refs_l0[sub_idx])
                    pred_cb[cy_base + cy_off:cy_base + cy_off + ch,
                            cx_base + cx_off:cx_base + cx_off + cw] = (
                        apply_chroma_prediction(
                            ref_frame.cb,
                            mb_x * 8 + cx_base + cx_off + (mvx >> 3),
                            mb_y * 8 + cy_base + cy_off + (mvy >> 3),
                            mvx & 7, mvy & 7, cw, ch,
                        )
                    )
                    pred_cr[cy_base + cy_off:cy_base + cy_off + ch,
                            cx_base + cx_off:cx_base + cx_off + cw] = (
                        apply_chroma_prediction(
                            ref_frame.cr,
                            mb_x * 8 + cx_base + cx_off + (mvx >> 3),
                            mb_y * 8 + cy_base + cy_off + (mvy >> 3),
                            mvx & 7, mvy & 7, cw, ch,
                        )
                    )
                elif pred_mode == "L1":
                    mvx, mvy = sub_mvs_l1[j]
                    ref_frame = self.state.ref_buffer.get_l1_frame(refs_l1[sub_idx])
                    pred_cb[cy_base + cy_off:cy_base + cy_off + ch,
                            cx_base + cx_off:cx_base + cx_off + cw] = (
                        apply_chroma_prediction(
                            ref_frame.cb,
                            mb_x * 8 + cx_base + cx_off + (mvx >> 3),
                            mb_y * 8 + cy_base + cy_off + (mvy >> 3),
                            mvx & 7, mvy & 7, cw, ch,
                        )
                    )
                    pred_cr[cy_base + cy_off:cy_base + cy_off + ch,
                            cx_base + cx_off:cx_base + cx_off + cw] = (
                        apply_chroma_prediction(
                            ref_frame.cr,
                            mb_x * 8 + cx_base + cx_off + (mvx >> 3),
                            mb_y * 8 + cy_base + cy_off + (mvy >> 3),
                            mvx & 7, mvy & 7, cw, ch,
                        )
                    )
                else:  # Bi
                    mvx0, mvy0 = sub_mvs_l0[j]
                    mvx1, mvy1 = sub_mvs_l1[j]
                    ref_l0 = self.state.ref_buffer.get_l0_frame(refs_l0[sub_idx])
                    ref_l1 = self.state.ref_buffer.get_l1_frame(refs_l1[sub_idx])
                    l0_cb = apply_chroma_prediction(
                        ref_l0.cb,
                        mb_x * 8 + cx_base + cx_off + (mvx0 >> 3),
                        mb_y * 8 + cy_base + cy_off + (mvy0 >> 3),
                        mvx0 & 7, mvy0 & 7, cw, ch,
                    )
                    l0_cr = apply_chroma_prediction(
                        ref_l0.cr,
                        mb_x * 8 + cx_base + cx_off + (mvx0 >> 3),
                        mb_y * 8 + cy_base + cy_off + (mvy0 >> 3),
                        mvx0 & 7, mvy0 & 7, cw, ch,
                    )
                    l1_cb = apply_chroma_prediction(
                        ref_l1.cb,
                        mb_x * 8 + cx_base + cx_off + (mvx1 >> 3),
                        mb_y * 8 + cy_base + cy_off + (mvy1 >> 3),
                        mvx1 & 7, mvy1 & 7, cw, ch,
                    )
                    l1_cr = apply_chroma_prediction(
                        ref_l1.cr,
                        mb_x * 8 + cx_base + cx_off + (mvx1 >> 3),
                        mb_y * 8 + cy_base + cy_off + (mvy1 >> 3),
                        mvx1 & 7, mvy1 & 7, cw, ch,
                    )
                    pred_cb[cy_base + cy_off:cy_base + cy_off + ch,
                            cx_base + cx_off:cx_base + cx_off + cw] = (
                        _do_bipred(l0_cb, l1_cb, refs_l0[sub_idx], refs_l1[sub_idx])
                    )
                    pred_cr[cy_base + cy_off:cy_base + cy_off + ch,
                            cx_base + cx_off:cx_base + cx_off + cw] = (
                        _do_bipred(l0_cr, l1_cr, refs_l0[sub_idx], refs_l1[sub_idx])
                    )

        return luma, pred_cb, pred_cr

    def _cabac_chroma_residual(
        self, residual: dict, plane: str,
        cbp_chroma: int, qp_chroma: int,
        nz_counts: np.ndarray, nz_offset: int,
    ) -> np.ndarray:
        """Reconstruct chroma residual from pre-parsed CABAC coefficients.

        Uses process_chroma_dc_coeffs for DC and same pattern as CAVLC path
        for AC blocks.
        """
        from reconstruct.macroblock import process_chroma_dc_coeffs
        from entropy.cavlc import CAVLCBlock

        chroma_residual = np.zeros((8, 8), dtype=np.int32)

        if cbp_chroma == 0:
            return chroma_residual

        # Process DC coefficients through hadamard + dequant
        dc_key = f'chroma_dc_{plane}'
        dc_raw = residual.get(dc_key)
        if dc_raw is not None:
            dc_arr = np.asarray(dc_raw, dtype=np.int32)
            dc_dequant = process_chroma_dc_coeffs(dc_arr, qp_chroma)
        else:
            dc_dequant = None

        # Build AC blocks (wrap in CAVLCBlock for build_chroma_residual)
        ac_key = f'chroma_ac_{plane}'
        ac_raw = residual.get(ac_key, [None] * 4)
        ac_blocks = []

        if cbp_chroma == 2:
            for i in range(4):
                if ac_raw[i] is not None:
                    ac_arr = np.asarray(ac_raw[i], dtype=np.int32)
                    nz_counts[nz_offset + i] = int(np.count_nonzero(ac_arr))
                    block = CAVLCBlock(
                        total_coeff=int(np.count_nonzero(ac_arr)),
                        trailing_ones=0,
                        coefficients=ac_arr,
                    )
                    ac_blocks.append(block)
                else:
                    nz_counts[nz_offset + i] = 0
                    ac_blocks.append(CAVLCBlock(
                        total_coeff=0, trailing_ones=0,
                        coefficients=np.zeros(15, dtype=np.int32),
                    ))

        return build_chroma_residual(dc_dequant, ac_blocks, cbp_chroma, qp_chroma)

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
        slice_header: 'SliceHeader' = None,
    ) -> int:
        """Decode a single B-macroblock.

        Handles all B-MB types from H.264 Table 7-14:
        - Direct (type 0)
        - L0/L1/Bi 16x16 (types 1-3)
        - Partitioned 16x8/8x16 (types 4-21)
        - Sub-partitioned 8x8 (type 22)

        Args:
            reader: BitReader
            mb_info: Parsed BMacroblockInfo
            mb_x, mb_y: Macroblock position
            qp: Current QP
            sps, pps: Parameter sets
            use_spatial: True for spatial direct mode
            current_poc: Picture order count
            slice_header: Slice header (for num_ref_idx_active)

        Returns:
            Updated QP after mb_qp_delta
        """
        mb_width = sps.pic_width_in_mbs
        mb_idx = mb_y * mb_width + mb_x
        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8

        # Use num_ref_idx_active from slice header (H.264 7.3.3)
        if slice_header is not None:
            num_l0 = getattr(slice_header, 'num_ref_idx_l0_active_minus1', 0) + 1
            num_l1 = getattr(slice_header, 'num_ref_idx_l1_active_minus1', 0) + 1
        else:
            num_l0 = len(self.state.l0_list) if self.state.l0_list else 1
            num_l1 = len(self.state.l1_list) if self.state.l1_list else 1

        part_modes = get_partition_pred_modes(mb_info)
        num_parts = mb_info.num_partitions
        part_size = mb_info.partition_size  # (w, h)
        weighted_bipred_idc = getattr(pps, 'weighted_bipred_idc', 0)

        if mb_info.is_direct:
            # B_Direct_16x16 — no ref_idx or MVD in bitstream
            luma_pred, cb_pred, cr_pred = reconstruct_b_direct_16x16(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                residual_luma=None,
                residual_cb=None,
                residual_cr=None,
                mb_x=mb_x,
                mb_y=mb_y,
                use_spatial=use_spatial,
                current_poc=current_poc,
                mv_cache_l1=self.state.mv_cache_l1,
                weighted_bipred_idc=weighted_bipred_idc,
            )

        elif mb_info.name == "B_8x8":
            # ── B_8x8: sub-macroblock partition types ──
            sub_mb_infos = []
            for i in range(4):
                sub_mb_infos.append(parse_b_sub_mb_type(reader.read_ue()))

            sub_pred_modes = [s.pred_mode for s in sub_mb_infos]

            # ref_idx_l0 for each sub-MB
            ref_idx_l0 = []
            for i in range(4):
                if not sub_mb_infos[i].is_direct and sub_pred_modes[i] in ("L0", "Bi"):
                    ref_idx_l0.append(reader.read_te(num_l0 - 1) if num_l0 > 1 else 0)
                else:
                    ref_idx_l0.append(0)

            # ref_idx_l1 for each sub-MB
            ref_idx_l1 = []
            for i in range(4):
                if not sub_mb_infos[i].is_direct and sub_pred_modes[i] in ("L1", "Bi"):
                    ref_idx_l1.append(reader.read_te(num_l1 - 1) if num_l1 > 1 else 0)
                else:
                    ref_idx_l1.append(0)

            # Read all L0 MVDs first (bitstream syntax sends all L0 before L1)
            mvd_l0_list = [(0, 0)] * 4
            for i in range(4):
                if not sub_mb_infos[i].is_direct and sub_pred_modes[i] in ("L0", "Bi"):
                    num_sub_parts = sub_mb_infos[i].num_partitions
                    for j in range(num_sub_parts):
                        mvd_x = reader.read_se()
                        mvd_y = reader.read_se()
                        if j == 0:
                            mvd_l0_list[i] = (mvd_x, mvd_y)

            # Read all L1 MVDs
            mvd_l1_list = [(0, 0)] * 4
            for i in range(4):
                if not sub_mb_infos[i].is_direct and sub_pred_modes[i] in ("L1", "Bi"):
                    num_sub_parts = sub_mb_infos[i].num_partitions
                    for j in range(num_sub_parts):
                        mvd_x = reader.read_se()
                        mvd_y = reader.read_se()
                        if j == 0:
                            mvd_l1_list[i] = (mvd_x, mvd_y)

            # Unified MV derivation loop: process sub-blocks sequentially
            # so each sub-block sees previously decoded neighbors in BOTH
            # L0 and L1 caches (H.264 8.4.1.2.2, 8.4.1.3.1)
            from inter.direct_mode import (
                derive_direct_spatial_sub8x8, derive_direct_temporal,
            )
            mvx_l0, mvy_l0 = [0] * 4, [0] * 4
            mvx_l1, mvy_l1 = [0] * 4, [0] * 4

            for i in range(4):
                if sub_mb_infos[i].is_direct:
                    # Per-sub-block direct MV derivation
                    if use_spatial:
                        (d_l0x, d_l0y, d_l1x, d_l1y,
                         pf_l0, pf_l1,
                         d_ri_l0, d_ri_l1) = derive_direct_spatial_sub8x8(
                            self.state.mv_cache, self.state.mv_cache_l1,
                            mb_x, mb_y, i, self.state.ref_buffer,
                        )
                    else:
                        d_l0x, d_l0y, d_l1x, d_l1y = derive_direct_temporal(
                            self.state.ref_buffer, current_poc, mb_x, mb_y,
                        )
                        pf_l0, pf_l1 = True, True
                        d_ri_l0, d_ri_l1 = 0, 0
                    mvx_l0[i], mvy_l0[i] = d_l0x, d_l0y
                    mvx_l1[i], mvy_l1[i] = d_l1x, d_l1y
                    ref_idx_l0[i] = max(d_ri_l0, 0)
                    ref_idx_l1[i] = max(d_ri_l1, 0)
                    # Update pred mode based on actual direct derivation flags
                    if pf_l0 and pf_l1:
                        sub_pred_modes[i] = "Bi"
                    elif pf_l0:
                        sub_pred_modes[i] = "L0"
                    else:
                        sub_pred_modes[i] = "L1"
                    # Update MV caches with derived ref_idx
                    if pf_l0:
                        self.state.mv_cache.set_mv_8x8(
                            mb_x, mb_y, i, mvx_l0[i], mvy_l0[i],
                            ref_idx=ref_idx_l0[i],
                        )
                    else:
                        self.state.mv_cache.mark_intra_8x8(mb_x, mb_y, i)
                    if self.state.mv_cache_l1 is not None:
                        if pf_l1:
                            self.state.mv_cache_l1.set_mv_8x8(
                                mb_x, mb_y, i, mvx_l1[i], mvy_l1[i],
                                ref_idx=ref_idx_l1[i],
                            )
                        else:
                            self.state.mv_cache_l1.mark_intra_8x8(
                                mb_x, mb_y, i
                            )
                else:
                    # L0 MV
                    if sub_pred_modes[i] in ("L0", "Bi"):
                        mvp_x, mvp_y = predict_mv_8x8(
                            self.state.mv_cache, mb_x, mb_y, i,
                            target_ref=ref_idx_l0[i],
                        )
                        mvx_l0[i] = mvp_x + mvd_l0_list[i][0]
                        mvy_l0[i] = mvp_y + mvd_l0_list[i][1]
                        self.state.mv_cache.set_mv_8x8(
                            mb_x, mb_y, i, mvx_l0[i], mvy_l0[i],
                            ref_idx=ref_idx_l0[i],
                        )
                    else:
                        self.state.mv_cache.mark_intra_8x8(mb_x, mb_y, i)

                    # L1 MV
                    if sub_pred_modes[i] in ("L1", "Bi"):
                        mvp_x, mvp_y = predict_mv_8x8(
                            self.state.mv_cache_l1, mb_x, mb_y, i,
                            target_ref=ref_idx_l1[i],
                        )
                        mvx_l1[i] = mvp_x + mvd_l1_list[i][0]
                        mvy_l1[i] = mvp_y + mvd_l1_list[i][1]
                        if self.state.mv_cache_l1 is not None:
                            self.state.mv_cache_l1.set_mv_8x8(
                                mb_x, mb_y, i, mvx_l1[i], mvy_l1[i],
                                ref_idx=ref_idx_l1[i],
                            )
                    else:
                        if self.state.mv_cache_l1 is not None:
                            self.state.mv_cache_l1.mark_intra_8x8(mb_x, mb_y, i)

            logger.debug(
                f"B_8x8 MB({mb_x},{mb_y}): "
                f"sub_types={[s.sub_mb_type for s in sub_mb_infos]} "
                f"modes={sub_pred_modes} "
                f"mvL0={list(zip(mvx_l0, mvy_l0))} "
                f"mvL1={list(zip(mvx_l1, mvy_l1))} "
                f"refL0={ref_idx_l0} refL1={ref_idx_l1}"
            )
            luma_pred, cb_pred, cr_pred = reconstruct_b_8x8(
                ref_buffer=self.state.ref_buffer,
                mv_cache=self.state.mv_cache,
                sub_mb_types=[s.sub_mb_type for s in sub_mb_infos],
                pred_modes=sub_pred_modes,
                ref_idx_l0=ref_idx_l0,
                ref_idx_l1=ref_idx_l1,
                mvx_l0=mvx_l0,
                mvy_l0=mvy_l0,
                mvx_l1=mvx_l1,
                mvy_l1=mvy_l1,
                residual_luma=None,
                residual_cb=None,
                residual_cr=None,
                mb_x=mb_x,
                mb_y=mb_y,
                weighted_bipred_idc=weighted_bipred_idc,
                current_poc=current_poc,
            )

        elif num_parts == 1:
            # ── 16x16 modes (L0, L1, Bi) ──
            ref_idx_l0 = [0]
            ref_idx_l1 = [0]

            if part_modes[0] in ("L0", "Bi"):
                if num_l0 > 1:
                    ref_idx_l0[0] = reader.read_te(num_l0 - 1)
            if part_modes[0] in ("L1", "Bi"):
                if num_l1 > 1:
                    ref_idx_l1[0] = reader.read_te(num_l1 - 1)

            mvx_l0, mvy_l0 = [0], [0]
            mvx_l1, mvy_l1 = [0], [0]

            if part_modes[0] in ("L0", "Bi"):
                mvd_x = reader.read_se()
                mvd_y = reader.read_se()
                mvp_x, mvp_y = predict_mv_16x16(
                    self.state.mv_cache, mb_x, mb_y
                )
                mvx_l0[0] = mvp_x + mvd_x
                mvy_l0[0] = mvp_y + mvd_y
                if hasattr(self, '_trace_b_mvd'):
                    logger.warning(f"B16x16 L0 MB({mb_x},{mb_y}) poc={current_poc}: mvd=({mvd_x},{mvd_y}) mvp=({mvp_x},{mvp_y}) final=({mvx_l0[0]},{mvy_l0[0]})")

            if part_modes[0] in ("L1", "Bi"):
                mvd_x = reader.read_se()
                mvd_y = reader.read_se()
                mvp_x, mvp_y = predict_mv_16x16(
                    self.state.mv_cache_l1, mb_x, mb_y
                )
                mvx_l1[0] = mvp_x + mvd_x
                mvy_l1[0] = mvp_y + mvd_y
                if hasattr(self, '_trace_b_mvd'):
                    logger.warning(f"B16x16 L1 MB({mb_x},{mb_y}) poc={current_poc}: mvd=({mvd_x},{mvd_y}) mvp=({mvp_x},{mvp_y}) final=({mvx_l1[0]},{mvy_l1[0]})")

            # Store MVs in caches
            if part_modes[0] in ("L0", "Bi"):
                self.state.mv_cache.set_mv_16x16(
                    mb_x, mb_y, mvx_l0[0], mvy_l0[0]
                )
            else:
                # L1-only: mark L0 as unavailable (MV=0, ref=-1)
                self.state.mv_cache.mark_intra(mb_x, mb_y)
            if self.state.mv_cache_l1 is not None:
                if part_modes[0] in ("L1", "Bi"):
                    self.state.mv_cache_l1.set_mv_16x16(
                        mb_x, mb_y, mvx_l1[0], mvy_l1[0]
                    )
                else:
                    # L0-only: mark L1 as unavailable (MV=0, ref=-1)
                    self.state.mv_cache_l1.mark_intra(mb_x, mb_y)

            if part_modes[0] == "L0":
                luma_pred, cb_pred, cr_pred = reconstruct_b_l0_16x16(
                    ref_buffer=self.state.ref_buffer,
                    ref_idx=ref_idx_l0[0],
                    mvx=mvx_l0[0], mvy=mvy_l0[0],
                    residual_luma=None, residual_cb=None, residual_cr=None,
                    mb_x=mb_x, mb_y=mb_y,
                )
            elif part_modes[0] == "L1":
                luma_pred, cb_pred, cr_pred = reconstruct_b_l1_16x16(
                    ref_buffer=self.state.ref_buffer,
                    ref_idx=ref_idx_l1[0],
                    mvx=mvx_l1[0], mvy=mvy_l1[0],
                    residual_luma=None, residual_cb=None, residual_cr=None,
                    mb_x=mb_x, mb_y=mb_y,
                )
            else:  # Bi
                luma_pred, cb_pred, cr_pred = reconstruct_b_bi_16x16(
                    ref_buffer=self.state.ref_buffer,
                    ref_idx_l0=ref_idx_l0[0],
                    mvx_l0=mvx_l0[0], mvy_l0=mvy_l0[0],
                    ref_idx_l1=ref_idx_l1[0],
                    mvx_l1=mvx_l1[0], mvy_l1=mvy_l1[0],
                    residual_luma=None, residual_cb=None, residual_cr=None,
                    mb_x=mb_x, mb_y=mb_y,
                    weighted_bipred_idc=weighted_bipred_idc,
                    current_poc=current_poc,
                )

        else:
            # ── Partitioned modes: 16x8 or 8x16 (types 4-21) ──
            ref_idx_l0 = [0, 0]
            ref_idx_l1 = [0, 0]

            # Parse ref_idx — all L0 first, then all L1 (H.264 7.3.5)
            for part in range(2):
                if part_modes[part] in ("L0", "Bi"):
                    if num_l0 > 1:
                        ref_idx_l0[part] = reader.read_te(num_l0 - 1)
            for part in range(2):
                if part_modes[part] in ("L1", "Bi"):
                    if num_l1 > 1:
                        ref_idx_l1[part] = reader.read_te(num_l1 - 1)

            # Parse MVDs — all L0 first, then all L1
            mvx_l0, mvy_l0 = [0, 0], [0, 0]
            mvx_l1, mvy_l1 = [0, 0], [0, 0]

            for part in range(2):
                if part_modes[part] in ("L0", "Bi"):
                    mvd_x = reader.read_se()
                    mvd_y = reader.read_se()
                    if part_size == (16, 8):
                        mvp_x, mvp_y = predict_mv_16x8(
                            self.state.mv_cache, mb_x, mb_y, part
                        )
                    else:
                        mvp_x, mvp_y = predict_mv_8x16(
                            self.state.mv_cache, mb_x, mb_y, part
                        )
                    mvx_l0[part] = mvp_x + mvd_x
                    mvy_l0[part] = mvp_y + mvd_y

                    # Store L0 MV immediately for next partition
                    if part_size == (16, 8):
                        self.state.mv_cache.set_mv_16x8(
                            mb_x, mb_y, part, mvx_l0[part], mvy_l0[part]
                        )
                    else:
                        self.state.mv_cache.set_mv_8x16(
                            mb_x, mb_y, part, mvx_l0[part], mvy_l0[part]
                        )
                else:
                    # L1-only: mark L0 as unavailable for this partition
                    if part_size == (16, 8):
                        self.state.mv_cache.mark_intra_16x8(
                            mb_x, mb_y, part
                        )
                    else:
                        self.state.mv_cache.mark_intra_8x16(
                            mb_x, mb_y, part
                        )

            for part in range(2):
                if part_modes[part] in ("L1", "Bi"):
                    mvd_x = reader.read_se()
                    mvd_y = reader.read_se()
                    if part_size == (16, 8):
                        mvp_x, mvp_y = predict_mv_16x8(
                            self.state.mv_cache_l1, mb_x, mb_y, part
                        )
                    else:
                        mvp_x, mvp_y = predict_mv_8x16(
                            self.state.mv_cache_l1, mb_x, mb_y, part
                        )
                    mvx_l1[part] = mvp_x + mvd_x
                    mvy_l1[part] = mvp_y + mvd_y

                    # Store L1 MV immediately for next partition
                    if self.state.mv_cache_l1 is not None:
                        if part_size == (16, 8):
                            self.state.mv_cache_l1.set_mv_16x8(
                                mb_x, mb_y, part, mvx_l1[part], mvy_l1[part]
                            )
                        else:
                            self.state.mv_cache_l1.set_mv_8x16(
                                mb_x, mb_y, part, mvx_l1[part], mvy_l1[part]
                            )
                else:
                    # L0-only: mark L1 as unavailable for this partition
                    if self.state.mv_cache_l1 is not None:
                        if part_size == (16, 8):
                            self.state.mv_cache_l1.mark_intra_16x8(
                                mb_x, mb_y, part
                            )
                        else:
                            self.state.mv_cache_l1.mark_intra_8x16(
                                mb_x, mb_y, part
                            )

            if part_size == (16, 8):
                luma_pred, cb_pred, cr_pred = reconstruct_b_16x8(
                    ref_buffer=self.state.ref_buffer,
                    pred_modes=part_modes,
                    ref_idx_l0=ref_idx_l0, ref_idx_l1=ref_idx_l1,
                    mvx_l0=mvx_l0, mvy_l0=mvy_l0,
                    mvx_l1=mvx_l1, mvy_l1=mvy_l1,
                    residual_luma=None, residual_cb=None, residual_cr=None,
                    mb_x=mb_x, mb_y=mb_y,
                    weighted_bipred_idc=weighted_bipred_idc,
                    current_poc=current_poc,
                )
            else:
                luma_pred, cb_pred, cr_pred = reconstruct_b_8x16(
                    ref_buffer=self.state.ref_buffer,
                    pred_modes=part_modes,
                    ref_idx_l0=ref_idx_l0, ref_idx_l1=ref_idx_l1,
                    mvx_l0=mvx_l0, mvy_l0=mvy_l0,
                    mvx_l1=mvx_l1, mvy_l1=mvy_l1,
                    residual_luma=None, residual_cb=None, residual_cr=None,
                    mb_x=mb_x, mb_y=mb_y,
                    weighted_bipred_idc=weighted_bipred_idc,
                    current_poc=current_poc,
                )

        # ── Parse coded_block_pattern (H.264 Table 9-4, inter column) ──
        cbp_coded = reader.read_ue()
        cbp_luma, cbp_chroma = decode_cbp_inter(cbp_coded)

        mb_qp_delta = 0
        if cbp_luma > 0 or cbp_chroma > 0:
            mb_qp_delta = reader.read_se()
        current_qp = (qp + mb_qp_delta + 52) % 52

        logger.debug(
            f"  B-MB CBP: luma={cbp_luma:#06b}, chroma={cbp_chroma}, "
            f"qp_delta={mb_qp_delta}, qp={current_qp}"
        )

        # ── Parse luma residual (16 4x4 blocks via CAVLC) ──
        nz_counts = np.zeros(24, dtype=np.int32)
        luma_residual = np.zeros((16, 16), dtype=np.int32)

        for block_idx in range(16):
            block_8x8_idx = block_idx // 4
            if not (cbp_luma & (1 << block_8x8_idx)):
                nz_counts[block_idx] = 0
                continue

            nA, nB = get_luma_neighbor_nz(
                block_idx, mb_x, mb_y, nz_counts,
                self.state.nz_counts, mb_width
            )
            nC = calculate_nC(nA, nB)
            block = decode_residual_block(reader, nC, max_coeffs=16)
            nz_counts[block_idx] = block.total_coeff

            if block.total_coeff > 0:
                coeffs_2d = np.zeros((4, 4), dtype=np.int32)
                for scan_idx in range(16):
                    raster_pos = ZIGZAG_4x4[scan_idx]
                    r, c = raster_pos // 4, raster_pos % 4
                    coeffs_2d[r, c] = block.coefficients[scan_idx]
                dequant = dequant_4x4(coeffs_2d, current_qp)
                residual_block = idct_4x4(dequant)
                row, col = BLOCK_SCAN_ORDER[block_idx]
                luma_residual[row:row+4, col:col+4] = residual_block

        # ── Parse chroma residual ──
        chroma_qp_offset = getattr(pps, 'chroma_qp_index_offset', 0)
        qpi = max(0, min(51, current_qp + chroma_qp_offset))
        qp_chroma = get_chroma_qp(qpi)

        cb_dc = decode_chroma_dc(reader, cbp_chroma, qp_chroma)
        cr_dc = decode_chroma_dc(reader, cbp_chroma, qp_chroma)

        cb_ac = decode_chroma_ac(
            reader, cbp_chroma, nz_counts, 16,
            mb_x, mb_y, self.state.nz_counts, mb_width
        )
        cr_ac = decode_chroma_ac(
            reader, cbp_chroma, nz_counts, 20,
            mb_x, mb_y, self.state.nz_counts, mb_width
        )

        cb_residual = build_chroma_residual(cb_dc, cb_ac, cbp_chroma, qp_chroma)
        cr_residual = build_chroma_residual(cr_dc, cr_ac, cbp_chroma, qp_chroma)

        # ── Combine prediction + residual, write to frame ──
        luma = np.clip(
            luma_pred.astype(np.int32) + luma_residual, 0, 255
        ).astype(np.uint8)
        cb = np.clip(
            cb_pred.astype(np.int32) + cb_residual, 0, 255
        ).astype(np.uint8)
        cr = np.clip(
            cr_pred.astype(np.int32) + cr_residual, 0, 255
        ).astype(np.uint8)

        self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
        self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
        self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

        # Store nz_counts for CAVLC context of neighboring MBs
        self.state.nz_counts[mb_idx] = nz_counts

        # ── Store deblocking metadata ──
        if self.state.mb_types is not None:
            self.state.mb_types[mb_idx] = 99  # inter sentinel
        if self.state.mb_qps is not None:
            self.state.mb_qps[mb_idx] = current_qp
        if self.state.mb_coeffs is not None:
            self.state.mb_coeffs[mb_idx, :16] = (nz_counts[:16] > 0)

        return current_qp

    def _process_b_skip_run(
        self,
        mb_x: int,
        mb_y: int,
        use_spatial: bool,
        current_poc: int,
        weighted_bipred_idc: int = 0,
        current_qp: int = 26,
    ) -> None:
        """Process a B_Skip macroblock.

        B_Skip uses direct mode MV derivation with no residual.

        Args:
            mb_x, mb_y: Macroblock position
            use_spatial: True for spatial direct mode
            current_poc: Picture order count
            weighted_bipred_idc: 0=none, 1=explicit, 2=implicit
        """
        luma, cb, cr = reconstruct_b_skip(
            ref_buffer=self.state.ref_buffer,
            mv_cache=self.state.mv_cache,
            mb_x=mb_x,
            mb_y=mb_y,
            use_spatial=use_spatial,
            current_poc=current_poc,
            mv_cache_l1=self.state.mv_cache_l1,
            weighted_bipred_idc=weighted_bipred_idc,
        )

        ly, lx = mb_y * 16, mb_x * 16
        cy, cx = mb_y * 8, mb_x * 8
        self.state.frame_luma[ly:ly + 16, lx:lx + 16] = luma
        self.state.frame_cb[cy:cy + 8, cx:cx + 8] = cb
        self.state.frame_cr[cy:cy + 8, cx:cx + 8] = cr

        # Store deblocking metadata for B_Skip
        mb_width = self.state.frame_luma.shape[1] // 16
        mb_idx_skip = mb_y * mb_width + mb_x
        self.state.nz_counts[mb_idx_skip] = 0
        if self.state.mb_types is not None:
            self.state.mb_types[mb_idx_skip] = 99  # Non-intra sentinel
        if self.state.mb_qps is not None:
            self.state.mb_qps[mb_idx_skip] = current_qp
        if self.state.mb_coeffs is not None:
            self.state.mb_coeffs[mb_idx_skip] = 0

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
        """Store MVs in reference frame for temporal/spatial direct mode.

        Stores the L0 MV at block (0,0) of each macroblock. Used by the
        colZeroFlag check in spatial direct mode (H.264 8.4.1.2.2).

        Args:
            ref_frame: Reference frame to store MVs in
        """
        if self.state.mv_cache is None:
            return

        sps = self.state.current_sps
        if sps is None:
            return

        # Store per-8x8-block MV and ref_idx for co-located lookups.
        # Map sub-block index to representative 4x4 block position.
        sub_to_4x4 = [(0, 0), (2, 0), (0, 2), (2, 2)]
        for mb_y in range(sps.frame_height_in_mbs):
            for mb_x in range(sps.pic_width_in_mbs):
                for si, (bx, by) in enumerate(sub_to_4x4):
                    mv = self.state.mv_cache.get_mv(mb_x, mb_y, bx, by)
                    ri = self.state.mv_cache.get_ref_idx(mb_x, mb_y, bx, by)
                    ref_frame.store_mv(
                        mb_x, mb_y, mv[0], mv[1],
                        ref_idx=ri, sub_idx=si,
                    )

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

        # H.264 9.3.3.1.1.3: condTermFlagN = 1 if available AND NOT skip
        mb_width = self.state.current_sps.pic_width_in_mbs
        left_cond = False
        top_cond = False

        if mb_x > 0:
            left_idx = mb_y * mb_width + (mb_x - 1)
            left_cond = self.state.mb_types[left_idx] != -1  # NOT skip
        if mb_y > 0:
            top_idx = (mb_y - 1) * mb_width + mb_x
            top_cond = self.state.mb_types[top_idx] != -1

        return decode_mb_skip_flag(
            cabac_decoder, self.state.cabac_contexts,
            slice_type, mb_x, mb_y, left_cond, top_cond
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
