# h264/decoder/tests/test_b_frames.py
"""RED TESTS: B-frame decoder integration.

B-frames require:
- L0/L1 reference list management
- B-slice detection
- B-macroblock type handling
- Direct mode MV derivation
- Bi-directional prediction

H.264 Spec Reference: Section 7.3.3 (B-slice header), Section 8.4 (inter prediction)

These tests SHOULD FAIL until B-frame support is implemented.
"""

import pytest
import numpy as np

from decoder.decoder import H264Decoder, DecoderState


class TestBSliceDetection:
    """Tests for B-slice detection."""

    def test_decoder_detects_b_slice(self):
        """Decoder should detect B-slice type."""
        from slice import SliceType

        # B-slice types are 1 and 6
        assert SliceType.is_b_slice(1) is True
        assert SliceType.is_b_slice(6) is True
        assert SliceType.is_b_slice(0) is False  # P
        assert SliceType.is_b_slice(2) is False  # I

    def test_decoder_has_b_slice_handler(self):
        """Decoder should have _decode_b_slice_data method."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_decode_b_slice_data'), \
            "Decoder should have _decode_b_slice_data method"


class TestBSliceRefLists:
    """Tests for B-slice reference list management."""

    def test_decoder_builds_ref_lists_for_b_slice(self):
        """Decoder should build L0/L1 lists for B-slices."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_build_b_slice_ref_lists'), \
            "Decoder should have _build_b_slice_ref_lists method"

    def test_decoder_state_has_l0_l1_lists(self):
        """DecoderState should track L0/L1 lists."""
        state = DecoderState()

        assert hasattr(state, 'l0_list'), \
            "DecoderState should have l0_list"
        assert hasattr(state, 'l1_list'), \
            "DecoderState should have l1_list"


class TestBMacroblockDecoding:
    """Tests for B-macroblock decoding in decoder."""

    def test_decoder_has_b_mb_decoder(self):
        """Decoder should have _decode_b_macroblock method."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_decode_b_macroblock'), \
            "Decoder should have _decode_b_macroblock method"

    def test_decoder_handles_b_skip_run(self):
        """Decoder should handle mb_skip_run in B-slices."""
        decoder = H264Decoder()

        # B-slices also use mb_skip_run like P-slices
        # but B_Skip uses direct mode instead of P_Skip semantics
        assert hasattr(decoder, '_process_b_skip_run'), \
            "Decoder should have _process_b_skip_run method"


class TestPOCCalculation:
    """Tests for Picture Order Count calculation."""

    def test_decoder_calculates_poc(self):
        """Decoder should calculate POC from slice header."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_calculate_poc'), \
            "Decoder should have _calculate_poc method"

    def test_poc_type_0_calculation(self):
        """POC type 0 uses pic_order_cnt_lsb."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_calc_poc_type_0'), \
            "Decoder should have _calc_poc_type_0 method"

    def test_poc_type_2_calculation(self):
        """POC type 2 derives POC from frame_num."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_calc_poc_type_2'), \
            "Decoder should have _calc_poc_type_2 method"


class TestBFrameDecodeSequence:
    """Tests for decoding I-P-B sequences."""

    def test_ipb_sequence_supported(self):
        """Decoder should support I → P → B decode order."""
        decoder = H264Decoder()

        # Decoder should track decode order vs display order
        assert hasattr(decoder, '_handle_frame_reordering'), \
            "Decoder should handle frame reordering"

    def test_b_frame_between_references(self):
        """B-frame should use I and P as references."""
        decoder = H264Decoder()

        # After decoding I (POC=0) and P (POC=2),
        # B (POC=1) should reference both
        assert hasattr(decoder.state, 'ref_buffer'), \
            "Decoder should have reference buffer"


class TestDirectModeFlagParsing:
    """Tests for direct_spatial_mv_pred_flag parsing."""

    def test_b_slice_has_direct_flag(self):
        """B-slice header should have direct_spatial_mv_pred_flag."""
        from slice import SliceHeader

        # This flag determines spatial vs temporal direct mode
        header = SliceHeader(
            first_mb_in_slice=0,
            slice_type=1,  # B-slice
            pic_parameter_set_id=0,
            frame_num=0,
            slice_qp_delta=0,
            header_bit_size=0,
            direct_spatial_mv_pred_flag=True,
        )

        assert header.direct_spatial_mv_pred_flag is True


class TestWeightedBipredSupport:
    """Tests for weighted bi-prediction in B-frames."""

    def test_decoder_supports_weighted_bipred(self):
        """Decoder should check weighted_bipred_idc in PPS."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_use_weighted_bipred'), \
            "Decoder should have _use_weighted_bipred method"

    def test_weighted_bipred_idc_values(self):
        """weighted_bipred_idc: 0=off, 1=explicit, 2=implicit."""
        from parameters import PPS

        pps = PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
        )

        assert hasattr(pps, 'weighted_bipred_idc'), \
            "PPS should have weighted_bipred_idc"


class TestCollocatedMV:
    """Tests for co-located MV storage for temporal direct."""

    def test_reference_frame_stores_mvs(self):
        """Reference frames should store MV field for temporal direct."""
        from inter.reference import ReferenceFrame

        frame = ReferenceFrame(
            luma=np.zeros((32, 32), dtype=np.uint8),
            cb=np.zeros((16, 16), dtype=np.uint8),
            cr=np.zeros((16, 16), dtype=np.uint8),
            frame_num=0,
        )

        assert hasattr(frame, 'mv_field') or hasattr(frame, 'store_mv'), \
            "ReferenceFrame should support MV storage for temporal direct"

    def test_decoder_stores_mvs_in_ref_frame(self):
        """Decoder should store MVs when adding frame to buffer."""
        decoder = H264Decoder()

        assert hasattr(decoder, '_store_mvs_in_ref'), \
            "Decoder should store MVs in reference frames"
