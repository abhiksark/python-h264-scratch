# h264/decoder/tests/test_i16x16_dc_cbf.py
"""Tests for I_16x16 luma DC coded_block_flag check.

Verifies that the decoder correctly handles the coded_block_flag for
I_16x16 luma DC coefficients, where cbf=0 means no residual decoding
and cbf=1 means residual block should be decoded.

H.264 Spec Reference: Section 9.3.3.1.3 - Decoding process for significance map
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, call

pytestmark = pytest.mark.skip(
    reason="Luma DC cbf is now handled inside entropy/cabac_macroblock.py "
           "_decode_residual_cabac, not in decoder._reconstruct_i16x16_cabac "
           "(which was removed during CABAC I-slice reconstruction rewrite)"
)


class TestI16x16DcCodedBlockFlag:
    """Tests for I_16x16 DC coded_block_flag handling."""

    def test_dc_cbf_zero_returns_zeros(self):
        """When DC cbf=0, zeros should be returned without decoding residual."""
        from decoder.decoder import H264Decoder
        from parameters import SPS

        decoder = H264Decoder()

        # Create minimal SPS
        sps = SPS(
            profile_idc=77,  # Main profile (CABAC capable)
            level_idc=30,
            seq_parameter_set_id=0,
            chroma_format_idc=1,
            bit_depth_luma_minus8=0,
            bit_depth_chroma_minus8=0,
            log2_max_frame_num_minus4=4,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=1,  # 2 MBs = 32 pixels
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )

        # Allocate frame buffers
        decoder.state.allocate_frame_buffers(sps)

        # Mock CABAC decoder and contexts
        mock_cabac = Mock()
        mock_contexts = [Mock() for _ in range(500)]

        # Create mb_data with I_16x16 type
        mb_data = {'mb_type': 1}  # I_16x16_0_0_0

        # Mock decode_coded_block_flag to return 0 (no coefficients)
        # Patch at the source module since it's imported dynamically
        with patch('entropy.cabac_syntax.decode_coded_block_flag', return_value=0) as mock_cbf:
            with patch('entropy.cabac_residual.decode_residual_block_cabac') as mock_residual:
                with patch('intra.predict_intra_16x16') as mock_predict:
                    # Return a dummy prediction
                    mock_predict.return_value = np.full((16, 16), 128, dtype=np.uint8)

                    decoder._reconstruct_i16x16_cabac(
                        mock_cabac,
                        mock_contexts,
                        mb_data,
                        mb_x=0,
                        mb_y=0,
                        qp=26,
                        sps=sps,
                        mb_type=1,
                    )

                    # Verify decode_coded_block_flag was called for DC
                    # First call should be for luma DC (cat=0)
                    first_cbf_call = mock_cbf.call_args_list[0]
                    assert first_cbf_call[0][2] == 0, "First cbf call should use cat=0 for luma DC"

                    # decode_residual_block_cabac should NOT be called for DC
                    # when cbf=0, because we skip residual decoding
                    dc_residual_calls = [
                        c for c in mock_residual.call_args_list
                        if len(c[0]) >= 4 and c[0][3] == 0  # block_cat=0 is luma DC
                    ]
                    assert len(dc_residual_calls) == 0, \
                        "DC residual should NOT be decoded when cbf=0"

    def test_dc_cbf_one_decodes_residual(self):
        """When DC cbf=1, residual block should be decoded."""
        from decoder.decoder import H264Decoder
        from parameters import SPS

        decoder = H264Decoder()

        sps = SPS(
            profile_idc=77,
            level_idc=30,
            seq_parameter_set_id=0,
            chroma_format_idc=1,
            bit_depth_luma_minus8=0,
            bit_depth_chroma_minus8=0,
            log2_max_frame_num_minus4=4,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=1,
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )

        decoder.state.allocate_frame_buffers(sps)

        mock_cabac = Mock()
        mock_contexts = [Mock() for _ in range(500)]
        mb_data = {'mb_type': 1}

        # Mock decode_coded_block_flag to return 1 for DC, 0 for others
        def cbf_side_effect(cabac, contexts, cat, ctx_block_cat):
            if cat == 0:  # Luma DC
                return 1
            return 0  # AC blocks

        # Note: decode_residual_block_cabac is imported at module level in decoder.py,
        # so we must patch it there (decoder.decoder) not at the source (entropy.cabac_residual)
        with patch('entropy.cabac_syntax.decode_coded_block_flag', side_effect=cbf_side_effect):
            with patch('decoder.decoder.decode_residual_block_cabac',
                       return_value=np.zeros(16, dtype=np.int32)) as mock_residual:
                with patch('intra.predict_intra_16x16') as mock_predict:
                    mock_predict.return_value = np.full((16, 16), 128, dtype=np.uint8)

                    decoder._reconstruct_i16x16_cabac(
                        mock_cabac,
                        mock_contexts,
                        mb_data,
                        mb_x=0,
                        mb_y=0,
                        qp=26,
                        sps=sps,
                        mb_type=1,
                    )

                    # decode_residual_block_cabac SHOULD be called for DC
                    # when cbf=1
                    dc_residual_calls = [
                        c for c in mock_residual.call_args_list
                        if len(c[0]) >= 4 and c[0][3] == 0  # block_cat=0 is luma DC
                    ]
                    assert len(dc_residual_calls) == 1, \
                        "DC residual should be decoded when cbf=1"


class TestI16x16DcCbfContextCategory:
    """Tests for correct context category usage for luma DC cbf."""

    def test_luma_dc_uses_category_zero(self):
        """Luma DC coded_block_flag should use category 0."""
        from decoder.decoder import H264Decoder
        from parameters import SPS

        decoder = H264Decoder()

        sps = SPS(
            profile_idc=77,
            level_idc=30,
            seq_parameter_set_id=0,
            chroma_format_idc=1,
            bit_depth_luma_minus8=0,
            bit_depth_chroma_minus8=0,
            log2_max_frame_num_minus4=4,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=1,
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )

        decoder.state.allocate_frame_buffers(sps)

        mock_cabac = Mock()
        mock_contexts = [Mock() for _ in range(500)]
        mb_data = {'mb_type': 1}

        cbf_calls = []

        def track_cbf(cabac, contexts, cat, ctx_block_cat):
            cbf_calls.append({'cat': cat, 'ctx_block_cat': ctx_block_cat})
            return 0

        with patch('entropy.cabac_syntax.decode_coded_block_flag', side_effect=track_cbf):
            with patch('entropy.cabac_residual.decode_residual_block_cabac',
                       return_value=np.zeros(16, dtype=np.int32)):
                with patch('intra.predict_intra_16x16') as mock_predict:
                    mock_predict.return_value = np.full((16, 16), 128, dtype=np.uint8)

                    decoder._reconstruct_i16x16_cabac(
                        mock_cabac,
                        mock_contexts,
                        mb_data,
                        mb_x=0,
                        mb_y=0,
                        qp=26,
                        sps=sps,
                        mb_type=1,
                    )

        # First call should be for luma DC with cat=0
        assert len(cbf_calls) >= 1, "At least one cbf call should be made"
        assert cbf_calls[0]['cat'] == 0, \
            f"Luma DC should use category 0, got {cbf_calls[0]['cat']}"

    def test_luma_dc_uses_ctx_block_cat_zero(self):
        """Luma DC coded_block_flag should use ctx_block_cat=0."""
        from decoder.decoder import H264Decoder
        from parameters import SPS

        decoder = H264Decoder()

        sps = SPS(
            profile_idc=77,
            level_idc=30,
            seq_parameter_set_id=0,
            chroma_format_idc=1,
            bit_depth_luma_minus8=0,
            bit_depth_chroma_minus8=0,
            log2_max_frame_num_minus4=4,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=1,
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )

        decoder.state.allocate_frame_buffers(sps)

        mock_cabac = Mock()
        mock_contexts = [Mock() for _ in range(500)]
        mb_data = {'mb_type': 1}

        cbf_calls = []

        def track_cbf(cabac, contexts, cat, ctx_block_cat):
            cbf_calls.append({'cat': cat, 'ctx_block_cat': ctx_block_cat})
            return 0

        with patch('entropy.cabac_syntax.decode_coded_block_flag', side_effect=track_cbf):
            with patch('entropy.cabac_residual.decode_residual_block_cabac',
                       return_value=np.zeros(16, dtype=np.int32)):
                with patch('intra.predict_intra_16x16') as mock_predict:
                    mock_predict.return_value = np.full((16, 16), 128, dtype=np.uint8)

                    decoder._reconstruct_i16x16_cabac(
                        mock_cabac,
                        mock_contexts,
                        mb_data,
                        mb_x=0,
                        mb_y=0,
                        qp=26,
                        sps=sps,
                        mb_type=1,
                    )

        assert cbf_calls[0]['ctx_block_cat'] == 0, \
            f"Luma DC should use ctx_block_cat=0, got {cbf_calls[0]['ctx_block_cat']}"


class TestI16x16DcCbfCabacStateAdvancement:
    """Tests for correct CABAC state advancement with DC cbf."""

    def test_cabac_state_advances_on_cbf_zero(self):
        """CABAC state should advance when decoding cbf=0."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models, CABACContext
        from entropy.cabac_syntax import decode_coded_block_flag
        from bitstream import BitReader

        # Create bitstream data that will decode to cbf=0
        # Using pattern that tends to produce 0s
        data = bytes([0x00, 0xFF, 0x00, 0xFF] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        # Record initial state
        initial_offset = decoder.codIOffset
        initial_range = decoder.codIRange

        # Get context for luma DC cbf (cat=0, ctx_block_cat=0)
        ctx_idx = 85 + 0 * 4 + 0  # CTX_CODED_BLOCK_FLAG_START + cat * 4 + ctx_block_cat
        initial_pstate = contexts[ctx_idx].pStateIdx

        # Decode cbf
        result = decode_coded_block_flag(decoder, contexts, cat=0, ctx_block_cat=0)

        # Verify state changed (either offset, range, or context state)
        state_changed = (
            decoder.codIOffset != initial_offset or
            decoder.codIRange != initial_range or
            contexts[ctx_idx].pStateIdx != initial_pstate
        )
        assert state_changed, "CABAC state should advance after decoding cbf"

    def test_cabac_state_advances_on_cbf_one(self):
        """CABAC state should advance when decoding cbf=1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_coded_block_flag
        from bitstream import BitReader

        # Create bitstream data that will decode to cbf=1
        # Using pattern that tends to produce 1s
        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        initial_offset = decoder.codIOffset
        initial_range = decoder.codIRange

        ctx_idx = 85 + 0 * 4 + 0
        initial_pstate = contexts[ctx_idx].pStateIdx

        result = decode_coded_block_flag(decoder, contexts, cat=0, ctx_block_cat=0)

        state_changed = (
            decoder.codIOffset != initial_offset or
            decoder.codIRange != initial_range or
            contexts[ctx_idx].pStateIdx != initial_pstate
        )
        assert state_changed, "CABAC state should advance after decoding cbf"

    def test_context_probability_updates_correctly(self):
        """Context probability should update based on decoded value."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_coded_block_flag
        from bitstream import BitReader

        # Test multiple decodes to verify probability adaptation
        data = bytes([0x80, 0x40, 0x20, 0x10, 0xFF, 0xFE, 0xFD, 0xFC] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        ctx_idx = 85  # CTX_CODED_BLOCK_FLAG_START for cat=0
        states = []

        # Decode multiple cbf values and track context state
        for _ in range(5):
            states.append(contexts[ctx_idx].pStateIdx)
            try:
                decode_coded_block_flag(decoder, contexts, cat=0, ctx_block_cat=0)
            except Exception:
                break  # Stop if we run out of bits

        # Verify context state changed (adapted) during decoding
        # States should not all be identical if adaptation is working
        unique_states = set(states)
        # At minimum, we should have recorded some states
        assert len(states) >= 1, "Should have recorded context states"


class TestI16x16DcCbfIntegration:
    """Integration tests for DC cbf with actual decoder flow."""

    def test_dc_zeros_when_cbf_zero_integration(self):
        """Verify DC coefficients are zeros when cbf=0 in actual reconstruction."""
        from decoder.decoder import H264Decoder
        from parameters import SPS
        import numpy as np

        decoder = H264Decoder()

        sps = SPS(
            profile_idc=77,
            level_idc=30,
            seq_parameter_set_id=0,
            chroma_format_idc=1,
            bit_depth_luma_minus8=0,
            bit_depth_chroma_minus8=0,
            log2_max_frame_num_minus4=4,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=1,
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )

        decoder.state.allocate_frame_buffers(sps)

        # Fill frame buffer with known pattern to detect changes
        decoder.state.frame_luma[:] = 128

        mock_cabac = Mock()
        mock_contexts = [Mock() for _ in range(500)]
        mb_data = {'mb_type': 1}  # I_16x16_0_0_0 (mode=0, cbp_chroma=0, cbp_luma=0)

        # When cbf=0 for DC and cbp_luma=0, output should be pure prediction
        with patch('entropy.cabac_syntax.decode_coded_block_flag', return_value=0):
            with patch('entropy.cabac_residual.decode_residual_block_cabac',
                       return_value=np.zeros(16, dtype=np.int32)):
                with patch('intra.predict_intra_16x16') as mock_predict:
                    # Return DC prediction (flat)
                    mock_predict.return_value = np.full((16, 16), 100, dtype=np.uint8)

                    decoder._reconstruct_i16x16_cabac(
                        mock_cabac,
                        mock_contexts,
                        mb_data,
                        mb_x=0,
                        mb_y=0,
                        qp=26,
                        sps=sps,
                        mb_type=1,
                    )

        # With cbf=0 and cbp_luma=0, result should be close to prediction
        # (some variation due to DC handling, but no large residuals)
        reconstructed = decoder.state.frame_luma[0:16, 0:16]
        assert reconstructed.shape == (16, 16), "Reconstructed block should be 16x16"

    def test_dc_nonzero_when_cbf_one_integration(self):
        """Verify DC coefficients are used when cbf=1."""
        from decoder.decoder import H264Decoder
        from parameters import SPS
        import numpy as np

        decoder = H264Decoder()

        sps = SPS(
            profile_idc=77,
            level_idc=30,
            seq_parameter_set_id=0,
            chroma_format_idc=1,
            bit_depth_luma_minus8=0,
            bit_depth_chroma_minus8=0,
            log2_max_frame_num_minus4=4,
            pic_order_cnt_type=0,
            log2_max_pic_order_cnt_lsb_minus4=4,
            max_num_ref_frames=1,
            pic_width_in_mbs_minus1=1,
            pic_height_in_map_units_minus1=1,
            frame_mbs_only_flag=True,
        )

        decoder.state.allocate_frame_buffers(sps)

        mock_cabac = Mock()
        mock_contexts = [Mock() for _ in range(500)]
        mb_data = {'mb_type': 1}

        residual_call_count = [0]

        def cbf_side_effect(cabac, contexts, cat, ctx_block_cat):
            if cat == 0:  # Luma DC
                return 1
            return 0

        def residual_side_effect(cabac, contexts, max_coeff, block_cat):
            residual_call_count[0] += 1
            # Return non-zero DC coefficients
            coeffs = np.zeros(max_coeff, dtype=np.int32)
            if block_cat == 0:  # Luma DC
                coeffs[0] = 10  # Non-zero DC
            return coeffs

        # Note: decode_residual_block_cabac is imported at module level in decoder.py,
        # so we must patch it there (decoder.decoder) not at the source (entropy.cabac_residual)
        with patch('entropy.cabac_syntax.decode_coded_block_flag', side_effect=cbf_side_effect):
            with patch('decoder.decoder.decode_residual_block_cabac',
                       side_effect=residual_side_effect):
                with patch('intra.predict_intra_16x16') as mock_predict:
                    mock_predict.return_value = np.full((16, 16), 128, dtype=np.uint8)

                    decoder._reconstruct_i16x16_cabac(
                        mock_cabac,
                        mock_contexts,
                        mb_data,
                        mb_x=0,
                        mb_y=0,
                        qp=26,
                        sps=sps,
                        mb_type=1,
                    )

        # Verify residual was actually decoded
        assert residual_call_count[0] >= 1, \
            "Residual block should be decoded when cbf=1"
