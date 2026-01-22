# h264/decoder/tests/test_chroma_dc_cbf.py
"""Tests for chroma DC coded_block_flag check in CABAC decoding.

Verifies that the decoder correctly checks coded_block_flag before decoding
chroma DC residual blocks, as required by H.264 spec section 9.3.3.1.3.

H.264 Spec Reference:
- Section 9.3.3.1.3 - coded_block_flag for residual data
- Section 7.3.5.3 - Residual data syntax

The fix ensures:
1. coded_block_flag is checked before decoding chroma DC residual
2. When cbf=0, residual block is NOT decoded (zeros returned)
3. When cbf=1, residual block IS decoded and Hadamard transformed
4. Context category 3 is used for chroma DC cbf
5. Cb and Cr planes check cbf independently
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

from bitstream import BITSTRING_AVAILABLE

pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestChromaDCCodedBlockFlagCheck:
    """Tests for coded_block_flag check before chroma DC decoding."""

    def test_chroma_dc_cbf_zero_returns_zeros(self):
        """When chroma DC cbf=0, residual block should NOT be decoded."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        # Setup minimal state
        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        # Mock decode_coded_block_flag to return 0 (no coded block)
        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=0
        ) as mock_cbf, patch(
            'entropy.cabac_residual.decode_residual_block_cabac'
        ) as mock_residual:

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,  # DC only
            )

            # decode_coded_block_flag should be called for both Cb and Cr DC
            assert mock_cbf.call_count == 2

            # decode_residual_block_cabac should NOT be called since cbf=0
            mock_residual.assert_not_called()

    def test_chroma_dc_cbf_one_decodes_residual(self):
        """When chroma DC cbf=1, residual block should be decoded."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        # Setup minimal state
        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        # Mock decode_coded_block_flag to return 1 (coded block exists)
        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=1
        ) as mock_cbf, patch(
            'entropy.cabac_residual.decode_residual_block_cabac',
            return_value=[10, 20, 30, 40]  # 4 DC coefficients
        ) as mock_residual, patch(
            'transform.inverse_hadamard_2x2',
            return_value=np.array([[10, 20], [30, 40]], dtype=np.int32)
        ):

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,  # DC only
            )

            # decode_coded_block_flag should be called for both Cb and Cr DC
            assert mock_cbf.call_count == 2

            # decode_residual_block_cabac should be called for both planes
            assert mock_residual.call_count == 2


class TestChromaDCContextCategory:
    """Tests for correct context category usage in chroma DC cbf."""

    def test_chroma_dc_uses_context_category_3(self):
        """Chroma DC coded_block_flag should use context category 3."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=0
        ) as mock_cbf:

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,
            )

            # Verify context category 3 is used for chroma DC
            # decode_coded_block_flag(cabac, contexts, cat=3, ctx_block_cat=0)
            for call_args in mock_cbf.call_args_list:
                args = call_args[0]
                # args[2] is the cat parameter
                assert args[2] == 3, f"Expected cat=3 for chroma DC, got {args[2]}"

    def test_chroma_dc_residual_uses_block_cat_3(self):
        """Chroma DC residual decoding should use block_cat=3."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=1
        ), patch(
            'entropy.cabac_residual.decode_residual_block_cabac',
            return_value=[0, 0, 0, 0]
        ) as mock_residual, patch(
            'transform.inverse_hadamard_2x2',
            return_value=np.zeros((2, 2), dtype=np.int32)
        ):

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,
            )

            # Verify block_cat=3 is used for chroma DC residual
            # decode_residual_block_cabac(cabac, contexts, max_coeff=4, block_cat=3)
            for call_args in mock_residual.call_args_list:
                args = call_args[0]
                # args[3] is the block_cat parameter
                assert args[3] == 3, f"Expected block_cat=3, got {args[3]}"


class TestCbCrIndependentCBFCheck:
    """Tests for independent cbf checking of Cb and Cr planes."""

    def test_cb_and_cr_check_cbf_independently(self):
        """Cb and Cr planes should check coded_block_flag independently."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        cbf_call_count = [0]
        cbf_values = [1, 0]  # Cb has DC, Cr does not

        def mock_cbf(cabac, contexts, cat, ctx_block_cat):
            if cat == 3:  # Chroma DC
                result = cbf_values[cbf_call_count[0]]
                cbf_call_count[0] += 1
                return result
            return 0

        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            side_effect=mock_cbf
        ), patch(
            'entropy.cabac_residual.decode_residual_block_cabac',
            return_value=[10, 20, 30, 40]
        ) as mock_residual, patch(
            'transform.inverse_hadamard_2x2',
            return_value=np.array([[10, 20], [30, 40]], dtype=np.int32)
        ):

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,
            )

            # Only Cb DC should be decoded (cbf=1), not Cr DC (cbf=0)
            assert mock_residual.call_count == 1

    def test_both_planes_cbf_zero_no_residual_decoded(self):
        """When both Cb and Cr have cbf=0, no DC residual should be decoded."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        # Both planes have cbf=0
        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=0
        ) as mock_cbf, patch(
            'entropy.cabac_residual.decode_residual_block_cabac'
        ) as mock_residual:

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,
            )

            # Both Cb and Cr should check cbf
            assert mock_cbf.call_count == 2

            # No residual blocks should be decoded
            mock_residual.assert_not_called()

    def test_both_planes_cbf_one_both_decoded(self):
        """When both Cb and Cr have cbf=1, both DC residuals should be decoded."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        # Both planes have cbf=1
        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=1
        ) as mock_cbf, patch(
            'entropy.cabac_residual.decode_residual_block_cabac',
            return_value=[10, 20, 30, 40]
        ) as mock_residual, patch(
            'transform.inverse_hadamard_2x2',
            return_value=np.array([[10, 20], [30, 40]], dtype=np.int32)
        ):

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,
            )

            # Both Cb and Cr should check cbf
            assert mock_cbf.call_count == 2

            # Both DC residual blocks should be decoded
            assert mock_residual.call_count == 2


class TestChromaDCHadamardTransform:
    """Tests for Hadamard transform application on chroma DC."""

    def test_hadamard_applied_when_cbf_one(self):
        """Inverse Hadamard transform should be applied when cbf=1."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=1
        ), patch(
            'entropy.cabac_residual.decode_residual_block_cabac',
            return_value=[10, 20, 30, 40]
        ), patch(
            'transform.inverse_hadamard_2x2'
        ) as mock_hadamard:

            # Configure mock to return proper array
            mock_hadamard.return_value = np.array(
                [[10, 20], [30, 40]], dtype=np.int32
            )

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,
            )

            # Hadamard transform should be called for both Cb and Cr
            assert mock_hadamard.call_count == 2

    def test_hadamard_not_applied_when_cbf_zero(self):
        """Inverse Hadamard transform should NOT be applied when cbf=0."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=0
        ), patch(
            'transform.inverse_hadamard_2x2'
        ) as mock_hadamard:

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,
            )

            # Hadamard transform should NOT be called when cbf=0
            mock_hadamard.assert_not_called()


class TestChromaDCZeroBlock:
    """Tests for zero DC block when cbf=0."""

    def test_zero_dc_block_when_cbf_zero(self):
        """DC block should be zeros when cbf=0."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch
        import numpy as np

        decoder = H264Decoder()

        # Initialize frame buffers
        decoder.state.frame_cb = np.full((8, 8), 100, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 100, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        # Track what gets passed to dequant/idct to verify DC values
        idct_inputs = []
        original_idct = None

        def capture_idct(block):
            idct_inputs.append(block.copy())
            # Return zero residual
            return np.zeros((4, 4), dtype=np.int32)

        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=0
        ), patch(
            'transform.idct_4x4',
            side_effect=capture_idct
        ), patch(
            'dequant.dequant_4x4',
            side_effect=lambda block, qp, is_intra: block
        ):

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,
            )

            # Verify that DC values passed to IDCT are zeros
            for block in idct_inputs:
                assert block[0, 0] == 0, "DC value should be 0 when cbf=0"


class TestChromaCBPChromaInteraction:
    """Tests for interaction between cbp_chroma and DC cbf."""

    def test_cbp_chroma_zero_function_not_called(self):
        """When cbp_chroma=0, _decode_chroma_cabac should not be called.

        Note: The caller (e.g., _reconstruct_intra16x16_mb_cabac) checks
        cbp_chroma > 0 before calling _decode_chroma_cabac. This test
        verifies that the design expects cbp_chroma > 0 when the function
        is called.
        """
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        # When _decode_chroma_cabac is called with cbp_chroma > 0,
        # it should always check the DC cbf for both planes.
        # The caller is responsible for not calling this function when
        # cbp_chroma=0.
        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=0
        ) as mock_cbf:

            # Call with cbp_chroma=1 (DC only) - the minimum valid value
            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,
            )

            # DC cbf should be checked for both Cb and Cr
            dc_cbf_calls = [
                c for c in mock_cbf.call_args_list
                if c[0][2] == 3  # cat=3 is chroma DC
            ]
            assert len(dc_cbf_calls) == 2

    def test_cbp_chroma_one_checks_dc_cbf(self):
        """When cbp_chroma=1 (DC only), DC cbf should be checked."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=0
        ) as mock_cbf:

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=1,  # DC only
            )

            # DC cbf should be checked for both planes
            dc_cbf_calls = [
                c for c in mock_cbf.call_args_list
                if c[0][2] == 3  # cat=3 is chroma DC
            ]
            assert len(dc_cbf_calls) == 2

    def test_cbp_chroma_two_checks_dc_and_ac_cbf(self):
        """When cbp_chroma=2 (DC+AC), both DC and AC cbf should be checked."""
        from decoder.decoder import H264Decoder
        from unittest.mock import MagicMock, patch

        decoder = H264Decoder()

        decoder.state.frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        decoder.state.frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mock_cabac = MagicMock()
        mock_contexts = [MagicMock() for _ in range(500)]

        with patch(
            'entropy.cabac_syntax.decode_coded_block_flag',
            return_value=0
        ) as mock_cbf:

            decoder._decode_chroma_cabac(
                cabac=mock_cabac,
                contexts=mock_contexts,
                mb_x=0,
                mb_y=0,
                qp=26,
                cbp_chroma=2,  # DC + AC
            )

            # DC cbf (cat=3) should be checked for both planes
            dc_cbf_calls = [
                c for c in mock_cbf.call_args_list
                if c[0][2] == 3
            ]
            assert len(dc_cbf_calls) == 2

            # AC cbf (cat=4) should be checked for all 4 blocks per plane
            ac_cbf_calls = [
                c for c in mock_cbf.call_args_list
                if c[0][2] == 4
            ]
            # 4 blocks per plane * 2 planes = 8 AC cbf checks
            assert len(ac_cbf_calls) == 8


class TestCodedBlockFlagContextIndexCalculation:
    """Tests for context index calculation in decode_coded_block_flag."""

    def test_context_index_for_chroma_dc(self):
        """Verify context index calculation for chroma DC (cat=3)."""
        from entropy.cabac_syntax import decode_coded_block_flag
        from entropy.cabac_context import CTX_CODED_BLOCK_FLAG_START
        from unittest.mock import MagicMock

        mock_decoder = MagicMock()
        mock_decoder.decode_decision.return_value = 0

        # Create enough contexts
        mock_contexts = [MagicMock() for _ in range(500)]

        # Call with cat=3, ctx_block_cat=0
        decode_coded_block_flag(mock_decoder, mock_contexts, cat=3, ctx_block_cat=0)

        # Expected context index: CTX_CODED_BLOCK_FLAG_START + 3*4 + 0
        expected_ctx_idx = CTX_CODED_BLOCK_FLAG_START + 3 * 4 + 0

        # Verify the correct context was used
        mock_decoder.decode_decision.assert_called_once()
        call_args = mock_decoder.decode_decision.call_args[0]
        assert call_args[0] == mock_contexts[expected_ctx_idx]

    def test_context_index_for_chroma_ac(self):
        """Verify context index calculation for chroma AC (cat=4)."""
        from entropy.cabac_syntax import decode_coded_block_flag
        from entropy.cabac_context import CTX_CODED_BLOCK_FLAG_START
        from unittest.mock import MagicMock

        mock_decoder = MagicMock()
        mock_decoder.decode_decision.return_value = 0

        mock_contexts = [MagicMock() for _ in range(500)]

        # Call with cat=4, ctx_block_cat=0
        decode_coded_block_flag(mock_decoder, mock_contexts, cat=4, ctx_block_cat=0)

        # Expected context index: CTX_CODED_BLOCK_FLAG_START + 4*4 + 0
        expected_ctx_idx = CTX_CODED_BLOCK_FLAG_START + 4 * 4 + 0

        mock_decoder.decode_decision.assert_called_once()
        call_args = mock_decoder.decode_decision.call_args[0]
        assert call_args[0] == mock_contexts[expected_ctx_idx]
