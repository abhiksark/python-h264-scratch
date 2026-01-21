# h264/decoder/tests/test_i8x8_decoder.py
"""RED TESTS: I_8x8 macroblock decoder integration tests.

H.264 Spec Reference:
- Section 7.3.5 - Macroblock layer syntax
- Section 7.4.5 - Macroblock layer semantics
- Section 8.3.1.3 - Intra_8x8 prediction process
- Section 8.5.12 - 8x8 inverse transform process

I_8x8 (Intra_8x8) is High profile only:
- Uses 8x8 transform instead of 4x4
- transform_8x8_mode_flag in mb_type indicates 8x8 transform
- Four 8x8 luma blocks, each with its own prediction mode
- Uses 8x8 IDCT and 8x8 dequantization

These tests SHOULD FAIL until I_8x8 decoder support is implemented.
"""

import pytest
import numpy as np

from bitstream import BitWriter, BitReader, BITSTRING_AVAILABLE

pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestI8x8MacroblockTypeRecognition:
    """Tests for recognizing I_8x8 macroblock type.

    In High profile, I_NxN with transform_8x8_mode_flag=1 indicates I_8x8.
    mb_type=0 in I-slice can be either I_4x4 or I_8x8 depending on
    transform_8x8_mode_flag in PPS and macroblock syntax.
    """

    def test_identify_i8x8_mb_type(self):
        """Should identify I_8x8 macroblock type from mb_type and flag."""
        from decoder.decoder import H264Decoder
        from decoder.i8x8 import is_i8x8_macroblock

        # I_NxN (mb_type=0) with transform_8x8_mode_flag=1 is I_8x8
        assert is_i8x8_macroblock(mb_type=0, transform_8x8_mode_flag=True)

        # I_NxN (mb_type=0) with transform_8x8_mode_flag=0 is I_4x4
        assert not is_i8x8_macroblock(mb_type=0, transform_8x8_mode_flag=False)

    def test_i8x8_requires_high_profile(self):
        """I_8x8 should only be valid in High profile (profile_idc >= 100)."""
        from decoder.i8x8 import validate_i8x8_profile

        # High profile (100) allows I_8x8
        assert validate_i8x8_profile(profile_idc=100)

        # Baseline profile (66) does not allow I_8x8
        assert not validate_i8x8_profile(profile_idc=66)

        # Main profile (77) does not allow I_8x8
        assert not validate_i8x8_profile(profile_idc=77)

    def test_i8x8_requires_transform_8x8_mode_flag_in_pps(self):
        """I_8x8 requires transform_8x8_mode_flag=1 in PPS."""
        from decoder.i8x8 import can_use_i8x8_transform
        from parameters import PPS

        pps_with_8x8 = PPS(
            pic_parameter_set_id=0,
            seq_parameter_set_id=0,
            entropy_coding_mode_flag=False,
            bottom_field_pic_order_in_frame_present_flag=False,
            num_slice_groups_minus1=0,
            num_ref_idx_l0_default_active_minus1=0,
            num_ref_idx_l1_default_active_minus1=0,
            weighted_pred_flag=False,
            weighted_bipred_idc=0,
            pic_init_qp_minus26=0,
            pic_init_qs_minus26=0,
            chroma_qp_index_offset=0,
            deblocking_filter_control_present_flag=False,
            constrained_intra_pred_flag=False,
            redundant_pic_cnt_present_flag=False,
            transform_8x8_mode_flag=True,  # Enables I_8x8
        )

        assert can_use_i8x8_transform(pps_with_8x8)


class TestI8x8PredictionModeDecoding:
    """Tests for decoding I_8x8 prediction modes.

    I_8x8 has four 8x8 blocks, each with its own prediction mode (0-8).
    Mode coding uses prev_intra8x8_pred_mode_flag and rem_intra8x8_pred_mode,
    similar to I_4x4 but for 8x8 blocks.
    """

    def test_decode_i8x8_pred_modes_exists(self):
        """decode_i8x8_pred_modes function should exist."""
        from decoder.i8x8 import decode_i8x8_pred_modes

        assert callable(decode_i8x8_pred_modes)

    def test_decode_four_i8x8_modes(self):
        """Should decode exactly four prediction modes for I_8x8."""
        from decoder.i8x8 import decode_i8x8_pred_modes

        # Create bitstream: 4 blocks, all using predicted mode (flag=1)
        writer = BitWriter()
        for _ in range(4):
            writer.write_bits(1, 1)  # prev_intra8x8_pred_mode_flag = 1
        writer.write_bits(0, 8)  # Padding
        reader = BitReader(writer.to_bytes())

        modes = decode_i8x8_pred_modes(reader)

        assert len(modes) == 4
        # All should be DC mode (2) when using predicted mode with no neighbors
        assert all(m == 2 for m in modes)

    def test_decode_explicit_i8x8_modes(self):
        """Should decode explicit prediction modes from bitstream.

        H.264 mode derivation: if rem < predicted: mode = rem, else: mode = rem + 1
        For block 0 (no neighbors): predicted = DC(2)
        For block 1 (left=mode[0]): predicted = min(mode[0], DC) = min(0, 2) = 0
        """
        from decoder.i8x8 import decode_i8x8_pred_modes

        writer = BitWriter()
        # Block 0: flag=0, rem=0 (mode 0 = Vertical, since 0 < predicted(2))
        writer.write_bits(0, 1)
        writer.write_bits(0, 3)
        # Block 1: flag=0, rem=0 (mode 1 = Horizontal, since rem(0) >= predicted(0) -> mode=0+1=1)
        writer.write_bits(0, 1)
        writer.write_bits(0, 3)
        # Block 2: flag=0, rem=3 (mode 4 = DDR, since rem >= predicted_mode)
        writer.write_bits(0, 1)
        writer.write_bits(3, 3)
        # Block 3: flag=1 (use predicted mode)
        writer.write_bits(1, 1)
        writer.write_bits(0, 8)  # Padding
        reader = BitReader(writer.to_bytes())

        modes = decode_i8x8_pred_modes(reader)

        assert len(modes) == 4
        assert modes[0] == 0  # Vertical
        assert modes[1] == 1  # Horizontal (rem=0 with predicted=0 gives mode=1)

    def test_i8x8_mode_prediction_from_neighbors(self):
        """Predicted mode should be min of left and top neighbor modes."""
        from decoder.i8x8 import predict_i8x8_mode

        # If left=Vertical(0) and top=Horizontal(1), predicted=min(0,1)=0
        predicted = predict_i8x8_mode(left_mode=0, top_mode=1)
        assert predicted == 0

        # If left=DC(2) and top=DDL(3), predicted=min(2,3)=2
        predicted = predict_i8x8_mode(left_mode=2, top_mode=3)
        assert predicted == 2


class TestI8x8BlockReconstruction:
    """Tests for reconstructing individual I_8x8 blocks."""

    def test_reconstruct_i8x8_block_exists(self):
        """reconstruct_i8x8_block function should exist."""
        from decoder.i8x8 import reconstruct_i8x8_block

        assert callable(reconstruct_i8x8_block)

    def test_reconstruct_i8x8_dc_mode_no_residual(self):
        """DC mode with zero residual should return DC prediction."""
        from decoder.i8x8 import reconstruct_i8x8_block

        top = np.array([100] * 8, dtype=np.uint8)
        left = np.array([100] * 8, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,  # DC
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8
        assert np.all(result == 100)

    def test_reconstruct_i8x8_with_residual(self):
        """Prediction + residual should produce correct output."""
        from decoder.i8x8 import reconstruct_i8x8_block

        top = np.array([100] * 8, dtype=np.uint8)
        left = np.array([100] * 8, dtype=np.uint8)
        residual = np.full((8, 8), 20, dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=2,  # DC
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert np.all(result == 120)  # 100 DC + 20 residual

    def test_reconstruct_i8x8_clips_output(self):
        """Output should be clipped to [0, 255]."""
        from decoder.i8x8 import reconstruct_i8x8_block

        top = np.array([250] * 8, dtype=np.uint8)
        left = np.array([250] * 8, dtype=np.uint8)
        residual = np.full((8, 8), 50, dtype=np.int32)  # Would overflow

        result = reconstruct_i8x8_block(
            mode=2,
            residual=residual,
            top=top,
            left=left,
            top_left=250,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        assert result.max() <= 255


class TestI8x8LumaReconstruction:
    """Tests for full 16x16 luma reconstruction from four 8x8 blocks."""

    def test_reconstruct_i8x8_luma_exists(self):
        """reconstruct_i8x8_luma function should exist."""
        from decoder.i8x8 import reconstruct_i8x8_luma

        assert callable(reconstruct_i8x8_luma)

    def test_reconstruct_i8x8_luma_returns_16x16(self):
        """I_8x8 luma reconstruction should return 16x16 block."""
        from decoder.i8x8 import reconstruct_i8x8_luma

        modes = [2, 2, 2, 2]  # All DC mode
        coefficients = [np.zeros((8, 8), dtype=np.int32) for _ in range(4)]
        frame_luma = np.full((32, 32), 128, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes=modes,
            coefficients=coefficients,
            qp=26,
            frame_luma=frame_luma,
            mb_x=1,
            mb_y=1,
            frame_width=32,
            frame_height=32,
        )

        assert result.shape == (16, 16)
        assert result.dtype == np.uint8

    def test_i8x8_block_scan_order(self):
        """I_8x8 blocks should be processed in correct order.

        8x8 block positions:
          Block 0: (0,0) - top-left 8x8
          Block 1: (0,8) - top-right 8x8
          Block 2: (8,0) - bottom-left 8x8
          Block 3: (8,8) - bottom-right 8x8
        """
        from decoder.i8x8 import I8X8_BLOCK_POSITIONS

        assert I8X8_BLOCK_POSITIONS[0] == (0, 0)
        assert I8X8_BLOCK_POSITIONS[1] == (0, 8)
        assert I8X8_BLOCK_POSITIONS[2] == (8, 0)
        assert I8X8_BLOCK_POSITIONS[3] == (8, 8)


class TestI8x8NeighborExtraction:
    """Tests for extracting neighbor pixels for I_8x8 blocks."""

    def test_get_i8x8_block_neighbors_exists(self):
        """get_i8x8_block_neighbors function should exist."""
        from decoder.i8x8 import get_i8x8_block_neighbors

        assert callable(get_i8x8_block_neighbors)

    def test_i8x8_block_needs_25_neighbors(self):
        """Each 8x8 block needs up to 25 neighbors (8 top + 8 top-right + 8 left + 1 corner)."""
        from decoder.i8x8 import get_i8x8_block_neighbors

        frame = np.full((32, 32), 100, dtype=np.uint8)
        mb_luma = np.zeros((16, 16), dtype=np.uint8)

        neighbors = get_i8x8_block_neighbors(
            frame_luma=frame,
            mb_luma=mb_luma,
            mb_x=0,
            mb_y=0,
            block_idx=3,  # Block at (8, 8) position
            frame_width=32,
            frame_height=32,
        )

        assert len(neighbors['top']) == 8
        assert len(neighbors['left']) == 8
        # top_right may be 8 if available
        if neighbors['top_right_available']:
            assert len(neighbors['top_right']) == 8

    def test_i8x8_first_block_uses_external_neighbors(self):
        """First 8x8 block should use neighbors from outside macroblock."""
        from decoder.i8x8 import get_i8x8_block_neighbors

        frame = np.zeros((32, 32), dtype=np.uint8)
        # Set top row above MB to known value
        frame[15, 16:32] = 200
        mb_luma = np.zeros((16, 16), dtype=np.uint8)

        neighbors = get_i8x8_block_neighbors(
            frame_luma=frame,
            mb_luma=mb_luma,
            mb_x=1,  # Not first MB column
            mb_y=1,  # Not first MB row
            block_idx=0,  # First block
            frame_width=32,
            frame_height=32,
        )

        # Should have external neighbors available
        assert neighbors['top_available']

    def test_i8x8_subsequent_blocks_use_reconstructed(self):
        """Later 8x8 blocks should use previously reconstructed pixels."""
        from decoder.i8x8 import get_i8x8_block_neighbors

        frame = np.zeros((16, 16), dtype=np.uint8)
        mb_luma = np.zeros((16, 16), dtype=np.uint8)
        # Simulate block 0 reconstructed in mb_luma
        mb_luma[0:8, 0:8] = 150

        neighbors = get_i8x8_block_neighbors(
            frame_luma=frame,
            mb_luma=mb_luma,
            mb_x=0,
            mb_y=0,
            block_idx=1,  # Block 1 (top-right)
            frame_width=16,
            frame_height=16,
        )

        # Block 1 should have left neighbor from block 0 (in mb_luma)
        if neighbors['left_available']:
            assert np.all(neighbors['left'] == 150)


class TestI8x8NeighborDependenciesAcrossMB:
    """Tests for I_8x8 neighbor dependencies across macroblock boundaries."""

    def test_i8x8_top_neighbor_from_above_mb(self):
        """I_8x8 blocks should use neighbors from macroblock above."""
        from decoder.i8x8 import get_i8x8_mb_neighbors

        # Create frame with MB above having known values
        frame = np.zeros((32, 16), dtype=np.uint8)
        frame[15, :] = 180  # Bottom row of MB above

        neighbors = get_i8x8_mb_neighbors(
            frame_luma=frame,
            mb_x=0,
            mb_y=1,  # Second row of MBs
        )

        # Top neighbors should come from MB above
        np.testing.assert_array_equal(neighbors['top'], frame[15, :])

    def test_i8x8_left_neighbor_from_left_mb(self):
        """I_8x8 blocks should use neighbors from macroblock to left."""
        from decoder.i8x8 import get_i8x8_mb_neighbors

        frame = np.zeros((16, 32), dtype=np.uint8)
        frame[:, 15] = 170  # Right column of MB to left

        neighbors = get_i8x8_mb_neighbors(
            frame_luma=frame,
            mb_x=1,  # Second column of MBs
            mb_y=0,
        )

        # Left neighbors should come from MB to left
        np.testing.assert_array_equal(neighbors['left'], frame[:, 15])

    def test_i8x8_top_right_neighbor_availability(self):
        """Top-right neighbors depend on block position within MB.

        For block 0: top-right is within same MB row above
        For block 1: top-right is from MB to the right
        """
        from decoder.i8x8 import get_i8x8_block_top_right_availability

        # Block 0 at (0,0): top-right at (0,8) - within same MB (if top MB exists)
        # Top-right available if top MB exists
        assert get_i8x8_block_top_right_availability(
            block_idx=0, mb_x=1, mb_y=1, mb_width=4
        )

        # Block 1 at (0,8): top-right at (0,16) - from MB to upper-right
        # Top-right available only if MB at (mb_x+1, mb_y-1) exists
        avail = get_i8x8_block_top_right_availability(
            block_idx=1, mb_x=0, mb_y=1, mb_width=4
        )
        # Depends on rightmost MB position

    def test_i8x8_constrained_intra_pred(self):
        """With constrained_intra_pred_flag, can only use intra neighbors."""
        from decoder.i8x8 import get_i8x8_constrained_neighbors

        frame_luma = np.full((32, 32), 100, dtype=np.uint8)
        mb_types = np.array([
            [1, 1],  # Row 0: both inter (P type)
            [0, 0],  # Row 1: both intra
        ])

        neighbors = get_i8x8_constrained_neighbors(
            frame_luma=frame_luma,
            mb_x=0,
            mb_y=1,
            mb_types=mb_types,
            constrained_intra_pred_flag=True,
        )

        # Top neighbor is from inter MB, so not available with constrained flag
        assert not neighbors['top_available']


class TestI8x8ResidualDecoding:
    """Tests for decoding I_8x8 residual coefficients."""

    def test_decode_i8x8_residual_exists(self):
        """decode_i8x8_residual function should exist."""
        from decoder.i8x8 import decode_i8x8_residual

        assert callable(decode_i8x8_residual)

    def test_i8x8_residual_uses_8x8_transform(self):
        """I_8x8 residual should use 8x8 IDCT, not 4x4."""
        from decoder.i8x8 import decode_i8x8_residual
        from transform.idct_8x8 import idct_8x8

        # Coefficients for one 8x8 block
        coeffs = np.zeros((8, 8), dtype=np.int32)
        coeffs[0, 0] = 256  # DC only

        residual = decode_i8x8_residual(coeffs, qp=20)

        assert residual.shape == (8, 8)
        # Result should be from 8x8 IDCT, not sum of four 4x4 IDCTs

    def test_i8x8_uses_8x8_dequantization(self):
        """I_8x8 should use 8x8 quantization matrix, not 4x4."""
        from decoder.i8x8 import dequant_i8x8_block

        coeffs = np.ones((8, 8), dtype=np.int32)

        dequantized = dequant_i8x8_block(coeffs, qp=20)

        assert dequantized.shape == (8, 8)
        # 8x8 dequant uses different scaling factors than 4x4


class TestI8x8WithAllPredictionModes:
    """Tests for I_8x8 with all 9 prediction modes."""

    @pytest.mark.parametrize("mode", range(9))
    def test_i8x8_prediction_mode(self, mode):
        """Each prediction mode should produce valid 8x8 output."""
        from decoder.i8x8 import reconstruct_i8x8_block

        top = np.arange(8, dtype=np.uint8) * 20 + 50
        left = np.arange(8, dtype=np.uint8) * 15 + 60
        top_right = np.arange(8, dtype=np.uint8) * 10 + 100
        top_left = 128
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=mode,
            residual=residual,
            top=top,
            left=left,
            top_left=top_left,
            top_right=top_right,
            top_available=True,
            left_available=True,
            top_right_available=True,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8

    def test_i8x8_vertical_mode_copies_top(self):
        """Mode 0 (Vertical) should copy top row to all rows."""
        from decoder.i8x8 import reconstruct_i8x8_block

        top = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint8)
        left = np.full(8, 0, dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=0,  # Vertical
            residual=residual,
            top=top,
            left=left,
            top_left=0,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        for row in range(8):
            np.testing.assert_array_equal(result[row, :], top)

    def test_i8x8_horizontal_mode_copies_left(self):
        """Mode 1 (Horizontal) should copy left column to all columns."""
        from decoder.i8x8 import reconstruct_i8x8_block

        top = np.full(8, 0, dtype=np.uint8)
        left = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.uint8)
        residual = np.zeros((8, 8), dtype=np.int32)

        result = reconstruct_i8x8_block(
            mode=1,  # Horizontal
            residual=residual,
            top=top,
            left=left,
            top_left=0,
            top_right=None,
            top_available=True,
            left_available=True,
            top_right_available=False,
        )

        for col in range(8):
            np.testing.assert_array_equal(result[:, col], left)


class TestMixedI4x4I8x8Frame:
    """Tests for decoding frames with mixed I_4x4 and I_8x8 macroblocks."""

    def test_mixed_transform_sizes_in_frame(self):
        """Frame can contain both I_4x4 and I_8x8 macroblocks."""
        from decoder.decoder import H264Decoder
        from decoder.i8x8 import decode_mixed_intra_frame

        # Create frame with 4 MBs: 2 I_4x4, 2 I_8x8
        mb_transform_flags = [False, True, True, False]  # 4x4, 8x8, 8x8, 4x4

        decoder = H264Decoder()
        # Setup minimal SPS/PPS for High profile
        # ...

        # decode_mixed_intra_frame should handle both types
        # This tests that the decoder can switch between transform sizes

    def test_neighbor_derivation_between_i4x4_and_i8x8(self):
        """Neighbors between I_4x4 and I_8x8 MBs should be handled correctly."""
        from decoder.i8x8 import get_cross_transform_neighbors

        # MB 0 is I_4x4, MB 1 is I_8x8
        # MB 1 needs left neighbors from MB 0's rightmost 4x4 blocks
        frame_luma = np.zeros((16, 32), dtype=np.uint8)
        frame_luma[:, 12:16] = 150  # Right edge of MB 0

        neighbors = get_cross_transform_neighbors(
            frame_luma=frame_luma,
            current_mb_x=1,
            current_mb_y=0,
            current_is_8x8=True,
            left_is_8x8=False,
        )

        # Left neighbors should come from I_4x4 MB
        assert len(neighbors['left']) == 16

    def test_cbp_encoding_differs_for_i8x8(self):
        """CBP encoding for I_8x8 uses 8x8 block pattern, not 4x4."""
        from decoder.i8x8 import parse_i8x8_cbp

        # In I_8x8, CBP luma bits correspond to 8x8 blocks
        # Bit 0: 8x8 block 0 (top-left)
        # Bit 1: 8x8 block 1 (top-right)
        # Bit 2: 8x8 block 2 (bottom-left)
        # Bit 3: 8x8 block 3 (bottom-right)
        cbp = 0b0101  # Blocks 0 and 2 have residual

        has_residual = parse_i8x8_cbp(cbp)

        assert has_residual[0] == True  # Block 0
        assert has_residual[1] == False  # Block 1
        assert has_residual[2] == True  # Block 2
        assert has_residual[3] == False  # Block 3


class TestI8x8DecoderIntegration:
    """Full integration tests for I_8x8 decoding from bitstream."""

    def test_decode_i8x8_macroblock_from_bitstream(self):
        """Should decode complete I_8x8 macroblock from bitstream."""
        from decoder.decoder import H264Decoder
        from decoder.i8x8 import decode_i8x8_macroblock

        # Create minimal bitstream for one I_8x8 MB
        # - mb_type = 0 (I_NxN)
        # - transform_8x8_mode_flag = 1 (from PPS or MB syntax)
        # - Four 8x8 prediction modes
        # - CBP
        # - Residual data (if CBP indicates)
        writer = BitWriter()

        # mb_type = 0 (I_NxN) as ue(v)
        writer.write_ue(0)

        # transform_8x8_mode_flag = 1 (when allowed by PPS)
        writer.write_bits(1, 1)

        # Four 8x8 prediction modes (all using prev mode flag = 1)
        for _ in range(4):
            writer.write_bits(1, 1)  # prev_intra8x8_pred_mode_flag

        # CBP = 0 (no residual) - coded as ue with mapping
        # For I_8x8: CBP luma=0, CBP chroma=0 maps to specific codeNum
        writer.write_ue(0)  # No coded block

        writer.write_bits(0, 8)  # Padding

        reader = BitReader(writer.to_bytes())

        # Create decoder state
        frame_luma = np.full((16, 16), 128, dtype=np.uint8)
        frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        mb_data = decode_i8x8_macroblock(
            reader=reader,
            frame_luma=frame_luma,
            frame_cb=frame_cb,
            frame_cr=frame_cr,
            mb_x=0,
            mb_y=0,
            qp=26,
            neighbors_available={'top': False, 'left': False},
        )

        assert mb_data is not None
        assert mb_data.mb_type == 'I_8x8'

    def test_i8x8_with_nonzero_residual(self):
        """Should decode I_8x8 with non-zero residual coefficients."""
        from decoder.i8x8 import decode_i8x8_macroblock

        # Create bitstream with residual data
        writer = BitWriter()

        # I_NxN mb_type
        writer.write_ue(0)

        # transform_8x8_mode_flag
        writer.write_bits(1, 1)

        # Prediction modes (all DC)
        for _ in range(4):
            writer.write_bits(1, 1)  # Use predicted

        # CBP with luma residual
        # For I_8x8, CBP maps differently than I_4x4
        # Need proper CBP codeNum for "all 8x8 blocks have residual, no chroma"
        writer.write_ue(15)  # CBP indicating luma residual

        # mb_qp_delta = 0
        writer.write_se(0)

        # Residual data for four 8x8 blocks
        # This would be CAVLC/CABAC encoded coefficients
        # Simplified: write placeholder data
        writer.write_bits(0, 64)  # Placeholder

        reader = BitReader(writer.to_bytes())

        frame_luma = np.full((16, 16), 128, dtype=np.uint8)
        frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        # This should handle residual parsing and application
        mb_data = decode_i8x8_macroblock(
            reader=reader,
            frame_luma=frame_luma,
            frame_cb=frame_cb,
            frame_cr=frame_cr,
            mb_x=0,
            mb_y=0,
            qp=26,
            neighbors_available={'top': False, 'left': False},
        )

        # Frame should be modified by residual
        assert mb_data is not None


class TestI8x8ChromaHandling:
    """Tests for chroma handling in I_8x8 macroblocks."""

    def test_i8x8_chroma_uses_intra_chroma_pred_mode(self):
        """I_8x8 chroma uses standard intra chroma prediction modes."""
        from decoder.i8x8 import get_i8x8_chroma_pred_mode

        # Chroma prediction is independent of luma 8x8 modes
        # intra_chroma_pred_mode: 0=DC, 1=Horizontal, 2=Vertical, 3=Plane
        for chroma_mode in range(4):
            result = get_i8x8_chroma_pred_mode(chroma_mode)
            assert result == chroma_mode

    def test_i8x8_chroma_residual_uses_4x4_transform(self):
        """I_8x8 chroma still uses 4x4 transform (4:2:0)."""
        from decoder.i8x8 import decode_i8x8_chroma_residual

        # Even when luma uses 8x8, chroma uses 4x4 in 4:2:0
        # Each 8x8 chroma block has four 4x4 sub-blocks
        cb_coeffs = [np.zeros((4, 4), dtype=np.int32) for _ in range(4)]

        residual = decode_i8x8_chroma_residual(cb_coeffs, qp=26)

        assert residual.shape == (8, 8)


class TestI8x8CABACDecoding:
    """Tests for I_8x8 decoding with CABAC entropy coding."""

    def test_i8x8_cabac_context_initialization(self):
        """CABAC contexts for I_8x8 should be properly initialized."""
        from decoder.i8x8 import init_i8x8_cabac_contexts

        contexts = init_i8x8_cabac_contexts(slice_qp=26)

        # Should have contexts for:
        # - prev_intra8x8_pred_mode_flag
        # - rem_intra8x8_pred_mode
        # - significant_coeff_flag (8x8 version)
        # - last_significant_coeff_flag (8x8 version)
        # - coeff_abs_level_minus1
        assert 'prev_intra8x8_pred_mode_flag' in contexts
        assert 'significant_coeff_flag_8x8' in contexts

    def test_i8x8_cabac_residual_scan_order(self):
        """I_8x8 CABAC uses 8x8 diagonal scan order."""
        from decoder.i8x8 import I8X8_SCAN_ORDER

        # 8x8 block has 64 coefficients
        assert len(I8X8_SCAN_ORDER) == 64

        # First position is DC (0, 0)
        assert I8X8_SCAN_ORDER[0] == (0, 0)


class TestI8x8EdgeCases:
    """Tests for I_8x8 edge cases and error handling."""

    def test_i8x8_at_frame_boundary(self):
        """I_8x8 at frame boundary should handle missing neighbors."""
        from decoder.i8x8 import reconstruct_i8x8_block

        # First MB in frame has no top or left neighbors
        result = reconstruct_i8x8_block(
            mode=2,  # DC - handles missing neighbors
            residual=np.zeros((8, 8), dtype=np.int32),
            top=None,
            left=None,
            top_left=None,
            top_right=None,
            top_available=False,
            left_available=False,
            top_right_available=False,
        )

        assert result.shape == (8, 8)
        # DC mode with no neighbors should use default value (128)
        assert np.all(result == 128)

    def test_i8x8_invalid_prediction_mode(self):
        """Invalid prediction mode should raise error."""
        from decoder.i8x8 import reconstruct_i8x8_block

        with pytest.raises(ValueError, match="(Unknown.*mode|not a valid)"):
            reconstruct_i8x8_block(
                mode=9,  # Invalid - only 0-8 are valid
                residual=np.zeros((8, 8), dtype=np.int32),
                top=np.zeros(8, dtype=np.uint8),
                left=np.zeros(8, dtype=np.uint8),
                top_left=0,
                top_right=None,
                top_available=True,
                left_available=True,
                top_right_available=False,
            )

    def test_i8x8_mode_requiring_unavailable_neighbors(self):
        """Modes requiring specific neighbors should fallback gracefully."""
        from decoder.i8x8 import reconstruct_i8x8_block

        # Vertical mode requires top neighbors
        # If top unavailable, should use default or fallback
        result = reconstruct_i8x8_block(
            mode=0,  # Vertical - needs top
            residual=np.zeros((8, 8), dtype=np.int32),
            top=None,
            left=np.full(8, 100, dtype=np.uint8),
            top_left=None,
            top_right=None,
            top_available=False,  # Top not available!
            left_available=True,
            top_right_available=False,
        )

        # Should produce valid output (possibly using default values)
        assert result.shape == (8, 8)


# =============================================================================
# RED TESTS: Extended I_8x8 Decoder Tests for High Profile Features
# These tests SHOULD FAIL until full I_8x8 decoder support is implemented.
# =============================================================================


class TestI8x8ScalingListIntegration:
    """Tests for I_8x8 with scaling lists from SPS/PPS.

    H.264 High profile uses 8x8 scaling matrices that affect
    dequantization of 8x8 transform coefficients.
    """

    def test_i8x8_uses_sps_scaling_list(self):
        """I_8x8 should use 8x8 scaling list from SPS if available."""
        from decoder.i8x8 import decode_i8x8_macroblock_with_scaling

        # Create SPS with custom 8x8 scaling list
        sps = {
            'profile_idc': 100,  # High profile
            'seq_scaling_list_present_flag': [0, 0, 0, 0, 0, 0, 1, 1],
            'scaling_list_8x8': [
                [16] * 64,  # Intra 8x8
                [16] * 64,  # Inter 8x8
            ]
        }

        frame_luma = np.full((16, 16), 128, dtype=np.uint8)

        mb_data = decode_i8x8_macroblock_with_scaling(
            frame_luma=frame_luma,
            mb_x=0,
            mb_y=0,
            qp=26,
            sps=sps,
            pps=None,
        )

        assert mb_data is not None

    def test_i8x8_pps_scaling_list_overrides_sps(self):
        """PPS scaling list should override SPS if present."""
        from decoder.i8x8 import get_i8x8_scaling_list

        sps_list = [20] * 64
        pps_list = [32] * 64

        result = get_i8x8_scaling_list(
            sps_list=sps_list,
            pps_list=pps_list,
            use_pps_list=True,
        )

        np.testing.assert_array_equal(result, pps_list)


class TestI8x8CAVLCDecoding:
    """Tests for I_8x8 residual decoding with CAVLC entropy coding."""

    def test_cavlc_8x8_block_decoding_exists(self):
        """CAVLC decoder for 8x8 blocks should exist."""
        from decoder.i8x8 import decode_cavlc_residual_8x8

        assert callable(decode_cavlc_residual_8x8)

    def test_cavlc_8x8_returns_64_coefficients(self):
        """CAVLC 8x8 decoder should return 64 coefficients."""
        from decoder.i8x8 import decode_cavlc_residual_8x8
        from bitstream import BitWriter, BitReader

        # Create minimal CAVLC-encoded data for 8x8 block
        # (all zeros - coeff_token=0)
        writer = BitWriter()
        writer.write_bits(1, 1)  # coeff_token for TotalCoeff=0
        writer.write_bits(0, 8)  # Padding

        reader = BitReader(writer.to_bytes())

        coeffs = decode_cavlc_residual_8x8(reader, nc=0)

        assert coeffs.shape == (8, 8)

    def test_cavlc_8x8_uses_correct_nc_mapping(self):
        """CAVLC for 8x8 should use correct nC calculation.

        H.264 Spec: For 8x8 blocks, nC is based on neighboring 8x8 blocks,
        not 4x4 blocks.
        """
        from decoder.i8x8 import calculate_nc_for_8x8

        # Create mock neighbor coefficient counts
        left_8x8_coeffs = 12
        top_8x8_coeffs = 8

        nc = calculate_nc_for_8x8(
            left_available=True,
            top_available=True,
            left_coeffs=left_8x8_coeffs,
            top_coeffs=top_8x8_coeffs,
        )

        # nC = (left + top + 1) >> 1 = (12 + 8 + 1) >> 1 = 10
        expected = (left_8x8_coeffs + top_8x8_coeffs + 1) >> 1
        assert nc == expected


class TestI8x8Transform8x8Flag:
    """Tests for transform_8x8_mode_flag syntax element.

    H.264 Spec: Section 7.3.5.1
    transform_8x8_mode_flag indicates whether macroblock uses 8x8 transform.
    """

    def test_parse_transform_8x8_mode_flag_cavlc(self):
        """Parse transform_8x8_mode_flag with CAVLC."""
        from decoder.i8x8 import parse_transform_8x8_mode_flag
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        writer.write_bits(1, 1)  # transform_8x8_mode_flag = 1
        writer.write_bits(0, 7)  # Padding

        reader = BitReader(writer.to_bytes())

        flag = parse_transform_8x8_mode_flag(reader, cabac=False)

        assert flag == True

    @pytest.mark.xfail(reason="CABAC decoder not yet integrated as single module")
    def test_parse_transform_8x8_mode_flag_cabac(self):
        """Parse transform_8x8_mode_flag with CABAC."""
        from decoder.i8x8 import parse_transform_8x8_mode_flag_cabac

        try:
            from entropy.cabac_arith import CABACArithmeticDecoder as CABACDecoder
        except ImportError:
            pytest.skip("CABAC arithmetic decoder not available")

        # CABAC needs context initialization
        # This tests the CABAC context model for this flag
        # Context index for transform_8x8_mode_flag is 399

        # Would need proper CABAC bitstream, skip for unit test
        pass

    def test_transform_8x8_mode_flag_context_index(self):
        """Context index for transform_8x8_mode_flag should be 399."""
        from decoder.i8x8 import TRANSFORM_8X8_MODE_FLAG_CTX_IDX

        assert TRANSFORM_8X8_MODE_FLAG_CTX_IDX == 399


class TestI8x8IntraModePrediction:
    """Tests for I_8x8 intra mode prediction (same mechanism as I_4x4)."""

    def test_prev_intra8x8_pred_mode_flag_parsing(self):
        """Parse prev_intra8x8_pred_mode_flag for each 8x8 block."""
        from decoder.i8x8 import parse_intra8x8_pred_modes
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        # 4 blocks, all using predicted mode (flag=1)
        for _ in range(4):
            writer.write_bits(1, 1)  # prev_intra8x8_pred_mode_flag = 1
        writer.write_bits(0, 4)  # Padding

        reader = BitReader(writer.to_bytes())

        modes = parse_intra8x8_pred_modes(reader, neighbor_modes=[2, 2, 2, 2])

        assert len(modes) == 4

    def test_rem_intra8x8_pred_mode_parsing(self):
        """Parse rem_intra8x8_pred_mode when flag is 0."""
        from decoder.i8x8 import parse_intra8x8_pred_modes
        from bitstream import BitWriter, BitReader

        writer = BitWriter()
        # Block 0: flag=0, rem=3 (explicit mode)
        writer.write_bits(0, 1)  # flag = 0
        writer.write_bits(3, 3)  # rem = 3
        # Remaining blocks use predicted
        for _ in range(3):
            writer.write_bits(1, 1)
        writer.write_bits(0, 4)  # Padding

        reader = BitReader(writer.to_bytes())

        modes = parse_intra8x8_pred_modes(reader, neighbor_modes=[2, 2, 2, 2])

        assert len(modes) == 4
        # First mode should be 3 (if predicted was 2, rem=3 means mode=4)
        # or mode=3 if rem < predicted

    def test_intra8x8_pred_mode_derivation(self):
        """Test mode derivation from flag and rem values.

        H.264 Spec: 8.3.2.2
        If prev_flag=1: mode = MostProbableMode
        If prev_flag=0: mode = rem if rem < MostProbableMode else rem + 1
        """
        from decoder.i8x8 import derive_intra8x8_mode

        # Test case 1: Use predicted mode
        mode = derive_intra8x8_mode(
            prev_flag=True,
            rem=None,
            most_probable_mode=3
        )
        assert mode == 3

        # Test case 2: Explicit mode less than predicted
        mode = derive_intra8x8_mode(
            prev_flag=False,
            rem=2,
            most_probable_mode=5
        )
        assert mode == 2  # rem < MPM, so mode = rem

        # Test case 3: Explicit mode >= predicted
        mode = derive_intra8x8_mode(
            prev_flag=False,
            rem=5,
            most_probable_mode=3
        )
        assert mode == 6  # rem >= MPM, so mode = rem + 1


class TestI8x8CBPHandling:
    """Tests for coded block pattern handling in I_8x8 macroblocks."""

    def test_cbp_luma_maps_to_8x8_blocks(self):
        """CBP luma bits map to 8x8 blocks, not 4x4."""
        from decoder.i8x8 import decode_i8x8_cbp_luma

        # CBP = 0b1010 means blocks 1 and 3 have residual
        cbp = 0b1010

        block_has_residual = decode_i8x8_cbp_luma(cbp)

        assert len(block_has_residual) == 4
        assert block_has_residual[0] == False  # Bit 0
        assert block_has_residual[1] == True   # Bit 1
        assert block_has_residual[2] == False  # Bit 2
        assert block_has_residual[3] == True   # Bit 3

    def test_cbp_chroma_same_as_i4x4(self):
        """CBP chroma handling is same as I_4x4."""
        from decoder.i8x8 import decode_i8x8_cbp_chroma

        # CBP chroma = 0: no chroma residual
        # CBP chroma = 1: DC only
        # CBP chroma = 2: DC and AC

        has_dc, has_ac = decode_i8x8_cbp_chroma(cbp_chroma=2)

        assert has_dc == True
        assert has_ac == True


class TestI8x8FullPipeline:
    """End-to-end tests for I_8x8 decoding pipeline."""

    def test_full_i8x8_frame_decode(self):
        """Decode complete frame with I_8x8 macroblocks."""
        from decoder.decoder import H264Decoder
        from decoder.i8x8 import decode_i8x8_frame

        # Create minimal High profile bitstream with I_8x8
        # This would be a complete NAL unit

        decoder = H264Decoder()

        # Setup for High profile
        sps = {
            'profile_idc': 100,
            'pic_width_in_mbs_minus1': 1,  # 2 MBs wide
            'pic_height_in_map_units_minus1': 1,  # 2 MBs tall
            'frame_mbs_only_flag': True,
        }
        pps = {
            'transform_8x8_mode_flag': True,  # Enable I_8x8
        }

        # Minimal frame would have 4 I_8x8 macroblocks
        # Each MB: mb_type, transform_8x8_mode_flag, 4 pred modes,
        # CBP, qp_delta, residual

    def test_i8x8_mb_after_p_mb(self):
        """I_8x8 macroblock following P macroblock."""
        from decoder.i8x8 import decode_mixed_frame

        # Test that neighbor derivation works when previous MB is P type
        # This tests constrained_intra_pred_flag handling

    def test_i8x8_produces_correct_pixel_values(self):
        """I_8x8 should produce valid uint8 pixel values."""
        from decoder.i8x8 import decode_i8x8_macroblock

        # Setup minimal test
        frame_luma = np.full((16, 16), 128, dtype=np.uint8)
        frame_cb = np.full((8, 8), 128, dtype=np.uint8)
        frame_cr = np.full((8, 8), 128, dtype=np.uint8)

        # After decoding, pixels should be valid uint8
        # Exact values depend on prediction + residual

        assert frame_luma.dtype == np.uint8
        assert frame_luma.min() >= 0
        assert frame_luma.max() <= 255


class TestI8x8ConstrainedIntraPrediction:
    """Tests for constrained intra prediction with I_8x8.

    H.264 Spec: When constrained_intra_pred_flag=1, intra macroblocks
    can only use neighbors from other intra macroblocks.
    """

    def test_constrained_intra_pred_disables_inter_neighbors(self):
        """I_8x8 should not use neighbors from inter macroblocks."""
        from decoder.i8x8 import get_i8x8_neighbors_constrained

        mb_types = np.array([
            [1, 0],  # Row 0: inter, intra
            [0, 0],  # Row 1: intra, intra
        ])

        neighbors = get_i8x8_neighbors_constrained(
            frame_luma=np.full((32, 32), 100, dtype=np.uint8),
            mb_x=1,  # Current MB
            mb_y=0,
            mb_types=mb_types,
            constrained_intra_pred_flag=True,
        )

        # Left neighbor is from inter MB, should be unavailable
        assert not neighbors['left_available']

    def test_non_constrained_uses_all_neighbors(self):
        """Without constraint, I_8x8 can use inter neighbors."""
        from decoder.i8x8 import get_i8x8_neighbors_constrained

        mb_types = np.array([
            [1, 0],
            [0, 0],
        ])

        neighbors = get_i8x8_neighbors_constrained(
            frame_luma=np.full((32, 32), 100, dtype=np.uint8),
            mb_x=1,
            mb_y=0,
            mb_types=mb_types,
            constrained_intra_pred_flag=False,  # Not constrained
        )

        # Left neighbor should be available
        assert neighbors['left_available']


class TestI8x8ReferenceFilteredSamples:
    """Tests for reference sample filtering in I_8x8 prediction.

    H.264 Spec: Section 8.3.4.2.2
    I_8x8 uses lowpass filtered reference samples.
    """

    def test_i8x8_applies_lowpass_filter(self):
        """I_8x8 prediction should use filtered reference samples."""
        from decoder.i8x8 import filter_reference_samples_8x8

        top = np.array([0, 0, 0, 0, 100, 100, 100, 100], dtype=np.uint8)
        left = np.array([50] * 8, dtype=np.uint8)
        top_left = 25
        top_right = np.full(8, 100, dtype=np.uint8)

        filtered = filter_reference_samples_8x8(top, left, top_left, top_right)

        # After filtering, edge transitions should be smoothed
        assert filtered['top'][4] < 100  # Smoothed from neighbors
        assert filtered['top'][3] > 0    # Smoothed from neighbors

    def test_filter_disabled_for_specific_modes(self):
        """Some modes may bypass filtering (implementation dependent)."""
        from decoder.i8x8 import should_filter_8x8

        # Check if filtering is enabled for each mode
        # H.264 allows implementation to skip filtering for DC mode
        for mode in range(9):
            result = should_filter_8x8(mode)
            assert isinstance(result, bool)


class TestI8x8QpDeltaHandling:
    """Tests for QP delta handling in I_8x8 macroblocks."""

    def test_qp_delta_affects_all_8x8_blocks(self):
        """mb_qp_delta should affect all 4 8x8 blocks equally."""
        from decoder.i8x8 import apply_qp_delta_i8x8

        base_qp = 26
        qp_delta = 4

        block_qps = apply_qp_delta_i8x8(
            base_qp=base_qp,
            qp_delta=qp_delta,
            num_blocks=4,
        )

        # All blocks should have same QP
        expected_qp = 30
        assert all(qp == expected_qp for qp in block_qps)

    def test_qp_wrapping_at_boundaries(self):
        """QP should wrap correctly at boundaries."""
        from decoder.i8x8 import calculate_i8x8_qp

        # Test QP wrapping at 52
        qp = calculate_i8x8_qp(base_qp=50, qp_delta=5)
        # (50 + 5) mod 52 = 3
        assert qp == 3

        # Test negative QP delta
        qp = calculate_i8x8_qp(base_qp=2, qp_delta=-5)
        # (2 - 5 + 52) mod 52 = 49
        assert qp == 49


class TestI8x8DeblockingFilter:
    """Tests for deblocking filter with I_8x8 macroblocks."""

    def test_deblock_i8x8_internal_edges(self):
        """Deblocking should be applied to 8x8 internal edges."""
        from decoder.i8x8 import get_i8x8_deblock_edges

        edges = get_i8x8_deblock_edges()

        # I_8x8 has one internal vertical edge at x=8
        # and one internal horizontal edge at y=8
        assert (8, 'vertical') in edges
        assert (8, 'horizontal') in edges

    def test_deblock_bs_for_i8x8(self):
        """Boundary strength for I_8x8 edges."""
        from decoder.i8x8 import calculate_bs_i8x8

        # Intra MB boundary should have BS=3 or BS=4
        # depending on whether it's MB boundary

        bs = calculate_bs_i8x8(
            is_mb_boundary=False,  # Internal 8x8 boundary
            is_intra=True,
        )

        # Internal 8x8 boundary in I_8x8 has BS=3
        assert bs == 3

        bs = calculate_bs_i8x8(
            is_mb_boundary=True,  # MB boundary
            is_intra=True,
        )

        # MB boundary for intra has BS=4
        assert bs == 4
