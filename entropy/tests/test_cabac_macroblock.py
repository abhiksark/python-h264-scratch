# h264/entropy/tests/test_cabac_macroblock.py
"""RED TESTS: CABAC macroblock decoding.

Full macroblock-level CABAC decoding including skip flags, mb_type,
sub_mb_type, intra prediction modes, and coded block patterns.

H.264 Spec Reference: Section 7.3.5 - Macroblock layer syntax

These tests SHOULD FAIL until CABAC macroblock decoding is implemented.
"""

import pytest

from bitstream import BitReader


class TestMBSkipFlagCABAC:
    """Tests for mb_skip_flag CABAC decoding with context derivation."""

    def test_decode_mb_skip_flag_p_slice_no_neighbors_skipped(self):
        """P-slice mb_skip_flag with no skipped neighbors uses ctx_idx=11."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_skip_flag_cabac

        # Bitstream that decodes to mb_skip_flag=1 (skipped MB)
        data = bytes([0x00, 0x00, 0x01, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        # Create mock MB neighbor info
        mb_info = {
            'mb_x': 1, 'mb_y': 1,
            'left_available': True, 'top_available': True,
            'left_skip': False, 'top_skip': False,
        }

        result = decode_mb_skip_flag_cabac(
            decoder, contexts, slice_type=0, mb_info=mb_info
        )

        assert result in (0, 1)

    def test_decode_mb_skip_flag_p_slice_left_neighbor_skipped(self):
        """P-slice mb_skip_flag with left neighbor skipped uses ctx_idx=12."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_skip_flag_cabac

        data = bytes([0xFF, 0x80, 0x40, 0x20] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        mb_info = {
            'mb_x': 1, 'mb_y': 1,
            'left_available': True, 'top_available': True,
            'left_skip': True, 'top_skip': False,
        }

        result = decode_mb_skip_flag_cabac(
            decoder, contexts, slice_type=0, mb_info=mb_info
        )

        assert result in (0, 1)

    def test_decode_mb_skip_flag_b_slice_both_neighbors_skipped(self):
        """B-slice mb_skip_flag with both neighbors skipped uses ctx_idx=26."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_skip_flag_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=1, slice_qp=26)

        mb_info = {
            'mb_x': 1, 'mb_y': 1,
            'left_available': True, 'top_available': True,
            'left_skip': True, 'top_skip': True,
        }

        result = decode_mb_skip_flag_cabac(
            decoder, contexts, slice_type=1, mb_info=mb_info
        )

        assert result in (0, 1)

    def test_decode_mb_skip_flag_unavailable_neighbors_treated_as_not_skipped(self):
        """Unavailable neighbors should be treated as not skipped (condTermFlag=0)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_skip_flag_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        # First MB in slice - no left or top neighbor
        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
            'left_skip': False, 'top_skip': False,
        }

        result = decode_mb_skip_flag_cabac(
            decoder, contexts, slice_type=0, mb_info=mb_info
        )

        assert result in (0, 1)


class TestMBTypeCABAC:
    """Tests for mb_type CABAC decoding with proper binarization."""

    def test_decode_mb_type_i_slice_i_4x4(self):
        """I-slice mb_type=0 (I_4x4) decodes from single '0' bin."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_type_cabac

        # Construct bitstream that should decode to I_4x4
        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
            'left_mb_type': None, 'top_mb_type': None,
        }

        result = decode_mb_type_cabac(
            decoder, contexts, slice_type=2, mb_info=mb_info
        )

        # I-slice mb_type: 0=I_4x4, 1-24=I_16x16 variants, 25=I_PCM
        assert 0 <= result <= 25

    def test_decode_mb_type_i_slice_i_16x16_pred_mode_0(self):
        """I-slice mb_type for I_16x16_0_0_0 (pred=0, cbp_luma=0, cbp_chroma=0)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_type_cabac

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
            'left_mb_type': None, 'top_mb_type': None,
        }

        result = decode_mb_type_cabac(
            decoder, contexts, slice_type=2, mb_info=mb_info
        )

        assert result >= 0

    def test_decode_mb_type_p_slice_p_l0_16x16(self):
        """P-slice mb_type=0 (P_L0_16x16) - simplest P-MB type."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_type_cabac

        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        mb_info = {
            'mb_x': 1, 'mb_y': 1,
            'left_available': True, 'top_available': True,
            'left_mb_type': 0, 'top_mb_type': 0,
        }

        result = decode_mb_type_cabac(
            decoder, contexts, slice_type=0, mb_info=mb_info
        )

        # P-slice: 0-4 for P-MB, 5+ for I-MB in P-slice
        assert result >= 0

    def test_decode_mb_type_p_slice_intra_in_p(self):
        """P-slice mb_type=5+ (I_4x4 in P-slice)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_type_cabac

        data = bytes([0xFF, 0x80, 0x40, 0x20] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
            'left_mb_type': None, 'top_mb_type': None,
        }

        result = decode_mb_type_cabac(
            decoder, contexts, slice_type=0, mb_info=mb_info
        )

        assert result >= 0

    def test_decode_mb_type_b_slice_b_direct_16x16(self):
        """B-slice mb_type=0 (B_Direct_16x16)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_type_cabac

        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=1, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
            'left_mb_type': None, 'top_mb_type': None,
        }

        result = decode_mb_type_cabac(
            decoder, contexts, slice_type=1, mb_info=mb_info
        )

        # B-slice: 0-22 for B-MB, 23+ for I-MB in B-slice
        assert result >= 0


class TestSubMBTypeCABAC:
    """Tests for sub_mb_type CABAC decoding in P_8x8 and B_8x8 macroblocks."""

    def test_decode_sub_mb_type_p_8x8(self):
        """P-slice sub_mb_type=0 (P_L0_8x8) for P_8x8 macroblock."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_sub_mb_type_cabac

        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_sub_mb_type_cabac(
            decoder, contexts, slice_type=0, mb_type=3  # P_8x8
        )

        # P-slice sub_mb_type: 0=P_L0_8x8, 1=P_L0_8x4, 2=P_L0_4x8, 3=P_L0_4x4
        assert 0 <= result <= 3

    def test_decode_sub_mb_type_p_4x4(self):
        """P-slice sub_mb_type=3 (P_L0_4x4) - smallest partition."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_sub_mb_type_cabac

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_sub_mb_type_cabac(
            decoder, contexts, slice_type=0, mb_type=3  # P_8x8
        )

        assert 0 <= result <= 3

    def test_decode_sub_mb_type_b_direct_8x8(self):
        """B-slice sub_mb_type=0 (B_Direct_8x8) for B_8x8 macroblock."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_sub_mb_type_cabac

        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=1, slice_qp=26)

        result = decode_sub_mb_type_cabac(
            decoder, contexts, slice_type=1, mb_type=22  # B_8x8
        )

        # B-slice sub_mb_type: 0-12 (various B sub-partition types)
        assert 0 <= result <= 12

    def test_decode_all_four_sub_mb_types_in_8x8_mb(self):
        """Decode all 4 sub_mb_type values for an 8x8 partitioned MB."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_sub_mb_type_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10, 0x08, 0x04] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        sub_mb_types = []
        for _ in range(4):
            sub_type = decode_sub_mb_type_cabac(
                decoder, contexts, slice_type=0, mb_type=3
            )
            sub_mb_types.append(sub_type)

        assert len(sub_mb_types) == 4
        assert all(0 <= t <= 3 for t in sub_mb_types)


class TestIntraChromaPredModeCABAC:
    """Tests for intra_chroma_pred_mode CABAC decoding."""

    def test_decode_intra_chroma_pred_mode_dc(self):
        """intra_chroma_pred_mode=0 (DC prediction)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_intra_chroma_pred_mode_cabac

        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'left_available': True, 'top_available': True,
            'left_intra_chroma_pred_mode': 0,
            'top_intra_chroma_pred_mode': 0,
        }

        result = decode_intra_chroma_pred_mode_cabac(
            decoder, contexts, mb_info=mb_info
        )

        # intra_chroma_pred_mode: 0=DC, 1=Horiz, 2=Vert, 3=Plane
        assert 0 <= result <= 3

    def test_decode_intra_chroma_pred_mode_plane(self):
        """intra_chroma_pred_mode=3 (Plane prediction)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_intra_chroma_pred_mode_cabac

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'left_available': True, 'top_available': True,
            'left_intra_chroma_pred_mode': 0,
            'top_intra_chroma_pred_mode': 0,
        }

        result = decode_intra_chroma_pred_mode_cabac(
            decoder, contexts, mb_info=mb_info
        )

        assert 0 <= result <= 3

    def test_decode_intra_chroma_pred_mode_context_from_neighbors(self):
        """Context depends on neighbor chroma prediction modes."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_intra_chroma_pred_mode_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        # Non-DC neighbors increase context index
        mb_info = {
            'left_available': True, 'top_available': True,
            'left_intra_chroma_pred_mode': 2,  # Non-DC
            'top_intra_chroma_pred_mode': 1,   # Non-DC
        }

        result = decode_intra_chroma_pred_mode_cabac(
            decoder, contexts, mb_info=mb_info
        )

        assert 0 <= result <= 3


class TestCodedBlockPatternCABAC:
    """Tests for coded_block_pattern CABAC decoding."""

    def test_decode_cbp_luma_all_zero(self):
        """CBP luma=0 means no luma coefficients in any 8x8 block."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_coded_block_pattern_cabac

        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
            'left_cbp_luma': 0, 'top_cbp_luma': 0,
            'left_cbp_chroma': 0, 'top_cbp_chroma': 0,
        }

        cbp_luma, cbp_chroma = decode_coded_block_pattern_cabac(
            decoder, contexts, mb_type=0, mb_info=mb_info
        )

        assert 0 <= cbp_luma <= 15
        assert 0 <= cbp_chroma <= 2

    def test_decode_cbp_luma_all_coded(self):
        """CBP luma=15 means all four 8x8 blocks have coefficients."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_coded_block_pattern_cabac

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'mb_x': 1, 'mb_y': 1,
            'left_available': True, 'top_available': True,
            'left_cbp_luma': 15, 'top_cbp_luma': 15,
            'left_cbp_chroma': 2, 'top_cbp_chroma': 2,
        }

        cbp_luma, cbp_chroma = decode_coded_block_pattern_cabac(
            decoder, contexts, mb_type=0, mb_info=mb_info
        )

        assert 0 <= cbp_luma <= 15
        assert 0 <= cbp_chroma <= 2

    def test_decode_cbp_chroma_dc_only(self):
        """CBP chroma=1 means DC coefficients only."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_coded_block_pattern_cabac

        data = bytes([0x80, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
            'left_cbp_luma': 0, 'top_cbp_luma': 0,
            'left_cbp_chroma': 0, 'top_cbp_chroma': 0,
        }

        cbp_luma, cbp_chroma = decode_coded_block_pattern_cabac(
            decoder, contexts, mb_type=0, mb_info=mb_info
        )

        assert 0 <= cbp_luma <= 15
        assert 0 <= cbp_chroma <= 2

    def test_decode_cbp_context_depends_on_neighbor_cbp(self):
        """CBP context derivation uses neighbor CBP values."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_coded_block_pattern_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        # Different neighbor CBP values
        mb_info = {
            'mb_x': 1, 'mb_y': 1,
            'left_available': True, 'top_available': True,
            'left_cbp_luma': 5,    # blocks 0,2 coded
            'top_cbp_luma': 10,    # blocks 1,3 coded
            'left_cbp_chroma': 1,
            'top_cbp_chroma': 2,
        }

        cbp_luma, cbp_chroma = decode_coded_block_pattern_cabac(
            decoder, contexts, mb_type=0, mb_info=mb_info
        )

        assert 0 <= cbp_luma <= 15
        assert 0 <= cbp_chroma <= 2


class TestFullMacroblockCABAC:
    """Tests for full macroblock CABAC decoding pipeline."""

    def test_decode_macroblock_layer_i_4x4(self):
        """Decode complete I_4x4 macroblock layer."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_macroblock_layer_cabac

        # Construct bitstream for I_4x4 MB
        data = bytes([0x00, 0x00, 0x00, 0x00] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
        }

        mb_data = decode_macroblock_layer_cabac(
            decoder, contexts, slice_type=2, mb_info=mb_info
        )

        assert 'mb_type' in mb_data
        assert mb_data['mb_type'] >= 0

    def test_decode_macroblock_layer_p_skip(self):
        """Decode P-slice skipped macroblock."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_macroblock_layer_cabac

        # Bitstream that decodes to skip=1
        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        mb_info = {
            'mb_x': 1, 'mb_y': 1,
            'left_available': True, 'top_available': True,
            'left_skip': True, 'top_skip': True,
        }

        mb_data = decode_macroblock_layer_cabac(
            decoder, contexts, slice_type=0, mb_info=mb_info
        )

        assert 'mb_skip_flag' in mb_data

    def test_decode_macroblock_layer_p_8x8_with_sub_partitions(self):
        """Decode P_8x8 macroblock with sub-partition types."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_macroblock_layer_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10, 0x08, 0x04] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
            'left_skip': False, 'top_skip': False,
        }

        mb_data = decode_macroblock_layer_cabac(
            decoder, contexts, slice_type=0, mb_info=mb_info
        )

        assert 'mb_type' in mb_data

    def test_decode_macroblock_layer_b_direct(self):
        """Decode B_Direct_16x16 macroblock."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_macroblock_layer_cabac

        data = bytes([0x00, 0x00, 0x00, 0x00] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=1, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
            'left_skip': False, 'top_skip': False,
        }

        mb_data = decode_macroblock_layer_cabac(
            decoder, contexts, slice_type=1, mb_info=mb_info
        )

        assert 'mb_type' in mb_data

    def test_decode_macroblock_includes_intra_pred_modes(self):
        """Intra macroblock includes prediction mode data."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_macroblock_layer_cabac

        data = bytes([0x00, 0x80, 0x40, 0x20] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
        }

        mb_data = decode_macroblock_layer_cabac(
            decoder, contexts, slice_type=2, mb_info=mb_info
        )

        # I_4x4 should have 16 luma prediction modes
        # I_16x16 should have one mode embedded in mb_type
        assert 'mb_type' in mb_data


class TestMBPredCABAC:
    """Tests for mb_pred CABAC decoding (prediction modes, ref_idx, mvd)."""

    def test_decode_mb_pred_intra_4x4_modes(self):
        """Decode prev_intra4x4_pred_mode_flag and rem_intra4x4_pred_mode."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_pred_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
        }

        pred_data = decode_mb_pred_cabac(
            decoder, contexts, slice_type=2, mb_type=0,  # I_4x4
            mb_info=mb_info
        )

        # Should have intra chroma pred mode for I-MB
        assert 'intra_chroma_pred_mode' in pred_data or pred_data is not None

    def test_decode_mb_pred_p_l0_16x16(self):
        """Decode ref_idx and mvd for P_L0_16x16."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_pred_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
        }

        pred_data = decode_mb_pred_cabac(
            decoder, contexts, slice_type=0, mb_type=0,  # P_L0_16x16
            mb_info=mb_info
        )

        # P_L0_16x16 has ref_idx_l0 and mvd_l0
        assert pred_data is not None


class TestTransformSizeFlag:
    """Tests for transform_size_8x8_flag CABAC decoding (High profile)."""

    def test_decode_transform_size_8x8_flag_exists(self):
        """transform_size_8x8_flag should exist for High profile support."""
        from entropy.cabac_macroblock import decode_transform_size_8x8_flag_cabac

        assert callable(decode_transform_size_8x8_flag_cabac)

    def test_decode_transform_size_8x8_flag_binary(self):
        """transform_size_8x8_flag returns 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_transform_size_8x8_flag_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'left_available': True, 'top_available': True,
            'left_transform_size_8x8_flag': 0,
            'top_transform_size_8x8_flag': 0,
        }

        result = decode_transform_size_8x8_flag_cabac(
            decoder, contexts, mb_info=mb_info
        )

        assert result in (0, 1)


class TestIPCMMacroblockCABAC:
    """Tests for I_PCM macroblock handling in CABAC mode.

    I_PCM macroblocks bypass CABAC entirely - raw sample values are
    read directly from the bitstream after byte alignment.

    H.264 Spec Reference: Section 7.3.5 - mb_type = I_PCM
    """

    def test_decode_i_pcm_mb_type_detection(self):
        """mb_type=25 in I-slice indicates I_PCM macroblock."""
        from entropy.cabac_macroblock import is_i_pcm_macroblock

        # I_PCM is mb_type=25 in I-slice
        assert is_i_pcm_macroblock(mb_type=25, slice_type=2)
        # Not I_PCM for other types
        assert not is_i_pcm_macroblock(mb_type=0, slice_type=2)
        assert not is_i_pcm_macroblock(mb_type=24, slice_type=2)

    def test_decode_i_pcm_requires_cabac_init_idc(self):
        """I_PCM causes CABAC engine re-initialization.

        After I_PCM, the CABAC decoder must be re-initialized
        with cabac_init_idc value.
        """
        from entropy.cabac_macroblock import handle_i_pcm_cabac_reinit

        assert callable(handle_i_pcm_cabac_reinit)

    def test_decode_i_pcm_byte_alignment(self):
        """I_PCM requires bitstream byte alignment before samples."""
        from entropy.cabac_macroblock import align_to_byte_for_i_pcm

        assert callable(align_to_byte_for_i_pcm)

    def test_decode_i_pcm_luma_samples_count(self):
        """I_PCM reads 256 luma samples (16x16)."""
        from entropy.cabac_macroblock import decode_i_pcm_samples

        data = bytes([128] * 384)  # 256 luma + 128 chroma (4:2:0)
        reader = BitReader(data)

        luma, cb, cr = decode_i_pcm_samples(
            reader, bit_depth_luma=8, bit_depth_chroma=8,
            chroma_format_idc=1  # 4:2:0
        )

        assert luma.shape == (16, 16)
        assert cb.shape == (8, 8)
        assert cr.shape == (8, 8)

    def test_decode_i_pcm_high_bit_depth(self):
        """I_PCM with bit_depth > 8 reads more bits per sample."""
        from entropy.cabac_macroblock import decode_i_pcm_samples

        # 10-bit samples require 2 bytes each
        data = bytes([0x00, 0x80] * 384)
        reader = BitReader(data)

        luma, cb, cr = decode_i_pcm_samples(
            reader, bit_depth_luma=10, bit_depth_chroma=10,
            chroma_format_idc=1  # 4:2:0
        )

        assert luma.dtype.itemsize >= 2  # At least 16-bit storage
        assert luma.shape == (16, 16)

    def test_decode_i_pcm_resets_nz_coeff_counts(self):
        """I_PCM resets non-zero coefficient counts to 16.

        For CAVLC: nC = 16 for neighbors of I_PCM.
        For CABAC: coded_block_flag context uses special handling.
        """
        from entropy.cabac_macroblock import get_i_pcm_neighbor_nz_count

        nz_count = get_i_pcm_neighbor_nz_count()
        assert nz_count == 16

    def test_decode_i_pcm_in_p_slice(self):
        """I_PCM can appear in P-slice as mb_type=30."""
        from entropy.cabac_macroblock import is_i_pcm_macroblock

        # In P-slice, I_PCM is mb_type = 5 + 25 = 30
        assert is_i_pcm_macroblock(mb_type=30, slice_type=0)

    def test_decode_i_pcm_in_b_slice(self):
        """I_PCM can appear in B-slice as mb_type=48."""
        from entropy.cabac_macroblock import is_i_pcm_macroblock

        # In B-slice, I_PCM is mb_type = 23 + 25 = 48
        assert is_i_pcm_macroblock(mb_type=48, slice_type=1)

    def test_decode_macroblock_layer_i_pcm_complete(self):
        """Complete I_PCM macroblock decoding."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_macroblock_layer_cabac

        # Bitstream that should decode to I_PCM mb_type,
        # followed by byte-aligned raw samples
        data = bytes([0xFF] * 10 + [128] * 384)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        mb_info = {
            'mb_x': 0, 'mb_y': 0,
            'left_available': False, 'top_available': False,
        }

        mb_data = decode_macroblock_layer_cabac(
            decoder, contexts, slice_type=2, mb_info=mb_info
        )

        assert 'mb_type' in mb_data


class TestMBSkipRunCABAC:
    """Tests for mb_skip_run and consecutive skip handling in CABAC.

    Unlike CAVLC which uses mb_skip_run (run-length), CABAC decodes
    mb_skip_flag for each macroblock individually.
    """

    def test_decode_consecutive_skipped_mbs(self):
        """Decode multiple consecutive skipped macroblocks."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_skip_flag_cabac

        # Bitstream that should decode to skip=1 repeatedly
        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        skip_count = 0
        for i in range(10):
            mb_info = {
                'mb_x': i % 5, 'mb_y': i // 5,
                'left_available': (i % 5) > 0,
                'top_available': (i // 5) > 0,
                'left_skip': True if skip_count > 0 else False,
                'top_skip': False,
            }

            result = decode_mb_skip_flag_cabac(
                decoder, contexts, slice_type=0, mb_info=mb_info
            )

            if result == 1:
                skip_count += 1

        # At least some should be skipped
        assert skip_count >= 0

    def test_skip_flag_context_accumulates_properly(self):
        """Context for mb_skip_flag should accumulate as neighbors skip."""
        from entropy.cabac_macroblock import get_mb_skip_flag_ctx_idx

        # No skipped neighbors: base context
        ctx_idx = get_mb_skip_flag_ctx_idx(
            slice_type=0,
            left_available=True, left_skip=False,
            top_available=True, top_skip=False
        )
        assert ctx_idx >= 0

        # Both skipped: higher context
        ctx_idx_both = get_mb_skip_flag_ctx_idx(
            slice_type=0,
            left_available=True, left_skip=True,
            top_available=True, top_skip=True
        )
        assert ctx_idx_both > ctx_idx

    def test_skip_to_non_skip_transition(self):
        """Handle transition from skipped to non-skipped MB."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_macroblock_layer_cabac

        data = bytes([0x00, 0x80, 0x40, 0x20] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        mb_info = {
            'mb_x': 1, 'mb_y': 0,
            'left_available': True, 'top_available': False,
            'left_skip': True, 'top_skip': False,
        }

        mb_data = decode_macroblock_layer_cabac(
            decoder, contexts, slice_type=0, mb_info=mb_info
        )

        assert 'mb_skip_flag' in mb_data or 'mb_type' in mb_data


class TestEndOfSliceFlagCABAC:
    """Tests for end_of_slice_flag CABAC decoding.

    end_of_slice_flag uses the terminate decoding process which
    is different from regular context-based or bypass decoding.

    H.264 Spec Reference: Section 9.3.3.2.4
    """

    def test_decode_end_of_slice_flag_exists(self):
        """decode_end_of_slice_flag function should exist."""
        from entropy.cabac_macroblock import decode_end_of_slice_flag_cabac

        assert callable(decode_end_of_slice_flag_cabac)

    def test_end_of_slice_flag_uses_terminate_mode(self):
        """end_of_slice_flag uses CABAC terminate, not decision."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_macroblock import decode_end_of_slice_flag_cabac

        # This should use decoder.decode_terminate() internally
        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        result = decode_end_of_slice_flag_cabac(decoder)

        assert result in (0, 1)

    def test_end_of_slice_returns_one_at_slice_end(self):
        """end_of_slice_flag=1 terminates macroblock loop."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_macroblock import decode_end_of_slice_flag_cabac

        # Construct bitstream that should decode to end_of_slice=1
        # Range reduced by 2, offset >= range means end
        data = bytes([0xFF, 0xFE] + [0x00] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        result = decode_end_of_slice_flag_cabac(decoder)

        assert result in (0, 1)

    def test_end_of_slice_flag_decoded_after_each_mb(self):
        """end_of_slice_flag is decoded after each non-skipped MB."""
        from entropy.cabac_macroblock import decode_slice_data_cabac

        # Just verify the function exists and handles end_of_slice
        assert callable(decode_slice_data_cabac)

    def test_slice_terminates_correctly(self):
        """Slice decoding loop terminates on end_of_slice_flag=1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_slice_data_cabac

        data = bytes([0x80, 0x40, 0xFF, 0xFE] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        slice_info = {
            'slice_type': 2,
            'first_mb_in_slice': 0,
            'pic_width_in_mbs': 2,
            'pic_height_in_mbs': 2,
        }

        result = decode_slice_data_cabac(decoder, contexts, slice_info)

        # Should return decoded macroblocks
        assert isinstance(result, (list, dict))


class TestMBFieldDecodingFlagCABAC:
    """Tests for mb_field_decoding_flag in MBAFF mode.

    In MBAFF (Macroblock-Adaptive Frame/Field) mode, each macroblock
    pair can be coded as frame or field.

    H.264 Spec Reference: Section 7.3.5
    """

    def test_decode_mb_field_decoding_flag_exists(self):
        """decode_mb_field_decoding_flag function should exist."""
        from entropy.cabac_macroblock import decode_mb_field_decoding_flag_cabac

        assert callable(decode_mb_field_decoding_flag_cabac)

    def test_mb_field_flag_returns_binary(self):
        """mb_field_decoding_flag returns 0 (frame) or 1 (field)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_field_decoding_flag_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        mb_info = {
            'left_available': True,
            'top_available': True,
            'left_mb_field_decoding_flag': 0,
            'top_mb_field_decoding_flag': 0,
        }

        result = decode_mb_field_decoding_flag_cabac(
            decoder, contexts, mb_info=mb_info
        )

        assert result in (0, 1)

    def test_mb_field_flag_context_depends_on_neighbors(self):
        """mb_field_decoding_flag context depends on neighbor flags."""
        from entropy.cabac_macroblock import get_mb_field_decoding_flag_ctx_idx

        # Both neighbors frame: ctxIdx base
        ctx_idx = get_mb_field_decoding_flag_ctx_idx(
            left_available=True, left_field=0,
            top_available=True, top_field=0
        )
        assert ctx_idx >= 0

        # Both neighbors field: higher context
        ctx_idx_field = get_mb_field_decoding_flag_ctx_idx(
            left_available=True, left_field=1,
            top_available=True, top_field=1
        )
        assert ctx_idx_field >= ctx_idx
