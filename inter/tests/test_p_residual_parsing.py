# h264/inter/tests/test_p_residual_parsing.py
"""Tests for P-macroblock residual bitstream parsing.

Tests the parsing of coded_block_pattern and residual coefficients
for P-macroblocks.

H.264 Spec Reference: Section 7.3.5.3 - Residual data syntax
"""

import pytest
import numpy as np

from bitstream import BitReader


class TestInterCBPMapping:
    """Tests for inter macroblock CBP code mapping.

    Inter MBs use a different VLC table for CBP than intra MBs.
    Table 9-4 in H.264 spec maps codeNum to (luma_cbp, chroma_cbp).
    """

    def test_cbp_mapping_table_exists(self):
        """CBP mapping table for inter MBs exists."""
        from inter.p_macroblock import INTER_CBP_TABLE
        assert len(INTER_CBP_TABLE) == 48  # 48 entries for inter

    def test_cbp_code_0_maps_to_cbp_0(self):
        """codeNum 0 maps to CBP 0 (no residual)."""
        from inter.p_macroblock import INTER_CBP_TABLE
        assert INTER_CBP_TABLE[0] == 0

    def test_cbp_code_1_maps_to_cbp_16(self):
        """codeNum 1 maps to CBP 16 (chroma DC only)."""
        from inter.p_macroblock import INTER_CBP_TABLE
        assert INTER_CBP_TABLE[1] == 16

    def test_cbp_code_2_maps_to_cbp_1(self):
        """codeNum 2 maps to CBP 1 (first 8x8 luma block)."""
        from inter.p_macroblock import INTER_CBP_TABLE
        assert INTER_CBP_TABLE[2] == 1

    def test_decode_cbp_from_bitstream(self):
        """Decode CBP from actual bitstream."""
        from inter.p_macroblock import decode_inter_cbp

        # Create bitstream with UE-coded value 0 (single 1 bit)
        data = bytes([0b10000000])
        reader = BitReader(data)

        cbp = decode_inter_cbp(reader)
        assert cbp == 0  # codeNum 0 -> CBP 0


class TestPMBResidualParsing:
    """Tests for parsing residual data in P-macroblocks."""

    def test_parse_no_residual_when_cbp_zero(self):
        """When CBP=0, no residual blocks are parsed."""
        from inter.p_macroblock import parse_p_residual

        # CBP=0 encoded as UE(0) = '1'
        data = bytes([0b10000000])
        reader = BitReader(data)

        residual = parse_p_residual(reader, cbp=0, qp=26)

        assert residual.luma is None
        assert residual.cb is None
        assert residual.cr is None

    def test_parse_qp_delta_when_cbp_nonzero(self):
        """When CBP!=0, mb_qp_delta is parsed."""
        from inter.p_macroblock import parse_p_mb_qp_delta

        # SE(0) = '1' (qp_delta = 0)
        data = bytes([0b10000000])
        reader = BitReader(data)

        qp_delta = parse_p_mb_qp_delta(reader)
        assert qp_delta == 0

    def test_parse_positive_qp_delta(self):
        """Positive QP delta."""
        from inter.p_macroblock import parse_p_mb_qp_delta

        # SE(1) = '010' (qp_delta = +1)
        data = bytes([0b01000000])
        reader = BitReader(data)

        qp_delta = parse_p_mb_qp_delta(reader)
        assert qp_delta == 1

    def test_parse_negative_qp_delta(self):
        """Negative QP delta."""
        from inter.p_macroblock import parse_p_mb_qp_delta

        # SE(-1) = '011' (qp_delta = -1)
        data = bytes([0b01100000])
        reader = BitReader(data)

        qp_delta = parse_p_mb_qp_delta(reader)
        assert qp_delta == -1


class TestResidualBlockLayout:
    """Tests for residual block organization within P-MB."""

    def test_luma_8x8_block_indices(self):
        """Luma 8x8 block indices map to correct 4x4 sub-blocks."""
        from inter.p_macroblock import get_luma_4x4_indices_for_8x8

        # 8x8 block 0 (top-left) contains 4x4 blocks 0,1,4,5
        indices = get_luma_4x4_indices_for_8x8(0)
        assert indices == [0, 1, 4, 5]

        # 8x8 block 1 (top-right) contains 4x4 blocks 2,3,6,7
        indices = get_luma_4x4_indices_for_8x8(1)
        assert indices == [2, 3, 6, 7]

        # 8x8 block 2 (bottom-left) contains 4x4 blocks 8,9,12,13
        indices = get_luma_4x4_indices_for_8x8(2)
        assert indices == [8, 9, 12, 13]

        # 8x8 block 3 (bottom-right) contains 4x4 blocks 10,11,14,15
        indices = get_luma_4x4_indices_for_8x8(3)
        assert indices == [10, 11, 14, 15]

    def test_4x4_block_position_in_mb(self):
        """4x4 block index maps to (x, y) position in macroblock.

        H.264 4x4 block scan order:
            0  1  4  5       positions: (0,0) (4,0) (8,0) (12,0)
            2  3  6  7                  (0,4) (4,4) (8,4) (12,4)
            8  9 12 13                  (0,8) (4,8) (8,12) (12,8)
           10 11 14 15                  (0,12)(4,12)(8,12)(12,12)
        """
        from inter.p_macroblock import get_4x4_position

        # Block 0 at (0, 0)
        assert get_4x4_position(0) == (0, 0)

        # Block 1 at (4, 0)
        assert get_4x4_position(1) == (4, 0)

        # Block 2 at (0, 4) - second row of first 8x8
        assert get_4x4_position(2) == (0, 4)

        # Block 4 at (8, 0) - top-left of second 8x8
        assert get_4x4_position(4) == (8, 0)

        # Block 5 at (12, 0)
        assert get_4x4_position(5) == (12, 0)

        # Block 15 at (12, 12)
        assert get_4x4_position(15) == (12, 12)


class TestChromaResidualParsing:
    """Tests for chroma residual parsing."""

    def test_chroma_cbp_dc_only(self):
        """Chroma CBP=1 means DC coefficients only."""
        from inter.p_macroblock import should_parse_chroma_dc, should_parse_chroma_ac

        assert should_parse_chroma_dc(chroma_cbp=1) is True
        assert should_parse_chroma_ac(chroma_cbp=1) is False

    def test_chroma_cbp_dc_and_ac(self):
        """Chroma CBP=2 means DC and AC coefficients."""
        from inter.p_macroblock import should_parse_chroma_dc, should_parse_chroma_ac

        assert should_parse_chroma_dc(chroma_cbp=2) is True
        assert should_parse_chroma_ac(chroma_cbp=2) is True

    def test_chroma_cbp_none(self):
        """Chroma CBP=0 means no chroma residual."""
        from inter.p_macroblock import should_parse_chroma_dc, should_parse_chroma_ac

        assert should_parse_chroma_dc(chroma_cbp=0) is False
        assert should_parse_chroma_ac(chroma_cbp=0) is False


class TestPResidualDataclass:
    """Tests for PResidual dataclass."""

    def test_create_empty_residual(self):
        """Create residual with no data."""
        from inter.p_macroblock import PResidual

        residual = PResidual()
        assert residual.luma is None
        assert residual.cb is None
        assert residual.cr is None

    def test_create_with_luma_only(self):
        """Create residual with luma data."""
        from inter.p_macroblock import PResidual

        luma = np.zeros((16, 16), dtype=np.int32)
        residual = PResidual(luma=luma)

        assert residual.luma is not None
        assert residual.luma.shape == (16, 16)
        assert residual.cb is None

    def test_create_with_all_components(self):
        """Create residual with all components."""
        from inter.p_macroblock import PResidual

        luma = np.zeros((16, 16), dtype=np.int32)
        cb = np.zeros((8, 8), dtype=np.int32)
        cr = np.zeros((8, 8), dtype=np.int32)

        residual = PResidual(luma=luma, cb=cb, cr=cr)

        assert residual.luma.shape == (16, 16)
        assert residual.cb.shape == (8, 8)
        assert residual.cr.shape == (8, 8)
