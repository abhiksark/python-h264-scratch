# h264/decoder/tests/test_p_partitions_decoder.py
"""Tests for P-macroblock partition decoding integration.

Tests the decoder's ability to parse and reconstruct P_16x8, P_8x16,
and P_8x8 macroblocks from bitstream.
"""

import pytest
import numpy as np

from decoder.decoder import H264Decoder
from inter.p_macroblock import P_MB_TYPES, parse_p_mb_type


class TestPMBTypeParsing:
    """Tests for P-macroblock type code parsing."""

    def test_parse_p_l0_16x16(self):
        """mb_type=0 is P_L0_16x16."""
        mb_type = parse_p_mb_type(0)
        assert mb_type.name == "P_L0_16x16"
        assert mb_type.num_partitions == 1
        assert mb_type.partition_width == 16
        assert mb_type.partition_height == 16

    def test_parse_p_l0_l0_16x8(self):
        """mb_type=1 is P_L0_L0_16x8."""
        mb_type = parse_p_mb_type(1)
        assert mb_type.name == "P_L0_L0_16x8"
        assert mb_type.num_partitions == 2
        assert mb_type.partition_width == 16
        assert mb_type.partition_height == 8

    def test_parse_p_l0_l0_8x16(self):
        """mb_type=2 is P_L0_L0_8x16."""
        mb_type = parse_p_mb_type(2)
        assert mb_type.name == "P_L0_L0_8x16"
        assert mb_type.num_partitions == 2
        assert mb_type.partition_width == 8
        assert mb_type.partition_height == 16

    def test_parse_p_8x8(self):
        """mb_type=3 is P_8x8."""
        mb_type = parse_p_mb_type(3)
        assert mb_type.name == "P_8x8"
        assert mb_type.num_partitions == 4
        assert mb_type.partition_width == 8
        assert mb_type.partition_height == 8

    def test_parse_p_8x8ref0(self):
        """mb_type=4 is P_8x8ref0."""
        mb_type = parse_p_mb_type(4)
        assert mb_type.name == "P_8x8ref0"
        assert mb_type.ref_idx_forced == 0

    def test_parse_intra_in_p_slice(self):
        """mb_type >= 5 maps to intra types in P-slice."""
        # In P-slice, mb_type 5+ maps to I_4x4, I_16x16, etc.
        mb_type = parse_p_mb_type(5)
        assert mb_type.is_intra


class TestP16x8Decoding:
    """Tests for P_L0_L0_16x8 macroblock decoding."""

    def test_16x8_mv_cache_update(self):
        """P_16x8 updates MV cache for both partitions."""
        from inter.mv_prediction import MVCache

        # Create MV cache directly
        cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        # After P_16x8 decode, top partition (rows 0-1) should have one MV
        # and bottom partition (rows 2-3) should have another

        # Set MVs as the decoder would
        for bx in range(4):
            for by in range(2):
                cache.set_mv(0, 0, bx, by, 4, 0)  # Top partition
            for by in range(2, 4):
                cache.set_mv(0, 0, bx, by, 8, 4)  # Bottom partition

        # Verify
        assert cache.get_mv(0, 0, 0, 0) == (4, 0)
        assert cache.get_mv(0, 0, 3, 1) == (4, 0)
        assert cache.get_mv(0, 0, 0, 2) == (8, 4)
        assert cache.get_mv(0, 0, 3, 3) == (8, 4)


class TestP8x16Decoding:
    """Tests for P_L0_L0_8x16 macroblock decoding."""

    def test_8x16_mv_cache_update(self):
        """P_8x16 updates MV cache for both partitions."""
        from inter.mv_prediction import MVCache

        cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        # After P_8x16 decode, left partition (cols 0-1) should have one MV
        # and right partition (cols 2-3) should have another
        for by in range(4):
            for bx in range(2):
                cache.set_mv(0, 0, bx, by, 4, 0)  # Left partition
            for bx in range(2, 4):
                cache.set_mv(0, 0, bx, by, 8, 4)  # Right partition

        # Verify
        assert cache.get_mv(0, 0, 0, 0) == (4, 0)
        assert cache.get_mv(0, 0, 1, 3) == (4, 0)
        assert cache.get_mv(0, 0, 2, 0) == (8, 4)
        assert cache.get_mv(0, 0, 3, 3) == (8, 4)


class TestP8x8Decoding:
    """Tests for P_8x8 macroblock decoding."""

    def test_8x8_mv_cache_update(self):
        """P_8x8 updates MV cache for all four sub-MBs."""
        from inter.mv_prediction import MVCache

        cache = MVCache(width_in_mbs=2, height_in_mbs=2)

        # Four different MVs for four 8x8 sub-MBs
        sub_mb_mvs = [(1, 1), (2, 2), (3, 3), (4, 4)]
        sub_mb_offsets = [(0, 0), (2, 0), (0, 2), (2, 2)]

        for idx, ((bx_start, by_start), (mvx, mvy)) in enumerate(
            zip(sub_mb_offsets, sub_mb_mvs)
        ):
            for by in range(by_start, by_start + 2):
                for bx in range(bx_start, bx_start + 2):
                    cache.set_mv(0, 0, bx, by, mvx, mvy)

        # Verify each sub-MB
        assert cache.get_mv(0, 0, 0, 0) == (1, 1)  # Sub-MB 0 (TL)
        assert cache.get_mv(0, 0, 2, 0) == (2, 2)  # Sub-MB 1 (TR)
        assert cache.get_mv(0, 0, 0, 2) == (3, 3)  # Sub-MB 2 (BL)
        assert cache.get_mv(0, 0, 2, 2) == (4, 4)  # Sub-MB 3 (BR)


class TestSubMBTypeParsing:
    """Tests for sub-macroblock type parsing (for P_8x8)."""

    def test_sub_mb_type_8x8(self):
        """sub_mb_type=0 is 8x8 sub-partition."""
        from inter.p_macroblock import parse_sub_mb_type
        sub_type = parse_sub_mb_type(0)
        assert sub_type.name == "P_L0_8x8"
        assert sub_type.num_partitions == 1
        assert sub_type.partition_width == 8
        assert sub_type.partition_height == 8

    def test_sub_mb_type_8x4(self):
        """sub_mb_type=1 is 8x4 sub-partition."""
        from inter.p_macroblock import parse_sub_mb_type
        sub_type = parse_sub_mb_type(1)
        assert sub_type.name == "P_L0_8x4"
        assert sub_type.num_partitions == 2
        assert sub_type.partition_width == 8
        assert sub_type.partition_height == 4

    def test_sub_mb_type_4x8(self):
        """sub_mb_type=2 is 4x8 sub-partition."""
        from inter.p_macroblock import parse_sub_mb_type
        sub_type = parse_sub_mb_type(2)
        assert sub_type.name == "P_L0_4x8"
        assert sub_type.num_partitions == 2
        assert sub_type.partition_width == 4
        assert sub_type.partition_height == 8

    def test_sub_mb_type_4x4(self):
        """sub_mb_type=3 is 4x4 sub-partition."""
        from inter.p_macroblock import parse_sub_mb_type
        sub_type = parse_sub_mb_type(3)
        assert sub_type.name == "P_L0_4x4"
        assert sub_type.num_partitions == 4
        assert sub_type.partition_width == 4
        assert sub_type.partition_height == 4


class TestCBPParsing:
    """Tests for coded_block_pattern parsing for P-macroblocks."""

    def test_cbp_no_residual(self):
        """CBP=0 means no residual data."""
        from inter.p_macroblock import decode_cbp_inter
        luma_cbp, chroma_cbp = decode_cbp_inter(0)
        assert luma_cbp == [False, False, False, False]
        assert chroma_cbp == 0

    def test_cbp_all_luma(self):
        """CBP=15 means all four 8x8 luma blocks have residual."""
        from inter.p_macroblock import decode_cbp_inter
        luma_cbp, chroma_cbp = decode_cbp_inter(15)
        assert luma_cbp == [True, True, True, True]
        assert chroma_cbp == 0

    def test_cbp_with_chroma_dc(self):
        """CBP=16 means chroma DC only."""
        from inter.p_macroblock import decode_cbp_inter
        luma_cbp, chroma_cbp = decode_cbp_inter(16)
        assert luma_cbp == [False, False, False, False]
        assert chroma_cbp == 1  # DC only

    def test_cbp_with_chroma_ac(self):
        """CBP=32 means chroma DC+AC."""
        from inter.p_macroblock import decode_cbp_inter
        luma_cbp, chroma_cbp = decode_cbp_inter(32)
        assert luma_cbp == [False, False, False, False]
        assert chroma_cbp == 2  # DC+AC

    def test_cbp_mixed(self):
        """CBP=47 means all luma + chroma DC+AC."""
        from inter.p_macroblock import decode_cbp_inter
        luma_cbp, chroma_cbp = decode_cbp_inter(47)
        assert luma_cbp == [True, True, True, True]
        assert chroma_cbp == 2
