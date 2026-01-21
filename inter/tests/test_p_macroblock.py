# h264/inter/tests/test_p_macroblock.py
"""Tests for P-macroblock type parsing and handling.

Tests the parsing of P-slice macroblock types and motion data.
H.264 Spec Reference: Table 7-13, 7-14
"""

import pytest
import numpy as np

from inter.p_macroblock import (
    PMBType,
    P_MB_TYPES,
    parse_p_mb_type,
    PMacroblockInfo,
)


class TestPMBTypes:
    """Tests for P-MB type definitions."""

    def test_p_l0_16x16_type(self):
        """P_L0_16x16 has one 16x16 partition."""
        mb_type = P_MB_TYPES[0]

        assert mb_type.name == "P_L0_16x16"
        assert mb_type.num_partitions == 1
        assert mb_type.partition_width == 16
        assert mb_type.partition_height == 16

    def test_p_l0_l0_16x8_type(self):
        """P_L0_L0_16x8 has two 16x8 partitions."""
        mb_type = P_MB_TYPES[1]

        assert mb_type.name == "P_L0_L0_16x8"
        assert mb_type.num_partitions == 2
        assert mb_type.partition_width == 16
        assert mb_type.partition_height == 8

    def test_p_l0_l0_8x16_type(self):
        """P_L0_L0_8x16 has two 8x16 partitions."""
        mb_type = P_MB_TYPES[2]

        assert mb_type.name == "P_L0_L0_8x16"
        assert mb_type.num_partitions == 2
        assert mb_type.partition_width == 8
        assert mb_type.partition_height == 16

    def test_p_8x8_type(self):
        """P_8x8 has four 8x8 partitions."""
        mb_type = P_MB_TYPES[3]

        assert mb_type.name == "P_8x8"
        assert mb_type.num_partitions == 4
        assert mb_type.partition_width == 8
        assert mb_type.partition_height == 8

    def test_p_8x8ref0_type(self):
        """P_8x8ref0 has four 8x8 partitions with ref_idx=0."""
        mb_type = P_MB_TYPES[4]

        assert mb_type.name == "P_8x8ref0"
        assert mb_type.num_partitions == 4
        assert mb_type.ref_idx_forced == 0


class TestParsePMBType:
    """Tests for parsing mb_type in P-slices."""

    def test_parse_type_0(self):
        """mb_type=0 is P_L0_16x16."""
        mb_type = parse_p_mb_type(0)
        assert mb_type.name == "P_L0_16x16"

    def test_parse_type_1(self):
        """mb_type=1 is P_L0_L0_16x8."""
        mb_type = parse_p_mb_type(1)
        assert mb_type.name == "P_L0_L0_16x8"

    def test_parse_type_2(self):
        """mb_type=2 is P_L0_L0_8x16."""
        mb_type = parse_p_mb_type(2)
        assert mb_type.name == "P_L0_L0_8x16"

    def test_parse_type_3(self):
        """mb_type=3 is P_8x8."""
        mb_type = parse_p_mb_type(3)
        assert mb_type.name == "P_8x8"

    def test_parse_type_4(self):
        """mb_type=4 is P_8x8ref0."""
        mb_type = parse_p_mb_type(4)
        assert mb_type.name == "P_8x8ref0"

    def test_parse_type_5_is_intra(self):
        """mb_type >= 5 is I-MB in P-slice (offset by 5)."""
        # In P-slices, mb_type 5-30 map to I-MB types 0-25
        mb_type = parse_p_mb_type(5)
        assert mb_type.is_intra is True
        assert mb_type.intra_type == 0  # I_4x4


class TestPMacroblockInfo:
    """Tests for P-MB info structure."""

    def test_create_p_16x16_info(self):
        """Create info for P_L0_16x16."""
        info = PMacroblockInfo(
            mb_type=P_MB_TYPES[0],
            ref_idx=[0],
            mvd=[(4, -2)],
        )

        assert info.mb_type.name == "P_L0_16x16"
        assert len(info.ref_idx) == 1
        assert info.ref_idx[0] == 0
        assert info.mvd[0] == (4, -2)

    def test_create_p_16x8_info(self):
        """Create info for P_L0_L0_16x8 with two partitions."""
        info = PMacroblockInfo(
            mb_type=P_MB_TYPES[1],
            ref_idx=[0, 1],
            mvd=[(2, 0), (-4, 6)],
        )

        assert info.mb_type.num_partitions == 2
        assert len(info.ref_idx) == 2
        assert len(info.mvd) == 2

    def test_create_p_8x8_info(self):
        """Create info for P_8x8 with four partitions."""
        info = PMacroblockInfo(
            mb_type=P_MB_TYPES[3],
            ref_idx=[0, 0, 1, 1],
            mvd=[(0, 0), (2, 2), (-2, -2), (4, 4)],
        )

        assert info.mb_type.num_partitions == 4
        assert len(info.ref_idx) == 4
        assert len(info.mvd) == 4

    def test_p_skip_info(self):
        """P_Skip has no explicit data - all inferred."""
        info = PMacroblockInfo.create_skip()

        assert info.is_skip is True
        assert info.ref_idx == [0]
        # MVD is (0, 0) for skip - actual MV is just the prediction

    def test_partition_positions_16x16(self):
        """Get partition positions for 16x16."""
        info = PMacroblockInfo(
            mb_type=P_MB_TYPES[0],
            ref_idx=[0],
            mvd=[(0, 0)],
        )

        positions = info.get_partition_positions()
        assert len(positions) == 1
        assert positions[0] == (0, 0, 16, 16)  # (x, y, w, h)

    def test_partition_positions_16x8(self):
        """Get partition positions for 16x8."""
        info = PMacroblockInfo(
            mb_type=P_MB_TYPES[1],
            ref_idx=[0, 0],
            mvd=[(0, 0), (0, 0)],
        )

        positions = info.get_partition_positions()
        assert len(positions) == 2
        assert positions[0] == (0, 0, 16, 8)   # Top
        assert positions[1] == (0, 8, 16, 8)   # Bottom

    def test_partition_positions_8x16(self):
        """Get partition positions for 8x16."""
        info = PMacroblockInfo(
            mb_type=P_MB_TYPES[2],
            ref_idx=[0, 0],
            mvd=[(0, 0), (0, 0)],
        )

        positions = info.get_partition_positions()
        assert len(positions) == 2
        assert positions[0] == (0, 0, 8, 16)   # Left
        assert positions[1] == (8, 0, 8, 16)   # Right

    def test_partition_positions_8x8(self):
        """Get partition positions for 8x8."""
        info = PMacroblockInfo(
            mb_type=P_MB_TYPES[3],
            ref_idx=[0, 0, 0, 0],
            mvd=[(0, 0)] * 4,
        )

        positions = info.get_partition_positions()
        assert len(positions) == 4
        assert positions[0] == (0, 0, 8, 8)   # Top-left
        assert positions[1] == (8, 0, 8, 8)   # Top-right
        assert positions[2] == (0, 8, 8, 8)   # Bottom-left
        assert positions[3] == (8, 8, 8, 8)   # Bottom-right


class TestSubMBTypes:
    """Tests for sub-macroblock types (within P_8x8)."""

    def test_sub_mb_8x8(self):
        """Sub-MB type 0 is 8x8."""
        from inter.p_macroblock import SUB_MB_TYPES

        sub_type = SUB_MB_TYPES[0]
        assert sub_type.name == "P_L0_8x8"
        assert sub_type.width == 8
        assert sub_type.height == 8
        assert sub_type.num_parts == 1

    def test_sub_mb_8x4(self):
        """Sub-MB type 1 is 8x4 (two 8x4 partitions)."""
        from inter.p_macroblock import SUB_MB_TYPES

        sub_type = SUB_MB_TYPES[1]
        assert sub_type.name == "P_L0_8x4"
        assert sub_type.width == 8
        assert sub_type.height == 4
        assert sub_type.num_parts == 2

    def test_sub_mb_4x8(self):
        """Sub-MB type 2 is 4x8 (two 4x8 partitions)."""
        from inter.p_macroblock import SUB_MB_TYPES

        sub_type = SUB_MB_TYPES[2]
        assert sub_type.name == "P_L0_4x8"
        assert sub_type.width == 4
        assert sub_type.height == 8
        assert sub_type.num_parts == 2

    def test_sub_mb_4x4(self):
        """Sub-MB type 3 is 4x4 (four 4x4 partitions)."""
        from inter.p_macroblock import SUB_MB_TYPES

        sub_type = SUB_MB_TYPES[3]
        assert sub_type.name == "P_L0_4x4"
        assert sub_type.width == 4
        assert sub_type.height == 4
        assert sub_type.num_parts == 4
