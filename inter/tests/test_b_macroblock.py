# h264/inter/tests/test_b_macroblock.py
"""RED TESTS: B-macroblock type parsing.

B-macroblocks have more complex prediction modes than P-macroblocks:
- B_Direct: Inferred MVs
- B_L0: Forward prediction only
- B_L1: Backward prediction only
- B_Bi: Bidirectional (average of L0 and L1)

H.264 Spec Reference: Table 7-14 - Macroblock types for B slices

These tests SHOULD FAIL until B-macroblock support is implemented.
"""

import pytest
import numpy as np

from bitstream import BitWriter, BitReader


class TestBMacroblockTypes:
    """Tests for B-macroblock type parsing."""

    def test_parse_b_mb_type_exists(self):
        """parse_b_mb_type function should exist."""
        from inter.b_macroblock import parse_b_mb_type

        assert callable(parse_b_mb_type)

    def test_b_direct_16x16(self):
        """mb_type=0 is B_Direct_16x16."""
        from inter.b_macroblock import parse_b_mb_type

        mb_type = parse_b_mb_type(0)

        assert mb_type.name == "B_Direct_16x16"
        assert mb_type.is_direct is True

    def test_b_l0_16x16(self):
        """mb_type=1 is B_L0_16x16."""
        from inter.b_macroblock import parse_b_mb_type

        mb_type = parse_b_mb_type(1)

        assert mb_type.name == "B_L0_16x16"
        assert mb_type.pred_mode == "L0"
        assert mb_type.num_partitions == 1

    def test_b_l1_16x16(self):
        """mb_type=2 is B_L1_16x16."""
        from inter.b_macroblock import parse_b_mb_type

        mb_type = parse_b_mb_type(2)

        assert mb_type.name == "B_L1_16x16"
        assert mb_type.pred_mode == "L1"
        assert mb_type.num_partitions == 1

    def test_b_bi_16x16(self):
        """mb_type=3 is B_Bi_16x16."""
        from inter.b_macroblock import parse_b_mb_type

        mb_type = parse_b_mb_type(3)

        assert mb_type.name == "B_Bi_16x16"
        assert mb_type.pred_mode == "Bi"
        assert mb_type.num_partitions == 1

    def test_b_l0_l0_16x8(self):
        """mb_type=4 is B_L0_L0_16x8."""
        from inter.b_macroblock import parse_b_mb_type

        mb_type = parse_b_mb_type(4)

        assert mb_type.name == "B_L0_L0_16x8"
        assert mb_type.num_partitions == 2
        assert mb_type.partition_size == (16, 8)

    def test_b_8x8(self):
        """mb_type=22 is B_8x8."""
        from inter.b_macroblock import parse_b_mb_type

        mb_type = parse_b_mb_type(22)

        assert mb_type.name == "B_8x8"
        assert mb_type.num_partitions == 4


class TestBSubMacroblockTypes:
    """Tests for B_8x8 sub-macroblock types."""

    def test_parse_b_sub_mb_type_exists(self):
        """parse_b_sub_mb_type function should exist."""
        from inter.b_macroblock import parse_b_sub_mb_type

        assert callable(parse_b_sub_mb_type)

    def test_b_direct_8x8(self):
        """sub_mb_type=0 is B_Direct_8x8."""
        from inter.b_macroblock import parse_b_sub_mb_type

        sub_type = parse_b_sub_mb_type(0)

        assert sub_type.name == "B_Direct_8x8"
        assert sub_type.is_direct is True

    def test_b_l0_8x8(self):
        """sub_mb_type=1 is B_L0_8x8."""
        from inter.b_macroblock import parse_b_sub_mb_type

        sub_type = parse_b_sub_mb_type(1)

        assert sub_type.name == "B_L0_8x8"
        assert sub_type.pred_mode == "L0"

    def test_b_l1_8x8(self):
        """sub_mb_type=2 is B_L1_8x8."""
        from inter.b_macroblock import parse_b_sub_mb_type

        sub_type = parse_b_sub_mb_type(2)

        assert sub_type.name == "B_L1_8x8"
        assert sub_type.pred_mode == "L1"

    def test_b_bi_8x8(self):
        """sub_mb_type=3 is B_Bi_8x8."""
        from inter.b_macroblock import parse_b_sub_mb_type

        sub_type = parse_b_sub_mb_type(3)

        assert sub_type.name == "B_Bi_8x8"
        assert sub_type.pred_mode == "Bi"


class TestBMacroblockInfo:
    """Tests for BMacroblockInfo dataclass."""

    def test_b_mb_info_exists(self):
        """BMacroblockInfo dataclass should exist."""
        from inter.b_macroblock import BMacroblockInfo

        assert BMacroblockInfo is not None

    def test_b_mb_info_has_required_fields(self):
        """BMacroblockInfo should have all required fields."""
        from inter.b_macroblock import BMacroblockInfo

        # Check required fields exist
        info = BMacroblockInfo(
            name="B_Bi_16x16",
            mb_type=3,
            is_direct=False,
            pred_mode="Bi",
            num_partitions=1,
        )

        assert info.name == "B_Bi_16x16"
        assert info.mb_type == 3
        assert info.is_direct is False
        assert info.pred_mode == "Bi"
        assert info.num_partitions == 1

    def test_b_mb_info_partition_info(self):
        """BMacroblockInfo should track partition sizes."""
        from inter.b_macroblock import BMacroblockInfo

        info = BMacroblockInfo(
            name="B_L0_L0_16x8",
            mb_type=4,
            is_direct=False,
            pred_mode="L0",
            num_partitions=2,
            partition_size=(16, 8),
        )

        assert info.partition_size == (16, 8)


class TestBSkipDetection:
    """Tests for B_Skip macroblock detection."""

    def test_b_skip_is_direct_mode(self):
        """B_Skip should be treated as B_Direct_16x16 with no residual."""
        from inter.b_macroblock import is_b_skip

        # B_Skip is signaled via mb_skip_run, not mb_type
        # When detected, it's equivalent to B_Direct_16x16
        assert callable(is_b_skip)

    def test_b_skip_has_no_residual(self):
        """B_Skip macroblocks have no coded residual."""
        from inter.b_macroblock import BMacroblockInfo

        # B_Skip should have cbp=0 (no residual)
        skip_info = BMacroblockInfo(
            name="B_Skip",
            mb_type=-1,  # Special value for skip
            is_direct=True,
            pred_mode="Direct",
            num_partitions=1,
            is_skip=True,
        )

        assert skip_info.is_skip is True
        assert skip_info.is_direct is True
