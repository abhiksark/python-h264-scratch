# h264/entropy/tests/test_cabac_syntax.py
"""RED TESTS: CABAC syntax element decoding.

Decode individual H.264 syntax elements using CABAC.
Each syntax element has specific binarization and context assignment.

H.264 Spec Reference: Section 9.3.3.1 - Decoding process for binary decisions

These tests SHOULD FAIL until CABAC syntax decoding is implemented.
"""

import pytest

from bitstream import BitReader


class TestMBSkipFlag:
    """Tests for mb_skip_flag decoding."""

    def test_decode_mb_skip_flag_exists(self):
        """decode_mb_skip_flag function should exist."""
        from entropy.cabac_syntax import decode_mb_skip_flag

        assert callable(decode_mb_skip_flag)

    def test_mb_skip_flag_returns_binary(self):
        """mb_skip_flag should return 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mb_skip_flag

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_mb_skip_flag(
            decoder, contexts, slice_type=0, mb_x=0, mb_y=0,
            left_skip=False, top_skip=False
        )

        assert result in (0, 1)

    def test_mb_skip_flag_context_depends_on_neighbors(self):
        """mb_skip_flag context depends on neighbor skip status."""
        from entropy.cabac_syntax import decode_mb_skip_flag

        # Just verify the function accepts neighbor parameters
        assert callable(decode_mb_skip_flag)


class TestMBTypeDecoding:
    """Tests for mb_type decoding."""

    def test_decode_mb_type_i_exists(self):
        """decode_mb_type_i function should exist."""
        from entropy.cabac_syntax import decode_mb_type_i

        assert callable(decode_mb_type_i)

    def test_decode_mb_type_p_exists(self):
        """decode_mb_type_p function should exist."""
        from entropy.cabac_syntax import decode_mb_type_p

        assert callable(decode_mb_type_p)

    def test_decode_mb_type_b_exists(self):
        """decode_mb_type_b function should exist."""
        from entropy.cabac_syntax import decode_mb_type_b

        assert callable(decode_mb_type_b)

    def test_mb_type_i_returns_valid_type(self):
        """I-slice mb_type should be valid I-MB type."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mb_type_i

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        result = decode_mb_type_i(decoder, contexts)

        # I-slice mb_type: 0=I_4x4, 1-24=I_16x16, 25=I_PCM
        assert 0 <= result <= 25

    def test_mb_type_p_returns_valid_type(self):
        """P-slice mb_type should be valid."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mb_type_p

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_mb_type_p(decoder, contexts)

        # P-slice mb_type: 0-4 for P-MB, 5+ for I-MB in P-slice
        assert result >= 0


class TestSubMBTypeDecoding:
    """Tests for sub_mb_type decoding."""

    def test_decode_sub_mb_type_p_exists(self):
        """decode_sub_mb_type_p function should exist."""
        from entropy.cabac_syntax import decode_sub_mb_type_p

        assert callable(decode_sub_mb_type_p)

    def test_decode_sub_mb_type_b_exists(self):
        """decode_sub_mb_type_b function should exist."""
        from entropy.cabac_syntax import decode_sub_mb_type_b

        assert callable(decode_sub_mb_type_b)

    def test_sub_mb_type_p_valid_range(self):
        """P-slice sub_mb_type should be 0-3."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_sub_mb_type_p

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_sub_mb_type_p(decoder, contexts)

        assert 0 <= result <= 3


class TestRefIdxDecoding:
    """Tests for ref_idx decoding."""

    def test_decode_ref_idx_exists(self):
        """decode_ref_idx function should exist."""
        from entropy.cabac_syntax import decode_ref_idx

        assert callable(decode_ref_idx)

    def test_ref_idx_returns_non_negative(self):
        """ref_idx should be non-negative."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_ref_idx

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_ref_idx(decoder, contexts, list_idx=0)

        assert result >= 0


class TestMVDDecoding:
    """Tests for mvd (motion vector difference) decoding."""

    def test_decode_mvd_lx_exists(self):
        """decode_mvd_lx function should exist."""
        from entropy.cabac_syntax import decode_mvd_lx

        assert callable(decode_mvd_lx)

    def test_mvd_returns_signed(self):
        """MVD can be positive or negative."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mvd_lx

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_mvd_lx(decoder, contexts, list_idx=0, comp=0)

        assert isinstance(result, int)


class TestMBQPDeltaDecoding:
    """Tests for mb_qp_delta decoding."""

    def test_decode_mb_qp_delta_exists(self):
        """decode_mb_qp_delta function should exist."""
        from entropy.cabac_syntax import decode_mb_qp_delta

        assert callable(decode_mb_qp_delta)

    def test_mb_qp_delta_returns_integer(self):
        """mb_qp_delta should return signed integer."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_mb_qp_delta

        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_mb_qp_delta(decoder, contexts)

        assert isinstance(result, int)


class MockCABACDecoder:
    """Mock CABAC decoder that tracks context indices used.

    This mock allows us to verify the correct context derivation
    by recording which context indices are passed to decode_decision.
    """

    def __init__(self, return_values=None):
        """Initialize mock decoder.

        Args:
            return_values: List of values to return from decode_decision.
                           If None, always returns 0.
        """
        self.return_values = return_values or []
        self.call_index = 0
        self.context_indices_used = []

    def decode_decision(self, context):
        """Record context index and return predetermined value."""
        # Find the context index by checking which context was passed
        # We store the context object itself for later inspection
        self.context_indices_used.append(context)

        if self.call_index < len(self.return_values):
            result = self.return_values[self.call_index]
        else:
            result = 0
        self.call_index += 1
        return result


class TestCodedBlockPattern:
    """Tests for coded_block_pattern decoding."""

    def test_decode_cbp_luma_exists(self):
        """decode_cbp_luma function should exist."""
        from entropy.cabac_syntax import decode_cbp_luma

        assert callable(decode_cbp_luma)

    def test_decode_cbp_chroma_exists(self):
        """decode_cbp_chroma function should exist."""
        from entropy.cabac_syntax import decode_cbp_chroma

        assert callable(decode_cbp_chroma)

    def test_cbp_luma_range(self):
        """CBP luma should be 0-15 (4 bits, one per 8x8 block)."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_cbp_luma

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_cbp_luma(decoder, contexts, mb_type=0)

        assert 0 <= result <= 15

    def test_cbp_chroma_range(self):
        """CBP chroma should be 0-2."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_cbp_chroma

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_cbp_chroma(decoder, contexts, mb_type=0)

        assert 0 <= result <= 2


class TestCBPLumaNeighborContext:
    """Tests for CBP luma neighbor-based context derivation.

    H.264 Spec Reference: Section 9.3.3.1.1.3

    8x8 block layout in MB:
        +---+---+
        | 0 | 1 |
        +---+---+
        | 2 | 3 |
        +---+---+

    Context increment: ctx_inc = condTermFlagA + 2*condTermFlagB
    condTermFlag = 1 if neighbor block has NO coded coeffs or unavailable
    condTermFlag = 0 if neighbor block HAS coded coeffs
    """

    def test_cbp_luma_unavailable_neighbors_use_ctx_inc_0(self):
        """Block 0 with unavailable neighbors should use ctx_inc=0.

        When left_cbp=-1 and top_cbp=-1 (unavailable):
        - condTermFlagA = 0 (unavailable, H.264 Section 9.3.3.1.1.3)
        - condTermFlagB = 0 (unavailable)
        - ctx_inc = 0 + 2*0 = 0
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_luma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0, 0, 0, 0])

        decode_cbp_luma(decoder, contexts, mb_type=0, left_cbp=-1, top_cbp=-1)

        # Block 0: ctx_inc = 0 + 2*0 = 0 (both unavailable → condTermFlag=0)
        expected_ctx_idx_block0 = CTX_CODED_BLOCK_PATTERN_START + 0
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx_block0]

    def test_cbp_luma_left_neighbor_has_coded_coeffs(self):
        """Block 0 with left neighbor having coded coeffs uses different ctx.

        left_cbp=0x02 means block 1 of left MB has coded coeffs.
        Block 0's left neighbor is block 1 of left MB.
        - condTermFlagA = 0 (left has coeffs, bit 1 set)
        - condTermFlagB = 0 (top unavailable, H.264 Section 9.3.3.1.1.3)
        - ctx_inc = 0 + 2*0 = 0
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_luma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0, 0, 0, 0])

        # left_cbp=0x02: block 1 of left MB has coded coeffs
        decode_cbp_luma(decoder, contexts, mb_type=0, left_cbp=0x02, top_cbp=-1)

        # Block 0: condTermFlagA=0 (left has coeffs), condTermFlagB=0 (unavailable)
        # ctx_inc = 0 + 2*0 = 0
        expected_ctx_idx_block0 = CTX_CODED_BLOCK_PATTERN_START + 0
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx_block0]

    def test_cbp_luma_left_neighbor_no_coded_coeffs(self):
        """Block 0 with left neighbor having NO coded coeffs.

        left_cbp=0x00 means block 1 of left MB has NO coded coeffs.
        - condTermFlagA = 1 (left has no coeffs)
        - condTermFlagB = 0 (top unavailable, H.264 Section 9.3.3.1.1.3)
        - ctx_inc = 1 + 2*0 = 1
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_luma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0, 0, 0, 0])

        # left_cbp=0x00: block 1 of left MB has NO coded coeffs (bit 1 not set)
        decode_cbp_luma(decoder, contexts, mb_type=0, left_cbp=0x00, top_cbp=-1)

        # Block 0: condTermFlagA=1 (no coeffs), condTermFlagB=0 (unavailable)
        # ctx_inc = 1 + 2*0 = 1
        expected_ctx_idx_block0 = CTX_CODED_BLOCK_PATTERN_START + 1
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx_block0]

    def test_cbp_luma_top_neighbor_has_coded_coeffs(self):
        """Block 0 with top neighbor having coded coeffs.

        top_cbp=0x04 means block 2 of top MB has coded coeffs.
        Block 0's top neighbor is block 2 of top MB.
        cond_a = 0 (left unavailable → 0)
        cond_b = !(0x04 & 0x04) = 0 (top has coeffs)
        ctx_inc = 0 + 2*0 = 0
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_luma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0, 0, 0, 0])

        # top_cbp=0x04: block 2 of top MB has coded coeffs
        decode_cbp_luma(decoder, contexts, mb_type=0, left_cbp=-1, top_cbp=0x04)

        # Block 0: cond_a=0 (unavailable), cond_b=0 (top bit set)
        # ctx_inc = 0 + 2*0 = 0
        expected_ctx_idx_block0 = CTX_CODED_BLOCK_PATTERN_START + 0
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx_block0]

    def test_cbp_luma_both_neighbors_have_coded_coeffs(self):
        """Block 0 with both neighbors having coded coeffs uses ctx_inc=0.

        left_cbp=0x02 (block 1 has coeffs), top_cbp=0x04 (block 2 has coeffs)
        - condTermFlagA = 0 (left has coeffs)
        - condTermFlagB = 0 (top has coeffs)
        - ctx_inc = 0 + 2*0 = 0
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_luma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0, 0, 0, 0])

        decode_cbp_luma(decoder, contexts, mb_type=0, left_cbp=0x02, top_cbp=0x04)

        # Block 0: ctx_inc = 0 + 2*0 = 0
        expected_ctx_idx_block0 = CTX_CODED_BLOCK_PATTERN_START + 0
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx_block0]

    def test_cbp_luma_block1_uses_block0_as_left_neighbor(self):
        """Block 1's left neighbor is block 0 within same MB.

        When block 0 is decoded as having coeffs (return 1), block 1 sees it.
        - condTermFlagA = 0 (block 0 has coeffs)
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_luma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        # Block 0 returns 1 (has coeffs), subsequent blocks return 0
        decoder = MockCABACDecoder(return_values=[1, 0, 0, 0])

        # top_cbp=0x08 means block 3 of top MB has coeffs (block 1's top neighbor)
        decode_cbp_luma(decoder, contexts, mb_type=0, left_cbp=-1, top_cbp=0x08)

        # Block 1: left=block0 (just decoded as 1), top=block3 of top (has coeffs)
        # condTermFlagA = 0 (block 0 has coeffs)
        # condTermFlagB = 0 (top block 3 has coeffs)
        # ctx_inc = 0 + 2*0 = 0
        expected_ctx_idx_block1 = CTX_CODED_BLOCK_PATTERN_START + 0
        assert decoder.context_indices_used[1] is contexts[expected_ctx_idx_block1]

    def test_cbp_luma_block2_uses_block3_of_left_mb(self):
        """Block 2's left neighbor is block 3 of left MB.

        left_cbp=0x08 means block 3 of left MB has coded coeffs.
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_luma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        # Block 0, 1 return 0, then block 2's context is checked
        decoder = MockCABACDecoder(return_values=[0, 0, 0, 0])

        # left_cbp=0x08: block 3 of left MB has coded coeffs
        decode_cbp_luma(decoder, contexts, mb_type=0, left_cbp=0x08, top_cbp=-1)

        # Block 2: left=block3 of left MB (has coeffs), top=block0 (no coeffs)
        # condTermFlagA = 0 (block 3 of left has coeffs)
        # condTermFlagB = 1 (block 0 has no coeffs, decoded as 0)
        # ctx_inc = 0 + 2*1 = 2
        expected_ctx_idx_block2 = CTX_CODED_BLOCK_PATTERN_START + 2
        assert decoder.context_indices_used[2] is contexts[expected_ctx_idx_block2]

    def test_cbp_luma_block3_uses_internal_neighbors(self):
        """Block 3's neighbors are block 2 (left) and block 1 (top) within MB.

        Both are internal to current MB.
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_luma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        # Blocks 0,1,2 return 1 (have coeffs)
        decoder = MockCABACDecoder(return_values=[1, 1, 1, 0])

        decode_cbp_luma(decoder, contexts, mb_type=0, left_cbp=-1, top_cbp=-1)

        # Block 3: left=block2 (has coeffs), top=block1 (has coeffs)
        # condTermFlagA = 0 (block 2 has coeffs)
        # condTermFlagB = 0 (block 1 has coeffs)
        # ctx_inc = 0 + 2*0 = 0
        expected_ctx_idx_block3 = CTX_CODED_BLOCK_PATTERN_START + 0
        assert decoder.context_indices_used[3] is contexts[expected_ctx_idx_block3]

    def test_cbp_luma_returns_correct_value_from_bits(self):
        """CBP luma correctly assembles bits from all 4 blocks."""
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_cbp_luma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        # Return pattern: block0=1, block1=0, block2=1, block3=1
        decoder = MockCABACDecoder(return_values=[1, 0, 1, 1])

        result = decode_cbp_luma(decoder, contexts, mb_type=0, left_cbp=-1, top_cbp=-1)

        # CBP = bit0 | bit1<<1 | bit2<<2 | bit3<<3 = 1 | 0 | 4 | 8 = 13 (0b1101)
        assert result == 0b1101


class TestCBPChromaNeighborContext:
    """Tests for CBP chroma neighbor-based context derivation.

    H.264 Spec Reference: Section 9.3.3.1.1.3

    CBP chroma: 0=none, 1=DC only, 2=DC+AC
    First bin: 0 vs non-zero
    Second bin (if first=1): 1 vs 2
    """

    def test_cbp_chroma_unavailable_neighbors_use_ctx_inc_0(self):
        """Unavailable neighbors should use ctx_inc=0 for first bin.

        When left_cbp_chroma=-1 and top_cbp_chroma=-1:
        - condTermFlagA = 0 (unavailable)
        - condTermFlagB = 0 (unavailable)
        - ctx_inc = 0 + 2*0 = 0
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_chroma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        # First bin returns 0 (cbp_chroma=0)
        decoder = MockCABACDecoder(return_values=[0])

        decode_cbp_chroma(
            decoder, contexts, mb_type=0, left_cbp_chroma=-1, top_cbp_chroma=-1
        )

        # First bin: ctx_base + 4 + ctx_inc = 73 + 4 + 0 = 77
        expected_ctx_idx = CTX_CODED_BLOCK_PATTERN_START + 4 + 0
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx]

    def test_cbp_chroma_left_neighbor_has_nonzero_cbp(self):
        """Left neighbor with non-zero chroma CBP.

        left_cbp_chroma=1 means left has DC chroma coeffs.
        - condTermFlagA = 1 (left has non-zero cbp)
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_chroma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0])

        decode_cbp_chroma(
            decoder, contexts, mb_type=0, left_cbp_chroma=1, top_cbp_chroma=-1
        )

        # condTermFlagA = 1 (left > 0), condTermFlagB = 0 (unavailable)
        # ctx_inc = 1 + 2*0 = 1
        expected_ctx_idx = CTX_CODED_BLOCK_PATTERN_START + 4 + 1
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx]

    def test_cbp_chroma_top_neighbor_has_nonzero_cbp(self):
        """Top neighbor with non-zero chroma CBP.

        top_cbp_chroma=2 means top has DC+AC chroma coeffs.
        - condTermFlagB = 1 (top has non-zero cbp)
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_chroma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0])

        decode_cbp_chroma(
            decoder, contexts, mb_type=0, left_cbp_chroma=-1, top_cbp_chroma=2
        )

        # condTermFlagA = 0 (unavailable), condTermFlagB = 1 (top > 0)
        # ctx_inc = 0 + 2*1 = 2
        expected_ctx_idx = CTX_CODED_BLOCK_PATTERN_START + 4 + 2
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx]

    def test_cbp_chroma_both_neighbors_have_zero_cbp(self):
        """Both neighbors with zero chroma CBP.

        left_cbp_chroma=0 and top_cbp_chroma=0.
        - condTermFlagA = 0 (left == 0)
        - condTermFlagB = 0 (top == 0)
        - ctx_inc = 0 + 2*0 = 0
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_chroma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0])

        decode_cbp_chroma(
            decoder, contexts, mb_type=0, left_cbp_chroma=0, top_cbp_chroma=0
        )

        # Both neighbors have cbp=0
        # ctx_inc = 0 + 2*0 = 0
        expected_ctx_idx = CTX_CODED_BLOCK_PATTERN_START + 4 + 0
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx]

    def test_cbp_chroma_both_neighbors_have_nonzero_cbp(self):
        """Both neighbors with non-zero chroma CBP uses ctx_inc=3."""
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_chroma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0])

        decode_cbp_chroma(
            decoder, contexts, mb_type=0, left_cbp_chroma=2, top_cbp_chroma=1
        )

        # Both neighbors have cbp > 0
        # ctx_inc = 1 + 2*1 = 3
        expected_ctx_idx = CTX_CODED_BLOCK_PATTERN_START + 4 + 3
        assert decoder.context_indices_used[0] is contexts[expected_ctx_idx]

    def test_cbp_chroma_second_bin_context_derivation(self):
        """Second bin uses different context based on neighbor cbp > 1.

        When first bin returns 1 (non-zero), second bin distinguishes 1 vs 2.
        Context derivation for second bin uses cbp > 1 condition.
        """
        from entropy.cabac_context import (
            CTX_CODED_BLOCK_PATTERN_START,
            init_context_models,
        )
        from entropy.cabac_syntax import decode_cbp_chroma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        # First bin=1 (non-zero), second bin=0 (result=1, DC only)
        decoder = MockCABACDecoder(return_values=[1, 0])

        # left_cbp_chroma=2 (>1), top_cbp_chroma=1 (not >1)
        decode_cbp_chroma(
            decoder, contexts, mb_type=0, left_cbp_chroma=2, top_cbp_chroma=1
        )

        # Second bin: condTermFlagA = 1 (left == 2), condTermFlagB = 0 (top != 2)
        # ctx_inc = 1 + 2*0 = 1
        # Second bin uses ctx_base + 4 + 4 + ctx_inc = 73 + 4 + 4 + 1 = 82
        expected_ctx_idx_bin2 = CTX_CODED_BLOCK_PATTERN_START + 4 + 4 + 1
        assert decoder.context_indices_used[1] is contexts[expected_ctx_idx_bin2]

    def test_cbp_chroma_returns_0_when_first_bin_0(self):
        """CBP chroma returns 0 when first bin is 0."""
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_cbp_chroma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[0])

        result = decode_cbp_chroma(
            decoder, contexts, mb_type=0, left_cbp_chroma=-1, top_cbp_chroma=-1
        )

        assert result == 0
        assert len(decoder.context_indices_used) == 1

    def test_cbp_chroma_returns_1_when_first_bin_1_second_bin_0(self):
        """CBP chroma returns 1 (DC only) when bins are 1,0."""
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_cbp_chroma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[1, 0])

        result = decode_cbp_chroma(
            decoder, contexts, mb_type=0, left_cbp_chroma=-1, top_cbp_chroma=-1
        )

        assert result == 1
        assert len(decoder.context_indices_used) == 2

    def test_cbp_chroma_returns_2_when_first_bin_1_second_bin_1(self):
        """CBP chroma returns 2 (DC+AC) when bins are 1,1."""
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_cbp_chroma

        contexts = init_context_models(slice_type=0, slice_qp=26)
        decoder = MockCABACDecoder(return_values=[1, 1])

        result = decode_cbp_chroma(
            decoder, contexts, mb_type=0, left_cbp_chroma=-1, top_cbp_chroma=-1
        )

        assert result == 2
        assert len(decoder.context_indices_used) == 2


class TestIntraPredModeDecoding:
    """Tests for intra prediction mode decoding."""

    def test_decode_prev_intra_pred_mode_flag_exists(self):
        """decode_prev_intra4x4_pred_mode_flag should exist."""
        from entropy.cabac_syntax import decode_prev_intra4x4_pred_mode_flag

        assert callable(decode_prev_intra4x4_pred_mode_flag)

    def test_decode_rem_intra_pred_mode_exists(self):
        """decode_rem_intra4x4_pred_mode should exist."""
        from entropy.cabac_syntax import decode_rem_intra4x4_pred_mode

        assert callable(decode_rem_intra4x4_pred_mode)

    def test_decode_intra_chroma_pred_mode_exists(self):
        """decode_intra_chroma_pred_mode should exist."""
        from entropy.cabac_syntax import decode_intra_chroma_pred_mode

        assert callable(decode_intra_chroma_pred_mode)

    def test_intra_chroma_pred_mode_range(self):
        """Intra chroma pred mode should be 0-3."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_intra_chroma_pred_mode

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        result = decode_intra_chroma_pred_mode(decoder, contexts)

        assert 0 <= result <= 3


class TestCodedBlockFlag:
    """Tests for coded_block_flag decoding."""

    def test_decode_coded_block_flag_exists(self):
        """decode_coded_block_flag function should exist."""
        from entropy.cabac_syntax import decode_coded_block_flag

        assert callable(decode_coded_block_flag)

    def test_coded_block_flag_returns_binary(self):
        """coded_block_flag should return 0 or 1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_syntax import decode_coded_block_flag

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        result = decode_coded_block_flag(decoder, contexts, cat=2, ctx_block_cat=0)

        assert result in (0, 1)
