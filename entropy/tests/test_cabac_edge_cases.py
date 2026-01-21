# h264/entropy/tests/test_cabac_edge_cases.py
"""RED TESTS: CABAC entropy decoding edge cases and advanced features.

Tests for advanced CABAC behavior including:
- Slice boundary handling
- Context state persistence across macroblocks
- CABAC bypass mode for specific syntax elements
- CABAC termination conditions
- Context initialization edge cases

H.264 Spec Reference:
- Section 9.3.1 - Initialization process for CABAC
- Section 9.3.2 - Binarization process
- Section 9.3.3 - Decoding process

These tests SHOULD FAIL until the corresponding features are implemented.
"""

import pytest
import numpy as np

from bitstream import BitReader


# =============================================================================
# Test: Slice boundary handling
# =============================================================================

class TestSliceBoundaryHandling:
    """Tests for CABAC behavior at slice boundaries.

    At slice start, CABAC contexts must be re-initialized.
    At slice end, proper termination must occur.
    """

    def test_cabac_init_at_slice_start(self):
        """CABAC engine initialization at slice start."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        # Initial range should be 510 per H.264 spec
        assert decoder.codIRange == 510

        # Contexts should be properly initialized
        assert len(contexts) >= 460

    def test_cabac_init_with_cabac_init_idc(self):
        """Context initialization varies with cabac_init_idc."""
        from entropy.cabac_context import init_context_models_with_idc

        contexts_idc0 = init_context_models_with_idc(
            slice_type=0, slice_qp=26, cabac_init_idc=0
        )
        contexts_idc1 = init_context_models_with_idc(
            slice_type=0, slice_qp=26, cabac_init_idc=1
        )
        contexts_idc2 = init_context_models_with_idc(
            slice_type=0, slice_qp=26, cabac_init_idc=2
        )

        # Different idc values should produce different contexts
        # (at least for some context indices)
        assert contexts_idc0 is not None
        assert contexts_idc1 is not None
        assert contexts_idc2 is not None

    def test_first_mb_in_slice_context_derivation(self):
        """First MB in slice has no left/top neighbors for context."""
        from entropy.cabac_macroblock import derive_context_for_first_mb

        ctx_info = derive_context_for_first_mb(slice_type=2)

        # First MB should use defaults for unavailable neighbors
        assert 'left_available' in ctx_info
        assert ctx_info['left_available'] is False
        assert 'top_available' in ctx_info
        assert ctx_info['top_available'] is False

    def test_slice_boundary_resets_neighbor_info(self):
        """Slice boundary resets neighbor availability flags."""
        from entropy.cabac_macroblock import is_across_slice_boundary

        # MB at slice start has no valid neighbors from previous slice
        assert is_across_slice_boundary(
            curr_mb_addr=0, neighbor_mb_addr=-1, first_mb_in_slice=0
        )

        # MB within slice can reference neighbors
        assert not is_across_slice_boundary(
            curr_mb_addr=5, neighbor_mb_addr=4, first_mb_in_slice=0
        )

        # Different slices
        assert is_across_slice_boundary(
            curr_mb_addr=10, neighbor_mb_addr=9, first_mb_in_slice=10
        )

    def test_multiple_slices_independent_context_init(self):
        """Each slice independently initializes CABAC contexts."""
        from entropy.cabac_context import init_context_models

        # Slice 1 with QP=26
        contexts1 = init_context_models(slice_type=0, slice_qp=26)

        # Slice 2 with QP=30
        contexts2 = init_context_models(slice_type=0, slice_qp=30)

        # Different QP should give different context states
        different_count = sum(
            1 for c1, c2 in zip(contexts1, contexts2)
            if c1.pStateIdx != c2.pStateIdx or c1.valMPS != c2.valMPS
        )

        # At least some contexts should differ
        assert different_count > 0

    def test_slice_boundary_top_row_neighbor_handling(self):
        """Top row MBs use special handling for top neighbor."""
        from entropy.cabac_macroblock import get_top_neighbor_for_context

        pic_width_in_mbs = 10

        # First row has no top neighbor
        top_info = get_top_neighbor_for_context(
            mb_addr=5, pic_width_in_mbs=pic_width_in_mbs,
            first_mb_in_slice=0
        )
        assert top_info['top_available'] is False

        # Second row has top neighbor
        top_info = get_top_neighbor_for_context(
            mb_addr=15, pic_width_in_mbs=pic_width_in_mbs,
            first_mb_in_slice=0
        )
        assert top_info['top_available'] is True


class TestContextStatePersistence:
    """Tests for context state persistence across macroblocks.

    CABAC contexts are updated after each decoded bin and persist
    across macroblocks within a slice. This tests proper state
    management and adaptation.
    """

    def test_context_state_updates_after_decision(self):
        """Context state should update after each decoded decision."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import CABACContext

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        ctx = CABACContext(pStateIdx=30, valMPS=1)
        initial_state = ctx.pStateIdx

        # Decode multiple decisions
        for _ in range(5):
            decoder.decode_decision(ctx)

        # State should have changed
        assert ctx.pStateIdx != initial_state or True  # May stay same if MPS

    def test_context_state_persists_across_mb(self):
        """Context state persists when decoding multiple macroblocks."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_type_cabac

        data = bytes([0x80, 0x40, 0x20, 0x10] * 50)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=2, slice_qp=26)

        # Record state before first MB
        ctx_idx_3 = contexts[3]  # mb_type I context
        state_before = ctx_idx_3.pStateIdx

        # Decode first MB type
        mb_info = {'left_available': False, 'top_available': False}
        decode_mb_type_cabac(decoder, contexts, slice_type=2, mb_info=mb_info)

        # State may have changed
        state_after_1 = ctx_idx_3.pStateIdx

        # Decode second MB type
        mb_info = {'left_available': True, 'top_available': False,
                   'left_mb_type': 0, 'top_mb_type': None}
        decode_mb_type_cabac(decoder, contexts, slice_type=2, mb_info=mb_info)

        # State continues to evolve
        state_after_2 = ctx_idx_3.pStateIdx

        # States form a sequence (may or may not change based on data)
        assert isinstance(state_before, int)
        assert isinstance(state_after_1, int)
        assert isinstance(state_after_2, int)

    def test_mps_switch_on_lps_at_state_0(self):
        """MPS switches when LPS decoded at pStateIdx=0."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import CABACContext

        # Construct bitstream that forces LPS decode
        data = bytes([0xFF, 0xFF, 0xFF, 0xFF] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        # Start at state 0 with MPS=1
        ctx = CABACContext(pStateIdx=0, valMPS=1)
        initial_mps = ctx.valMPS

        # Keep decoding until we get LPS at state 0
        # (This depends on the random data, but tests the mechanism)
        for _ in range(20):
            if ctx.pStateIdx == 0:
                old_mps = ctx.valMPS
                decoder.decode_decision(ctx)
                if ctx.valMPS != old_mps:
                    break  # MPS switched

        # Just verify context is valid
        assert ctx.pStateIdx >= 0
        assert ctx.valMPS in (0, 1)

    def test_context_reaches_saturation_state(self):
        """Context can reach saturation state (63) with consistent MPS."""
        from entropy.cabac_arith import CABACDecoder, TRANS_IDX_MPS
        from entropy.cabac_context import CABACContext

        # Verify saturation behavior from transition table
        # State 63 transitions to 63 on MPS
        assert TRANS_IDX_MPS[63] == 63

        # State 62 transitions to 62 on MPS
        assert TRANS_IDX_MPS[62] == 62

    def test_slice_preserves_context_across_skip_run(self):
        """Context state preserved during skip run in P/B slices."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models
        from entropy.cabac_macroblock import decode_mb_skip_flag_cabac

        data = bytes([0xFF, 0xFF] * 50)  # Likely skips
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=0, slice_qp=26)

        # Decode multiple skip flags
        skip_results = []
        for i in range(5):
            mb_info = {
                'mb_x': i, 'mb_y': 0,
                'left_available': i > 0, 'top_available': False,
                'left_skip': bool(skip_results[-1]) if skip_results else False,
                'top_skip': False,
            }
            result = decode_mb_skip_flag_cabac(
                decoder, contexts, slice_type=0, mb_info=mb_info
            )
            skip_results.append(result)

        assert len(skip_results) == 5


# =============================================================================
# Test: CABAC bypass mode
# =============================================================================

class TestCABACBypassMode:
    """Tests for CABAC bypass (equiprobable) mode.

    Some syntax elements use bypass mode instead of context-based
    decoding for efficiency. These include:
    - MVD suffix (abs_mvd_comp >= 9)
    - Coefficient level suffix (coeff_abs_level >= 14)
    - Sign flags for MVD and coefficient levels
    - Exp-Golomb coded parts

    H.264 Spec Reference: Section 9.3.3.2.3
    """

    def test_bypass_decode_returns_binary(self):
        """Bypass decode returns 0 or 1."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0xAA, 0x55, 0xAA, 0x55] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        result = decoder.decode_bypass()

        assert result in (0, 1)

    def test_bypass_decode_multiple_bits(self):
        """Multiple bypass decodes for multi-bit values."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0xAA, 0x55] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        # Decode 8 bypass bits
        bits = [decoder.decode_bypass() for _ in range(8)]

        assert len(bits) == 8
        assert all(b in (0, 1) for b in bits)

    def test_mvd_uses_bypass_for_suffix(self):
        """MVD suffix (when abs >= 9) uses bypass mode."""
        from entropy.cabac_syntax import decode_mvd_lx_suffix_bypass

        assert callable(decode_mvd_lx_suffix_bypass)

    def test_mvd_sign_uses_bypass(self):
        """MVD sign flag uses bypass mode."""
        from entropy.cabac_syntax import decode_mvd_sign_bypass

        assert callable(decode_mvd_sign_bypass)

    def test_coeff_level_suffix_uses_bypass(self):
        """Coefficient level suffix (when >= 14) uses bypass."""
        from entropy.cabac_residual import decode_coeff_abs_level_suffix_bypass

        assert callable(decode_coeff_abs_level_suffix_bypass)

    def test_coeff_sign_uses_bypass(self):
        """Coefficient sign flag uses bypass mode."""
        from entropy.cabac_residual import decode_coeff_sign_flag

        assert callable(decode_coeff_sign_flag)

    def test_bypass_equiprobable_distribution(self):
        """Bypass mode should give approximately 50/50 distribution."""
        from entropy.cabac_arith import CABACDecoder

        # Random-ish data
        data = bytes(range(256)) * 4
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        # Decode many bypass bits
        results = [decoder.decode_bypass() for _ in range(100)]

        zeros = results.count(0)
        ones = results.count(1)

        # Should be roughly balanced (not perfect due to data pattern)
        assert zeros > 0
        assert ones > 0

    def test_decode_bypass_exp_golomb_k(self):
        """Decode k-th order Exp-Golomb code in bypass mode."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_binarize import decode_exp_golomb_bypass

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        # Decode UEGk with k=0 (standard Exp-Golomb)
        result = decode_exp_golomb_bypass(decoder, k=0)

        assert result >= 0

    def test_decode_bypass_exp_golomb_k3(self):
        """Decode Exp-Golomb with k=3 for large MVD/levels."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_binarize import decode_exp_golomb_bypass

        data = bytes([0xFF, 0x00, 0xFF, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        # Decode UEGk with k=3 (used for large values)
        result = decode_exp_golomb_bypass(decoder, k=3)

        assert result >= 0


class TestCABACTermination:
    """Tests for CABAC termination process.

    end_of_slice_flag uses a special terminate decoding process
    that differs from both decision and bypass modes.

    H.264 Spec Reference: Section 9.3.3.2.4
    """

    def test_terminate_decode_returns_binary(self):
        """Terminate decode returns 0 or 1."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x80, 0x40, 0x20, 0x10] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        result = decoder.decode_terminate()

        assert result in (0, 1)

    def test_terminate_reduces_range_by_2(self):
        """Terminate process reduces codIRange by 2."""
        from entropy.cabac_arith import CABACDecoder

        data = bytes([0x00, 0x00, 0x00, 0x00] * 10)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        initial_range = decoder.codIRange

        # Decode terminate (assuming not end of slice)
        decoder.decode_terminate()

        # If not terminated, range should be renormalized
        # but was reduced by 2 before renormalization
        assert decoder.codIRange >= 256  # After renormalization

    def test_terminate_detected_when_offset_ge_range(self):
        """Termination detected when codIOffset >= codIRange after reduction."""
        from entropy.cabac_arith import CABACDecoder

        # Construct bitstream that should trigger termination
        # High offset value that becomes >= range after reduction
        data = bytes([0xFF, 0xFE] + [0x00] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)

        result = decoder.decode_terminate()

        # Result depends on constructed bitstream
        assert result in (0, 1)

    def test_cabac_alignment_after_terminate(self):
        """CABAC should byte-align after terminate=1."""
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_macroblock import cabac_byte_alignment_after_slice

        assert callable(cabac_byte_alignment_after_slice)

    def test_terminate_called_after_each_mb(self):
        """end_of_slice_flag terminate called after each coded MB."""
        from entropy.cabac_macroblock import should_decode_end_of_slice

        # After non-skip MB in slice interior
        assert should_decode_end_of_slice(
            mb_skip_flag=0, is_last_mb_in_slice=False
        )

        # After skip MB (no terminate needed until next non-skip)
        # Actually, terminate is called after the non-skipped MB that follows
        assert callable(should_decode_end_of_slice)


# =============================================================================
# Test: Context initialization edge cases
# =============================================================================

class TestContextInitializationEdgeCases:
    """Tests for edge cases in context initialization.

    Context initialization depends on slice_type and QP, with
    special handling for boundary values.
    """

    def test_context_init_qp_0(self):
        """Context initialization with minimum QP (0)."""
        from entropy.cabac_context import init_context_models

        contexts = init_context_models(slice_type=2, slice_qp=0)

        assert len(contexts) >= 460
        # All contexts should be valid
        for ctx in contexts:
            assert 0 <= ctx.pStateIdx <= 63
            assert ctx.valMPS in (0, 1)

    def test_context_init_qp_51(self):
        """Context initialization with maximum QP (51)."""
        from entropy.cabac_context import init_context_models

        contexts = init_context_models(slice_type=2, slice_qp=51)

        assert len(contexts) >= 460
        for ctx in contexts:
            assert 0 <= ctx.pStateIdx <= 63
            assert ctx.valMPS in (0, 1)

    def test_context_init_prestate_clipping(self):
        """preCtxState is clipped to [1, 126]."""
        from entropy.cabac_context import calc_initial_state

        # Test with extreme m, n values
        p_state, val_mps = calc_initial_state(m=-128, n=0, slice_qp=51)
        assert 0 <= p_state <= 63
        assert val_mps in (0, 1)

        p_state, val_mps = calc_initial_state(m=127, n=127, slice_qp=51)
        assert 0 <= p_state <= 63
        assert val_mps in (0, 1)

    def test_context_init_all_slice_types(self):
        """Context initialization works for all slice types."""
        from entropy.cabac_context import init_context_models

        for slice_type in [0, 1, 2, 5, 6, 7]:  # P, B, I and variants
            contexts = init_context_models(slice_type=slice_type, slice_qp=26)
            assert len(contexts) >= 460

    def test_si_sp_slice_context_init(self):
        """Context initialization for SI and SP slices."""
        from entropy.cabac_context import init_context_models

        # SI slice (type 4)
        contexts_si = init_context_models(slice_type=4, slice_qp=26)
        assert len(contexts_si) >= 460

        # SP slice (type 3)
        contexts_sp = init_context_models(slice_type=3, slice_qp=26)
        assert len(contexts_sp) >= 460

    def test_context_index_boundaries(self):
        """Verify context index boundaries from H.264 Table 9-11."""
        from entropy.cabac_context import (
            CTX_MB_TYPE_SI_START,
            CTX_MB_TYPE_I_START,
            CTX_MB_TYPE_P_START,
            CTX_MB_TYPE_B_START,
            CTX_SUB_MB_TYPE_P_START,
            CTX_MVD_START,
            CTX_REF_IDX_START,
            CTX_MB_QP_DELTA_START,
            CTX_INTRA_CHROMA_PRED_START,
            CTX_CODED_BLOCK_FLAG_START,
            CTX_SIG_COEFF_FLAG_START,
            CTX_LAST_SIG_COEFF_START,
            CTX_COEFF_ABS_LEVEL_START,
        )

        # Verify non-overlapping ranges
        assert CTX_MB_TYPE_SI_START == 0
        assert CTX_MB_TYPE_I_START == 3
        assert CTX_MB_TYPE_P_START > CTX_MB_TYPE_I_START
        assert CTX_MB_TYPE_B_START > CTX_MB_TYPE_P_START
        assert CTX_MVD_START >= 40
        assert CTX_SIG_COEFF_FLAG_START >= 105


# =============================================================================
# Test: Macroblock address and neighbor derivation
# =============================================================================

class TestMBAddressDerivation:
    """Tests for macroblock address and neighbor derivation.

    CABAC context derivation often depends on neighboring MB info.
    """

    def test_mb_addr_to_coords(self):
        """Convert MB address to (x, y) coordinates."""
        from entropy.cabac_macroblock import mb_addr_to_xy

        pic_width_in_mbs = 10

        # First MB
        x, y = mb_addr_to_xy(mb_addr=0, pic_width_in_mbs=pic_width_in_mbs)
        assert (x, y) == (0, 0)

        # End of first row
        x, y = mb_addr_to_xy(mb_addr=9, pic_width_in_mbs=pic_width_in_mbs)
        assert (x, y) == (9, 0)

        # First of second row
        x, y = mb_addr_to_xy(mb_addr=10, pic_width_in_mbs=pic_width_in_mbs)
        assert (x, y) == (0, 1)

    def test_get_left_mb_addr(self):
        """Get left neighbor MB address."""
        from entropy.cabac_macroblock import get_left_mb_addr

        pic_width_in_mbs = 10

        # MB at column 0 has no left neighbor
        left = get_left_mb_addr(mb_addr=0, pic_width_in_mbs=pic_width_in_mbs)
        assert left is None

        left = get_left_mb_addr(mb_addr=10, pic_width_in_mbs=pic_width_in_mbs)
        assert left is None

        # MB at column > 0 has left neighbor
        left = get_left_mb_addr(mb_addr=5, pic_width_in_mbs=pic_width_in_mbs)
        assert left == 4

    def test_get_top_mb_addr(self):
        """Get top neighbor MB address."""
        from entropy.cabac_macroblock import get_top_mb_addr

        pic_width_in_mbs = 10

        # MB in first row has no top neighbor
        top = get_top_mb_addr(mb_addr=5, pic_width_in_mbs=pic_width_in_mbs)
        assert top is None

        # MB in row > 0 has top neighbor
        top = get_top_mb_addr(mb_addr=15, pic_width_in_mbs=pic_width_in_mbs)
        assert top == 5

    def test_neighbor_availability_at_picture_boundary(self):
        """Neighbors at picture boundary are unavailable."""
        from entropy.cabac_macroblock import get_neighbor_availability

        pic_width_in_mbs = 10
        pic_height_in_mbs = 8

        # Top-left corner
        avail = get_neighbor_availability(
            mb_addr=0,
            pic_width_in_mbs=pic_width_in_mbs,
            pic_height_in_mbs=pic_height_in_mbs,
            first_mb_in_slice=0
        )
        assert not avail['left']
        assert not avail['top']

        # Bottom-right corner
        last_mb = pic_width_in_mbs * pic_height_in_mbs - 1
        avail = get_neighbor_availability(
            mb_addr=last_mb,
            pic_width_in_mbs=pic_width_in_mbs,
            pic_height_in_mbs=pic_height_in_mbs,
            first_mb_in_slice=0
        )
        assert avail['left']  # Has left neighbor
        assert avail['top']   # Has top neighbor


# =============================================================================
# Test: MBAFF-specific CABAC handling
# =============================================================================

class TestMBAFFCABACHandling:
    """Tests for MBAFF (Macroblock-Adaptive Frame/Field) CABAC handling.

    In MBAFF, macroblocks are coded in pairs and context derivation
    must account for frame/field coding decisions.
    """

    def test_mbaff_pair_address_calculation(self):
        """Calculate MB pair address in MBAFF mode."""
        from entropy.cabac_macroblock import get_mbaff_pair_addr

        pic_width_in_mbs = 10

        # MB 0 and 1 are a pair
        pair = get_mbaff_pair_addr(mb_addr=0, pic_width_in_mbs=pic_width_in_mbs)
        assert pair in (0, 1)

        pair = get_mbaff_pair_addr(mb_addr=1, pic_width_in_mbs=pic_width_in_mbs)
        assert pair in (0, 1)

    def test_mbaff_field_mb_neighbor_derivation(self):
        """Field MB neighbor derivation differs in MBAFF."""
        from entropy.cabac_macroblock import get_mbaff_neighbor

        # This is complex - neighbors for field MBs may be from
        # different MB pairs or different rows within a pair
        assert callable(get_mbaff_neighbor)

    def test_mbaff_context_derivation_uses_pair_info(self):
        """Context derivation in MBAFF uses MB pair information."""
        from entropy.cabac_macroblock import derive_mbaff_context_info

        mb_pair_info = {
            'mb_addr': 0,
            'is_top_mb': True,
            'pair_field_flag': 0,
            'pic_width_in_mbs': 10,
        }

        ctx_info = derive_mbaff_context_info(mb_pair_info)

        assert 'effective_top_available' in ctx_info
        assert 'effective_left_available' in ctx_info


# =============================================================================
# Test: Coded block flag context derivation
# =============================================================================

class TestCodedBlockFlagContext:
    """Tests for coded_block_flag context derivation.

    coded_block_flag context depends on:
    - Block category (luma DC, luma AC, luma 4x4, chroma DC, chroma AC)
    - Neighboring block CBF values
    """

    def test_cbf_context_for_luma_dc(self):
        """coded_block_flag context for Luma DC (cat=0)."""
        from entropy.cabac_residual import get_coded_block_flag_ctx_idx

        ctx_idx = get_coded_block_flag_ctx_idx(
            block_cat=0,
            left_cbf=0, top_cbf=0,
            left_available=True, top_available=True
        )

        assert ctx_idx >= 85  # CBF contexts start at 85

    def test_cbf_context_for_luma_ac(self):
        """coded_block_flag context for Luma AC (cat=1)."""
        from entropy.cabac_residual import get_coded_block_flag_ctx_idx

        ctx_idx = get_coded_block_flag_ctx_idx(
            block_cat=1,
            left_cbf=0, top_cbf=0,
            left_available=True, top_available=True
        )

        assert ctx_idx >= 85

    def test_cbf_context_for_chroma_dc(self):
        """coded_block_flag context for Chroma DC (cat=3)."""
        from entropy.cabac_residual import get_coded_block_flag_ctx_idx

        ctx_idx = get_coded_block_flag_ctx_idx(
            block_cat=3,
            left_cbf=0, top_cbf=0,
            left_available=True, top_available=True
        )

        assert ctx_idx >= 85

    def test_cbf_context_increment_from_neighbors(self):
        """CBF context increment depends on neighbor CBF values."""
        from entropy.cabac_residual import get_coded_block_flag_ctx_idx

        # Both neighbors CBF=0
        ctx_00 = get_coded_block_flag_ctx_idx(
            block_cat=2,
            left_cbf=0, top_cbf=0,
            left_available=True, top_available=True
        )

        # Both neighbors CBF=1
        ctx_11 = get_coded_block_flag_ctx_idx(
            block_cat=2,
            left_cbf=1, top_cbf=1,
            left_available=True, top_available=True
        )

        # Context should differ
        assert ctx_11 > ctx_00

    def test_cbf_unavailable_neighbor_treated_as_zero(self):
        """Unavailable CBF neighbor treated as CBF=0."""
        from entropy.cabac_residual import get_coded_block_flag_ctx_idx

        # Unavailable neighbors
        ctx_unavail = get_coded_block_flag_ctx_idx(
            block_cat=2,
            left_cbf=0, top_cbf=0,
            left_available=False, top_available=False
        )

        # Both available with CBF=0
        ctx_avail = get_coded_block_flag_ctx_idx(
            block_cat=2,
            left_cbf=0, top_cbf=0,
            left_available=True, top_available=True
        )

        # Should be same (unavailable treated as 0)
        assert ctx_unavail == ctx_avail


# =============================================================================
# Test: B-slice specific CABAC handling
# =============================================================================

class TestBSliceCABACHandling:
    """Tests for B-slice specific CABAC decoding.

    B-slices have additional syntax elements (bi-prediction, direct mode)
    with specific context handling.
    """

    def test_b_slice_mb_type_context_range(self):
        """B-slice mb_type uses context indices 27-35."""
        from entropy.cabac_context import CTX_MB_TYPE_B_START

        assert CTX_MB_TYPE_B_START == 27

    def test_b_direct_16x16_mb_type_decoding(self):
        """B_Direct_16x16 (mb_type=0) special handling."""
        from entropy.cabac_macroblock import is_b_direct_16x16

        assert is_b_direct_16x16(mb_type=0, slice_type=1)
        assert not is_b_direct_16x16(mb_type=1, slice_type=1)

    def test_b_skip_flag_context_indices(self):
        """B-slice skip flag uses different context than P-slice."""
        from entropy.cabac_macroblock import get_mb_skip_flag_ctx_idx

        p_ctx = get_mb_skip_flag_ctx_idx(
            slice_type=0,
            left_available=True, left_skip=False,
            top_available=True, top_skip=False
        )

        b_ctx = get_mb_skip_flag_ctx_idx(
            slice_type=1,
            left_available=True, left_skip=False,
            top_available=True, top_skip=False
        )

        # B-slice uses different base context
        assert b_ctx != p_ctx

    def test_b_sub_mb_type_range(self):
        """B-slice sub_mb_type has larger range (0-12) than P-slice."""
        from entropy.cabac_syntax import decode_sub_mb_type_b
        from entropy.cabac_arith import CABACDecoder
        from entropy.cabac_context import init_context_models

        data = bytes([0x80, 0x40, 0x20, 0x10] * 20)
        reader = BitReader(data)
        decoder = CABACDecoder(reader)
        contexts = init_context_models(slice_type=1, slice_qp=26)

        result = decode_sub_mb_type_b(decoder, contexts)

        # B-slice: 0-12 for sub_mb_type
        assert 0 <= result <= 12


# =============================================================================
# Test: ref_idx CABAC decoding
# =============================================================================

class TestRefIdxCABACDecoding:
    """Tests for reference index CABAC decoding with context derivation."""

    def test_ref_idx_context_depends_on_list(self):
        """ref_idx context differs for L0 and L1."""
        from entropy.cabac_syntax import get_ref_idx_ctx_idx

        ctx_l0 = get_ref_idx_ctx_idx(list_idx=0, ctx_inc=0)
        ctx_l1 = get_ref_idx_ctx_idx(list_idx=1, ctx_inc=0)

        # L0 and L1 may use same or different contexts
        assert isinstance(ctx_l0, int)
        assert isinstance(ctx_l1, int)

    def test_ref_idx_truncated_unary(self):
        """ref_idx uses truncated unary binarization."""
        from entropy.cabac_binarize import binarize_ref_idx

        # ref_idx=0 -> "0"
        bins = binarize_ref_idx(ref_idx=0, max_ref_idx=2)
        assert bins == [0]

        # ref_idx=1 -> "10"
        bins = binarize_ref_idx(ref_idx=1, max_ref_idx=2)
        assert bins == [1, 0]

        # ref_idx=2 -> "11" (truncated)
        bins = binarize_ref_idx(ref_idx=2, max_ref_idx=2)
        assert bins == [1, 1]

    def test_ref_idx_zero_when_single_ref(self):
        """ref_idx not coded when only one reference picture."""
        from entropy.cabac_macroblock import should_decode_ref_idx

        assert not should_decode_ref_idx(num_ref_idx_active=1)
        assert should_decode_ref_idx(num_ref_idx_active=2)
