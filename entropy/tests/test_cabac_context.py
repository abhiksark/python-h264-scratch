# h264/entropy/tests/test_cabac_context.py
"""RED TESTS: CABAC context model initialization.

CABAC uses 460 context models, each with:
- pStateIdx: Probability state (0-63)
- valMPS: Most probable symbol (0 or 1)

Context initialization depends on slice type and QP.

H.264 Spec Reference: Section 9.3.1.1 - Initialization process for context variables

These tests SHOULD FAIL until CABAC context initialization is implemented.
"""

import pytest


class TestCABACContextModel:
    """Tests for CABACContext data structure."""

    def test_cabac_context_exists(self):
        """CABACContext class should exist."""
        from entropy.cabac_context import CABACContext

        assert CABACContext is not None

    def test_cabac_context_has_pstate(self):
        """CABACContext should have pStateIdx attribute."""
        from entropy.cabac_context import CABACContext

        ctx = CABACContext(pStateIdx=32, valMPS=0)
        assert hasattr(ctx, 'pStateIdx')
        assert ctx.pStateIdx == 32

    def test_cabac_context_has_valmps(self):
        """CABACContext should have valMPS attribute."""
        from entropy.cabac_context import CABACContext

        ctx = CABACContext(pStateIdx=32, valMPS=1)
        assert hasattr(ctx, 'valMPS')
        assert ctx.valMPS == 1

    def test_cabac_context_pstate_range(self):
        """pStateIdx should be in range [0, 63]."""
        from entropy.cabac_context import CABACContext

        ctx = CABACContext(pStateIdx=0, valMPS=0)
        assert 0 <= ctx.pStateIdx <= 63

        ctx = CABACContext(pStateIdx=63, valMPS=1)
        assert 0 <= ctx.pStateIdx <= 63

    def test_cabac_context_valmps_binary(self):
        """valMPS should be 0 or 1."""
        from entropy.cabac_context import CABACContext

        ctx0 = CABACContext(pStateIdx=32, valMPS=0)
        ctx1 = CABACContext(pStateIdx=32, valMPS=1)

        assert ctx0.valMPS in (0, 1)
        assert ctx1.valMPS in (0, 1)


class TestContextInitialization:
    """Tests for context model initialization."""

    def test_init_context_models_exists(self):
        """init_context_models function should exist."""
        from entropy.cabac_context import init_context_models

        assert callable(init_context_models)

    def test_init_returns_list_of_contexts(self):
        """init_context_models should return list of CABACContext."""
        from entropy.cabac_context import init_context_models, CABACContext

        contexts = init_context_models(slice_type=2, slice_qp=26)

        assert isinstance(contexts, list)
        assert len(contexts) > 0
        assert isinstance(contexts[0], CABACContext)

    def test_init_returns_460_contexts(self):
        """CABAC uses 460 context models (ctxIdx 0-459)."""
        from entropy.cabac_context import init_context_models

        contexts = init_context_models(slice_type=2, slice_qp=26)

        assert len(contexts) == 460

    def test_init_i_slice_contexts(self):
        """Initialize contexts for I-slice (slice_type=2)."""
        from entropy.cabac_context import init_context_models

        contexts = init_context_models(slice_type=2, slice_qp=26)

        # Should initialize without error
        assert len(contexts) == 460

    def test_init_p_slice_contexts(self):
        """Initialize contexts for P-slice (slice_type=0)."""
        from entropy.cabac_context import init_context_models

        contexts = init_context_models(slice_type=0, slice_qp=26)

        assert len(contexts) == 460

    def test_init_b_slice_contexts(self):
        """Initialize contexts for B-slice (slice_type=1)."""
        from entropy.cabac_context import init_context_models

        contexts = init_context_models(slice_type=1, slice_qp=26)

        assert len(contexts) == 460


class TestContextInitFormula:
    """Tests for context initialization formula (9-5 and 9-6)."""

    def test_calc_initial_state_exists(self):
        """calc_initial_state function should exist."""
        from entropy.cabac_context import calc_initial_state

        assert callable(calc_initial_state)

    def test_initial_state_from_m_n(self):
        """Initial state from m, n parameters.

        preCtxState = Clip3(1, 126, ((m * SliceQP) >> 4) + n)
        """
        from entropy.cabac_context import calc_initial_state

        # For m=0, n=64, QP=26: preCtxState = ((0*26)>>4) + 64 = 64
        pState, valMPS = calc_initial_state(m=0, n=64, slice_qp=26)

        # preCtxState=64 -> valMPS=1, pStateIdx=64-64=0
        assert valMPS == 1
        assert pState == 0

    def test_initial_state_mps_derivation(self):
        """valMPS = (preCtxState >= 64) ? 1 : 0."""
        from entropy.cabac_context import calc_initial_state

        # preCtxState < 64 -> valMPS=0
        pState, valMPS = calc_initial_state(m=0, n=32, slice_qp=26)
        assert valMPS == 0

        # preCtxState >= 64 -> valMPS=1
        pState, valMPS = calc_initial_state(m=0, n=80, slice_qp=26)
        assert valMPS == 1

    def test_initial_state_pstateIdx_derivation(self):
        """pStateIdx derivation based on valMPS.

        If valMPS=1: pStateIdx = preCtxState - 64
        If valMPS=0: pStateIdx = 63 - preCtxState
        """
        from entropy.cabac_context import calc_initial_state

        # preCtxState=80 -> valMPS=1, pStateIdx = 80-64 = 16
        pState, valMPS = calc_initial_state(m=0, n=80, slice_qp=26)
        assert valMPS == 1
        assert pState == 16

        # preCtxState=32 -> valMPS=0, pStateIdx = 63-32 = 31
        pState, valMPS = calc_initial_state(m=0, n=32, slice_qp=26)
        assert valMPS == 0
        assert pState == 31

    def test_initial_state_clipping(self):
        """preCtxState is clipped to [1, 126]."""
        from entropy.cabac_context import calc_initial_state

        # Very negative result -> clips to 1
        pState, valMPS = calc_initial_state(m=-100, n=-100, slice_qp=51)
        assert valMPS == 0  # preCtxState=1 < 64
        assert pState == 62  # 63 - 1 = 62

        # Very positive result -> clips to 126
        pState, valMPS = calc_initial_state(m=100, n=200, slice_qp=51)
        assert valMPS == 1  # preCtxState=126 >= 64
        assert pState == 62  # 126 - 64 = 62


class TestContextCategories:
    """Tests for context index ranges."""

    def test_mb_type_context_range(self):
        """mb_type contexts should be in proper range."""
        from entropy.cabac_context import (
            CTX_MB_TYPE_SI_START,
            CTX_MB_TYPE_I_START,
            CTX_MB_TYPE_P_START,
            CTX_MB_TYPE_B_START,
        )

        # Verify ranges exist and are distinct
        assert CTX_MB_TYPE_SI_START == 0
        assert CTX_MB_TYPE_I_START == 3
        assert CTX_MB_TYPE_P_START == 14
        assert CTX_MB_TYPE_B_START == 27

    def test_mvd_context_range(self):
        """mvd contexts start at index 40."""
        from entropy.cabac_context import CTX_MVD_START

        assert CTX_MVD_START == 40

    def test_ref_idx_context_range(self):
        """ref_idx contexts start at index 54."""
        from entropy.cabac_context import CTX_REF_IDX_START

        assert CTX_REF_IDX_START == 54

    def test_coded_block_flag_context_range(self):
        """coded_block_flag contexts start at index 85."""
        from entropy.cabac_context import CTX_CODED_BLOCK_FLAG_START

        assert CTX_CODED_BLOCK_FLAG_START == 85

    def test_significant_coeff_context_range(self):
        """significant_coeff_flag contexts start at index 105."""
        from entropy.cabac_context import CTX_SIG_COEFF_FLAG_START

        assert CTX_SIG_COEFF_FLAG_START == 105


class TestSliceTypeVariation:
    """Tests for slice-type-dependent initialization."""

    def test_different_slice_types_give_different_contexts(self):
        """Different slice types should initialize differently."""
        from entropy.cabac_context import init_context_models

        i_ctx = init_context_models(slice_type=2, slice_qp=26)
        p_ctx = init_context_models(slice_type=0, slice_qp=26)
        b_ctx = init_context_models(slice_type=1, slice_qp=26)

        # At least some contexts should differ between slice types
        i_states = [c.pStateIdx for c in i_ctx[:50]]
        p_states = [c.pStateIdx for c in p_ctx[:50]]
        b_states = [c.pStateIdx for c in b_ctx[:50]]

        # Not all identical
        assert i_states != p_states or i_states != b_states

    def test_qp_affects_initialization(self):
        """Different QP values should affect context initialization."""
        from entropy.cabac_context import init_context_models

        ctx_qp10 = init_context_models(slice_type=0, slice_qp=10)
        ctx_qp40 = init_context_models(slice_type=0, slice_qp=40)

        # States should differ due to QP
        states_10 = [c.pStateIdx for c in ctx_qp10[:50]]
        states_40 = [c.pStateIdx for c in ctx_qp40[:50]]

        assert states_10 != states_40
