# h264/entropy/tests/test_cabac_context_init.py
"""Tests for complete CABAC context initialization tables.

Verifies that all 460 context entries are populated with real (m, n) values
from the H.264 spec tables 9-12 through 9-23, not placeholders.

H.264 Spec Reference: Section 9.3.1.1
"""

import pytest


class TestInitTableCompleteness:
    """Verify all 460 entries are present and non-placeholder."""

    def test_init_i_has_460_entries(self):
        """_INIT_I must have exactly 460 entries."""
        from entropy.cabac_context import _INIT_I

        assert len(_INIT_I) == 460

    def test_init_pb_idc0_has_460_entries(self):
        """_INIT_PB_IDC0 must have exactly 460 entries."""
        from entropy.cabac_context import _INIT_PB_IDC0

        assert len(_INIT_PB_IDC0) == 460

    def test_init_pb_idc1_has_460_entries(self):
        """_INIT_PB_IDC1 must have exactly 460 entries."""
        from entropy.cabac_context import _INIT_PB_IDC1

        assert len(_INIT_PB_IDC1) == 460

    def test_init_pb_idc2_has_460_entries(self):
        """_INIT_PB_IDC2 must have exactly 460 entries."""
        from entropy.cabac_context import _INIT_PB_IDC2

        assert len(_INIT_PB_IDC2) == 460

    def test_sig_coeff_flag_not_placeholder(self):
        """significant_coeff_flag contexts (105-165) must have real values."""
        from entropy.cabac_context import _INIT_I

        assert _INIT_I[105] != (0, 64), (
            "significant_coeff_flag context still has placeholder"
        )

    def test_coeff_abs_level_not_placeholder(self):
        """coeff_abs_level_minus1 contexts (227-275) must have real values."""
        from entropy.cabac_context import _INIT_I

        assert _INIT_I[227] != (0, 64), (
            "coeff_abs_level context still has placeholder"
        )

    def test_last_sig_coeff_not_placeholder(self):
        """last_significant_coeff_flag contexts (166-226) must have real values."""
        from entropy.cabac_context import _INIT_I

        assert _INIT_I[166] != (0, 64), (
            "last_significant_coeff context still has placeholder"
        )

    def test_coded_block_flag_not_placeholder(self):
        """coded_block_flag contexts (85-104) must have real values."""
        from entropy.cabac_context import _INIT_I

        assert _INIT_I[85] != (0, 64), (
            "coded_block_flag context still has placeholder"
        )


class TestContextInitValues:
    """Spot-check specific context initialization values."""

    def test_context_init_i_slice_qp26_ctx0(self):
        """I-slice QP=26, context 0: m=20, n=-15.

        preCtxState = Clip3(1, 126, (20*26)>>4 + (-15))
                    = Clip3(1, 126, 32 + (-15)) = 17
        valMPS = 0 (17 < 64)
        pStateIdx = 63 - 17 = 46
        """
        from entropy.cabac_context import init_context_models

        contexts = init_context_models(slice_type=2, slice_qp=26)
        assert len(contexts) == 460
        assert contexts[0].pStateIdx == 46
        assert contexts[0].valMPS == 0

    def test_context_init_i_slice_qp26_ctx276(self):
        """Context 276 (end_of_slice_flag) has m=0, n=0.

        preCtxState = Clip3(1, 126, 0 + 0) = 1
        valMPS = 0 (1 < 64)
        pStateIdx = 63 - 1 = 62
        """
        from entropy.cabac_context import init_context_models

        contexts = init_context_models(slice_type=2, slice_qp=26)
        assert contexts[276].pStateIdx == 62
        assert contexts[276].valMPS == 0

    def test_context_init_p_slice_qp26_ctx14(self):
        """P-slice (idc=0) QP=26, context 14: m=1, n=9.

        preCtxState = Clip3(1, 126, (1*26)>>4 + 9) = Clip3(1, 126, 1 + 9) = 10
        valMPS = 0 (10 < 64)
        pStateIdx = 63 - 10 = 53
        """
        from entropy.cabac_context import init_context_models

        contexts = init_context_models(slice_type=0, slice_qp=26)
        assert contexts[14].pStateIdx == 53
        assert contexts[14].valMPS == 0

    def test_i_slice_unused_contexts_are_zero(self):
        """I-slice contexts 11-59 should be (0, 0) since they are unused."""
        from entropy.cabac_context import _INIT_I

        # I-slice doesn't use P/B mb_type, sub_mb_type, mvd, ref_idx ranges
        for idx in range(11, 60):
            assert _INIT_I[idx] == (0, 0), (
                f"I-slice context {idx} should be (0, 0), got {_INIT_I[idx]}"
            )


class TestCabacInitIdc:
    """Test cabac_init_idc table selection."""

    def test_idc_selects_different_tables(self):
        """Different cabac_init_idc values should give different contexts."""
        from entropy.cabac_context import init_context_models_with_idc

        ctx0 = init_context_models_with_idc(slice_type=0, slice_qp=26,
                                            cabac_init_idc=0)
        ctx1 = init_context_models_with_idc(slice_type=0, slice_qp=26,
                                            cabac_init_idc=1)
        ctx2 = init_context_models_with_idc(slice_type=0, slice_qp=26,
                                            cabac_init_idc=2)

        states0 = [(c.pStateIdx, c.valMPS) for c in ctx0]
        states1 = [(c.pStateIdx, c.valMPS) for c in ctx1]
        states2 = [(c.pStateIdx, c.valMPS) for c in ctx2]

        # Each idc should give different states
        assert states0 != states1
        assert states1 != states2
        assert states0 != states2

    def test_i_slice_ignores_idc(self):
        """I-slice should use the same table regardless of cabac_init_idc."""
        from entropy.cabac_context import init_context_models_with_idc

        ctx0 = init_context_models_with_idc(slice_type=2, slice_qp=26,
                                            cabac_init_idc=0)
        ctx1 = init_context_models_with_idc(slice_type=2, slice_qp=26,
                                            cabac_init_idc=1)
        ctx2 = init_context_models_with_idc(slice_type=2, slice_qp=26,
                                            cabac_init_idc=2)

        states0 = [(c.pStateIdx, c.valMPS) for c in ctx0]
        states1 = [(c.pStateIdx, c.valMPS) for c in ctx1]
        states2 = [(c.pStateIdx, c.valMPS) for c in ctx2]

        assert states0 == states1 == states2

    def test_pb_tables_are_distinct(self):
        """The three PB init tables should have distinct values."""
        from entropy.cabac_context import (
            _INIT_PB_IDC0, _INIT_PB_IDC1, _INIT_PB_IDC2,
        )

        assert _INIT_PB_IDC0 != _INIT_PB_IDC1
        assert _INIT_PB_IDC1 != _INIT_PB_IDC2
        assert _INIT_PB_IDC0 != _INIT_PB_IDC2

    def test_all_tables_share_i_slice_prefix(self):
        """All PB tables share the same first 11 entries (I-slice mb_type)."""
        from entropy.cabac_context import (
            _INIT_I, _INIT_PB_IDC0, _INIT_PB_IDC1, _INIT_PB_IDC2,
        )

        # First 11 entries (ctxIdx 0-10) are the same across all tables
        assert _INIT_I[:11] == _INIT_PB_IDC0[:11]
        assert _INIT_I[:11] == _INIT_PB_IDC1[:11]
        assert _INIT_I[:11] == _INIT_PB_IDC2[:11]


class TestValueRanges:
    """Verify (m, n) values are in valid int8 range."""

    @pytest.mark.parametrize("table_name", [
        "_INIT_I", "_INIT_PB_IDC0", "_INIT_PB_IDC1", "_INIT_PB_IDC2",
    ])
    def test_m_values_in_range(self, table_name):
        """m values should be in int8 range [-128, 127]."""
        import entropy.cabac_context as mod

        table = getattr(mod, table_name)
        for idx, (m, n) in enumerate(table):
            assert -128 <= m <= 127, (
                f"{table_name}[{idx}]: m={m} out of int8 range"
            )

    @pytest.mark.parametrize("table_name", [
        "_INIT_I", "_INIT_PB_IDC0", "_INIT_PB_IDC1", "_INIT_PB_IDC2",
    ])
    def test_n_values_in_range(self, table_name):
        """n values should be in int8 range [-128, 127]."""
        import entropy.cabac_context as mod

        table = getattr(mod, table_name)
        for idx, (m, n) in enumerate(table):
            assert -128 <= n <= 127, (
                f"{table_name}[{idx}]: n={n} out of int8 range"
            )
