# h264/parameters/tests/test_hrd.py
"""Tests for HRD (Hypothetical Reference Decoder) parameters parsing (TDD - RED phase).

H.264 Spec Reference:
- Annex E.1.2: HRD parameters syntax
- Annex E.2.2: HRD parameters semantics
- Annex C: Hypothetical reference decoder

The HRD is used for:
- Buffer management and underflow/overflow prevention
- Timing verification for compliant bitstreams
- Constant bit rate (CBR) and variable bit rate (VBR) scheduling

These tests are written TDD-style and should FAIL until the
HRD parsing features are properly implemented.
"""

import pytest

from bitstream import BitWriter, BITSTRING_AVAILABLE


pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestHRDParametersParsing:
    """Test HRD parameters parsing from bitstream.

    H.264 Spec: Annex E.1.2
    """

    @pytest.mark.xfail(reason="HRD parameters parsing not implemented")
    def test_parse_minimal_hrd_parameters(self):
        """Parse minimal HRD parameters with single CPB."""
        from parameters.hrd import parse_hrd_parameters, HRDParameters

        writer = BitWriter()
        writer.write_ue(0)  # cpb_cnt_minus1 = 0 (1 CPB)
        writer.write_bits(4, 4)  # bit_rate_scale
        writer.write_bits(6, 4)  # cpb_size_scale
        # SchedSelIdx = 0
        writer.write_ue(62499)  # bit_rate_value_minus1[0]
        writer.write_ue(62499)  # cpb_size_value_minus1[0]
        writer.write_flag(False)  # cbr_flag[0]
        writer.write_bits(23, 5)  # initial_cpb_removal_delay_length_minus1
        writer.write_bits(23, 5)  # cpb_removal_delay_length_minus1
        writer.write_bits(23, 5)  # dpb_output_delay_length_minus1
        writer.write_bits(24, 5)  # time_offset_length

        data = writer.to_bytes()
        hrd = parse_hrd_parameters(data)

        assert isinstance(hrd, HRDParameters)
        assert hrd.cpb_cnt_minus1 == 0
        assert hrd.cpb_cnt == 1
        assert hrd.bit_rate_scale == 4
        assert hrd.cpb_size_scale == 6

    @pytest.mark.xfail(reason="HRD parameters parsing not implemented")
    def test_parse_hrd_with_multiple_cpbs(self):
        """Parse HRD parameters with multiple CPBs (Schedule Selection)."""
        from parameters.hrd import parse_hrd_parameters

        writer = BitWriter()
        writer.write_ue(2)  # cpb_cnt_minus1 = 2 (3 CPBs)
        writer.write_bits(4, 4)  # bit_rate_scale
        writer.write_bits(6, 4)  # cpb_size_scale
        # 3 CPB configurations
        for i in range(3):
            writer.write_ue(62499 + i * 1000)  # bit_rate_value_minus1[i]
            writer.write_ue(62499 + i * 2000)  # cpb_size_value_minus1[i]
            writer.write_flag(i % 2 == 0)  # cbr_flag[i]
        writer.write_bits(23, 5)
        writer.write_bits(23, 5)
        writer.write_bits(23, 5)
        writer.write_bits(24, 5)

        data = writer.to_bytes()
        hrd = parse_hrd_parameters(data)

        assert hrd.cpb_cnt == 3
        assert len(hrd.bit_rate_value_minus1) == 3
        assert len(hrd.cpb_size_value_minus1) == 3
        assert len(hrd.cbr_flag) == 3
        assert hrd.cbr_flag[0] is True
        assert hrd.cbr_flag[1] is False
        assert hrd.cbr_flag[2] is True


class TestHRDBitRateCalculation:
    """Test HRD bit rate calculation.

    H.264 Spec: Section E.2.2
    BitRate[i] = (bit_rate_value_minus1[i] + 1) * 2^(6 + bit_rate_scale)
    """

    @pytest.mark.xfail(reason="HRD bit rate calculation not implemented")
    def test_calculate_bit_rate_low(self):
        """Calculate bit rate for low bitrate stream."""
        from parameters.hrd import HRDParameters

        # bit_rate_value_minus1 = 62499, bit_rate_scale = 4
        # BitRate = (62499 + 1) * 2^(6+4) = 62500 * 1024 = 64,000,000 bps = 64 Mbps
        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        assert hrd.bit_rate(0) == 64_000_000

    @pytest.mark.xfail(reason="HRD bit rate calculation not implemented")
    def test_calculate_bit_rate_high(self):
        """Calculate bit rate for high bitrate stream (4K)."""
        from parameters.hrd import HRDParameters

        # For a ~100 Mbps stream
        # bit_rate_scale = 6, bit_rate_value_minus1 = 24413
        # BitRate = (24413 + 1) * 2^(6+6) = 24414 * 4096 = 100,007,424 bps
        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=6,
            cpb_size_scale=6,
            bit_rate_value_minus1=[24413],
            cpb_size_value_minus1=[24413],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        assert abs(hrd.bit_rate(0) - 100_007_424) < 1000

    @pytest.mark.xfail(reason="HRD bit rate calculation not implemented")
    def test_calculate_bit_rate_multiple_cpbs(self):
        """Calculate bit rates for multiple CPBs."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=2,
            bit_rate_scale=4,
            bit_rate_value_minus1=[31249, 62499, 124999],  # ~32, 64, 128 Mbps
            cpb_size_value_minus1=[31249, 62499, 124999],
            cpb_size_scale=6,
            cbr_flag=[False, False, False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        # Check each CPB has different bit rate
        assert hrd.bit_rate(0) < hrd.bit_rate(1) < hrd.bit_rate(2)


class TestHRDCPBSizeCalculation:
    """Test HRD CPB size calculation.

    H.264 Spec: Section E.2.2
    CpbSize[i] = (cpb_size_value_minus1[i] + 1) * 2^(4 + cpb_size_scale)
    """

    @pytest.mark.xfail(reason="HRD CPB size calculation not implemented")
    def test_calculate_cpb_size(self):
        """Calculate CPB size in bits."""
        from parameters.hrd import HRDParameters

        # cpb_size_value_minus1 = 62499, cpb_size_scale = 6
        # CpbSize = (62499 + 1) * 2^(4+6) = 62500 * 1024 = 64,000,000 bits
        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        assert hrd.cpb_size(0) == 64_000_000  # bits
        assert hrd.cpb_size_bytes(0) == 8_000_000  # bytes

    @pytest.mark.xfail(reason="HRD CPB size calculation not implemented")
    def test_calculate_cpb_size_small_buffer(self):
        """Calculate small CPB size for low-delay applications."""
        from parameters.hrd import HRDParameters

        # Small buffer for low latency: cpb_size_scale = 2
        # cpb_size_value_minus1 = 9999
        # CpbSize = (9999 + 1) * 2^(4+2) = 10000 * 64 = 640,000 bits = 80 KB
        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=2,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[9999],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        assert hrd.cpb_size(0) == 640_000
        assert hrd.cpb_size_bytes(0) == 80_000


class TestHRDDelayLengths:
    """Test HRD delay length fields.

    H.264 Spec: Section E.2.2
    """

    @pytest.mark.xfail(reason="HRD delay lengths not implemented")
    def test_initial_cpb_removal_delay_length(self):
        """Verify initial_cpb_removal_delay_length calculation."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=15,
            dpb_output_delay_length_minus1=7,
            time_offset_length=24
        )

        assert hrd.initial_cpb_removal_delay_length == 24
        assert hrd.cpb_removal_delay_length == 16
        assert hrd.dpb_output_delay_length == 8

    @pytest.mark.xfail(reason="HRD delay lengths not implemented")
    def test_time_offset_length_zero(self):
        """Time offset length can be 0 (no time offset in timestamps)."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=0
        )

        assert hrd.time_offset_length == 0


class TestHRDCBRVBRMode:
    """Test HRD CBR/VBR mode handling.

    H.264 Spec: Section E.2.2
    cbr_flag[i] = 1: constant bit rate operation
    cbr_flag[i] = 0: variable bit rate operation
    """

    @pytest.mark.xfail(reason="HRD CBR/VBR mode not implemented")
    def test_cbr_mode(self):
        """CBR (Constant Bit Rate) mode detection."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[True],  # CBR mode
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        assert hrd.is_cbr(0) is True
        assert hrd.is_vbr(0) is False

    @pytest.mark.xfail(reason="HRD CBR/VBR mode not implemented")
    def test_vbr_mode(self):
        """VBR (Variable Bit Rate) mode detection."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],  # VBR mode
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        assert hrd.is_cbr(0) is False
        assert hrd.is_vbr(0) is True

    @pytest.mark.xfail(reason="HRD CBR/VBR mode not implemented")
    def test_mixed_cbr_vbr_cpbs(self):
        """Multiple CPBs with mixed CBR/VBR modes."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=1,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499, 124999],
            cpb_size_value_minus1=[62499, 124999],
            cbr_flag=[True, False],  # CPB 0: CBR, CPB 1: VBR
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        assert hrd.is_cbr(0) is True
        assert hrd.is_vbr(1) is True


class TestHRDDataclass:
    """Test HRDParameters dataclass properties."""

    @pytest.mark.xfail(reason="HRD dataclass not implemented")
    def test_hrd_dataclass_fields(self):
        """HRDParameters has all required fields."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        assert hasattr(hrd, 'cpb_cnt_minus1')
        assert hasattr(hrd, 'bit_rate_scale')
        assert hasattr(hrd, 'cpb_size_scale')
        assert hasattr(hrd, 'bit_rate_value_minus1')
        assert hasattr(hrd, 'cpb_size_value_minus1')
        assert hasattr(hrd, 'cbr_flag')
        assert hasattr(hrd, 'initial_cpb_removal_delay_length_minus1')
        assert hasattr(hrd, 'cpb_removal_delay_length_minus1')
        assert hasattr(hrd, 'dpb_output_delay_length_minus1')
        assert hasattr(hrd, 'time_offset_length')

    @pytest.mark.xfail(reason="HRD dataclass not implemented")
    def test_hrd_repr(self):
        """HRDParameters has readable repr."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        s = repr(hrd)
        assert 'cpb_cnt' in s.lower() or 'hrd' in s.lower()


class TestHRDIntegrationWithVUI:
    """Test HRD parameters integration with VUI parsing."""

    @pytest.mark.xfail(reason="HRD VUI integration not implemented")
    def test_vui_stores_nal_hrd_parameters(self):
        """VUI should store parsed NAL HRD parameters."""
        from parameters.sps import VUIParameters
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        vui = VUIParameters(
            nal_hrd_parameters_present_flag=True,
            nal_hrd_parameters=hrd
        )

        assert vui.nal_hrd_parameters is not None
        assert vui.nal_hrd_parameters.cpb_cnt == 1

    @pytest.mark.xfail(reason="HRD VUI integration not implemented")
    def test_vui_stores_vcl_hrd_parameters(self):
        """VUI should store parsed VCL HRD parameters."""
        from parameters.sps import VUIParameters
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[True],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        vui = VUIParameters(
            vcl_hrd_parameters_present_flag=True,
            vcl_hrd_parameters=hrd
        )

        assert vui.vcl_hrd_parameters is not None
        assert vui.vcl_hrd_parameters.is_cbr(0) is True

    @pytest.mark.xfail(reason="HRD VUI integration not implemented")
    def test_parse_vui_with_hrd(self):
        """Parse complete VUI with HRD parameters."""
        from parameters.sps import parse_vui_parameters

        writer = BitWriter()
        writer.write_flag(False)  # aspect_ratio
        writer.write_flag(False)  # overscan
        writer.write_flag(False)  # video_signal
        writer.write_flag(False)  # chroma_loc
        writer.write_flag(False)  # timing
        writer.write_flag(True)   # nal_hrd_parameters_present_flag
        # HRD parameters
        writer.write_ue(0)  # cpb_cnt_minus1
        writer.write_bits(4, 4)  # bit_rate_scale
        writer.write_bits(6, 4)  # cpb_size_scale
        writer.write_ue(62499)  # bit_rate_value_minus1[0]
        writer.write_ue(62499)  # cpb_size_value_minus1[0]
        writer.write_flag(False)  # cbr_flag[0]
        writer.write_bits(23, 5)
        writer.write_bits(23, 5)
        writer.write_bits(23, 5)
        writer.write_bits(24, 5)
        writer.write_flag(False)  # vcl_hrd
        writer.write_flag(False)  # low_delay_hrd
        writer.write_flag(False)  # pic_struct
        writer.write_flag(False)  # bitstream_restriction

        data = writer.to_bytes()
        vui = parse_vui_parameters(data)

        assert vui.nal_hrd_parameters_present_flag is True
        assert vui.nal_hrd_parameters is not None
        assert vui.nal_hrd_parameters.bit_rate_scale == 4


class TestHRDBufferingPeriodIntegration:
    """Test HRD parameters used with buffering_period SEI."""

    @pytest.mark.xfail(reason="HRD buffering period integration not implemented")
    def test_hrd_provides_delay_lengths_for_sei(self):
        """HRD provides delay field lengths for SEI parsing."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=15,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        # These values are used when parsing buffering_period SEI
        assert hrd.initial_cpb_removal_delay_length == 24
        # These values are used when parsing pic_timing SEI
        assert hrd.cpb_removal_delay_length == 16
        assert hrd.dpb_output_delay_length == 24

    @pytest.mark.xfail(reason="HRD buffering period integration not implemented")
    def test_hrd_cpb_count_for_sei(self):
        """HRD CPB count determines number of delay values in SEI."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=2,  # 3 CPBs
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499, 62499, 62499],
            cpb_size_value_minus1=[62499, 62499, 62499],
            cbr_flag=[False, False, False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        # buffering_period SEI will have 3 pairs of delay values
        assert hrd.cpb_cnt == 3


class TestHRDModuleInterface:
    """Test HRD module interface."""

    @pytest.mark.xfail(reason="HRD module not implemented")
    def test_module_exports_hrd_parameters(self):
        """Module exports HRDParameters dataclass."""
        from parameters import hrd

        assert hasattr(hrd, 'HRDParameters')

    @pytest.mark.xfail(reason="HRD module not implemented")
    def test_module_exports_parse_function(self):
        """Module exports parse_hrd_parameters function."""
        from parameters import hrd

        assert hasattr(hrd, 'parse_hrd_parameters')
        assert callable(hrd.parse_hrd_parameters)


class TestHRDEdgeCases:
    """Test edge cases in HRD parameter handling."""

    @pytest.mark.xfail(reason="HRD edge cases not implemented")
    def test_maximum_cpb_count(self):
        """Handle maximum CPB count (32 CPBs)."""
        from parameters.hrd import parse_hrd_parameters

        writer = BitWriter()
        writer.write_ue(31)  # cpb_cnt_minus1 = 31 (32 CPBs)
        writer.write_bits(4, 4)
        writer.write_bits(6, 4)
        for _ in range(32):
            writer.write_ue(62499)
            writer.write_ue(62499)
            writer.write_flag(False)
        writer.write_bits(23, 5)
        writer.write_bits(23, 5)
        writer.write_bits(23, 5)
        writer.write_bits(24, 5)

        data = writer.to_bytes()
        hrd = parse_hrd_parameters(data)

        assert hrd.cpb_cnt == 32

    @pytest.mark.xfail(reason="HRD edge cases not implemented")
    def test_minimum_delay_lengths(self):
        """Handle minimum delay length values (1 bit)."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=0,
            cpb_size_scale=0,
            bit_rate_value_minus1=[0],
            cpb_size_value_minus1=[0],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=0,  # 1 bit
            cpb_removal_delay_length_minus1=0,  # 1 bit
            dpb_output_delay_length_minus1=0,  # 1 bit
            time_offset_length=0
        )

        assert hrd.initial_cpb_removal_delay_length == 1
        assert hrd.cpb_removal_delay_length == 1
        assert hrd.dpb_output_delay_length == 1

    @pytest.mark.xfail(reason="HRD edge cases not implemented")
    def test_maximum_delay_lengths(self):
        """Handle maximum delay length values (32 bits)."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=0,
            cpb_size_scale=0,
            bit_rate_value_minus1=[0],
            cpb_size_value_minus1=[0],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=31,  # 32 bits
            cpb_removal_delay_length_minus1=31,  # 32 bits
            dpb_output_delay_length_minus1=31,  # 32 bits
            time_offset_length=31
        )

        assert hrd.initial_cpb_removal_delay_length == 32
        assert hrd.cpb_removal_delay_length == 32
        assert hrd.dpb_output_delay_length == 32
        assert hrd.time_offset_length == 31

    @pytest.mark.xfail(reason="HRD edge cases not implemented")
    def test_very_high_bit_rate(self):
        """Calculate very high bit rate (streaming 8K)."""
        from parameters.hrd import HRDParameters

        # For ~500 Mbps stream (8K streaming)
        # bit_rate_scale = 8, bit_rate_value_minus1 = 30517
        # BitRate = (30517 + 1) * 2^(6+8) = 30518 * 16384 = 500,006,912 bps
        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=8,
            cpb_size_scale=8,
            bit_rate_value_minus1=[30517],
            cpb_size_value_minus1=[30517],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        assert hrd.bit_rate(0) == 500_006_912


class TestHRDHelperMethods:
    """Test HRD helper methods."""

    @pytest.mark.xfail(reason="HRD helper methods not implemented")
    def test_buffer_fullness_time(self):
        """Calculate time to fill buffer at given bit rate."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[True],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        # Time to fill buffer = cpb_size / bit_rate
        # 64,000,000 bits / 64,000,000 bps = 1.0 second
        fill_time = hrd.buffer_fill_time(0)
        assert abs(fill_time - 1.0) < 0.001

    @pytest.mark.xfail(reason="HRD helper methods not implemented")
    def test_max_buffer_delay(self):
        """Calculate maximum buffering delay in 90kHz units."""
        from parameters.hrd import HRDParameters

        hrd = HRDParameters(
            cpb_cnt_minus1=0,
            bit_rate_scale=4,
            cpb_size_scale=6,
            bit_rate_value_minus1=[62499],
            cpb_size_value_minus1=[62499],
            cbr_flag=[False],
            initial_cpb_removal_delay_length_minus1=23,
            cpb_removal_delay_length_minus1=23,
            dpb_output_delay_length_minus1=23,
            time_offset_length=24
        )

        # Maximum value in initial_cpb_removal_delay field
        # With 24 bits: max = 2^24 - 1 = 16777215
        # In 90kHz units: 16777215 / 90000 = 186.4 seconds max delay
        max_delay_90khz = hrd.max_initial_cpb_removal_delay()
        assert max_delay_90khz == (1 << 24) - 1
