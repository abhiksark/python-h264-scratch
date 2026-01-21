# h264/parameters/tests/test_sei.py
"""Tests for SEI (Supplemental Enhancement Information) parsing (TDD - RED phase).

H.264 Spec Reference:
- Annex D: Supplemental enhancement information
- Section D.1: SEI payload syntax
- Section D.2: SEI payload semantics

These tests are written TDD-style and should FAIL until the
SEI parsing features are properly implemented.
"""

import uuid
import pytest

from bitstream import BitWriter, BITSTRING_AVAILABLE


pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


# H.264 Spec Table D-1: SEI payload type codes
SEI_TYPE_BUFFERING_PERIOD = 0
SEI_TYPE_PIC_TIMING = 1
SEI_TYPE_PAN_SCAN_RECT = 2
SEI_TYPE_FILLER_PAYLOAD = 3
SEI_TYPE_USER_DATA_REGISTERED = 4
SEI_TYPE_USER_DATA_UNREGISTERED = 5
SEI_TYPE_RECOVERY_POINT = 6
SEI_TYPE_DEC_REF_PIC_MARKING_REPETITION = 7
SEI_TYPE_SPARE_PIC = 8
SEI_TYPE_SCENE_INFO = 9
SEI_TYPE_SUB_SEQ_INFO = 10
SEI_TYPE_SUB_SEQ_LAYER_CHARACTERISTICS = 11
SEI_TYPE_SUB_SEQ_CHARACTERISTICS = 12
SEI_TYPE_FULL_FRAME_FREEZE = 13
SEI_TYPE_FULL_FRAME_FREEZE_RELEASE = 14
SEI_TYPE_FULL_FRAME_SNAPSHOT = 15
SEI_TYPE_PROGRESSIVE_REFINEMENT_SEGMENT_START = 16
SEI_TYPE_PROGRESSIVE_REFINEMENT_SEGMENT_END = 17
SEI_TYPE_MOTION_CONSTRAINED_SLICE_GROUP_SET = 18
SEI_TYPE_FILM_GRAIN_CHARACTERISTICS = 19
SEI_TYPE_DEBLOCKING_FILTER_DISPLAY_PREFERENCE = 20
SEI_TYPE_STEREO_VIDEO_INFO = 21
SEI_TYPE_POST_FILTER_HINT = 22
SEI_TYPE_TONE_MAPPING_INFO = 23
SEI_TYPE_SCALABILITY_INFO = 24
SEI_TYPE_SUB_PIC_SCALABLE_LAYER = 25
SEI_TYPE_NON_REQUIRED_LAYER_REP = 26
SEI_TYPE_PRIORITY_LAYER_INFO = 27
SEI_TYPE_LAYERS_NOT_PRESENT = 28
SEI_TYPE_LAYER_DEPENDENCY_CHANGE = 29
SEI_TYPE_SCALABLE_NESTING = 30
SEI_TYPE_BASE_LAYER_TEMPORAL_HRD = 31
SEI_TYPE_QUALITY_LAYER_INTEGRITY_CHECK = 32
SEI_TYPE_REDUNDANT_PIC_PROPERTY = 33
SEI_TYPE_TL0_DEP_REP_INDEX = 34
SEI_TYPE_TL_SWITCHING_POINT = 35
SEI_TYPE_PARALLEL_DECODING_INFO = 36
SEI_TYPE_MVC_SCALABLE_NESTING = 37
SEI_TYPE_VIEW_SCALABILITY_INFO = 38
SEI_TYPE_MULTIVIEW_SCENE_INFO = 39
SEI_TYPE_MULTIVIEW_ACQUISITION_INFO = 40
SEI_TYPE_NON_REQUIRED_VIEW_COMPONENT = 41
SEI_TYPE_VIEW_DEPENDENCY_CHANGE = 42
SEI_TYPE_OPERATION_POINTS_NOT_PRESENT = 43
SEI_TYPE_BASE_VIEW_TEMPORAL_HRD = 44
SEI_TYPE_FRAME_PACKING_ARRANGEMENT = 45

# x264/FFmpeg common UUIDs
UUID_X264 = uuid.UUID("dc45e9bd-e6d9-48b7-962c-d820d923eeef")
UUID_APPLE_PRORES = uuid.UUID("a85e3b77-e47e-4883-8b0d-2e6e5a929e7d")


class TestSEIPayloadTypeSizeParsing:
    """Test SEI payload type and size parsing.

    H.264 Spec: Section D.1.1
    """

    @pytest.mark.xfail(reason="SEI parsing not implemented")
    def test_parse_single_byte_payload_type(self):
        """Parse SEI with single-byte payload type (< 255)."""
        from parameters.sei import parse_sei_message

        writer = BitWriter()
        writer.write_bits(SEI_TYPE_PIC_TIMING, 8)  # payload_type = 1
        writer.write_bits(4, 8)  # payload_size = 4
        writer.write_bits(0, 32)  # 4 bytes of payload data

        data = writer.to_bytes()
        sei = parse_sei_message(data)

        assert sei.payload_type == SEI_TYPE_PIC_TIMING
        assert sei.payload_size == 4

    @pytest.mark.xfail(reason="SEI parsing not implemented")
    def test_parse_multi_byte_payload_type(self):
        """Parse SEI with multi-byte payload type (>= 255)."""
        from parameters.sei import parse_sei_message

        writer = BitWriter()
        # Type 260 = 255 + 5 (two bytes)
        writer.write_bits(255, 8)  # ff_byte
        writer.write_bits(5, 8)  # last_payload_type_byte
        writer.write_bits(0, 8)  # payload_size = 0

        data = writer.to_bytes()
        sei = parse_sei_message(data)

        assert sei.payload_type == 260

    @pytest.mark.xfail(reason="SEI parsing not implemented")
    def test_parse_single_byte_payload_size(self):
        """Parse SEI with single-byte payload size (< 255)."""
        from parameters.sei import parse_sei_message

        writer = BitWriter()
        writer.write_bits(SEI_TYPE_FILLER_PAYLOAD, 8)  # payload_type
        writer.write_bits(100, 8)  # payload_size = 100
        for _ in range(100):
            writer.write_bits(0xFF, 8)

        data = writer.to_bytes()
        sei = parse_sei_message(data)

        assert sei.payload_size == 100
        assert len(sei.payload_data) == 100

    @pytest.mark.xfail(reason="SEI parsing not implemented")
    def test_parse_multi_byte_payload_size(self):
        """Parse SEI with multi-byte payload size (>= 255)."""
        from parameters.sei import parse_sei_message

        writer = BitWriter()
        writer.write_bits(SEI_TYPE_FILLER_PAYLOAD, 8)
        # Size 300 = 255 + 45 (two bytes)
        writer.write_bits(255, 8)  # ff_byte
        writer.write_bits(45, 8)   # last_payload_size_byte
        for _ in range(300):
            writer.write_bits(0xFF, 8)

        data = writer.to_bytes()
        sei = parse_sei_message(data)

        assert sei.payload_size == 300

    @pytest.mark.xfail(reason="SEI parsing not implemented")
    def test_parse_zero_size_payload(self):
        """Parse SEI with zero-size payload."""
        from parameters.sei import parse_sei_message

        writer = BitWriter()
        writer.write_bits(SEI_TYPE_FULL_FRAME_FREEZE_RELEASE, 8)
        writer.write_bits(0, 8)  # payload_size = 0

        data = writer.to_bytes()
        sei = parse_sei_message(data)

        assert sei.payload_size == 0
        assert sei.payload_data == b''


class TestMultipleSEIMessagesInNAL:
    """Test parsing multiple SEI messages in a single NAL unit.

    H.264 Spec: Section 7.3.2.3
    """

    @pytest.mark.xfail(reason="Multiple SEI parsing not implemented")
    def test_parse_two_sei_messages(self):
        """Parse NAL unit with two SEI messages."""
        from parameters.sei import parse_sei_nal_unit

        writer = BitWriter()
        # First SEI: pic_timing
        writer.write_bits(SEI_TYPE_PIC_TIMING, 8)
        writer.write_bits(4, 8)
        writer.write_bits(0, 32)
        # Second SEI: recovery_point
        writer.write_bits(SEI_TYPE_RECOVERY_POINT, 8)
        writer.write_bits(3, 8)
        writer.write_bits(0, 24)
        # RBSP trailing bits
        writer.write_bits(0x80, 8)

        data = writer.to_bytes()
        messages = parse_sei_nal_unit(data)

        assert len(messages) == 2
        assert messages[0].payload_type == SEI_TYPE_PIC_TIMING
        assert messages[1].payload_type == SEI_TYPE_RECOVERY_POINT

    @pytest.mark.xfail(reason="Multiple SEI parsing not implemented")
    def test_parse_three_sei_messages(self):
        """Parse NAL unit with three SEI messages."""
        from parameters.sei import parse_sei_nal_unit

        writer = BitWriter()
        # buffering_period
        writer.write_bits(SEI_TYPE_BUFFERING_PERIOD, 8)
        writer.write_bits(5, 8)
        writer.write_bits(0, 40)
        # pic_timing
        writer.write_bits(SEI_TYPE_PIC_TIMING, 8)
        writer.write_bits(4, 8)
        writer.write_bits(0, 32)
        # user_data_unregistered
        writer.write_bits(SEI_TYPE_USER_DATA_UNREGISTERED, 8)
        writer.write_bits(20, 8)  # 16 bytes UUID + 4 bytes data
        for b in UUID_X264.bytes:
            writer.write_bits(b, 8)
        writer.write_bits(0, 32)
        writer.write_bits(0x80, 8)

        data = writer.to_bytes()
        messages = parse_sei_nal_unit(data)

        assert len(messages) == 3

    @pytest.mark.xfail(reason="Multiple SEI parsing not implemented")
    def test_parse_sei_until_rbsp_trailing(self):
        """Stop parsing at RBSP trailing bits."""
        from parameters.sei import parse_sei_nal_unit

        writer = BitWriter()
        writer.write_bits(SEI_TYPE_FILLER_PAYLOAD, 8)
        writer.write_bits(2, 8)
        writer.write_bits(0xFFFF, 16)
        # RBSP trailing bits: 1 followed by zeros
        writer.write_flag(True)
        writer.write_bits(0, 7)

        data = writer.to_bytes()
        messages = parse_sei_nal_unit(data)

        assert len(messages) == 1


class TestBufferingPeriodSEI:
    """Test buffering_period SEI message parsing.

    H.264 Spec: Section D.1.2, D.2.2
    """

    @pytest.mark.xfail(reason="buffering_period SEI not implemented")
    def test_buffering_period_nal_hrd(self):
        """Parse buffering_period with NAL HRD parameters."""
        from parameters.sei import parse_buffering_period_sei, SEIBufferingPeriod

        # Assume HRD parameters: cpb_cnt=1, initial_cpb_removal_delay_length=24
        writer = BitWriter()
        writer.write_ue(0)  # seq_parameter_set_id
        # For NAL HRD (1 CPB)
        writer.write_bits(90000, 24)  # initial_cpb_removal_delay[0]
        writer.write_bits(90000, 24)  # initial_cpb_removal_delay_offset[0]

        data = writer.to_bytes()
        bp = parse_buffering_period_sei(
            data,
            nal_hrd_params={'cpb_cnt': 1, 'initial_cpb_removal_delay_length': 24},
            vcl_hrd_params=None
        )

        assert isinstance(bp, SEIBufferingPeriod)
        assert bp.seq_parameter_set_id == 0
        assert bp.initial_cpb_removal_delay[0] == 90000
        assert bp.initial_cpb_removal_delay_offset[0] == 90000

    @pytest.mark.xfail(reason="buffering_period SEI not implemented")
    def test_buffering_period_vcl_hrd(self):
        """Parse buffering_period with VCL HRD parameters."""
        from parameters.sei import parse_buffering_period_sei

        writer = BitWriter()
        writer.write_ue(0)
        # NAL HRD (2 CPBs)
        for _ in range(2):
            writer.write_bits(90000, 24)
            writer.write_bits(90000, 24)
        # VCL HRD (2 CPBs)
        for _ in range(2):
            writer.write_bits(45000, 24)
            writer.write_bits(45000, 24)

        data = writer.to_bytes()
        bp = parse_buffering_period_sei(
            data,
            nal_hrd_params={'cpb_cnt': 2, 'initial_cpb_removal_delay_length': 24},
            vcl_hrd_params={'cpb_cnt': 2, 'initial_cpb_removal_delay_length': 24}
        )

        assert len(bp.initial_cpb_removal_delay) == 2
        assert len(bp.vcl_initial_cpb_removal_delay) == 2

    @pytest.mark.xfail(reason="buffering_period SEI not implemented")
    def test_buffering_period_different_sps_id(self):
        """Parse buffering_period referencing different SPS."""
        from parameters.sei import parse_buffering_period_sei

        writer = BitWriter()
        writer.write_ue(3)  # seq_parameter_set_id = 3

        data = writer.to_bytes()
        bp = parse_buffering_period_sei(
            data,
            nal_hrd_params=None,
            vcl_hrd_params=None
        )

        assert bp.seq_parameter_set_id == 3


class TestPicTimingSEI:
    """Test pic_timing SEI message parsing.

    H.264 Spec: Section D.1.3, D.2.3
    """

    @pytest.mark.xfail(reason="pic_timing SEI not implemented")
    def test_pic_timing_with_cpb_dpb_delay(self):
        """Parse pic_timing with CPB/DPB removal delays."""
        from parameters.sei import parse_pic_timing_sei, SEIPicTiming

        writer = BitWriter()
        # Assume cpb_removal_delay_length=24, dpb_output_delay_length=24
        writer.write_bits(1000, 24)  # cpb_removal_delay
        writer.write_bits(2000, 24)  # dpb_output_delay

        data = writer.to_bytes()
        pt = parse_pic_timing_sei(
            data,
            cpb_removal_delay_length=24,
            dpb_output_delay_length=24,
            pic_struct_present_flag=False
        )

        assert isinstance(pt, SEIPicTiming)
        assert pt.cpb_removal_delay == 1000
        assert pt.dpb_output_delay == 2000

    @pytest.mark.xfail(reason="pic_timing SEI not implemented")
    def test_pic_timing_with_pic_struct(self):
        """Parse pic_timing with pic_struct field."""
        from parameters.sei import parse_pic_timing_sei

        writer = BitWriter()
        writer.write_bits(1000, 24)  # cpb_removal_delay
        writer.write_bits(2000, 24)  # dpb_output_delay
        writer.write_bits(0, 4)  # pic_struct = 0 (frame)
        # NumClockTS = 1 for pic_struct=0
        writer.write_flag(False)  # clock_timestamp_flag[0]

        data = writer.to_bytes()
        pt = parse_pic_timing_sei(
            data,
            cpb_removal_delay_length=24,
            dpb_output_delay_length=24,
            pic_struct_present_flag=True
        )

        assert pt.pic_struct == 0

    @pytest.mark.xfail(reason="pic_timing SEI not implemented")
    def test_pic_timing_pic_struct_top_bottom_fields(self):
        """Parse pic_timing for interlaced content (top/bottom fields)."""
        from parameters.sei import parse_pic_timing_sei

        writer = BitWriter()
        writer.write_bits(1000, 24)
        writer.write_bits(2000, 24)
        writer.write_bits(1, 4)  # pic_struct = 1 (top field)
        writer.write_flag(False)

        data = writer.to_bytes()
        pt = parse_pic_timing_sei(
            data,
            cpb_removal_delay_length=24,
            dpb_output_delay_length=24,
            pic_struct_present_flag=True
        )

        assert pt.pic_struct == 1
        assert pt.is_top_field is True

    @pytest.mark.xfail(reason="pic_timing SEI not implemented")
    def test_pic_timing_with_clock_timestamp(self):
        """Parse pic_timing with clock timestamp information."""
        from parameters.sei import parse_pic_timing_sei

        writer = BitWriter()
        writer.write_bits(1000, 24)
        writer.write_bits(2000, 24)
        writer.write_bits(0, 4)  # pic_struct = 0
        writer.write_flag(True)  # clock_timestamp_flag[0] = 1
        # Clock timestamp fields
        writer.write_bits(0, 2)   # ct_type
        writer.write_flag(False)  # nuit_field_based_flag
        writer.write_bits(5, 5)   # counting_type
        writer.write_flag(True)   # full_timestamp_flag
        writer.write_flag(False)  # discontinuity_flag
        writer.write_flag(False)  # cnt_dropped_flag
        writer.write_bits(0, 8)   # n_frames
        # Full timestamp
        writer.write_bits(10, 6)  # seconds_value
        writer.write_bits(30, 6)  # minutes_value
        writer.write_bits(14, 5)  # hours_value
        writer.write_bits(0, 5)   # time_offset (assume 5 bits)

        data = writer.to_bytes()
        pt = parse_pic_timing_sei(
            data,
            cpb_removal_delay_length=24,
            dpb_output_delay_length=24,
            pic_struct_present_flag=True,
            time_offset_length=5
        )

        assert pt.clock_timestamps[0].hours == 14
        assert pt.clock_timestamps[0].minutes == 30
        assert pt.clock_timestamps[0].seconds == 10


class TestUserDataRegisteredSEI:
    """Test user_data_registered_itu_t_t35 SEI message parsing.

    H.264 Spec: Section D.1.6, D.2.6
    """

    @pytest.mark.xfail(reason="user_data_registered SEI not implemented")
    def test_user_data_registered_atsc(self):
        """Parse user_data_registered with ATSC country code."""
        from parameters.sei import parse_user_data_registered_sei

        writer = BitWriter()
        writer.write_bits(0xB5, 8)  # itu_t_t35_country_code (US = 0xB5)
        writer.write_bits(0x31, 8)  # itu_t_t35_provider_code (ATSC)
        writer.write_bits(0x47413934, 32)  # "GA94" user identifier
        # CC data
        writer.write_bits(0x03, 8)  # user_data_type_code
        for _ in range(10):
            writer.write_bits(0, 8)

        data = writer.to_bytes()
        ud = parse_user_data_registered_sei(data, payload_size=len(data))

        assert ud.country_code == 0xB5
        assert ud.provider_code == 0x31
        assert ud.user_identifier == b'GA94'

    @pytest.mark.xfail(reason="user_data_registered SEI not implemented")
    def test_user_data_registered_afd(self):
        """Parse user_data_registered with AFD (Active Format Description)."""
        from parameters.sei import parse_user_data_registered_sei

        writer = BitWriter()
        writer.write_bits(0xB5, 8)  # country_code (US)
        writer.write_bits(0x31, 8)  # provider_code (ATSC)
        writer.write_bits(0x44544731, 32)  # "DTG1" identifier
        writer.write_bits(0x41, 8)  # AFD data

        data = writer.to_bytes()
        ud = parse_user_data_registered_sei(data, payload_size=len(data))

        assert ud.user_identifier == b'DTG1'
        assert ud.has_afd is True

    @pytest.mark.xfail(reason="user_data_registered SEI not implemented")
    def test_user_data_registered_cea608(self):
        """Parse user_data_registered with CEA-608 closed captions."""
        from parameters.sei import parse_user_data_registered_sei

        writer = BitWriter()
        writer.write_bits(0xB5, 8)
        writer.write_bits(0x31, 8)
        writer.write_bits(0x47413934, 32)  # GA94
        writer.write_bits(0x03, 8)  # user_data_type_code for CEA-608
        # cc_data
        writer.write_bits(0xC0 | 3, 8)  # process_cc_data_flag + cc_count
        for _ in range(3):
            writer.write_bits(0xFC, 8)  # cc_valid + cc_type
            writer.write_bits(0x80, 8)  # cc_data_1
            writer.write_bits(0x80, 8)  # cc_data_2

        data = writer.to_bytes()
        ud = parse_user_data_registered_sei(data, payload_size=len(data))

        assert ud.has_closed_captions is True
        assert len(ud.closed_caption_data) == 3


class TestUserDataUnregisteredSEI:
    """Test user_data_unregistered SEI message parsing.

    H.264 Spec: Section D.1.7, D.2.7
    """

    @pytest.mark.xfail(reason="user_data_unregistered SEI not implemented")
    def test_user_data_unregistered_x264(self):
        """Parse user_data_unregistered with x264 encoder info."""
        from parameters.sei import parse_user_data_unregistered_sei

        writer = BitWriter()
        # x264 UUID
        for b in UUID_X264.bytes:
            writer.write_bits(b, 8)
        # x264 encoder string
        encoder_info = b"x264 - core 164 r3095"
        for b in encoder_info:
            writer.write_bits(b, 8)

        data = writer.to_bytes()
        ud = parse_user_data_unregistered_sei(data, payload_size=len(data))

        assert ud.uuid == UUID_X264
        assert b"x264" in ud.user_data

    @pytest.mark.xfail(reason="user_data_unregistered SEI not implemented")
    def test_user_data_unregistered_custom_uuid(self):
        """Parse user_data_unregistered with custom UUID."""
        from parameters.sei import parse_user_data_unregistered_sei

        custom_uuid = uuid.uuid4()
        writer = BitWriter()
        for b in custom_uuid.bytes:
            writer.write_bits(b, 8)
        custom_data = b"Custom encoder data v1.0"
        for b in custom_data:
            writer.write_bits(b, 8)

        data = writer.to_bytes()
        ud = parse_user_data_unregistered_sei(data, payload_size=len(data))

        assert ud.uuid == custom_uuid
        assert ud.user_data == custom_data

    @pytest.mark.xfail(reason="user_data_unregistered SEI not implemented")
    def test_user_data_unregistered_empty_data(self):
        """Parse user_data_unregistered with only UUID (no user data)."""
        from parameters.sei import parse_user_data_unregistered_sei

        writer = BitWriter()
        for b in UUID_X264.bytes:
            writer.write_bits(b, 8)

        data = writer.to_bytes()
        ud = parse_user_data_unregistered_sei(data, payload_size=16)

        assert ud.uuid == UUID_X264
        assert ud.user_data == b''

    @pytest.mark.xfail(reason="user_data_unregistered SEI not implemented")
    def test_user_data_unregistered_ffmpeg(self):
        """Parse user_data_unregistered with FFmpeg Lavc info."""
        from parameters.sei import parse_user_data_unregistered_sei

        # FFmpeg Lavc UUID (common pattern)
        ffmpeg_uuid = uuid.UUID("17ee8c60-f84d-11d9-8cd6-0800200c9a66")
        writer = BitWriter()
        for b in ffmpeg_uuid.bytes:
            writer.write_bits(b, 8)
        encoder_info = b"Lavc60.3.100 libx264"
        for b in encoder_info:
            writer.write_bits(b, 8)

        data = writer.to_bytes()
        ud = parse_user_data_unregistered_sei(data, payload_size=len(data))

        assert b"Lavc" in ud.user_data or b"libx264" in ud.user_data


class TestRecoveryPointSEI:
    """Test recovery_point SEI message parsing.

    H.264 Spec: Section D.1.8, D.2.8
    """

    @pytest.mark.xfail(reason="recovery_point SEI not implemented")
    def test_recovery_point_immediate(self):
        """Parse recovery_point with immediate recovery."""
        from parameters.sei import parse_recovery_point_sei, SEIRecoveryPoint

        writer = BitWriter()
        writer.write_ue(0)  # recovery_frame_cnt = 0 (immediate)
        writer.write_flag(True)   # exact_match_flag
        writer.write_flag(False)  # broken_link_flag
        writer.write_bits(0, 2)   # changing_slice_group_idc

        data = writer.to_bytes()
        rp = parse_recovery_point_sei(data)

        assert isinstance(rp, SEIRecoveryPoint)
        assert rp.recovery_frame_cnt == 0
        assert rp.exact_match_flag is True
        assert rp.broken_link_flag is False
        assert rp.changing_slice_group_idc == 0

    @pytest.mark.xfail(reason="recovery_point SEI not implemented")
    def test_recovery_point_delayed(self):
        """Parse recovery_point with delayed recovery."""
        from parameters.sei import parse_recovery_point_sei

        writer = BitWriter()
        writer.write_ue(5)  # recovery_frame_cnt = 5
        writer.write_flag(False)  # exact_match_flag
        writer.write_flag(True)   # broken_link_flag
        writer.write_bits(1, 2)   # changing_slice_group_idc

        data = writer.to_bytes()
        rp = parse_recovery_point_sei(data)

        assert rp.recovery_frame_cnt == 5
        assert rp.exact_match_flag is False
        assert rp.broken_link_flag is True
        assert rp.changing_slice_group_idc == 1

    @pytest.mark.xfail(reason="recovery_point SEI not implemented")
    def test_recovery_point_for_random_access(self):
        """Recovery point used for random access (open GOP)."""
        from parameters.sei import parse_recovery_point_sei

        writer = BitWriter()
        writer.write_ue(3)  # Recovery after 3 frames
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_bits(0, 2)

        data = writer.to_bytes()
        rp = parse_recovery_point_sei(data)

        assert rp.recovery_frame_cnt == 3
        assert rp.is_random_access_point is True


class TestFillerPayloadSEI:
    """Test filler_payload SEI message parsing.

    H.264 Spec: Section D.1.5, D.2.5
    """

    @pytest.mark.xfail(reason="filler_payload SEI not implemented")
    def test_filler_payload_basic(self):
        """Parse basic filler_payload SEI."""
        from parameters.sei import parse_filler_payload_sei

        writer = BitWriter()
        for _ in range(16):
            writer.write_bits(0xFF, 8)  # All 0xFF bytes

        data = writer.to_bytes()
        fp = parse_filler_payload_sei(data, payload_size=16)

        assert fp.size == 16
        assert all(b == 0xFF for b in fp.data)

    @pytest.mark.xfail(reason="filler_payload SEI not implemented")
    def test_filler_payload_large(self):
        """Parse large filler_payload SEI."""
        from parameters.sei import parse_filler_payload_sei

        writer = BitWriter()
        for _ in range(1024):
            writer.write_bits(0xFF, 8)

        data = writer.to_bytes()
        fp = parse_filler_payload_sei(data, payload_size=1024)

        assert fp.size == 1024


class TestSEIModuleInterface:
    """Test SEI module interface and dataclasses."""

    @pytest.mark.xfail(reason="SEI module not implemented")
    def test_sei_message_dataclass(self):
        """SEIMessage dataclass has required fields."""
        from parameters.sei import SEIMessage

        sei = SEIMessage(
            payload_type=SEI_TYPE_PIC_TIMING,
            payload_size=4,
            payload_data=b'\x00\x00\x00\x00'
        )

        assert sei.payload_type == SEI_TYPE_PIC_TIMING
        assert sei.payload_size == 4
        assert sei.type_name == "pic_timing"

    @pytest.mark.xfail(reason="SEI module not implemented")
    def test_sei_type_name_lookup(self):
        """SEI type name lookup for known types."""
        from parameters.sei import get_sei_type_name

        assert get_sei_type_name(SEI_TYPE_BUFFERING_PERIOD) == "buffering_period"
        assert get_sei_type_name(SEI_TYPE_PIC_TIMING) == "pic_timing"
        assert get_sei_type_name(SEI_TYPE_USER_DATA_REGISTERED) == "user_data_registered_itu_t_t35"
        assert get_sei_type_name(SEI_TYPE_USER_DATA_UNREGISTERED) == "user_data_unregistered"
        assert get_sei_type_name(SEI_TYPE_RECOVERY_POINT) == "recovery_point"

    @pytest.mark.xfail(reason="SEI module not implemented")
    def test_sei_type_name_unknown(self):
        """SEI type name for unknown types."""
        from parameters.sei import get_sei_type_name

        assert "unknown" in get_sei_type_name(999).lower() or "999" in get_sei_type_name(999)


class TestSEIPayloadDispatch:
    """Test SEI payload type dispatch to specific parsers."""

    @pytest.mark.xfail(reason="SEI payload dispatch not implemented")
    def test_dispatch_buffering_period(self):
        """Dispatch to buffering_period parser."""
        from parameters.sei import parse_sei_payload

        data = b'\x00\x00\x00\x00\x00'  # Minimal buffering_period
        result = parse_sei_payload(SEI_TYPE_BUFFERING_PERIOD, data, context={})

        assert result.payload_type == SEI_TYPE_BUFFERING_PERIOD

    @pytest.mark.xfail(reason="SEI payload dispatch not implemented")
    def test_dispatch_pic_timing(self):
        """Dispatch to pic_timing parser."""
        from parameters.sei import parse_sei_payload

        data = b'\x00\x00\x00\x00'
        result = parse_sei_payload(SEI_TYPE_PIC_TIMING, data, context={
            'cpb_removal_delay_length': 16,
            'dpb_output_delay_length': 16,
            'pic_struct_present_flag': False
        })

        assert result.payload_type == SEI_TYPE_PIC_TIMING

    @pytest.mark.xfail(reason="SEI payload dispatch not implemented")
    def test_dispatch_recovery_point(self):
        """Dispatch to recovery_point parser."""
        from parameters.sei import parse_sei_payload

        writer = BitWriter()
        writer.write_ue(0)
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_bits(0, 2)

        result = parse_sei_payload(SEI_TYPE_RECOVERY_POINT, writer.to_bytes(), context={})

        assert result.recovery_frame_cnt == 0

    @pytest.mark.xfail(reason="SEI payload dispatch not implemented")
    def test_dispatch_unknown_type(self):
        """Unknown SEI type returns generic SEIMessage."""
        from parameters.sei import parse_sei_payload, SEIMessage

        data = b'\x01\x02\x03\x04'
        result = parse_sei_payload(999, data, context={})

        assert isinstance(result, SEIMessage)
        assert result.payload_type == 999
        assert result.payload_data == data


class TestSEIIntegrationWithNALParser:
    """Test SEI parsing integration with NAL unit parser."""

    @pytest.mark.xfail(reason="SEI NAL integration not implemented")
    def test_parse_sei_nal_unit(self):
        """Parse complete SEI NAL unit from bitstream."""
        from bitstream import NALUnitType, parse_nal_unit
        from parameters.sei import parse_sei_nal_unit

        writer = BitWriter()
        # NAL header for SEI (nal_ref_idc=0, nal_unit_type=6)
        writer.write_bits(0x06, 8)
        # SEI message
        writer.write_bits(SEI_TYPE_RECOVERY_POINT, 8)
        writer.write_bits(3, 8)  # payload_size
        writer.write_ue(0)
        writer.write_flag(True)
        writer.write_flag(False)
        writer.write_bits(0, 2)
        # RBSP trailing
        writer.write_bits(0x80, 8)

        data = writer.to_bytes()
        nal = parse_nal_unit(data)

        assert nal.nal_unit_type == NALUnitType.SEI

        messages = parse_sei_nal_unit(nal.rbsp)
        assert len(messages) >= 1

    @pytest.mark.xfail(reason="SEI NAL integration not implemented")
    def test_extract_sei_from_stream(self):
        """Extract all SEI messages from a bitstream."""
        from bitstream import extract_nal_units, NALUnitType
        from parameters.sei import extract_sei_messages

        # Create a simple bitstream with SPS, PPS, SEI, and slice
        # (This would need actual NAL unit construction)
        bitstream = b'\x00\x00\x00\x01'  # Start code placeholder

        nal_units = extract_nal_units(bitstream)
        sei_nals = [n for n in nal_units if n.nal_unit_type == NALUnitType.SEI]
        sei_messages = extract_sei_messages(sei_nals)

        # Would verify SEI messages are correctly extracted
        assert isinstance(sei_messages, list)


class TestSEIEdgeCases:
    """Test edge cases in SEI parsing."""

    @pytest.mark.xfail(reason="SEI edge case handling not implemented")
    def test_malformed_sei_too_short(self):
        """Handle malformed SEI with truncated data."""
        from parameters.sei import parse_sei_message

        # Payload size says 100 but only 5 bytes present
        writer = BitWriter()
        writer.write_bits(SEI_TYPE_FILLER_PAYLOAD, 8)
        writer.write_bits(100, 8)
        writer.write_bits(0, 40)  # Only 5 bytes

        data = writer.to_bytes()

        with pytest.raises(ValueError, match="truncated|insufficient"):
            parse_sei_message(data)

    @pytest.mark.xfail(reason="SEI edge case handling not implemented")
    def test_sei_with_emulation_prevention(self):
        """Handle SEI payload with emulation prevention bytes."""
        from parameters.sei import parse_sei_nal_unit

        # SEI data that contains 0x000003 sequence
        writer = BitWriter()
        writer.write_bits(SEI_TYPE_USER_DATA_UNREGISTERED, 8)
        writer.write_bits(20, 8)
        for b in UUID_X264.bytes:
            writer.write_bits(b, 8)
        # Data that would need emulation prevention
        writer.write_bits(0, 8)
        writer.write_bits(0, 8)
        writer.write_bits(3, 8)  # Would be 0x000003 in raw form
        writer.write_bits(0, 8)
        writer.write_bits(0x80, 8)

        data = writer.to_bytes()
        # Parser should handle this correctly after emulation prevention removal
        messages = parse_sei_nal_unit(data)

        assert len(messages) >= 1

    @pytest.mark.xfail(reason="SEI edge case handling not implemented")
    def test_very_large_sei_payload(self):
        """Handle very large SEI payload (multi-byte size)."""
        from parameters.sei import parse_sei_message

        writer = BitWriter()
        writer.write_bits(SEI_TYPE_FILLER_PAYLOAD, 8)
        # Size 1000 = 255 + 255 + 255 + 235 (four bytes)
        writer.write_bits(255, 8)
        writer.write_bits(255, 8)
        writer.write_bits(255, 8)
        writer.write_bits(235, 8)
        for _ in range(1000):
            writer.write_bits(0xFF, 8)

        data = writer.to_bytes()
        sei = parse_sei_message(data)

        assert sei.payload_size == 1000
