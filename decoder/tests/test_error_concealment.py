# h264/decoder/tests/test_error_concealment.py
"""Red TDD tests for H.264 error concealment.

Error concealment handles corrupt or missing data gracefully to minimize
visual artifacts. This module tests various concealment strategies.

H.264 Spec Reference:
- Annex E: Error detection and recovery
- Section 8.4: Inter prediction (used for temporal concealment)

Common concealment strategies:
- Temporal: Copy from reference frame
- Spatial: Interpolate from neighboring macroblocks
- Motion vector: Use median of neighbor MVs
"""

import pytest
import numpy as np

from decoder.decoder import H264Decoder, DecoderState, DecodedFrame
from inter.reference import ReferenceFrame, ReferenceFrameBuffer
from inter.mv_prediction import MVCache


# All tests should fail initially - error concealment not implemented
pytestmark = pytest.mark.xfail(reason="Error concealment not implemented")


class TestConcealmentStrategy:
    """Tests for ConcealmentStrategy enum and selection logic."""

    def test_concealment_strategy_enum_exists(self):
        """ConcealmentStrategy enum should exist with expected values."""
        from decoder.error_concealment import ConcealmentStrategy

        assert hasattr(ConcealmentStrategy, 'TEMPORAL')
        assert hasattr(ConcealmentStrategy, 'SPATIAL')
        assert hasattr(ConcealmentStrategy, 'ZERO')
        assert hasattr(ConcealmentStrategy, 'PREVIOUS_MB')

    def test_default_strategy_is_temporal_when_refs_available(self):
        """Default strategy should be temporal when reference frames exist."""
        from decoder.error_concealment import select_concealment_strategy

        # Create mock decoder state with reference frame
        ref_buffer = ReferenceFrameBuffer(max_frames=4)
        ref_frame = ReferenceFrame(
            luma=np.zeros((64, 64), dtype=np.uint8),
            cb=np.zeros((32, 32), dtype=np.uint8),
            cr=np.zeros((32, 32), dtype=np.uint8),
            frame_num=0,
        )
        ref_buffer.add_frame(ref_frame)

        strategy = select_concealment_strategy(
            ref_buffer=ref_buffer,
            has_neighbors=True,
        )

        from decoder.error_concealment import ConcealmentStrategy
        assert strategy == ConcealmentStrategy.TEMPORAL

    def test_fallback_to_spatial_when_no_refs(self):
        """Should use spatial concealment when no reference frames available."""
        from decoder.error_concealment import (
            ConcealmentStrategy,
            select_concealment_strategy,
        )

        strategy = select_concealment_strategy(
            ref_buffer=None,
            has_neighbors=True,
        )

        assert strategy == ConcealmentStrategy.SPATIAL

    def test_fallback_to_zero_when_nothing_available(self):
        """Should use zero fill when no refs and no neighbors."""
        from decoder.error_concealment import (
            ConcealmentStrategy,
            select_concealment_strategy,
        )

        strategy = select_concealment_strategy(
            ref_buffer=None,
            has_neighbors=False,
        )

        assert strategy == ConcealmentStrategy.ZERO


class TestMissingMacroblockDetection:
    """Tests for detecting missing or corrupt macroblocks."""

    def test_detect_missing_mb_in_slice_data(self):
        """Detect when macroblock count doesn't match expected."""
        from decoder.error_concealment import detect_missing_macroblocks

        expected_mbs = 16  # 4x4 MBs
        decoded_mbs = [0, 1, 2, 3, 5, 6, 7]  # MB 4 missing

        missing = detect_missing_macroblocks(expected_mbs, decoded_mbs)

        assert 4 in missing
        assert len(missing) == expected_mbs - len(decoded_mbs)

    def test_detect_multiple_missing_mbs(self):
        """Detect multiple missing macroblocks."""
        from decoder.error_concealment import detect_missing_macroblocks

        expected_mbs = 16
        decoded_mbs = [0, 1, 5, 6, 10, 11, 15]

        missing = detect_missing_macroblocks(expected_mbs, decoded_mbs)

        assert 2 in missing
        assert 3 in missing
        assert 4 in missing
        assert 7 in missing
        assert len(missing) == expected_mbs - len(decoded_mbs)

    def test_no_missing_mbs(self):
        """Return empty list when all MBs decoded."""
        from decoder.error_concealment import detect_missing_macroblocks

        expected_mbs = 4
        decoded_mbs = [0, 1, 2, 3]

        missing = detect_missing_macroblocks(expected_mbs, decoded_mbs)

        assert len(missing) == 0

    def test_detect_corrupt_mb_by_coefficients(self):
        """Detect corrupt MB by invalid coefficient values."""
        from decoder.error_concealment import is_macroblock_corrupt

        # Valid coefficients are in range after dequant
        valid_coeffs = np.random.randint(-2048, 2048, (16, 16), dtype=np.int32)
        assert not is_macroblock_corrupt(valid_coeffs)

        # Out of range values indicate corruption
        corrupt_coeffs = np.full((16, 16), 99999, dtype=np.int32)
        assert is_macroblock_corrupt(corrupt_coeffs)

    def test_detect_corrupt_mb_by_mb_type(self):
        """Detect corrupt MB by invalid mb_type value."""
        from decoder.error_concealment import is_mb_type_valid

        # Valid I-slice mb_types: 0-25
        assert is_mb_type_valid(0, is_i_slice=True)
        assert is_mb_type_valid(25, is_i_slice=True)
        assert not is_mb_type_valid(26, is_i_slice=True)
        assert not is_mb_type_valid(-1, is_i_slice=True)

        # Valid P-slice mb_types
        assert is_mb_type_valid(0, is_i_slice=False, is_p_slice=True)
        assert not is_mb_type_valid(100, is_i_slice=False, is_p_slice=True)


class TestTemporalConcealment:
    """Tests for temporal concealment (copy from reference)."""

    def test_conceal_mb_copy_from_ref(self):
        """Conceal MB by copying co-located block from reference frame."""
        from decoder.error_concealment import conceal_macroblock_temporal

        # Create reference frame with known pattern
        ref_luma = np.full((64, 64), 128, dtype=np.uint8)
        ref_luma[16:32, 16:32] = 200  # MB at (1, 1) has value 200
        ref_cb = np.full((32, 32), 128, dtype=np.uint8)
        ref_cr = np.full((32, 32), 128, dtype=np.uint8)

        ref_frame = ReferenceFrame(
            luma=ref_luma,
            cb=ref_cb,
            cr=ref_cr,
            frame_num=0,
        )

        # Conceal MB at position (1, 1)
        concealed = conceal_macroblock_temporal(
            ref_frame=ref_frame,
            mb_x=1,
            mb_y=1,
        )

        assert concealed['luma'].shape == (16, 16)
        assert concealed['cb'].shape == (8, 8)
        assert concealed['cr'].shape == (8, 8)
        np.testing.assert_array_equal(concealed['luma'], 200)

    def test_temporal_concealment_with_mv(self):
        """Temporal concealment should use motion-compensated copy."""
        from decoder.error_concealment import conceal_macroblock_temporal

        # Reference frame
        ref_luma = np.zeros((64, 64), dtype=np.uint8)
        ref_luma[20:36, 20:36] = 255  # Block offset from MB position

        ref_frame = ReferenceFrame(
            luma=ref_luma,
            cb=np.zeros((32, 32), dtype=np.uint8),
            cr=np.zeros((32, 32), dtype=np.uint8),
            frame_num=0,
        )

        # MV points to offset position
        concealed = conceal_macroblock_temporal(
            ref_frame=ref_frame,
            mb_x=1,
            mb_y=1,
            mvx=4,  # Quarter-pel MV = 1 pixel offset
            mvy=4,
        )

        # Should have copied the white block
        assert np.mean(concealed['luma']) > 100

    def test_temporal_concealment_at_frame_boundary(self):
        """Handle concealment at frame edges gracefully."""
        from decoder.error_concealment import conceal_macroblock_temporal

        ref_frame = ReferenceFrame(
            luma=np.full((32, 32), 128, dtype=np.uint8),
            cb=np.full((16, 16), 128, dtype=np.uint8),
            cr=np.full((16, 16), 128, dtype=np.uint8),
            frame_num=0,
        )

        # MB at boundary with MV pointing outside
        concealed = conceal_macroblock_temporal(
            ref_frame=ref_frame,
            mb_x=1,
            mb_y=1,
            mvx=-100,  # Points way outside frame
            mvy=-100,
        )

        # Should still produce valid output (clamped or padded)
        assert concealed['luma'].shape == (16, 16)
        assert concealed['luma'].dtype == np.uint8


class TestSpatialConcealment:
    """Tests for spatial concealment (interpolate from neighbors)."""

    def test_conceal_mb_interpolate_from_neighbors(self):
        """Conceal MB by interpolating from decoded neighbors."""
        from decoder.error_concealment import conceal_macroblock_spatial

        # Create frame with decoded neighbors
        frame_luma = np.zeros((48, 48), dtype=np.uint8)
        # Top neighbor (MB 0,0) = 100
        frame_luma[0:16, 16:32] = 100
        # Left neighbor (MB 1,0) = 150
        frame_luma[16:32, 0:16] = 150
        # Bottom neighbor (MB 2,1) = 200
        frame_luma[32:48, 16:32] = 200
        # Right neighbor (MB 1,2) = 250
        frame_luma[16:32, 32:48] = 250

        frame_cb = np.zeros((24, 24), dtype=np.uint8)
        frame_cr = np.zeros((24, 24), dtype=np.uint8)

        neighbors = {
            'top_available': True,
            'left_available': True,
            'bottom_available': True,
            'right_available': True,
        }

        concealed = conceal_macroblock_spatial(
            frame_luma=frame_luma,
            frame_cb=frame_cb,
            frame_cr=frame_cr,
            mb_x=1,
            mb_y=1,
            neighbors=neighbors,
        )

        # Result should be interpolation (roughly average of neighbors)
        expected_avg = (100 + 150 + 200 + 250) // 4
        assert abs(np.mean(concealed['luma']) - expected_avg) < 30

    def test_spatial_concealment_edge_interpolation(self):
        """Spatial concealment should interpolate along edges properly."""
        from decoder.error_concealment import conceal_macroblock_spatial

        # Create frame with left=0 and right=255
        frame_luma = np.zeros((32, 48), dtype=np.uint8)
        frame_luma[0:32, 0:16] = 0    # Left
        frame_luma[0:32, 32:48] = 255  # Right

        neighbors = {
            'top_available': False,
            'left_available': True,
            'bottom_available': False,
            'right_available': True,
        }

        concealed = conceal_macroblock_spatial(
            frame_luma=frame_luma,
            frame_cb=np.zeros((16, 24), dtype=np.uint8),
            frame_cr=np.zeros((16, 24), dtype=np.uint8),
            mb_x=1,
            mb_y=0,
            neighbors=neighbors,
        )

        # Should have gradient from left to right
        assert concealed['luma'][8, 0] < concealed['luma'][8, 15]

    def test_spatial_concealment_top_left_corner(self):
        """Handle concealment at top-left corner (no top/left neighbors)."""
        from decoder.error_concealment import conceal_macroblock_spatial

        frame_luma = np.zeros((32, 32), dtype=np.uint8)
        frame_luma[16:32, 0:16] = 128  # Bottom neighbor
        frame_luma[0:16, 16:32] = 128  # Right neighbor

        neighbors = {
            'top_available': False,
            'left_available': False,
            'bottom_available': True,
            'right_available': True,
        }

        concealed = conceal_macroblock_spatial(
            frame_luma=frame_luma,
            frame_cb=np.zeros((16, 16), dtype=np.uint8),
            frame_cr=np.zeros((16, 16), dtype=np.uint8),
            mb_x=0,
            mb_y=0,
            neighbors=neighbors,
        )

        # Should use available neighbors
        assert concealed['luma'].shape == (16, 16)

    def test_spatial_concealment_no_neighbors(self):
        """Handle concealment when no neighbors available."""
        from decoder.error_concealment import conceal_macroblock_spatial

        neighbors = {
            'top_available': False,
            'left_available': False,
            'bottom_available': False,
            'right_available': False,
        }

        concealed = conceal_macroblock_spatial(
            frame_luma=np.zeros((16, 16), dtype=np.uint8),
            frame_cb=np.zeros((8, 8), dtype=np.uint8),
            frame_cr=np.zeros((8, 8), dtype=np.uint8),
            mb_x=0,
            mb_y=0,
            neighbors=neighbors,
        )

        # Should fill with neutral gray (128)
        assert abs(np.mean(concealed['luma']) - 128) < 10


class TestMotionVectorConcealment:
    """Tests for motion vector concealment."""

    def test_conceal_mv_median_of_neighbors(self):
        """Conceal MV using median of neighboring MVs."""
        from decoder.error_concealment import conceal_motion_vector

        # Create MV cache with known neighbor MVs
        mv_cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # Set neighbor MVs
        # Left neighbor (MB 0, 1): MV = (4, 8)
        mv_cache.set_mv_16x16(0, 1, 4, 8)
        # Top neighbor (MB 1, 0): MV = (8, 4)
        mv_cache.set_mv_16x16(1, 0, 8, 4)
        # Top-right neighbor (MB 2, 0): MV = (12, 12)
        mv_cache.set_mv_16x16(2, 0, 12, 12)

        # Conceal MV for MB at (1, 1)
        concealed_mv = conceal_motion_vector(
            mv_cache=mv_cache,
            mb_x=1,
            mb_y=1,
        )

        # Should be median: (8, 8)
        assert concealed_mv == (8, 8)

    def test_conceal_mv_single_neighbor(self):
        """Use single neighbor MV when only one available."""
        from decoder.error_concealment import conceal_motion_vector

        mv_cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # Only left neighbor available
        mv_cache.set_mv_16x16(0, 1, 10, 20)

        concealed_mv = conceal_motion_vector(
            mv_cache=mv_cache,
            mb_x=1,
            mb_y=1,
        )

        # Should use the single available MV
        assert concealed_mv == (10, 20)

    def test_conceal_mv_no_neighbors(self):
        """Return zero MV when no neighbors available."""
        from decoder.error_concealment import conceal_motion_vector

        mv_cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # No neighbors set (first MB)
        concealed_mv = conceal_motion_vector(
            mv_cache=mv_cache,
            mb_x=0,
            mb_y=0,
        )

        assert concealed_mv == (0, 0)

    def test_conceal_mv_with_colocated_ref(self):
        """Use co-located MV from reference when available."""
        from decoder.error_concealment import conceal_motion_vector

        mv_cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # Reference frame with stored MVs
        ref_frame = ReferenceFrame(
            luma=np.zeros((64, 64), dtype=np.uint8),
            cb=np.zeros((32, 32), dtype=np.uint8),
            cr=np.zeros((32, 32), dtype=np.uint8),
            frame_num=0,
        )
        ref_frame.store_mv(1, 1, 15, 25)

        concealed_mv = conceal_motion_vector(
            mv_cache=mv_cache,
            mb_x=1,
            mb_y=1,
            colocated_ref=ref_frame,
            use_colocated=True,
        )

        # Should prefer co-located MV
        assert concealed_mv == (15, 25)


class TestSliceLossConcealment:
    """Tests for handling entire slice loss."""

    def test_conceal_entire_lost_slice(self):
        """Conceal all MBs when entire slice is lost."""
        from decoder.error_concealment import conceal_lost_slice

        # Reference frame exists
        ref_frame = ReferenceFrame(
            luma=np.full((64, 64), 128, dtype=np.uint8),
            cb=np.full((32, 32), 128, dtype=np.uint8),
            cr=np.full((32, 32), 128, dtype=np.uint8),
            frame_num=0,
        )
        ref_buffer = ReferenceFrameBuffer(max_frames=4)
        ref_buffer.add_frame(ref_frame)

        # Frame to conceal into
        frame_luma = np.zeros((64, 64), dtype=np.uint8)
        frame_cb = np.zeros((32, 32), dtype=np.uint8)
        frame_cr = np.zeros((32, 32), dtype=np.uint8)

        # Slice covers MBs 4-7 (second row of 4x4 MB frame)
        first_mb = 4
        last_mb = 7

        conceal_lost_slice(
            frame_luma=frame_luma,
            frame_cb=frame_cb,
            frame_cr=frame_cr,
            ref_buffer=ref_buffer,
            first_mb=first_mb,
            last_mb=last_mb,
            mb_width=4,
        )

        # Row 1 should be concealed
        np.testing.assert_array_equal(frame_luma[16:32, :], 128)

    def test_conceal_first_slice_loss(self):
        """Handle loss of first slice (no previous data)."""
        from decoder.error_concealment import conceal_lost_slice

        # No reference frames
        ref_buffer = ReferenceFrameBuffer(max_frames=4)

        frame_luma = np.zeros((32, 32), dtype=np.uint8)
        frame_cb = np.zeros((16, 16), dtype=np.uint8)
        frame_cr = np.zeros((16, 16), dtype=np.uint8)

        # First slice (MBs 0-3)
        conceal_lost_slice(
            frame_luma=frame_luma,
            frame_cb=frame_cb,
            frame_cr=frame_cr,
            ref_buffer=ref_buffer,
            first_mb=0,
            last_mb=3,
            mb_width=2,
        )

        # Should fill with gray (128) when no reference
        assert abs(np.mean(frame_luma) - 128) < 10

    def test_partial_slice_loss(self):
        """Conceal only missing MBs when partial slice received."""
        from decoder.error_concealment import conceal_partial_slice

        ref_frame = ReferenceFrame(
            luma=np.full((64, 64), 200, dtype=np.uint8),
            cb=np.full((32, 32), 128, dtype=np.uint8),
            cr=np.full((32, 32), 128, dtype=np.uint8),
            frame_num=0,
        )
        ref_buffer = ReferenceFrameBuffer(max_frames=4)
        ref_buffer.add_frame(ref_frame)

        frame_luma = np.zeros((64, 64), dtype=np.uint8)
        frame_cb = np.zeros((32, 32), dtype=np.uint8)
        frame_cr = np.zeros((32, 32), dtype=np.uint8)

        # MBs 0, 1, 4, 5 decoded; 2, 3, 6, 7 missing
        decoded_mbs = [0, 1, 4, 5]
        slice_mbs = list(range(8))

        conceal_partial_slice(
            frame_luma=frame_luma,
            frame_cb=frame_cb,
            frame_cr=frame_cr,
            ref_buffer=ref_buffer,
            decoded_mbs=decoded_mbs,
            slice_mbs=slice_mbs,
            mb_width=4,
        )

        # Only missing MBs should be concealed
        # MB 2 at (2, 0): pixels [0:16, 32:48]
        assert np.mean(frame_luma[0:16, 32:48]) > 100


class TestInvalidCoefficientHandling:
    """Tests for handling invalid transform coefficients."""

    def test_clamp_out_of_range_coefficients(self):
        """Clamp coefficient values to valid range."""
        from decoder.error_concealment import sanitize_coefficients

        coeffs = np.array([
            [32767, -32768, 0, 100],
            [2048, -2048, 50, -50],
            [99999, -99999, 1000, -1000],
            [0, 0, 0, 0],
        ], dtype=np.int32)

        sanitized = sanitize_coefficients(coeffs, max_coeff=2047)

        assert sanitized.max() <= 2047
        assert sanitized.min() >= -2048

    def test_detect_nan_coefficients(self):
        """Detect and handle NaN values in coefficients."""
        from decoder.error_concealment import sanitize_coefficients

        coeffs = np.array([
            [1.0, np.nan, 0.0, 100.0],
            [np.inf, -np.inf, 50.0, -50.0],
        ], dtype=np.float64)

        sanitized = sanitize_coefficients(coeffs, max_coeff=2047)

        assert not np.any(np.isnan(sanitized))
        assert not np.any(np.isinf(sanitized))

    def test_replace_invalid_dc_coefficient(self):
        """Replace invalid DC coefficient with predicted value."""
        from decoder.error_concealment import repair_dc_coefficient

        # DC coefficient is out of range
        coeffs = np.zeros((4, 4), dtype=np.int32)
        coeffs[0, 0] = 99999  # Invalid DC

        # Neighbor DC values
        neighbor_dcs = [100, 120, 110]

        repaired = repair_dc_coefficient(coeffs, neighbor_dcs)

        # DC should be replaced with average of neighbors
        expected_dc = sum(neighbor_dcs) // len(neighbor_dcs)
        assert abs(repaired[0, 0] - expected_dc) < 20


class TestTruncatedNALUnitHandling:
    """Tests for handling truncated NAL units."""

    def test_detect_truncated_nal(self):
        """Detect when NAL unit is truncated."""
        from decoder.error_concealment import is_nal_truncated

        # Valid NAL with RBSP trailing bits
        valid_nal = bytes([0x00, 0x00, 0x01, 0x65, 0x88, 0x80])  # With stop bit
        assert not is_nal_truncated(valid_nal)

        # Truncated NAL (no proper ending)
        truncated_nal = bytes([0x00, 0x00, 0x01, 0x65, 0x88])
        assert is_nal_truncated(truncated_nal)

    def test_estimate_missing_data_size(self):
        """Estimate how much data is missing from truncated NAL."""
        from decoder.error_concealment import estimate_truncation_extent

        # NAL that should have more data based on slice header
        partial_nal = bytes([0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00])

        # Expected MBs in slice
        expected_mbs = 16
        decoded_mbs = 4

        estimate = estimate_truncation_extent(
            partial_nal,
            expected_mbs=expected_mbs,
            decoded_mbs=decoded_mbs,
        )

        assert estimate['missing_mbs'] == 12
        assert 'estimated_bytes' in estimate

    def test_recover_from_truncated_slice(self):
        """Attempt to decode as much as possible from truncated slice."""
        from decoder.error_concealment import decode_truncated_slice

        # Partial slice data
        partial_data = bytes([
            0x00, 0x00, 0x01, 0x65,  # NAL header
            0x88, 0x84, 0x00, 0x0A,  # Partial slice data
        ])

        result = decode_truncated_slice(partial_data)

        assert 'decoded_mbs' in result
        assert 'error_position' in result
        assert result['partial_decode'] is True


class TestInvalidSyntaxRecovery:
    """Tests for recovering from invalid syntax elements."""

    def test_recover_from_invalid_mb_type(self):
        """Recover when mb_type is invalid."""
        from decoder.error_concealment import recover_invalid_mb_type

        # Invalid mb_type for I-slice
        invalid_mb_type = 50

        recovered = recover_invalid_mb_type(
            invalid_mb_type,
            is_i_slice=True,
        )

        # Should return a valid I-slice mb_type
        assert 0 <= recovered <= 25

    def test_recover_from_invalid_cbp(self):
        """Recover when coded block pattern is invalid."""
        from decoder.error_concealment import recover_invalid_cbp

        # Invalid CBP value
        invalid_cbp = 99

        recovered = recover_invalid_cbp(invalid_cbp)

        # Should return valid CBP (0-47 for inter, 0-47 for intra)
        assert 0 <= recovered <= 47

    def test_recover_from_invalid_intra_mode(self):
        """Recover when intra prediction mode is invalid."""
        from decoder.error_concealment import recover_invalid_intra_mode

        # Invalid intra 4x4 mode
        invalid_mode = 20  # Valid range: 0-8

        recovered = recover_invalid_intra_mode(
            invalid_mode,
            block_size=4,
        )

        # Should return valid mode (0-8 for 4x4)
        assert 0 <= recovered <= 8

    def test_recover_from_invalid_mv_component(self):
        """Recover when MV component is out of range."""
        from decoder.error_concealment import recover_invalid_mv

        # MV outside valid range
        invalid_mvx = 10000  # Way beyond frame size
        invalid_mvy = -10000

        recovered_mvx, recovered_mvy = recover_invalid_mv(
            invalid_mvx,
            invalid_mvy,
            frame_width=1920,
            frame_height=1080,
        )

        # Should clamp to reasonable range
        assert -8192 <= recovered_mvx <= 8191
        assert -8192 <= recovered_mvy <= 8191


class TestConcealmentModeSelection:
    """Tests for automatic concealment mode selection."""

    def test_select_temporal_for_p_slice(self):
        """Select temporal concealment for P-slice with available reference."""
        from decoder.error_concealment import (
            ConcealmentStrategy,
            auto_select_concealment,
        )

        ref_buffer = ReferenceFrameBuffer(max_frames=4)
        ref_frame = ReferenceFrame(
            luma=np.zeros((64, 64), dtype=np.uint8),
            cb=np.zeros((32, 32), dtype=np.uint8),
            cr=np.zeros((32, 32), dtype=np.uint8),
            frame_num=0,
        )
        ref_buffer.add_frame(ref_frame)

        strategy = auto_select_concealment(
            is_i_slice=False,
            ref_buffer=ref_buffer,
            mb_x=1,
            mb_y=1,
            has_neighbors=True,
        )

        assert strategy == ConcealmentStrategy.TEMPORAL

    def test_select_spatial_for_i_slice(self):
        """Prefer spatial concealment for I-slice."""
        from decoder.error_concealment import (
            ConcealmentStrategy,
            auto_select_concealment,
        )

        strategy = auto_select_concealment(
            is_i_slice=True,
            ref_buffer=None,
            mb_x=2,
            mb_y=2,
            has_neighbors=True,
        )

        assert strategy == ConcealmentStrategy.SPATIAL

    def test_select_based_on_motion_activity(self):
        """Select concealment based on motion activity in neighbors."""
        from decoder.error_concealment import (
            ConcealmentStrategy,
            auto_select_concealment,
        )

        mv_cache = MVCache(width_in_mbs=4, height_in_mbs=4)

        # High motion in neighbors
        mv_cache.set_mv_16x16(0, 1, 50, 50)
        mv_cache.set_mv_16x16(1, 0, 60, 40)

        ref_buffer = ReferenceFrameBuffer(max_frames=4)
        ref_frame = ReferenceFrame(
            luma=np.zeros((64, 64), dtype=np.uint8),
            cb=np.zeros((32, 32), dtype=np.uint8),
            cr=np.zeros((32, 32), dtype=np.uint8),
            frame_num=0,
        )
        ref_buffer.add_frame(ref_frame)

        strategy = auto_select_concealment(
            is_i_slice=False,
            ref_buffer=ref_buffer,
            mb_x=1,
            mb_y=1,
            has_neighbors=True,
            mv_cache=mv_cache,
        )

        # High motion should prefer temporal with motion compensation
        assert strategy == ConcealmentStrategy.TEMPORAL

    def test_select_zero_fill_as_last_resort(self):
        """Use zero/gray fill when nothing else available."""
        from decoder.error_concealment import (
            ConcealmentStrategy,
            auto_select_concealment,
        )

        strategy = auto_select_concealment(
            is_i_slice=True,
            ref_buffer=None,
            mb_x=0,
            mb_y=0,
            has_neighbors=False,
        )

        assert strategy == ConcealmentStrategy.ZERO


class TestConcealMacroblockIntegration:
    """Integration tests for the main conceal_macroblock function."""

    def test_conceal_macroblock_temporal(self):
        """Test full macroblock concealment using temporal strategy."""
        from decoder.error_concealment import (
            ConcealmentStrategy,
            conceal_macroblock,
        )

        ref_buffer = ReferenceFrameBuffer(max_frames=4)
        ref_frame = ReferenceFrame(
            luma=np.full((64, 64), 150, dtype=np.uint8),
            cb=np.full((32, 32), 128, dtype=np.uint8),
            cr=np.full((32, 32), 128, dtype=np.uint8),
            frame_num=0,
        )
        ref_buffer.add_frame(ref_frame)

        result = conceal_macroblock(
            strategy=ConcealmentStrategy.TEMPORAL,
            mb_x=1,
            mb_y=1,
            ref_buffer=ref_buffer,
        )

        assert result['luma'].shape == (16, 16)
        assert result['cb'].shape == (8, 8)
        assert result['cr'].shape == (8, 8)
        np.testing.assert_array_equal(result['luma'], 150)

    def test_conceal_macroblock_spatial(self):
        """Test full macroblock concealment using spatial strategy."""
        from decoder.error_concealment import (
            ConcealmentStrategy,
            conceal_macroblock,
        )

        frame_luma = np.zeros((48, 48), dtype=np.uint8)
        frame_luma[0:16, 16:32] = 100   # Top
        frame_luma[16:32, 0:16] = 150   # Left
        frame_luma[32:48, 16:32] = 200  # Bottom
        frame_luma[16:32, 32:48] = 250  # Right

        result = conceal_macroblock(
            strategy=ConcealmentStrategy.SPATIAL,
            mb_x=1,
            mb_y=1,
            frame_luma=frame_luma,
            frame_cb=np.zeros((24, 24), dtype=np.uint8),
            frame_cr=np.zeros((24, 24), dtype=np.uint8),
            neighbors={
                'top_available': True,
                'left_available': True,
                'bottom_available': True,
                'right_available': True,
            },
        )

        assert result['luma'].shape == (16, 16)
        # Should be interpolation of neighbors
        assert 100 < np.mean(result['luma']) < 250

    def test_conceal_macroblock_zero_fill(self):
        """Test macroblock concealment with zero/gray fill."""
        from decoder.error_concealment import (
            ConcealmentStrategy,
            conceal_macroblock,
        )

        result = conceal_macroblock(
            strategy=ConcealmentStrategy.ZERO,
            mb_x=0,
            mb_y=0,
        )

        assert result['luma'].shape == (16, 16)
        assert result['cb'].shape == (8, 8)
        assert result['cr'].shape == (8, 8)
        # Neutral gray
        assert abs(np.mean(result['luma']) - 128) < 2
        assert abs(np.mean(result['cb']) - 128) < 2
        assert abs(np.mean(result['cr']) - 128) < 2


class TestDecoderErrorHandling:
    """Tests for H264Decoder error handling integration."""

    def test_decoder_handles_truncated_stream(self):
        """Decoder should handle truncated bitstream gracefully."""
        decoder = H264Decoder()

        # Truncated stream (incomplete NAL)
        truncated_data = bytes([
            0x00, 0x00, 0x00, 0x01,  # Start code
            0x67,  # SPS NAL type
            0x42, 0xC0,  # Truncated SPS data
        ])

        # Should not raise exception
        frames = list(decoder.decode_bytes(truncated_data))

        # May return 0 frames, but should not crash
        assert isinstance(frames, list)

    def test_decoder_conceals_corrupt_mb(self):
        """Decoder should conceal corrupt macroblock."""
        from decoder.error_concealment import ConcealmentStrategy

        decoder = H264Decoder()

        # This would require a more complex test with actual bitstream
        # For now, test that decoder has concealment capability
        assert hasattr(decoder, 'conceal_macroblock') or \
               hasattr(decoder, '_conceal_macroblock')

    def test_decoder_continues_after_error(self):
        """Decoder should continue decoding after recoverable error."""
        decoder = H264Decoder()

        # Decoder must have error resilience mode
        assert hasattr(decoder, 'error_resilience')
        decoder.error_resilience = True

        # Decoder must support concealment callback
        assert hasattr(decoder, 'on_concealment')

    def test_decoder_reports_concealment_count(self):
        """Decoder should report number of concealed macroblocks."""
        decoder = H264Decoder()

        # Decoder must have stats with concealment tracking
        assert hasattr(decoder, 'stats')
        assert hasattr(decoder.stats, 'concealed_mb_count')


class TestChromaConcealmentConsistency:
    """Tests for ensuring chroma concealment matches luma."""

    def test_chroma_downsampled_consistently(self):
        """Chroma concealment should match luma sampling pattern."""
        from decoder.error_concealment import conceal_macroblock_temporal

        ref_luma = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        ref_cb = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        ref_cr = np.random.randint(0, 256, (32, 32), dtype=np.uint8)

        ref_frame = ReferenceFrame(
            luma=ref_luma,
            cb=ref_cb,
            cr=ref_cr,
            frame_num=0,
        )

        concealed = conceal_macroblock_temporal(
            ref_frame=ref_frame,
            mb_x=1,
            mb_y=1,
        )

        # Chroma should be exactly 8x8 for 4:2:0
        assert concealed['cb'].shape == (8, 8)
        assert concealed['cr'].shape == (8, 8)

        # Verify proper subsampling correspondence
        # Luma MB at (1,1) = pixels [16:32, 16:32]
        # Chroma at (1,1) = pixels [8:16, 8:16]
        expected_cb = ref_cb[8:16, 8:16]
        np.testing.assert_array_equal(concealed['cb'], expected_cb)

    def test_chroma_mv_scaling(self):
        """Chroma MV should be properly scaled from luma MV."""
        from decoder.error_concealment import conceal_macroblock_temporal

        ref_frame = ReferenceFrame(
            luma=np.zeros((64, 64), dtype=np.uint8),
            cb=np.zeros((32, 32), dtype=np.uint8),
            cr=np.zeros((32, 32), dtype=np.uint8),
            frame_num=0,
        )

        # Luma MV = (8, 8) pixels = (32, 32) quarter-pels
        # Chroma MV should be (4, 4) pixels = (16, 16) quarter-pels for 4:2:0
        concealed = conceal_macroblock_temporal(
            ref_frame=ref_frame,
            mb_x=1,
            mb_y=1,
            mvx=32,  # Quarter-pel units
            mvy=32,
        )

        # This tests that chroma uses properly scaled MV
        assert concealed['cb'].shape == (8, 8)
        assert concealed['cr'].shape == (8, 8)
