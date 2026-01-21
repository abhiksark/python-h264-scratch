# h264/decoder/tests/test_mbaff.py
"""RED TESTS: MBAFF (Macroblock-Adaptive Frame-Field) decoder tests.

H.264 Spec Reference:
- Section 7.3.4 - Slice data syntax (mb_field_decoding_flag)
- Section 7.4.4 - Slice data semantics
- Section 8.4.1.2 - Motion vectors in MBAFF
- Section 8.4.1.3 - Derivation of motion vectors for MBAFF
- Section 8.7.2.2 - Deblocking filter in MBAFF

MBAFF (Macroblock-Adaptive Frame-Field) allows mixing frame and field
macroblocks within the same picture:
- Picture is divided into macroblock pairs (vertically adjacent MBs)
- Each pair can be coded as frame or field mode
- mb_field_decoding_flag signals the coding mode for each pair
- Affects motion vector prediction, neighbor derivation, and deblocking

These tests SHOULD FAIL until MBAFF support is implemented.
"""

import pytest
import numpy as np

from bitstream import BitWriter, BitReader, BITSTRING_AVAILABLE

pytestmark = [
    pytest.mark.skipif(
        not BITSTRING_AVAILABLE,
        reason="bitstring library not installed"
    ),
    pytest.mark.xfail(reason="MBAFF not implemented"),
]


class TestMbFieldDecodingFlagParsing:
    """Tests for mb_field_decoding_flag parsing.

    H.264 Spec Section 7.3.4: mb_field_decoding_flag is present when:
    - MbaffFrameFlag is true (frame_mbs_only_flag=0 AND mb_adaptive_frame_field_flag=1)
    - Current macroblock is the top MB of a macroblock pair
    - Either mb_skip_flag=0 or it's the first MB in the pair

    The flag applies to the entire macroblock pair (top and bottom MBs).
    """

    def test_parse_mb_field_decoding_flag_exists(self):
        """parse_mb_field_decoding_flag function should exist."""
        from decoder.mbaff import parse_mb_field_decoding_flag

        assert callable(parse_mb_field_decoding_flag)

    def test_flag_present_for_top_mb_of_pair(self):
        """mb_field_decoding_flag should be read for top MB of pair."""
        from decoder.mbaff import MBAFFDecoder

        writer = BitWriter()
        writer.write_bits(1, 1)  # mb_field_decoding_flag = 1 (field mode)
        writer.write_bits(0, 8)  # Padding
        reader = BitReader(writer.to_bytes())

        decoder = MBAFFDecoder()

        flag = decoder.parse_mb_field_decoding_flag(
            reader=reader,
            mb_addr=0,  # Top MB of first pair
            mbaff_frame_flag=True,
        )

        assert flag is True  # Field mode

    def test_flag_not_present_for_bottom_mb(self):
        """mb_field_decoding_flag should NOT be read for bottom MB."""
        from decoder.mbaff import MBAFFDecoder

        writer = BitWriter()
        writer.write_bits(0b10101010, 8)  # Junk data - should not be consumed
        reader = BitReader(writer.to_bytes())

        decoder = MBAFFDecoder()
        # Set top MB's flag to field mode
        decoder.set_pair_field_flag(pair_idx=0, is_field=True)

        flag = decoder.parse_mb_field_decoding_flag(
            reader=reader,
            mb_addr=1,  # Bottom MB of first pair
            mbaff_frame_flag=True,
        )

        # Should return the pair's flag without reading from bitstream
        assert flag is True
        assert reader.position == 0  # Nothing consumed

    def test_flag_not_present_when_mbaff_disabled(self):
        """mb_field_decoding_flag should not be read when MBAFF disabled."""
        from decoder.mbaff import MBAFFDecoder

        writer = BitWriter()
        writer.write_bits(0b11111111, 8)
        reader = BitReader(writer.to_bytes())

        decoder = MBAFFDecoder()

        flag = decoder.parse_mb_field_decoding_flag(
            reader=reader,
            mb_addr=0,
            mbaff_frame_flag=False,  # MBAFF disabled
        )

        assert flag is False  # Frame mode by default
        assert reader.position == 0  # Nothing consumed

    def test_flag_inheritance_in_skip_run(self):
        """Skipped MBs inherit mb_field_decoding_flag from previous pair."""
        from decoder.mbaff import MBAFFDecoder

        decoder = MBAFFDecoder()

        # First pair is field mode
        decoder.set_pair_field_flag(pair_idx=0, is_field=True)

        # Second pair is skipped - should use context to infer field flag
        inferred_flag = decoder.infer_field_flag_for_skip(
            mb_addr=2,  # Top MB of second pair
            prev_pair_field_flag=True,
        )

        # H.264 Spec: When both MBs in pair are skipped, flag is inferred
        assert inferred_flag is not None


class TestMacroblockPairAddressing:
    """Tests for macroblock pair addressing in MBAFF.

    H.264 Spec Section 6.4.1: In MBAFF mode, macroblocks are organized
    into pairs. Address calculations differ between frame and field mode.

    Pair layout (for frame with width W macroblocks):
      Pair 0: MB 0 (top), MB 1 (bottom)
      Pair 1: MB 2 (top), MB 3 (bottom)
      ...
    """

    def test_mb_to_pair_index(self):
        """Convert macroblock address to pair index."""
        from decoder.mbaff import mb_addr_to_pair_idx

        # First pair
        assert mb_addr_to_pair_idx(0) == 0  # Top MB of pair 0
        assert mb_addr_to_pair_idx(1) == 0  # Bottom MB of pair 0

        # Second pair
        assert mb_addr_to_pair_idx(2) == 1  # Top MB of pair 1
        assert mb_addr_to_pair_idx(3) == 1  # Bottom MB of pair 1

        # General formula: pair_idx = mb_addr // 2
        assert mb_addr_to_pair_idx(10) == 5

    def test_is_top_mb_of_pair(self):
        """Determine if macroblock is top of its pair."""
        from decoder.mbaff import is_top_mb_of_pair

        assert is_top_mb_of_pair(0) is True
        assert is_top_mb_of_pair(1) is False
        assert is_top_mb_of_pair(2) is True
        assert is_top_mb_of_pair(3) is False
        assert is_top_mb_of_pair(100) is True
        assert is_top_mb_of_pair(101) is False

    def test_get_pair_top_and_bottom(self):
        """Get both MB addresses for a pair."""
        from decoder.mbaff import get_pair_mb_addrs

        top, bottom = get_pair_mb_addrs(pair_idx=0)
        assert top == 0
        assert bottom == 1

        top, bottom = get_pair_mb_addrs(pair_idx=5)
        assert top == 10
        assert bottom == 11

    def test_mb_pair_raster_scan_order(self):
        """Macroblock pairs follow raster scan order.

        For a 4-MB-wide frame (2 pairs per row):
          Row 0: Pair 0 (MB 0,1), Pair 1 (MB 2,3)
          Row 1: Pair 2 (MB 4,5), Pair 3 (MB 6,7)
        """
        from decoder.mbaff import get_pair_position

        # Width in pairs = frame_width_in_mbs
        mb_width = 4

        # Pair 0 at (0, 0)
        x, y = get_pair_position(pair_idx=0, mb_width=mb_width)
        assert (x, y) == (0, 0)

        # Pair 1 at (1, 0)
        x, y = get_pair_position(pair_idx=1, mb_width=mb_width)
        assert (x, y) == (1, 0)

        # Pair 4 at (0, 1) - second row
        x, y = get_pair_position(pair_idx=4, mb_width=mb_width)
        assert (x, y) == (0, 1)


class TestNeighborAvailabilityMBAFF:
    """Tests for neighbor availability in MBAFF mode.

    H.264 Spec Section 6.4.9-12: Neighbor derivation in MBAFF considers:
    - Whether current and neighbor MBs are frame or field coded
    - Vertical position within the pair (top or bottom MB)
    - Frame/field mode mismatch between neighbors
    """

    def test_left_neighbor_same_mode(self):
        """Left neighbor when both MBs use same coding mode."""
        from decoder.mbaff import get_left_neighbor_mbaff

        # Both frame mode
        left = get_left_neighbor_mbaff(
            mb_addr=2,
            current_is_field=False,
            neighbor_is_field=False,
            mb_width=4,
        )

        assert left == 0  # Normal left neighbor

    def test_left_neighbor_field_to_frame(self):
        """Left neighbor when current is field, neighbor is frame.

        When current MB is field-coded but left is frame-coded:
        - Top field MB sees top part of left frame MB
        - Bottom field MB sees bottom part of left frame MB
        """
        from decoder.mbaff import get_left_neighbor_mbaff

        # Current is top MB of field pair, left is frame
        left_for_top = get_left_neighbor_mbaff(
            mb_addr=2,  # Top MB
            current_is_field=True,
            neighbor_is_field=False,
            mb_width=4,
        )

        left_for_bottom = get_left_neighbor_mbaff(
            mb_addr=3,  # Bottom MB
            current_is_field=True,
            neighbor_is_field=False,
            mb_width=4,
        )

        # Both see the same frame-coded pair, but different parts
        assert left_for_top is not None
        assert left_for_bottom is not None

    def test_left_neighbor_frame_to_field(self):
        """Left neighbor when current is frame, neighbor is field.

        When current MB is frame-coded but left is field-coded:
        - Top frame MB sees top field MB
        - Bottom frame MB sees bottom field MB
        """
        from decoder.mbaff import get_left_neighbor_mbaff

        # Current is top MB of frame pair, left is field
        left_for_top = get_left_neighbor_mbaff(
            mb_addr=2,
            current_is_field=False,
            neighbor_is_field=True,
            mb_width=4,
        )

        # Should map to appropriate field MB
        assert left_for_top is not None

    def test_top_neighbor_frame_mode(self):
        """Top neighbor in frame mode.

        For frame-coded MBs:
        - Top MB of pair looks at bottom MB of pair above
        - Bottom MB of pair looks at top MB of same pair
        """
        from decoder.mbaff import get_top_neighbor_mbaff

        mb_width = 4

        # Top MB of pair 4 (second row)
        top = get_top_neighbor_mbaff(
            mb_addr=8,  # Top MB of pair 4
            current_is_field=False,
            mb_width=mb_width,
        )

        # Should be bottom MB of pair above (pair 0's bottom = MB 1)
        # Wait, with 4 MB width, pair 4 is at position (0, 1)
        # Above it is pair 0 at (0, 0), whose bottom MB is 1
        assert top == 1

    def test_top_neighbor_field_mode(self):
        """Top neighbor in field mode.

        For field-coded MBs (parity matters):
        - Top field MB looks at top field MB of pair above
        - Bottom field MB looks at bottom field MB of pair above

        H.264 Spec: Field MBs reference same parity field MBs.
        """
        from decoder.mbaff import get_top_neighbor_mbaff

        mb_width = 4

        # Top field MB of pair 4
        top_for_top_field = get_top_neighbor_mbaff(
            mb_addr=8,
            current_is_field=True,
            mb_width=mb_width,
        )

        # Bottom field MB of pair 4
        top_for_bottom_field = get_top_neighbor_mbaff(
            mb_addr=9,
            current_is_field=True,
            mb_width=mb_width,
        )

        # Field MBs see same-parity neighbors
        # Top field sees top field of pair 0
        # Bottom field sees bottom field of pair 0
        assert top_for_top_field == 0  # Top of pair 0
        assert top_for_bottom_field == 1  # Bottom of pair 0

    def test_neighbor_unavailable_at_boundary(self):
        """Neighbors unavailable at picture boundaries."""
        from decoder.mbaff import get_left_neighbor_mbaff, get_top_neighbor_mbaff

        mb_width = 4

        # First pair has no left neighbor
        left = get_left_neighbor_mbaff(
            mb_addr=0,
            current_is_field=False,
            neighbor_is_field=False,
            mb_width=mb_width,
        )
        assert left is None

        # First row has no top neighbor
        top = get_top_neighbor_mbaff(
            mb_addr=0,
            current_is_field=False,
            mb_width=mb_width,
        )
        assert top is None


class TestMotionVectorPredictionMBAFF:
    """Tests for motion vector prediction in MBAFF.

    H.264 Spec Section 8.4.1.2-3: MV prediction in MBAFF must account for:
    - Vertical MV scaling when frame/field mode differs between blocks
    - Proper neighbor selection based on coding mode
    - Reference index mapping between frame and field pictures
    """

    def test_mv_prediction_frame_to_frame(self):
        """MV prediction when all blocks are frame-coded."""
        from decoder.mbaff import predict_mv_mbaff

        # Setup: current and all neighbors are frame-coded
        mv_a = np.array([10, 20], dtype=np.int32)  # Left
        mv_b = np.array([15, 25], dtype=np.int32)  # Top
        mv_c = np.array([12, 22], dtype=np.int32)  # Top-right

        predicted = predict_mv_mbaff(
            mv_a=mv_a,
            mv_b=mv_b,
            mv_c=mv_c,
            current_is_field=False,
            neighbor_a_is_field=False,
            neighbor_b_is_field=False,
            neighbor_c_is_field=False,
        )

        # Standard median prediction (no scaling)
        expected = np.array([12, 22], dtype=np.int32)  # Median
        np.testing.assert_array_equal(predicted, expected)

    def test_mv_scaling_field_to_frame(self):
        """MV scaling when predicting field MB from frame neighbor.

        When current is field-coded and neighbor is frame-coded:
        - Vertical MV component must be scaled by 2
        - Field MVs represent half the vertical motion
        """
        from decoder.mbaff import scale_mv_for_field_prediction

        # Frame MV
        frame_mv = np.array([10, 20], dtype=np.int32)

        # Scale for field prediction
        field_mv = scale_mv_for_field_prediction(
            mv=frame_mv,
            current_is_field=True,
            neighbor_is_field=False,
        )

        # Vertical component scaled by 0.5 (field sees half vertical distance)
        assert field_mv[0] == 10  # Horizontal unchanged
        assert field_mv[1] == 10  # Vertical halved

    def test_mv_scaling_frame_to_field(self):
        """MV scaling when predicting frame MB from field neighbor.

        When current is frame-coded and neighbor is field-coded:
        - Vertical MV component must be scaled by 2
        """
        from decoder.mbaff import scale_mv_for_field_prediction

        # Field MV
        field_mv = np.array([10, 10], dtype=np.int32)

        # Scale for frame prediction
        frame_mv = scale_mv_for_field_prediction(
            mv=field_mv,
            current_is_field=False,
            neighbor_is_field=True,
        )

        # Vertical component scaled by 2 (frame sees double vertical distance)
        assert frame_mv[0] == 10  # Horizontal unchanged
        assert frame_mv[1] == 20  # Vertical doubled

    def test_mv_prediction_mixed_neighbors(self):
        """MV prediction with mixed frame/field neighbors."""
        from decoder.mbaff import predict_mv_mbaff

        # Left: field-coded, Top: frame-coded, Top-right: unavailable
        mv_a = np.array([8, 8], dtype=np.int32)   # Field neighbor
        mv_b = np.array([12, 24], dtype=np.int32)  # Frame neighbor

        predicted = predict_mv_mbaff(
            mv_a=mv_a,
            mv_b=mv_b,
            mv_c=None,  # Top-right unavailable
            current_is_field=True,
            neighbor_a_is_field=True,
            neighbor_b_is_field=False,  # Frame - needs scaling
            neighbor_c_is_field=None,
        )

        # Should scale mv_b before median calculation
        assert predicted is not None
        assert predicted.shape == (2,)

    def test_direct_mode_mv_mbaff(self):
        """Direct mode MV derivation in MBAFF.

        H.264 Spec: Co-located block determination considers
        field/frame coding of current and reference.
        """
        from decoder.mbaff import derive_direct_mv_mbaff

        # Current: field MB, Reference: frame picture
        mv_l0, mv_l1 = derive_direct_mv_mbaff(
            current_is_field=True,
            colocated_is_field=False,
            colocated_mv=np.array([20, 40], dtype=np.int32),
            colocated_ref=0,
            poc_current=4,
            poc_l0=0,
            poc_l1=8,
        )

        # MVs should be scaled appropriately
        assert mv_l0 is not None
        assert mv_l1 is not None


class TestDeblockingFilterMBAFF:
    """Tests for deblocking filter adaptation in MBAFF.

    H.264 Spec Section 8.7.2.2: Deblocking in MBAFF considers:
    - Transform size based on field/frame coding
    - Edge positions different for field vs frame MBs
    - Boundary strength calculation with field/frame neighbors
    """

    def test_deblock_edge_positions_frame_mb(self):
        """Edge positions for frame-coded MB in MBAFF."""
        from decoder.mbaff import get_deblock_edges_mbaff

        edges = get_deblock_edges_mbaff(
            is_field=False,
            direction='vertical',
        )

        # Frame MB: standard 4x4 edge positions
        # Vertical edges at x = 0, 4, 8, 12
        edge_x = set(x for x, y in edges)
        assert 0 in edge_x
        assert 4 in edge_x
        assert 8 in edge_x
        assert 12 in edge_x

    def test_deblock_edge_positions_field_mb(self):
        """Edge positions for field-coded MB in MBAFF.

        Field MBs use 4x8 transform blocks (vertically stretched).
        Internal horizontal edges are at different positions.
        """
        from decoder.mbaff import get_deblock_edges_mbaff

        h_edges = get_deblock_edges_mbaff(
            is_field=True,
            direction='horizontal',
        )

        # Field MB: horizontal edges may differ
        # Internal edges at 8-pixel boundaries for field
        edge_y = set(y for x, y in h_edges)
        assert 0 in edge_y  # MB boundary always filtered

    def test_boundary_strength_mbaff_mixed(self):
        """Boundary strength between frame and field MBs.

        At boundary between differently-coded MBs, bS calculation
        must consider the mixed nature.
        """
        from decoder.mbaff import calc_bs_mbaff_boundary

        bs = calc_bs_mbaff_boundary(
            current_is_field=True,
            neighbor_is_field=False,
            current_is_intra=False,
            neighbor_is_intra=False,
            current_mv=np.array([0, 0], dtype=np.int32),
            neighbor_mv=np.array([0, 0], dtype=np.int32),
            current_ref=0,
            neighbor_ref=0,
            current_has_coeff=False,
            neighbor_has_coeff=False,
        )

        # bS depends on scaled MV comparison
        assert bs >= 0
        assert bs <= 4

    def test_deblock_mb_pair_vertical_edge(self):
        """Deblock vertical edge at pair boundary."""
        from decoder.mbaff import deblock_mbaff_pair_boundary

        # Create a frame with two pairs side by side
        luma = np.zeros((32, 32), dtype=np.uint8)  # 2x1 pairs
        luma[:, :16] = 100
        luma[:, 16:] = 200

        pair_left_info = {
            'is_field': False,
            'is_intra': [False, False],  # Top and bottom MB
        }
        pair_right_info = {
            'is_field': True,
            'is_intra': [True, True],
        }

        luma_out = deblock_mbaff_pair_boundary(
            luma=luma,
            pair_left=pair_left_info,
            pair_right=pair_right_info,
            qp=26,
        )

        # Edge should be filtered (values should change)
        assert not np.array_equal(luma_out, luma)


class TestVerticalSubsamplingFieldMBs:
    """Tests for vertical subsampling in field-coded MBs.

    In field mode, each field MB represents alternate lines.
    Motion compensation and reconstruction use vertical subsampling.
    """

    def test_is_field_macroblock_utility(self):
        """Utility function to check if MB is field-coded."""
        from decoder.mbaff import is_field_macroblock

        # Setup MBAFF state
        pair_field_flags = {0: True, 1: False}  # Pair 0 field, pair 1 frame

        assert is_field_macroblock(mb_addr=0, pair_field_flags=pair_field_flags)
        assert is_field_macroblock(mb_addr=1, pair_field_flags=pair_field_flags)
        assert not is_field_macroblock(mb_addr=2, pair_field_flags=pair_field_flags)
        assert not is_field_macroblock(mb_addr=3, pair_field_flags=pair_field_flags)

    def test_field_mb_line_mapping(self):
        """Map field MB lines to frame lines.

        Field coding: Top MB contains even lines, bottom MB contains odd lines.
        """
        from decoder.mbaff import get_field_line_in_frame

        # Top field MB, line 0 -> frame line 0
        assert get_field_line_in_frame(field_line=0, is_bottom_field=False, pair_y=0) == 0

        # Top field MB, line 1 -> frame line 2
        assert get_field_line_in_frame(field_line=1, is_bottom_field=False, pair_y=0) == 2

        # Bottom field MB, line 0 -> frame line 1
        assert get_field_line_in_frame(field_line=0, is_bottom_field=True, pair_y=0) == 1

        # Bottom field MB, line 1 -> frame line 3
        assert get_field_line_in_frame(field_line=1, is_bottom_field=True, pair_y=0) == 3

    def test_motion_comp_vertical_subsampling(self):
        """Motion compensation with vertical subsampling for fields."""
        from decoder.mbaff import motion_comp_field_mb

        # Reference frame (progressive)
        ref_luma = np.arange(32 * 16, dtype=np.uint8).reshape(32, 16)

        # Motion compensate for top field MB
        pred = motion_comp_field_mb(
            ref_luma=ref_luma,
            mv=np.array([0, 0], dtype=np.int32),
            mb_x=0,
            mb_y=0,
            is_bottom_field=False,
        )

        # Should extract even lines only
        assert pred.shape == (16, 16)

    def test_reconstruction_interleaving(self):
        """Reconstructed field MBs must interleave into frame."""
        from decoder.mbaff import interleave_field_pair

        # Two field MBs (top and bottom)
        top_field_mb = np.full((16, 16), 100, dtype=np.uint8)
        bottom_field_mb = np.full((16, 16), 200, dtype=np.uint8)

        frame = interleave_field_pair(
            top_field=top_field_mb,
            bottom_field=bottom_field_mb,
        )

        # Frame should have interleaved lines
        assert frame.shape == (32, 16)
        # Even lines from top field
        np.testing.assert_array_equal(frame[0, :], 100)
        np.testing.assert_array_equal(frame[2, :], 100)
        # Odd lines from bottom field
        np.testing.assert_array_equal(frame[1, :], 200)
        np.testing.assert_array_equal(frame[3, :], 200)


class TestRefPicListReorderingFields:
    """Tests for reference picture list reordering with fields.

    H.264 Spec Section 8.2.4.2: In MBAFF, reference lists can contain
    both frame and field references. Reordering operations must
    handle field/frame picture type differences.
    """

    def test_ref_list_contains_fields(self):
        """Reference list can contain separate fields."""
        from decoder.mbaff import build_ref_list_mbaff

        # Create reference buffer with frame and field pictures
        ref_buffer = [
            {'poc': 0, 'is_field': False},  # Frame
            {'poc': 2, 'is_field': True, 'is_bottom': False},  # Top field
            {'poc': 2, 'is_field': True, 'is_bottom': True},   # Bottom field
        ]

        # For field-coded current picture
        ref_list = build_ref_list_mbaff(
            ref_buffer=ref_buffer,
            current_poc=4,
            current_is_field=True,
        )

        assert len(ref_list) > 0

    def test_field_ref_pic_list_modification(self):
        """Reference picture list modification for fields.

        modification_of_pic_nums_idc values:
        - 4, 5: Long-term reference field
        """
        from decoder.mbaff import modify_ref_list_mbaff

        initial_list = [
            {'poc': 0, 'is_field': True, 'is_bottom': False},
            {'poc': 2, 'is_field': True, 'is_bottom': False},
        ]

        # Reorder: swap positions
        modifications = [
            {'idc': 0, 'abs_diff_pic_num_minus1': 1},  # Short-term
            {'idc': 3},  # End
        ]

        modified = modify_ref_list_mbaff(
            ref_list=initial_list.copy(),
            modifications=modifications,
            current_pic_num=4,
        )

        assert len(modified) == len(initial_list)

    def test_parity_based_ref_selection(self):
        """Field MBs prefer same-parity reference fields.

        H.264 Spec: When current is top field, prefer top field references.
        """
        from decoder.mbaff import get_preferred_ref_parity

        # Top field prefers top field references
        parity = get_preferred_ref_parity(
            current_is_bottom=False,
        )
        assert parity == 'top'

        # Bottom field prefers bottom field references
        parity = get_preferred_ref_parity(
            current_is_bottom=True,
        )
        assert parity == 'bottom'


class TestChromaInterpolationMBAFF:
    """Tests for chroma interpolation differences in MBAFF.

    Chroma interpolation in field mode must account for
    vertical subsampling of field pictures.
    """

    def test_chroma_mc_field_mode(self):
        """Chroma motion compensation for field MB."""
        from decoder.mbaff import chroma_motion_comp_field

        # Reference chroma (8x8 per MB in 4:2:0)
        ref_cb = np.arange(16 * 8, dtype=np.uint8).reshape(16, 8)

        # Fractional MV (1/8 pixel precision for chroma)
        mv = np.array([1, 2], dtype=np.int32)  # 1/8 pixel

        pred = chroma_motion_comp_field(
            ref_chroma=ref_cb,
            mv=mv,
            mb_x=0,
            mb_y=0,
            is_bottom_field=False,
        )

        # Field chroma prediction
        assert pred.shape == (8, 8)

    def test_chroma_vertical_scale_field(self):
        """Chroma MV vertical component scaling for fields.

        In field mode, vertical chroma MV must account for
        field structure (half vertical resolution per field).
        """
        from decoder.mbaff import scale_chroma_mv_field

        # Frame MV
        mv = np.array([4, 8], dtype=np.int32)

        # Scale for field chroma
        scaled = scale_chroma_mv_field(
            mv=mv,
            current_is_field=True,
            ref_is_field=False,
        )

        # Vertical component adjusted
        assert scaled[1] != mv[1]  # Should be different


class TestTransformSelectionMBAFF:
    """Tests for transform selection in MBAFF (4x4 vs 4x8 for fields).

    H.264 Spec: Field MBs may use 4x8 transforms instead of 4x4
    to better match field structure.
    """

    def test_transform_block_size_frame_mb(self):
        """Frame MB uses standard 4x4 transform."""
        from decoder.mbaff import get_transform_size_mbaff

        size = get_transform_size_mbaff(
            is_field=False,
            transform_8x8_mode_flag=False,
        )

        assert size == (4, 4)

    def test_transform_block_size_field_mb(self):
        """Field MB may use 4x8 transform.

        The 4x8 transform is more efficient for field content
        due to reduced vertical correlation.
        """
        from decoder.mbaff import get_transform_size_mbaff

        # Field MB without 8x8 mode
        size = get_transform_size_mbaff(
            is_field=True,
            transform_8x8_mode_flag=False,
        )

        # May return 4x4 or 4x8 depending on implementation
        assert size in [(4, 4), (4, 8)]

    def test_idct_selection_field_mb(self):
        """IDCT selection based on field/frame coding."""
        from decoder.mbaff import get_idct_function_mbaff

        # Frame MB uses standard 4x4 IDCT
        idct_frame = get_idct_function_mbaff(is_field=False)
        assert callable(idct_frame)

        # Field MB may use different IDCT
        idct_field = get_idct_function_mbaff(is_field=True)
        assert callable(idct_field)

    def test_dequant_scaling_field_mb(self):
        """Dequantization scaling for field MBs.

        Field MBs may use different quantization matrices.
        """
        from decoder.mbaff import get_dequant_scale_mbaff

        # Get scaling factors
        scale_frame = get_dequant_scale_mbaff(qp=26, is_field=False)
        scale_field = get_dequant_scale_mbaff(qp=26, is_field=True)

        # Both should return valid scaling
        assert scale_frame is not None
        assert scale_field is not None


class TestMBAFFDecoderIntegration:
    """Integration tests for full MBAFF decoding."""

    def test_mbaff_decoder_class_exists(self):
        """MBAFFDecoder class should exist."""
        from decoder.mbaff import MBAFFDecoder

        decoder = MBAFFDecoder()
        assert decoder is not None

    def test_decode_mbaff_pair(self):
        """Decode a macroblock pair in MBAFF mode."""
        from decoder.mbaff import MBAFFDecoder

        decoder = MBAFFDecoder()

        # Create minimal bitstream for MB pair
        writer = BitWriter()

        # mb_field_decoding_flag = 0 (frame mode)
        writer.write_bits(0, 1)

        # Top MB: I_16x16 (mb_type = some value)
        writer.write_ue(1)  # Simplified

        # Bottom MB: follows same pair
        writer.write_ue(1)

        writer.write_bits(0, 16)  # Padding
        reader = BitReader(writer.to_bytes())

        # Decode pair
        result = decoder.decode_mbaff_pair(
            reader=reader,
            pair_idx=0,
            frame_luma=np.zeros((32, 16), dtype=np.uint8),
            frame_cb=np.zeros((16, 8), dtype=np.uint8),
            frame_cr=np.zeros((16, 8), dtype=np.uint8),
            qp=26,
        )

        assert result is not None
        assert 'top_mb' in result
        assert 'bottom_mb' in result

    def test_mbaff_frame_decoding(self):
        """Decode entire MBAFF frame."""
        from decoder.mbaff import MBAFFDecoder

        decoder = MBAFFDecoder()

        # Small frame: 2x2 pairs (64x64 pixels)
        frame_width = 4  # in MBs
        frame_height = 4  # in MBs (2 pairs high)

        result = decoder.decode_mbaff_frame(
            reader=None,  # Would be actual bitstream
            frame_width=frame_width,
            frame_height=frame_height,
        )

        # Should return decoded frame
        assert result is not None

    def test_mbaff_with_slice_header(self):
        """MBAFF decoding uses slice header information."""
        from slice.slice_header import SliceHeader
        from decoder.mbaff import MBAFFDecoder

        header = SliceHeader(
            first_mb_in_slice=0,
            slice_type=2,  # I-slice
            pic_parameter_set_id=0,
            frame_num=0,
            field_pic_flag=False,  # MBAFF frame, not field picture
            slice_qp_delta=0,
            header_bit_size=0,
        )

        decoder = MBAFFDecoder()
        decoder.configure_from_header(header)

        assert decoder.is_configured


class TestMBAFFEdgeCases:
    """Tests for MBAFF edge cases and error handling."""

    def test_single_mb_width_frame(self):
        """MBAFF with 1-MB-wide frame (vertical strip)."""
        from decoder.mbaff import get_pair_position, get_left_neighbor_mbaff

        mb_width = 1

        # All pairs have no left neighbor
        left = get_left_neighbor_mbaff(
            mb_addr=0,
            current_is_field=False,
            neighbor_is_field=False,
            mb_width=mb_width,
        )
        assert left is None

        # Pairs stack vertically
        x, y = get_pair_position(pair_idx=2, mb_width=mb_width)
        assert x == 0
        assert y == 2

    def test_alternating_field_frame_pairs(self):
        """Handle alternating field/frame pairs in same frame."""
        from decoder.mbaff import MBAFFDecoder

        decoder = MBAFFDecoder()

        # Set alternating pattern
        decoder.set_pair_field_flag(pair_idx=0, is_field=False)  # Frame
        decoder.set_pair_field_flag(pair_idx=1, is_field=True)   # Field
        decoder.set_pair_field_flag(pair_idx=2, is_field=False)  # Frame
        decoder.set_pair_field_flag(pair_idx=3, is_field=True)   # Field

        # Verify pattern
        assert not decoder.is_pair_field_coded(0)
        assert decoder.is_pair_field_coded(1)
        assert not decoder.is_pair_field_coded(2)
        assert decoder.is_pair_field_coded(3)

    def test_mbaff_with_skip_run(self):
        """Handle mb_skip_run in MBAFF mode.

        When MBs are skipped, field flag must be inferred.
        """
        from decoder.mbaff import MBAFFDecoder

        decoder = MBAFFDecoder()

        # First pair coded as field
        decoder.set_pair_field_flag(pair_idx=0, is_field=True)

        # Process skip run starting at pair 1
        skip_results = decoder.process_skip_run(
            start_mb_addr=2,
            skip_count=4,  # Skip 2 pairs (4 MBs)
        )

        # Skipped pairs should have inferred field flags
        assert len(skip_results) == 4

    def test_invalid_mb_addr_handling(self):
        """Handle invalid macroblock addresses gracefully."""
        from decoder.mbaff import mb_addr_to_pair_idx, is_top_mb_of_pair

        # Negative address
        with pytest.raises(ValueError):
            mb_addr_to_pair_idx(-1)

        # Very large address should still work
        pair = mb_addr_to_pair_idx(10000)
        assert pair == 5000


class TestMBAFFIntraPrediction:
    """Tests for intra prediction in MBAFF mode."""

    def test_intra_neighbor_derivation_field(self):
        """Intra prediction neighbor derivation for field MBs.

        Field MBs use same-parity lines for neighbors.
        """
        from decoder.mbaff import get_intra_neighbors_mbaff

        neighbors = get_intra_neighbors_mbaff(
            mb_addr=0,
            is_field=True,
            is_bottom_mb=False,
            frame_luma=np.zeros((32, 16), dtype=np.uint8),
        )

        assert 'top' in neighbors
        assert 'left' in neighbors

    def test_intra_16x16_prediction_field_mb(self):
        """Intra 16x16 prediction for field-coded MB."""
        from decoder.mbaff import intra_16x16_predict_field

        # Field MB with available neighbors
        top = np.full(16, 100, dtype=np.uint8)
        left = np.full(16, 100, dtype=np.uint8)

        # DC mode prediction
        pred = intra_16x16_predict_field(
            mode=0,  # DC
            top=top,
            left=left,
            top_available=True,
            left_available=True,
        )

        assert pred.shape == (16, 16)

    def test_intra_4x4_prediction_cross_field_frame(self):
        """Intra 4x4 prediction across field/frame boundary.

        When predicting field MB with frame-coded neighbor.
        """
        from decoder.mbaff import get_intra_4x4_neighbors_mbaff

        neighbors = get_intra_4x4_neighbors_mbaff(
            block_idx=0,
            mb_addr=2,
            current_is_field=True,
            left_mb_is_field=False,  # Frame-coded left neighbor
        )

        # Should handle line mapping correctly
        assert neighbors is not None


class TestMBAFFCABACContexts:
    """Tests for CABAC context adaptation in MBAFF."""

    def test_cabac_ctx_mb_field_decoding_flag(self):
        """CABAC context for mb_field_decoding_flag.

        H.264 Spec: ctxIdxInc depends on neighbors' field flags.
        """
        from decoder.mbaff import get_ctx_idx_mb_field_flag

        # Both neighbors frame-coded
        ctx = get_ctx_idx_mb_field_flag(
            left_is_field=False,
            top_is_field=False,
        )
        assert ctx == 0

        # Left field, top frame
        ctx = get_ctx_idx_mb_field_flag(
            left_is_field=True,
            top_is_field=False,
        )
        assert ctx == 1

        # Both field
        ctx = get_ctx_idx_mb_field_flag(
            left_is_field=True,
            top_is_field=True,
        )
        assert ctx == 2

    def test_cabac_ctx_adaptation_field_mb(self):
        """CABAC context adaptation for field-coded MBs.

        Some syntax elements use different contexts in field mode.
        """
        from decoder.mbaff import get_cabac_ctx_field_adaptation

        # mb_type context may differ
        ctx_frame = get_cabac_ctx_field_adaptation('mb_type', is_field=False)
        ctx_field = get_cabac_ctx_field_adaptation('mb_type', is_field=True)

        # Contexts should be valid
        assert ctx_frame >= 0
        assert ctx_field >= 0
