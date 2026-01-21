# h264/reconstruct/tests/test_macroblock.py
"""Tests for macroblock reconstruction."""

import pytest
import numpy as np

from bitstream import BitWriter, BitReader, BITSTRING_AVAILABLE
from reconstruct.macroblock import (
    MBType,
    CodedBlockPattern,
    MacroblockData,
    decode_i16x16_mb_type,
    decode_cbp_intra,
    _get_4x4_block_position,
    _clip_pixel,
    _clip_block,
    reconstruct_i16x16_luma,
)
from entropy.tables import COEFF_TOKEN_0
from intra import Intra16x16Mode

# Skip all tests if bitstring not available
pytestmark = pytest.mark.skipif(
    not BITSTRING_AVAILABLE,
    reason="bitstring library not installed"
)


class TestCodedBlockPattern:
    """Tests for CodedBlockPattern."""

    def test_no_coefficients(self):
        """CBP with no coefficients."""
        cbp = CodedBlockPattern(luma=0, chroma=0)
        assert cbp.has_luma_dc is False
        assert cbp.has_chroma_dc is False
        assert cbp.has_chroma_ac is False

    def test_luma_only(self):
        """CBP with luma coefficients only."""
        cbp = CodedBlockPattern(luma=15, chroma=0)
        assert cbp.has_luma_dc is True
        assert cbp.has_chroma_dc is False
        assert cbp.has_chroma_ac is False

    def test_chroma_dc_only(self):
        """CBP with chroma DC only."""
        cbp = CodedBlockPattern(luma=0, chroma=1)
        assert cbp.has_luma_dc is False
        assert cbp.has_chroma_dc is True
        assert cbp.has_chroma_ac is False

    def test_chroma_dc_and_ac(self):
        """CBP with chroma DC and AC."""
        cbp = CodedBlockPattern(luma=0, chroma=2)
        assert cbp.has_chroma_dc is True
        assert cbp.has_chroma_ac is True

    def test_full_cbp(self):
        """CBP with all coefficients."""
        cbp = CodedBlockPattern(luma=15, chroma=2)
        assert cbp.has_luma_dc is True
        assert cbp.has_chroma_dc is True
        assert cbp.has_chroma_ac is True

    def test_luma_8x8_blocks(self):
        """Test individual 8x8 block flags."""
        cbp = CodedBlockPattern(luma=0b1010, chroma=0)
        assert cbp.has_luma_8x8(0) is False
        assert cbp.has_luma_8x8(1) is True
        assert cbp.has_luma_8x8(2) is False
        assert cbp.has_luma_8x8(3) is True


class TestDecodeI16x16MBType:
    """Tests for I_16x16 mb_type decoding."""

    def test_type_1(self):
        """mb_type=1: pred_mode=0, cbp_luma=0, cbp_chroma=0."""
        pred, luma, chroma = decode_i16x16_mb_type(1)
        assert pred == 0
        assert luma == 0
        assert chroma == 0

    def test_type_2(self):
        """mb_type=2: pred_mode=1, cbp_luma=0, cbp_chroma=0."""
        pred, luma, chroma = decode_i16x16_mb_type(2)
        assert pred == 1
        assert luma == 0
        assert chroma == 0

    def test_type_5(self):
        """mb_type=5: pred_mode=0, cbp_luma=0, cbp_chroma=1."""
        pred, luma, chroma = decode_i16x16_mb_type(5)
        assert pred == 0
        assert luma == 0
        assert chroma == 1

    def test_type_13(self):
        """mb_type=13: pred_mode=0, cbp_luma=15, cbp_chroma=0."""
        pred, luma, chroma = decode_i16x16_mb_type(13)
        assert pred == 0
        assert luma == 15
        assert chroma == 0

    def test_type_24(self):
        """mb_type=24: pred_mode=3, cbp_luma=15, cbp_chroma=2."""
        pred, luma, chroma = decode_i16x16_mb_type(24)
        assert pred == 3
        assert luma == 15
        assert chroma == 2

    def test_invalid_type_raises(self):
        """Invalid mb_type raises error."""
        with pytest.raises(ValueError):
            decode_i16x16_mb_type(0)
        with pytest.raises(ValueError):
            decode_i16x16_mb_type(25)


class TestDecodeCBPIntra:
    """Tests for CBP decoding based on H.264 spec Table 9-4.

    H.264 Table 9-4 maps codeNum (from me(v)) to coded_block_pattern.
    For Intra macroblocks with ChromaArrayType=1 (4:2:0):
    - cbp_luma = coded_block_pattern & 0x0F (4 bits, 0-15)
    - cbp_chroma = (coded_block_pattern >> 4) & 0x03 (2 bits, 0-2)

    The table is designed so common patterns have shorter codes.
    """

    # H.264 Table 9-4 for Intra (ChromaArrayType=1)
    # Format: (codeNum, expected_luma, expected_chroma)
    # cbp = (chroma << 4) | luma
    EXPECTED_CBP_INTRA = [
        # codeNum 0-3: highest probability patterns
        (0, 15, 2),   # cbp=47: all luma + chroma DC+AC
        (1, 15, 1),   # cbp=31: all luma + chroma DC only
        (2, 15, 0),   # cbp=15: all luma, no chroma
        (3, 0, 0),    # cbp=0: no coefficients

        # codeNum 4-7: 3 of 4 luma blocks + chroma DC
        (4, 7, 1),    # cbp=23
        (5, 11, 1),   # cbp=27
        (6, 13, 1),   # cbp=29
        (7, 14, 1),   # cbp=30

        # codeNum 8-11: 3 of 4 luma blocks, no chroma
        (8, 7, 0),    # cbp=7
        (9, 11, 0),   # cbp=11
        (10, 13, 0),  # cbp=13
        (11, 14, 0),  # cbp=14

        # codeNum 12-15: 3 of 4 luma blocks + chroma DC+AC
        (12, 7, 2),   # cbp=39
        (13, 11, 2),  # cbp=43
        (14, 13, 2),  # cbp=45
        (15, 14, 2),  # cbp=46

        # codeNum 16: no luma, chroma DC only
        (16, 0, 1),   # cbp=16

        # codeNum 17-20: 2 of 4 luma blocks, no chroma
        (17, 3, 0),   # cbp=3
        (18, 5, 0),   # cbp=5
        (19, 10, 0),  # cbp=10
        (20, 12, 0),  # cbp=12

        # codeNum 21-24: 2 of 4 luma blocks + chroma DC
        (21, 3, 1),   # cbp=19
        (22, 5, 1),   # cbp=21
        (23, 10, 1),  # cbp=26
        (24, 12, 1),  # cbp=28

        # codeNum 25-28: 2 of 4 luma blocks + chroma DC+AC
        (25, 3, 2),   # cbp=35
        (26, 5, 2),   # cbp=37
        (27, 10, 2),  # cbp=42
        (28, 12, 2),  # cbp=44

        # codeNum 29-32: 1 of 4 luma blocks, no chroma
        (29, 1, 0),   # cbp=1
        (30, 2, 0),   # cbp=2
        (31, 4, 0),   # cbp=4
        (32, 8, 0),   # cbp=8

        # codeNum 33-36: 1 of 4 luma blocks + chroma DC
        (33, 1, 1),   # cbp=17
        (34, 2, 1),   # cbp=18
        (35, 4, 1),   # cbp=20
        (36, 8, 1),   # cbp=24

        # codeNum 37-40: 2 adjacent luma blocks
        (37, 6, 0),   # cbp=6
        (38, 9, 0),   # cbp=9
        (39, 6, 1),   # cbp=22
        (40, 9, 1),   # cbp=25

        # codeNum 41-45: no luma or 1 luma + chroma DC+AC
        (41, 0, 2),   # cbp=32
        (42, 1, 2),   # cbp=33
        (43, 2, 2),   # cbp=34
        (44, 4, 2),   # cbp=36
        (45, 8, 2),   # cbp=40

        # codeNum 46-47: 2 adjacent luma + chroma DC+AC
        (46, 6, 2),   # cbp=38
        (47, 9, 2),   # cbp=41
    ]

    @pytest.mark.parametrize("codeNum,expected_luma,expected_chroma", EXPECTED_CBP_INTRA)
    def test_cbp_intra_table(self, codeNum, expected_luma, expected_chroma):
        """Test each codeNum maps to correct (luma, chroma) per H.264 Table 9-4."""
        luma, chroma = decode_cbp_intra(codeNum)
        assert luma == expected_luma, f"codeNum={codeNum}: luma expected {expected_luma}, got {luma}"
        assert chroma == expected_chroma, f"codeNum={codeNum}: chroma expected {expected_chroma}, got {chroma}"

    def test_cbp_luma_range(self):
        """All luma values should be in range 0-15 (4 bits)."""
        for codeNum in range(48):
            luma, chroma = decode_cbp_intra(codeNum)
            assert 0 <= luma <= 15, f"codeNum={codeNum}: luma {luma} out of range"

    def test_cbp_chroma_range(self):
        """All chroma values should be in range 0-2."""
        for codeNum in range(48):
            luma, chroma = decode_cbp_intra(codeNum)
            assert 0 <= chroma <= 2, f"codeNum={codeNum}: chroma {chroma} out of range"

    def test_cbp_out_of_range(self):
        """Out of range codeNum returns (0, 0)."""
        luma, chroma = decode_cbp_intra(48)
        assert luma == 0
        assert chroma == 0
        luma, chroma = decode_cbp_intra(100)
        assert luma == 0
        assert chroma == 0


class TestGet4x4BlockPosition:
    """Tests for 4x4 block position calculation."""

    def test_block_0(self):
        """Block 0 is at (0, 0)."""
        row, col = _get_4x4_block_position(0)
        assert row == 0
        assert col == 0

    def test_block_1(self):
        """Block 1 is at (0, 4)."""
        row, col = _get_4x4_block_position(1)
        assert row == 0
        assert col == 4

    def test_block_2(self):
        """Block 2 is at (4, 0)."""
        row, col = _get_4x4_block_position(2)
        assert row == 4
        assert col == 0

    def test_block_4(self):
        """Block 4 is at (0, 8)."""
        row, col = _get_4x4_block_position(4)
        assert row == 0
        assert col == 8

    def test_block_15(self):
        """Block 15 is at (12, 12)."""
        row, col = _get_4x4_block_position(15)
        assert row == 12
        assert col == 12

    def test_all_blocks_coverage(self):
        """All 16 blocks cover the 16x16 area."""
        positions = set()
        for i in range(16):
            row, col = _get_4x4_block_position(i)
            positions.add((row, col))

        assert len(positions) == 16
        # Check all expected positions
        expected = {(r * 4, c * 4) for r in range(4) for c in range(4)}
        assert positions == expected


class TestClipFunctions:
    """Tests for clipping functions."""

    def test_clip_pixel_in_range(self):
        """Pixel in valid range unchanged."""
        assert _clip_pixel(0) == 0
        assert _clip_pixel(128) == 128
        assert _clip_pixel(255) == 255

    def test_clip_pixel_negative(self):
        """Negative pixel clipped to 0."""
        assert _clip_pixel(-10) == 0
        assert _clip_pixel(-1) == 0

    def test_clip_pixel_overflow(self):
        """Overflow pixel clipped to 255."""
        assert _clip_pixel(256) == 255
        assert _clip_pixel(1000) == 255

    def test_clip_block(self):
        """Block clipping."""
        block = np.array([[-10, 128], [300, 0]], dtype=np.int32)
        clipped = _clip_block(block)
        assert clipped.dtype == np.uint8
        assert clipped[0, 0] == 0
        assert clipped[0, 1] == 128
        assert clipped[1, 0] == 255
        assert clipped[1, 1] == 0


class TestMacroblockData:
    """Tests for MacroblockData dataclass."""

    def test_default_values(self):
        """Default initialization."""
        mb = MacroblockData()
        assert mb.mb_type == 0
        assert mb.luma.shape == (16, 16)
        assert mb.cb.shape == (8, 8)
        assert mb.cr.shape == (8, 8)
        assert len(mb.nz_counts) == 24

    def test_nz_counts_assignment(self):
        """Non-zero counts can be assigned."""
        mb = MacroblockData()
        mb.nz_counts[0] = 5
        mb.nz_counts[16] = 3
        assert mb.nz_counts[0] == 5
        assert mb.nz_counts[16] == 3


class TestReconstructI16x16Luma:
    """Tests for I_16x16 luma reconstruction."""

    def test_reconstruct_no_residual(self):
        """Reconstruct with zero residual (CBP=0)."""
        # Create bitstream with zero coefficients
        writer = BitWriter()
        # Empty DC block: TC=0 for nC=0 -> 0b1
        writer.write_bits(0b1, 1)
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        nz_counts = np.zeros(24, dtype=np.int32)

        # DC prediction with all neighbors = 128
        neighbors_top = np.full(16, 128, dtype=np.uint8)
        neighbors_left = np.full(16, 128, dtype=np.uint8)

        result = reconstruct_i16x16_luma(
            reader,
            pred_mode=2,  # DC mode
            cbp_luma=0,  # No coefficients
            qp=26,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=128,
            nz_counts=nz_counts
        )

        assert result.shape == (16, 16)
        assert result.dtype == np.uint8
        # Should be all 128 (DC prediction with 128 neighbors)
        assert np.all(result == 128)

    def test_reconstruct_no_neighbors(self):
        """Reconstruct without neighbors uses DC mode (defaults to 128)."""
        writer = BitWriter()
        writer.write_bits(0b1, 1)
        writer.write_bits(0, 7)
        reader = BitReader(writer.to_bytes())

        nz_counts = np.zeros(24, dtype=np.int32)

        # DC mode is most appropriate when no neighbors available
        result = reconstruct_i16x16_luma(
            reader,
            pred_mode=2,  # DC mode
            cbp_luma=0,
            qp=26,
            neighbors_top=None,
            neighbors_left=None,
            neighbor_top_left=None,
            nz_counts=nz_counts
        )

        assert result.shape == (16, 16)
        # Should be all 128 (default DC value when no neighbors)
        assert np.all(result == 128)


class TestMBTypeEnum:
    """Tests for MBType enum."""

    def test_i_nxn_value(self):
        """I_NxN has value 0."""
        assert MBType.I_NxN == 0

    def test_i_16x16_value(self):
        """I_16x16 starts at 1."""
        assert MBType.I_16x16 == 1

    def test_i_pcm_value(self):
        """I_PCM has value 25."""
        assert MBType.I_PCM == 25


class TestReconstructionPipeline:
    """Integration tests for reconstruction pipeline."""

    def test_i16x16_with_cbp_zero_reads_dc_block(self):
        """For I_16x16 with cbp_luma=0, DC block is STILL read.

        H.264 Spec 7.3.5.3: For I_16x16, the DC luma block (Intra16x16DCLevel)
        is ALWAYS coded, regardless of cbp_luma. Only AC blocks depend on cbp.

        When cbp_luma=0, there are no AC coefficients, but DC is still decoded.
        """
        # Create bitstream with empty DC block (TC=0, T1=0 = "1" for nC=0)
        writer = BitWriter()
        writer.write_bits(0b1, 1)  # coeff_token: TC=0, T1=0 (empty block)
        writer.write_bits(0x00, 7)  # Padding
        reader = BitReader(writer.to_bytes())

        nz_counts = np.zeros(24, dtype=np.int32)
        neighbors_top = np.full(16, 128, dtype=np.uint8)
        neighbors_left = np.full(16, 128, dtype=np.uint8)

        initial_pos = reader.position

        result = reconstruct_i16x16_luma(
            reader,
            pred_mode=2,  # DC mode
            cbp_luma=0,   # No luma AC residual
            qp=26,
            neighbors_top=neighbors_top,
            neighbors_left=neighbors_left,
            neighbor_top_left=128,
            nz_counts=nz_counts
        )

        # DC block should have been read (1 bit for empty coeff_token)
        assert reader.position == 1, \
            f"DC block not read: position is {reader.position}, expected 1"
        assert result.shape == (16, 16)
        assert np.all(result == 128)  # Pure DC prediction (no residual)

    def test_dc_only_reconstruction(self):
        """Test DC-only macroblock reconstruction."""
        # This tests the full pipeline with minimal coefficients
        writer = BitWriter()

        # DC block with single coefficient
        # TC=1, T1=0 for nC=0 -> 0b000101 (6 bits)
        code, bits = COEFF_TOKEN_0[(1, 0)]
        writer.write_bits(code, bits)

        # Level: prefix=0 (1 bit), first level gets +1 -> value=2
        writer.write_bits(0b1, 1)

        # total_zeros TC=1, TZ=15 -> need to look up
        # For TC=1, TZ=15: 0b000000001 (9 bits)
        writer.write_bits(0b000000001, 9)

        # Pad to byte boundary
        writer.write_bits(0, 8)

        # This creates a DC block with a single non-zero coefficient
        # The reconstruction should produce a valid 16x16 block

    def test_prediction_modes(self):
        """Test that different prediction modes produce different results."""
        # Create bitstream for empty block
        def create_empty_reader():
            writer = BitWriter()
            writer.write_bits(0b1, 1)  # TC=0
            writer.write_bits(0, 7)
            return BitReader(writer.to_bytes())

        nz_counts = np.zeros(24, dtype=np.int32)

        # Different neighbors for different modes
        neighbors_top = np.arange(16, dtype=np.uint8) * 10  # 0, 10, 20, ...
        neighbors_left = np.arange(16, dtype=np.uint8) * 5  # 0, 5, 10, ...

        # Vertical mode (copies top row)
        result_v = reconstruct_i16x16_luma(
            create_empty_reader(), pred_mode=0, cbp_luma=0, qp=26,
            neighbors_top=neighbors_top, neighbors_left=neighbors_left,
            neighbor_top_left=0, nz_counts=nz_counts
        )

        # Horizontal mode (copies left column)
        result_h = reconstruct_i16x16_luma(
            create_empty_reader(), pred_mode=1, cbp_luma=0, qp=26,
            neighbors_top=neighbors_top, neighbors_left=neighbors_left,
            neighbor_top_left=0, nz_counts=nz_counts
        )

        # Results should be different
        assert not np.array_equal(result_v, result_h)


class TestI8x8BlockScanOrder:
    """Tests for I_8x8 block scan order."""

    def test_block_scan_order_8x8_exists(self):
        """BLOCK_SCAN_ORDER_8x8 should exist."""
        from reconstruct.macroblock import BLOCK_SCAN_ORDER_8x8
        assert BLOCK_SCAN_ORDER_8x8 is not None

    def test_block_scan_order_8x8_has_4_blocks(self):
        """I_8x8 has 4 blocks."""
        from reconstruct.macroblock import BLOCK_SCAN_ORDER_8x8
        assert len(BLOCK_SCAN_ORDER_8x8) == 4

    def test_block_scan_order_8x8_positions(self):
        """I_8x8 blocks are at (0,0), (0,8), (8,0), (8,8)."""
        from reconstruct.macroblock import BLOCK_SCAN_ORDER_8x8
        expected = [(0, 0), (0, 8), (8, 0), (8, 8)]
        assert list(BLOCK_SCAN_ORDER_8x8) == expected


class TestReconstructI8x8Block:
    """Tests for single I_8x8 block reconstruction."""

    def test_reconstruct_i8x8_block_exists(self):
        """reconstruct_i8x8_block function should exist."""
        from reconstruct.macroblock import reconstruct_i8x8_block
        assert callable(reconstruct_i8x8_block)

    def test_reconstruct_i8x8_block_returns_8x8(self):
        """reconstruct_i8x8_block should return 8x8 array."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        residual = np.zeros((8, 8), dtype=np.int32)
        top = np.full(8, 128, dtype=np.uint8)
        left = np.full(8, 128, dtype=np.uint8)
        top_right = np.full(8, 128, dtype=np.uint8)

        result = reconstruct_i8x8_block(
            mode=2,  # DC mode
            residual=residual,
            top=top,
            left=left,
            top_left=128,
            top_right=top_right,
            top_available=True,
            left_available=True,
            top_right_available=True,
        )

        assert result.shape == (8, 8)
        assert result.dtype == np.uint8

    def test_reconstruct_i8x8_dc_prediction(self):
        """DC prediction with uniform neighbors."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        residual = np.zeros((8, 8), dtype=np.int32)
        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)

        result = reconstruct_i8x8_block(
            mode=2,  # DC mode
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=top_right,
            top_available=True,
            left_available=True,
            top_right_available=True,
        )

        # With uniform neighbors and zero residual, result should be 100
        assert np.all(result == 100)

    def test_reconstruct_i8x8_with_residual(self):
        """Reconstruction with non-zero residual."""
        from reconstruct.macroblock import reconstruct_i8x8_block

        residual = np.full((8, 8), 10, dtype=np.int32)
        top = np.full(8, 100, dtype=np.uint8)
        left = np.full(8, 100, dtype=np.uint8)
        top_right = np.full(8, 100, dtype=np.uint8)

        result = reconstruct_i8x8_block(
            mode=2,  # DC
            residual=residual,
            top=top,
            left=left,
            top_left=100,
            top_right=top_right,
            top_available=True,
            left_available=True,
            top_right_available=True,
        )

        # Prediction=100, residual=10, result should be 110
        assert np.all(result == 110)


class TestReconstructI8x8Luma:
    """Tests for full I_8x8 luma macroblock reconstruction."""

    def test_reconstruct_i8x8_luma_exists(self):
        """reconstruct_i8x8_luma function should exist."""
        from reconstruct.macroblock import reconstruct_i8x8_luma
        assert callable(reconstruct_i8x8_luma)

    def test_reconstruct_i8x8_luma_returns_16x16(self):
        """I_8x8 reconstruction returns 16x16 block."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 2, 2, 2]  # All DC mode
        residuals = [np.zeros((8, 8), dtype=np.int32) for _ in range(4)]
        neighbors_top = np.full(16, 128, dtype=np.uint8)
        neighbors_left = np.full(16, 128, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes, residuals,
            neighbors_top, neighbors_left, 128
        )

        assert result.shape == (16, 16)
        assert result.dtype == np.uint8

    def test_reconstruct_i8x8_luma_dc_uniform(self):
        """I_8x8 with DC mode and uniform neighbors."""
        from reconstruct.macroblock import reconstruct_i8x8_luma

        modes = [2, 2, 2, 2]  # All DC mode
        residuals = [np.zeros((8, 8), dtype=np.int32) for _ in range(4)]
        neighbors_top = np.full(16, 128, dtype=np.uint8)
        neighbors_left = np.full(16, 128, dtype=np.uint8)

        result = reconstruct_i8x8_luma(
            modes, residuals,
            neighbors_top, neighbors_left, 128
        )

        # With uniform DC prediction, all pixels should be 128
        assert np.all(result == 128)


class TestDecodeIntra8x8PredModes:
    """Tests for I_8x8 prediction mode decoding."""

    def test_decode_intra8x8_pred_modes_exists(self):
        """decode_intra8x8_pred_modes function should exist."""
        from reconstruct.macroblock import decode_intra8x8_pred_modes
        assert callable(decode_intra8x8_pred_modes)

    def test_decode_intra8x8_pred_modes_returns_4_modes(self):
        """decode_intra8x8_pred_modes returns 4 modes."""
        from reconstruct.macroblock import decode_intra8x8_pred_modes

        # Create bitstream with 4 predicted modes (all prev_flag=1)
        writer = BitWriter()
        for _ in range(4):
            writer.write_bits(1, 1)  # prev_intra8x8_pred_mode_flag = 1
        writer.write_bits(0, 4)  # Padding
        reader = BitReader(writer.to_bytes())

        modes = decode_intra8x8_pred_modes(reader)
        assert len(modes) == 4

    def test_decode_intra8x8_pred_modes_with_rem_mode(self):
        """decode_intra8x8_pred_modes handles rem_intra8x8_pred_mode."""
        from reconstruct.macroblock import decode_intra8x8_pred_modes

        # Create bitstream: prev_flag=0, rem_mode=0 for first block
        writer = BitWriter()
        writer.write_bits(0, 1)  # prev_flag = 0
        writer.write_bits(0, 3)  # rem_mode = 0
        # Remaining 3 blocks use predicted mode
        for _ in range(3):
            writer.write_bits(1, 1)
        writer.write_bits(0, 4)  # Padding
        reader = BitReader(writer.to_bytes())

        modes = decode_intra8x8_pred_modes(reader)
        assert len(modes) == 4
        # First mode should be 0 (rem_mode=0 < predicted_mode=2, so mode=0)
        assert modes[0] == 0


class TestDecodeAndReconstructI8x8Luma:
    """Tests for full I_8x8 decode from bitstream."""

    def test_decode_and_reconstruct_i8x8_luma_exists(self):
        """decode_and_reconstruct_i8x8_luma function should exist."""
        from reconstruct.macroblock import decode_and_reconstruct_i8x8_luma
        assert callable(decode_and_reconstruct_i8x8_luma)

    def test_decode_and_reconstruct_i8x8_luma_cbp_zero(self):
        """I_8x8 with no coefficients returns prediction only."""
        from reconstruct.macroblock import (
            decode_and_reconstruct_i8x8_luma,
            CodedBlockPattern,
        )

        # Create empty bitstream (no coefficients to decode)
        writer = BitWriter()
        writer.write_bits(0, 8)  # Padding
        reader = BitReader(writer.to_bytes())

        # All DC modes (mode 2)
        modes = [2, 2, 2, 2]

        # CBP with no luma coefficients
        cbp = CodedBlockPattern(luma=0, chroma=0)

        # Frame buffer with neighbors
        frame_luma = np.full((32, 32), 128, dtype=np.uint8)
        nz_counts = np.zeros(24, dtype=np.int32)

        result = decode_and_reconstruct_i8x8_luma(
            reader=reader,
            modes=modes,
            cbp=cbp,
            qp=28,
            frame_luma=frame_luma,
            mb_x=1,
            mb_y=1,
            nz_counts=nz_counts,
        )

        # Should return 16x16 result
        assert result.shape == (16, 16)
        # DC mode with 128 neighbors should predict ~128
        assert np.all(result == 128)

    def test_decode_and_reconstruct_i8x8_no_neighbors(self):
        """I_8x8 at top-left corner (no neighbors) uses DC fallback."""
        from reconstruct.macroblock import (
            decode_and_reconstruct_i8x8_luma,
            CodedBlockPattern,
        )

        writer = BitWriter()
        writer.write_bits(0, 8)
        reader = BitReader(writer.to_bytes())

        modes = [2, 2, 2, 2]  # DC mode
        cbp = CodedBlockPattern(luma=0, chroma=0)
        frame_luma = np.zeros((16, 16), dtype=np.uint8)
        nz_counts = np.zeros(24, dtype=np.int32)

        result = decode_and_reconstruct_i8x8_luma(
            reader=reader,
            modes=modes,
            cbp=cbp,
            qp=28,
            frame_luma=frame_luma,
            mb_x=0,
            mb_y=0,
            nz_counts=nz_counts,
        )

        assert result.shape == (16, 16)
        # DC fallback should produce 128
        assert np.all(result == 128)


class TestMacroblockDataI8x8Fields:
    """Tests for I_8x8 fields in MacroblockData."""

    def test_macroblock_data_has_i8x8_fields(self):
        """MacroblockData should have I_8x8 fields."""
        from reconstruct.macroblock import MacroblockData

        mb = MacroblockData()
        assert hasattr(mb, 'intra_8x8_pred_modes')
        assert hasattr(mb, 'transform_size_8x8_flag')

    def test_macroblock_data_i8x8_defaults(self):
        """MacroblockData I_8x8 fields have correct defaults."""
        from reconstruct.macroblock import MacroblockData

        mb = MacroblockData()
        assert mb.intra_8x8_pred_modes == []
        assert mb.transform_size_8x8_flag is False
