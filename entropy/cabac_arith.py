# h264/entropy/cabac_arith.py
"""CABAC binary arithmetic decoder.

The arithmetic decoder maintains:
- codIRange: Current interval range (256-510 after renormalization)
- codIOffset: Current position within interval (from bitstream)

Decoding process:
1. Calculate LPS range from rangeTabLPS[pStateIdx][qCodIRangeIdx]
2. Compare codIOffset with threshold (codIRange - codIRangeLPS)
3. If codIOffset < threshold: MPS decoded, narrow range to upper part
4. If codIOffset >= threshold: LPS decoded, narrow to lower part
5. Update context probability via state transition tables
6. Renormalize range to [256, 510]

H.264 Spec Reference: Section 9.3.3 - Arithmetic decoding process
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bitstream import BitReader
    from entropy.cabac_context import CABACContext


# State transition table for LPS (H.264 Table 9-45)
# transIdxLPS[pStateIdx] gives new state after decoding LPS
TRANS_IDX_LPS = [
    0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
]

# State transition table for MPS (H.264 Table 9-45)
# transIdxMPS[pStateIdx] gives new state after decoding MPS
TRANS_IDX_MPS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
]

# LPS range table (H.264 Table 9-46)
# rangeTabLPS[pStateIdx][qCodIRangeIdx] where qCodIRangeIdx = (codIRange >> 6) & 3
RANGE_TAB_LPS = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216],
    [123, 150, 178, 205], [116, 142, 169, 195], [111, 135, 160, 185],
    [105, 128, 152, 175], [100, 122, 144, 166], [95, 116, 137, 158],
    [90, 110, 130, 150], [85, 104, 123, 142], [81, 99, 117, 135],
    [77, 94, 111, 128], [73, 89, 105, 122], [69, 85, 100, 116],
    [66, 80, 95, 110], [62, 76, 90, 104], [59, 72, 86, 99],
    [56, 69, 81, 94], [53, 65, 77, 89], [51, 62, 73, 85],
    [48, 59, 69, 80], [46, 56, 66, 76], [43, 53, 63, 72],
    [41, 50, 59, 69], [39, 48, 56, 65], [37, 45, 54, 62],
    [35, 43, 51, 59], [33, 41, 48, 56], [32, 39, 46, 53],
    [30, 37, 43, 50], [29, 35, 41, 48], [27, 33, 39, 45],
    [26, 31, 37, 43], [24, 30, 35, 41], [23, 28, 33, 39],
    [22, 27, 32, 37], [21, 26, 30, 35], [20, 24, 29, 33],
    [19, 23, 27, 31], [18, 22, 26, 30], [17, 21, 25, 28],
    [16, 20, 23, 27], [15, 19, 22, 25], [14, 18, 21, 24],
    [14, 17, 20, 23], [13, 16, 19, 22], [12, 15, 18, 21],
    [12, 14, 17, 20], [11, 14, 16, 19], [11, 13, 15, 18],
    [10, 12, 15, 17], [10, 12, 14, 16], [9, 11, 13, 15],
    [9, 11, 12, 14], [8, 10, 12, 14], [8, 9, 11, 13],
    [7, 9, 11, 12], [7, 9, 10, 12], [7, 8, 10, 11],
    [6, 8, 9, 11], [6, 7, 9, 10], [6, 7, 8, 9],
    [2, 2, 2, 2],
]


class CABACDecoder:
    """CABAC binary arithmetic decoder.

    Attributes:
        codIRange: Current interval range (256-510)
        codIOffset: Current position within interval
        reader: BitReader for input stream
    """

    def __init__(self, reader: 'BitReader'):
        """Initialize decoder from bitstream.

        Args:
            reader: BitReader positioned at start of CABAC data
        """
        self.reader = reader

        # Initialize range to 510 (spec section 9.3.1.2)
        self.codIRange = 510

        # Read initial 9 bits for codIOffset
        self.codIOffset = reader.read_bits(9)

    def decode_decision(self, ctx: 'CABACContext') -> int:
        """Decode one bin using context-based arithmetic decoding.

        H.264 Section 9.3.3.2 - Arithmetic decoding process for a binary decision

        Args:
            ctx: Context model (will be updated)

        Returns:
            Decoded bin value (0 or 1)
        """
        # Get range index for table lookup
        q_cod_i_range_idx = (self.codIRange >> 6) & 3

        # Get LPS range from table
        cod_i_range_lps = RANGE_TAB_LPS[ctx.pStateIdx][q_cod_i_range_idx]

        # Narrow range for MPS
        self.codIRange -= cod_i_range_lps

        if self.codIOffset >= self.codIRange:
            # LPS decoded
            bin_val = 1 - ctx.valMPS
            self.codIOffset -= self.codIRange
            self.codIRange = cod_i_range_lps

            # State transition for LPS
            if ctx.pStateIdx == 0:
                # Switch MPS
                ctx.valMPS = 1 - ctx.valMPS

            ctx.pStateIdx = TRANS_IDX_LPS[ctx.pStateIdx]
        else:
            # MPS decoded
            bin_val = ctx.valMPS

            # State transition for MPS
            ctx.pStateIdx = TRANS_IDX_MPS[ctx.pStateIdx]

        # Renormalize
        self.renormalize()

        return bin_val

    def decode_bypass(self) -> int:
        """Decode one bin using bypass (equiprobable) mode.

        H.264 Section 9.3.3.2.3 - Bypass decoding process

        Returns:
            Decoded bin value (0 or 1)
        """
        # Double the offset
        self.codIOffset = (self.codIOffset << 1) | self.reader.read_bits(1)

        if self.codIOffset >= self.codIRange:
            # Decode 1
            self.codIOffset -= self.codIRange
            return 1
        else:
            # Decode 0
            return 0

    def decode_terminate(self) -> int:
        """Decode end_of_slice_flag.

        H.264 Section 9.3.3.2.4 - Decoding process for end_of_slice_flag

        Returns:
            1 if end of slice, 0 otherwise
        """
        # Reduce range by 2
        self.codIRange -= 2

        if self.codIOffset >= self.codIRange:
            # End of slice
            return 1
        else:
            # Not end, renormalize
            self.renormalize()
            return 0

    def renormalize(self) -> None:
        """Renormalize range to maintain precision.

        H.264 Section 9.3.3.2.2 - Renormalization process

        Keeps codIRange in [256, 510] by doubling when needed.
        """
        while self.codIRange < 256:
            self.codIRange <<= 1
            self.codIOffset = (self.codIOffset << 1) | self.reader.read_bits(1)

    def byte_align(self) -> None:
        """Align bitstream reader to next byte boundary.

        Used after I_PCM macroblock to resume CABAC from byte-aligned position.
        Reinitializes the arithmetic decoder state.

        H.264 Section 9.3.1.2 - Initialization process
        """
        # Align underlying reader to byte boundary
        self.reader.byte_align()

        # Reinitialize CABAC state
        self.codIRange = 510
        self.codIOffset = self.reader.read_bits(9)

    def init_after_pcm(self) -> None:
        self.reader.byte_align()
        self.codIRange = 510
        self.codIOffset = self.reader.read_bits(9)

    def read_bits(self, n: int) -> int:
        """Read n bypass bits for I_PCM samples.

        Args:
            n: Number of bits to read

        Returns:
            Value read from bitstream
        """
        return self.reader.read_bits(n)
