# entropy/tests/test_levels_bits.py
"""Test level VLC decoding consumes correct number of bits.

H.264 Spec Reference: Section 9.2.2

Level decoding uses adaptive VLC with:
- level_prefix: unary code (N zeros + 1 terminator = N+1 bits)
- level_suffix: variable bits depending on level_prefix and suffix_length state
- suffix_bits calculation:
  * level_prefix < 14: suffix_bits = suffix_length
  * level_prefix == 14: suffix_bits = 4 if suffix_length==0 else suffix_length
  * level_prefix >= 15: suffix_bits = (level_prefix - 3) if suffix_length==0 else suffix_length

Expected total bits per level: level_prefix + 1 + suffix_bits
"""
import pytest
from bitstream import BitReader
from entropy.cavlc import decode_levels


class TestLevelBitsSimple:
    """Test basic level codes with suffix_length=0."""

    def test_level_prefix_0_suffix_0(self):
        """Level with prefix=0, suffix_length=0 -> 1 bit total.

        Code: '1' (terminator only)
        level_code=0 -> level=1, but with trailing_ones=0, first level gets +1 -> level=2
        Expected: level=2, 1 bit consumed
        """
        # Code: '1' (prefix=0, no suffix)
        data = bytes([0b10000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert levels[0] == 2  # Magnitude adjustment: 1 + 1 = 2
        assert reader.position == 1

    def test_level_prefix_1_suffix_0(self):
        """Level with prefix=1, suffix_length=0 -> 2 bits total.

        Code: '01' (1 zero + terminator)
        level_code=1 -> level=-1, but with trailing_ones=0, first level gets -1 -> level=-2
        Expected: level=-2, 2 bits consumed
        """
        data = bytes([0b01000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert levels[0] == -2  # Magnitude adjustment: -1 - 1 = -2
        assert reader.position == 2

    def test_level_prefix_2_suffix_0(self):
        """Level with prefix=2, suffix_length=0 -> 3 bits total.

        Code: '001' (2 zeros + terminator)
        level_code=2 -> level=2, but with trailing_ones=0, first level gets +1 -> level=3
        Expected: level=3, 3 bits consumed
        """
        data = bytes([0b00100000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert levels[0] == 3  # Magnitude adjustment: 2 + 1 = 3
        assert reader.position == 3

    def test_level_prefix_3_suffix_0(self):
        """Level with prefix=3, suffix_length=0 -> 4 bits total.

        Code: '0001' (3 zeros + terminator)
        level_code=3 -> level=-2, but with trailing_ones=0, first level gets -1 -> level=-3
        Expected: level=-3, 4 bits consumed
        """
        data = bytes([0b00010000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert levels[0] == -3  # Magnitude adjustment: -2 - 1 = -3
        assert reader.position == 4


class TestLevelBitsSuffixLength1:
    """Test level codes with suffix_length=1."""

    def test_level_prefix_0_suffix_1(self):
        """First level triggers suffix_length=1, then second level uses it.

        First level: prefix=0, suffix_length=0 -> '1' (1 bit)
        After first level: suffix_length becomes 1
        Second level: prefix=0, suffix_length=1 -> '1' + '0' (2 bits)
        Total: 3 bits
        """
        # First: '1' (level=1), Second: '10' (prefix=0, suffix=0)
        data = bytes([0b11000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=2, trailing_ones=0)

        assert len(levels) == 2
        assert reader.position == 3

    def test_level_prefix_2_suffix_1(self):
        """Test second level uses suffix_length=1 from state transition.

        First level: prefix=0, suffix_length=0 -> 1 bit, suffix_length becomes 1
        Second level: prefix=2, suffix_length=1 -> 2+1+1=4 bits
        Total: 1 + 4 = 5 bits
        """
        # First level: '1' (1 bit)
        # Second level: '001' + '0' (4 bits)
        data = bytes([0b10010000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=2, trailing_ones=0)

        assert len(levels) == 2
        assert reader.position == 5


class TestLevelBitsSuffixLength2:
    """Test level codes with suffix_length=2."""

    def test_level_prefix_0_suffix_2(self):
        """Level with prefix=0, suffix_length=2 -> 1+2=3 bits.

        Setup: Decode two levels to reach suffix_length=2
        - First level: prefix=2, suffix_length=0 -> level=2, suffix_length becomes 1
        - Second level: prefix=3, suffix_length=1 -> level magnitude > threshold, suffix_length becomes 2
        - Third level: prefix=0, suffix_length=2 -> 1+2=3 bits
        """
        # First: '001' (prefix=2, suffix_length=0, level=2) = 3 bits, suffix_length->1
        # Second: '0001' + '0' (prefix=3, suffix_length=1) = 5 bits, suffix_length->2
        # Third: '1' + '00' (prefix=0, suffix_length=2) = 3 bits
        # Total: 3 + 5 + 3 = 11 bits
        data = bytes([0b00100010, 0b10000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=3, trailing_ones=0)

        assert len(levels) == 3
        assert reader.position == 11


class TestLevelBitsPrefix14:
    """Test special case: level_prefix==14."""

    def test_prefix_14_suffix_length_0(self):
        """Level_prefix=14 with suffix_length=0 uses 4 suffix bits.

        Code: 14 zeros + terminator + 4 suffix bits = 15 + 4 = 19 bits
        """
        # 14 zeros + '1' + 4 suffix bits '0000'
        data = bytes([0b00000000, 0b00000010, 0b00000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert reader.position == 19

    def test_prefix_14_suffix_length_1(self):
        """Level_prefix=14 with suffix_length=1 uses 1 suffix bit.

        Setup: Build up to suffix_length=1 with first level
        First level: prefix=0 -> suffix_length becomes 1
        Second level: prefix=14, suffix_length=1 -> 14+1+1=16 bits
        Total: 1 + 16 = 17 bits
        """
        # First level: '1' (1 bit)
        # Second level: 14 zeros + '1' + '0' (16 bits)
        data = bytes([0b10000000, 0b00000001, 0b00000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=2, trailing_ones=0)

        assert len(levels) == 2
        assert reader.position == 17

    def test_prefix_14_suffix_length_2(self):
        """Level_prefix=14 with suffix_length=2 uses 2 suffix bits.

        Setup: Build up to suffix_length=2
        First: prefix=2, suffix_length=0 -> 3 bits, level=3, suffix_length->1
        Second: prefix=3, suffix_length=1 -> 5 bits, level=-4, suffix_length->2
        Third: prefix=14, suffix_length=2 -> 17 bits
        Total: 3 + 5 + 17 = 25 bits
        """
        # First: '001' (3 bits)
        # Second: '0001' + '0' (5 bits)
        # Third: 14 zeros + '1' + '00' (17 bits)
        data = bytes([0b00100010, 0b00000000, 0b00000010, 0b00000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=3, trailing_ones=0)

        assert len(levels) == 3
        assert reader.position == 25


class TestLevelBitsExtendedEscape:
    """Test extended escape: level_prefix >= 15."""

    def test_prefix_15_suffix_length_0(self):
        """Level_prefix=15 with suffix_length=0 uses (15-3)=12 suffix bits.

        Code: 15 zeros + terminator + 12 suffix bits = 16 + 12 = 28 bits
        """
        # 15 zeros + '1' + 12 suffix bits (all zeros for simplicity)
        data = bytes([
            0b00000000,  # 8 zeros
            0b00000001,  # 7 zeros + terminator
            0b00000000, 0b00000000,  # 12 suffix bits (2 bytes but need only 12 bits)
        ])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert reader.position == 28

    def test_prefix_16_suffix_length_0(self):
        """Level_prefix=16 with suffix_length=0 uses (16-3)=13 suffix bits.

        Code: 16 zeros + terminator + 13 suffix bits = 17 + 13 = 30 bits
        """
        # 16 zeros + '1' + 13 suffix bits
        data = bytes([
            0b00000000,  # 8 zeros
            0b00000000,  # 8 zeros
            0b10000000, 0b00000000,  # terminator + 13 suffix bits
        ])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert reader.position == 30

    def test_prefix_15_suffix_length_1(self):
        """Level_prefix=15 with suffix_length=1: ALWAYS uses level_prefix-3=12 suffix bits.

        Per H.264 spec 9.2.2, escape (prefix >= 15) ALWAYS reads
        level_prefix - 3 suffix bits regardless of suffix_length.
        First level: prefix=0 -> 1 bit, suffix_length becomes 1
        Second level: prefix=15, suffix_bits=12 -> 16+12=28 bits
        Total: 1 + 28 = 29 bits
        """
        # First: '1' (1 bit)
        # Second: 15 zeros + '1' + 12 suffix bits (all zeros)
        data = bytes([0b10000000, 0b00000000, 0b10000000, 0b00000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=2, trailing_ones=0)

        assert len(levels) == 2
        assert reader.position == 29

    def test_prefix_20_suffix_length_0(self):
        """Level_prefix=20 with suffix_length=0 uses (20-3)=17 suffix bits.

        Code: 20 zeros + terminator + 17 suffix bits = 21 + 17 = 38 bits
        """
        # 20 zeros + '1' + 17 suffix bits
        data = bytes([
            0b00000000,  # 8 zeros
            0b00000000,  # 8 zeros
            0b00001000,  # 4 zeros + terminator + 3 suffix bits
            0b00000000, 0b00000000,  # 14 more suffix bits
        ])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert reader.position == 38


class TestLevelBitsMultipleLevels:
    """Test bit consumption across multiple levels with suffix_length state changes."""

    def test_two_levels_suffix_state_transition(self):
        """Test suffix_length state transition between levels.

        First level: prefix=0, suffix_length=0 -> 1 bit, level=1, suffix_length->1
        Second level: prefix=1, suffix_length=1 -> 2+1=3 bits
        Total: 4 bits
        """
        # First: '1' (1 bit)
        # Second: '01' + '0' (3 bits)
        data = bytes([0b10100000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=2, trailing_ones=0)

        assert len(levels) == 2
        assert reader.position == 4

    def test_three_levels_progressive_suffix_increase(self):
        """Test suffix_length increasing through multiple levels.

        Start: total_coeff=3, trailing_ones=0, suffix_length=0
        Level 1: prefix=2 -> level=2, suffix_length->1 (abs(level)=2 > threshold=0)
        Level 2: prefix=3, suffix=0 -> level=-4, suffix_length->2 (abs(level)=4 > threshold=3)
        Level 3: prefix=0, suffix=0b11 -> suffix_length=2, 3 bits
        """
        # Level 1: '001' (3 bits)
        # Level 2: '0001' + '0' (5 bits)
        # Level 3: '1' + '11' (3 bits)
        # Total: 11 bits
        data = bytes([0b00100010, 0b11100000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=3, trailing_ones=0)

        assert len(levels) == 3
        assert reader.position == 11

    def test_four_levels_reach_suffix_length_3(self):
        """Test reaching suffix_length=3.

        Suffix_length increases when abs(level) > threshold.
        Note: first level gets magnitude +1 adjustment (trailing_ones=0).

        Level 0: prefix=4, suffix_length=0 -> 5 bits, level=4 (adjusted)
            suffix_length: 0->1->2 (abs(4) > 3)
        Level 1: prefix=7, suffix=01, suffix_length=2 -> 10 bits, level=-15
            suffix_length: 2->3 (abs(15) > 6)
        Level 2: prefix=5, suffix=000, suffix_length=3 -> 9 bits, level=21
            suffix_length: 3->4 (abs(21) > 12)
        Level 3: prefix=0, suffix=0000, suffix_length=4 -> 5 bits, level=1
        Total: 5 + 10 + 9 + 5 = 29 bits
        """
        data = bytes([0b00001000, 0b00001010, 0b00001000, 0b10000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=4, trailing_ones=0)

        assert len(levels) == 4
        assert reader.position == 29


class TestLevelBitsTrailingOnesImpact:
    """Test how trailing_ones affects level decoding."""

    def test_first_level_magnitude_adjustment_t1_lt_3(self):
        """When trailing_ones < 3, first level magnitude is increased by 1.

        trailing_ones=0: first level magnitude += 1
        Code: '1' (prefix=0) with suffix_length=0
        Normal: level_code=0 -> level=1
        Adjusted: level=1+1=2 (positive) or -1-1=-2 (negative)
        Bits consumed: 1 bit
        """
        data = bytes([0b10000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        # After adjustment, level should be 2 (not 1)
        assert abs(levels[0]) >= 2
        assert reader.position == 1

    def test_no_magnitude_adjustment_when_t1_eq_3(self):
        """When trailing_ones = 3, no magnitude adjustment.

        trailing_ones=3: no adjustment to first level
        This is rare but valid case.
        """
        # With trailing_ones=3, total_coeff must be >= 3
        # First non-T1 level has no adjustment
        data = bytes([0b10000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=4, trailing_ones=3)

        assert len(levels) == 1
        assert reader.position == 1

    def test_multiple_levels_with_t1_eq_2(self):
        """Test multiple levels with trailing_ones=2.

        First level gets magnitude adjustment.
        Bits consumed should still be correct.
        """
        # Two levels, trailing_ones=2
        # Level 1: '1' (1 bit)
        # Level 2: '01' + '0' (3 bits, suffix_length=1 after first)
        data = bytes([0b10100000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=4, trailing_ones=2)

        assert len(levels) == 2
        assert reader.position == 4


class TestLevelBitsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_level_prefix_13(self):
        """Test boundary before prefix=14 special case.

        prefix=13, suffix_length=0 -> 13 zeros + terminator + 0 suffix = 14 bits
        """
        # 13 zeros + '1' + padding
        # Byte 1: 8 zeros, Byte 2: 5 zeros + '1' + 2 bits padding
        data = bytes([0b00000000, 0b00000100])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert reader.position == 14

    def test_suffix_length_6_max(self):
        """Test higher suffix_length values work correctly.

        Build up to suffix_length=3.
        Note: first level gets magnitude +1 adjustment (trailing_ones=0).

        Level 0: prefix=6, suffix_length=0 -> 7 bits, level=5 (adjusted)
            suffix_length: 0->1->2 (abs(5) > 3)
        Level 1: prefix=5, suffix=00, suffix_length=2 -> 8 bits, level=11
            suffix_length: 2->3 (abs(11) > 6)
        Level 2: prefix=3, suffix=010, suffix_length=3 -> 7 bits, level=14
        Total: 7 + 8 + 7 = 22 bits
        """
        data = bytes([0b00000010, 0b00001000, 0b00101000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=3, trailing_ones=0)

        assert len(levels) == 3
        assert reader.position == 22

    def test_no_levels_when_t1_equals_tc(self):
        """Test when all coefficients are trailing ones.

        total_coeff=3, trailing_ones=3 -> no levels to decode
        """
        data = bytes([0b00000000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=3, trailing_ones=3)

        assert len(levels) == 0
        assert reader.position == 0  # No bits consumed


class TestLevelBitsRealWorldPatterns:
    """Test realistic level patterns from actual H.264 streams."""

    def test_typical_inter_block_pattern(self):
        """Test typical inter block: few small levels.

        Common pattern: 1-3 levels with small magnitudes
        """
        # 3 levels: prefix=0, prefix=1, prefix=0 with suffix_length transitions
        # Level 1: '1' (1 bit), suffix_length->1
        # Level 2: '01' + '1' (3 bits), suffix_length stays 1
        # Level 3: '1' + '0' (2 bits)
        # Total: 6 bits
        data = bytes([0b10111000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=3, trailing_ones=0)

        assert len(levels) == 3
        assert reader.position == 6

    def test_dc_coefficient_large_value(self):
        """Test DC coefficient with large magnitude.

        DC coefficients often have larger values.
        """
        # Large level: prefix=10, suffix_length=0 -> 11 bits
        # 10 zeros + '1' + padding
        data = bytes([0b00000000, 0b00100000])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=1, trailing_ones=0)

        assert len(levels) == 1
        assert reader.position == 11

    def test_mixed_large_and_small_levels(self):
        """Test mix of large and small levels.

        Level 0: prefix=2, suffix_length=0 -> 3 bits, level=3 (adjusted)
            suffix_length: 0->1, abs(3) <= 3 -> stays 1
        Level 1: prefix=1, suffix=0, suffix_length=1 -> 3 bits, level=2
        Level 2: prefix=0, suffix=1, suffix_length=1 -> 2 bits, level=-1
        Total: 3 + 3 + 2 = 8 bits
        """
        data = bytes([0b00101011])
        reader = BitReader(data)

        levels = decode_levels(reader, total_coeff=3, trailing_ones=0)

        assert len(levels) == 3
        assert reader.position == 8
