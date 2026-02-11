# entropy/tests/test_vlc_validation.py
"""Test VLC bit consumption validation utilities."""
import pytest
from bitstream import BitReader
from entropy.cavlc import validate_vlc_bits_consumed


def test_validate_vlc_bits_consumed_correct():
    """Validation passes when bit consumption matches."""
    data = bytes([0b10110000])  # ue(2) = '011'
    reader = BitReader(data)
    pos_before = reader.position

    reader.read_bits(3)  # Read '011'

    # Should not raise
    validate_vlc_bits_consumed(
        reader, pos_before, expected_bits=3,
        context="test_ue", code_value=2
    )


def test_validate_vlc_bits_consumed_mismatch():
    """Validation fails when bit consumption doesn't match."""
    data = bytes([0b10110000])
    reader = BitReader(data)
    pos_before = reader.position

    reader.read_bits(2)  # Read only 2 bits instead of 3

    with pytest.raises(ValueError, match="VLC bit consumption mismatch"):
        validate_vlc_bits_consumed(
            reader, pos_before, expected_bits=3,
            context="test_ue", code_value=2
        )
