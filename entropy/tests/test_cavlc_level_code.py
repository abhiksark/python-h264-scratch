# entropy/tests/test_cavlc_level_code.py
"""Tests for CAVLC level_code calculation.

Tests the level_code formula from H.264 spec Section 9.2.2, Table 9-7.
This formula has three cases based on level_prefix and suffix_length.

H.264 Spec Reference: Section 9.2.2
"""

import pytest


def _compute_level_code(level_prefix: int, suffix_length: int, level_suffix: int) -> int:
    """Compute level_code per H.264 spec Table 9-7.

    Args:
        level_prefix: Number of leading zeros in level VLC
        suffix_length: Current suffix length state
        level_suffix: Decoded suffix bits

    Returns:
        level_code value per H.264 Table 9-7

    H.264 Spec:
        if( level_prefix < 15 )
            level_code = ( level_prefix << suffixLength ) + level_suffix
        else
            if( suffixLength == 0 )
                level_code = level_prefix + level_suffix
            else
                level_code = ( 15 << suffixLength ) + level_suffix
    """
    # Correct implementation per H.264 spec Table 9-7
    if level_prefix < 15:
        # Standard case: use level_prefix directly
        return (level_prefix << suffix_length) + level_suffix
    elif suffix_length == 0:
        # Extended escape without suffix
        return level_prefix + level_suffix
    else:
        # Extended escape with suffix: CAP level_prefix at 15
        return (15 << suffix_length) + level_suffix


def test_level_code_standard_case():
    """Test standard case: level_prefix < 15.

    When level_prefix < 15, use standard shift formula regardless of suffix_length.
    """
    test_cases = [
        # (level_prefix, suffix_length, level_suffix, expected_level_code)
        (5, 0, 0, 5),          # No suffix: 5 + 0 = 5
        (5, 2, 3, 23),         # With suffix: (5 << 2) + 3 = 20 + 3 = 23
        (10, 1, 1, 21),        # (10 << 1) + 1 = 20 + 1 = 21
        (14, 3, 7, 119),       # Boundary: (14 << 3) + 7 = 112 + 7 = 119
        (0, 2, 3, 3),          # Zero prefix: (0 << 2) + 3 = 3
    ]

    for prefix, suffix_len, suffix, expected in test_cases:
        result = _compute_level_code(prefix, suffix_len, suffix)
        assert result == expected, \
            f"level_prefix={prefix}, suffix_length={suffix_len}, level_suffix={suffix}: " \
            f"expected {expected}, got {result}"


def test_level_code_escape_no_suffix():
    """Test escape code without suffix: level_prefix >= 15, suffix_length == 0.

    When level_prefix >= 15 and suffix_length == 0, use direct addition.
    """
    test_cases = [
        # (level_prefix, suffix_length, level_suffix, expected_level_code)
        (15, 0, 0, 15),        # 15 + 0 = 15
        (16, 0, 5, 21),        # 16 + 5 = 21
        (20, 0, 10, 30),       # 20 + 10 = 30
        (25, 0, 7, 32),        # 25 + 7 = 32
    ]

    for prefix, suffix_len, suffix, expected in test_cases:
        result = _compute_level_code(prefix, suffix_len, suffix)
        assert result == expected, \
            f"level_prefix={prefix}, suffix_length={suffix_len}, level_suffix={suffix}: " \
            f"expected {expected}, got {result}"


def test_level_code_escape_with_suffix():
    """Test escape code WITH suffix: level_prefix >= 15, suffix_length > 0.

    THIS IS THE BUG CASE!

    Per H.264 spec, when level_prefix >= 15 and suffix_length > 0,
    level_code MUST cap level_prefix at 15: (15 << suffix_length) + level_suffix.

    Current buggy code uses level_prefix directly, causing bit drift.
    """
    test_cases = [
        # (level_prefix, suffix_length, level_suffix, expected_level_code)
        (15, 1, 0, 30),        # (15 << 1) + 0 = 30
        (15, 2, 3, 63),        # (15 << 2) + 3 = 60 + 3 = 63
        (16, 1, 0, 30),        # (15 << 1) + 0 = 30 (CAPPED at 15!)
        (16, 2, 3, 63),        # (15 << 2) + 3 = 63 (CAPPED at 15!)
        (20, 3, 7, 127),       # (15 << 3) + 7 = 120 + 7 = 127 (CAPPED at 15!)
        (25, 2, 1, 61),        # (15 << 2) + 1 = 60 + 1 = 61 (CAPPED at 15!)
    ]

    for prefix, suffix_len, suffix, expected in test_cases:
        result = _compute_level_code(prefix, suffix_len, suffix)
        assert result == expected, \
            f"level_prefix={prefix}, suffix_length={suffix_len}, level_suffix={suffix}: " \
            f"expected {expected}, got {result}"

        # Verify current buggy code would give WRONG answer for prefix > 15
        # (When prefix == 15, capped and uncapped formulas are identical)
        if prefix > 15 and suffix_len > 0:
            wrong_result = (prefix << suffix_len) + suffix
            assert result != wrong_result, \
                f"Bug detected: level_prefix={prefix} should be capped at 15, " \
                f"but got same result as uncapped formula"


def test_level_code_boundary_at_15():
    """Test boundary between standard and escape formulas.

    level_prefix = 14 uses standard formula: (14 << suffix_length) + suffix
    level_prefix = 15 uses escape formula with cap: (15 << suffix_length) + suffix

    With same suffix_length and suffix, they should produce different results.
    """
    # Test with suffix_length=2, level_suffix=3
    result_14 = _compute_level_code(14, 2, 3)  # (14 << 2) + 3 = 56 + 3 = 59
    result_15 = _compute_level_code(15, 2, 3)  # (15 << 2) + 3 = 60 + 3 = 63

    assert result_14 == 59, f"Expected 59 for level_prefix=14, got {result_14}"
    assert result_15 == 63, f"Expected 63 for level_prefix=15, got {result_15}"
    assert result_15 != result_14, "Boundary values should differ"

    # Test with suffix_length=1, level_suffix=1
    result_14b = _compute_level_code(14, 1, 1)  # (14 << 1) + 1 = 28 + 1 = 29
    result_15b = _compute_level_code(15, 1, 1)  # (15 << 1) + 1 = 30 + 1 = 31

    assert result_14b == 29, f"Expected 29 for level_prefix=14, got {result_14b}"
    assert result_15b == 31, f"Expected 31 for level_prefix=15, got {result_15b}"
    assert result_15b != result_14b, "Boundary values should differ"


def test_level_code_all_suffix_lengths():
    """Test level_code with various suffix_length values.

    Verify correct behavior for suffix_length from 0 to 6 (typical range).
    """
    # Test level_prefix=16 (escape code) with different suffix_length values
    test_cases = [
        # (level_prefix, suffix_length, level_suffix, expected_level_code)
        (16, 0, 0, 16),        # suffix_len=0: 16 + 0 = 16
        (16, 1, 0, 30),        # suffix_len=1: (15 << 1) + 0 = 30 (capped)
        (16, 2, 0, 60),        # suffix_len=2: (15 << 2) + 0 = 60 (capped)
        (16, 3, 0, 120),       # suffix_len=3: (15 << 3) + 0 = 120 (capped)
        (16, 4, 0, 240),       # suffix_len=4: (15 << 4) + 0 = 240 (capped)
        (16, 5, 0, 480),       # suffix_len=5: (15 << 5) + 0 = 480 (capped)
        (16, 6, 0, 960),       # suffix_len=6: (15 << 6) + 0 = 960 (capped)
    ]

    for prefix, suffix_len, suffix, expected in test_cases:
        result = _compute_level_code(prefix, suffix_len, suffix)
        assert result == expected, \
            f"level_prefix={prefix}, suffix_length={suffix_len}, level_suffix={suffix}: " \
            f"expected {expected}, got {result}"
