# h264/parameters/__init__.py
"""H.264 Parameter Set parsing module.

Handles parsing of SPS (Sequence Parameter Set) and PPS (Picture Parameter Set)
which contain the configuration data needed for video decoding.

H.264 Spec Reference:
- Section 7.3.2: Parameter set syntax
- Section 7.4.2: Parameter set semantics
"""

from .sps import (
    SPS,
    VUIParameters,
    parse_sps,
    PROFILE_BASELINE,
    PROFILE_MAIN,
    PROFILE_EXTENDED,
    PROFILE_HIGH,
    PROFILE_HIGH_10,
    PROFILE_HIGH_422,
    PROFILE_HIGH_444,
)

from .pps import (
    PPS,
    parse_pps,
)

__all__ = [
    # SPS
    "SPS",
    "VUIParameters",
    "parse_sps",
    "PROFILE_BASELINE",
    "PROFILE_MAIN",
    "PROFILE_EXTENDED",
    "PROFILE_HIGH",
    "PROFILE_HIGH_10",
    "PROFILE_HIGH_422",
    "PROFILE_HIGH_444",
    # PPS
    "PPS",
    "parse_pps",
]
