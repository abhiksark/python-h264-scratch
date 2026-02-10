# reconstruct/mb_state.py
"""Macroblock decoding state machine.

H.264 Spec: Section 7.3.5 - Macroblock layer syntax
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from bitstream.bit_reader import BitReader


class MBState(Enum):
    """Macroblock decode states following H.264 spec order."""

    START_MB = auto()        # Initial state, MB header parsed
    LUMA_RESIDUAL = auto()   # Decoding luma blocks 0-15
    CHROMA_DC = auto()       # Decoding Cb DC + Cr DC
    CHROMA_AC = auto()       # Decoding Cb AC + Cr AC (4 blocks each)
    MB_COMPLETE = auto()     # MB fully decoded


class MBStateValidationError(Exception):
    """Exception raised when state machine validation fails.

    Includes full context for debugging: current state, bit position,
    what was expected vs what was found.
    """

    def __init__(
        self,
        message: str,
        context: 'MBDecodeContext',
        expected_state: Optional[MBState] = None,
        actual_state: Optional[MBState] = None,
    ):
        self.message = message
        self.context = context
        self.expected_state = expected_state
        self.actual_state = actual_state

        # Build detailed error message
        error_parts = [
            f"MB State Validation Error at MB({context.mb_x},{context.mb_y})",
            f"Bit Position: {context.bit_position_current}",
            f"Current State: {context.current_state.name}",
        ]

        if expected_state and actual_state:
            error_parts.append(f"Expected State: {expected_state.name}")
            error_parts.append(f"Actual State: {actual_state.name}")

        error_parts.append(f"Error: {message}")

        # Add context details
        error_parts.append(f"CBP: luma={context.cbp_luma:04b}, chroma={context.cbp_chroma}")
        error_parts.append(f"Blocks decoded: luma={context.luma_blocks_decoded}")

        full_message = "\n  ".join(error_parts)
        super().__init__(full_message)


@dataclass
class MBDecodeContext:
    """Context for macroblock decode state machine.

    Tracks all information needed for state validation and transitions.
    """

    # Macroblock position
    mb_x: int
    mb_y: int

    # Macroblock parameters
    mb_type: int
    cbp_luma: int  # Coded block pattern for luma (0-15)
    cbp_chroma: int  # Coded block pattern for chroma (0-2)
    qp: int  # Quantization parameter

    # Bit position tracking
    bit_position_start: int  # Bit position at start of MB residual decode
    bit_position_current: int = 0  # Current bit position (updated during decode)

    # State tracking
    current_state: MBState = field(default=MBState.START_MB)

    # Decode results (populated as we decode)
    luma_blocks_decoded: int = 0  # Count of luma blocks decoded
    chroma_dc_decoded: bool = False  # Both Cb and Cr DC decoded
    chroma_ac_decoded: bool = False  # Both Cb and Cr AC decoded

    def __post_init__(self):
        """Initialize bit position."""
        if self.bit_position_current == 0:
            self.bit_position_current = self.bit_position_start


class MacroblockDecoder:
    """State machine for decoding a macroblock.

    Enforces H.264 decode order: luma → chroma DC → chroma AC
    Validates bit positions at state boundaries.
    Fails fast with detailed context on any validation error.
    """

    # Valid state transitions per H.264 spec
    VALID_TRANSITIONS = {
        MBState.START_MB: {MBState.LUMA_RESIDUAL},
        MBState.LUMA_RESIDUAL: {MBState.CHROMA_DC},
        MBState.CHROMA_DC: {MBState.CHROMA_AC},
        MBState.CHROMA_AC: {MBState.MB_COMPLETE},
        MBState.MB_COMPLETE: set(),  # Terminal state
    }

    def __init__(
        self,
        reader: BitReader,
        mb_x: int,
        mb_y: int,
        mb_type: int,
        cbp_luma: int,
        cbp_chroma: int,
        qp: int,
    ):
        """Initialize decoder with MB parameters.

        Args:
            reader: Bit reader positioned at start of MB residual data
            mb_x: Macroblock X coordinate
            mb_y: Macroblock Y coordinate
            mb_type: Macroblock type
            cbp_luma: Luma coded block pattern (0-15)
            cbp_chroma: Chroma coded block pattern (0-2)
            qp: Quantization parameter
        """
        self.reader = reader

        self.context = MBDecodeContext(
            mb_x=mb_x,
            mb_y=mb_y,
            mb_type=mb_type,
            cbp_luma=cbp_luma,
            cbp_chroma=cbp_chroma,
            qp=qp,
            bit_position_start=reader.position,
            bit_position_current=reader.position,
        )

    def _validate_transition(self, next_state: MBState) -> None:
        """Validate state transition is legal.

        Args:
            next_state: State to transition to

        Raises:
            MBStateValidationError: If transition is invalid
        """
        current = self.context.current_state
        valid_next = self.VALID_TRANSITIONS.get(current, set())

        if next_state not in valid_next:
            raise MBStateValidationError(
                message=f"Invalid state transition from {current.name} to {next_state.name}",
                context=self.context,
                expected_state=next_state,
                actual_state=current,
            )

    def _transition_to(self, next_state: MBState) -> None:
        """Transition to next state after validation.

        Args:
            next_state: State to transition to
        """
        self._validate_transition(next_state)
        self.context.current_state = next_state
        self.context.bit_position_current = self.reader.position
