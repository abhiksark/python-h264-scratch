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

    def _validate_bit_position_reasonable(
        self,
        state_name: str,
        max_bits_per_block: int = 200
    ) -> None:
        """Validate bit position hasn't advanced unreasonably.

        Args:
            state_name: Name of state for error message
            max_bits_per_block: Maximum reasonable bits per block

        Raises:
            MBStateValidationError: If bit consumption exceeds reasonable bounds
        """
        bits_consumed = self.reader.position - self.context.bit_position_current

        # Calculate maximum reasonable bits for this MB
        # Luma: up to 16 blocks × max_bits_per_block
        # Chroma DC: 2 blocks × 50 bits (much smaller)
        # Chroma AC: 8 blocks × max_bits_per_block
        max_reasonable_bits = (16 + 8) * max_bits_per_block + 2 * 50

        if bits_consumed > max_reasonable_bits:
            raise MBStateValidationError(
                message=f"State {state_name} exceeded reasonable bit budget: "
                        f"consumed {bits_consumed} bits, max reasonable {max_reasonable_bits}",
                context=self.context,
            )

        if bits_consumed < 0:
            raise MBStateValidationError(
                message=f"State {state_name} bit position went backwards: {bits_consumed}",
                context=self.context,
            )

    def decode_luma_residual(self) -> None:
        """Decode luma residual blocks in state machine context.

        H.264 Spec: Section 7.3.5.3 - Residual block data for luma.

        Raises:
            MBStateValidationError: If not in correct state or validation fails
        """
        # Validate we're in the right state
        if self.context.current_state != MBState.START_MB:
            raise MBStateValidationError(
                message="decode_luma_residual called from wrong state",
                context=self.context,
                expected_state=MBState.START_MB,
                actual_state=self.context.current_state,
            )

        # Transition to LUMA_RESIDUAL state
        self._transition_to(MBState.LUMA_RESIDUAL)

        # Count blocks we should decode based on CBP
        # CBP luma has 4 bits, one per 8x8 quadrant (0-3)
        # Each quadrant contains 4 4x4 blocks
        expected_blocks = 0
        for quadrant in range(4):  # All 4 quadrants (blocks 0-15)
            if self.context.cbp_luma & (1 << quadrant):
                expected_blocks += 4  # All 4 blocks in this quadrant

        # For now, just validate the count matches
        # (Actual decode logic will be integrated later)
        self.context.luma_blocks_decoded = expected_blocks

        # Validate bit position is reasonable
        self._validate_bit_position_reasonable("LUMA_RESIDUAL")

        # Transition to next state
        self._transition_to(MBState.CHROMA_DC)

    def decode_chroma_dc(self) -> None:
        """Decode chroma DC residual blocks in state machine context.

        H.264 Spec: Section 7.3.5.3 - Chroma DC residual.
        Order: Cb DC, then Cr DC (both 2x2 blocks).

        Raises:
            MBStateValidationError: If not in correct state or validation fails
        """
        # Validate we're in the right state (luma already transitioned us to CHROMA_DC)
        if self.context.current_state != MBState.CHROMA_DC:
            raise MBStateValidationError(
                message="decode_chroma_dc called from wrong state",
                context=self.context,
                expected_state=MBState.CHROMA_DC,
                actual_state=self.context.current_state,
            )

        bit_position_before = self.reader.position

        if self.context.cbp_chroma >= 1:
            # Decode Cb DC (2x2 block, 4 coefficients)
            # Then Cr DC (2x2 block, 4 coefficients)
            # For now, just validate bit position advances
            # (Actual decode logic will be integrated later)

            self.context.chroma_dc_decoded = True

            # Validate bit position advanced
            bits_consumed = self.reader.position - bit_position_before
            if bits_consumed < 0:
                raise MBStateValidationError(
                    message=f"Bit position went backwards in chroma DC: {bits_consumed}",
                    context=self.context,
                )
        else:
            # CBP chroma = 0, no DC to decode
            self.context.chroma_dc_decoded = False

        # Transition to next state
        self._transition_to(MBState.CHROMA_AC)

    def decode_chroma_ac(self) -> None:
        """Decode chroma AC residual blocks in state machine context.

        H.264 Spec: Section 7.3.5.3 - Chroma AC residual.
        Order: Cb AC (4 blocks), then Cr AC (4 blocks).

        Raises:
            MBStateValidationError: If not in correct state or validation fails
        """
        # Validate we're in the right state (chroma DC already transitioned us to CHROMA_AC)
        if self.context.current_state != MBState.CHROMA_AC:
            raise MBStateValidationError(
                message="decode_chroma_ac called from wrong state",
                context=self.context,
                expected_state=MBState.CHROMA_AC,
                actual_state=self.context.current_state,
            )

        bit_position_before = self.reader.position

        if self.context.cbp_chroma >= 2:
            # Decode Cb AC (4 blocks of 15 coefficients each)
            # Then Cr AC (4 blocks of 15 coefficients each)
            # For now, just validate bit position advances
            # (Actual decode logic will be integrated later)

            self.context.chroma_ac_decoded = True

            # Validate bit position advanced
            bits_consumed = self.reader.position - bit_position_before
            if bits_consumed < 0:
                raise MBStateValidationError(
                    message=f"Bit position went backwards in chroma AC: {bits_consumed}",
                    context=self.context,
                )
        else:
            # CBP chroma < 2, no AC to decode
            self.context.chroma_ac_decoded = False

        # Transition to terminal state
        self._transition_to(MBState.MB_COMPLETE)
