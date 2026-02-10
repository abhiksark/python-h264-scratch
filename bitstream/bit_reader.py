# h264/bitstream/bit_reader.py
"""Bit-level reader for H.264 bitstream parsing.

This module provides bit-level reading capabilities needed for parsing
H.264 syntax elements like Exponential-Golomb codes, fixed-length codes,
and flags.

Two implementations available:
- Pure NumPy (default, no external dependencies)
- bitstring-based (legacy, for comparison)

H.264 Spec Reference:
- Section 9.1: Parsing process for Exp-Golomb coded syntax elements
"""

import logging
from typing import Optional

# Try to import bitstring for legacy support
try:
    from bitstring import BitStream, ReadError
    BITSTRING_AVAILABLE = True
except ImportError:
    BITSTRING_AVAILABLE = False
    BitStream = None
    ReadError = Exception

# Import pure NumPy implementation
from .numpy_bit_reader import NumpyBitReader, NumpyBitWriter

import numpy as np

logger = logging.getLogger(__name__)

# Default to NumPy implementation (set to False to use bitstring if available)
USE_NUMPY_READER = False  # TODO: Fix NumpyBitReader compatibility


class BitReader:
    """Bit-level reader for H.264 RBSP data.

    Provides methods to read bits, bytes, and Exp-Golomb coded values
    from a byte sequence.

    Attributes:
        data: Original byte data
        position: Current bit position
    """

    def __init__(self, data: bytes):
        """Initialize BitReader with byte data.

        Args:
            data: RBSP bytes to read from
        """
        if USE_NUMPY_READER:
            # Use NumPy-based reader (no external dependencies)
            self._stream = NumpyBitReader(data)
            self._data = data
            logger.debug(f"BitReader (NumPy) initialized with {len(data)} bytes ({len(data)*8} bits)")
        else:
            # Use bitstring-based reader
            if not BITSTRING_AVAILABLE:
                raise ImportError(
                    "bitstring library required. Install with: pip install bitstring"
                )
            self._stream = BitStream(bytes=data)
            self._data = data
            logger.debug(f"BitReader (bitstring) initialized with {len(data)} bytes ({len(data)*8} bits)")

    @property
    def position(self) -> int:
        """Current bit position."""
        return self._stream.pos

    @position.setter
    def position(self, pos: int):
        """Set current bit position."""
        self._stream.pos = pos

    @property
    def bits_remaining(self) -> int:
        """Number of bits remaining to read."""
        return len(self._stream) - self._stream.pos

    @property
    def bytes_remaining(self) -> int:
        """Number of complete bytes remaining."""
        return self.bits_remaining // 8

    def read_bits(self, n: int) -> int:
        """Read n bits as unsigned integer.

        Args:
            n: Number of bits to read (1-32)

        Returns:
            Unsigned integer value

        Raises:
            ReadError: If not enough bits available
        """
        if n == 0:
            return 0

        try:
            value = self._stream.read(f'uint:{n}')
            logger.debug(f"read_bits({n}) = {value} (0b{value:0{n}b})")
            return value
        except ReadError as e:
            raise ReadError(f"Cannot read {n} bits at position {self.position}: {e}")

    def read_bit(self) -> int:
        """Read single bit.

        Returns:
            0 or 1
        """
        return self.read_bits(1)

    def read_flag(self) -> bool:
        """Read single bit as boolean flag.

        Returns:
            True if bit is 1, False if 0
        """
        return self.read_bit() == 1

    def read_byte(self) -> int:
        """Read 8 bits as unsigned byte.

        Returns:
            Unsigned byte value (0-255)
        """
        return self.read_bits(8)

    def read_bytes(self, n: int) -> bytes:
        """Read n bytes.

        Args:
            n: Number of bytes to read

        Returns:
            Bytes object
        """
        if USE_NUMPY_READER:
            # NumpyBitReader doesn't have a bytes read method, read bits instead
            value = self._stream.read_bits(n * 8)
            return value.to_bytes(n, byteorder='big')
        else:
            return self._stream.read(f'bytes:{n}')

    def read_ue(self) -> int:
        """Read unsigned Exp-Golomb coded value (ue(v)).

        Exp-Golomb coding:
        1. Count leading zeros until first 1 bit
        2. Read that many more bits after the 1
        3. Value = (1 << n) + suffix - 1

        Returns:
            Decoded unsigned integer

        H.264 Spec: Section 9.1

        Examples:
            1        -> 0
            010      -> 1
            011      -> 2
            00100    -> 3
            00101    -> 4
            00110    -> 5
            00111    -> 6
        """
        # Count leading zeros
        leading_zeros = 0
        while self.read_bit() == 0:
            leading_zeros += 1
            if leading_zeros > 32:
                raise ValueError("Exp-Golomb value too large (>32 leading zeros)")

        if leading_zeros == 0:
            # First bit was 1, value is 0
            return 0

        # Read the suffix bits
        suffix = self.read_bits(leading_zeros)

        # Calculate value
        value = (1 << leading_zeros) - 1 + suffix

        logger.debug(f"read_ue() = {value} (zeros={leading_zeros}, suffix={suffix})")
        return value

    def read_se(self) -> int:
        """Read signed Exp-Golomb coded value (se(v)).

        Signed Exp-Golomb maps unsigned values to signed:
        0 -> 0
        1 -> 1
        2 -> -1
        3 -> 2
        4 -> -2
        ...

        Returns:
            Decoded signed integer

        H.264 Spec: Section 9.1.1

        Formula: value = (-1)^(k+1) * ceil(k/2)
                 where k is the unsigned Exp-Golomb value
        """
        k = self.read_ue()

        if k == 0:
            return 0

        # Map to signed
        value = ((k + 1) >> 1)
        if k % 2 == 0:
            value = -value

        logger.debug(f"read_se() = {value} (k={k})")
        return value

    def more_rbsp_data(self) -> bool:
        """Check if more RBSP data exists before RBSP trailing bits.

        Per H.264 spec 7.2, checks for rbsp_stop_one_bit pattern.

        Returns:
            True if more data exists, False if at RBSP trailing bits
        """
        if USE_NUMPY_READER:
            return self._stream.more_rbsp_data()
        else:
            # For bitstring-based reader, check bits_remaining
            return self.bits_remaining > 8

    def read_te(self, max_value: int) -> int:
        """Read truncated Exp-Golomb coded value (te(v)).

        Used when the maximum possible value is known.
        If max_value == 1, reads single bit.
        Otherwise, reads as ue(v).

        Args:
            max_value: Maximum possible value

        Returns:
            Decoded value (0 to max_value)

        H.264 Spec: Section 9.1.2
        """
        if max_value == 1:
            return 1 - self.read_bit()
        else:
            return self.read_ue()

    def peek_bits(self, n: int) -> int:
        """Peek at next n bits without advancing position.

        Args:
            n: Number of bits to peek

        Returns:
            Unsigned integer value
        """
        pos = self.position
        value = self.read_bits(n)
        self.position = pos
        return value

    def byte_align(self):
        """Skip bits to align to next byte boundary."""
        remainder = self.position % 8
        if remainder != 0:
            skip = 8 - remainder
            self._stream.pos += skip
            logger.debug(f"Byte aligned: skipped {skip} bits")

    def skip_bits(self, n: int):
        """Skip n bits.

        Args:
            n: Number of bits to skip
        """
        self._stream.pos += n

    def more_rbsp_data(self) -> bool:
        """Check if more RBSP data remains.

        Returns:
            True if more data to read (not just trailing bits)

        H.264 Spec: Section 7.2
        """
        if self.bits_remaining == 0:
            return False

        # Check for rbsp_trailing_bits pattern
        # (1 followed by zeros to byte boundary)
        remaining = self.bits_remaining
        if remaining < 8:
            # Check if remaining bits are trailing
            pos = self.position
            try:
                bits = self.read_bits(remaining)
                # Should be 1 followed by zeros
                expected = 1 << (remaining - 1)
                return bits != expected
            finally:
                self.position = pos

        return True


class BitWriter:
    """Bit-level writer for creating test bitstreams.

    Primarily used for testing - encodes values in H.264 format.
    """

    def __init__(self):
        """Initialize empty bit buffer."""
        self._bits = []

    def write_bits(self, value: int, n: int):
        """Write n bits from value.

        Args:
            value: Integer value to write
            n: Number of bits to write
        """
        for i in range(n - 1, -1, -1):
            self._bits.append((value >> i) & 1)

    def write_bit(self, value: int):
        """Write single bit."""
        self._bits.append(value & 1)

    def write_flag(self, value: bool):
        """Write boolean as single bit."""
        self._bits.append(1 if value else 0)

    def write_ue(self, value: int):
        """Write unsigned Exp-Golomb coded value.

        Args:
            value: Non-negative integer to encode
        """
        if value == 0:
            self.write_bit(1)
            return

        # Calculate code_num = value + 1
        code_num = value + 1

        # Count bits needed
        n = code_num.bit_length()

        # Write n-1 leading zeros
        for _ in range(n - 1):
            self.write_bit(0)

        # Write the code_num
        self.write_bits(code_num, n)

    def write_se(self, value: int):
        """Write signed Exp-Golomb coded value.

        Args:
            value: Signed integer to encode
        """
        if value <= 0:
            k = -2 * value
        else:
            k = 2 * value - 1
        self.write_ue(k)

    def to_bytes(self) -> bytes:
        """Convert bit buffer to bytes.

        Pads with zeros to byte boundary if needed.

        Returns:
            Bytes representation
        """
        # Pad to byte boundary
        while len(self._bits) % 8 != 0:
            self._bits.append(0)

        # Convert to bytes
        result = bytearray()
        for i in range(0, len(self._bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self._bits[i + j]
            result.append(byte)

        return bytes(result)


# Provide aliases based on configuration
# When USE_NUMPY_READER is True, the classes in this module still exist
# but NumpyBitReader/NumpyBitWriter are the recommended implementations.
# For backward compatibility, we keep both available.

# Rename original classes to indicate they use bitstring
BitstringBitReader = BitReader
BitstringBitWriter = BitWriter


def create_bit_reader(data: bytes) -> "BitReader":
    """Factory function to create appropriate BitReader.

    Returns NumpyBitReader by default, falls back to bitstring-based
    reader if explicitly configured.

    Args:
        data: RBSP bytes to read from

    Returns:
        BitReader instance (NumPy or bitstring-based)
    """
    if USE_NUMPY_READER:
        return NumpyBitReader(data)
    elif BITSTRING_AVAILABLE:
        return BitstringBitReader(data)
    else:
        return NumpyBitReader(data)
