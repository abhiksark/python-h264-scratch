# h264/bitstream/numpy_bit_reader.py
"""Pure NumPy bit-level reader for H.264 bitstream parsing.

This module provides bit-level reading using only NumPy, replacing
the bitstring library dependency.

H.264 Spec Reference:
- Section 9.1: Parsing process for Exp-Golomb coded syntax elements
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Pre-computed bit masks for efficiency
_BIT_MASKS = np.array([0, 1, 3, 7, 15, 31, 63, 127, 255], dtype=np.uint32)


class NumpyBitReader:
    """Pure NumPy bit-level reader for H.264 RBSP data.

    Provides methods to read bits, bytes, and Exp-Golomb coded values
    from a byte sequence without external dependencies.

    Attributes:
        data: Original byte data as numpy array
        _byte_pos: Current byte position
        _bit_offset: Bit offset within current byte (0-7, MSB first)
    """

    def __init__(self, data: bytes):
        """Initialize BitReader with byte data.

        Args:
            data: RBSP bytes to read from
        """
        self._data = np.frombuffer(data, dtype=np.uint8)
        self._byte_pos = 0
        self._bit_offset = 0  # 0 = MSB, 7 = LSB
        self._total_bits = len(data) * 8

        logger.debug(
            f"BitReader initialized with {len(data)} bytes ({self._total_bits} bits)"
        )

    @property
    def position(self) -> int:
        """Current bit position."""
        return self._byte_pos * 8 + self._bit_offset

    @position.setter
    def position(self, pos: int):
        """Set current bit position."""
        self._byte_pos = pos // 8
        self._bit_offset = pos % 8

    @property
    def bits_remaining(self) -> int:
        """Number of bits remaining to read."""
        return self._total_bits - self.position

    @property
    def bytes_remaining(self) -> int:
        """Number of complete bytes remaining."""
        return self.bits_remaining // 8

    def read_bits(self, n: int) -> int:
        """Read n bits as unsigned integer.

        Handles reads that cross byte boundaries efficiently.

        Args:
            n: Number of bits to read (0-32)

        Returns:
            Unsigned integer value

        Raises:
            ValueError: If not enough bits available
        """
        if n == 0:
            return 0

        if n > 32:
            raise ValueError(f"Cannot read more than 32 bits at once, got {n}")

        if n > self.bits_remaining:
            raise ValueError(
                f"Cannot read {n} bits at position {self.position}, "
                f"only {self.bits_remaining} remaining"
            )

        result = 0
        bits_needed = n

        while bits_needed > 0:
            # Bits available in current byte (from current offset to end)
            bits_in_byte = 8 - self._bit_offset
            bits_to_read = min(bits_needed, bits_in_byte)

            # Extract bits from current byte
            # Shift right to align desired bits to LSB, then mask
            shift = bits_in_byte - bits_to_read
            mask = _BIT_MASKS[bits_to_read]
            byte_val = int(self._data[self._byte_pos])
            extracted = (byte_val >> shift) & mask

            # Add to result (shift left to make room for new bits)
            result = (result << bits_to_read) | extracted

            # Update position
            bits_needed -= bits_to_read
            self._bit_offset += bits_to_read

            if self._bit_offset >= 8:
                self._byte_pos += 1
                self._bit_offset = 0

        logger.debug(f"read_bits({n}) = {result} (0b{result:0{n}b})")
        return result

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
        return bool(self.read_bit() == 1)

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
        if self._bit_offset == 0:
            # Byte-aligned read - fast path
            if self._byte_pos + n > len(self._data):
                raise ValueError(
                    f"Cannot read {n} bytes at position {self.position}"
                )
            result = bytes(self._data[self._byte_pos : self._byte_pos + n])
            self._byte_pos += n
            return result
        else:
            # Non-aligned read - slow path
            result = bytearray(n)
            for i in range(n):
                result[i] = self.read_bits(8)
            return bytes(result)

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
        0 -> 0, 1 -> 1, 2 -> -1, 3 -> 2, 4 -> -2, ...

        Returns:
            Decoded signed integer

        H.264 Spec: Section 9.1.1
        """
        k = int(self.read_ue())  # Convert to Python int for signed arithmetic

        if k == 0:
            return 0

        # Map to signed: value = (-1)^(k+1) * ceil(k/2)
        value = ((k + 1) >> 1)
        if k % 2 == 0:
            value = -value

        logger.debug(f"read_se() = {value} (k={k})")
        return value

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
        if self._bit_offset != 0:
            skip = 8 - self._bit_offset
            self._bit_offset = 0
            self._byte_pos += 1
            logger.debug(f"Byte aligned: skipped {skip} bits")

    def skip_bits(self, n: int):
        """Skip n bits.

        Args:
            n: Number of bits to skip
        """
        new_pos = self.position + n
        self._byte_pos = new_pos // 8
        self._bit_offset = new_pos % 8

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
                return bool(bits != expected)
            finally:
                self.position = pos

        return True


class NumpyBitWriter:
    """Pure NumPy bit-level writer for creating test bitstreams.

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

        # Convert to bytes using numpy for efficiency
        bits_array = np.array(self._bits, dtype=np.uint8)
        n_bytes = len(bits_array) // 8

        # Reshape and pack bits into bytes
        bits_reshaped = bits_array.reshape(n_bytes, 8)
        # Multiply by powers of 2 and sum each row
        powers = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
        result = np.sum(bits_reshaped * powers, axis=1, dtype=np.uint8)

        return bytes(result)
