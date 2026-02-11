# debug/mb_position_tracker.py
"""Track bit positions at MB parsing boundaries.

This module provides utilities to track and log bit positions during
macroblock decoding, helping identify bit drift issues.
"""
from typing import Dict, List, Tuple
from bitstream import BitReader


class MBPositionTracker:
    """Track bit positions during MB decode for debugging."""

    def __init__(self):
        self.checkpoints: List[Dict] = []
        self.current_mb: Tuple[int, int] = (0, 0)

    def checkpoint(self, reader: BitReader, label: str, **kwargs):
        """Record current bit position with label.

        Args:
            reader: BitReader instance
            label: Description of checkpoint
            **kwargs: Additional metadata to store
        """
        self.checkpoints.append({
            'mb': self.current_mb,
            'position': reader.position,
            'label': label,
            **kwargs
        })

    def start_mb(self, mb_x: int, mb_y: int, reader: BitReader):
        """Mark start of MB decode.

        Args:
            mb_x: Macroblock X coordinate
            mb_y: Macroblock Y coordinate
            reader: BitReader instance
        """
        self.current_mb = (mb_x, mb_y)
        self.checkpoint(reader, 'MB_START')

    def end_mb(self, reader: BitReader, status: str = 'OK'):
        """Mark end of MB decode.

        Args:
            reader: BitReader instance
            status: Decode status (default: 'OK')
        """
        self.checkpoint(reader, 'MB_END', status=status)

    def get_mb_summary(self, mb_x: int, mb_y: int) -> Dict:
        """Get summary for specific MB.

        Args:
            mb_x: Macroblock X coordinate
            mb_y: Macroblock Y coordinate

        Returns:
            Dictionary with MB summary or None if not found
        """
        mb_checkpoints = [c for c in self.checkpoints if c['mb'] == (mb_x, mb_y)]
        if len(mb_checkpoints) < 2:
            return None

        start = mb_checkpoints[0]['position']
        end = mb_checkpoints[-1]['position']

        return {
            'mb': (mb_x, mb_y),
            'start': start,
            'end': end,
            'consumed': end - start,
            'checkpoints': mb_checkpoints
        }

    def print_summary(self):
        """Print summary of all MBs."""
        mbs = {}
        for cp in self.checkpoints:
            mb = cp['mb']
            if mb not in mbs:
                mbs[mb] = []
            mbs[mb].append(cp)

        for mb, cps in sorted(mbs.items()):
            if len(cps) >= 2:
                start = cps[0]['position']
                end = cps[-1]['position']
                print(f"MB{mb}: {start:4d} -> {end:4d} ({end-start:4d} bits)")
                for cp in cps[1:-1]:  # Skip START and END
                    print(f"  [{cp['position']:4d}] {cp['label']}")


# Global tracker instance
_tracker = MBPositionTracker()


def get_tracker() -> MBPositionTracker:
    """Get global position tracker.

    Returns:
        Global MBPositionTracker instance
    """
    return _tracker
