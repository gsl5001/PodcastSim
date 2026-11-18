"""Read NIST RTTM turn files (written by S1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Turn:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def read_rttm(path: str | Path) -> list[Turn]:
    """Parse an RTTM file into Turns. RTTM columns (space-separated): 0=type
    ("SPEAKER"), 3=start_s, 4=duration_s, 7=speaker label. Lines that are not a
    well-formed SPEAKER row (<8 fields or a different type) are skipped."""
    turns: list[Turn] = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER":
            start, dur = float(f[3]), float(f[4])
            turns.append(Turn(start, start + dur, f[7]))
    return turns


def turns_by_speaker(turns: list[Turn]) -> dict[str, list[Turn]]:
    """Group turns by speaker label, preserving order within each speaker."""
    out: dict[str, list[Turn]] = {}
    for t in turns:
        out.setdefault(t.speaker, []).append(t)
    return out
