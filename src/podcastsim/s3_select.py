"""Turn selection for S3 training clips (plan S8.1). Pure logic, GPU-free.

Step 1 of S8: "host-assigned turns with zero overlap after 0.25 s padding, not
inside music/speech_over_music spans, duration >= 1.0 s." Overlaps are discarded,
not separated (S8.1, S14.11).
"""

from __future__ import annotations

from dataclasses import dataclass

from .rttm import Turn


@dataclass
class Candidate:
    start: float
    end: float
    host_id: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def select_host_turns(
    turns: list[Turn],
    host_map: dict[str, str],
    overlap_regions: list[tuple[float, float]],
    exclude_spans: list[tuple[float, float]],
    *,
    min_duration_s: float = 1.0,
    overlap_pad_s: float = 0.25,
) -> list[Candidate]:
    """Host turns that are clean enough to become training clips.

    `overlap_regions` are the padded multi-speaker regions from S1; a turn
    touching any is dropped (overlap contaminates timbre). `exclude_spans` are
    music / speech_over_music spans from S0. `host_map` maps a local speaker
    label to a host id or "guest"; only real hosts are kept.
    """
    out: list[Candidate] = []
    for t in turns:
        host = host_map.get(t.speaker, "guest")
        if host == "guest":
            continue
        if t.duration < min_duration_s:
            continue
        s, e = t.start - overlap_pad_s, t.end + overlap_pad_s
        if any(_overlaps(s, e, o0, o1) for o0, o1 in overlap_regions):
            continue
        if any(_overlaps(t.start, t.end, x0, x1) for x0, x1 in exclude_spans):
            continue
        out.append(Candidate(t.start, t.end, host))
    return out


def host_hours(candidates: list[Candidate]) -> dict[str, float]:
    hours: dict[str, float] = {}
    for c in candidates:
        hours[c.host_id] = hours.get(c.host_id, 0.0) + c.duration / 3600
    return hours
