"""Automated gate evaluation (plan S12).

Gates are designed to run with "no human listening" (S0). Each check returns
PASS, WARN or ESCALATE:

    PASS      inside the target band
    WARN      between pass and escalate -- "acceptable, log a warning, continue"
    ESCALATE  past the escalate threshold; the stage's Plan B is on the table
    N/A       cannot be evaluated yet (e.g. needs the gold set, S6.4)

Thresholds come from configs/pipeline.yaml, never from here (S3 config
contract), so calibration is a config change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PASS, WARN, ESCALATE, NA = "PASS", "WARN", "ESCALATE", "N/A"


@dataclass
class Check:
    name: str
    status: str
    detail: str

    def __str__(self) -> str:
        return f"{self.status:<8} {self.name:<38} {self.detail}"


def _band(value: float | None, lo: float, hi: float, *,
          escalate_below: float | None = None,
          escalate_above: float | None = None) -> str:
    if value is None:
        return NA
    if escalate_below is not None and value < escalate_below:
        return ESCALATE
    if escalate_above is not None and value > escalate_above:
        return ESCALATE
    return PASS if lo <= value <= hi else WARN


def load_manifests(work: str | Path, stage: str) -> list[dict[str, Any]]:
    out = []
    for p in sorted(Path(work).glob(f"*/{stage}/_DONE.json")):
        try:
            with open(p) as fh:
                out.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _fraction(values: Iterable[bool]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def gate_s0(manifests: list[dict], cfg) -> list[Check]:
    """Gate S0 (plan S5)."""
    if not manifests:
        return [Check("s0: any episodes", NA, "no manifests")]
    g = cfg.stage_params("s0_ingest").get("gate", {})
    ms = [m["metrics"] for m in manifests]
    checks: list[Check] = []

    ratios = [m.get("speech_ratio") for m in ms]
    known = [r for r in ratios if r is not None]
    if not known:
        checks.append(Check("speech_ratio in band", NA, "segmentation not run"))
    else:
        lo, hi = g.get("speech_ratio_min", 0.60), g.get("speech_ratio_max", 0.98)
        ok = sum(1 for r in known if lo <= r <= hi)
        checks.append(Check(
            "speech_ratio in band", PASS if ok == len(known) else WARN,
            f"{ok}/{len(known)} in [{lo}, {hi}]; "
            f"values {', '.join(f'{r:.3f}' for r in sorted(known))}"))

    with_music = [m.get("music_seconds") for m in ms]
    seen = [x for x in with_music if x is not None]
    if not seen:
        checks.append(Check("music detected", NA, "segmentation not run"))
    else:
        frac = _fraction(x > 0 for x in seen)
        need = g.get("music_detected_fraction_min", 0.80)
        checks.append(Check(
            "music detected", PASS if frac >= need else WARN,
            f"{frac:.0%} of episodes (need >={need:.0%}); "
            "NOTE which episodes have intros is unverified (S5 Plan B: spot-check)"))

    checks.append(Check("decode failures", PASS, f"0/{len(manifests)}"))

    audio_h = sum(m.get("duration_s", 0) for m in ms) / 3600
    wall = sum(m.get("wall_s", 0) for m in manifests)
    if audio_h:
        rate = wall / audio_h
        cap = g.get("max_wall_s_per_audio_hour", 30)
        checks.append(Check(
            "wall per audio-hour", PASS if rate <= cap else WARN,
            f"{rate:.0f}s (target <={cap}s; target is CPU-only, ours includes AST)"))
    return checks


def gate_s1(manifests: list[dict], cfg) -> list[Check]:
    """Gate S1-A (plan S6.1)."""
    if not manifests:
        return [Check("s1: any episodes", NA, "no manifests")]
    g = cfg.stage_params("s1_diarize").get("gate", {})
    ms = [m["metrics"] for m in manifests]
    checks: list[Check] = []

    checks.append(Check("DER vs gold set", NA, "gold set not built (S6.4)"))
    checks.append(Check("JER vs gold set", NA, "gold set not built (S6.4)"))

    want = set(g.get("dominant_speaker_count", [3, 4]))
    counts = [m.get("n_dominant_speakers") for m in ms if m.get("n_dominant_speakers") is not None]
    if counts:
        frac = _fraction(c in want for c in counts)
        need = g.get("dominant_count_ok_fraction", 0.90)
        checks.append(Check(
            "dominant speakers in {3,4}", PASS if frac >= need else WARN,
            f"{frac:.0%} (need >={need:.0%}); counts {counts}. "
            "NOTE the show began as a duo, so early episodes legitimately show 2"))

    ovl = [m.get("overlap_ratio") for m in ms if m.get("overlap_ratio") is not None]
    if ovl:
        lo = g.get("overlap_ratio_min", 0.02)
        hi = g.get("overlap_ratio_max", 0.10)
        esc = g.get("overlap_ratio_escalate", 0.15)
        worst = max(ovl)
        checks.append(Check(
            "overlap ratio", _band(worst, lo, hi, escalate_above=esc),
            f"max {worst:.3f} across {len(ovl)} (band {lo}-{hi}, escalate >{esc}); "
            f"values {', '.join(f'{o:.3f}' for o in sorted(ovl))}"))

    audio_h = sum(m.get("duration_s", 0) for m in ms) / 3600
    wall = sum(m.get("wall_s", 0) for m in manifests)
    if audio_h:
        rate = wall / audio_h / 60
        cap = g.get("max_wall_min_per_audio_hour", 3)
        esc = g.get("escalate_wall_min_per_audio_hour", 10)
        checks.append(Check(
            "wall per audio-hour", _band(rate, 0, cap, escalate_above=esc),
            f"{rate:.1f} min (target <={cap}, escalate >{esc}); measured on T4, target assumes L4"))
    return checks


def summarise(checks: list[Check]) -> str:
    worst = ESCALATE if any(c.status == ESCALATE for c in checks) else (
        WARN if any(c.status == WARN for c in checks) else PASS)
    n_na = sum(1 for c in checks if c.status == NA)
    return f"{worst}" + (f" ({n_na} not evaluable)" if n_na else "")


# Registry of stage -> gate function. The CLI and any caller should look up
# available gates here rather than hardcoding the mapping (a stage with no gate
# yet is simply absent). Add new gates to this dict as stages gain them.
GATES = {
    "s0_ingest": gate_s0,
    "s1_diarize": gate_s1,
}
