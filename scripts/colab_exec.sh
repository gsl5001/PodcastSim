#!/usr/bin/env bash
# The ONE way to run a Python file on a Colab session.
#
# `colab exec` reattaches to a long-lived kernel, which caused three separate
# failures (see docs/RUNBOOK.md). Rather than remembering three mitigations at
# every call site, they live here:
#
#   1. --timeout is always passed. The default abandons the output stream after
#      ~30 s while the kernel keeps running, so a finished run looks failed.
#   2. A preamble purges project modules from sys.modules, so a freshly synced
#      source file is actually used instead of a stale import.
#   3. The same preamble asserts that every compiled dep's live version matches
#      its installed distribution, and fails LOUDLY if not -- a numpy upgraded
#      on disk but stale in the kernel previously surfaced as an unrelated
#      transformers import error.
#
# Usage: scripts/colab_exec.sh <session> <script.py> [timeout_s]
set -euo pipefail

SESSION="$1"; SCRIPT="$2"; TIMEOUT="${3:-1800}"
[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT" >&2; exit 1; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
COMBINED="$TMP/$(basename "$SCRIPT")"

cat > "$COMBINED" <<'PREAMBLE'
# --- injected by scripts/colab_exec.sh -------------------------------------
import importlib as _il, sys as _sys

for _n in [m for m in _sys.modules if m == "podcastsim" or m.startswith("podcastsim.")]:
    del _sys.modules[_n]
_il.invalidate_caches()

def _assert_fresh(*packages):
    """Fail loudly when the kernel holds an older build than is installed."""
    import importlib.metadata as _md
    stale = []
    for _p in packages:
        try:
            _mod = _il.import_module(_p)
            _live, _dist = getattr(_mod, "__version__", None), _md.version(_p)
        except Exception:
            continue
        # Compare base versions only: torch reports "2.11.0+cu128" while its
        # distribution metadata says "2.11.0", which is not staleness.
        _lb, _db = _live.split("+")[0] if _live else "", _dist.split("+")[0] if _dist else ""
        if _lb and _db and _lb != _db:
            stale.append(f"{_p}: kernel={_live} installed={_dist}")
    if stale:
        raise RuntimeError(
            "stale kernel modules -- run `colab restart-kernel`:\n  "
            + "\n  ".join(stale)
        )

_assert_fresh("numpy", "scipy", "transformers", "torch")
del _assert_fresh
# --- end injected preamble --------------------------------------------------
PREAMBLE
cat "$SCRIPT" >> "$COMBINED"

exec colab --auth=adc exec -s "$SESSION" --timeout "$TIMEOUT" -f "$COMBINED"
