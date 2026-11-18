#!/usr/bin/env bash
# Recreate a working Colab session from scratch (plan S3 `scripts/colab_bootstrap.sh`).
#
# Free-tier runtimes are reclaimed without warning (S2.1), and /content does not
# survive. This makes a lost session cost one command instead of a rebuild.
#
# Usage: scripts/colab_bootstrap.sh [session] [gpu] [episode ...]
set -euo pipefail

SESSION="${1:-ps}"; GPU="${2:-T4}"; shift 2 || true
EPISODES=("$@")
COLAB=(colab --auth=adc)
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

echo "[boot] session=$SESSION gpu=$GPU episodes=${EPISODES[*]:-none}"

# `colab status` succeeds against a stale LOCAL record, so a reclaimed VM looks
# alive. Probe the kernel instead -- the only thing that proves it is really up.
alive=0
if "${COLAB[@]}" status -s "$SESSION" >/dev/null 2>&1; then
  if echo 'print("ALIVE")' | "${COLAB[@]}" exec -s "$SESSION" --timeout 120 2>/dev/null | grep -q ALIVE; then
    alive=1
  fi
fi
if [ "$alive" -eq 1 ]; then
  echo "[boot] reusing live session"
else
  echo "[boot] no live session; creating"
  "${COLAB[@]}" stop -s "$SESSION" >/dev/null 2>&1 || true
  "${COLAB[@]}" new --gpu "$GPU" -s "$SESSION"
fi

./scripts/colab_sync.sh "$SESSION" 2>&1 | grep -vE "Deprecation|extractall" || true

# HF token for the gated pyannote repo (S14.12). Never baked into the image.
if [ -f "$HOME/.config/podcastsim/hf_token" ]; then
  "${COLAB[@]}" upload -s "$SESSION" "$HOME/.config/podcastsim/hf_token" /content/hf_token >/dev/null
  echo "[boot] hf token staged"
fi

# pyannote is not on the Colab image; everything else (torch, transformers,
# torchaudio, ffmpeg 4.4.2) already is.
cat > /tmp/_need.py <<'PY'
try:
    import pyannote.audio; print("PYANNOTE_OK")
except ImportError:
    print("PYANNOTE_MISSING")
PY
if ./scripts/colab_exec.sh "$SESSION" /tmp/_need.py 300 2>/dev/null | grep -q PYANNOTE_MISSING; then
  echo "[boot] installing pyannote.audio"
  "${COLAB[@]}" install -s "$SESSION" "pyannote.audio>=4.0,<5" >/dev/null
  # An install can upgrade a compiled dep (it pulled numpy 2.1.3 -> 2.5.2) while
  # the kernel keeps the OLD module in sys.modules -- kernel state persists
  # across exec. scipy then calls into a numpy that is not there, and the error
  # surfaces far away as "Could not import ASTForAudioClassification".
  echo "[boot] restarting kernel to pick up upgraded compiled deps"
  "${COLAB[@]}" restart-kernel -s "$SESSION" >/dev/null 2>&1 || true
fi

# Restore prior stage state so completed episodes are skipped, not recomputed.
# This is what makes a reclaimed VM cheap (plan S0 rule 3 / S2.2).
if [ -n "$(find data/work -name '_DONE.json' 2>/dev/null | head -1)" ]; then
  tar czf /tmp/_work_restore.tgz -C data/work .
  "${COLAB[@]}" upload -s "$SESSION" /tmp/_work_restore.tgz /content/work_restore.tgz >/dev/null
  ./scripts/colab_exec.sh "$SESSION" scripts/_colab_restore_work.py 600 | tail -1
fi

for ep in "${EPISODES[@]:-}"; do
  [ -z "$ep" ] && continue
  ./scripts/colab_put.sh "$SESSION" "data/raw/$ep" "/content/raw/$ep" | tail -1
done

echo "[boot] ready"
