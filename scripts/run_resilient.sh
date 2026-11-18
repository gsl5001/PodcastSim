#!/usr/bin/env bash
# Run a stage over episodes, one at a time, surviving free-tier VM reclamation.
#
# Free-tier Colab reclaims runtimes within ~15-20 min (plan S2.1), faster than a
# multi-episode diarization batch. So: process ONE episode per exec (each ~5 min,
# fits the window), push its result to Drive immediately, and if the VM dies just
# re-bootstrap and continue. Episodes already done in Drive are skipped, so every
# attempt makes progress.
#
# Usage: scripts/run_resilient.sh <stage> <episode.mp3> [episode.mp3 ...]
set -uo pipefail
STAGE="$1"; shift
EPISODES=("$@")
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
COLAB=(colab --auth=adc)
STATE="gdrive_rw:podcastsim_state/work"

ensure_session() {
  if ! echo 'print("A")' | "${COLAB[@]}" exec -s ps --timeout 60 2>/dev/null | grep -q A; then
    echo "[resilient] (re)building session"
    "${COLAB[@]}" stop -s ps >/dev/null 2>&1 || true
    ./scripts/colab_bootstrap.sh ps T4 >/dev/null 2>&1
    "${COLAB[@]}" upload -s ps "$HOME/.config/rclone/rclone.conf" /content/rclone.conf >/dev/null
    ./scripts/colab_exec.sh ps scripts/_colab_setup_rclone.py 600 >/dev/null 2>&1
  fi
}

for EP in "${EPISODES[@]}"; do
  BASE="${EP%.mp3}"
  # is it already done in Drive?
  if rclone lsf -R "$STATE" 2>/dev/null | grep "${STAGE}/_DONE.json" | grep -q "__${BASE}/"; then
    echo "[resilient] $EP already done ($STAGE)"; continue
  fi

  for attempt in 1 2 3; do
    echo "[resilient] $EP attempt $attempt"
    ensure_session
    # make sure the audio is on the VM
    echo "{\"episodes\":[\"$EP\"]}" > /tmp/fetch.json
    "${COLAB[@]}" upload -s ps /tmp/fetch.json /content/fetch.json >/dev/null 2>&1
    ./scripts/colab_exec.sh ps scripts/_colab_fetch_audio.py 600 >/dev/null 2>&1
    # run the stage on just this episode
    echo "{\"stage\":\"$STAGE\",\"episodes\":\"$EP\"}" > /tmp/run.json
    "${COLAB[@]}" upload -s ps /tmp/run.json /content/run.json >/dev/null 2>&1
    OUT=$(./scripts/colab_exec.sh ps scripts/colab_run_stage.py 1500 2>&1)
    if echo "$OUT" | grep -qE "done ${EP}|skip ${EP}"; then
      echo "$OUT" | grep -E "^\[run\] (done|skip)"
      break
    fi
    echo "[resilient] $EP attempt $attempt did not complete; retrying"
    sleep 5
  done
done
echo "[resilient] finished"
