#!/usr/bin/env bash
# Sync pipeline state between the local tree and Drive.
#
# Plan S2.2: "State lives on Google Drive ... never on the runtime disk."
# Two remotes by design:
#   gdrive     scope=drive.readonly  -- source audio, structurally unwritable
#   gdrive_rw  scope=drive.file      -- pipeline state, cannot see anything else
# The read-only source remote is what enforces plan rule 2 ("Never delete
# source MP3s") against any bug on our side.
#
# Usage: scripts/drive_sync.sh push|pull [local_dir] [remote_subdir]
set -euo pipefail
ACTION="${1:-push}"; LOCAL="${2:-data/work}"; REMOTE="${3:-work}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
DEST="gdrive_rw:podcastsim_state/$REMOTE"
RC=(rclone --drive-acknowledge-abuse -q)

case "$ACTION" in
  push) mkdir -p "$LOCAL"; "${RC[@]}" copy "$LOCAL" "$DEST" ;;
  pull) mkdir -p "$LOCAL"; "${RC[@]}" copy "$DEST" "$LOCAL" ;;
  *) echo "usage: $0 push|pull [local] [remote]" >&2; exit 2 ;;
esac
echo "[drive] $ACTION $LOCAL <-> $DEST"
"${RC[@]}" size "$DEST" 2>/dev/null | tail -2
