# scripts/

Run helpers for the GPU stages (Colab and Kaggle). The pipeline library lives in
`src/podcastsim/`; these drive it on remote compute and move state to/from Google
Drive. Convention: a leading `_` means "internal helper, invoked by another
script, not run directly."

## Colab (current compute path)

| Script | Purpose |
|---|---|
| `colab_bootstrap.sh` | Provision a Colab session: install deps, sync rclone config + package. |
| `colab_exec.sh`      | The one safe way to run a Python file on a Colab session (timeout + stale-module purge + version asserts). |
| `colab_run_stage.py` | Run a pipeline stage across episodes on Colab. |
| `colab_smoke.py`     | Minimal "what hardware did we get" check (`colab run --gpu T4 scripts/colab_smoke.py`). |
| `colab_sync.sh` / `colab_put.sh` / `colab_pull_work.sh` | Move package/state/results between local, the VM, and Drive. |
| `_colab_setup_rclone.py` / `_colab_restore_work.py` / `_colab_fetch_audio.py` | Internal setup helpers used during bootstrap. |

See `docs/RUNBOOK.md` for auth (`--auth=adc`) and `docs/COLAB_SKILL.md` for CLI gotchas.

## Kaggle (used for the heavy, long-running stages — stable 12 h sessions)

| Script | Purpose |
|---|---|
| `kaggle/s3_trih_kernel.py` | S3: CrisperWhisper verbatim clip cutting across episodes. |
| `kaggle/train_trih_kernel.py` | S4: fine-tune the host voice models. |
| `kaggle/synth_trih_kernel.py` | Synthesize sample clips from the fine-tuned voices. |

Pushed with the Kaggle CLI (`kaggle kernels push`); they auto-run on a GPU
session and read/write Drive via rclone (secrets in a private Kaggle dataset).

## Other

| Script | Purpose |
|---|---|
| `drive_sync.sh`    | rclone sync helper between local and the Drive state remote. |
| `run_resilient.sh` | Re-run a command across session reclamation (retry/resume wrapper). |
