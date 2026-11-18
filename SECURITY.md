# Security notes

This is a single-operator research pipeline. Summary of its security posture and
the residual hardening items, from an audit of the code and git history.

## Design (good by default)
- **Secrets are never committed.** Tokens (HF, Kaggle, Google/rclone) live in
  `~/.config`/`$HOME` 0600 files or environment variables and are `.gitignore`d
  (`rclone.conf`, `hf_token`, `kaggle.json`, `.env`, `*.pem`, `*.key`). The
  `_colab_setup_rclone` helper chmods staged secrets to 0600.
- **Least privilege Drive access:** a read-only remote for source audio and a
  `drive.file`-scoped remote for state (can only touch what it created).
- **Safe loading:** configs via `yaml.safe_load`; fine-tuned voice checkpoints via
  `safetensors` (not `torch.load`/pickle); diarization embeddings saved/loaded
  without pickle (`allow_pickle=False`, string label dtype).
- No `eval`/`exec`/`os.system`; in-package subprocess calls are list-form ffmpeg.

## Operator responsibilities
- **Keep the Kaggle secrets dataset and all kernels PRIVATE.** They carry the
  rclone OAuth config + HF token; Kaggle datasets/kernels are public by default.
  Prefer Kaggle *Secrets* add-ons over a dataset where possible.
- Never commit anything under `data/`, `models/`, `downloads/` (gitignored).

## Residual hardening (low-risk, not yet applied)
- Pin the runtime-fetched third-party code to reviewed commits:
  `stlohrey/chatterbox-finetuning` (git clone) and the `torch.hub` Silero VAD ref.
- Verify a pinned SHA-256 of the rclone archive before extracting/executing it.
- In the Kaggle kernels, pass rclone filenames as argv (or `shlex.quote`) instead
  of shell-interpolating them (currently safe because all names are
  pipeline-sanitized; latent only).
- `downloads/fetch_trih.py` (local only, gitignored): restrict download URL
  schemes to http/https.
