# Runbook

Written as we go (plan S3). Everything here is verified in this environment unless
marked otherwise.

## One-time setup

### Local (macOS dev machine)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

### Colab CLI

`google-colab-cli` needs two pins in this environment — without them install fails
or every `exec` dies with `AttributeError`. See `docs/DECISIONS.md` for why.

```bash
uv tool install -c ~/.config/podcastsim/colab-cli-constraints.txt google-colab-cli
```

Authenticate with **ADC** (the agent-friendly path; all four scopes required):

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
colab --auth=adc whoami     # expect 4 scopes + the right email
```

Always pass `--auth=adc` explicitly: the bundled skill claims it is the default,
but `colab --help` on 0.6.0 says `oauth2`.

## Daily commands

```bash
podcastsim check                      # validate configs, no data touched
podcastsim index                      # scan data/raw, refresh index, flag duplicates
podcastsim status                     # per-stage completion across the archive
podcastsim run s0_ingest --dry-run    # what would run
podcastsim run s0_ingest -e all       # run a stage
podcastsim run s1_diarize -e 3f2a     # one episode by digest prefix
podcastsim run s1_diarize -e all --force   # ignore existing _DONE.json
```

Selectors accept `all`, a source path (`12.mp3`), a full episode id, or a digest
prefix.

## Colab

```bash
colab --auth=adc run --gpu T4 scripts/colab_smoke.py   # provisions, runs, tears down
colab --auth=adc sessions                              # check nothing is leaking
colab --auth=adc stop -s <name>                        # always, if you used `new`
```

`colab run` self-cleans even when the script errors; `colab new` does not — an
unstopped session burns compute indefinitely. Check `sessions` after any crash.

**Not agent-runnable** (they need a TTY): `colab repl`, `colab console`,
`colab auth`, `colab drivemount`.

## Resumability

A stage writes `data/work/{episode_id}/{stage}/_DONE.json` **only on success**.
So:

- killing a run mid-episode loses that episode and nothing else;
- re-running skips completed episodes;
- editing that stage's section of `configs/pipeline.yaml` changes its
  `params_hash` and re-runs it — editing a *different* stage's section does not.

To see why something will re-run, `podcastsim run <stage> --dry-run`.

## Gotchas

- **Never `num_speakers=3`** (S14.16). Guests exist; use min/max and let S2 merge.
- **All analysis is 16 kHz** (S14.2); cut training clips from the native-rate decode.
- **Decode each MP3 once** and slice by sample index (S14.3) — MP3 encoder delay
  and seek inaccuracy shift clip boundaries.
- **Shard clips** (S14.13). Drive is slow for many small files.

## Orphaned sessions (free tier)

If a runtime is reclaimed mid-run, the CLI prints:

```
[colab] Session 'ps' appears to be lost (404/401). Cleaning up.
```

That cleanup deletes the **local** record (`~/.config/colab-cli/sessions.json`
becomes `{}`) while the **server-side assignment survives**. The result:

- `colab sessions` shows it as `[?] gpu-t4-...`
- `colab stop -s <that-name>` returns "not found" — stop only works on sessions
  with a local record
- `colab new` then fails with `TooManyAssignmentsError: ... Precondition Failed`,
  because the orphan is holding a free-tier slot

**Fix:** terminate it in the browser — colab.research.google.com -> Runtime ->
Manage sessions -> Terminate. It also times out on its own once the keep-alive
daemon dies, but that can take an hour.

**Avoid:** prefer `colab run` (self-cleans even on error) for one-shot work, and
`scripts/colab_bootstrap.sh` for sessions, which probes the kernel rather than
trusting `colab status` — status succeeds against a stale local record.

## Always restart the kernel after `colab install`

`colab install` upgrades packages on disk, but **kernel state persists across
`exec`** — so a module already imported stays at the old version. When the
upgraded package is compiled (numpy, scipy), the mismatch surfaces somewhere
unrelated:

```
ModuleNotFoundError: Could not import module 'ASTForAudioClassification'.
  ...
AttributeError: module 'numpy._core._multiarray_umath' has no attribute '_blas_supports_fpe'
```

That was numpy 2.1.3 live in the kernel while 2.5.2 sat on disk, with scipy
built against 2.5.x. `colab restart-kernel -s <name>` fixes it.
`scripts/colab_bootstrap.sh` now does this automatically after any install.

Diagnose with: `python -c "import numpy, importlib.metadata as md; \
print(numpy.__version__, md.version('numpy'))"` — if those two disagree, the
kernel is stale.

## Stale kernel state bites three ways

`colab exec` reattaches to the SAME kernel, so anything already imported stays
live. This has caused three distinct failures:

1. **Upgraded compiled deps** — `colab install` moved numpy 2.1.3 -> 2.5.2 on
   disk while the kernel kept 2.1.3, and scipy (built against 2.5.x) failed far
   away as `Could not import ASTForAudioClassification`. Fix: restart the kernel
   after any install.
2. **Re-synced source** — `colab_sync.sh` replaced `src/podcastsim` on disk, but
   the kernel served the old modules from `sys.modules`. A `CODE_VERSION` bump
   had no effect and the stage skipped as already-done. Fix:
   `scripts/colab_run_stage.py` now purges `podcastsim*` from `sys.modules`
   before importing.
3. **Long-running exec** — the client abandons the output stream after ~30 s
   while the kernel keeps working, so a completed run looks like a failure.
   Fix: always pass `--timeout`, and check `colab status` before concluding
   anything failed.

Rule of thumb: **the kernel is a long-lived process, not a fresh interpreter.**
If a change on disk does not seem to take effect, suspect `sys.modules` first.

## Kaggle GPU (alternative to Colab, no reclamation)

Free Colab reclaims the VM every ~15 min, which makes long jobs (65-min episode
diarization, voice training) a grind of retries. Kaggle gives ~12 h background
sessions and ~30 GPU-h/week free, so one run does a whole stage across every
episode. Execution is batch (push a kernel, it runs to completion, results land
in Drive), not interactive.

### One-time setup

1. Kaggle account, **phone-verified** (Settings -> Phone Verification). Required
   for GPU and internet access in kernels; without it the kernel has neither.
2. API token: Settings -> API -> Create New Token -> `kaggle.json`. Save it:
   ```
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
   (`{"username": "...", "key": "..."}`)
3. Create the private secrets dataset (bundles rclone.conf + hf_token so the
   kernel can reach Drive):
   ```
   # bundle rclone.conf + hf_token into a private Kaggle dataset the kernel attaches
   kaggle datasets create -p <secrets_dir>   # or `datasets version` to update
   ```

### Running a stage

The heavy stages run as standalone kernels pushed with the Kaggle CLI (they
auto-run on a GPU session and read/write Drive via rclone):

```
kaggle kernels push -p <dir>     # dir holds the kernel .py + kernel-metadata.json
kaggle kernels status <user>/<slug>
```

The TRIH kernels are in `scripts/kaggle/` (`s3_trih_kernel.py`,
`train_trih_kernel.py`, `synth_trih_kernel.py`); each pulls package + state +
audio from Drive, runs, and pushes results back per episode -- same persistence
model as Colab, so `podcastsim identify` / `gates` work locally afterward once
you `drive_sync.sh pull`.

Data flow is identical to Colab: `gdrive` (readonly source audio), `gdrive_rw`
(drive.file state). Nothing about S0-S3 changes.
