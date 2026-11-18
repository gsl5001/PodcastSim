# PodcastSim architecture

A map of how the pipeline fits together. Read this before diving into
`docs/PODCASTSIM_PLAN.md` (the full spec) or the source.

## What it does

Podcast MP3s → cloned host voices. The pipeline ingests episodes, finds who
speaks when, figures out which speakers are the recurring hosts, cuts a clean
transcribed training set per host, and fine-tunes a voice model per host. The
project targets **two hosts** (configurable — see `configs/hosts.yaml` and
`podcastsim identify --hosts N`). (Rendering scripted episodes — S5 — is not built yet.)

## The two "S" numbering schemes (important)

There are **two different `S` namespaces** and they do **not** line up:

- **Pipeline stages S0–S5** — the runtime stages (`s0_ingest` … `s5_render`), one
  module each under `src/podcastsim/stages/`. (S6 QA was dropped.)
- **Plan sections S0–S15** — numbered sections of `docs/PODCASTSIM_PLAN.md`
  (and `S1.3`, `S7.2`, etc. sub-points).

A stage module's docstring cites the plan *section* it implements, so the
numbers differ. The mapping:

| Pipeline stage | Implements plan section | What it does |
|---|---|---|
| `s0_ingest`  | plan S5 | decode to 16 kHz analysis WAV, probe metadata/bandwidth, AST speech/music segmentation |
| `s1_diarize` | plan S6 | pyannote diarization → who speaks when (RTTM); chunks long episodes |
| `s2_identify`| plan S7 | embed each diarized speaker, cluster across episodes → the 2 hosts |
| `s3_dataset` | plan S8 | cut clean host turns, VAD-segment, transcribe (verbatim), filter → per-host clips |
| `s4_voices`  | plan S9 | fine-tune a voice model per host (**currently runs via `scripts/`, not this stage**) |
| `s5_render`  | plan S10 | render scripted episodes (not built) |

"S14.x" references in comments are the plan's infra/operational section; "S15"
is ethics/consent. (S6 QA — plan S11 — was dropped; its S15 provenance
constraints moved under `s5_render` config.)

## Data flow

```
   source MP3s (Google Drive / data/raw)
        │
   S0   │  decode → data/work/{ep}/s0_ingest/   (audio16k.wav, meta.json, segments_music.json)
        ▼
   S1   │  diarize → data/work/{ep}/s1_diarize/ (exclusive.rttm, overlap.json, embeddings.npz)
        ▼
   S2   │  embed+cluster → data/work/{ep}/s2_identify/ (host_map.json, local_embeddings.npz)
        ▼
   S3   │  cut+transcribe → data/work/{ep}/s3_dataset/clips.jsonl  (+ clip FLACs in data/hosts/)
        │
        │  aggregate_host_metadata(work, hosts)   ← bridges per-episode → per-host
        ▼
             data/hosts/{host}/metadata.jsonl  +  clips/
        │
   S4   │  fine-tune → Drive s4/finetune_verbatim/{host}/t3_cfg.safetensors
        ▼
   (S5: render — not built)
```

Every stage writes a `_DONE.json` manifest; a stage with a valid manifest for
the current inputs/params is **skipped** on re-run (resumability). See
`manifest.py` (`StageGuard`).

## Package layout: orchestrators vs implementation

Two layers:

- **`src/podcastsim/stages/sN_*.py`** — thin **orchestrators**. Each exposes
  `run_episode(episode, cfg, paths, *, force=False) -> Manifest | None`
  (returns `None` when skipped). They handle manifests, input hashing, and
  calling into the implementation modules.
- **`src/podcastsim/*.py`** (flat) — the **implementation** (the actual DSP,
  diarization, clustering, clip-cutting logic).

Stage → implementation modules it uses:

| Stage | Implementation modules |
|---|---|
| `s0_ingest`  | `audio.py`, `segmentation.py` |
| `s1_diarize` | `diarize_pyannote.py`, `chunking.py`, `rttm.py` |
| `s2_identify`| `embed_speaker.py`, `identify.py`, `rttm.py` |
| `s3_dataset` | `s3_select.py`, `s3_clips.py` (+ `clip_quality.py`), `rttm.py` |
| `s4_voices`  | `s4_refbank.py` (reference selection); **training/synthesis live in `scripts/kaggle/`**) |

Cross-cutting: `config.py` (config + the canonical `STAGES` tuple), `episodes.py`
(episode index + selectors + exclusions), `ids.py` (content-hash episode ids),
`manifest.py` (resumability), `gates.py` (automated quality gates), `cli.py`
(the `podcastsim` command).

## Where S4+ actually runs

The `stages/s4_voices.py` and `s5_render.py` modules are **stubs**.
Voice fine-tuning and synthesis currently run as standalone kernels on Colab/Kaggle
under `scripts/` — see `scripts/README.md`. `s4_refbank.py` (reference-clip
selection) is the one piece of S4 logic that lives in the package; it consumes
the per-host `metadata.jsonl` produced by `s3_clips.aggregate_host_metadata`.

## Compute

GPU stages (S1–S4) run on **Colab** (GPU runtime, state synced to Google Drive
via rclone); S0 is CPU-only and runs locally. The `scripts/colab_*` tooling and
`docs/RUNBOOK.md` cover auth and running. (Kaggle was used earlier; the
`scripts/kaggle/` kernels remain but Colab is the current path.)
