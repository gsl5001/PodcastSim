# Changelog

The decisions that shaped PodcastSim, newest first — what we chose, what we chose
it *over*, and why.

---

## 2026-09-05 — CrisperWhisper for verbatim transcripts  ⭐

- **Switched S3 ASR from faster-whisper large-v3 → CrisperWhisper**
  (`nyrahealth/CrisperWhisper`). *Why:* the clones should reproduce the hosts'
  **real disfluent delivery** — the "um"s, false starts and stammers. Plain
  Whisper cleans those up; CrisperWhisper keeps them verbatim. Verified on the
  first episodes: it preserves stammers ("I, I will", "company company") and
  discourse fillers, and marks hesitation pauses. (Trade-off we accepted: it maps
  filled pauses to pauses rather than literal "um" tokens.)
- **Per-episode `clips.jsonl`, not a shared per-host `metadata.jsonl`.** *Why:* a
  shared append file corrupts when a run resumes mid-way; per-episode files are
  idempotent. A later aggregation step assembles the per-host dataset.

## 2026-09-04 — Hosts found; fine-tuning validated

- **Hosts found by unsupervised clustering, not hand-picked enrollment clips.**
  S2 embeds every diarized speaker and clusters across episodes; the recurring
  clusters are the hosts. *Over:* the plan's S7.2 enrollment approach. It
  **self-validated** — one host is absent from the later-era episodes, and the
  clustering reproduced exactly that pattern on its own.
- **Full T3 fine-tune beats zero-shot** (clearly more like the host). Exaggeration
  is per-host. Plan set: gather more data + retrain with verbatim labels.
- **Drop troublesome episodes rather than risk mixing speakers.** Live episodes
  (audience + guests, different acoustics) are excluded — bad training data is
  worse than less data.

## 2026-09-03 — First datasets

- S2 per-episode embedding; the bootstrap first found the founding duo, then the
  full set once more episodes were processed.
- First S3 training clips produced; clipping-filter bug fixed (MP3 overshoot on
  hot masters is not digital clipping — peak-normalise *then* test).

## 2026-09-02 — Foundations

- **State lives on Google Drive** via rclone (two remotes: read-only source,
  read-write state), with **per-episode `_DONE.json` manifests** so a killed
  session resumes exactly where it stopped. *Why:* free compute reclaims VMs
  constantly; the pipeline has to be interruption-proof.
- **AST for S0 segmentation**, not inaSpeechSegmenter. *Over:* ~3 GB of
  TensorFlow+ONNX CUDA extras and a dedicated venv — AST is already needed in S3,
  so S0 and S3 share one model and one threshold set.
- **Played-media detection** added (a gap in the plan): comedy podcasts play
  YouTube/film clips whose voices would otherwise cluster as fake speakers and
  poison a host's data.
- **Analysis copy is WAV, not FLAC.** Both lossless and bit-identical, but WAV
  needs no decode at read time; a FLAC analysis copy made pyannote runs 3× slower.
- **`max_speakers=8`** (plan suggested 6). Over-segmentation is the *safe* error:
  a host split across clusters is repaired downstream, but a played clip merged
  into a host cluster is unrecoverable.
- **Long episodes chunked (~39 min) and speaker-relinked** by embedding, so a
  single diarization pass survives a free-tier VM's ~15-minute reclamation window.
- **WeSpeaker ResNet34-LM embeddings** (pyannote's own), not the plan's ResNet293
  — the `wespeaker` package won't install on the Colab image.
