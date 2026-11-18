"""Cross-episode host discovery by clustering (plan S7.2, Plan B).

"pool all local-speaker centroids from all episodes ... run agglomerative
clustering with cosine distance ... inspect the largest three clusters -- they
are the hosts. Recurring guests appear as medium clusters."

This runs locally on the per-episode embeddings S2 produced. No enrollment clips
needed: the recurring voices reveal themselves. The output is a report the owner
confirms and names -- which then seeds enrollment for Plan A.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LocalSpeaker:
    episode_id: str
    label: str
    centroid: np.ndarray
    speech_s: float
    date: str | None

    @property
    def key(self) -> str:
        return f"{self.episode_id}:{self.label}"


@dataclass
class Cluster:
    cluster_id: int
    members: list[LocalSpeaker] = field(default_factory=list)

    @property
    def total_speech_s(self) -> float:
        return sum(m.speech_s for m in self.members)

    @property
    def n_episodes(self) -> int:
        return len({m.episode_id for m in self.members})

    def centroid(self) -> np.ndarray:
        c = np.mean([m.centroid for m in self.members], axis=0)
        n = np.linalg.norm(c)
        return c / n if n else c


def load_local_speakers(work_root: str | Path) -> list[LocalSpeaker]:
    """Gather every episode's local-speaker embeddings from S2 output."""
    work = Path(work_root)
    speakers: list[LocalSpeaker] = []
    for npz in sorted(work.glob("*/s2_identify/local_embeddings.npz")):
        ep_dir = npz.parent
        episode_id = ep_dir.parent.name
        data = np.load(npz, allow_pickle=False)  # labels are plain strings; never unpickle Drive-sourced files
        labels = data["labels"]
        cents = data["centroids"]
        meta = {}
        try:
            meta = json.load(open(ep_dir / "local_speakers.json"))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        date = meta.get("publish_date")
        speech = {r["local_speaker"]: r["total_exclusive_s"]
                  for r in meta.get("speakers", [])}
        for lbl, cen in zip(labels, cents):
            speakers.append(LocalSpeaker(
                episode_id=episode_id, label=str(lbl),
                centroid=np.asarray(cen, dtype=np.float32),
                speech_s=float(speech.get(str(lbl), 0.0)), date=date))
    return speakers


def _cosine_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    sim = np.clip(v @ v.T, -1.0, 1.0)
    return 1.0 - sim


def cluster_speakers(speakers: list[LocalSpeaker], sim_threshold: float
                     ) -> list[Cluster]:
    """Agglomerative clustering with cosine distance (plan S7.2).

    distance_threshold = 1 - sim_threshold: two local speakers merge when their
    cosine similarity is at least sim_threshold.
    """
    if not speakers:
        return []
    from sklearn.cluster import AgglomerativeClustering

    vectors = np.stack([s.centroid for s in speakers])
    if len(speakers) == 1:
        labels = np.array([0])
    else:
        dist = _cosine_distance_matrix(vectors)
        model = AgglomerativeClustering(
            n_clusters=None, metric="precomputed", linkage="average",
            distance_threshold=1.0 - sim_threshold)
        labels = model.fit_predict(dist)

    clusters: dict[int, Cluster] = {}
    for spk, lab in zip(speakers, labels):
        clusters.setdefault(int(lab), Cluster(int(lab))).members.append(spk)
    ranked = sorted(clusters.values(), key=lambda c: c.total_speech_s, reverse=True)
    for i, c in enumerate(ranked):
        c.cluster_id = i
    return ranked


def assign_hosts(clusters: list[Cluster], min_episodes: int = 2, n_hosts: int = 2,
                 min_speech_share: float = 0.20) -> dict[str, dict[str, str]]:
    """Per-episode {local_label: host_id} from the global clustering.

    A cluster is a host only if it BOTH recurs across >= `min_episodes` episodes
    AND carries at least `min_speech_share` of the largest cluster's speech time.
    Recurrence alone is not enough: a recurring guest also spans episodes (S7.2),
    but hosts dominate speaking time. On the fixtures this correctly yields 2
    hosts (14.7 and 5.9 min) and rejects a 1.0-min recurring fragment that
    appears in the duo-era episode 1 and so cannot be host C.

    Host ids are provisional (host_0, host_1, ...) until the owner names them.
    Everything else is `guest`.
    """
    if not clusters:
        return {}
    top_speech = max(c.total_speech_s for c in clusters)
    host_clusters = [c for c in clusters
                     if c.n_episodes >= min_episodes
                     and c.total_speech_s >= min_speech_share * top_speech][:n_hosts]
    host_of_cluster = {c.cluster_id: f"host_{i}" for i, c in enumerate(host_clusters)}

    per_episode: dict[str, dict[str, str]] = {}
    for c in clusters:
        hid = host_of_cluster.get(c.cluster_id, "guest")
        for m in c.members:
            per_episode.setdefault(m.episode_id, {})[m.label] = hid
    return per_episode


def summarise_clusters(clusters: list[Cluster], n_hosts: int = 2) -> dict:
    """A report: the top-N clusters are the candidate hosts."""
    rows = []
    for c in clusters:
        rows.append({
            "cluster_id": c.cluster_id,
            "candidate": "host" if c.cluster_id < n_hosts else "guest/other",
            "total_speech_min": round(c.total_speech_s / 60, 1),
            "n_episodes": c.n_episodes,
            "n_local_speakers": len(c.members),
            "members": sorted(f"{m.episode_id[:12]}:{m.label} ({m.speech_s/60:.1f}m)"
                              for m in c.members),
        })
    return {"n_clusters": len(clusters), "n_hosts_expected": n_hosts, "clusters": rows}
