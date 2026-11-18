"""`podcastsim <command>` (plan S3).

CLI + YAML scripts only -- plan S1.2 rules out a GUI for v1.
"""

from __future__ import annotations

import json

from pathlib import Path

import typer

from . import episodes as ep_mod
from .config import DEFAULT_CONFIG, STAGES, Config, ConfigError, load_hosts
from .manifest import read_manifest
from .stages import load as load_stage

app = typer.Typer(add_completion=False, help="Podcast archive -> cloned hosts -> episodes.")


def _paths(cfg: Config) -> dict[str, Path]:
    return {k: Path(v) for k, v in cfg.get("paths").items()}


def _load(config: Path) -> tuple[Config, dict[str, Path]]:
    try:
        cfg = Config.load(config)
    except ConfigError as e:
        typer.secho(f"config error: {e}", fg="red", err=True)
        raise typer.Exit(2)
    return cfg, _paths(cfg)


@app.command("index")
def index_cmd(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """Scan data/raw and refresh the episode index."""
    cfg, paths = _load(config)
    if not paths["raw"].is_dir():
        typer.secho(f"raw directory not found: {paths['raw']}", fg="red", err=True)
        raise typer.Exit(2)

    found = ep_mod.refresh(paths["raw"], paths["work"])
    total_gb = sum(e.size for e in found.values()) / 2**30
    typer.echo(f"indexed {len(found)} episodes ({total_gb:.2f} GiB) -> {ep_mod.index_path(paths['work'])}")

    dups = ep_mod.duplicates(found)
    if dups:
        typer.secho(f"{len(dups)} duplicate digest(s):", fg="yellow")
        for digest, sources in dups.items():
            typer.echo(f"  {digest}: {', '.join(sources)}")


@app.command("run")
def run_cmd(
    stage: str = typer.Argument(..., help=f"one of: {', '.join(STAGES)}"),
    episodes: str = typer.Option("all", "--episodes", "-e", help="'all', a source path, an id, or a digest prefix"),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    force: bool = typer.Option(False, "--force", help="re-run even if _DONE.json matches"),
    dry_run: bool = typer.Option(False, "--dry-run", help="report what would run, touch nothing"),
) -> None:
    """Run one stage over selected episodes."""
    if stage not in STAGES:
        typer.secho(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}", fg="red", err=True)
        raise typer.Exit(2)

    cfg, paths = _load(config)
    known = ep_mod.load_index(paths["work"])
    if not known:
        typer.secho("empty episode index -- run `podcastsim index` first", fg="red", err=True)
        raise typer.Exit(2)

    selected = ep_mod.resolve(known, episodes)
    if not selected:
        typer.secho(f"no episodes matched {episodes!r}", fg="red", err=True)
        raise typer.Exit(2)

    exclusions = ep_mod.load_exclusions()
    kept = []
    for ep in selected:
        reason = ep_mod.is_excluded(ep, exclusions)
        if reason:
            typer.secho(f"  excluded {ep.source}: {reason}", fg="yellow")
        else:
            kept.append(ep)
    selected = kept
    if not selected:
        typer.secho("all selected episodes are excluded", fg="yellow")
        return

    module = load_stage(stage)
    ran = skipped = failed = 0

    for ep in selected:
        m = read_manifest(paths["work"], ep.episode_id, stage)
        if dry_run:
            typer.echo(f"  [{'done' if m else 'todo'}] {ep.source}")
            continue
        try:
            result = module.run_episode(ep, cfg, paths, force=force)
        except NotImplementedError as e:
            typer.secho(f"  {e}", fg="yellow")
            raise typer.Exit(3)
        except Exception as e:  # one bad episode must not kill the batch
            failed += 1
            typer.secho(f"  FAIL {ep.source}: {type(e).__name__}: {e}", fg="red")
            continue
        if result is None:
            skipped += 1
        else:
            ran += 1

    if not dry_run:
        typer.echo(f"{stage}: {ran} ran, {skipped} skipped, {failed} failed")
    if failed:
        raise typer.Exit(1)


@app.command("status")
def status_cmd(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """Per-stage completion counts across the indexed archive."""
    cfg, paths = _load(config)
    known = ep_mod.load_index(paths["work"])
    if not known:
        typer.echo("no episode index yet -- run `podcastsim index`")
        return

    typer.echo(f"{len(known)} episodes indexed\n")
    for stage in STAGES:
        done = sum(
            read_manifest(paths["work"], e.episode_id, stage) is not None
            for e in known.values()
        )
        bar = "#" * int(20 * done / len(known))
        typer.echo(f"  {stage:12s} {done:>4}/{len(known):<4} |{bar:<20}|")


@app.command("gates")
def gates_cmd(
    stage: str = typer.Option("all", "--stage", "-s"),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """Evaluate automated gates against the manifests on disk (plan S12)."""
    from .gates import ESCALATE, GATES, load_manifests, summarise

    cfg, paths = _load(config)
    if stage != "all" and stage not in GATES:
        typer.secho(
            f"no gate defined for stage '{stage}' (have: {', '.join(sorted(GATES))})",
            fg="red")
        raise typer.Exit(2)
    wanted = GATES if stage == "all" else {stage: GATES[stage]}
    worst_any = False

    for name, fn in wanted.items():
        mans = load_manifests(paths["work"], name)
        checks = fn(mans, cfg)
        typer.echo(f"\n=== {name} ({len(mans)} episodes) -> {summarise(checks)} ===")
        for c in checks:
            colour = {"PASS": "green", "WARN": "yellow",
                      "ESCALATE": "red", "N/A": "blue"}[c.status]
            typer.secho(str(c), fg=colour)
        worst_any |= any(c.status == ESCALATE for c in checks)

    if worst_any:
        raise typer.Exit(1)


@app.command("identify")
def identify_cmd(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    n_hosts: int = typer.Option(2, "--hosts"),
) -> None:
    """Cluster S2 embeddings to find the recurring hosts (plan S7.2)."""
    from .identify import (assign_hosts, cluster_speakers, load_local_speakers,
                           summarise_clusters)

    cfg, paths = _load(config)
    speakers = load_local_speakers(paths["work"])
    if not speakers:
        typer.secho("no S2 embeddings found -- run `podcastsim run s2_identify` first",
                    fg="red", err=True)
        raise typer.Exit(2)

    sim = float(cfg.get("s2_identify.sim_threshold", 0.55))
    clusters = cluster_speakers(speakers, sim)
    report = summarise_clusters(clusters, n_hosts)

    typer.echo(f"{len(speakers)} local speakers across "
               f"{len({s.episode_id for s in speakers})} episodes")
    typer.echo(f"clustered at sim>={sim} -> {report['n_clusters']} clusters\n")
    for row in report["clusters"]:
        colour = "green" if row["candidate"] == "host" else "yellow"
        typer.secho(f"  cluster {row['cluster_id']} [{row['candidate']}]: "
                    f"{row['total_speech_min']}m across {row['n_episodes']} episodes, "
                    f"{row['n_local_speakers']} local speakers", fg=colour)
        for m in row["members"]:
            typer.echo(f"      {m}")

    out = paths["out"] / "s2_clusters.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    # Persist the per-episode host assignment S3 consumes.
    assignment = assign_hosts(clusters, min_episodes=2, n_hosts=n_hosts)
    n_hosts_found = len({h for ep in assignment.values() for h in ep.values() if h != "guest"})
    for episode_id, host_map in assignment.items():
        d = paths["work"] / episode_id / "s2_identify"
        if d.is_dir():
            (d / "host_map.json").write_text(json.dumps(host_map, indent=2))
    typer.echo(f"\nreport -> {out}")
    typer.secho(f"assigned {n_hosts_found} provisional host(s) across "
                f"{len(assignment)} episodes -> host_map.json per episode", fg="green")
    if n_hosts_found < n_hosts:
        typer.secho(f"  only {n_hosts_found} of {n_hosts} hosts recur across >=2 episodes; "
                    "the rest need more episodes or enrollment clips", fg="yellow")


@app.command("check")
def check_cmd(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """Validate configs without touching data."""
    cfg, paths = _load(config)
    typer.secho(f"config OK: {config}", fg="green")
    for stage in STAGES:
        typer.echo(f"  {stage:12s} {len(cfg.stage_params(stage))} keys")
    try:
        hosts = load_hosts()
        # Hosts are found by clustering (S2), not enrollment clips, so an empty
        # enrollment_clips list is expected -- see configs/hosts.yaml.
        typer.secho(f"hosts OK: {len(hosts)} defined", fg="green")
    except ConfigError as e:
        typer.secho(f"hosts config error: {e}", fg="red", err=True)
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
