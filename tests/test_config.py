"""Config loading and the stage-isolation property (plan S3 config contract)."""

import pytest

from podcastsim.config import STAGES, Config, ConfigError, load_hosts
from podcastsim.manifest import canonical_hash

pytestmark = pytest.mark.unit

REAL_CONFIG = "configs/pipeline.yaml"
REAL_HOSTS = "configs/hosts.yaml"


def write_yaml(tmp_path, body: str):
    p = tmp_path / "pipeline.yaml"
    p.write_text(body)
    return p


def minimal_body(**overrides) -> str:
    lines = []
    for s in STAGES:
        lines.append(f"{s}:")
        lines.append(f"  value: {overrides.get(s, 1)}")
    return "\n".join(lines) + "\n"


# --- the real shipped config -------------------------------------------------

def test_real_config_loads_and_validates():
    Config.load(REAL_CONFIG)  # raises if a stage section is missing


def test_real_config_has_every_stage_section():
    cfg = Config.load(REAL_CONFIG)
    for s in STAGES:
        assert cfg.stage_params(s), f"{s} section is empty"


@pytest.mark.parametrize(
    "dotted,expected",
    [
        # Spot-check thresholds against the plan's gate tables.
        ("s0_ingest.gate.speech_ratio_min", 0.60),        # Gate S0
        ("s1_diarize.min_speakers", 2),                   # S14.16: never num_speakers=3
        # Deliberately above S6.1's suggested 6 -- see pipeline.yaml for why
        # played clips make over-segmentation the safe error.
        ("s1_diarize.max_speakers", 8),
        ("s1_diarize.scoring.collar", 0.25),              # S14.4
        ("s1_diarize.gate.der_pass", 0.10),               # Gate S1-A
        ("s2_identify.margin", 0.10),                     # S7.3
        ("s3_dataset.filters.dnsmos_ovrl_min", 3.0),      # S8.6
        ("s3_dataset.loudness.target_lufs", -23.0),       # S8.7
        ("s4_voices.gate.sim_o_min", 0.60),               # Gate S4-A
        ("s4_voices.gate.wer_max", 0.08),
        ("s5_render.provenance.require_id3_disclaimer", True),  # S15 ethics
    ],
)
def test_plan_thresholds_present(dotted, expected):
    assert Config.load(REAL_CONFIG).get(dotted) == expected


def test_analysis_rate_is_16k():
    """S14.2: 'All analysis models are 16 kHz.'"""
    assert Config.load(REAL_CONFIG).get("s0_ingest.analysis_sample_rate") == 16000


def test_never_pins_num_speakers():
    """S14.16: 'never num_speakers=3; guests exist.'"""
    s1 = Config.load(REAL_CONFIG).stage_params("s1_diarize")
    assert "num_speakers" not in s1
    assert s1["min_speakers"] < s1["max_speakers"]


# --- stage isolation ---------------------------------------------------------

def test_stage_params_excludes_other_stages(tmp_path):
    cfg = Config.load(write_yaml(tmp_path, minimal_body()))
    assert cfg.stage_params("s1_diarize") == {"value": 1}


def test_editing_one_stage_does_not_invalidate_another(tmp_path):
    """The property that protects 200 hours of finished S1 work when S4's
    gates get recalibrated (plan S12.3)."""
    before = Config.load(write_yaml(tmp_path, minimal_body()))
    s1_before = canonical_hash(before.stage_params("s1_diarize"))

    after = Config.load(write_yaml(tmp_path, minimal_body(s4_voices=999)))
    assert canonical_hash(after.stage_params("s1_diarize")) == s1_before
    assert canonical_hash(after.stage_params("s4_voices")) != canonical_hash(
        before.stage_params("s4_voices")
    )


def test_real_config_stage_hashes_all_differ():
    cfg = Config.load(REAL_CONFIG)
    hashes = {s: canonical_hash(cfg.stage_params(s)) for s in STAGES}
    assert len(set(hashes.values())) == len(STAGES)


# --- error handling ----------------------------------------------------------

def test_missing_stage_section_raises(tmp_path):
    body = "\n".join(f"{s}:\n  value: 1" for s in STAGES if s != "s5_render")
    with pytest.raises(ConfigError, match="s5_render"):
        Config.load(write_yaml(tmp_path, body))


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        Config.load(tmp_path / "nope.yaml")


def test_non_mapping_top_level_raises(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        Config.load(write_yaml(tmp_path, "- just\n- a\n- list\n"))


def test_unknown_stage_raises(tmp_path):
    cfg = Config.load(write_yaml(tmp_path, minimal_body()))
    with pytest.raises(ConfigError, match="unknown stage"):
        cfg.stage_params("s9_nonexistent")


def test_missing_key_raises_but_default_wins(tmp_path):
    cfg = Config.load(write_yaml(tmp_path, minimal_body()))
    with pytest.raises(ConfigError, match="missing config key"):
        cfg.get("s1_diarize.nope")
    assert cfg.get("s1_diarize.nope", 42) == 42
    assert cfg.get("s1_diarize.nope", None) is None  # None is a real default


# --- hosts -------------------------------------------------------------------

def test_real_hosts_file_has_two_hosts():
    """Plan S1.3 (revised): the project now targets two hosts."""
    hosts = load_hosts(REAL_HOSTS)
    assert len(hosts) == 2
    assert [h.host_id for h in hosts] == ["host_0", "host_1"]


def test_duplicate_host_id_raises(tmp_path):
    p = tmp_path / "hosts.yaml"
    p.write_text("hosts:\n  - host_id: host_a\n  - host_id: host_a\n")
    with pytest.raises(ConfigError, match="duplicate"):
        load_hosts(p)


def test_host_without_id_raises(tmp_path):
    p = tmp_path / "hosts.yaml"
    p.write_text("hosts:\n  - display_name: nameless\n")
    with pytest.raises(ConfigError, match="host_id"):
        load_hosts(p)


def test_hosts_missing_list_raises(tmp_path):
    p = tmp_path / "hosts.yaml"
    p.write_text("something_else: 1\n")
    with pytest.raises(ConfigError, match="hosts"):
        load_hosts(p)
