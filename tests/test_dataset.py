import json

import pytest
import yaml
from conftest import needs_sumo
from trafma.collect import collect, load_collect_config
from trafma.config import ConfigError

pytestmark = needs_sumo


def write(tmp_path, splits):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"dataset_id": "t", "output_dir": str(tmp_path / "data"),
                                 "base": {"run": {"duration_s": 400}},
                                 "scenarios": [{"name": "moderate", "vehicles": 80}],
                                 "splits": splits}))
    return p


def test_overlapping_split_seeds_rejected(tmp_path):
    with pytest.raises(ConfigError, match="disjoint"):
        load_collect_config(write(tmp_path, {"train": [1, 2], "test": [2]}))


def test_missing_sections_rejected(tmp_path):
    p = tmp_path / "x.yaml"; p.write_text("dataset_id: a\n")
    with pytest.raises(ConfigError):
        load_collect_config(p)


@pytest.fixture(scope="module")
def man(tmp_path_factory):
    tp = tmp_path_factory.mktemp("ds")
    return collect(write(tp, {"train": [1], "test": [2]}), out=lambda *a: None), tp / "data" / "t"


def test_integrity_block_clean(man):
    m, _ = man
    i = m["integrity"]
    assert i["split_run_overlap"] == [] and i["windows_violating_run_or_horizon_rule"] == 0
    assert i["evaluator_keys_found_in_agent_logs"] == [] and i["all_runs_have_complete_rsu_logs"]
    assert i["total_collisions"] == i["total_teleports"] == 0


def test_windows_never_cross_runs_and_labels_are_in_the_future(man):
    m, d = man
    H = m["definitions"]["horizon_s"]
    for run in m["runs"]:
        for l in open(d / "runs" / run / "windows.jsonl"):
            w = json.loads(l)
            assert w["client"].startswith(run + "/")
            assert w["label_t"] == pytest.approx(w["end_t"] + H)


def test_vehicle_windows_require_full_incoming_history(man):
    m, d = man
    hist = m["definitions"]["history_len"]
    run = next(iter(m["runs"]))
    wins = [json.loads(l) for l in open(d / "runs" / run / "windows.jsonl")]
    vw = [w for w in wins if w["type"] == "vehicle"]
    assert vw
    by = {}
    for w in vw:
        by.setdefault(w["client"], []).append(w["end_t"])
    for cid, ends in list(by.items())[:15]:
        vid = cid.split("/")[1]
        recs = [json.loads(l) for l in open(d / "runs" / run / "agents" / f"{vid}.jsonl")]
        scope = {r["t"]: r["view"]["scope"] for r in recs}
        for e in ends:
            assert all(scope.get(e - k) == "incoming_local" for k in range(hist))


def test_trainable_rule_and_manifest_files(man):
    m, d = man
    mw = m["definitions"]["min_windows_trainable"]
    assert all(c["trainable"] == (c["windows"] >= mw) for c in m["clients"].values())
    assert (d / "dataset_manifest.json").is_file() and (d / "dataset_report.txt").is_file()
    assert len([c for c in m["clients"].values() if c["type"] == "rsu"]) == 8     # 4 RSUs x 2 runs
