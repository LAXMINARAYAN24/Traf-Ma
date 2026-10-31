"""Dataset-generation configs expand to valid, disjoint, well-formed run grids (SUMO is stubbed out)."""
from pathlib import Path

import pytest
import trafma.collect as C

ROOT = Path(__file__).resolve().parent.parent


def expand(monkeypatch, tmp_path, cfg_name):
    seen = []
    monkeypatch.setattr(C, "_run_one", lambda cd: (seen.append(cd) or {"run": cd["run"]["run_id"], "status": "stub"}))
    monkeypatch.setattr(C, "build_dataset_manifest", lambda *a, **k: {})
    monkeypatch.setattr(C, "format_dataset_report", lambda m: "")
    monkeypatch.chdir(tmp_path)
    C.collect(ROOT / "configs" / cfg_name, jobs=1, out=lambda *_: None)
    return seen


def test_original_v1_config_unchanged(monkeypatch, tmp_path):
    assert len(expand(monkeypatch, tmp_path, "data.yaml")) == 24


@pytest.mark.parametrize("name,bcast", [("data_v2.yaml", None), ("data_v2_broadcast.yaml", True)])
def test_v2_grid(monkeypatch, tmp_path, name, bcast):
    runs = expand(monkeypatch, tmp_path, name)
    assert len(runs) == 8 * 20
    ids = [r["run"]["run_id"] for r in runs]
    assert len(set(ids)) == len(ids)
    scen = {i.rsplit("_s", 1)[0] for i in ids}
    assert scen == {"free", "light_moderate", "moderate", "moderate_heavy", "heavy",
                    "asymmetric", "asymmetric_ew", "single_arm"}
    seeds = {r["run"]["seed"] for r in runs}
    assert seeds == set(range(100, 110)) | set(range(200, 205)) | set(range(300, 305))
    assert not seeds & set(range(400, 405))                      # calibration seeds stay reserved for Phase 3
    assert {r.get("agents", {}).get("signal_broadcast") for r in runs} == {bcast}
    by = {r["run"]["run_id"]: r for r in runs}
    ew = by["asymmetric_ew_s100"]["demand"]
    assert ew["profile"] == "asymmetric" and ew["approach_weights"] == {"N": 0.1, "S": 0.1, "E": 0.4, "W": 0.4}
    assert by["single_arm_s300"]["demand"]["profile"] == "custom"
    assert [by[f"{s}_s100"]["run"]["vehicles"] for s in ("free", "light_moderate", "moderate", "moderate_heavy", "heavy")] \
        == [60, 120, 220, 300, 380]


def test_v2_and_broadcast_variants_have_identical_run_grids(monkeypatch, tmp_path):
    a = {(r["run"]["run_id"], r["run"]["vehicles"], r["run"]["seed"]) for r in expand(monkeypatch, tmp_path, "data_v2.yaml")}
    b = {(r["run"]["run_id"], r["run"]["vehicles"], r["run"]["seed"])
         for r in expand(monkeypatch, tmp_path, "data_v2_broadcast.yaml")}
    assert a == b


def test_dev_grid_is_a_seed_subset_of_the_full_grid(monkeypatch, tmp_path):
    full = {r["run"]["run_id"]: r for r in expand(monkeypatch, tmp_path, "data_v2.yaml")}
    dev = {r["run"]["run_id"]: r for r in expand(monkeypatch, tmp_path, "data_v2_dev.yaml")}
    assert len(dev) == 8 * 6 and set(dev) <= set(full)
    strip = lambda d: {k: v for k, v in d.items() if k != "output_dir"}     # only the output folder may differ
    for rid, r in dev.items():                      # identical scenario definition, so identical runs
        assert strip(r["run"]) == strip(full[rid]["run"]) and r["demand"] == full[rid]["demand"]
