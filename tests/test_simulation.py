import json
from pathlib import Path

import pytest
from conftest import needs_sumo
from trafma.config import ConfigError, config_from_dict, load_config
from trafma.gate import load_manifest, run_gate

pytestmark = needs_sumo


def run(tmp_path, rid, quiet=True, **kw):
    from trafma.runner import run_simulation
    ov = {"run": {"output_dir": str(tmp_path), "run_id": rid, "duration_s": kw.pop("duration_s", 300),
                  "vehicles": kw.pop("vehicles", 20)}}
    for k, v in kw.items():
        ov.setdefault(k.split("__")[0], {})[k.split("__")[1]] = v if k.split("__")[0] != "run" else v
    return run_simulation(load_config(overrides=ov), quiet=quiet)


def fails(res):
    return [r for r in res if r[1] == "FAIL"]


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("runs")
    return d, run(d, "base", vehicles=40, duration_s=400)


def test_clean_run_passes_phase1_gate(base):
    d, res = base
    assert res.status == "completed" and res.manifest["traci_closed_cleanly"] is True
    g = run_gate(res.run_dir)
    assert not fails(g), fails(g)
    c = res.manifest["counters"]
    assert c["collisions"] == c["teleports"] == c["emergency_stops"] == 0


def test_four_rsus_all_reporting_and_all_vehicles_relayed(base):
    m = base[1].manifest
    assert m["registry"]["rsus"] == ["rsu_N", "rsu_E", "rsu_S", "rsu_W"]
    assert m["coordinator"]["relayed_accepted"] == m["counters"]["vehicle_reports_sent"] > 0
    assert m["coordinator"]["direct_accepted"] == m["counters"]["rsu_summaries_sent"] == 4 * 400


def test_every_vehicle_traced_vehicle_to_rsu_to_coordinator(base):
    rd = base[1].run_dir
    sends = relays = accepts = 0
    for line in open(rd / "messages.jsonl"):
        r = json.loads(line)
        if r["link"] == "v2i" and r["ev"] == "send":
            sends += 1
            assert r["to"] == "rsu_" + r["to"][-1] and r["from"].startswith("veh_")
        if r["link"] == "i2c" and r["ev"] == "relay":
            relays += 1
    assert sends > 0 and relays > 0


def test_ownership_parent_matches_origin_road(base):
    rd = base[1].run_dir
    for ev in (json.loads(l) for l in open(rd / "evaluator" / "events.jsonl")):
        if ev["ev"] == "depart":
            first = json.loads(open(rd / "agents" / f"{ev['vid']}.jsonl").readline())
            assert first["parent"] == f"rsu_{ev['origin']}"
            assert first["view"]["edge_id"] == f"{ev['origin']}_in"


def test_rsu_sees_only_its_own_road_in_logs(base):
    rd = base[1].run_dir
    origin = {}
    for l in open(rd / "evaluator" / "events.jsonl"):
        e = json.loads(l)
        if e["ev"] == "depart":
            origin[e["vid"]] = e["origin"]
    for a in "NESW":
        for l in open(rd / "agents" / f"rsu_{a}.jsonl"):
            r = json.loads(l)
            assert all(origin[v] == a for v in r["detected"])


def test_agent_logs_hold_no_evaluator_truth(base):
    rd = base[1].run_dir
    for p in (rd / "agents").glob("*.jsonl"):
        rec = json.loads(p.read_text().splitlines()[0])
        assert not ({"zone_queue_m", "congestion_class", "edge_queue_m", "label"} & set(rec["view"]))


def test_same_seed_reproduces_trace(base, tmp_path):
    again = run(tmp_path, "again", vehicles=40, duration_s=400)
    a, b = base[1].manifest, again.manifest
    assert a["trace_sha256"] == b["trace_sha256"] and a["files"]["routes_sha256"] == b["files"]["routes_sha256"]
    assert a["files"]["net_sha256"] == b["files"]["net_sha256"]


def test_different_seed_changes_trace(base, tmp_path):
    other = run(tmp_path, "other", vehicles=40, duration_s=400, run__seed=1)
    assert other.manifest["trace_sha256"] != base[1].manifest["trace_sha256"]


def test_different_n_changes_population(tmp_path):
    a, b = run(tmp_path, "n10", vehicles=10), run(tmp_path, "n30", vehicles=30)
    assert (a.manifest["vehicles_scheduled"], b.manifest["vehicles_scheduled"]) == (10, 30)
    assert b.manifest["vehicles_departed"] == 30


@pytest.mark.parametrize("n", [0, 1])
def test_edge_n_runs(tmp_path, n):
    res = run(tmp_path, f"n{n}", vehicles=n, duration_s=200)
    assert res.status == "completed" and res.manifest["vehicles_departed"] == n
    assert res.manifest["counters"]["rsu_summaries_sent"] == 4 * 200            # RSUs exist throughout
    assert not fails(run_gate(res.run_dir))


def test_n100_runs_and_reports_unfinished_honestly(tmp_path):
    res = run(tmp_path, "n100", vehicles=100, duration_s=600)
    m = res.manifest
    assert m["vehicles_departed"] + len(m["pending_ids"]) == 100
    assert m["vehicles_arrived"] + m["vehicles_active_at_end"] == m["vehicles_departed"]
    if m["vehicles_active_at_end"]:
        assert res.status == "sim_timeout" and m["unfinished_ids"]
    assert not fails(run_gate(res.run_dir))


def test_invalid_n_rejected_before_any_sumo_start(tmp_path):
    from trafma.runner import run_simulation
    with pytest.raises(ConfigError):
        run_simulation(load_config(overrides={"run": {"vehicles": -5, "output_dir": str(tmp_path)}}))
    assert not list(Path(tmp_path).iterdir())


def test_existing_run_dir_not_overwritten(base):
    from trafma.runner import run_simulation
    d = base[0]
    with pytest.raises(ConfigError, match="already exists"):
        run_simulation(load_config(overrides={"run": {"output_dir": str(d), "run_id": "base"}}), quiet=True)


def test_stop_when_empty(tmp_path):
    res = run(tmp_path, "empty", vehicles=5, duration_s=600, run__stop_when_empty=True)
    assert res.status == "completed" and res.manifest["end_sim_time_s"] < 600


def test_lossy_delayed_duplicating_links_keep_accounting_exact(tmp_path):
    from trafma.runner import run_simulation
    cfg = load_config(overrides={"run": {"output_dir": str(tmp_path), "run_id": "lossy", "vehicles": 30,
                                         "duration_s": 400},
                                 "comms": {"v2i": {"loss_prob": 0.2, "duplicate_prob": 0.3, "delay_s": 1.0},
                                           "i2c": {"loss_prob": 0.1, "duplicate_prob": 0.3, "delay_s": 1.0}}})
    m = run_simulation(cfg, quiet=True).manifest
    d, c = m["reports_disposition"], m["counters"]
    assert sum(d.values()) == c["vehicle_reports_sent"]                          # every report has one fate
    assert d.get("coordinator_accepted", 0) == m["coordinator"]["relayed_accepted"]  # duplicates never double-counted
    assert m["coordinator"]["rejected"]["duplicate"] > 0 and c["v2i_lost"] > 0
    assert d.get("coordinator_accepted", 0) < c["vehicle_reports_sent"]
    assert not fails(run_gate(Path(m and Path(tmp_path) / "lossy")))


def test_v2i_range_limit_drops_are_traced(tmp_path):
    from trafma.runner import run_simulation
    cfg = load_config(overrides={"run": {"output_dir": str(tmp_path), "run_id": "range", "vehicles": 10,
                                         "duration_s": 300}, "agents": {"v2i_range_m": 120}})
    m = run_simulation(cfg, quiet=True).manifest
    assert m["counters"]["v2i_dropped_range"] > 0
    assert m["reports_disposition"]["dropped_range"] == m["counters"]["v2i_dropped_range"]


def test_wall_timeout_reported(tmp_path):
    from trafma.runner import run_simulation
    cfg = load_config(overrides={"run": {"output_dir": str(tmp_path), "run_id": "wall", "vehicles": 20,
                                         "duration_s": 600, "wall_timeout_s": 0.2}})
    res = run_simulation(cfg, quiet=True)
    assert res.status == "wall_timeout" and res.manifest["traci_closed_cleanly"] is True


def test_signal_cycle_visible_to_rsus(base):
    seen = {a: set() for a in "NESW"}
    for a in "NESW":
        for l in open(base[1].run_dir / "agents" / f"rsu_{a}.jsonl"):
            seen[a].add(json.loads(l)["view"]["signal_state"])
    assert all({"G", "y", "r"} <= s for s in seen.values())


def test_no_two_rsus_green_at_once(base):
    per_t = {}
    for a in "NESW":
        for l in open(base[1].run_dir / "agents" / f"rsu_{a}.jsonl"):
            r = json.loads(l)
            per_t.setdefault(r["t"], []).append(r["view"]["signal_state"])
    assert all(s.count("G") <= 1 for s in per_t.values())
