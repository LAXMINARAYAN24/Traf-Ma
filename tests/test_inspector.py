"""`trafma inspect`: one agent's local view + the path its reports took, from saved logs only."""
import json

import pytest
from conftest import needs_sumo
from trafma.cli import main
from trafma.inspector import InspectError, inspect_rsu, inspect_vehicle


def _w(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def vrec(t, scope="incoming_local", dist=100.0):
    v = {"scope": scope, "edge_id": "N_in", "speed_mps": 5.0, "accel_mps2": 0.0, "lane_pos_m": 100.0,
         "waiting_time_s": 0.0, "dist_to_stop_line_m": dist if scope == "incoming_local" else None,
         "signal_state": None, "neighbor_count": 0 if scope == "incoming_local" else None,
         "neighbor_halting_count": 0 if scope == "incoming_local" else None, "leader_gap_m": None, "unavailable": []}
    return {"kind": "observation", "msg_id": f"veh_0001#{int(t) - 1}", "parent": "rsu_N", "t": float(t), "view": v}


def ev(t, e, frm, to, link, msg, reason=None):
    r = {"ev": e, "from": frm, "to": to, "link": link, "msg": msg, "t": float(t), "type": None}
    if reason:
        r["reason"] = reason
    return r


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "r1"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "run_id": "r1", "seed": 7, "vehicles": 3, "config": {"run": {"message_trace": "full"}},
        "rsu_child_stats": {"rsu_N": {"accepted": 2, "duplicate": 0}}}), encoding="utf-8")
    _w(d / "agents" / "veh_0001.jsonl", [vrec(1), vrec(2), vrec(3), vrec(4, "in_junction")])
    # OLD-format RSU records (no mean_accel / mean_dist) must still render
    _w(d / "agents" / "rsu_N.jsonl", [{"kind": "observation", "t": 10.0, "detected": ["veh_0001"], "msg_id": "rsu_N#9",
          "view": {"signal_state": "G", "signal_remaining_s": 12.0, "vehicle_count": 1, "halting_count": 0,
                   "queue_length_m": 0.0, "mean_speed_mps": 4.0}}])
    _w(d / "messages.jsonl", [
        ev(1, "send", "veh_0001", "rsu_N", "v2i", "veh_0001#0"), ev(1, "deliver", "veh_0001", "rsu_N", "v2i", "veh_0001#0"),
        ev(1, "accept", "rsu_N", "coordinator", "i2c", "veh_0001#0", "relayed_child"),
        ev(2, "send", "veh_0001", "rsu_N", "v2i", "veh_0001#1"), ev(2, "lost", "veh_0001", "rsu_N", "v2i", "veh_0001#1"),
        ev(3, "send", "veh_0001", "rsu_N", "v2i", "veh_0001#2"), ev(3, "deliver", "veh_0001", "rsu_N", "v2i", "veh_0001#2"),
        ev(3, "coord_reject", "rsu_N", "coordinator", "i2c", "veh_0001#2", "stale"),
        ev(1, "send", "veh_0002", "rsu_E", "v2i", "veh_0002#0")])          # another vehicle: must be ignored
    return d


def test_vehicle_view_and_message_paths(run_dir):
    lines = []
    r = inspect_vehicle(run_dir, "veh_0001", every=1, out=lines.append)
    text = "\n".join(lines)
    assert r["parent"] == "rsu_N" and r["reports"] == 4 and r["paths"] == 3
    assert r["outcomes"] == {"accepted by coordinator": 1, "lost on link": 1, "rejected by coordinator": 1}
    assert "ACCEPTED by coordinator via rsu_N" in text and "LOST" in text and "COORD_REJECT" in text and "(stale)" in text
    assert "in_junction" in text and "veh_0002" not in text
    assert "-" in text                                           # unavailable shown as '-', never 0.0 for masked fields


def test_missing_trace_is_explained_not_crashed(run_dir):
    (run_dir / "messages.jsonl").unlink()
    lines = []
    r = inspect_vehicle(run_dir, "veh_0001", out=lines.append)
    assert r["paths"] == 0 and any("no transport trace" in l for l in lines)


def test_unknown_agent_errors(run_dir, tmp_path):
    with pytest.raises(InspectError, match="no log for vehicle"):
        inspect_vehicle(run_dir, "veh_9999", out=lambda *_: None)
    with pytest.raises(InspectError, match="RSU must be one of"):
        inspect_rsu(run_dir, "X", out=lambda *_: None)
    with pytest.raises(InspectError, match="not a run directory"):
        inspect_vehicle(tmp_path / "nope", "veh_0001", out=lambda *_: None)


def test_rsu_view_handles_old_format_logs(run_dir):
    lines = []
    r = inspect_rsu(run_dir, "n", every=10, out=lines.append)
    assert r["observations"] == 1 and any("veh_0001" in l for l in lines)


def test_cli_exit_codes(run_dir, capsys):
    assert main(["inspect", "--run", str(run_dir), "--vehicle", "veh_0001", "--every", "1"]) == 0
    assert main(["inspect", "--run", str(run_dir), "--rsu", "N"]) == 0
    assert main(["inspect", "--run", str(run_dir), "--vehicle", "veh_9999"]) == 2
    assert "inspect error" in capsys.readouterr().err
    with pytest.raises(SystemExit):                              # --vehicle and --rsu are mutually exclusive / one required
        main(["inspect", "--run", str(run_dir)])


@needs_sumo
def test_inspect_real_run_every_report_accepted(tmp_path):
    from trafma.config import load_config
    from trafma.runner import run_simulation
    res = run_simulation(load_config(overrides={"run": {"output_dir": str(tmp_path), "run_id": "real",
                                                        "duration_s": 200, "vehicles": 10}}), quiet=True)
    vid = sorted(p.stem for p in (res.run_dir / "agents").glob("veh_*.jsonl"))[0]
    r = inspect_vehicle(res.run_dir, vid, out=lambda *_: None)
    assert r["paths"] == r["reports"] > 0 and r["outcomes"] == {"accepted by coordinator": r["reports"]}
    assert inspect_rsu(res.run_dir, "N", out=lambda *_: None)["observations"] == 200
