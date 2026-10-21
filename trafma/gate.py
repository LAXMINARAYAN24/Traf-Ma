"""Phase-1 gate: machine-checks the plan's gate checklist against a saved run directory."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import APPROACHES
from .config import config_from_dict
from .sensors import VEHICLE_MASKED  # noqa: F401  (masked-field list is the contract)

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "SKIP", "INFO"


def _jl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_manifest(run_dir) -> dict:
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))


def reproduce(run_dir) -> dict:
    """Re-run with the identical config/seed and compare non-wall-clock fingerprints."""
    from .runner import run_simulation
    run_dir = Path(run_dir)
    m = load_manifest(run_dir)
    cfg = config_from_dict(m["config"])
    cfg.run.run_id = m["run_id"] + "_repro"
    cfg.run.output_dir = str(run_dir.parent / "_repro")
    cfg.run.gui = False
    res = run_simulation(cfg, gui=False, quiet=True)
    n = res.manifest
    keys = ["trace_sha256"]
    fkeys = ["net_sha256", "routes_sha256", "topology_sha256"]
    same = all(n[k] == m[k] for k in keys) and all(n["files"][k] == m["files"][k] for k in fkeys)
    return {"identical": same, "trace_sha256": (m["trace_sha256"][:12], n["trace_sha256"][:12]),
            "routes_sha256": (m["files"]["routes_sha256"][:12], n["files"]["routes_sha256"][:12]),
            "repro_dir": str(res.run_dir)}


def run_gate(run_dir, do_reproduce=False, do_pytest=False, tests_dir=None) -> list:
    run_dir = Path(run_dir)
    m = load_manifest(run_dir)
    cfg = config_from_dict(m["config"])
    out = []
    add = lambda name, ok, detail="": out.append((name, ok, detail))

    # 1 finish + clean TraCI close
    ok = m["status"] in ("completed", "sim_timeout", "wall_timeout") and m["traci_closed_cleanly"] is True
    add("Run finished or reported a defined timeout; TraCI closed cleanly", PASS if ok else FAIL,
        f"status={m['status']}, closed_cleanly={m['traci_closed_cleanly']}")

    # 2 four RSUs exist throughout
    steps = int(round(m["end_sim_time_s"] / cfg.run.report_step_s))
    bad = []
    for a in APPROACHES:
        n = sum(1 for _ in _jl(run_dir / "agents" / f"rsu_{a}.jsonl"))
        if n != steps:
            bad.append(f"rsu_{a}:{n}/{steps}")
    ok = m["registry"]["rsus"] == [f"rsu_{a}" for a in APPROACHES] and not bad
    add("Four RSUs (one per road) exist throughout", PASS if ok else FAIL,
        "all four have a record at every report step" if ok else f"gaps: {bad}")

    # 3 registry matches events
    ev = list(_jl(run_dir / "evaluator" / "events.jsonl"))
    dep = [e for e in ev if e["ev"] == "depart"]
    arr = [e for e in ev if e["ev"] == "arrive"]
    ok = (len(dep) == m["vehicles_departed"] == m["registry"]["events_depart"] and
          len(arr) == m["vehicles_arrived"] == m["registry"]["events_arrive"] and
          m["vehicles_active_at_end"] == len(dep) - len(arr) and m["counters"]["registry_mismatch"] == 0 and
          len(m["pending_ids"]) == m["vehicles_scheduled"] - m["vehicles_departed"])
    add("Vehicle registry matches departed/active/arrived events", PASS if ok else FAIL,
        f"scheduled {m['vehicles_scheduled']}, departed {len(dep)}, arrived {len(arr)}, "
        f"active {m['vehicles_active_at_end']}, pending {len(m['pending_ids'])}")

    # 4 ownership persists + out-of-coverage masked
    viol = []
    origin = {e["vid"]: e["origin"] for e in dep}
    for vid, o in origin.items():
        p = run_dir / "agents" / f"{vid}.jsonl"
        if not p.is_file():
            viol.append(f"{vid}: no log")
            continue
        for r in _jl(p):
            v = r["view"]
            if r["parent"] != f"rsu_{o}":
                viol.append(f"{vid}: parent changed")
                break
            if v["scope"] == "incoming_local" and v["edge_id"] != f"{o}_in":
                viol.append(f"{vid}: incoming scope on foreign edge {v['edge_id']}")
                break
            if v["scope"] != "incoming_local" and any(v.get(k) is not None for k in VEHICLE_MASKED):
                viol.append(f"{vid}: unmasked out-of-coverage field at t={r['t']}")
                break
    add("Ownership persists to arrival; out-of-coverage observations masked", PASS if not viol else FAIL,
        f"{len(origin)} vehicles checked" if not viol else "; ".join(viol[:3]))

    # 5 reproducibility
    if do_reproduce:
        rr = reproduce(run_dir)
        add("Identical seed/config/simulator reproduce the non-wall-clock trace", PASS if rr["identical"] else FAIL,
            f"trace {rr['trace_sha256'][0]} vs {rr['trace_sha256'][1]}; routes {rr['routes_sha256'][0]} vs {rr['routes_sha256'][1]}")
    else:
        add("Identical seed/config/simulator reproduce the non-wall-clock trace", SKIP, "pass --reproduce to re-run")

    # 6 vehicle messages reach coordinator through parent, no duplicate counted
    disp = m["reports_disposition"]
    sent = m["counters"]["vehicle_reports_sent"]
    acc = disp.get("coordinator_accepted", 0)
    unresolved = m["reports_in_flight_or_unresolved"]
    accounted = sum(disp.values()) == sent
    no_dup = acc == m["coordinator"]["relayed_accepted"]
    lossless = (cfg.comms.v2i.loss_prob == 0 and cfg.comms.i2c.loss_prob == 0 and cfg.comms.v2i.delay_s == 0
                and cfg.comms.i2c.delay_s == 0 and cfg.agents.v2i_range_m >= 400)
    all_reach = (acc == sent) if lossless else True
    add("Every vehicle message reaches the coordinator via its parent RSU; no duplicate contribution",
        PASS if (accounted and no_dup and all_reach and unresolved == 0 or (accounted and no_dup and not lossless)) else FAIL,
        f"sent {sent}; coordinator-accepted {acc}; fates {disp}; coordinator rejects {m['coordinator']['rejected']}")

    # 7 checks exist (unit tests)
    td = Path(tests_dir) if tests_dir else Path(__file__).resolve().parent.parent / "tests"
    have = td.is_dir() and any(td.glob("test_*.py"))
    if do_pytest and have:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(td)], capture_output=True, text=True)
        add("Checks for locality, empty regions, missing values, stale messages, malformed config pass",
            PASS if r.returncode == 0 else FAIL, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:])
    else:
        add("Checks for locality, empty regions, missing values, stale messages, malformed config pass",
            SKIP if have else FAIL, "pass --pytest to execute" if have else "tests directory missing")

    # 8 artefacts saved
    need = ["manifest.json", "messages.jsonl", "global_snapshots.jsonl", "global_snapshot_final.json",
            "trace.jsonl", "summary.txt", "scenario/net.net.xml", "scenario/routes.rou.xml",
            "scenario/scenario.sumocfg", "evaluator/truth.jsonl"]
    missing = [f for f in need if not (run_dir / f).is_file()]
    missing += [f"agents/{vid}.jsonl" for vid in origin if not (run_dir / "agents" / f"{vid}.jsonl").is_file()]
    add("Manifest, per-agent logs, message trace and global snapshot saved", PASS if not missing else FAIL,
        "all present" if not missing else f"missing {missing[:4]}")

    # extra: safety + routes/approaches (plan P1.2 evidence)
    c = m["counters"]
    ok = c["collisions"] == 0 and c["teleports"] == 0 and m.get("sumo_error_log_bytes") in (0, None)
    add("No unexplained collisions/teleports/SUMO errors", PASS if ok else FAIL,
        f"collisions {c['collisions']}, teleports {c['teleports']}, error log bytes {m.get('sumo_error_log_bytes')}")
    used = sorted(set(origin.values()))
    if m["vehicles_departed"] >= 40 and cfg.demand.profile == "balanced":
        add("All four approaches used", PASS if len(used) == 4 else FAIL, f"origins used: {used}")
    else:
        add("All four approaches used", INFO, f"origins used: {used} (checked only for balanced N>=40)")
    return out


def format_gate(results) -> str:
    L = ["Phase 1 gate"]
    for name, st, detail in results:
        L.append(f"  [{st:4}] {name}\n         {detail}")
    fails = [r for r in results if r[1] == FAIL]
    skips = [r for r in results if r[1] == SKIP]
    L.append(f"\nRESULT: {'FAIL' if fails else 'PASS'}"
             + (f" ({len(skips)} check(s) skipped)" if skips else ""))
    return "\n".join(L)
