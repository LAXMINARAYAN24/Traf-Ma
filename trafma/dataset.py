"""P2.1 - Labels, windows, coverage audit, label histograms and split manifest for generated runs.

Agents' own logs never contain labels. Labels come only from evaluator/truth.jsonl and are joined
here, offline, at (end_t + horizon). A window never crosses a run boundary, and splits are by whole run.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from . import APPROACHES, LABEL_SCHEMA
from .config import config_from_dict
from .gate import _jl, load_manifest

FORBIDDEN_IN_AGENT_LOGS = {"zone_queue_m", "zone_queue_ratio", "zone_queue_count", "congestion_class",
                           "edge_queue_m", "edge_count", "zone_halting", "label", "labels"}


def _hist(classes):
    h = [0, 0, 0]
    for c in classes:
        h[c] += 1
    return h


def _tv(p, q):
    sp, sq = sum(p) or 1, sum(q) or 1
    return 0.5 * sum(abs(a / sp - b / sq) for a, b in zip(p, q))


def _agent_keys(rec) -> set:
    return set(rec.keys()) | set(rec.get("view", {}).keys())


def process_run(run_dir: Path) -> dict:
    """Write labels.jsonl + windows.jsonl for a run and return its per-client stats."""
    m = load_manifest(run_dir)
    cfg = config_from_dict(m["config"])
    H, hist = cfg.labels.horizon_s, cfg.labels.history_len
    dt = cfg.run.report_step_s
    truth = {}
    for r in _jl(run_dir / "evaluator" / "truth.jsonl"):
        truth[round(r["t"], 6)] = r["approaches"]
    times = sorted(truth)
    t_end = times[-1] if times else 0.0

    def label(approach, t_obs):
        t2 = round(t_obs + H, 6)
        if t2 not in truth:
            return None
        a = truth[t2][approach]
        return {"label_t": t2, "queue_m": a["zone_queue_m"], "cls": a["congestion_class"],
                "ratio": a["zone_queue_ratio"]}

    with open(run_dir / "labels.jsonl", "w", encoding="utf-8") as fl:
        for t in times:
            for a in APPROACHES:
                lb = label(a, t)
                if lb:
                    fl.write(json.dumps({"t": t, "approach": a, "horizon_s": H, **lb}, sort_keys=True) + "\n")

    clients, leak = {}, []
    wf = open(run_dir / "windows.jsonl", "w", encoding="utf-8")

    def emit(cid, ctype, approach, end_t, lb):
        wf.write(json.dumps({"client": cid, "type": ctype, "approach": approach, "end_t": end_t, **lb},
                            sort_keys=True) + "\n")
        c = clients[cid]
        c["windows"] += 1
        c["classes"].append(lb["cls"])
        c["queues"].append(lb["queue_m"])

    # RSU clients
    for a in APPROACHES:
        cid = f"{m['run_id']}/rsu_{a}"
        clients[cid] = {"type": "rsu", "approach": a, "obs_steps": 0, "windows": 0, "classes": [], "queues": []}
        recs = list(_jl(run_dir / "agents" / f"rsu_{a}.jsonl"))
        clients[cid]["obs_steps"] = len(recs)
        for rec in recs[:3]:
            leak += sorted(_agent_keys(rec) & FORBIDDEN_IN_AGENT_LOGS)
        ts = [round(r["t"], 6) for r in recs]
        for i in range(hist - 1, len(ts)):
            if abs(ts[i] - ts[i - hist + 1] - dt * (hist - 1)) > 1e-6:
                continue
            lb = label(a, ts[i])
            if lb:
                emit(cid, "rsu", a, ts[i], lb)
    # vehicle clients (one client per vehicle trajectory)
    for p in sorted((run_dir / "agents").glob("veh_*.jsonl")):
        vid = p.stem
        recs = list(_jl(p))
        if not recs:
            continue
        a = recs[0]["parent"].split("_", 1)[1]
        cid = f"{m['run_id']}/{vid}"
        clients[cid] = {"type": "vehicle", "approach": a, "obs_steps": len(recs), "windows": 0,
                        "classes": [], "queues": []}
        for rec in recs[:2]:
            leak += sorted(_agent_keys(rec) & FORBIDDEN_IN_AGENT_LOGS)
        ok = [r["view"]["scope"] == "incoming_local" for r in recs]
        ts = [round(r["t"], 6) for r in recs]
        run_len = 0
        for i in range(len(recs)):
            run_len = run_len + 1 if ok[i] and (i == 0 or abs(ts[i] - ts[i - 1] - dt) < 1e-6) else (1 if ok[i] else 0)
            if run_len >= hist:
                lb = label(a, ts[i])
                if lb:
                    emit(cid, "vehicle", a, ts[i], lb)
    wf.close()

    out = {}
    for cid, c in clients.items():
        out[cid] = {"type": c["type"], "approach": c["approach"], "obs_steps": c["obs_steps"],
                    "windows": c["windows"], "label_hist": _hist(c["classes"]),
                    "mean_label_queue_m": round(sum(c["queues"]) / len(c["queues"]), 3) if c["queues"] else None,
                    "trainable": c["windows"] >= cfg.labels.min_windows}
    steps = int(round(m["end_sim_time_s"] / dt))
    audit = {"run_id": m["run_id"], "status": m["status"], "steps": steps,
             "rsu_records": {a: clients[f"{m['run_id']}/rsu_{a}"]["obs_steps"] for a in APPROACHES},
             "rsu_complete": all(clients[f"{m['run_id']}/rsu_{a}"]["obs_steps"] == steps for a in APPROACHES),
             "vehicles_scheduled": m["vehicles_scheduled"], "vehicles_departed": m["vehicles_departed"],
             "vehicles_arrived": m["vehicles_arrived"], "vehicles_unfinished": m["vehicles_active_at_end"],
             "vehicles_never_inserted": len(m["pending_ids"]),
             "collisions": m["counters"]["collisions"], "teleports": m["counters"]["teleports"],
             "label_leak_keys_in_agent_logs": sorted(set(leak))}
    return {"clients": out, "audit": audit, "label_end_t": t_end}


def build_dataset_manifest(ds_dir: Path, run_index: list, base_cfg: dict, splits: dict) -> dict:
    """run_index: [{"run": run_id, "scenario": name, "seed": s, "split": split}]"""
    ds_dir = Path(ds_dir)
    cfg = config_from_dict(base_cfg)
    per_run, all_clients = {}, {}
    for r in run_index:
        res = process_run(ds_dir / "runs" / r["run"])
        per_run[r["run"]] = {**r, **res["audit"]}
        for cid, c in res["clients"].items():
            all_clients[cid] = {**c, "run": r["run"], "scenario": r["scenario"], "split": r["split"]}

    # ---- integrity checks
    by_split = defaultdict(set)
    for r in run_index:
        by_split[r["split"]].add(r["run"])
    splits_list = list(by_split)
    overlap = [(a, b, sorted(by_split[a] & by_split[b])) for i, a in enumerate(splits_list)
               for b in splits_list[i + 1:] if by_split[a] & by_split[b]]
    cross = 0
    H = cfg.labels.horizon_s
    for r in run_index:
        for w in _jl(ds_dir / "runs" / r["run"] / "windows.jsonl"):
            if not w["client"].startswith(r["run"] + "/") or abs(w["label_t"] - w["end_t"] - H) > 1e-6 or w["label_t"] <= w["end_t"]:
                cross += 1
    leaks = sorted({k for r in per_run.values() for k in r["label_leak_keys_in_agent_logs"]})

    # ---- aggregates per split x client type
    agg = {}
    for sp in sorted(by_split):
        for ct in ("rsu", "vehicle"):
            cl = [c for c in all_clients.values() if c["split"] == sp and c["type"] == ct]
            tr = [c for c in cl if c["trainable"]]
            h = [sum(c["label_hist"][k] for c in cl) for k in range(3)]
            agg[f"{sp}/{ct}"] = {"clients": len(cl), "trainable_clients": len(tr),
                                 "windows": sum(c["windows"] for c in cl), "label_hist": h}
    # ---- measured non-IID skew among trainable training clients (total-variation distance to pooled)
    skew = {}
    for ct in ("rsu", "vehicle"):
        cl = [c for c in all_clients.values() if c["split"] == "train" and c["type"] == ct and c["trainable"]]
        if cl:
            pooled = [sum(c["label_hist"][k] for c in cl) for k in range(3)]
            tv = [_tv(c["label_hist"], pooled) for c in cl]
            skew[ct] = {"n_clients": len(cl), "mean_tv_to_pooled": round(sum(tv) / len(tv), 4),
                        "max_tv_to_pooled": round(max(tv), 4), "pooled_label_hist": pooled}
    manifest = {
        "label_schema": LABEL_SCHEMA,
        "definitions": {
            "horizon_s": H, "history_len": cfg.labels.history_len, "min_windows_trainable": cfg.labels.min_windows,
            "report_step_s": cfg.run.report_step_s, "target_zone_m": cfg.agents.rsu_zone_m,
            "congestion_rule": f"queue_zone/zone_len < {cfg.labels.congestion_thresholds[0]} -> 0 free; "
                               f"< {cfg.labels.congestion_thresholds[1]} -> 1 moderate; else 2 heavy",
            "queue_definition": f"contiguous halted chain (speed<={cfg.agents.halting_speed_mps} m/s) starting within "
                                f"{cfg.agents.queue_start_m} m of the stop line, bumper gaps <= {cfg.agents.queue_gap_m} m, "
                                "measured to rear bumper, capped at the target zone",
            "capacity_veh_in_zone": round(cfg.agents.rsu_zone_m / (cfg.vehicle.length_m + cfg.vehicle.min_gap_m), 2),
            "window_rule_rsu": "last history_len consecutive report steps (all present)",
            "window_rule_vehicle": "last history_len consecutive report steps, ALL with scope=incoming_local",
            "missing_value_policy": "None = unavailable (mask); never zero-filled; empty RSU zone has no mean speed/wait",
            "label_target": "evaluator zone queue/class of the client's own approach at end_t + horizon",
        },
        "runs": per_run, "splits": {k: sorted(v) for k, v in by_split.items()},
        "aggregates": agg, "noniid_skew_train": skew,
        "integrity": {"split_run_overlap": overlap, "windows_violating_run_or_horizon_rule": cross,
                      "evaluator_keys_found_in_agent_logs": leaks,
                      "all_runs_have_complete_rsu_logs": all(r["rsu_complete"] for r in per_run.values()),
                      "total_collisions": sum(r["collisions"] for r in per_run.values()),
                      "total_teleports": sum(r["teleports"] for r in per_run.values())},
        "clients": all_clients,
    }
    (ds_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def format_dataset_report(man: dict) -> str:
    L = ["=== dataset report ===", "congestion rule : " + man["definitions"]["congestion_rule"],
         f"horizon {man['definitions']['horizon_s']} s, history {man['definitions']['history_len']} steps, "
         f"trainable if >= {man['definitions']['min_windows_trainable']} windows", ""]
    L.append(f"{'run':24} {'split':5} {'N':>4} {'dep':>4} {'arr':>4} {'unf':>3} {'coll':>4} rsuLogs")
    for r, v in sorted(man["runs"].items()):
        L.append(f"{r:24} {v['split']:5} {v['vehicles_scheduled']:4d} {v['vehicles_departed']:4d} "
                 f"{v['vehicles_arrived']:4d} {v['vehicles_unfinished']:3d} {v['collisions']:4d} "
                 f"{'ok' if v['rsu_complete'] else 'GAPS'}")
    L.append("\nsplit/client-type   clients trainable windows   label hist [free, moderate, heavy]")
    for k, a in man["aggregates"].items():
        L.append(f"{k:19} {a['clients']:7d} {a['trainable_clients']:9d} {a['windows']:7d}   {a['label_hist']}")
    L.append("\nmeasured non-IID skew (train, trainable clients; TV distance of client label dist. to pooled):")
    for ct, s in man["noniid_skew_train"].items():
        L.append(f"  {ct:8} n={s['n_clients']} mean TV {s['mean_tv_to_pooled']}, max {s['max_tv_to_pooled']}, pooled {s['pooled_label_hist']}")
    i = man["integrity"]
    L.append(f"\nintegrity: split overlap={i['split_run_overlap'] or 'none'}; window rule violations="
             f"{i['windows_violating_run_or_horizon_rule']}; evaluator keys in agent logs={i['evaluator_keys_found_in_agent_logs'] or 'none'}; "
             f"complete RSU logs={i['all_runs_have_complete_rsu_logs']}; collisions={i['total_collisions']}, teleports={i['total_teleports']}")
    return "\n".join(L)
