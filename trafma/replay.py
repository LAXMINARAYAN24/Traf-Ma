"""Replay from saved logs (no SUMO needed) and optional deterministic re-run verification."""
from __future__ import annotations

import json
from pathlib import Path

from . import APPROACHES
from .gate import load_manifest, reproduce, _jl


def replay_run(run_dir, t_from=0.0, t_to=None, every=10.0, verify=False, out=print):
    run_dir = Path(run_dir)
    m = load_manifest(run_dir)
    out(f"Replay of {m['run_id']}  (seed {m['seed']}, N={m['vehicles']}, status {m['status']})")
    out(f"{'t':>6} | " + " | ".join(f"{a}: sig q_true q_rsu n  rep" for a in APPROACHES) + " | active")
    truth = {r["t"]: r["approaches"] for r in _jl(run_dir / "evaluator" / "truth.jsonl")}
    ev = {}
    active = 0
    for e in _jl(run_dir / "evaluator" / "events.jsonl"):
        ev.setdefault(e["t"], []).append(e)
    act_at = {}
    for t in sorted(set(truth) | set(ev)):
        for e in ev.get(t, []):
            active += 1 if e["ev"] == "depart" else -1 if e["ev"] == "arrive" else 0
        act_at[t] = active
    for g in _jl(run_dir / "global_snapshots.jsonl"):
        t = g["time"]
        if t < t_from or (t_to is not None and t > t_to) or (t % every) > 1e-9:
            continue
        cells = []
        for a in APPROACHES:
            ap = g["approaches"][a]
            d = ap["direct"]["observation"] if ap["direct"] else None
            cells.append(f"{a}: {d['signal_state'] if d else '?':>3} {truth[t][a]['zone_queue_m']:6.1f} "
                         f"{d['queue_length_m'] if d else float('nan'):5.1f} {d['vehicle_count'] if d else -1:2d} "
                         f"{ap['relayed_child']['reporting_vehicles']:3d}")
        out(f"{t:6.0f} | " + " | ".join(cells) + f" | {act_at.get(t, 0):3d}")
    out("(sig = RSU-observed signal; q_true = evaluator truth; q_rsu = RSU-observed queue; "
        "n = vehicles in RSU zone; rep = vehicles reporting via this RSU)")
    if verify:
        r = reproduce(run_dir)
        out(f"Deterministic re-run: {'IDENTICAL' if r['identical'] else 'DIFFERENT'} "
            f"(trace {r['trace_sha256'][0]} vs {r['trace_sha256'][1]})")
        return r
    return None
