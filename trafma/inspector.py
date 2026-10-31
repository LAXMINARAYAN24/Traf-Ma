"""Inspect ONE agent of a saved run, from saved logs only (no SUMO needed).

    python -m trafma inspect --run runs/<id> --vehicle veh_0005     # its local view + the path its reports took
    python -m trafma inspect --run runs/<id> --rsu E                # one RSU's view, signal and detected vehicles

This is the Phase 1 demo story made executable: "inspect each agent's local view and trace vehicle
updates through the RSU to the global coordinator". It reads only `agents/<id>.jsonl` (what that agent
itself saw) and `messages.jsonl` (the transport trace). It never reads evaluator truth.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from . import APPROACHES


class InspectError(ValueError):
    """Raised when the requested agent or log is not available."""


def _jl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _fmt(x, nd=1, none="-"):
    if x is None:
        return none
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _on_grid(t: float, every: float, t_from: float, t_to) -> bool:
    if t < t_from or (t_to is not None and t > t_to):
        return False
    if every <= 0:
        return True
    r = t % every
    return r < 1e-6 or every - r < 1e-6


def _manifest(run_dir: Path) -> dict:
    p = run_dir / "manifest.json"
    if not p.is_file():
        raise InspectError(f"not a run directory (no manifest.json): {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _message_paths(run_dir: Path, vid: str):
    """msg_id -> [events] for every transport event of this vehicle's reports. Returns (paths, trace_level)."""
    paths = defaultdict(list)
    p = run_dir / "messages.jsonl"
    if not p.is_file():
        return paths
    prefix = vid + "#"
    for r in _jl(p):
        if str(r.get("msg", "")).startswith(prefix):
            paths[r["msg"]].append(r)
    return paths


def _describe_event(e: dict) -> str:
    ev, frm, to, link = e["ev"], e["from"], e["to"], e.get("link", "")
    why = f" ({e['reason']})" if e.get("reason") else ""
    if ev == "send":
        return f"sent {frm}->{to} [{link}]"
    if ev == "deliver":
        return f"delivered to {to} [{link}]"
    if ev == "accept":
        return f"ACCEPTED by {to} via {frm} [{link}]{why}"
    if ev == "relay":
        return f"relayed {frm}->{to} [{link}]"
    return f"{ev.upper()} {frm}->{to} [{link}]{why}"


def inspect_vehicle(run_dir, vid: str, every: float = 10.0, t_from: float = 0.0, t_to=None,
                    max_paths: int = 5, out=print) -> dict:
    run_dir = Path(run_dir)
    m = _manifest(run_dir)
    log = run_dir / "agents" / f"{vid}.jsonl"
    if not log.is_file():
        known = sorted(p.stem for p in (run_dir / "agents").glob("veh_*.jsonl"))
        raise InspectError(f"no log for vehicle '{vid}' in {run_dir.name}. "
                           f"{len(known)} vehicle logs exist, e.g. {known[:5]}")
    recs = list(_jl(log))
    if not recs:
        raise InspectError(f"vehicle log {log.name} is empty")
    parent = recs[0]["parent"]
    scopes = Counter(r["view"]["scope"] for r in recs)
    out(f"=== vehicle {vid}  (run {m['run_id']}, seed {m['seed']}, N={m['vehicles']}) ===")
    out(f"parent RSU          : {parent}  (ownership is fixed at departure and persists until arrival)")
    out(f"reports             : {len(recs)} (t={recs[0]['t']:.0f} .. {recs[-1]['t']:.0f} s)")
    out("scope over its life : " + ", ".join(f"{k} x{v}" for k, v in sorted(scopes.items())))
    out("")
    out("What this vehicle could see (only its own view; '-' = unavailable, never zero)")
    out(f"{'t':>6} {'scope':<15} {'dist_m':>7} {'v_mps':>6} {'sig':>4} {'sig_rem':>7} {'nbrs':>4} {'halt':>4} {'lead_gap':>8}")
    shown = 0
    for r in recs:
        if not _on_grid(r["t"], every, t_from, t_to):
            continue
        v = r["view"]
        out(f"{r['t']:6.0f} {v['scope']:<15} {_fmt(v.get('dist_to_stop_line_m')):>7} {_fmt(v['speed_mps']):>6} "
            f"{_fmt(v.get('signal_state')):>4} {_fmt(v.get('signal_remaining_s')):>7} "
            f"{_fmt(v.get('neighbor_count')):>4} {_fmt(v.get('neighbor_halting_count')):>4} "
            f"{_fmt(v.get('leader_gap_m')):>8}")
        shown += 1
    if not shown:
        out("  (no records on that time grid; try --every 1)")

    paths = _message_paths(run_dir, vid)
    out("")
    if not paths:
        out(f"Message path: no transport trace for this vehicle (run was saved with message_trace="
            f"'{m['config']['run'].get('message_trace')}'; only 'full' keeps successful sends).")
        return {"vehicle": vid, "parent": parent, "reports": len(recs), "paths": 0}
    final = Counter()
    for evs in paths.values():
        kinds = {e["ev"] for e in evs}
        if "accept" in kinds:
            final["accepted by coordinator"] += 1
        elif "coord_reject" in kinds:
            final["rejected by coordinator"] += 1
        elif "rsu_reject" in kinds:
            final["rejected by RSU"] += 1
        elif "lost" in kinds:
            final["lost on link"] += 1
        elif "dropped_range" in kinds:
            final["dropped (out of V2I range)"] += 1
        else:
            final["unresolved at end of run"] += 1
    out(f"Message path vehicle -> {parent} -> coordinator  ({len(paths)} reports; outcome: "
        + ", ".join(f"{k} {v}" for k, v in final.items()) + ")")
    for mid in sorted(paths, key=lambda s: int(s.split("#")[1]))[:max_paths]:
        out(f"  {mid}")
        for e in paths[mid]:
            out(f"    t={e['t']:6.1f}  {_describe_event(e)}")
    if len(paths) > max_paths:
        out(f"  ... {len(paths) - max_paths} more (use --paths N)")
    return {"vehicle": vid, "parent": parent, "reports": len(recs), "paths": len(paths), "outcomes": dict(final)}


def inspect_rsu(run_dir, approach: str, every: float = 10.0, t_from: float = 0.0, t_to=None, out=print) -> dict:
    run_dir = Path(run_dir)
    m = _manifest(run_dir)
    approach = approach.upper()
    if approach not in APPROACHES:
        raise InspectError(f"RSU must be one of {list(APPROACHES)}, got {approach!r}")
    log = run_dir / "agents" / f"rsu_{approach}.jsonl"
    if not log.is_file():
        raise InspectError(f"no log for rsu_{approach} in {run_dir.name}")
    recs = list(_jl(log))
    stats = (m.get("rsu_child_stats") or {}).get(f"rsu_{approach}", {})
    out(f"=== rsu_{approach}  (run {m['run_id']}, seed {m['seed']}, N={m['vehicles']}) ===")
    out(f"observations        : {len(recs)} (one per report step, whole run)")
    if stats:
        out("children handled    : " + ", ".join(f"{k} {v}" for k, v in stats.items()))
    out("")
    out("What this RSU sensed on its own road (zone only; '-' = unavailable, never zero)")
    out(f"{'t':>6} {'sig':>3} {'rem_s':>5} {'n':>3} {'halt':>4} {'queue_m':>7} {'v_mps':>6} {'acc':>6} {'dist_m':>7}  detected")
    shown = 0
    for r in recs:
        if not _on_grid(r["t"], every, t_from, t_to):
            continue
        v = r["view"]
        det = r.get("detected", [])
        out(f"{r['t']:6.0f} {_fmt(v['signal_state']):>3} {_fmt(v['signal_remaining_s'], 0):>5} {v['vehicle_count']:>3} "
            f"{v['halting_count']:>4} {_fmt(v['queue_length_m']):>7} {_fmt(v['mean_speed_mps']):>6} "
            f"{_fmt(v.get('mean_accel_mps2'), 2):>6} {_fmt(v.get('mean_dist_to_stop_m')):>7}  "
            f"{','.join(det[:4])}{' ...' if len(det) > 4 else ''}")
        shown += 1
    if not shown:
        out("  (no records on that time grid; try --every 1)")
    return {"rsu": f"rsu_{approach}", "observations": len(recs)}
