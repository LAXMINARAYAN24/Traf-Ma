"""One simulation run: SUMO <-> TraCI <-> agents <-> channels <-> RSUs <-> coordinator, with full logging."""
from __future__ import annotations

import datetime as dt
import platform
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import APPROACHES, __version__
from .agents import AgentRegistry, COORDINATOR_ID
from .channel import Channel
from .config import Config, ConfigError
from .coordinator import Coordinator
from .demand import generate_demand, write_routes
from .evaluator import world_truth
from .messages import VEHICLE_REPORT, RSU_SUMMARY
from .netgen import Topology, build_network, write_sumocfg, in_edge, sha256_file
from .runlog import RunLogger, sha256_path
from .sensors import rsu_view, vehicle_view
from .world import TraciWorld, SumoStartError

ACCEPTED = {"coordinator_accepted"}


@dataclass
class RunResult:
    run_dir: Path
    status: str
    manifest: dict


def _make_run_id(cfg: Config) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return cfg.run.run_id or f"p1_n{cfg.run.vehicles}_s{cfg.run.seed}_{stamp}"


def _versions(world: Optional[TraciWorld]) -> dict:
    try:
        import traci
        tv = getattr(traci, "__version__", "unknown")
    except Exception:
        tv = "missing"
    try:
        import sumolib
        sv = getattr(sumolib, "__version__", "unknown")
    except Exception:
        sv = "missing"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "trafma": __version__,
            "traci": tv, "sumolib": sv, "sumo": getattr(world, "sumo_version", "unknown")}


def _gui_color(topo, v, parent, cfg) -> tuple:
    g = topo.approaches[parent]
    if v.edge_id == g.in_edge:
        d = g.in_len - v.lane_pos
        if v.speed <= cfg.agents.halting_speed_mps:
            return (230, 40, 40, 255)          # halted
        return (30, 190, 80, 255) if d <= cfg.agents.rsu_zone_m else (240, 200, 0, 255)
    return (140, 140, 140, 255)                # junction / outgoing


def _parse_tripinfo(p: Path) -> dict:
    if not p.is_file():
        return {}
    try:
        rows = [t.attrib for t in ET.parse(p).getroot().iter("tripinfo")]
    except ET.ParseError:
        return {}
    if not rows:
        return {"count": 0}
    f = lambda k: [float(r[k]) for r in rows if k in r]
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {"count": len(rows), "mean_duration_s": avg(f("duration")), "mean_waiting_s": avg(f("waitingTime")),
            "mean_time_loss_s": avg(f("timeLoss")), "max_depart_delay_s": max(f("departDelay"), default=None)}


def run_simulation(cfg: Config, gui: Optional[bool] = None, quiet: bool = False) -> RunResult:
    cfg.validate()
    gui = cfg.run.gui if gui is None else gui
    run_id = _make_run_id(cfg)
    run_dir = Path(cfg.run.output_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ConfigError(f"run directory already exists and is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))

    # ------------------------------------------------ scenario files (P1.2, P1.3)
    sc = run_dir / "scenario"
    net, topo_path, add, gs = build_network(cfg, sc)
    topo = Topology.load(topo_path)
    specs = generate_demand(cfg)
    spec_by_id = {s.vid: s for s in specs}
    routes = write_routes(specs, cfg, sc / "routes.rou.xml")
    sumocfg = write_sumocfg(cfg, sc, net, routes, add, gs)

    log = RunLogger(run_dir, cfg.run.message_trace)
    manifest = {"run_id": run_id, "status": "running", "seed": cfg.run.seed, "vehicles": cfg.run.vehicles,
                "config": cfg.to_dict(), "gui": gui,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "files": {"net_sha256": sha256_path(net), "routes_sha256": sha256_path(routes),
                          "topology_sha256": sha256_path(topo_path), "sumocfg": "scenario/scenario.sumocfg"}}
    log.write_json("manifest.json", manifest)

    world = TraciWorld(sumocfg, cfg, topo.tls_id, label=f"trafma-{run_id}")
    registry = AgentRegistry(cfg)
    coord = Coordinator(cfg)
    seed = cfg.comms.seed_offset + cfg.run.seed
    v2i, i2c = Channel("v2i", cfg.comms.v2i, seed), Channel("i2c", cfg.comms.i2c, seed)

    disp: dict = {}                       # msg_id -> disposition (vehicle reports)
    cnt = {"vehicle_reports_sent": 0, "rsu_summaries_sent": 0, "v2i_lost": 0, "v2i_dropped_range": 0,
           "v2i_duplicated": 0, "i2c_lost": 0, "i2c_duplicated": 0, "registry_mismatch": 0,
           "collisions": 0, "teleports": 0, "emergency_stops": 0, "steps": 0}
    truth_acc = {a: {"n": 0, "q_sum": 0.0, "q_max": 0.0, "obs_q_sum": 0.0, "obs_n": 0} for a in APPROACHES}
    departed_ids: set = set()
    status, err = "running", None
    t_wall0 = time.monotonic()
    report_every = max(int(round(cfg.run.report_step_s / cfg.run.step_s)), 1)
    step_i = 0
    last_snapshot = None

    def trace_msg(now, ev, env_d, frm, to, link, reason=None, exception=False):
        rec = {"t": now, "ev": ev, "msg": env_d.get("msg_id") if isinstance(env_d, dict) else str(env_d),
               "type": env_d.get("msg_type") if isinstance(env_d, dict) else None, "from": frm, "to": to,
               "link": link}
        if reason:
            rec["reason"] = reason
        log.msg(rec, exception or ev not in ("send", "deliver", "accept", "relay"))

    def set_disp(mid, val):
        if disp.get(mid) not in ACCEPTED:
            disp[mid] = val

    try:
        world.start(gui=gui)
        say(f"[trafma] run {run_id}: N={cfg.run.vehicles} seed={cfg.run.seed} "
            f"{'GUI' if gui else 'headless'} SUMO {world.sumo_version}")
        while True:
            events = world.step()
            step_i += 1
            now = round(world.time, 6)
            cnt["steps"] += 1
            snap = world.snapshot()

            # ---- evaluator-side event logging
            for vid in events.departed:
                spec = spec_by_id.get(vid)
                v = snap.vehicles.get(vid)
                if spec is None or v is None or v.edge_id != in_edge(spec.origin):
                    cnt["registry_mismatch"] += 1
                    origin = spec.origin if spec else v.edge_id.split("_")[0]
                else:
                    origin = spec.origin
                registry.on_depart(vid, origin, now)
                departed_ids.add(vid)
                log.event({"t": now, "ev": "depart", "vid": vid, "origin": origin})
            for vid in events.arrived:
                registry.on_arrive(vid, now)
                log.close_agent(vid)
                log.event({"t": now, "ev": "arrive", "vid": vid})
            for key, lst in (("collisions", events.collisions), ("teleports", events.teleports),
                             ("emergency_stops", events.emergency_stops)):
                for vid in lst:
                    cnt[key] += 1
                    log.event({"t": now, "ev": key[:-1], "vid": vid})

            is_report = step_i % report_every == 0
            if is_report:
                # ---- RSUs observe + summarise (direct evidence)
                for a, rsu in registry.rsus.items():
                    view = rsu_view(snap, topo, a, cfg)
                    detected = view.pop("_detected")
                    env = rsu.summarise(view, now)
                    log.agent(rsu.id, {"t": now, "kind": "observation", "view": env.payload, "detected": detected,
                                       "msg_id": env.msg_id})
                    cnt["rsu_summaries_sent"] += 1
                    out = i2c.send(env.to_dict(), now)
                    trace_msg(now, "send", env.to_dict(), rsu.id, COORDINATOR_ID, "i2c")
                    if "lost" in out:
                        trace_msg(now, "lost", env.to_dict(), rsu.id, COORDINATOR_ID, "i2c")
                        cnt["i2c_lost"] += 1
                    if "duplicated" in out:
                        cnt["i2c_duplicated"] += 1

                # ---- vehicles observe + report to parent RSU over V2I
                for vid, agent in sorted(registry.vehicles.items()):
                    view = vehicle_view(snap, topo, vid, agent.origin, cfg)
                    if view is None:
                        continue
                    env = agent.report(view, now)
                    log.agent(vid, {"t": now, "kind": "observation", "parent": agent.parent,
                                    "view": env.payload, "msg_id": env.msg_id})
                    cnt["vehicle_reports_sent"] += 1
                    disp[env.msg_id] = "sent"
                    ev, rg = snap.vehicles[vid], topo.approaches[agent.origin].rsu_xy
                    dist = ((ev.x - rg[0]) ** 2 + (ev.y - rg[1]) ** 2) ** 0.5
                    d = env.to_dict()
                    trace_msg(now, "send", d, vid, agent.parent, "v2i")
                    if dist > cfg.agents.v2i_range_m:
                        cnt["v2i_dropped_range"] += 1
                        set_disp(env.msg_id, "dropped_range")
                        trace_msg(now, "dropped_range", d, vid, agent.parent, "v2i", f"{dist:.1f}m")
                        continue
                    out = v2i.send(d, now)
                    if "lost" in out:
                        cnt["v2i_lost"] += 1
                        set_disp(env.msg_id, "lost_v2i")
                        trace_msg(now, "lost", d, vid, agent.parent, "v2i")
                    if "duplicated" in out:
                        cnt["v2i_duplicated"] += 1
                    if cfg.run.gui and gui:
                        world.color_vehicle(vid, _gui_color(topo, ev, agent.origin, cfg))

            # ---- V2I deliveries -> RSU accept/reject -> relay bundles over I2C
            for d, tag in v2i.deliver_due(now):
                rsu = registry.rsus[registry.parent_of(d["origin_id"]).split("_", 1)[1]]
                ok, why = rsu.receive_from_child(d, now)
                if ok:
                    set_disp(d["msg_id"], "at_rsu")
                    trace_msg(now, "deliver", d, d["origin_id"], rsu.id, "v2i")
                else:
                    if tag == "orig" or disp.get(d["msg_id"]) == "sent":
                        set_disp(d["msg_id"], f"rsu_rejected:{why.split(':')[0]}")
                    trace_msg(now, "rsu_reject", d, d["origin_id"], rsu.id, "v2i", why)
            for a, rsu in registry.rsus.items():
                bundle = rsu.flush_relay(now)
                if bundle is None:
                    continue
                bd = bundle.to_dict()
                out = i2c.send(bd, now)
                trace_msg(now, "relay", bd, rsu.id, COORDINATOR_ID, "i2c")
                if "lost" in out:
                    cnt["i2c_lost"] += 1
                    for r in bd["payload"]["reports"]:
                        set_disp(r["msg_id"], "lost_i2c")
                    trace_msg(now, "lost", bd, rsu.id, COORDINATOR_ID, "i2c")
                if "duplicated" in out:
                    cnt["i2c_duplicated"] += 1

            # ---- I2C deliveries -> coordinator
            for d, tag in i2c.deliver_due(now):
                for mid, outcome, why in coord.ingest(d, now):
                    if outcome == "accepted":
                        if mid in disp:
                            disp[mid] = "coordinator_accepted"
                        trace_msg(now, "accept", {"msg_id": mid, "msg_type": None}, d["sender_id"],
                                  COORDINATOR_ID, "i2c", why)
                    else:
                        if mid in disp:
                            set_disp(mid, f"coord_rejected:{why.split(':')[0]}")
                        trace_msg(now, "coord_reject", {"msg_id": mid, "msg_type": None}, d["sender_id"],
                                  COORDINATOR_ID, "i2c", why)

            if is_report:
                last_snapshot = coord.snapshot(now)
                log.glob(last_snapshot)
                tr = world_truth(snap, topo, cfg)
                log.truth({"t": now, "approaches": tr})
                for a in APPROACHES:
                    ta = truth_acc[a]
                    ta["n"] += 1
                    ta["q_sum"] += tr[a]["zone_queue_m"]
                    ta["q_max"] = max(ta["q_max"], tr[a]["zone_queue_m"])
                    dobs = last_snapshot["approaches"][a]["direct"]
                    if dobs and not dobs["stale"]:
                        ta["obs_q_sum"] += dobs["observation"]["queue_length_m"]
                        ta["obs_n"] += 1
                log.trace({"t": now, "ph": snap.signal.phase_index,
                           "v": [[v.vid, v.edge_id, round(v.lane_pos, 2), round(v.speed, 3)]
                                 for v in snap.vehicles.values()]})

            # ---- termination
            n_pending = len(specs) - len(departed_ids)
            if now >= cfg.run.duration_s - 1e-9:
                status = "completed" if (n_pending == 0 and not registry.vehicles) else "sim_timeout"
                break
            if cfg.run.stop_when_empty and n_pending == 0 and not registry.vehicles:
                status = "completed"
                break
            if time.monotonic() - t_wall0 > cfg.run.wall_timeout_s:
                status = "wall_timeout"
                break
    except SumoStartError as e:
        status, err = "error", f"{e}"
    except Exception as e:
        status, err = "error", "".join(traceback.format_exception_only(type(e), e)).strip()
        manifest["traceback"] = traceback.format_exc()
    finally:
        world.close()
        log.close()

    # ------------------------------------------------ finalise
    end_t = round(world_time_safe(last_snapshot, cfg, step_i), 6)
    pending = sorted(set(spec_by_id) - departed_ids)
    unfinished = sorted(registry.vehicles)
    totals = coord.totals()
    unique = {}
    for m, v in disp.items():
        k = v.split(":")[0]
        unique[k] = unique.get(k, 0) + 1
    in_flight_reports = sum(1 for v in disp.values() if v in ("sent", "at_rsu"))
    sumo_err = (sc / "sumo_errors.log")
    manifest.update({
        "status": status, "error": err, "end_sim_time_s": end_t,
        "wall_time_s": round(time.monotonic() - t_wall0, 3),
        "traci_closed_cleanly": world.closed_cleanly,
        "versions": _versions(world),
        "vehicles_scheduled": len(specs), "vehicles_departed": len(departed_ids),
        "vehicles_arrived": len(registry.retired), "vehicles_active_at_end": len(unfinished),
        "pending_ids": pending, "unfinished_ids": unfinished,
        "registry": {"rsus": [r.id for r in registry.rsus.values()],
                     "events_depart": sum(1 for e in registry.events if e[1] == "depart"),
                     "events_arrive": sum(1 for e in registry.events if e[1] == "arrive")},
        "counters": cnt,
        "reports_disposition": dict(sorted(unique.items())),
        "reports_in_flight_or_unresolved": in_flight_reports,
        "coordinator": totals,
        "rsu_child_stats": {r.id: r.stats for r in registry.rsus.values()},
        "sumo_error_log_bytes": sumo_err.stat().st_size if sumo_err.is_file() else None,
        "tripinfo": _parse_tripinfo(sc / "tripinfo.xml"),
        "trace_sha256": sha256_path(run_dir / "trace.jsonl"),
        "approach_summary": {a: {"mean_zone_queue_m": round(t["q_sum"] / t["n"], 3) if t["n"] else None,
                                 "max_zone_queue_m": round(t["q_max"], 3),
                                 "mean_rsu_observed_queue_m": round(t["obs_q_sum"] / t["obs_n"], 3) if t["obs_n"] else None}
                             for a, t in truth_acc.items()},
        "message_trace_lines": log.n_msg_lines,
    })
    manifest["files"]["messages_sha256"] = sha256_path(run_dir / "messages.jsonl")
    if last_snapshot is not None:
        (run_dir / "global_snapshot_final.json").write_text(
            __import__("json").dumps(last_snapshot, indent=1, sort_keys=True), encoding="utf-8")
    (run_dir / "manifest.json").write_text(__import__("json").dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    (run_dir / "summary.txt").write_text(format_summary(manifest), encoding="utf-8")
    say(format_summary(manifest))
    if status == "error":
        raise RuntimeError(f"run failed: {err}")
    return RunResult(run_dir, status, manifest)


def world_time_safe(last_snapshot, cfg, step_i) -> float:
    return step_i * cfg.run.step_s


def format_summary(m: dict) -> str:
    c, co = m["counters"], m["coordinator"]
    L = [f"=== TRAF-MA run {m['run_id']} ===",
         f"status            : {m['status']}  (sim end t={m['end_sim_time_s']} s, wall {m['wall_time_s']} s, "
         f"TraCI closed cleanly: {m['traci_closed_cleanly']})",
         f"seed / N          : {m['seed']} / {m['vehicles']}",
         f"vehicles          : scheduled {m['vehicles_scheduled']}, departed {m['vehicles_departed']}, "
         f"arrived {m['vehicles_arrived']}, still active {m['vehicles_active_at_end']}, "
         f"never inserted {len(m['pending_ids'])}",
         f"RSUs              : {', '.join(m['registry']['rsus'])}",
         f"safety            : collisions {c['collisions']}, teleports {c['teleports']}, "
         f"emergency stops {c['emergency_stops']}",
         f"messages          : vehicle reports sent {c['vehicle_reports_sent']} -> coordinator accepted "
         f"{co['relayed_accepted']} (relayed via parent RSU); RSU summaries {co['direct_accepted']}/{c['rsu_summaries_sent']}",
         f"                    rejected {co['rejected']}",
         f"report fate       : {m['reports_disposition']}"]
    if m.get("tripinfo"):
        t = m["tripinfo"]
        L.append(f"trips             : n={t.get('count')} mean duration {t.get('mean_duration_s')} s, "
                 f"mean waiting {t.get('mean_waiting_s')} s, max depart delay {t.get('max_depart_delay_s')} s")
    L.append("approach queue (zone, truth vs RSU-observed), metres:")
    for a, s in m["approach_summary"].items():
        L.append(f"   {a}: mean {s['mean_zone_queue_m']} / obs {s['mean_rsu_observed_queue_m']}, max {s['max_zone_queue_m']}")
    L.append(f"trace sha256      : {m['trace_sha256'][:16]}...")
    return "\n".join(L)
