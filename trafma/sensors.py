"""P1.4 - Observation boundary. The ONLY functions that turn world truth into agent-visible data.

RSU view      : vehicles on its own incoming road within `rsu_zone_m` of the stop line, vehicles on
                its own outgoing road within `rsu_outbound_zone_m`, and the signal head of its road.
                Nothing about any other approach is readable.
Vehicle view  : own kinematics always; neighbours/leader/signal ONLY while the vehicle is on its own
                incoming approach, within `vehicle_radius_m`. Out-of-coverage values are None and
                listed in `unavailable` - unknown is never encoded as zero.
                With `agents.signal_broadcast=true` the vehicle's own RSU also broadcasts the signal state and
                time-to-change to it anywhere on the incoming approach (default false = original behaviour).
"""
from __future__ import annotations

from typing import Optional

from .config import Config
from .netgen import Topology
from .world import WorldSnapshot, VehicleState


# ------------------------------------------------------------------ signal helpers
def approach_signal_char(state: str, link_indices: list) -> str:
    chars = {state[i] for i in link_indices}
    if chars & {"G", "g"}:
        return "G"
    if "y" in chars or "Y" in chars:
        return "y"
    return "r"


def time_to_state_change(topo: Topology, approach: str, phase_index: int, remaining: float) -> float:
    """Seconds until this approach's displayed state (G/y/r) next changes, from the fixed program."""
    idx = topo.approaches[approach].link_indices
    phases = topo.signal_phases
    cur = approach_signal_char(phases[phase_index]["state"], idx)
    t = remaining
    for k in range(1, len(phases) + 1):
        ph = phases[(phase_index + k) % len(phases)]
        if approach_signal_char(ph["state"], idx) != cur:
            return round(t, 6)
        t += ph["duration"]
    return round(t, 6)


# ------------------------------------------------------------------ queue logic (shared with evaluator)
def queue_chain(items: list, cfg: Config):
    """items: [(dist_to_stop_line, speed, length)] -> (queue_length_m, queue_vehicle_count).

    A queue is the contiguous chain of halted vehicles (speed <= halting_speed) that starts within
    `queue_start_m` of the stop line, extended while bumper-to-bumper gaps <= `queue_gap_m`.
    Queue length is measured to the rear bumper of the last vehicle in the chain."""
    a = cfg.agents
    halted = sorted((d, ln) for d, sp, ln in items if sp <= a.halting_speed_mps)
    if not halted or halted[0][0] > a.queue_start_m:
        return 0.0, 0
    chain = [halted[0]]
    for d, ln in halted[1:]:
        pd, _ = chain[-1]
        if d - pd - ln <= a.queue_gap_m:
            chain.append((d, ln))
        else:
            break
    d_last, ln_last = chain[-1]
    return d_last + ln_last, len(chain)


def zone_items(world: WorldSnapshot, topo: Topology, approach: str, zone_m: float):
    g = topo.approaches[approach]
    out = []
    for v in world.vehicles.values():
        if v.edge_id == g.in_edge:
            d = g.in_len - v.lane_pos
            if d <= zone_m:
                out.append((max(d, 0.0), v))
    return out


# ------------------------------------------------------------------ RSU
def rsu_view(world: WorldSnapshot, topo: Topology, approach: str, cfg: Config) -> dict:
    a = cfg.agents
    g = topo.approaches[approach]
    zone = min(a.rsu_zone_m, g.in_len)
    inz = zone_items(world, topo, approach, zone)
    n = len(inz)
    q_m, q_n = queue_chain([(d, v.speed, v.length) for d, v in inz], cfg)
    q_m = min(q_m, zone)
    halting = sum(1 for _, v in inz if v.speed <= a.halting_speed_mps)
    out_n = sum(1 for v in world.vehicles.values()
                if v.edge_id == g.out_edge and v.lane_pos <= a.rsu_outbound_zone_m)
    sig = world.signal
    unavailable = []
    if n:
        mean_speed = sum(v.speed for _, v in inz) / n
        mean_accel = sum(v.accel for _, v in inz) / n
        mean_dist = sum(d for d, _ in inz) / n          # metres to the stop line (not normalised here)
        mean_wait = sum(v.waiting_s for _, v in inz) / n
        max_wait = max(v.waiting_s for _, v in inz)
    else:
        mean_speed = mean_accel = mean_dist = mean_wait = max_wait = None
        unavailable += ["mean_speed_mps", "mean_accel_mps2", "mean_dist_to_stop_m",
                        "mean_waiting_time_s", "max_waiting_time_s"]
    return {
        "approach": approach, "zone_m": zone, "coverage_m": zone,
        "vehicle_count": n, "halting_count": halting,
        "queue_length_m": round(q_m, 3), "queue_vehicle_count": q_n,
        "mean_speed_mps": None if mean_speed is None else round(mean_speed, 3),
        "mean_accel_mps2": None if mean_accel is None else round(mean_accel, 3),
        "mean_dist_to_stop_m": None if mean_dist is None else round(mean_dist, 3),
        "density_veh_per_km": round(n / (zone / 1000.0), 3),
        "mean_waiting_time_s": None if mean_wait is None else round(mean_wait, 3),
        "max_waiting_time_s": None if max_wait is None else round(max_wait, 3),
        "outbound_count": out_n,
        "signal_state": approach_signal_char(sig.state, g.link_indices),
        "signal_remaining_s": time_to_state_change(topo, approach, sig.phase_index, sig.remaining_s),
        "signal_phase_index": sig.phase_index,
        "unavailable": unavailable,
        "_detected": sorted(v.vid for _, v in inz),   # local debug only; stripped before any message
    }


# ------------------------------------------------------------------ vehicle
VEHICLE_MASKED = ["dist_to_stop_line_m", "signal_state", "signal_remaining_s", "leader_gap_m", "leader_speed_mps",
                  "neighbor_count", "neighbor_ahead_count", "neighbor_behind_count",
                  "neighbor_halting_count", "neighbor_mean_speed_mps"]


def vehicle_scope(topo: Topology, edge_id: str, parent: str) -> str:
    if edge_id == topo.approaches[parent].in_edge:
        return "incoming_local"
    if edge_id.startswith(":"):
        return "in_junction"
    return "out_of_coverage"


def vehicle_view(world: WorldSnapshot, topo: Topology, vid: str, parent: str, cfg: Config) -> Optional[dict]:
    me = world.vehicles.get(vid)
    if me is None:
        return None
    a = cfg.agents
    g = topo.approaches[parent]
    scope = vehicle_scope(topo, me.edge_id, parent)
    view = {"scope": scope, "edge_id": me.edge_id, "speed_mps": round(me.speed, 3),
            "accel_mps2": round(me.accel, 3), "lane_pos_m": round(me.lane_pos, 3),
            "waiting_time_s": round(me.waiting_s, 3)}
    if scope != "incoming_local":
        for k in VEHICLE_MASKED:
            view[k] = None
        view["unavailable"] = list(VEHICLE_MASKED)
        return view
    dist = g.in_len - me.lane_pos
    view["dist_to_stop_line_m"] = round(max(dist, 0.0), 3)
    unavailable = []
    if a.signal_broadcast or dist <= a.vehicle_radius_m:   # head readable when close, or if the RSU broadcasts it
        view["signal_state"] = approach_signal_char(world.signal.state, g.link_indices)
    else:
        view["signal_state"] = None
        unavailable.append("signal_state")
    if a.signal_broadcast:              # the timer is only ever available through the RSU broadcast
        view["signal_remaining_s"] = time_to_state_change(topo, parent, world.signal.phase_index,
                                                          world.signal.remaining_s)
    else:
        view["signal_remaining_s"] = None
        unavailable.append("signal_remaining_s")
    nb = [v for v in world.vehicles.values()
          if v.vid != vid and v.edge_id == me.edge_id and abs(v.lane_pos - me.lane_pos) <= a.vehicle_radius_m]
    ahead = [v for v in nb if v.lane_pos > me.lane_pos]
    view["neighbor_count"] = len(nb)
    view["neighbor_ahead_count"] = len(ahead)
    view["neighbor_behind_count"] = len(nb) - len(ahead)
    view["neighbor_halting_count"] = sum(1 for v in nb if v.speed <= a.halting_speed_mps)
    if nb:
        view["neighbor_mean_speed_mps"] = round(sum(v.speed for v in nb) / len(nb), 3)
    else:
        view["neighbor_mean_speed_mps"] = None
        unavailable.append("neighbor_mean_speed_mps")
    if ahead:
        ld = min(ahead, key=lambda v: v.lane_pos)
        view["leader_gap_m"] = round(ld.lane_pos - me.lane_pos - ld.length, 3)
        view["leader_speed_mps"] = round(ld.speed, 3)
    else:
        view["leader_gap_m"] = None
        view["leader_speed_mps"] = None
        unavailable += ["leader_gap_m", "leader_speed_mps"]
    view["unavailable"] = unavailable
    return view
