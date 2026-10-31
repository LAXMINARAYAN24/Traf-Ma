"""P1.5 - Typed, versioned, provenance-carrying messages and their validator."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import MSG_SCHEMA, MSG_VERSION

VEHICLE_REPORT, RSU_SUMMARY, RELAY_BUNDLE = "vehicle_report", "rsu_summary", "relay_bundle"

EVIDENCE_DIRECT = "direct"            # RSU's own sensing
EVIDENCE_RELAYED_CHILD = "relayed_child"   # a vehicle's report, forwarded by its parent RSU
EVIDENCE_RELAY = "relay"              # the bundle wrapper itself

VEHICLE_FIELDS = ["scope", "edge_id", "speed_mps", "accel_mps2", "lane_pos_m", "waiting_time_s",
                  "dist_to_stop_line_m", "signal_state", "leader_gap_m", "leader_speed_mps",
                  "neighbor_count", "neighbor_ahead_count", "neighbor_behind_count",
                  "neighbor_halting_count", "neighbor_mean_speed_mps", "signal_remaining_s"]
RSU_FIELDS = ["approach", "zone_m", "coverage_m", "vehicle_count", "halting_count", "queue_length_m",
              "queue_vehicle_count", "mean_speed_mps", "mean_accel_mps2", "mean_dist_to_stop_m",
              "density_veh_per_km", "mean_waiting_time_s", "max_waiting_time_s", "outbound_count", "signal_state", "signal_remaining_s",
              "signal_phase_index"]
PAYLOAD_FIELDS = {VEHICLE_REPORT: VEHICLE_FIELDS, RSU_SUMMARY: RSU_FIELDS}
RESERVED = {"trust": "unavailable", "signature": "unavailable"}   # filled in Phase 3


@dataclass
class Envelope:
    msg_id: str
    msg_type: str
    origin_id: str
    sender_id: str
    receiver_id: str
    seq: int
    sim_time: float
    hops: list
    evidence: str
    payload: dict
    schema: str = MSG_SCHEMA
    version: str = MSG_VERSION
    reserved: dict = field(default_factory=lambda: dict(RESERVED))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Envelope":
        return Envelope(**d)


def make_msg_id(origin: str, seq: int) -> str:
    return f"{origin}#{seq}"


def build_payload(kind: str, view: dict) -> dict:
    keys = PAYLOAD_FIELDS[kind]
    p = {k: view.get(k) for k in keys}
    p["unavailable"] = sorted(k for k in keys if p[k] is None)
    return p


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(d) -> list:
    """Return a list of problems ('' list == valid). Never raises on malformed input."""
    if not isinstance(d, dict):
        return ["not a mapping"]
    probs = []
    need = ["schema", "version", "msg_id", "msg_type", "origin_id", "sender_id", "receiver_id",
            "seq", "sim_time", "hops", "evidence", "payload"]
    for k in need:
        if k not in d:
            probs.append(f"missing {k}")
    if probs:
        return probs
    if d["schema"] != MSG_SCHEMA:
        probs.append(f"bad schema {d['schema']!r}")
    if d["version"].split(".")[0] != MSG_VERSION.split(".")[0] if isinstance(d["version"], str) else True:
        probs.append(f"incompatible version {d['version']!r}")
    if d["msg_type"] not in (VEHICLE_REPORT, RSU_SUMMARY, RELAY_BUNDLE):
        probs.append(f"unknown msg_type {d['msg_type']!r}")
        return probs
    if not isinstance(d["seq"], int) or isinstance(d["seq"], bool) or d["seq"] < 0:
        probs.append("bad seq")
    if not _finite(d["sim_time"]) or d["sim_time"] < 0:
        probs.append("bad sim_time")
    if not isinstance(d["hops"], list) or not d["hops"] or not all(isinstance(h, str) for h in d["hops"]):
        probs.append("bad hops")
    if not isinstance(d["payload"], dict):
        probs.append("payload not a mapping")
        return probs
    if isinstance(d["seq"], int) and d["msg_id"] != make_msg_id(d["origin_id"], d["seq"]):
        probs.append("msg_id does not match origin/seq")
    t, p = d["msg_type"], d["payload"]
    if t == RELAY_BUNDLE:
        if not isinstance(p.get("reports"), list) or "rsu_id" not in p:
            probs.append("bad relay payload")
        return probs
    for k in PAYLOAD_FIELDS[t]:
        if k not in p:
            probs.append(f"payload missing {k}")
    if "unavailable" not in p or not isinstance(p.get("unavailable"), list):
        probs.append("payload missing unavailable list")
        return probs
    for k in PAYLOAD_FIELDS[t]:
        if k in p:
            if p[k] is None and k not in p["unavailable"]:
                probs.append(f"{k} is None but not declared unavailable")
            if p[k] is not None and k in p["unavailable"]:
                probs.append(f"{k} declared unavailable but has a value")
            if isinstance(p[k], float) and not math.isfinite(p[k]):
                probs.append(f"{k} not finite")
    exp = EVIDENCE_DIRECT if t == RSU_SUMMARY else EVIDENCE_RELAYED_CHILD
    if d["evidence"] != exp:
        probs.append(f"evidence {d['evidence']!r} inconsistent with {t}")
    return probs


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
