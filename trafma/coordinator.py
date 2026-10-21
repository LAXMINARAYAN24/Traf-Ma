"""P1.5 - Edge coordinator: ingest, validate, de-duplicate and keep a global per-approach state.

RSU-sensed ('direct') and vehicle-reported-via-RSU ('relayed_child') evidence are stored side by side
and are NEVER summed. Unknown is None, never zero."""
from __future__ import annotations

from . import APPROACHES
from .config import Config
from .messages import (validate, VEHICLE_REPORT, RSU_SUMMARY, RELAY_BUNDLE, EVIDENCE_DIRECT,
                       EVIDENCE_RELAYED_CHILD)

REJECT_REASONS = ("invalid", "duplicate", "stale", "out_of_order", "bad_provenance")


class Coordinator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._seen: set = set()
        self._last_seq: dict = {}
        self.direct: dict = {a: None for a in APPROACHES}       # latest accepted RSU summary
        self.child: dict = {a: {} for a in APPROACHES}           # approach -> vid -> latest accepted report
        self.counts = {a: {"direct_accepted": 0, "relayed_accepted": 0,
                           "rejected": {r: 0 for r in REJECT_REASONS}} for a in APPROACHES}
        self.unattributed_rejects = {r: 0 for r in REJECT_REASONS}

    # -------------------------------------------------------------- ingestion
    def ingest(self, d: dict, now: float) -> list:
        """Process one delivered message. Returns [(msg_id, outcome, reason)] for tracing
        (one entry per bundled report for relay bundles)."""
        probs = validate(d)
        if probs:
            self.unattributed_rejects["invalid"] += 1
            return [(d.get("msg_id", "?") if isinstance(d, dict) else "?", "rejected", "invalid:" + probs[0])]
        if d["msg_type"] == RSU_SUMMARY:
            oid = d["origin_id"]
            approach = oid.split("_", 1)[1] if oid.startswith("rsu_") else None
            if approach not in APPROACHES:
                self.unattributed_rejects["bad_provenance"] += 1
                return [(d["msg_id"], "rejected", "bad_provenance:unknown rsu")]
            # the slot is chosen by the sender's identity, never by a payload claim
            return [self._ingest_one(d, now, approach=approach, via=None)]
        if d["msg_type"] == RELAY_BUNDLE:
            rid = d["payload"]["rsu_id"]
            if d["sender_id"] != rid:
                self.unattributed_rejects["bad_provenance"] += 1
                return [(d["msg_id"], "rejected", "bad_provenance:sender!=rsu_id")]
            approach = rid.split("_", 1)[1] if rid.startswith("rsu_") else None
            if approach not in APPROACHES:
                self.unattributed_rejects["bad_provenance"] += 1
                return [(d["msg_id"], "rejected", "bad_provenance:unknown rsu")]
            return [self._ingest_one(r, now, approach=approach, via=rid) for r in d["payload"]["reports"]]
        self.unattributed_rejects["invalid"] += 1
        return [(d["msg_id"], "rejected", "invalid:unexpected type")]

    def _reject(self, approach, reason, msg_id):
        key = reason.split(":")[0]
        (self.counts[approach]["rejected"] if approach in self.counts else self.unattributed_rejects)[key] += 1
        return (msg_id, "rejected", reason)

    def _ingest_one(self, d, now, approach, via):
        mid = d.get("msg_id", "?") if isinstance(d, dict) else "?"
        probs = validate(d)
        if probs:
            return self._reject(approach, "invalid:" + probs[0], mid)
        is_child = d["msg_type"] == VEHICLE_REPORT
        if is_child:
            # provenance: vehicle -> its parent RSU -> coordinator, parent matching the sending RSU
            if via is None or d["hops"] != [d["origin_id"], via]:
                return self._reject(approach, "bad_provenance:hops", mid)
        elif d["sender_id"] != d["origin_id"] or d["payload"]["approach"] != approach:
            return self._reject(approach, "bad_provenance:rsu", mid)
        if d["msg_id"] in self._seen:
            return self._reject(approach, "duplicate", mid)
        if d["seq"] <= self._last_seq.get(d["origin_id"], -1):
            return self._reject(approach, "out_of_order", mid)
        if d["sim_time"] < now - self.cfg.comms.stale_after_s:
            return self._reject(approach, "stale", mid)
        self._seen.add(d["msg_id"])
        self._last_seq[d["origin_id"]] = d["seq"]
        if is_child:
            self.child[approach][d["origin_id"]] = {"sim_time": d["sim_time"], "payload": d["payload"]}
            self.counts[approach]["relayed_accepted"] += 1
        else:
            self.direct[approach] = {"sim_time": d["sim_time"], "payload": d["payload"]}
            self.counts[approach]["direct_accepted"] += 1
        return (mid, "accepted", "direct" if not is_child else "relayed_child")

    # -------------------------------------------------------------- global state
    def snapshot(self, now: float) -> dict:
        fresh = self.cfg.comms.stale_after_s
        out = {"time": now, "approaches": {}}
        for a in APPROACHES:
            dr = self.direct[a]
            direct = None
            if dr is not None:
                age = round(now - dr["sim_time"], 6)
                direct = {"age_s": age, "stale": age > fresh, "observation": dr["payload"]}
            # prune children that left (no fresh report) from the live view
            live = {v: r for v, r in self.child[a].items() if now - r["sim_time"] <= fresh}
            self.child[a] = live
            inc = [r["payload"] for r in live.values() if r["payload"]["scope"] == "incoming_local"]
            spd = [p["speed_mps"] for p in inc if p["speed_mps"] is not None]
            relayed = {
                "reporting_vehicles": len(live),
                "incoming_local_reporters": len(inc),
                "post_junction_reporters": len(live) - len(inc),
                "halted_reporters": sum(1 for p in inc if p["speed_mps"] is not None and
                                        p["speed_mps"] <= self.cfg.agents.halting_speed_mps),
                "mean_reported_speed_mps": round(sum(spd) / len(spd), 3) if spd else None,
                "max_reported_waiting_s": max((p["waiting_time_s"] for p in inc), default=None),
            }
            out["approaches"][a] = {
                "direct": direct,                 # RSU-sensed evidence
                "relayed_child": relayed,         # vehicle evidence relayed by this RSU (kept separate)
                "status": "no_data" if direct is None else ("stale" if direct["stale"] else "ok"),
                "contributions": {"direct_accepted": self.counts[a]["direct_accepted"],
                                  "relayed_accepted": self.counts[a]["relayed_accepted"],
                                  "rejected": dict(self.counts[a]["rejected"])},
            }
        return out

    def totals(self) -> dict:
        return {
            "direct_accepted": sum(c["direct_accepted"] for c in self.counts.values()),
            "relayed_accepted": sum(c["relayed_accepted"] for c in self.counts.values()),
            "rejected": {r: sum(c["rejected"][r] for c in self.counts.values()) + self.unattributed_rejects[r]
                         for r in REJECT_REASONS},
        }
