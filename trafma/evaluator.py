"""Evaluator-only ground truth and labels. Never imported by agent code paths."""
from __future__ import annotations

from . import APPROACHES
from .config import Config
from .netgen import Topology
from .sensors import queue_chain, zone_items
from .world import WorldSnapshot


def congestion_class(queue_m: float, zone_m: float, thresholds) -> int:
    """0 = free, 1 = moderate, 2 = heavy, from queue/zone ratio (editable thresholds)."""
    r = queue_m / zone_m if zone_m > 0 else 0.0
    if r < thresholds[0]:
        return 0
    if r < thresholds[1]:
        return 1
    return 2


def approach_truth(world: WorldSnapshot, topo: Topology, approach: str, cfg: Config) -> dict:
    g = topo.approaches[approach]
    zone = min(cfg.agents.rsu_zone_m, g.in_len)
    zi = zone_items(world, topo, approach, zone)
    q_m, q_n = queue_chain([(d, v.speed, v.length) for d, v in zi], cfg)
    q_m = min(q_m, zone)
    ei = zone_items(world, topo, approach, g.in_len)       # whole incoming road
    eq_m, eq_n = queue_chain([(d, v.speed, v.length) for d, v in ei], cfg)
    return {
        "approach": approach,
        "zone_count": len(zi),
        "zone_halting": sum(1 for _, v in zi if v.speed <= cfg.agents.halting_speed_mps),
        "zone_queue_m": round(q_m, 3), "zone_queue_count": q_n,
        "zone_queue_ratio": round(q_m / zone, 4),
        "congestion_class": congestion_class(q_m, zone, cfg.labels.congestion_thresholds),
        "edge_count": len(ei), "edge_queue_m": round(eq_m, 3),
    }


def world_truth(world: WorldSnapshot, topo: Topology, cfg: Config) -> dict:
    return {a: approach_truth(world, topo, a, cfg) for a in APPROACHES}
