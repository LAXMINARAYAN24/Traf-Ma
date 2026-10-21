"""P1.3 - Seeded, reproducible demand. Same (seed, config) -> byte-identical routes file."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from . import APPROACHES
from .config import Config, ConfigError
from .netgen import in_edge, out_edge, movement_dest, route_id, movements

ASYMMETRIC_DEFAULT = {"N": 0.4, "S": 0.4, "E": 0.1, "W": 0.1}


@dataclass(frozen=True)
class VehicleSpec:
    vid: str
    depart_s: float
    origin: str
    dest: str
    turn: str

    @property
    def route(self) -> str:
        return route_id(self.origin, self.dest)


def approach_weights(cfg: Config) -> dict:
    d = cfg.demand
    if d.profile == "balanced":
        w = {a: 1.0 for a in APPROACHES}
    elif d.profile == "asymmetric":
        w = dict(d.approach_weights or ASYMMETRIC_DEFAULT)
    else:
        w = dict(d.approach_weights)
    tot = sum(w.values())
    return {a: w[a] / tot for a in APPROACHES}


def depart_window(cfg: Config) -> float:
    return max(cfg.run.duration_s - cfg.demand.depart_tail_s, 0.0)


def generate_demand(cfg: Config) -> list:
    """Deterministic list of VehicleSpec sorted by departure time."""
    cfg.validate()
    n = cfg.run.vehicles
    if n == 0:
        return []
    rng = random.Random(f"trafma-demand:{cfg.run.seed}")
    res = cfg.demand.depart_resolution_s
    win = depart_window(cfg)
    if win <= 0:
        raise ConfigError("no departure window: run.duration_s <= demand.depart_tail_s")
    w = approach_weights(cfg)
    turns = ("straight", "left", "right")
    tp = [cfg.demand.turn_probs[t] for t in turns]
    times = sorted(round(rng.uniform(0, win) / res) * res for _ in range(n))
    specs = []
    for i, t in enumerate(times):
        o = rng.choices(APPROACHES, weights=[w[a] for a in APPROACHES])[0]
        turn = rng.choices(turns, weights=tp)[0]
        specs.append(VehicleSpec(f"veh_{i:04d}", round(t, 6), o, movement_dest(o, turn), turn))
    return specs


def write_routes(specs: list, cfg: Config, path: Path) -> Path:
    v = cfg.vehicle
    lines = ['<routes>',
             f'  <vType id="car" length="{v.length_m}" minGap="{v.min_gap_m}" accel="{v.accel_mps2}" '
             f'decel="{v.decel_mps2}" tau="{v.tau_s}" sigma="{v.sigma}" maxSpeed="{v.max_speed_mps}" '
             f'guiShape="passenger" color="255,200,0"/>']
    for o, _t, d in movements():
        lines.append(f'  <route id="{route_id(o, d)}" edges="{in_edge(o)} {out_edge(d)}"/>')
    for s in specs:
        lines.append(f'  <vehicle id="{s.vid}" type="car" route="{s.route}" depart="{s.depart_s:.2f}" '
                     f'departLane="best" departPos="base" departSpeed="max"/>')
    lines.append('</routes>')
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)
