"""Typed, validated configuration. All numeric values are *editable defaults*, not findings."""
from __future__ import annotations

import copy
import dataclasses
import math
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from . import APPROACHES


class ConfigError(ValueError):
    """Raised for any invalid / malformed configuration."""


@dataclass
class RunCfg:
    seed: int = 42
    vehicles: int = 20                 # N vehicle agents scheduled over the run
    duration_s: float = 600.0
    step_s: float = 1.0                # SUMO step length
    report_step_s: float = 1.0         # agents observe / report every report_step_s
    gui: bool = False
    gui_delay_ms: int = 50
    wall_timeout_s: float = 900.0      # wall-clock guard; status becomes "wall_timeout"
    stop_when_empty: bool = False      # stop once every vehicle has arrived
    output_dir: str = "runs"
    run_id: Optional[str] = None
    message_trace: str = "full"        # "full" | "exceptions" (only drops/rejections)


@dataclass
class NetworkCfg:
    arm_length_m: float = 300.0
    speed_limit_mps: float = 13.89
    lanes_per_direction: int = 1


@dataclass
class SignalCfg:
    # Split phasing: each approach gets its own fully protected green -> conflict-free by construction.
    green_s: float = 25.0
    yellow_s: float = 3.0
    all_red_s: float = 2.0


@dataclass
class DemandCfg:
    profile: str = "balanced"          # "balanced" | "asymmetric" | "custom"
    approach_weights: Optional[dict] = None
    turn_probs: dict = field(default_factory=lambda: {"straight": 0.6, "left": 0.2, "right": 0.2})
    depart_tail_s: float = 120.0       # departures stop this long before the end so vehicles can clear
    depart_resolution_s: float = 0.1
    max_vehicles: int = 5000


@dataclass
class VehicleTypeCfg:
    length_m: float = 5.0
    min_gap_m: float = 2.5
    accel_mps2: float = 2.6
    decel_mps2: float = 4.5
    tau_s: float = 1.0
    sigma: float = 0.5                 # driver imperfection (seeded by SUMO --seed)
    max_speed_mps: float = 13.89


@dataclass
class AgentCfg:
    rsu_zone_m: float = 150.0          # RSU sensing/target zone before each stop line
    rsu_outbound_zone_m: float = 100.0 # RSU also counts vehicles leaving on its outgoing road
    vehicle_radius_m: float = 100.0    # vehicle sensing radius, restricted to its own incoming approach
    v2i_range_m: float = 400.0         # max vehicle->RSU distance for a delivery
    rsu_lateral_offset_m: float = 8.0  # RSU sits beside the road, this far from the lane centre
    halting_speed_mps: float = 0.1
    queue_gap_m: float = 10.0          # max gap between consecutive halted vehicles in one queue chain
    queue_start_m: float = 15.0        # first halted vehicle must be this close to the stop line


@dataclass
class LinkCfg:
    loss_prob: float = 0.0
    delay_s: float = 0.0
    duplicate_prob: float = 0.0


@dataclass
class CommsCfg:
    v2i: LinkCfg = field(default_factory=LinkCfg)   # vehicle -> RSU
    i2c: LinkCfg = field(default_factory=LinkCfg)   # RSU -> coordinator
    stale_after_s: float = 5.0
    seed_offset: int = 7


@dataclass
class LabelCfg:
    horizon_s: float = 30.0
    history_len: int = 10              # observations per window
    min_windows: int = 16              # min valid windows for a client to be trainable
    congestion_thresholds: list = field(default_factory=lambda: [0.10, 0.35])  # on queue/zone ratio


@dataclass
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    network: NetworkCfg = field(default_factory=NetworkCfg)
    signal: SignalCfg = field(default_factory=SignalCfg)
    demand: DemandCfg = field(default_factory=DemandCfg)
    vehicle: VehicleTypeCfg = field(default_factory=VehicleTypeCfg)
    agents: AgentCfg = field(default_factory=AgentCfg)
    comms: CommsCfg = field(default_factory=CommsCfg)
    labels: LabelCfg = field(default_factory=LabelCfg)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def validate(self) -> "Config":
        r, n, s, d, v, a, c, lb = (self.run, self.network, self.signal, self.demand,
                                   self.vehicle, self.agents, self.comms, self.labels)

        def num(x, name, lo=None, hi=None, strict_lo=False):
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ConfigError(f"{name} must be a finite number, got {x!r}")
            if lo is not None and (x <= lo if strict_lo else x < lo):
                raise ConfigError(f"{name} must be {'>' if strict_lo else '>='} {lo}, got {x}")
            if hi is not None and x > hi:
                raise ConfigError(f"{name} must be <= {hi}, got {x}")

        if isinstance(r.vehicles, bool) or not isinstance(r.vehicles, int):
            raise ConfigError(f"run.vehicles (N) must be an integer, got {r.vehicles!r}")
        if r.vehicles < 0:
            raise ConfigError(f"run.vehicles (N) must be >= 0, got {r.vehicles}")
        if r.vehicles > d.max_vehicles:
            raise ConfigError(f"run.vehicles={r.vehicles} exceeds demand.max_vehicles={d.max_vehicles}")
        if isinstance(r.seed, bool) or not isinstance(r.seed, int):
            raise ConfigError(f"run.seed must be an integer, got {r.seed!r}")
        num(r.duration_s, "run.duration_s", 0, strict_lo=True)
        num(r.step_s, "run.step_s", 0, 10, strict_lo=True)
        num(r.report_step_s, "run.report_step_s", 0, strict_lo=True)
        ratio = r.report_step_s / r.step_s
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise ConfigError("run.report_step_s must be a positive multiple of run.step_s")
        num(r.wall_timeout_s, "run.wall_timeout_s", 0, strict_lo=True)
        if r.message_trace not in ("full", "exceptions"):
            raise ConfigError("run.message_trace must be 'full' or 'exceptions'")
        num(n.arm_length_m, "network.arm_length_m", 50)
        num(n.speed_limit_mps, "network.speed_limit_mps", 1, 50)
        if n.lanes_per_direction != 1:
            raise ConfigError("network.lanes_per_direction: only 1 is supported (plan default)")
        for k in ("green_s", "yellow_s", "all_red_s"):
            num(getattr(s, k), f"signal.{k}", 0, strict_lo=True)
        if d.profile not in ("balanced", "asymmetric", "custom"):
            raise ConfigError("demand.profile must be balanced | asymmetric | custom")
        if d.profile == "custom" and not d.approach_weights:
            raise ConfigError("demand.profile=custom requires demand.approach_weights")
        if d.approach_weights is not None:
            if set(d.approach_weights) != set(APPROACHES):
                raise ConfigError(f"demand.approach_weights must have exactly keys {APPROACHES}")
            for k, w in d.approach_weights.items():
                num(w, f"demand.approach_weights[{k}]", 0)
            if sum(d.approach_weights.values()) <= 0:
                raise ConfigError("demand.approach_weights must not all be zero")
        if set(d.turn_probs) != {"straight", "left", "right"}:
            raise ConfigError("demand.turn_probs must have keys straight/left/right")
        for k, p in d.turn_probs.items():
            num(p, f"demand.turn_probs[{k}]", 0, 1)
        if abs(sum(d.turn_probs.values()) - 1.0) > 1e-6:
            raise ConfigError("demand.turn_probs must sum to 1")
        num(d.depart_tail_s, "demand.depart_tail_s", 0)
        num(d.depart_resolution_s, "demand.depart_resolution_s", 0, strict_lo=True)
        for k in ("length_m", "min_gap_m", "accel_mps2", "decel_mps2", "tau_s", "max_speed_mps"):
            num(getattr(v, k), f"vehicle.{k}", 0, strict_lo=True)
        num(v.sigma, "vehicle.sigma", 0, 1)
        num(a.rsu_zone_m, "agents.rsu_zone_m", 10)
        num(a.rsu_outbound_zone_m, "agents.rsu_outbound_zone_m", 0)
        num(a.vehicle_radius_m, "agents.vehicle_radius_m", 1)
        num(a.v2i_range_m, "agents.v2i_range_m", 1)
        num(a.halting_speed_mps, "agents.halting_speed_mps", 0)
        num(a.queue_gap_m, "agents.queue_gap_m", 0)
        num(a.queue_start_m, "agents.queue_start_m", 0)
        if a.rsu_zone_m > n.arm_length_m - 20:
            raise ConfigError("agents.rsu_zone_m must fit on the approach (arm_length_m - 20)")
        if a.rsu_outbound_zone_m > n.arm_length_m - 20:
            raise ConfigError("agents.rsu_outbound_zone_m must fit on the outgoing road")
        for name, link in (("v2i", c.v2i), ("i2c", c.i2c)):
            num(link.loss_prob, f"comms.{name}.loss_prob", 0, 1)
            num(link.duplicate_prob, f"comms.{name}.duplicate_prob", 0, 1)
            num(link.delay_s, f"comms.{name}.delay_s", 0)
        num(c.stale_after_s, "comms.stale_after_s", 0, strict_lo=True)
        num(lb.horizon_s, "labels.horizon_s", 0, strict_lo=True)
        if not isinstance(lb.history_len, int) or lb.history_len < 1:
            raise ConfigError("labels.history_len must be a positive integer")
        if not isinstance(lb.min_windows, int) or lb.min_windows < 1:
            raise ConfigError("labels.min_windows must be a positive integer")
        th = lb.congestion_thresholds
        if not (isinstance(th, (list, tuple)) and len(th) == 2 and 0 < th[0] < th[1] <= 1):
            raise ConfigError("labels.congestion_thresholds must be [t1, t2] with 0 < t1 < t2 <= 1")
        return self


def _build(cls, data: Any, path: str = ""):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'} must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"unknown key(s) in {path or 'config'}: {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        t = hints[f.name]
        if dataclasses.is_dataclass(t):
            kwargs[f.name] = _build(t, val, f"{path}.{f.name}" if path else f.name)
        else:
            kwargs[f.name] = copy.deepcopy(val)
    return cls(**kwargs)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_from_dict(d: dict) -> Config:
    return _build(Config, d).validate()


def load_config(path: Optional[str | Path] = None, overrides: Optional[dict] = None) -> Config:
    data: dict = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"malformed YAML in {p}: {e}") from e
        if not isinstance(data, dict):
            raise ConfigError(f"{p} must contain a mapping at top level")
    data = deep_merge(data, overrides or {})
    return config_from_dict(data)


def parse_set_overrides(items: list) -> dict:
    """['agents.rsu_zone_m=120', 'run.gui=true'] -> nested dict (values parsed as YAML scalars)."""
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise ConfigError(f"--set expects key=value, got {it!r}")
        key, raw = it.split("=", 1)
        val = yaml.safe_load(raw)
        cur = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return out
