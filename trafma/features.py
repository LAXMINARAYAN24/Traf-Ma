"""Deterministic adapter from agent-local logs to the Phase 2 feature schema."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

from . import APPROACHES

FEATURE_SCHEMA_VERSION = "trafma.features/1"
SEQUENCE_LENGTH = 10
VALUE_NAMES = (
    "local_mean_speed_mps",
    "local_mean_accel_mps2",
    "local_mean_dist_to_stop_m",
    "local_density_ratio",
    "local_halted_count_ratio",
    "local_mean_waiting_time_s",
    "signal_remaining_s",
    "signal_red",
    "signal_yellow",
    "signal_green",
)
MASK_NAMES = tuple(f"{name}_available" for name in VALUE_NAMES)
FEATURE_NAMES = VALUE_NAMES + MASK_NAMES + ("client_vehicle", "client_rsu")
FEATURE_DIM = len(FEATURE_NAMES)
FORBIDDEN_FEATURE_KEYS = {
    "zone_queue_m", "zone_queue_ratio", "zone_queue_count", "congestion_class",
    "edge_queue_m", "edge_count", "zone_halting", "label", "labels", "label_t",
}
SIGNAL_CHANNELS = {"r": 0, "y": 1, "G": 2}
WAITING_TIME_SCALE_S = 300.0

FEATURE_SCHEMA = {
    "version": FEATURE_SCHEMA_VERSION,
    "sequence_length": SEQUENCE_LENGTH,
    "dimension": FEATURE_DIM,
    "ordered_features": [
        {"name": name, "unit": unit, "source": source, "normalization": normalization}
        for name, unit, source, normalization in (
            ("local_mean_speed_mps", "m/s", "RSU mean_speed_mps; vehicle speed and same-approach neighbour mean",
             "clip(value / configured max_speed_mps, 0, 1)"),
            ("local_mean_accel_mps2", "m/s^2", "RSU mean_accel_mps2; vehicle accel_mps2 proxy",
             "clip(value / (2 * max(configured max_accel_mps2, max_decel_mps2)), -1, 1)"),
            ("local_mean_dist_to_stop_m", "m", "RSU mean_dist_to_stop_m; vehicle dist_to_stop_line_m proxy",
             "clip(value / configured arm_length_m, 0, 1)"),
            ("local_density_ratio", "ratio", "RSU vehicle_count / zone capacity; vehicle local count / clipped visible capacity",
             "clip(value, 0, 1)"),
            ("local_halted_count_ratio", "ratio", "RSU halting_count / zone capacity; vehicle local halted count / clipped visible capacity",
             "clip(value, 0, 1)"),
            ("local_mean_waiting_time_s", "s", "RSU mean_waiting_time_s; vehicle waiting_time_s proxy",
             f"clip(value / {WAITING_TIME_SCALE_S}, 0, 1)"),
            ("signal_remaining_s", "s", "Agent-local signal_remaining_s when present",
             "clip(value / configured signal cycle seconds, 0, 1)"),
            ("signal_red", "one-hot", "Agent-local signal_state", "fixed R/Y/G one-hot"),
            ("signal_yellow", "one-hot", "Agent-local signal_state", "fixed R/Y/G one-hot"),
            ("signal_green", "one-hot", "Agent-local signal_state", "fixed R/Y/G one-hot"),
        )
    ],
    "masks": "One availability mask for each ordered value; all three signal masks match signal_state availability.",
    "client_type": {"vehicle": [1, 0], "rsu": [0, 1]},
    "missing_value_policy": "Unavailable normalized values are 0 only when their mask is 0; measured zero has mask 1.",
    "target_excluded": True,
}
FEATURE_SCHEMA_HASH = hashlib.sha256(
    json.dumps(FEATURE_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


class FeatureError(ValueError):
    """Raised when an agent record violates the feature or observation contract."""


@dataclass(frozen=True)
class FeatureContext:
    zone_m: float
    arm_length_m: float
    incoming_edge_length_m: float
    vehicle_radius_m: float
    vehicle_length_m: float
    min_gap_m: float
    max_speed_mps: float
    max_accel_mps2: float
    max_decel_mps2: float
    halting_speed_mps: float
    signal_cycle_s: float

    @classmethod
    def from_run(cls, config: dict, topology: dict, approach: str) -> "FeatureContext":
        if approach not in APPROACHES:
            raise FeatureError(f"unknown approach: {approach!r}")
        try:
            a = config["agents"]
            n = config["network"]
            v = config["vehicle"]
            g = topology["approaches"][approach]
            ctx = cls(
                zone_m=min(float(a["rsu_zone_m"]), float(g["in_len"])),
                arm_length_m=float(n["arm_length_m"]),
                incoming_edge_length_m=float(g["in_len"]),
                vehicle_radius_m=float(a["vehicle_radius_m"]),
                vehicle_length_m=float(v["length_m"]),
                min_gap_m=float(v["min_gap_m"]),
                max_speed_mps=float(v["max_speed_mps"]),
                max_accel_mps2=float(v["accel_mps2"]),
                max_decel_mps2=float(v["decel_mps2"]),
                halting_speed_mps=float(a["halting_speed_mps"]),
                signal_cycle_s=float(topology["cycle_s"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FeatureError(f"run config/topology lacks valid feature geometry: {exc}") from exc
        if any(not math.isfinite(value) or value <= 0 for value in asdict(ctx).values()):
            raise FeatureError("feature geometry and normalization references must be finite and positive")
        return ctx


@dataclass(frozen=True)
class FeatureWindow:
    dataset_id: str
    run_id: str
    split: str
    scenario: str
    client_id: str
    client_type: str
    approach: str
    end_t: float
    zone_m: float
    features: list[list[float]]
    current_queue_m: Optional[float]
    queue_m: float
    congestion_class: int


def _read_number(view: dict, key: str, *, required: bool = False) -> Optional[float]:
    unavailable = set(view.get("unavailable", []))
    value = view.get(key)
    if key in unavailable and value is not None:
        raise FeatureError(f"{key} is marked unavailable but has value {value!r}")
    if value is None:
        if required:
            raise FeatureError(f"required observation {key} is unavailable")
        if key in view and key not in unavailable:
            raise FeatureError(f"{key} is null but missing from unavailable")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FeatureError(f"{key} must be a finite number or null, got {value!r}")
    return float(value)


def _bounded(value: float, key: str, lower: float, upper: float, tolerance: float = 1e-3) -> float:
    if value < lower - tolerance or value > upper + tolerance:
        raise FeatureError(f"{key}={value} is outside [{lower}, {upper}]")
    return min(max(value, lower), upper)


def adapt_observation(view: dict, client_type: str, context: FeatureContext) -> list[float]:
    """Return one ordered 22-value feature vector from a single agent-local view."""
    if client_type not in ("vehicle", "rsu"):
        raise FeatureError(f"unsupported client type: {client_type!r}")
    if client_type == "vehicle" and view.get("scope") != "incoming_local":
        raise FeatureError("vehicle features require scope=incoming_local")

    unavailable = set(view.get("unavailable", []))
    if client_type == "rsu":
        speed = _read_number(view, "mean_speed_mps")
        accel = _read_number(view, "mean_accel_mps2")
        distance = _read_number(view, "mean_dist_to_stop_m")
        count = _read_number(view, "vehicle_count", required=True)
        halted = _read_number(view, "halting_count", required=True)
        waiting = _read_number(view, "mean_waiting_time_s")
        zone = _read_number(view, "zone_m", required=True)
        if abs(zone - context.zone_m) > 1e-3:
            raise FeatureError(f"RSU record zone_m={zone} disagrees with configured zone {context.zone_m}")
        if count < 0 or halted < 0 or halted > count:
            raise FeatureError("RSU vehicle_count/halting_count are inconsistent")
        density = count / (zone / (context.vehicle_length_m + context.min_gap_m))
        halted_ratio = halted / (zone / (context.vehicle_length_m + context.min_gap_m))
    else:
        speed = _read_number(view, "speed_mps", required=True)
        neighbor_count = _read_number(view, "neighbor_count", required=True)
        if neighbor_count < 0 or not neighbor_count.is_integer():
            raise FeatureError("neighbor_count must be a nonnegative integer")
        neighbor_count = int(neighbor_count)
        neighbor_speed = _read_number(view, "neighbor_mean_speed_mps")
        if neighbor_count and neighbor_speed is None:
            raise FeatureError("neighbor mean speed is unavailable despite a positive neighbor_count")
        if neighbor_count:
            speed = (speed + neighbor_count * neighbor_speed) / (neighbor_count + 1)
        accel = _read_number(view, "accel_mps2", required=True)
        distance = _read_number(view, "dist_to_stop_line_m", required=True)
        waiting = _read_number(view, "waiting_time_s", required=True)
        position = _read_number(view, "lane_pos_m", required=True)
        neighbors_halted = _read_number(view, "neighbor_halting_count", required=True)
        own_halted = 1 if _read_number(view, "speed_mps", required=True) <= context.halting_speed_mps else 0
        visible_length = max(
            0.0,
            min(context.incoming_edge_length_m, position + context.vehicle_radius_m)
            - max(0.0, position - context.vehicle_radius_m),
        )
        capacity = visible_length / (context.vehicle_length_m + context.min_gap_m)
        if capacity <= 0:
            density = halted_ratio = None
        else:
            density = (neighbor_count + 1) / capacity
            halted_ratio = (neighbors_halted + own_halted) / capacity

    if speed is not None:
        _bounded(speed, "local_mean_speed_mps", 0.0, context.max_speed_mps)
    if accel is not None:
        acceleration_scale = 2 * max(context.max_accel_mps2, context.max_decel_mps2)
        _bounded(accel, "local_mean_accel_mps2", -acceleration_scale, acceleration_scale)
    if distance is not None:
        _bounded(distance, "local_mean_dist_to_stop_m", 0.0, context.arm_length_m)
    if waiting is not None and waiting < 0:
        raise FeatureError(f"local_mean_waiting_time_s={waiting} must be nonnegative")
    signal_remaining = _read_number(view, "signal_remaining_s")
    if signal_remaining is not None:
        _bounded(signal_remaining, "signal_remaining_s", 0.0, context.signal_cycle_s)
    signal_state = view.get("signal_state")
    if signal_state is None:
        if "signal_state" in view and "signal_state" not in unavailable:
            raise FeatureError("signal_state is null but missing from unavailable")
        signal_values = [0.0, 0.0, 0.0]
        signal_available = False
    else:
        if signal_state not in ("r", "y", "G", "g"):
            raise FeatureError(f"unknown signal state {signal_state!r}")
        signal_values = [0.0, 0.0, 0.0]
        signal_values[SIGNAL_CHANNELS["G" if signal_state == "g" else signal_state]] = 1.0
        signal_available = True

    normalized = [
        None if speed is None else min(max(speed / context.max_speed_mps, 0.0), 1.0),
        None if accel is None else min(max(
            accel / (2 * max(context.max_accel_mps2, context.max_decel_mps2)), -1.0
        ), 1.0),
        None if distance is None else min(max(distance / context.arm_length_m, 0.0), 1.0),
        None if density is None else min(max(density, 0.0), 1.0),
        None if halted_ratio is None else min(max(halted_ratio, 0.0), 1.0),
        None if waiting is None else min(max(waiting / WAITING_TIME_SCALE_S, 0.0), 1.0),
        None if signal_remaining is None else min(max(signal_remaining / context.signal_cycle_s, 0.0), 1.0),
    ]
    vector = [0.0 if value is None else value for value in normalized] + signal_values
    masks = [1.0 if value is not None else 0.0 for value in normalized]
    masks.extend([1.0 if signal_available else 0.0] * 3)
    vector.extend(masks)
    vector.extend([1.0, 0.0] if client_type == "vehicle" else [0.0, 1.0])
    if len(vector) != FEATURE_DIM or not all(math.isfinite(value) for value in vector):
        raise FeatureError("feature adapter produced an invalid vector")
    return vector


def _jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise FeatureError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def _target_windows(
    run_dir: Path, run_id: str, split: str, requested_split: Optional[str], horizon_s: float
):
    by_client = defaultdict(dict)
    for window in _jsonl(run_dir / "windows.jsonl"):
        if requested_split is not None and split != requested_split:
            continue
        if window.get("client", "").split("/", 1)[0] != run_id:
            raise FeatureError(f"window client crosses run boundary in {run_dir}: {window.get('client')!r}")
        if window.get("label_t") is None or abs(window["label_t"] - window["end_t"] - horizon_s) > 1e-6:
            raise FeatureError(f"window target is not aligned at end_t + {horizon_s:g} s in {run_dir}")
        by_client[window["client"]][round(float(window["end_t"]), 6)] = window
    return by_client


def iter_feature_windows(
    dataset_dir: Union[str, Path],
    split: Optional[str] = None,
    client_type: Optional[str] = None,
    client_ids: Optional[set[str]] = None,
    approach_filter: Optional[str] = None,
) -> Iterable[FeatureWindow]:
    """Stream windows built only from each agent's own consecutive logs and saved offline labels."""
    root = Path(dataset_dir)
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FeatureError(f"dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if split is not None and split not in manifest.get("splits", {}):
        raise FeatureError(f"unknown dataset split {split!r}")
    if client_type is not None and client_type not in ("vehicle", "rsu"):
        raise FeatureError(f"unknown client type {client_type!r}")
    if approach_filter is not None and approach_filter not in APPROACHES:
        raise FeatureError(f"unknown approach {approach_filter!r}")
    dataset_id = root.name
    for run_id, run_meta in sorted(manifest.get("runs", {}).items()):
        run_split = run_meta.get("split")
        if split is not None and run_split != split:
            continue
        run_dir = root / "runs" / run_id
        if not run_dir.is_dir():
            raise FeatureError(f"run directory missing for dataset manifest entry: {run_dir}")
        run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config = run_manifest["config"]
        topology = json.loads((run_dir / "scenario" / "topology.json").read_text(encoding="utf-8"))
        history_len = int(config["labels"]["history_len"])
        report_step = float(config["run"]["report_step_s"])
        targets = _target_windows(run_dir, run_id, run_split, split, float(config["labels"]["horizon_s"]))
        if client_ids is not None:
            targets = {client: labels for client, labels in targets.items() if client in client_ids}

        for approach in APPROACHES:
            if approach_filter is not None and approach != approach_filter:
                continue
            if client_type in (None, "rsu"):
                cid = f"{run_id}/rsu_{approach}"
                if client_ids is not None and cid not in client_ids:
                    continue
                target_map = targets.get(cid, {})
                if target_map:
                    ctx = FeatureContext.from_run(config, topology, approach)
                    records = _jsonl(run_dir / "agents" / f"rsu_{approach}.jsonl")
                    history = deque(maxlen=history_len)
                    previous_t = None
                    for record in records:
                        t = float(record["t"])
                        if previous_t is not None and abs(t - previous_t - report_step) > 1e-6:
                            history.clear()
                        previous_t = t
                        history.append(adapt_observation(record["view"], "rsu", ctx))
                        if len(history) == history_len and round(t, 6) in target_map:
                            label = target_map.pop(round(t, 6))
                            yield _make_window(dataset_id, run_id, run_split, run_meta, cid, "rsu",
                                               approach, t, ctx.zone_m, history, label,
                                               _read_number(record["view"], "queue_length_m", required=True),
                                               config["labels"]["congestion_thresholds"])
                    if target_map:
                        raise FeatureError(f"saved RSU windows could not be matched to observations: {cid}")

        if client_type in (None, "vehicle"):
            agents_dir = run_dir / "agents"
            for log_path in sorted(agents_dir.glob("veh_*.jsonl")):
                cid = f"{run_id}/{log_path.stem}"
                if client_ids is not None and cid not in client_ids:
                    continue
                client_meta = manifest.get("clients", {}).get(cid, {})
                if approach_filter is not None and client_meta.get("approach") != approach_filter:
                    continue
                target_map = targets.get(cid, {})
                if not target_map:
                    continue
                history = deque(maxlen=history_len)
                previous_t = None
                ctx = None
                approach = None
                for record in _jsonl(log_path):
                    view = record["view"]
                    t = float(record["t"])
                    if view.get("scope") != "incoming_local":
                        history.clear()
                        previous_t = t
                        continue
                    if previous_t is not None and abs(t - previous_t - report_step) > 1e-6:
                        history.clear()
                    previous_t = t
                    parent = record.get("parent")
                    if not isinstance(parent, str) or not parent.startswith("rsu_"):
                        raise FeatureError(f"vehicle record has invalid parent RSU: {cid}")
                    approach = parent.removeprefix("rsu_")
                    if approach not in APPROACHES:
                        raise FeatureError(f"vehicle record has unknown parent approach: {parent!r}")
                    if ctx is None:
                        ctx = FeatureContext.from_run(config, topology, approach)
                    history.append(adapt_observation(view, "vehicle", ctx))
                    if len(history) == history_len and round(t, 6) in target_map:
                        label = target_map.pop(round(t, 6))
                        yield _make_window(dataset_id, run_id, run_split, run_meta, cid, "vehicle",
                                           approach, t, ctx.zone_m, history, label, None,
                                           config["labels"]["congestion_thresholds"])
                if target_map:
                    raise FeatureError(f"saved vehicle windows could not be matched to observations: {cid}")


def _make_window(dataset_id, run_id, split, run_meta, cid, ctype, approach, end_t, zone_m,
                 history, label, current_queue_m, thresholds):
    queue_m = float(label["queue_m"])
    congestion_class = int(label["cls"])
    if not math.isfinite(queue_m) or queue_m < 0 or queue_m > zone_m + 1e-3:
        raise FeatureError(f"invalid future queue label for {cid} at t={end_t}: {queue_m}")
    if congestion_class not in (0, 1, 2):
        raise FeatureError(f"invalid congestion class for {cid} at t={end_t}: {congestion_class}")
    ratio = queue_m / zone_m
    expected_class = 0 if ratio < thresholds[0] else 1 if ratio < thresholds[1] else 2
    if congestion_class != expected_class:
        raise FeatureError(f"future queue/class mismatch for {cid} at t={end_t}")
    if len(history) != SEQUENCE_LENGTH:
        raise FeatureError(f"expected {SEQUENCE_LENGTH} timesteps, got {len(history)}")
    if current_queue_m is not None and (
        not math.isfinite(current_queue_m) or current_queue_m < 0 or current_queue_m > zone_m + 1e-3
    ):
        raise FeatureError(f"invalid local current queue observation for {cid} at t={end_t}")
    return FeatureWindow(dataset_id, run_id, split, run_meta.get("scenario", "unknown"), cid,
                         ctype, approach, float(end_t), float(zone_m), list(history),
                         current_queue_m, queue_m, congestion_class)


def audit_dataset(dataset_dir: Union[str, Path]) -> dict:
    """Audit feature availability, finite vectors, labels isolation and whole-run splits."""
    root = Path(dataset_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    split_overlap = []
    split_names = sorted(splits)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = sorted(set(splits[left]) & set(splits[right]))
            if overlap:
                split_overlap.append({"left": left, "right": right, "runs": overlap})

    summary = {
        "dataset_id": root.name,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "schema": FEATURE_SCHEMA,
        "counts": {"runs": 0, "records": {"rsu": 0, "vehicle": 0},
                   "incoming_local_vehicle_records": 0, "vehicle_records_outside_scope": 0},
        "available": {kind: {name: 0 for name in VALUE_NAMES} for kind in ("rsu", "vehicle")},
        "scanned": {"rsu": 0, "vehicle": 0},
        "errors": [],
        "split_overlap": split_overlap,
        "broadcast_values": [],
    }
    for run_id, run_meta in sorted(manifest.get("runs", {}).items()):
        run_dir = root / "runs" / run_id
        if not run_dir.is_dir():
            summary["errors"].append(f"missing run directory: {run_dir}")
            continue
        summary["counts"]["runs"] += 1
        run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config = run_manifest["config"]
        summary["broadcast_values"].append(bool(config["agents"]["signal_broadcast"]))
        topology = json.loads((run_dir / "scenario" / "topology.json").read_text(encoding="utf-8"))
        for approach in APPROACHES:
            ctx = FeatureContext.from_run(config, topology, approach)
            path = run_dir / "agents" / f"rsu_{approach}.jsonl"
            for line_number, record in enumerate(_jsonl(path), 1):
                summary["counts"]["records"]["rsu"] += 1
                summary["scanned"]["rsu"] += 1
                if FORBIDDEN_FEATURE_KEYS.intersection(record) or FORBIDDEN_FEATURE_KEYS.intersection(record.get("view", {})):
                    summary["errors"].append(f"label key in agent log: {path}:{line_number}")
                    continue
                try:
                    vector = adapt_observation(record["view"], "rsu", ctx)
                except FeatureError as exc:
                    if len(summary["errors"]) < 50:
                        summary["errors"].append(f"{path}:{line_number}: {exc}")
                    continue
                for index, name in enumerate(VALUE_NAMES):
                    summary["available"]["rsu"][name] += int(vector[10 + index] == 1.0)
        for path in sorted((run_dir / "agents").glob("veh_*.jsonl")):
            for line_number, record in enumerate(_jsonl(path), 1):
                summary["counts"]["records"]["vehicle"] += 1
                if FORBIDDEN_FEATURE_KEYS.intersection(record) or FORBIDDEN_FEATURE_KEYS.intersection(record.get("view", {})):
                    summary["errors"].append(f"label key in agent log: {path}:{line_number}")
                    continue
                if record.get("view", {}).get("scope") != "incoming_local":
                    summary["counts"]["vehicle_records_outside_scope"] += 1
                    continue
                summary["counts"]["incoming_local_vehicle_records"] += 1
                summary["scanned"]["vehicle"] += 1
                parent = record.get("parent")
                if not isinstance(parent, str) or not parent.startswith("rsu_") or parent[4:] not in APPROACHES:
                    summary["errors"].append(f"{path}:{line_number}: invalid vehicle parent {parent!r}")
                    continue
                ctx = FeatureContext.from_run(config, topology, parent[4:])
                try:
                    vector = adapt_observation(record["view"], "vehicle", ctx)
                except FeatureError as exc:
                    if len(summary["errors"]) < 50:
                        summary["errors"].append(f"{path}:{line_number}: {exc}")
                    continue
                for index, name in enumerate(VALUE_NAMES):
                    summary["available"]["vehicle"][name] += int(vector[10 + index] == 1.0)
    if len(set(summary["broadcast_values"])) > 1:
        summary["errors"].append("dataset mixes signal-broadcast settings")
    summary["broadcast_values"] = sorted(set(summary["broadcast_values"]))
    summary["availability_fraction"] = {
        kind: {
            name: count / summary["scanned"][kind] if summary["scanned"][kind] else None
            for name, count in values.items()
        }
        for kind, values in summary["available"].items()
    }
    summary["passed"] = not summary["errors"] and not summary["split_overlap"]
    return summary
