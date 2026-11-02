"""Resumable population-wide local-model evaluation."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Union

import yaml

from .baseline import BaselineError, load_learning_config
from .training import TrainingError, run_gru_training


class LocalEvaluationError(ValueError):
    """Raised when the all-client local evaluation cannot be initialized."""


_WINDOW_BANDS = (
    ("16-31", 16, 31),
    ("32-63", 32, 63),
    ("64-127", 64, 127),
    ("128+", 128, None),
)


def _window_band(windows: int) -> str:
    for name, lower, upper in _WINDOW_BANDS:
        if windows >= lower and (upper is None or windows <= upper):
            return name
    raise LocalEvaluationError(f"eligible client has invalid window count: {windows}")


def _select_representative_clients(
    eligible: list[tuple[str, dict]], count: int, seed: int
) -> list[tuple[str, dict]]:
    """Select a deterministic, coverage-oriented client sample."""
    if count < 1:
        raise LocalEvaluationError("representative_clients must be a positive integer")
    if count >= len(eligible):
        return list(eligible)

    import random

    candidates = sorted(eligible)
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected_ids: set[str] = set()

    dimensions = (
        ("type", lambda client: str(client.get("type"))),
        ("approach", lambda client: str(client.get("approach"))),
        ("scenario", lambda client: str(client.get("scenario"))),
        ("window_band", lambda client: _window_band(int(client["windows"]))),
    )
    for _name, value_for in dimensions:
        for value in sorted({value_for(client) for _, client in eligible}):
            for client_id, client in candidates:
                if client_id not in selected_ids and value_for(client) == value:
                    selected_ids.add(client_id)
                    break

    buckets: dict[tuple[str, str, str, str], list[tuple[str, dict]]] = defaultdict(list)
    for client_id, client in candidates:
        if client_id not in selected_ids:
            buckets[(
                str(client.get("type")),
                str(client.get("approach")),
                str(client.get("scenario")),
                _window_band(int(client["windows"])),
            )].append((client_id, client))
    bucket_keys = list(buckets)
    rng.shuffle(bucket_keys)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    while len(selected_ids) < count:
        added = False
        for bucket_key in bucket_keys:
            bucket = buckets[bucket_key]
            if bucket:
                client_id, _client = bucket.pop()
                selected_ids.add(client_id)
                added = True
                if len(selected_ids) == count:
                    break
        if not added:
            break

    selected = [item for item in eligible if item[0] in selected_ids]
    return sorted(selected)


def _selection_strata(eligible: list[tuple[str, dict]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _client_id, client in eligible:
        key = "|".join((
            str(client.get("type")),
            str(client.get("approach")),
            str(client.get("scenario")),
            _window_band(int(client["windows"])),
        ))
        counts[key] += 1
    return dict(sorted(counts.items()))


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise LocalEvaluationError("cannot calculate a percentile for an empty group")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(rows: list[dict]) -> dict:
    metrics = [float(row["mae_m"]) for row in rows if row.get("status") == "completed"]
    if not metrics:
        return {"clients": 0, "median_mae_m": None, "iqr_mae_m": None, "worst_decile_mae_m": None}
    return {
        "clients": len(metrics),
        "median_mae_m": statistics.median(metrics),
        "iqr_mae_m": _percentile(metrics, 0.75) - _percentile(metrics, 0.25),
        "worst_decile_mae_m": _percentile(metrics, 0.9),
    }


def _read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LocalEvaluationError(f"invalid client record at {path}:{line_number}") from exc
            client_id = row.get("client_id")
            if not isinstance(client_id, str) or not client_id:
                raise LocalEvaluationError(f"client record has no client_id at {path}:{line_number}")
            records.append(row)
    return records


def _write_summary(output_dir: Path, manifest: dict, rows: list[dict], config_hash: str) -> None:
    completed = [row for row in rows if row.get("status") == "completed"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in completed:
        for dimension in ("client_type", "approach", "scenario"):
            grouped[f"{dimension}:{row[dimension]}"].append(row)
    summary = {
        "dataset_id": manifest.get("dataset_id"),
        "minimum_client_windows": manifest.get("minimum_client_windows"),
        "eligible_client_count": manifest.get("eligible_client_count"),
        "selected_client_count": manifest.get("selected_client_count"),
        "not_selected_due_to_limit": manifest.get("not_selected_due_to_limit"),
        "client_limit": manifest.get("client_limit"),
        "evaluation_mode": manifest.get("evaluation_mode"),
        "selection_seed": manifest.get("selection_seed"),
        "representative_clients": manifest.get("representative_clients"),
        "selected_client_ids": manifest.get("selected_client_ids"),
        "selection_strata": manifest.get("selection_strata"),
        "ineligible_client_count": manifest.get("ineligible_client_count"),
        "completed_client_count": len(completed),
        "failed_client_count": sum(row.get("status") == "failed" for row in rows),
        "config_sha256": config_hash,
        "distributions": {
            key: _distribution(value) for key, value in sorted(grouped.items())
        },
        "ineligible_reasons": dict(Counter(
            row.get("reason", "unknown") for row in rows if row.get("status") == "ineligible"
        )),
        "failed_reasons": dict(Counter(
            row.get("error_type", "unknown") for row in rows if row.get("status") == "failed"
        )),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["client_id", "client_type", "approach", "scenario", "status",
                  "train_windows", "test_n", "mae_m", "rmse_m", "macro_f1", "balanced_accuracy",
                  "artifact", "reason", "error"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_broad_local_evaluation(
    config_path: Union[str, Path],
    dataset_override: Optional[Union[str, Path]] = None,
    output: Optional[Union[str, Path]] = None,
    training_seed: Optional[int] = None,
    max_clients: Optional[int] = None,
    resume: bool = False,
    device: Optional[str] = None,
    selection_mode: str = "exhaustive",
    selection_seed: int = 20261101,
    representative_clients: int = 256,
) -> Path:
    """Evaluate eligible clients exhaustively or with a representative stratified sample."""
    config = load_learning_config(config_path)
    dataset_dir = Path(dataset_override or config["dataset"])
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise LocalEvaluationError(f"dataset manifest not found: {manifest_path}")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    minimum = int(config["baseline"]["minimum_client_windows"])
    clients = dataset_manifest.get("clients", {})
    if not isinstance(clients, dict):
        raise LocalEvaluationError("dataset manifest clients must be an object")

    output_dir = Path(output) if output else Path(config["output_dir"]) / (
        f"local_population_{dataset_dir.name}_"
        f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if output_dir.exists():
        if not resume:
            raise LocalEvaluationError(f"evaluation output already exists: {output_dir}")
        if not output_dir.is_dir():
            raise LocalEvaluationError(f"evaluation output is not a directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    records_path = output_dir / "clients.jsonl"

    eligible = []
    rows = []
    for client_id, client in sorted(clients.items()):
        if client.get("split") != "train":
            continue
        windows = int(client.get("windows", 0))
        if client.get("trainable") and windows >= minimum:
            eligible.append((client_id, client))
        else:
            rows.append({
                "client_id": client_id,
                "client_type": client.get("type"),
                "approach": client.get("approach"),
                "scenario": client.get("scenario"),
                "status": "ineligible",
                "train_windows": windows,
                "reason": (
                    "not_trainable" if not client.get("trainable")
                    else f"fewer_than_{minimum}_windows"
                ),
            })

    eligible_total_count = len(eligible)
    if selection_mode not in ("exhaustive", "stratified"):
        raise LocalEvaluationError("selection_mode must be 'exhaustive' or 'stratified'")
    if selection_mode == "stratified":
        if max_clients is not None:
            raise LocalEvaluationError(
                "max_clients cannot be combined with stratified selection; use representative_clients"
            )
        if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
            raise LocalEvaluationError("selection_seed must be an integer")
        if isinstance(representative_clients, bool) or representative_clients < 1:
            raise LocalEvaluationError("representative_clients must be a positive integer")
        eligible = _select_representative_clients(
            eligible, representative_clients, selection_seed
        )
    elif max_clients is not None:
        if isinstance(max_clients, bool) or max_clients < 1:
            raise LocalEvaluationError("max_clients must be a positive integer")
        eligible = eligible[:max_clients]

    config_for_clients = dict(config)
    config_for_clients["dataset"] = str(dataset_dir)
    config_for_clients["output_dir"] = str(models_dir)
    config_for_clients["training"] = dict(config["training"])
    if training_seed is not None:
        config_for_clients["training"]["seed"] = training_seed
    if device is not None:
        if device not in ("cpu", "cuda"):
            raise LocalEvaluationError("device must be 'cpu' or 'cuda'")
        config_for_clients["training"]["device"] = device
    config_path_for_clients = output_dir / "local_training_config.yaml"
    config_path_for_clients.write_text(
        yaml.safe_dump(config_for_clients, sort_keys=False), encoding="utf-8"
    )
    config_hash = hashlib.sha256(config_path_for_clients.read_bytes()).hexdigest()

    existing_rows = _read_records(records_path)
    existing_manifest_path = output_dir / "manifest.json"
    if resume and existing_manifest_path.is_file():
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        prior_mode = existing_manifest.get("evaluation_mode", "exhaustive_local_only")
        current_mode = (
            "representative_stratified_local_only"
            if selection_mode == "stratified"
            else "exhaustive_local_only"
        )
        if prior_mode != current_mode:
            raise LocalEvaluationError(
                f"resume selection mode mismatch: output uses {prior_mode}, requested {current_mode}"
            )
        prior_ids = existing_manifest.get("selected_client_ids")
        current_ids = [client_id for client_id, _client in eligible]
        if prior_ids is not None and prior_ids != current_ids:
            raise LocalEvaluationError(
                "resume client selection mismatch; keep selection mode, seed, and client count unchanged"
            )
    existing = {
        row["client_id"]: row for row in existing_rows
        if row.get("status") == "completed"
    }
    if existing_rows:
        rows.extend(existing_rows)
    row_by_client = {row["client_id"]: row for row in rows}
    with records_path.open("a", encoding="utf-8") as records:
        if not existing_rows:
            for row in rows:
                records.write(json.dumps(row, sort_keys=True) + "\n")
            records.flush()
        for client_id, client in eligible:
            if client_id in existing:
                continue
            try:
                artifact = run_gru_training(
                    config_path_for_clients,
                    method="local",
                    dataset_override=str(dataset_dir),
                    client_id=client_id,
                    training_seed=training_seed,
                )
                metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
                test = metrics["test"]["overall"]
                row = {
                    "client_id": client_id,
                    "client_type": client["type"],
                    "approach": client["approach"],
                    "scenario": client["scenario"],
                    "status": "completed",
                    "train_windows": int(client["windows"]),
                    "test_n": test["n"],
                    "mae_m": test["mae_m"],
                    "rmse_m": test["rmse_m"],
                    "macro_f1": test["macro_f1"],
                    "balanced_accuracy": test["balanced_accuracy"],
                    "artifact": str(artifact),
                }
            except (BaselineError, TrainingError, OSError, KeyError, json.JSONDecodeError) as exc:
                row = {
                    "client_id": client_id,
                    "client_type": client["type"],
                    "approach": client["approach"],
                    "scenario": client["scenario"],
                    "status": "failed",
                    "train_windows": int(client["windows"]),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            records.write(json.dumps(row, sort_keys=True) + "\n")
            records.flush()
            row_by_client[client_id] = row

    rows = list(row_by_client.values())
    ineligible_count = sum(row.get("status") == "ineligible" for row in rows)
    summary_manifest = {
        "dataset_id": dataset_dir.name,
        "evaluation_mode": (
            "representative_stratified_local_only"
            if selection_mode == "stratified"
            else "exhaustive_local_only"
        ),
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "minimum_client_windows": minimum,
        "eligible_client_count": eligible_total_count,
        "selected_client_count": len(eligible),
        "not_selected_due_to_limit": eligible_total_count - len(eligible),
        "client_limit": max_clients,
        "selection_seed": selection_seed if selection_mode == "stratified" else None,
        "representative_clients": (
            representative_clients if selection_mode == "stratified" else None
        ),
        "selected_client_ids": [client_id for client_id, _client in eligible],
        "selection_strata": _selection_strata(eligible),
        "ineligible_client_count": ineligible_count,
        "training_seed": training_seed if training_seed is not None else config["training"]["seed"],
        "config_sha256": config_hash,
        "experiment_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_summary(output_dir, summary_manifest, rows, config_hash)
    return output_dir
