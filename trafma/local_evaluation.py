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
) -> Path:
    """Train/evaluate every eligible client, resuming completed client records."""
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
    if max_clients is not None:
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
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "minimum_client_windows": minimum,
        "eligible_client_count": eligible_total_count,
        "selected_client_count": len(eligible),
        "not_selected_due_to_limit": eligible_total_count - len(eligible),
        "client_limit": max_clients,
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
