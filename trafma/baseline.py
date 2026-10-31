"""Reproducible, non-neural Phase 2 baselines and evaluation artifacts."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Union

import yaml

from .features import FEATURE_SCHEMA_HASH, iter_feature_windows


class BaselineError(ValueError):
    """Raised when an experiment config or dataset cannot support evaluation."""


def load_learning_config(path: Union[str, Path]) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise BaselineError(f"learning config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    baseline = config.get("baseline", {})
    alpha = baseline.get("ridge_alpha")
    minimum_windows = baseline.get("minimum_client_windows")
    if not isinstance(config.get("dataset"), str) or not config["dataset"]:
        raise BaselineError("learning config must define dataset")
    if not isinstance(config.get("output_dir"), str) or not config["output_dir"]:
        raise BaselineError("learning config must define output_dir")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha < 0:
        raise BaselineError("baseline.ridge_alpha must be a finite nonnegative number")
    if isinstance(minimum_windows, bool) or not isinstance(minimum_windows, int) or minimum_windows < 1:
        raise BaselineError("baseline.minimum_client_windows must be a positive integer")
    if baseline.get("method") != "pooled_signal_aware_ridge":
        raise BaselineError("baseline.method must be pooled_signal_aware_ridge")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BaselineError("learning seed must be an integer")
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> Optional[str]:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _repository_revision() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True, check=False
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
    except OSError:
        return {"commit": None, "worktree_dirty": None}
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _fit_ridge(dataset_dir: Path, train_clients: set[str], alpha: float):
    import numpy as np

    dimension = 23  # 22 ordered input features plus an intercept.
    xtx = np.zeros((dimension, dimension), dtype=np.float64)
    xty = np.zeros(dimension, dtype=np.float64)
    count = 0
    started = time.perf_counter()
    for sample in iter_feature_windows(dataset_dir, split="train"):
        if sample.client_id not in train_clients:
            continue
        x = np.asarray(sample.features[-1] + [1.0], dtype=np.float64)
        xtx += np.outer(x, x)
        xty += x * sample.queue_m
        count += 1
    if count == 0:
        raise BaselineError("no eligible training windows are available")
    regularizer = np.eye(dimension, dtype=np.float64) * alpha
    regularizer[-1, -1] = 0.0
    try:
        coefficients = np.linalg.solve(xtx + regularizer, xty)
    except np.linalg.LinAlgError as exc:
        raise BaselineError(f"ridge fit failed: {exc}") from exc
    return coefficients, count, time.perf_counter() - started


class _Metrics:
    def __init__(self):
        self.n = 0
        self.abs_error = 0.0
        self.squared_error = 0.0
        self.confusion = [[0, 0, 0] for _ in range(3)]

    def add(self, true_queue: float, pred_queue: float, true_class: int, pred_class: int):
        error = pred_queue - true_queue
        self.n += 1
        self.abs_error += abs(error)
        self.squared_error += error * error
        self.confusion[true_class][pred_class] += 1

    def result(self) -> dict:
        if self.n == 0:
            return {"n": 0, "mae_m": None, "rmse_m": None, "accuracy": None,
                    "macro_f1": None, "balanced_accuracy": None, "class_metrics": [],
                    "confusion_matrix": self.confusion}
        class_metrics = []
        recalls = []
        for cls in range(3):
            tp = self.confusion[cls][cls]
            support = sum(self.confusion[cls])
            predicted = sum(row[cls] for row in self.confusion)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            class_metrics.append({"class": cls, "support": support, "precision": precision,
                                 "recall": recall, "f1": f1})
            if support:
                recalls.append(recall)
        correct = sum(self.confusion[cls][cls] for cls in range(3))
        return {
            "n": self.n,
            "mae_m": self.abs_error / self.n,
            "rmse_m": math.sqrt(self.squared_error / self.n),
            "accuracy": correct / self.n,
            "macro_f1": sum(row["f1"] for row in class_metrics) / 3,
            "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
            "class_metrics": class_metrics,
            "confusion_matrix": self.confusion,
        }


def _class_for(queue_m: float, zone_m: float, thresholds: list[float]) -> int:
    ratio = queue_m / zone_m
    if ratio < thresholds[0]:
        return 0
    if ratio < thresholds[1]:
        return 1
    return 2


def _add_group(groups: dict, key: tuple, true_queue: float, pred_queue: float,
               true_class: int, pred_class: int):
    accumulator = groups.setdefault(key, _Metrics())
    accumulator.add(true_queue, pred_queue, true_class, pred_class)


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "runs": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "runs": len(values),
    }


def _write_predictions(dataset_dir: Path, split: str, coefficients, thresholds: list[float],
                       majority_class: int, writer):
    import numpy as np

    for sample in iter_feature_windows(dataset_dir, split=split):
        true_class = sample.congestion_class
        x = np.asarray(sample.features[-1] + [1.0], dtype=np.float64)
        signal_prediction = min(max(float(x @ coefficients), 0.0), sample.zone_m)
        signal_class = _class_for(signal_prediction, sample.zone_m, thresholds)
        persistence_prediction = sample.current_queue_m
        if persistence_prediction is not None:
            persistence_class = _class_for(persistence_prediction, sample.zone_m, thresholds)
        else:
            persistence_class = None
        writer.writerow({
            "split": split,
            "run_id": sample.run_id,
            "scenario": sample.scenario,
            "client_id": sample.client_id,
            "client_type": sample.client_type,
            "approach": sample.approach,
            "end_t": sample.end_t,
            "true_queue_m": sample.queue_m,
            "true_class": true_class,
            "persistence_pred_queue_m": "" if persistence_prediction is None else persistence_prediction,
            "persistence_pred_class": "" if persistence_class is None else persistence_class,
            "signal_aware_pred_queue_m": signal_prediction,
            "signal_aware_pred_class": signal_class,
            "majority_pred_class": majority_class,
        })


def _metrics_from_predictions(predictions_path: Path) -> dict:
    """Compute every reported metric from the persisted prediction rows."""
    metrics = {
        split: {method: defaultdict(_Metrics) for method in ("persistence", "signal_aware_ridge")}
        for split in ("val", "test")
    }
    majority_metrics = {split: defaultdict(_Metrics) for split in ("val", "test")}
    per_run = defaultdict(
        lambda: defaultdict(
            lambda: {method: defaultdict(_Metrics) for method in ("persistence", "signal_aware_ridge")}
        )
    )
    with predictions_path.open(newline="", encoding="utf-8") as predictions:
        for row in csv.DictReader(predictions):
            split = row["split"]
            true_queue = float(row["true_queue_m"])
            true_class = int(row["true_class"])
            signal_prediction = float(row["signal_aware_pred_queue_m"])
            signal_class = int(row["signal_aware_pred_class"])
            majority_class = int(row["majority_pred_class"])
            groups = (
                ("overall", "all"),
                ("scenario", row["scenario"]),
                ("approach", row["approach"]),
                ("client_type", row["client_type"]),
                ("run", row["run_id"]),
            )
            for group_key in groups:
                _add_group(metrics[split]["signal_aware_ridge"], group_key,
                           true_queue, signal_prediction, true_class, signal_class)
                _add_group(majority_metrics[split], group_key, true_queue, true_queue,
                           true_class, majority_class)
            _add_group(per_run[split][row["run_id"]]["signal_aware_ridge"], "all",
                       true_queue, signal_prediction, true_class, signal_class)

            if row["persistence_pred_queue_m"]:
                persistence_prediction = float(row["persistence_pred_queue_m"])
                persistence_class = int(row["persistence_pred_class"])
                for group_key in groups:
                    _add_group(metrics[split]["persistence"], group_key,
                               true_queue, persistence_prediction, true_class, persistence_class)
                _add_group(per_run[split][row["run_id"]]["persistence"], "all",
                           true_queue, persistence_prediction, true_class, persistence_class)

    def class_only(value):
        metrics_result = value.result()
        metrics_result.pop("mae_m")
        metrics_result.pop("rmse_m")
        return metrics_result

    result = {}
    for split in metrics:
        result[split] = {}
        for method in metrics[split]:
            result[split][method] = {
                "overall": metrics[split][method][("overall", "all")].result(),
                "by_scenario": {key: value.result()
                                for (dim, key), value in metrics[split][method].items()
                                if dim == "scenario"},
                "by_approach": {key: value.result()
                                for (dim, key), value in metrics[split][method].items()
                                if dim == "approach"},
                "by_client_type": {key: value.result()
                                   for (dim, key), value in metrics[split][method].items()
                                   if dim == "client_type"},
                "by_run": {key: value.result()
                           for (dim, key), value in metrics[split][method].items() if dim == "run"},
                "run_level_mae_m": _mean_std([
                    per_run[split][run_id][method]["all"].result()["mae_m"]
                    for run_id in sorted(per_run[split])
                    if per_run[split][run_id][method]["all"].result()["mae_m"] is not None
                ]),
            }
        result[split]["majority_class"] = {
            "overall": class_only(majority_metrics[split][("overall", "all")]),
            "by_scenario": {
                key: class_only(value) for (dim, key), value in majority_metrics[split].items()
                if dim == "scenario"
            },
            "by_approach": {
                key: class_only(value) for (dim, key), value in majority_metrics[split].items()
                if dim == "approach"
            },
            "by_client_type": {
                key: class_only(value) for (dim, key), value in majority_metrics[split].items()
                if dim == "client_type"
            },
            "by_run": {
                key: class_only(value) for (dim, key), value in majority_metrics[split].items()
                if dim == "run"
            },
        }
    return result


def run_baselines(config_path: Union[str, Path], dataset_override: Optional[str] = None) -> Path:
    """Fit the train-only signal-aware ridge reference and save validation/test artifacts."""
    config = load_learning_config(config_path)
    dataset_dir = Path(dataset_override or config["dataset"])
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise BaselineError(f"dataset manifest not found: {manifest_path}")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_names = set(dataset_manifest.get("splits", {}))
    if not {"train", "val", "test"} <= split_names:
        raise BaselineError("baseline dataset must have disjoint train, val and test run splits")
    overlap = set(dataset_manifest["splits"]["train"]) & (
        set(dataset_manifest["splits"]["val"]) | set(dataset_manifest["splits"]["test"])
    )
    overlap |= set(dataset_manifest["splits"]["val"]) & set(dataset_manifest["splits"]["test"])
    if overlap:
        raise BaselineError(f"dataset split runs overlap: {sorted(overlap)}")

    minimum_windows = config["baseline"]["minimum_client_windows"]
    run_splits = {name: sorted(values) for name, values in dataset_manifest["splits"].items()}
    train_clients = {
        cid for cid, client in dataset_manifest["clients"].items()
        if client["split"] == "train" and client["windows"] >= minimum_windows
    }
    thresholds = None
    for run_id in sorted(set().union(*[set(value) for value in run_splits.values()])):
        run_manifest = json.loads((dataset_dir / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        current_thresholds = run_manifest["config"]["labels"]["congestion_thresholds"]
        if thresholds is None:
            thresholds = current_thresholds
        elif thresholds != current_thresholds:
            raise BaselineError("dataset runs use inconsistent congestion thresholds")
    training_label_hist = [
        sum(dataset_manifest["clients"][client_id]["label_hist"][label] for client_id in train_clients)
        for label in range(3)
    ]
    if not any(training_label_hist):
        raise BaselineError("eligible training clients have no class labels")
    majority_class = max(range(3), key=lambda label: training_label_hist[label])
    start_utc = dt.datetime.now(dt.timezone.utc)
    coefficients, training_samples, training_time_s = _fit_ridge(
        dataset_dir, train_clients, float(config["baseline"]["ridge_alpha"])
    )

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_id = f"baseline_{dataset_dir.name}_{stamp}"
    output_root = Path(config["output_dir"])
    output_dir = output_root / experiment_id
    if output_dir.exists():
        raise BaselineError(f"experiment output already exists: {output_dir}")
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "predictions.csv"
    csv_fields = [
        "split", "run_id", "scenario", "client_id", "client_type", "approach", "end_t",
        "true_queue_m", "true_class", "persistence_pred_queue_m", "persistence_pred_class",
        "signal_aware_pred_queue_m", "signal_aware_pred_class", "majority_pred_class",
    ]
    with prediction_path.open("w", newline="", encoding="utf-8") as predictions:
        writer = csv.DictWriter(predictions, fieldnames=csv_fields)
        writer.writeheader()
        for split in ("val", "test"):
            _write_predictions(dataset_dir, split, coefficients, thresholds, majority_class, writer)
    results = _metrics_from_predictions(prediction_path)

    (output_dir / "checkpoints" / "signal_aware_ridge.json").write_text(
        json.dumps({
            "feature_schema_version": "trafma.features/1",
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "features": "last timestep's 22-value feature vector plus intercept",
            "ridge_alpha": float(config["baseline"]["ridge_alpha"]),
            "eligible_training_clients": len(train_clients),
            "distinct_training_windows": training_samples,
            "coefficients": coefficients.tolist(),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "checkpoints" / "majority_class.json").write_text(json.dumps({
        "method": "train_only_majority_class",
        "class": majority_class,
        "training_label_histogram": training_label_hist,
        "eligible_training_clients": len(train_clients),
    }, indent=2, sort_keys=True), encoding="utf-8")
    end_utc = dt.datetime.now(dt.timezone.utc)
    dataset_hash = _sha256_file(manifest_path)
    config_resolved = {
        **config,
        "dataset": str(dataset_dir),
        "dataset_manifest_sha256": dataset_hash,
        "dataset_definitions": dataset_manifest["definitions"],
        "dataset_splits": run_splits,
        "feature_schema_version": "trafma.features/1",
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "aggregation_policy": "overlapping windows are samples; simulation-run summaries are reported separately",
    }
    config_hash = hashlib.sha256(
        json.dumps(config_resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    resolved_config = yaml.safe_dump(config_resolved, sort_keys=False)
    (output_dir / "resolved_config.yaml").write_text(resolved_config, encoding="utf-8")
    package_versions = {
        name: _package_version(name)
        for name in ("numpy", "scikit-learn", "torch", "eclipse-sumo", "traci", "sumolib", "pyyaml")
    }
    run_versions = []
    for run_id in sorted(set().union(*[set(value) for value in run_splits.values()])):
        run_manifest = json.loads((dataset_dir / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
        run_versions.append(run_manifest.get("versions", {}))
    dataset_versions = run_versions[0] if run_versions else {}
    dataset_versions_consistent = all(versions == dataset_versions for versions in run_versions)
    repository_revision = _repository_revision()
    experiment_manifest = {
        "experiment_id": experiment_id,
        "method": "non-neural baselines",
        "offline_supervision": "saved evaluator labels joined only for training/evaluation, never feature inputs",
        "dataset_id": dataset_dir.name,
        "dataset_manifest_sha256": dataset_hash,
        "config_sha256": config_hash,
        "train_run_ids": run_splits["train"],
        "validation_run_ids": run_splits["val"],
        "test_run_ids": run_splits["test"],
        "feature_schema_version": "trafma.features/1",
        "feature_schema_sha256": FEATURE_SCHEMA_HASH,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": {"processor": platform.processor(), "machine": platform.machine(),
                "logical_cores": os.cpu_count()},
        "gpu": "not used by this baseline",
        "code_commit": repository_revision["commit"],
        "code_worktree_dirty": repository_revision["worktree_dirty"],
        "dataset_simulator_versions": dataset_versions,
        "dataset_simulator_versions_consistent": dataset_versions_consistent,
        "packages": package_versions,
        "experiment_started_utc": start_utc.isoformat(),
        "experiment_ended_utc": end_utc.isoformat(),
        "wall_duration_s": (end_utc - start_utc).total_seconds(),
        "training": {
            "eligible_clients": len(train_clients),
            "minimum_windows_per_client": minimum_windows,
            "distinct_windows": training_samples,
            "fit_wall_time_s": training_time_s,
            "signal_aware_method": "train-only scalar Ridge regression over local features; class derived from predicted queue using fixed thresholds",
            "majority_class": majority_class,
            "majority_class_histogram": training_label_hist,
            "persistence_coverage": "RSU rows only; vehicle persistence is unavailable because the vehicle log has no local queue-length observation",
        },
        "seed": config["seed"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(experiment_manifest, indent=2, sort_keys=True),
                                               encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=["method", "windows", "fit_wall_time_s", "ridge_alpha"])
        writer.writeheader()
        writer.writerow({"method": "signal_aware_ridge", "windows": training_samples,
                         "fit_wall_time_s": training_time_s, "ridge_alpha": config["baseline"]["ridge_alpha"]})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics_csv:
        writer = csv.DictWriter(metrics_csv, fieldnames=["split", "method", "group", "n", "mae_m",
                                                         "rmse_m", "macro_f1", "balanced_accuracy", "accuracy"])
        writer.writeheader()
        for split, methods in results.items():
            for method, content in methods.items():
                for group, values in (("overall", content["overall"]), *[
                    (f"{dimension}:{key}", metric)
                    for dimension in ("by_scenario", "by_approach", "by_client_type", "by_run")
                    for key, metric in content[dimension].items()
                ]):
                    writer.writerow({"split": split, "method": method, "group": group,
                                     "n": values["n"], "mae_m": values.get("mae_m"),
                                     "rmse_m": values.get("rmse_m"), "macro_f1": values["macro_f1"],
                                     "balanced_accuracy": values["balanced_accuracy"],
                                     "accuracy": values["accuracy"]})
    summary = [
        f"# Baseline experiment: {experiment_id}",
        "",
        f"- Dataset: `{dataset_dir.name}` (manifest SHA-256 `{dataset_hash}`)",
        f"- Feature schema: `trafma.features/1` (SHA-256 `{FEATURE_SCHEMA_HASH}`)",
        f"- Train-only Ridge windows: {training_samples} from {len(train_clients)} eligible clients.",
        "- Persistence uses each RSU's observed current queue; vehicle persistence is unavailable.",
        "- Ridge predictions use only local features from the final input timestep; predicted class uses the frozen 0.10/0.35 queue-to-zone thresholds.",
        "- Windows overlap; overall window metrics are descriptive. Run-level MAE mean and standard deviation are also saved.",
        "",
        "| Split | Method | N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, methods in results.items():
        for method, content in methods.items():
            metric = content["overall"]
            summary.append(
                f"| {split} | {method} | {metric['n']} | {metric.get('mae_m', 'n/a')} "
                f"| {metric.get('rmse_m', 'n/a')} "
                f"| {metric['macro_f1']} | {metric['balanced_accuracy']} |"
            )
    summary.extend([
        "",
        "## Matched RSU-only comparison",
        "",
        "Both methods below use the same held-out RSU windows. Persistence is not available on vehicle rows.",
        "",
        "| Split | Method | N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for split, methods in results.items():
        for method, content in methods.items():
            metric = content["by_client_type"].get("rsu")
            if metric is not None:
                summary.append(
                    f"| {split} | {method} | {metric['n']} | {metric.get('mae_m', 'n/a')} "
                    f"| {metric.get('rmse_m', 'n/a')} "
                    f"| {metric['macro_f1']} | {metric['balanced_accuracy']} |"
                )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return output_dir
