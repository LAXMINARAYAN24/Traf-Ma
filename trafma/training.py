"""Reproducible CPU-first GRU training for permitted offline feature windows."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .baseline import BaselineError, _Metrics, load_learning_config
from .features import FEATURE_DIM, FEATURE_SCHEMA_HASH, SEQUENCE_LENGTH, FeatureWindow, iter_feature_windows
from .models import GRUPredictor, multitask_loss


class TrainingError(ValueError):
    """Raised when a configured training experiment cannot be completed safely."""


def _validate_training(config: dict) -> dict:
    training = config.get("training")
    if not isinstance(training, dict):
        raise TrainingError("learning config must define training settings")
    numeric = {"learning_rate": (0.0, None)}
    for key, (lower, upper) in numeric.items():
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TrainingError(f"training.{key} must be finite")
        if value <= lower or (upper is not None and value >= upper):
            raise TrainingError(f"training.{key} must be in ({lower}, {upper})")
    dropout = training.get("dropout")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout):
        raise TrainingError("training.dropout must be finite")
    if not 0.0 <= dropout < 1.0:
        raise TrainingError("training.dropout must be in [0, 1)")
    for key in ("regression_loss_weight", "classification_loss_weight"):
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise TrainingError(f"training.{key} must be finite and nonnegative")
    for key in ("seed", "hidden_size", "layers", "shared_size", "batch_size", "epochs",
                "early_stopping_patience", "max_train_windows", "max_validation_windows", "local_epochs"):
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TrainingError(f"training.{key} must be a positive integer")
    if training.get("device") != "cpu":
        raise TrainingError("the current Phase 2 training runner supports the configured CPU device only")
    if not isinstance(training.get("deterministic"), bool):
        raise TrainingError("training.deterministic must be boolean")
    if training["regression_loss_weight"] == 0 and training["classification_loss_weight"] == 0:
        raise TrainingError("at least one task loss weight must be positive")
    return training


def _seed_everything(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _sample_windows(dataset_dir: Path, split: str, maximum: int, seed: int,
                    client_type: Optional[str] = None,
                    client_ids: Optional[set[str]] = None,
                    approach: Optional[str] = None) -> list[FeatureWindow]:
    """Deterministic reservoir sample bounded in memory."""
    rng = random.Random(seed)
    selected: list[FeatureWindow] = []
    seen = 0
    for sample in iter_feature_windows(
        dataset_dir, split=split, client_type=client_type, client_ids=client_ids,
        approach_filter=approach,
    ):
        seen += 1
        if len(selected) < maximum:
            selected.append(sample)
        else:
            index = rng.randrange(seen)
            if index < maximum:
                selected[index] = sample
    return selected


def _to_dataset(samples: list[FeatureWindow]) -> TensorDataset:
    features = np.asarray([sample.features for sample in samples], dtype=np.float32)
    queue_ratios = np.asarray([sample.queue_m / sample.zone_m for sample in samples], dtype=np.float32)
    labels = np.asarray([sample.congestion_class for sample in samples], dtype=np.int64)
    return TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(queue_ratios),
        torch.from_numpy(labels),
    )


def _loss_for(model, batches, training, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"loss": 0.0, "regression": 0.0, "classification": 0.0, "samples": 0}
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for features, queue_ratio, labels in batches:
            outputs = model(features)
            loss, components = multitask_loss(
                outputs,
                queue_ratio,
                labels,
                regression_weight=float(training["regression_loss_weight"]),
                classification_weight=float(training["classification_loss_weight"]),
            )
            if not torch.isfinite(loss):
                raise TrainingError("training produced non-finite loss")
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise TrainingError("training produced non-finite gradients")
                optimizer.step()
            n = len(labels)
            totals["loss"] += float(loss.detach()) * n
            totals["regression"] += float(components["regression"].detach()) * n
            totals["classification"] += float(components["classification"].detach()) * n
            totals["samples"] += n
    if totals["samples"] == 0:
        raise TrainingError("training or validation split has no selected windows")
    return {name: totals[name] / totals["samples"] for name in ("loss", "regression", "classification")}


def _train_model(training, train_samples, val_samples):
    _seed_everything(int(training["seed"]), bool(training["deterministic"]))
    model = GRUPredictor(
        input_size=FEATURE_DIM,
        hidden_size=int(training["hidden_size"]),
        num_layers=int(training["layers"]),
        dropout=float(training["dropout"]),
        shared_size=int(training["shared_size"]),
    )
    train_loader = DataLoader(
        _to_dataset(train_samples),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(training["seed"])),
        num_workers=0,
    )
    val_loader = DataLoader(
        _to_dataset(val_samples),
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    history = []
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        train_loss = _loss_for(model, train_loader, training, optimizer)
        validation_loss = _loss_for(model, val_loader, training)
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in train_loss.items()},
                        **{f"validation_{key}": value for key, value in validation_loss.items()}})
        if validation_loss["loss"] < best_loss:
            best_loss = validation_loss["loss"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["early_stopping_patience"]):
                break
    if best_state is None:
        raise TrainingError("no finite validation checkpoint was produced")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def _prediction_metrics_from_saved(path: Path, requested_split: str) -> dict:
    metrics = defaultdict(_Metrics)
    run_metrics = defaultdict(_Metrics)
    with path.open(newline="", encoding="utf-8") as predictions:
        for row in csv.DictReader(predictions):
            if row["split"] != requested_split:
                continue
            groups = (
                ("overall", "all"),
                ("scenario", row["scenario"]),
                ("approach", row["approach"]),
                ("client_type", row["client_type"]),
                ("run", row["run_id"]),
            )
            true_queue = float(row["true_queue_m"])
            predicted_queue = float(row["pred_queue_m"])
            true_class = int(row["true_class"])
            predicted_class = int(row["pred_class"])
            for key in groups:
                metrics[key].add(true_queue, predicted_queue, true_class, predicted_class)
            run_metrics[row["run_id"]].add(true_queue, predicted_queue, true_class, predicted_class)
    return {
        "overall": metrics[("overall", "all")].result(),
        "by_scenario": {key: value.result() for (dim, key), value in metrics.items() if dim == "scenario"},
        "by_approach": {key: value.result() for (dim, key), value in metrics.items() if dim == "approach"},
        "by_client_type": {key: value.result() for (dim, key), value in metrics.items()
                           if dim == "client_type"},
        "by_run": {key: value.result() for (dim, key), value in metrics.items() if dim == "run"},
        "run_level_mae_m": {
            "mean": statistics.mean([value.result()["mae_m"] for value in run_metrics.values()]),
            "std": statistics.stdev([value.result()["mae_m"] for value in run_metrics.values()])
            if len(run_metrics) > 1 else 0.0,
            "runs": len(run_metrics),
        } if run_metrics else {"mean": None, "std": None, "runs": 0},
    }


def _predict(model, samples: list[FeatureWindow], batch_size: int):
    loader = DataLoader(_to_dataset(samples), batch_size=batch_size, shuffle=False, num_workers=0)
    predicted_ratios = []
    predicted_classes = []
    per_sample_latency = []
    model.eval()
    with torch.no_grad():
        for features, _queue_ratios, _labels in loader:
            started = time.perf_counter()
            output = model(features)
            per_sample_latency.append((time.perf_counter() - started) / len(features))
            predicted_ratios.extend(output["queue_ratio"].tolist())
            predicted_classes.extend(output["class_logits"].argmax(dim=1).tolist())
    predicted_queue = [
        min(max(ratio, 0.0), 1.0) * sample.zone_m
        for ratio, sample in zip(predicted_ratios, samples)
    ]
    return predicted_queue, predicted_classes, per_sample_latency


def _repo_state() -> dict:
    commit_result = subprocess.run(["git", "rev-parse", "--verify", "HEAD"],
                                   capture_output=True, text=True, check=False)
    status_result = subprocess.run(["git", "status", "--porcelain"],
                                   capture_output=True, text=True, check=False)
    return {
        "commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "worktree_dirty": bool(status_result.stdout.strip()) if status_result.returncode == 0 else None,
    }


def run_gru_training(config_path: Union[str, Path], method: str = "central",
                     dataset_override: Optional[str] = None,
                     client_id: Optional[str] = None) -> Path:
    """Train a central pooled GRU or one explicit offline trajectory-client GRU."""
    if method not in ("central", "local"):
        raise TrainingError("method must be central or local")
    if method == "local" and not client_id:
        raise TrainingError("local training requires --client with an eligible training client ID")
    if method == "central" and client_id:
        raise TrainingError("--client is only valid with --method local")
    config = load_learning_config(config_path)
    training = _validate_training(config)
    dataset_dir = Path(dataset_override or config["dataset"])
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise TrainingError(f"dataset manifest not found: {manifest_path}")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _seed_everything(int(training["seed"]), bool(training["deterministic"]))
    start_utc = dt.datetime.now(dt.timezone.utc)
    wall_started = time.perf_counter()
    if method == "local":
        client = dataset_manifest.get("clients", {}).get(client_id)
        if client is None or client.get("split") != "train" or not client.get("trainable"):
            raise TrainingError(f"local client must be an eligible training client: {client_id!r}")
        if client.get("windows", 0) < int(config["baseline"]["minimum_client_windows"]):
            raise TrainingError(f"local client is below the minimum-window eligibility rule: {client_id!r}")
        client_type = client["type"]
        approach = client["approach"]
        train_samples = _sample_windows(
            dataset_dir, "train", int(training["max_train_windows"]), int(training["seed"]),
            client_type=client_type, client_ids={client_id},
        )
        val_samples = _sample_windows(
            dataset_dir, "val", int(training["max_validation_windows"]), int(training["seed"]) + 1,
            client_type=client_type, approach=approach,
        )
        test_samples = _sample_windows(
            dataset_dir, "test", int(training["max_validation_windows"]), int(training["seed"]) + 2,
            client_type=client_type, approach=approach,
        )
    else:
        train_samples = _sample_windows(dataset_dir, "train", int(training["max_train_windows"]),
                                        int(training["seed"]))
        val_samples = _sample_windows(dataset_dir, "val", int(training["max_validation_windows"]),
                                      int(training["seed"]) + 1)
        test_samples = _sample_windows(dataset_dir, "test", int(training["max_validation_windows"]),
                                       int(training["seed"]) + 2)
    if not train_samples or not val_samples or not test_samples:
        raise TrainingError("train, validation and test splits must each contain sampled windows")
    if set(sample.run_id for sample in train_samples) & set(sample.run_id for sample in val_samples + test_samples):
        raise TrainingError("whole-run split leakage detected in selected GRU windows")

    train_started = time.perf_counter()
    model, history, best_epoch = _train_model(training, train_samples, val_samples)
    training_duration = time.perf_counter() - train_started
    output_root = Path(config["output_dir"])
    experiment_id = f"{method}_gru_{dataset_dir.name}_{start_utc.strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = output_root / experiment_id
    if output_dir.exists():
        raise TrainingError(f"experiment output already exists: {output_dir}")
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "checkpoints" / "gru.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": FEATURE_DIM,
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": training["hidden_size"],
            "num_layers": training["layers"],
            "dropout": training["dropout"],
            "shared_size": training["shared_size"],
        },
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "best_epoch": best_epoch,
    }, checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("feature_schema_hash") != FEATURE_SCHEMA_HASH:
        raise TrainingError("saved checkpoint feature schema does not match the current adapter")
    model_config = dict(checkpoint["model_config"])
    model_config.pop("sequence_length")
    loaded_model = GRUPredictor(**model_config)
    loaded_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    validation_queue, validation_classes, validation_latency = _predict(
        loaded_model, val_samples, int(training["batch_size"])
    )
    test_queue, test_classes, inference_times = _predict(
        loaded_model, test_samples, int(training["batch_size"])
    )
    config_resolved = {**config, "dataset": str(dataset_dir),
                       "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                       "feature_schema_hash": FEATURE_SCHEMA_HASH}
    config_hash = hashlib.sha256(json.dumps(config_resolved, sort_keys=True).encode("utf-8")).hexdigest()
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config_resolved, sort_keys=False),
                                                      encoding="utf-8")
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["split", "run_id", "scenario", "client_id", "client_type", "approach", "end_t",
                  "true_queue_m", "true_class", "pred_queue_m", "pred_class"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for split, samples, queues, classes in (
            ("val", val_samples, validation_queue, validation_classes),
            ("test", test_samples, test_queue, test_classes),
        ):
            for sample, queue, cls in zip(samples, queues, classes):
                writer.writerow({
                    "split": split, "run_id": sample.run_id, "scenario": sample.scenario,
                    "client_id": sample.client_id, "client_type": sample.client_type,
                    "approach": sample.approach, "end_t": sample.end_t,
                    "true_queue_m": sample.queue_m, "true_class": sample.congestion_class,
                    "pred_queue_m": queue, "pred_class": cls,
                })
    metrics = {
        split: _prediction_metrics_from_saved(output_dir / "predictions.csv", split)
        for split in ("val", "test")
    }
    repo = _repo_state()
    manifest = {
        "experiment_id": experiment_id,
        "method": ("central_pooled_offline_gru" if method == "central"
                   else "local_offline_trajectory_gru"),
        "local_training_client": client_id,
        "dataset_id": dataset_dir.name,
        "dataset_manifest_sha256": config_resolved["dataset_manifest_sha256"],
        "config_sha256": config_hash,
        "feature_schema_version": "trafma.features/1",
        "feature_schema_sha256": FEATURE_SCHEMA_HASH,
        "train_run_ids": sorted(dataset_manifest["splits"]["train"]),
        "validation_run_ids": sorted(dataset_manifest["splits"]["val"]),
        "test_run_ids": sorted(dataset_manifest["splits"]["test"]),
        "selected_window_counts": {"train": len(train_samples), "validation": len(val_samples),
                                   "test": len(test_samples)},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": {"processor": platform.processor(), "logical_cores": os.cpu_count()},
        "device": "cpu",
        "code_commit": repo["commit"],
        "code_worktree_dirty": repo["worktree_dirty"],
        "seed": training["seed"],
        "deterministic_algorithms": training["deterministic"],
        "checkpoint_sha256": checkpoint_hash,
        "best_epoch": best_epoch,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_wall_time_s": training_duration,
        "inference_latency_ms_mean": statistics.mean(inference_times) * 1000.0,
        "validation_inference_latency_ms_mean": statistics.mean(validation_latency) * 1000.0,
        "wall_duration_s": time.perf_counter() - wall_started,
        "experiment_started_utc": start_utc.isoformat(),
        "experiment_ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "metrics.csv").write_text(
        "split,method,group,n,mae_m,rmse_m,macro_f1,balanced_accuracy,accuracy\n"
        + "".join(",".join([
            split, manifest["method"], "overall",
            str(metrics[split]["overall"]["n"]),
            str(metrics[split]["overall"]["mae_m"]),
            str(metrics[split]["overall"]["rmse_m"]),
            str(metrics[split]["overall"]["macro_f1"]),
            str(metrics[split]["overall"]["balanced_accuracy"]),
            str(metrics[split]["overall"]["accuracy"]),
        ]) + "\n" for split in ("val", "test")),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        "\n".join([
            f"# {method.title()} GRU experiment: {experiment_id}",
            "",
            f"- Dataset: `{dataset_dir.name}`; feature schema hash `{FEATURE_SCHEMA_HASH}`.",
            f"- Train/validation/test sampled windows: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}.",
            f"- Best checkpoint epoch: {best_epoch}; parameter count: {manifest['parameters']}.",
            ("- Model is a pooled offline reference; it does not preserve client-local training."
             if method == "central" else
             f"- Model trained only on trajectory client `{client_id}`; test rows are held-out runs "
             f"matched on client type `{client_type}` and approach `{approach}`."),
            "- Metrics were recomputed from saved `predictions.csv`; test rows were not used for checkpoint selection.",
            "",
            "| Split | N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {split} | {metrics[split]['overall']['n']} | {metrics[split]['overall']['mae_m']} "
                f"| {metrics[split]['overall']['rmse_m']} | {metrics[split]['overall']['macro_f1']} "
                f"| {metrics[split]['overall']['balanced_accuracy']} |"
                for split in ("val", "test")
            ],
        ]) + "\n",
        encoding="utf-8",
    )
    return output_dir
