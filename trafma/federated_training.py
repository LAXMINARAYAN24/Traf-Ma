"""Reproducible offline flat FedAvg experiments over saved trajectory clients."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .baseline import load_learning_config
from .features import FEATURE_DIM, FEATURE_SCHEMA_HASH, SEQUENCE_LENGTH, FeatureWindow, iter_feature_windows
from .federation import ModelUpdate, aggregate_flat, aggregate_hierarchical, model_hash
from .models import GRUPredictor
from .training import (
    TrainingError,
    _loss_for,
    _prediction_metrics_from_saved,
    _predict,
    _repo_state,
    _sample_windows,
    _seed_everything,
    _to_dataset,
    _validate_training,
)


def _client_tensor_map(model) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in model.state_dict().items()
    }


def _select_clients(manifest: dict, max_clients: int, minimum_windows: int, seed: int) -> list[str]:
    eligible = sorted(
        client_id for client_id, client in manifest.get("clients", {}).items()
        if client.get("split") == "train"
        and client.get("trainable")
        and client.get("windows", 0) >= minimum_windows
    )
    if len(eligible) < 2:
        raise TrainingError("flat FedAvg requires at least two eligible training clients")
    selected_count = min(max_clients, len(eligible))
    selected = random.Random(seed).sample(eligible, selected_count)
    return sorted(selected)


def _select_hierarchical_vehicles(manifest: dict, max_clients: int, minimum_windows: int,
                                 seed: int) -> list[str]:
    eligible_by_approach = {approach: [] for approach in ("N", "E", "S", "W")}
    for client_id, client in manifest.get("clients", {}).items():
        if (client.get("split") == "train" and client.get("trainable")
                and client.get("type") == "vehicle"
                and client.get("windows", 0) >= minimum_windows):
            eligible_by_approach[client["approach"]].append(client_id)
    rng = random.Random(seed)
    for clients in eligible_by_approach.values():
        rng.shuffle(clients)
    selected = []
    while len(selected) < max_clients:
        added = False
        for approach in ("N", "E", "S", "W"):
            if eligible_by_approach[approach]:
                selected.append(eligible_by_approach[approach].pop())
                added = True
                if len(selected) == max_clients:
                    break
        if not added:
            break
    if len(selected) < 2:
        raise TrainingError("hierarchical FedAvg requires at least two eligible vehicle clients")
    return sorted(selected)


def _collect_client_windows(dataset_dir: Path, source_ids: list[str], destination_ids: list[str],
                            source_to_destination: dict[str, str], maximum: Optional[int],
                            seed: int) -> tuple[dict[str, list[FeatureWindow]], dict[str, int]]:
    selected = {client_id: [] for client_id in destination_ids}
    seen = {client_id: 0 for client_id in destination_ids}
    rngs = {
        client_id: random.Random(int.from_bytes(
            hashlib.sha256(f"{seed}:{client_id}".encode()).digest()[:4], "big"
        ))
        for client_id in destination_ids
    }
    for sample in iter_feature_windows(
        dataset_dir, split="train", client_ids=set(source_ids)
    ):
        client_id = source_to_destination[sample.client_id]
        seen[client_id] += 1
        if maximum is None or len(selected[client_id]) < maximum:
            selected[client_id].append(sample)
        else:
            index = rngs[client_id].randrange(seen[client_id])
            if index < maximum:
                selected[client_id][index] = sample
    return selected, seen


def _partition_iid_windows(samples: list[FeatureWindow], client_count: int,
                           seed: int) -> dict[str, list[FeatureWindow]]:
    partitions = {f"iid_client_{index:03d}": [] for index in range(client_count)}
    by_class = {label: [] for label in range(3)}
    rng = random.Random(seed)
    for sample in samples:
        by_class[sample.congestion_class].append(sample)
    for rows in by_class.values():
        rng.shuffle(rows)
        for index, sample in enumerate(rows):
            partitions[f"iid_client_{index % client_count:03d}"].append(sample)
    return partitions


def _client_training(model_config: dict, base_state: dict, samples: list[FeatureWindow],
                     training: dict, seed: int) -> tuple[dict[str, np.ndarray], float, float]:
    _seed_everything(seed, bool(training["deterministic"]))
    model = GRUPredictor(**model_config)
    model.load_state_dict(base_state, strict=True)
    dataset = _to_dataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    started = time.perf_counter()
    loss = None
    for _ in range(int(training["local_epochs"])):
        loss = _loss_for(model, loader, training, optimizer)["loss"]
    if loss is None:
        raise TrainingError("local training completed without an epoch")
    return _client_tensor_map(model), float(loss), time.perf_counter() - started


def _write_predictions(path: Path, model, samples: list[FeatureWindow], batch_size: int,
                       split: str) -> None:
    queues, classes, _latency = _predict(model, samples, batch_size)
    fields = [
        "split", "run_id", "scenario", "client_id", "client_type", "approach", "end_t",
        "true_queue_m", "true_class", "pred_queue_m", "pred_class",
    ]
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if file.tell() == 0:
            writer.writeheader()
        for sample, queue, cls in zip(samples, queues, classes):
            writer.writerow({
                "split": split, "run_id": sample.run_id, "scenario": sample.scenario,
                "client_id": sample.client_id, "client_type": sample.client_type,
                "approach": sample.approach, "end_t": sample.end_t,
                "true_queue_m": sample.queue_m, "true_class": sample.congestion_class,
                "pred_queue_m": queue, "pred_class": cls,
            })


def run_fedavg_training(config_path: Union[str, Path],
                        dataset_override: Optional[str] = None,
                        hierarchy: bool = False,
                        iid_control: bool = False,
                        transient_participation: bool = False) -> Path:
    """Run fixed-cohort flat or two-stage hierarchical FedAvg trajectory replay."""
    config = load_learning_config(config_path)
    training = _validate_training(config)
    federation = config.get("federation")
    if not isinstance(federation, dict):
        raise TrainingError("learning config must define federation settings")
    rounds = federation.get("rounds")
    max_clients = federation.get("max_clients")
    deadline_s = federation.get("deadline_s")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise TrainingError("federation.rounds must be a positive integer")
    if isinstance(max_clients, bool) or not isinstance(max_clients, int) or max_clients < 2:
        raise TrainingError("federation.max_clients must be an integer of at least two")
    if hierarchy and iid_control:
        raise TrainingError("IID control and hierarchical aggregation cannot be selected together")
    if iid_control and transient_participation:
        raise TrainingError("transient participation is available for natural flat or hierarchical replay, not IID control")
    if hierarchy and max_clients < 6:
        raise TrainingError("hierarchical FedAvg requires max_clients >= 6 (four RSUs and two vehicles)")
    if (isinstance(deadline_s, bool) or not isinstance(deadline_s, (int, float))
            or not math.isfinite(deadline_s) or deadline_s <= 0):
        raise TrainingError("federation.deadline_s must be positive")
    if federation.get("aggregation") != "sample_weighted_model_weights":
        raise TrainingError("federation.aggregation must be sample_weighted_model_weights")
    transient_rate = federation.get("transient_vehicle_participation_rate")
    if (isinstance(transient_rate, bool) or not isinstance(transient_rate, (int, float))
            or not math.isfinite(transient_rate) or not 0 < transient_rate <= 1):
        raise TrainingError("federation.transient_vehicle_participation_rate must be in (0, 1]")
    max_windows_per_client = federation.get("max_windows_per_client")
    if (isinstance(max_windows_per_client, bool)
            or not isinstance(max_windows_per_client, int) or max_windows_per_client < 1):
        raise TrainingError("federation.max_windows_per_client must be a positive integer")

    dataset_dir = Path(dataset_override or config["dataset"])
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise TrainingError(f"dataset manifest not found: {manifest_path}")
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not {"train", "val", "test"} <= set(dataset_manifest.get("splits", {})):
        raise TrainingError("FedAvg dataset must have train, validation and test run splits")
    minimum_windows = int(config["baseline"]["minimum_client_windows"])
    if hierarchy:
        vehicle_client_ids = _select_hierarchical_vehicles(
            dataset_manifest, max_clients - 4, minimum_windows, int(training["seed"])
        )
        persistent_rsu_ids = [f"rsu_{approach}" for approach in ("N", "E", "S", "W")]
        rsu_source_ids = [
            client_id for client_id, metadata in dataset_manifest.get("clients", {}).items()
            if metadata.get("split") == "train" and metadata.get("trainable")
            and metadata.get("type") == "rsu"
        ]
        source_client_ids = vehicle_client_ids + rsu_source_ids
        client_ids = sorted(vehicle_client_ids + persistent_rsu_ids)
        source_to_destination = {client_id: client_id for client_id in vehicle_client_ids}
        client_meta_by_id = {
            client_id: dataset_manifest["clients"][client_id] for client_id in vehicle_client_ids
        }
        for approach in ("N", "E", "S", "W"):
            persistent_id = f"rsu_{approach}"
            client_meta_by_id[persistent_id] = {"type": "rsu", "approach": approach}
            for source_id in rsu_source_ids:
                if dataset_manifest["clients"][source_id]["approach"] == approach:
                    source_to_destination[source_id] = persistent_id
        sample_cap = max_windows_per_client
    else:
        client_ids = _select_clients(dataset_manifest, max_clients, minimum_windows,
                                     int(training["seed"]))
        source_client_ids = list(client_ids)
        source_to_destination = {client_id: client_id for client_id in client_ids}
        client_meta_by_id = {
            client_id: dataset_manifest["clients"][client_id] for client_id in client_ids
        }
        sample_cap = None
    _seed_everything(int(training["seed"]), bool(training["deterministic"]))
    model_config = {
        "input_size": FEATURE_DIM,
        "hidden_size": int(training["hidden_size"]),
        "num_layers": int(training["layers"]),
        "dropout": float(training["dropout"]),
        "shared_size": int(training["shared_size"]),
    }
    global_model = GRUPredictor(**model_config)
    base_state = {key: value.detach().clone() for key, value in global_model.state_dict().items()}
    base_tensors = _client_tensor_map(global_model)
    client_samples, available_window_counts = _collect_client_windows(
        dataset_dir, source_client_ids, client_ids, source_to_destination,
        sample_cap, int(training["seed"]),
    )
    if iid_control:
        source_client_ids = list(client_ids)
        pooled_samples = [sample for rows in client_samples.values() for sample in rows]
        client_samples = _partition_iid_windows(pooled_samples, max_clients, int(config["iid"]["seed"]))
        client_ids = sorted(client_samples)
        client_meta_by_id = {
            client_id: {"type": "synthetic_iid", "approach": "mixed"}
            for client_id in client_ids
        }
    else:
        client_samples = {
            client_id: samples for client_id, samples in client_samples.items()
            if len(samples) >= minimum_windows
        }
        client_ids = sorted(client_samples)
    if len(client_ids) < 2:
        raise TrainingError("fewer than two selected clients produced the minimum valid windows")
    if any(len(samples) < minimum_windows for samples in client_samples.values()):
        raise TrainingError("a selected client partition is below the minimum-window eligibility rule")
    train_samples = [
        sample for samples in client_samples.values() for sample in samples
    ]
    val_samples = _sample_windows(
        dataset_dir, "val", int(training["max_validation_windows"]), int(training["seed"]) + 1
    )
    test_samples = _sample_windows(
        dataset_dir, "test", int(training["max_validation_windows"]), int(training["seed"]) + 2
    )
    if not val_samples or not test_samples:
        raise TrainingError("validation and test splits must each contain sampled windows")
    if set(sample.run_id for sample in train_samples) & {
        sample.run_id for sample in val_samples + test_samples
    }:
        raise TrainingError("whole-run split leakage detected in selected FedAvg windows")

    start_utc = dt.datetime.now(dt.timezone.utc)
    wall_started = time.perf_counter()
    output_root = Path(config["output_dir"])
    method_name = (
        "hierarchical_fedavg" if hierarchy else
        "flat_fedavg_iid" if iid_control else
        "flat_fedavg"
    )
    experiment_id = f"{method_name}_{dataset_dir.name}_{start_utc.strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = output_root / experiment_id
    if output_dir.exists():
        raise TrainingError(f"experiment output already exists: {output_dir}")
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    round_records = []
    with (output_dir / "client_rounds.csv").open("w", newline="", encoding="utf-8") as client_file, \
            (output_dir / "updates.jsonl").open("w", encoding="utf-8") as update_file:
        client_writer = csv.DictWriter(
            client_file,
            fieldnames=["round_id", "client_id", "sample_count", "training_time_s", "training_loss",
                        "serialized_bytes", "accepted", "rejection_reason"],
        )
        client_writer.writeheader()
        model_version = 0
        current_hash = model_hash(base_tensors)
        previous_active_ids: set[str] = set()
        vehicle_client_ids = [
            client_id for client_id in client_ids
            if client_meta_by_id[client_id]["type"] != "rsu"
        ]
        persistent_rsu_ids = [
            client_id for client_id in client_ids
            if client_meta_by_id[client_id]["type"] == "rsu"
        ]
        for round_id in range(1, rounds + 1):
            round_started = time.perf_counter()
            round_origin = round_started
            if transient_participation:
                active_count = max(1, math.ceil(len(vehicle_client_ids) * transient_rate))
                round_rng = random.Random(int(training["seed"]) + round_id * 7919)
                active_vehicle_ids = sorted(round_rng.sample(vehicle_client_ids, active_count))
                active_client_ids = sorted(active_vehicle_ids + persistent_rsu_ids)
            else:
                active_client_ids = list(client_ids)
            active_ids = set(active_client_ids)
            client_arrivals = sorted(active_ids - previous_active_ids)
            client_departures = sorted(previous_active_ids - active_ids)
            updates = []
            local_durations = 0.0
            for client_id in active_client_ids:
                client_seed = int.from_bytes(
                    hashlib.sha256(f"{training['seed']}:{round_id}:{client_id}".encode()).digest()[:4],
                    "big",
                )
                tensors, local_loss, local_duration = _client_training(
                    model_config, base_state, client_samples[client_id], training, client_seed
                )
                local_durations += local_duration
                client_meta = client_meta_by_id[client_id]
                group_id = (
                    f"rsu_{client_meta['approach']}"
                    if client_meta["type"] in ("vehicle", "rsu") else None
                )
                update = ModelUpdate(
                    client_id=client_id,
                    client_type=client_meta["type"],
                    round_id=round_id,
                    base_model_version=model_version,
                    base_model_hash=current_hash,
                    feature_schema_hash=FEATURE_SCHEMA_HASH,
                    sample_count=len(client_samples[client_id]),
                    tensors=tensors,
                    received_at_s=time.perf_counter() - round_origin,
                    group_id=group_id,
                    training_time_s=local_duration,
                    training_metadata={
                        "epochs": float(training["local_epochs"]),
                        "training_loss": local_loss,
                    },
                )
                updates.append(update)
            aggregator = aggregate_hierarchical if hierarchy else aggregate_flat
            result = aggregator(
                updates, base_tensors, round_id=round_id, model_version=model_version,
                deadline_s=float(deadline_s),
            )
            accepted = set(result.accepted_clients)
            for update in updates:
                rejection = result.rejected_clients.get(update.client_id, "")
                client_writer.writerow({
                    "round_id": round_id,
                    "client_id": update.client_id,
                    "sample_count": update.sample_count,
                    "training_time_s": update.training_time_s,
                    "training_loss": update.training_metadata["training_loss"],
                    "serialized_bytes": result.serialized_bytes[update.client_id],
                    "accepted": update.client_id in accepted,
                    "rejection_reason": rejection,
                })
                update_file.write(json.dumps({
                    "round_id": round_id,
                    "client_id": update.client_id,
                    "client_type": update.client_type,
                    "group_id": update.group_id,
                    "sample_count": update.sample_count,
                    "base_model_version": update.base_model_version,
                    "base_model_hash": update.base_model_hash,
                    "feature_schema_hash": update.feature_schema_hash,
                    "update_model_hash": model_hash(update.tensors),
                    "received_at_s": update.received_at_s,
                    "training_time_s": update.training_time_s,
                    "training_metadata": dict(update.training_metadata),
                    "serialized_bytes": result.serialized_bytes[update.client_id],
                    "accepted": update.client_id in accepted,
                    "rejection_reason": rejection,
                }, sort_keys=True) + "\n")
            base_tensors = result.tensors
            base_state = {
                name: torch.as_tensor(value.copy(), dtype=global_model.state_dict()[name].dtype)
                for name, value in base_tensors.items()
            }
            global_model.load_state_dict(base_state, strict=True)
            if result.sample_mass:
                model_version += 1
            current_hash = model_hash(base_tensors)
            validation_loader = DataLoader(
                _to_dataset(val_samples), batch_size=int(training["batch_size"]),
                shuffle=False, num_workers=0,
            )
            validation_loss = _loss_for(global_model, validation_loader, training)["loss"]
            round_record = {
                "round_id": round_id,
                "model_version": model_version,
                "base_model_hash": updates[0].base_model_hash,
                "global_model_hash": current_hash,
                "eligible_clients": len(client_ids),
                "connected_clients": len(active_client_ids),
                "connected_client_ids": active_client_ids,
                "dropped_clients": sorted(set(client_ids) - active_ids),
                "client_arrivals": client_arrivals,
                "client_departures": client_departures,
                "accepted_clients": result.accepted_clients,
                "rejected_clients": result.rejected_clients,
                "sample_mass": result.sample_mass,
                "serialized_bytes": sum(result.serialized_bytes.values()),
                "local_training_time_s": local_durations,
                "aggregation_time_s": result.aggregation_time_s,
                "round_duration_s": time.perf_counter() - round_started,
                "validation_loss": validation_loss,
                "skipped": result.skipped,
                "groups": result.groups,
            }
            round_records.append(round_record)
            previous_active_ids = active_ids
            torch.save({
                "model_state_dict": base_state,
                "model_config": {**model_config, "sequence_length": SEQUENCE_LENGTH},
                "feature_schema_hash": FEATURE_SCHEMA_HASH,
                "model_version": model_version,
                "round_id": round_id,
                "model_hash": current_hash,
            }, output_dir / "checkpoints" / f"round_{round_id:03d}.pt")

    final_checkpoint = output_dir / "checkpoints" / f"round_{rounds:03d}.pt"
    checkpoint_hash = hashlib.sha256(final_checkpoint.read_bytes()).hexdigest()
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as prediction_file:
        fields = [
            "split", "run_id", "scenario", "client_id", "client_type", "approach", "end_t",
            "true_queue_m", "true_class", "pred_queue_m", "pred_class",
        ]
        csv.DictWriter(prediction_file, fieldnames=fields).writeheader()
    for split, samples in (("val", val_samples), ("test", test_samples)):
        _write_predictions(output_dir / "predictions.csv", global_model, samples,
                           int(training["batch_size"]), split)
    metrics = {
        split: _prediction_metrics_from_saved(output_dir / "predictions.csv", split)
        for split in ("val", "test")
    }
    (output_dir / "rounds.json").write_text(json.dumps(round_records, indent=2, sort_keys=True),
                                             encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True),
                                              encoding="utf-8")
    resolved_config = {**config, "dataset": str(dataset_dir)}
    resolved_text = yaml.safe_dump(resolved_config, sort_keys=False)
    (output_dir / "resolved_config.yaml").write_text(resolved_text, encoding="utf-8")
    repo = _repo_state()
    eligible_clients = [
        client for client in dataset_manifest.get("clients", {}).values()
        if client.get("split") == "train"
        and client.get("trainable")
        and client.get("windows", 0) >= minimum_windows
    ]
    if hierarchy:
        eligible_count = sum(
            client.get("type") == "vehicle" for client in eligible_clients
        ) + 4
    else:
        eligible_count = len(eligible_clients)
    manifest = {
        "experiment_id": experiment_id,
        "method": ("hierarchical_sample_weighted_fedavg" if hierarchy
                   else "flat_synthetic_iid_fedavg" if iid_control
                   else "flat_sample_weighted_fedavg"),
        "hierarchical": hierarchy,
        "iid_control": iid_control,
        "transient_participation": transient_participation,
        "transient_vehicle_participation_rate": transient_rate if transient_participation else None,
        "simulation_type": "offline_saved_trajectory_replay",
        "dataset_id": dataset_dir.name,
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(resolved_text.encode("utf-8")).hexdigest(),
        "feature_schema_sha256": FEATURE_SCHEMA_HASH,
        "train_run_ids": sorted(dataset_manifest["splits"]["train"]),
        "validation_run_ids": sorted(dataset_manifest["splits"]["val"]),
        "test_run_ids": sorted(dataset_manifest["splits"]["test"]),
        "selected_client_ids": client_ids,
        "source_trajectory_client_ids": source_client_ids if iid_control else None,
        "selected_client_count": len(client_ids),
        "eligible_client_count": eligible_count,
        "selected_clients_by_type": dict(Counter(
            client_meta_by_id[client_id]["type"] for client_id in client_ids
        )),
        "selected_clients_by_approach": dict(Counter(
            client_meta_by_id[client_id]["approach"] for client_id in client_ids
        )),
        "selected_client_label_counts": {
            client_id: dict(Counter(str(sample.congestion_class) for sample in client_samples[client_id]))
            for client_id in client_ids
        },
        "available_training_windows_by_client": available_window_counts,
        "max_windows_per_client": max_windows_per_client if hierarchy else None,
        "selected_train_windows": sum(map(len, client_samples.values())),
        "rounds": round_records,
        "model_parameters": sum(parameter.numel() for parameter in global_model.parameters()),
        "checkpoint_sha256": checkpoint_hash,
        "predictions_sha256": hashlib.sha256((output_dir / "predictions.csv").read_bytes()).hexdigest(),
        "validation_window_count": len(val_samples),
        "test_window_count": len(test_samples),
        "seed": training["seed"],
        "deterministic_algorithms": training["deterministic"],
        "device": "cpu",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "code_commit": repo["commit"],
        "code_worktree_dirty": repo["worktree_dirty"],
        "experiment_started_utc": start_utc.isoformat(),
        "experiment_ended_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_duration_s": time.perf_counter() - wall_started,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "rounds.csv").open("w", newline="", encoding="utf-8") as rounds_file:
        writer = csv.DictWriter(rounds_file, fieldnames=[
            "round_id", "model_version", "base_model_hash", "global_model_hash",
            "eligible_clients", "connected_clients", "connected_client_ids", "dropped_clients",
            "client_arrivals", "client_departures", "sample_mass", "accepted_clients", "rejected_clients",
            "serialized_bytes", "local_training_time_s",
            "aggregation_time_s", "round_duration_s", "validation_loss", "skipped", "groups",
        ])
        writer.writeheader()
        for row in round_records:
            writer.writerow({
                **row,
                "accepted_clients": json.dumps(row["accepted_clients"]),
                "rejected_clients": json.dumps(row["rejected_clients"], sort_keys=True),
                "groups": json.dumps(row["groups"], sort_keys=True),
                "connected_client_ids": json.dumps(row["connected_client_ids"]),
                "dropped_clients": json.dumps(row["dropped_clients"]),
                "client_arrivals": json.dumps(row["client_arrivals"]),
                "client_departures": json.dumps(row["client_departures"]),
            })
    summary = [
        f"# {'Hierarchical' if hierarchy else 'Flat'} FedAvg experiment: {experiment_id}",
        "",
        f"- Dataset: `{dataset_dir.name}`; feature schema hash `{FEATURE_SCHEMA_HASH}`.",
        f"- Fixed replay cohort: {len(client_ids)} clients, {sum(map(len, client_samples.values()))} distinct training windows.",
        f"- Rounds: {rounds}; local epochs per client per round: {training['local_epochs']}.",
        ("- Four persistent RSU learners train only on their own sensor histories, capped at "
         f"{max_windows_per_client} sampled windows each; child vehicle updates are aggregated once at their parent RSU."
         if hierarchy else
         "- Synthetic IID partitions are generated offline from the same natural replay cohort; this is a diagnostic control, not a local deployment."
         if iid_control else
         "- Flat aggregation treats each selected saved trajectory as an independent replay client."),
        ("- Vehicle participants are deterministically sampled at "
         f"{transient_rate:.0%} per round; these are offline replay arrivals/departures, not live connections."
         if transient_participation else
         "- The same fixed replay cohort participates in every round."),
        "- This is an offline saved-trajectory replay, not an online networked deployment.",
        "- Serialized byte counts use the federation module's explicit JSON tensor representation; they are not measured network traffic.",
        "- Test data was not used for client selection, local training, or round selection.",
        "",
        "| Split | N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summary.extend(
        f"| {split} | {metrics[split]['overall']['n']} | {metrics[split]['overall']['mae_m']} "
        f"| {metrics[split]['overall']['rmse_m']} | {metrics[split]['overall']['macro_f1']} "
        f"| {metrics[split]['overall']['balanced_accuracy']} |"
        for split in ("val", "test")
    )
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return output_dir
