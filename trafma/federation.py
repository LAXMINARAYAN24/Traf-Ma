"""Inspectable, sample-weighted flat and RSU-edge FedAvg aggregation."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from . import APPROACHES
from .features import FEATURE_SCHEMA_HASH

TensorMap = Dict[str, np.ndarray]


@dataclass(frozen=True)
class ModelUpdate:
    client_id: str
    client_type: str
    round_id: int
    base_model_version: int
    base_model_hash: str
    feature_schema_hash: str
    sample_count: int
    tensors: TensorMap
    received_at_s: float
    group_id: Optional[str] = None
    training_time_s: float = 0.0
    # This allowlist intentionally excludes training examples, labels and histories.
    training_metadata: Mapping[str, float] = field(default_factory=dict)


@dataclass
class AggregationResult:
    tensors: TensorMap
    sample_mass: int
    accepted_clients: list[str]
    rejected_clients: dict[str, str]
    serialized_bytes: dict[str, int]
    aggregation_time_s: float
    skipped: bool
    groups: dict = field(default_factory=dict)


def model_hash(tensors: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = np.ascontiguousarray(tensors[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def _update_size(update: ModelUpdate) -> int:
    # JSON serialization is transparent and reproducible; it transmits weights and metadata only.
    payload = {
        "client_id": update.client_id,
        "client_type": update.client_type,
        "round_id": update.round_id,
        "base_model_version": update.base_model_version,
        "base_model_hash": update.base_model_hash,
        "feature_schema_hash": update.feature_schema_hash,
        "sample_count": update.sample_count,
        "received_at_s": update.received_at_s,
        "group_id": update.group_id,
        "training_time_s": update.training_time_s,
        "training_metadata": dict(update.training_metadata),
        "tensors": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
            for name, value in sorted(update.tensors.items())
        },
    }
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8"))


def _signature(tensors: Mapping[str, np.ndarray]) -> dict[str, Tuple[Tuple[int, ...], str]]:
    return {name: (tuple(value.shape), str(value.dtype)) for name, value in tensors.items()}


def _validate_metadata(update: ModelUpdate) -> Optional[str]:
    if not isinstance(update.training_metadata, Mapping):
        return "invalid_training_metadata"
    permitted = {"epochs", "training_loss", "training_time_s"}
    if set(update.training_metadata) - permitted:
        return "disallowed_training_metadata"
    for key, value in update.training_metadata.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return f"invalid_training_metadata:{key}"
    if not math.isfinite(update.training_time_s) or update.training_time_s < 0:
        return "invalid_training_time"
    return None


def _validate_update(
    update: ModelUpdate,
    *,
    round_id: int,
    model_version: int,
    base_hash: str,
    expected_signature: dict,
    deadline_s: float,
) -> Optional[str]:
    if update.round_id != round_id:
        return "wrong_round"
    if not math.isfinite(update.received_at_s) or update.received_at_s < 0:
        return "invalid_received_time"
    if update.received_at_s > deadline_s:
        return "late"
    if update.base_model_version != model_version or update.base_model_hash != base_hash:
        return "wrong_base_model"
    if update.feature_schema_hash != FEATURE_SCHEMA_HASH:
        return "wrong_feature_schema"
    if isinstance(update.sample_count, bool) or not isinstance(update.sample_count, int) or update.sample_count <= 0:
        return "invalid_sample_count"
    if not update.client_id:
        return "invalid_client_id"
    if update.client_type not in ("vehicle", "rsu", "synthetic_iid"):
        return "invalid_client_type"
    if _signature(update.tensors) != expected_signature:
        return "tensor_signature_mismatch"
    if not all(np.isfinite(value).all() for value in update.tensors.values()):
        return "non_finite_tensor"
    return _validate_metadata(update)


def _weighted_average(updates: Iterable[ModelUpdate]) -> tuple[TensorMap, int]:
    accepted = list(updates)
    sample_mass = sum(update.sample_count for update in accepted)
    if not sample_mass:
        return {}, 0
    names = sorted(accepted[0].tensors)
    averaged = {}
    for name in names:
        accumulator = np.zeros_like(accepted[0].tensors[name], dtype=np.float64)
        for update in accepted:
            accumulator += update.tensors[name].astype(np.float64) * update.sample_count
        averaged[name] = (accumulator / sample_mass).astype(accepted[0].tensors[name].dtype)
    return averaged, sample_mass


def aggregate_flat(
    updates: Iterable[ModelUpdate],
    current_tensors: Mapping[str, np.ndarray],
    *,
    round_id: int,
    model_version: int,
    deadline_s: float,
) -> AggregationResult:
    """Validate a frozen update set and perform one sample-weighted FedAvg step."""
    started = time.perf_counter()
    base_hash = model_hash(current_tensors)
    expected = _signature(current_tensors)
    accepted: list[ModelUpdate] = []
    rejected: dict[str, str] = {}
    serialized_bytes = {}
    seen = set()
    for update in updates:
        serialized_bytes[update.client_id] = _update_size(update)
        if update.client_id in seen:
            rejected[update.client_id] = "duplicate_client"
            accepted = [item for item in accepted if item.client_id != update.client_id]
            continue
        seen.add(update.client_id)
        reason = _validate_update(
            update,
            round_id=round_id,
            model_version=model_version,
            base_hash=base_hash,
            expected_signature=expected,
            deadline_s=deadline_s,
        )
        if reason:
            rejected[update.client_id] = reason
        else:
            accepted.append(update)
    tensors, mass = _weighted_average(accepted)
    skipped = mass == 0
    if skipped:
        tensors = {name: np.array(value, copy=True) for name, value in current_tensors.items()}
    return AggregationResult(
        tensors=tensors,
        sample_mass=mass,
        accepted_clients=[item.client_id for item in accepted],
        rejected_clients=rejected,
        serialized_bytes=serialized_bytes,
        aggregation_time_s=time.perf_counter() - started,
        skipped=skipped,
    )


def aggregate_hierarchical(
    updates: Iterable[ModelUpdate],
    current_tensors: Mapping[str, np.ndarray],
    *,
    round_id: int,
    model_version: int,
    deadline_s: float,
) -> AggregationResult:
    """Aggregate vehicle and independent RSU sensor updates at their RSU group, then globally."""
    started = time.perf_counter()
    group_updates = {f"rsu_{approach}": [] for approach in APPROACHES}
    materialized = list(updates)
    routed_updates = []
    route_rejections = {}
    route_sizes = {update.client_id: _update_size(update) for update in materialized}
    for update in materialized:
        if update.group_id not in group_updates:
            route_rejections[update.client_id] = "invalid_group"
            continue
        if update.client_type == "vehicle":
            routed_updates.append(update)
            continue
        if update.client_type == "rsu" and update.client_id == update.group_id:
            routed_updates.append(update)
            continue
        if update.client_type == "rsu":
            route_rejections[update.client_id] = "invalid_rsu_learner_identity"
            continue
        route_rejections[update.client_id] = "invalid_client_type"
    flattened = aggregate_flat(
        routed_updates, current_tensors, round_id=round_id,
        model_version=model_version, deadline_s=deadline_s,
    )
    flattened.serialized_bytes.update(route_sizes)
    flattened.rejected_clients.update(route_rejections)
    if flattened.skipped:
        flattened.groups = {
            group_id: {"sample_mass": 0, "client_ids": [], "tensors": None}
            for group_id in group_updates
        }
        return flattened

    accepted_ids = set(flattened.accepted_clients)
    for update in routed_updates:
        if update.client_id in accepted_ids:
            group_updates[update.group_id].append(update)

    groups = {}
    valid_group_updates = []
    for group_id, candidates in group_updates.items():
        if not candidates:
            groups[group_id] = {"sample_mass": 0, "client_ids": [], "tensors": None}
            continue
        group_result = aggregate_flat(
            candidates, current_tensors, round_id=round_id,
            model_version=model_version, deadline_s=deadline_s,
        )
        groups[group_id] = {
            "sample_mass": group_result.sample_mass,
            "client_ids": group_result.accepted_clients,
            "tensors": group_result.tensors if group_result.sample_mass else None,
        }
        if group_result.sample_mass:
            valid_group_updates.append(ModelUpdate(
                client_id=group_id,
                client_type="rsu",
                round_id=round_id,
                base_model_version=model_version,
                base_model_hash=model_hash(current_tensors),
                feature_schema_hash=FEATURE_SCHEMA_HASH,
                sample_count=group_result.sample_mass,
                tensors=group_result.tensors,
                received_at_s=0.0,
                group_id=None,
            ))
    global_result = aggregate_flat(
        valid_group_updates, current_tensors, round_id=round_id,
        model_version=model_version, deadline_s=deadline_s,
    )
    flattened.tensors = global_result.tensors
    flattened.groups = {
        group_id: {key: value for key, value in data.items() if key != "tensors"}
        for group_id, data in groups.items()
    }
    flattened.aggregation_time_s = time.perf_counter() - started
    return flattened
