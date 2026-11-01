"""Optional live-SUMO federated learner with delayed, client-local supervision."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from . import APPROACHES
from .features import (
    FEATURE_DIM,
    FEATURE_SCHEMA_HASH,
    SEQUENCE_LENGTH,
    FeatureContext,
    FeatureWindow,
    adapt_observation,
)
from .federated_training import _client_tensor_map, _client_training
from .federation import ModelUpdate, aggregate_hierarchical, model_hash
from .models import GRUPredictor
from .training import _seed_everything, _validate_training


class LiveFederationError(ValueError):
    """Raised when live learning configuration or supervised input is invalid."""


@dataclass
class _PendingWindow:
    client_id: str
    approach: str
    end_t: float
    zone_m: float
    features: list[list[float]]


@dataclass
class _Client:
    client_id: str
    client_type: str
    approach: str
    active: bool
    history: deque = field(default_factory=lambda: deque(maxlen=SEQUENCE_LENGTH))
    valid_observations: int = 0
    valid_windows: int = 0
    matured_labels: int = 0
    training_windows: list[FeatureWindow] = field(default_factory=list)


def _class_for(queue_m: float, zone_m: float, thresholds: list[float]) -> int:
    ratio = queue_m / zone_m
    if ratio < thresholds[0]:
        return 0
    if ratio < thresholds[1]:
        return 1
    return 2


class LiveFederation:
    """Collect agent-local windows and call hierarchical aggregation only at round boundaries."""

    def __init__(self, learning_config_path: str | Path):
        config_path = Path(learning_config_path)
        if not config_path.is_file():
            raise LiveFederationError(f"learning config not found: {config_path}")
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        self.training = _validate_training(self.config)
        self.federation = self.config.get("federation")
        if not isinstance(self.federation, dict):
            raise LiveFederationError("learning config must define federation settings")
        self.max_clients = self.federation.get("max_clients", 16)
        if (isinstance(self.max_clients, bool) or not isinstance(self.max_clients, int)
                or self.max_clients < 5):
            raise LiveFederationError("federation.max_clients must be at least five for live RSU/vehicle participation")
        self.rsu_self_update = self.federation.get("rsu_self_update", True)
        if not isinstance(self.rsu_self_update, bool):
            raise LiveFederationError("federation.rsu_self_update must be boolean")
        self.round_interval_s = self.federation.get("live_round_interval_s", 60.0)
        if (isinstance(self.round_interval_s, bool)
                or not isinstance(self.round_interval_s, (int, float))
                or not math.isfinite(self.round_interval_s) or self.round_interval_s <= 0):
            raise LiveFederationError("federation.live_round_interval_s must be positive")
        self.deadline_s = self.federation.get("deadline_s", 300.0)
        if (isinstance(self.deadline_s, bool) or not isinstance(self.deadline_s, (int, float))
                or not math.isfinite(self.deadline_s) or self.deadline_s <= 0):
            raise LiveFederationError("federation.deadline_s must be positive")
        self.minimum_windows = self.config.get("baseline", {}).get("minimum_client_windows", 16)
        if isinstance(self.minimum_windows, bool) or not isinstance(self.minimum_windows, int) \
                or self.minimum_windows < 16:
            raise LiveFederationError("live federation requires a minimum of at least 16 training windows")
        self.horizon_s = 30.0
        self.max_windows = self.federation.get("max_windows_per_client", 2000)
        if isinstance(self.max_windows, bool) or not isinstance(self.max_windows, int) or self.max_windows < 16:
            raise LiveFederationError("federation.max_windows_per_client must be at least 16")

        model_config = {
            "input_size": FEATURE_DIM,
            "hidden_size": int(self.training["hidden_size"]),
            "num_layers": int(self.training["layers"]),
            "dropout": float(self.training["dropout"]),
            "shared_size": int(self.training["shared_size"]),
        }
        _seed_everything(int(self.training["seed"]), bool(self.training["deterministic"]))
        self.model = GRUPredictor(**model_config)
        self.model_config = model_config
        self.current_tensors = _client_tensor_map(self.model)
        self.model_version = 0
        self.round_id = 0
        self.clients: dict[str, _Client] = {}
        self.contexts: dict[str, FeatureContext] = {}
        self.pending: dict[float, list[_PendingWindow]] = defaultdict(list)
        self.run_id = ""
        self.run_dir: Optional[Path] = None
        self.log_file = None
        self.next_round_s = float(self.round_interval_s)
        self._closed = False
        self._wall_started = 0.0
        self._round_count = 0
        self._vehicle_round_participants = 0
        self._rsu_round_participants = 0
        self._skipped_rounds = 0
        self._feature_schema_hash = FEATURE_SCHEMA_HASH

    def start(self, run_id: str, run_dir: Path, config: dict, topology_path: Path) -> None:
        if self.log_file is not None:
            raise LiveFederationError("live federation has already been started")
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        labels = config.get("labels", {})
        self.horizon_s = float(labels.get("horizon_s", self.horizon_s))
        history_len = int(labels.get("history_len", SEQUENCE_LENGTH))
        if self.horizon_s != 30.0 or history_len != SEQUENCE_LENGTH:
            raise LiveFederationError("live federation requires the frozen 10-step, 30-second label contract")
        run_config = config.get("run", {})
        step_s = float(run_config.get("step_s", 1.0))
        report_step_s = float(run_config.get("report_step_s", 1.0))
        if not math.isfinite(step_s) or step_s <= 0 or not math.isfinite(report_step_s) \
                or report_step_s <= 0:
            raise LiveFederationError("simulation step and report interval must be positive")
        report_every = max(int(round(report_step_s / step_s)), 1)
        effective_report_step_s = report_every * step_s
        if (abs(effective_report_step_s - report_step_s) > 1e-6
                or abs(self.horizon_s / effective_report_step_s
                       - round(self.horizon_s / effective_report_step_s)) > 1e-6
                or abs(float(self.round_interval_s) / effective_report_step_s
                       - round(float(self.round_interval_s) / effective_report_step_s)) > 1e-6):
            raise LiveFederationError("label horizon and round interval must align with report steps")
        self.thresholds = list(labels.get("congestion_thresholds", [0.10, 0.35]))
        if self.thresholds != [0.10, 0.35]:
            raise LiveFederationError("live federation requires frozen congestion thresholds [0.10, 0.35]")
        topology = json.loads(Path(topology_path).read_text(encoding="utf-8"))
        for approach in APPROACHES:
            client_id = f"rsu_{approach}"
            self.clients[client_id] = _Client(client_id, "rsu", approach, True)
            self.contexts[client_id] = FeatureContext.from_run(config, topology, approach)
        self.log_file = (self.run_dir / "online_federation.jsonl").open("w", encoding="utf-8")
        (self.run_dir / "online_checkpoints").mkdir(parents=True, exist_ok=False)
        self._wall_started = time.perf_counter()
        self._write({
            "event": "start",
            "run_id": run_id,
            "simulated_time_s": 0.0,
            "round_interval_s": self.round_interval_s,
            "horizon_s": self.horizon_s,
            "minimum_valid_windows": self.minimum_windows,
            "max_clients": self.max_clients,
            "rsu_self_update": self.rsu_self_update,
            "feature_schema_hash": self._feature_schema_hash,
            "initial_model_hash": model_hash(self.current_tensors),
            "supervision": "evaluator labels are released only at their end_t + 30 s maturity time to the owning local cache",
        })

    def _write(self, record: dict) -> None:
        if self.log_file is None:
            raise LiveFederationError("live federation has not been started")
        self.log_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def on_vehicle_departure(self, vehicle_id: str, approach: str, now: float) -> None:
        if approach not in APPROACHES:
            raise LiveFederationError(f"unknown vehicle approach: {approach!r}")
        if vehicle_id in self.clients:
            raise LiveFederationError(f"duplicate live vehicle identity: {vehicle_id}")
        self.clients[vehicle_id] = _Client(vehicle_id, "vehicle", approach, True)
        self.contexts[vehicle_id] = self.contexts[f"rsu_{approach}"]
        self._write({
            "event": "vehicle_joined",
            "simulated_time_s": now,
            "client_id": vehicle_id,
            "client_type": "vehicle",
            "approach": approach,
        })

    def on_vehicle_arrival(self, vehicle_id: str, now: float) -> None:
        client = self.clients.get(vehicle_id)
        if client is None or client.client_type != "vehicle":
            raise LiveFederationError(f"vehicle is not registered for live learning: {vehicle_id}")
        client.active = False
        client.history.clear()
        self._write({
            "event": "vehicle_departed",
            "simulated_time_s": now,
            "client_id": vehicle_id,
            "valid_observations": client.valid_observations,
            "valid_windows": client.valid_windows,
            "matured_labels": client.matured_labels,
            "training_windows": len(client.training_windows),
            "status": self._eligibility(client)[0],
        })

    def observe(self, client_id: str, now: float, view: dict) -> None:
        client = self.clients.get(client_id)
        if client is None:
            raise LiveFederationError(f"observation belongs to an unregistered live client: {client_id}")
        if not client.active:
            raise LiveFederationError(f"departed client produced a live observation: {client_id}")
        if client.client_type == "vehicle" and view.get("scope") != "incoming_local":
            client.history.clear()
            return
        vector = adapt_observation(view, client.client_type, self.contexts[client_id])
        client.history.append(vector)
        client.valid_observations += 1
        if len(client.history) < SEQUENCE_LENGTH:
            return
        features = [list(row) for row in client.history]
        client.valid_windows += 1
        zone_m = self.contexts[client_id].zone_m
        due_t = round(now + self.horizon_s, 6)
        self.pending[due_t].append(
            _PendingWindow(client_id, client.approach, float(now), zone_m, features)
        )
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.from_numpy(np.asarray([features], dtype=np.float32)))
        self._write({
            "event": "prediction",
            "simulated_time_s": now,
            "client_id": client_id,
            "client_type": client.client_type,
            "approach": client.approach,
            "model_version": self.model_version,
            "model_hash": model_hash(self.current_tensors),
            "forecast_horizon_s": self.horizon_s,
            "queue_pred_m": float(output["queue_ratio"][0]) * zone_m,
            "class_scores_raw": [float(value) for value in output["class_logits"][0]],
            "class_scores_calibrated": False,
        })

    def release_matured_labels(self, now: float, truth: dict) -> None:
        due_times = sorted(due for due in self.pending if due <= now + 1e-6)
        for due_t in due_times:
            if abs(due_t - now) > 1e-6:
                raise LiveFederationError(
                    f"matured label at {due_t:g}s cannot be released at observation time {now:g}s"
                )
            for pending in self.pending.pop(due_t):
                client = self.clients[pending.client_id]
                target = truth[pending.approach]
                queue_m = float(target["zone_queue_m"])
                if not math.isfinite(queue_m) or not 0 <= queue_m <= pending.zone_m + 1e-3:
                    raise LiveFederationError(f"invalid matured label for {pending.client_id} at {now}")
                client.matured_labels += 1
                if client.active:
                    congestion_class = _class_for(
                        queue_m, pending.zone_m, self.thresholds,
                    )
                    client.training_windows.append(FeatureWindow(
                        dataset_id="live_sumo",
                        run_id=self.run_id,
                        split="train",
                        scenario="live",
                        client_id=pending.client_id,
                        client_type=client.client_type,
                        approach=client.approach,
                        end_t=pending.end_t,
                        zone_m=pending.zone_m,
                        features=pending.features,
                        current_queue_m=None,
                        queue_m=queue_m,
                        congestion_class=congestion_class,
                    ))
                    if len(client.training_windows) > self.max_windows:
                        client.training_windows.pop(0)
                self._write({
                    "event": "label_matured",
                    "simulated_time_s": now,
                    "label_end_t": due_t,
                    "client_id": pending.client_id,
                    "client_type": client.client_type,
                    "approach": pending.approach,
                    "active_at_release": client.active,
                    "matured_label_count": client.matured_labels,
                    "eligible_training_windows": len(client.training_windows),
                })

    def _eligibility(self, client: _Client) -> tuple[str, str]:
        if not client.active:
            return "departed_inference_only", "vehicle_has_departed"
        if client.client_type == "rsu" and not self.rsu_self_update:
            return "aggregator_only", "rsu_self_update_disabled"
        if client.valid_observations < SEQUENCE_LENGTH:
            return "inference_only", "fewer_than_10_valid_observations"
        if client.matured_labels < 1:
            return "inference_only", "waiting_for_matured_t_plus_30_label"
        if len(client.training_windows) < self.minimum_windows:
            return "inference_only", "fewer_than_16_matured_training_windows"
        return "trainable", "eligible"

    def _eligible_statuses(self) -> tuple[list[_Client], list[dict]]:
        eligible = []
        statuses = []
        for client in sorted(self.clients.values(), key=lambda value: value.client_id):
            status, reason = self._eligibility(client)
            statuses.append({
                "client_id": client.client_id,
                "client_type": client.client_type,
                "approach": client.approach,
                "active": client.active,
                "status": status,
                "reason": reason,
                "valid_observations": client.valid_observations,
                "valid_windows": client.valid_windows,
                "matured_labels": client.matured_labels,
                "training_windows": len(client.training_windows),
            })
            if status == "trainable":
                eligible.append(client)
        return eligible, statuses

    def _select_participants(self, eligible: list[_Client]) -> list[_Client]:
        rsus = sorted(
            (client for client in eligible if client.client_type == "rsu"),
            key=lambda client: client.client_id,
        )
        vehicles = sorted(
            (client for client in eligible if client.client_type == "vehicle"),
            key=lambda client: client.client_id,
        )
        vehicle_limit = self.max_clients - 4
        if len(vehicles) > vehicle_limit:
            rng = random.Random(int(self.training["seed"]) + self.round_id * 104729)
            vehicles = sorted(rng.sample(vehicles, vehicle_limit), key=lambda client: client.client_id)
        selected = (rsus if self.rsu_self_update else []) + vehicles
        return sorted(selected, key=lambda client: client.client_id)

    def run_due_rounds(self, now: float) -> None:
        if self.log_file is None or self.run_dir is None:
            raise LiveFederationError("live federation has not been started")
        while now + 1e-6 >= self.next_round_s:
            boundary = self.next_round_s
            self.next_round_s += float(self.round_interval_s)
            self._run_round(boundary, now)

    def _run_round(self, boundary: float, now: float) -> None:
        if self.log_file is None or self.run_dir is None:
            raise LiveFederationError("live federation has not been started")
        eligible, statuses = self._eligible_statuses()
        participants = self._select_participants(eligible)
        self.round_id += 1
        start = time.perf_counter()
        round_origin = start
        base_tensors = self.current_tensors
        base_hash = model_hash(base_tensors)
        base_state = {
            name: torch.as_tensor(value.copy(), dtype=self.model.state_dict()[name].dtype)
            for name, value in base_tensors.items()
        }
        updates = []
        for client in participants:
            client_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.training['seed']}:{self.round_id}:{client.client_id}".encode()
                ).digest()[:4],
                "big",
            )
            training = dict(self.training)
            training["seed"] = client_seed
            tensors, loss, duration = _client_training(
                self.model_config, base_state, client.training_windows, training, client_seed
            )
            updates.append(ModelUpdate(
                client_id=client.client_id,
                client_type=client.client_type,
                round_id=self.round_id,
                base_model_version=self.model_version,
                base_model_hash=base_hash,
                feature_schema_hash=FEATURE_SCHEMA_HASH,
                sample_count=len(client.training_windows),
                tensors=tensors,
                received_at_s=time.perf_counter() - round_origin,
                group_id=f"rsu_{client.approach}",
                training_time_s=duration,
                training_metadata={
                    "epochs": float(self.training["local_epochs"]),
                    "training_loss": loss,
                },
            ))
        result = aggregate_hierarchical(
            updates,
            base_tensors,
            round_id=self.round_id,
            model_version=self.model_version,
            deadline_s=float(self.deadline_s),
            rsu_self_update=self.rsu_self_update,
        )
        self.current_tensors = result.tensors
        if result.sample_mass:
            self.model_version += 1
            state = {
                name: torch.as_tensor(value.copy(), dtype=self.model.state_dict()[name].dtype)
                for name, value in self.current_tensors.items()
            }
            self.model.load_state_dict(state, strict=True)
        self._round_count += 1
        if result.skipped:
            self._skipped_rounds += 1
        self._vehicle_round_participants += sum(
            client.client_type == "vehicle" for client in participants
        )
        self._rsu_round_participants += sum(
            client.client_type == "rsu" for client in participants
        )
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {**self.model_config, "sequence_length": SEQUENCE_LENGTH},
            "feature_schema_hash": FEATURE_SCHEMA_HASH,
            "model_version": self.model_version,
            "round_id": self.round_id,
            "model_hash": model_hash(self.current_tensors),
        }, self.run_dir / "online_checkpoints" / f"round_{self.round_id:03d}.pt")
        self._write({
            "event": "round",
            "simulated_time_s": now,
            "scheduled_boundary_s": boundary,
            "round_id": self.round_id,
            "base_model_version": self.model_version - (1 if result.sample_mass else 0),
            "model_version": self.model_version,
            "base_model_hash": base_hash,
            "global_model_hash": model_hash(self.current_tensors),
            "rsu_self_update": self.rsu_self_update,
            "max_clients": self.max_clients,
            "eligible_client_ids": [client.client_id for client in eligible],
            "participant_ids": [client.client_id for client in participants],
            "client_status": statuses,
            "accepted_clients": result.accepted_clients,
            "rejected_clients": result.rejected_clients,
            "groups": result.groups,
            "sample_mass": result.sample_mass,
            "serialized_bytes": sum(result.serialized_bytes.values()),
            "local_training_wall_s": sum(update.training_time_s for update in updates),
            "aggregation_wall_s": result.aggregation_time_s,
            "round_wall_s": time.perf_counter() - start,
            "skipped": result.skipped,
            "simulated_time_is_not_wall_time": True,
        })

    def close(self) -> dict:
        if self._closed:
            return self.summary()
        if self.log_file is not None:
            self._write({
                "event": "finish",
                "simulated_time_s": None,
                "rounds": self._round_count,
                "skipped_rounds": self._skipped_rounds,
                "vehicle_round_participants": self._vehicle_round_participants,
                "rsu_round_participants": self._rsu_round_participants,
                "remaining_unmatured_windows": sum(map(len, self.pending.values())),
                "wall_duration_s": time.perf_counter() - self._wall_started,
            })
            self.log_file.close()
            self.log_file = None
        self._closed = True
        return self.summary()

    def summary(self) -> dict:
        eligible, statuses = self._eligible_statuses()
        return {
            "enabled": self.log_file is not None or self._closed,
            "simulation_type": "live_sumo_federation",
            "rsu_self_update": self.rsu_self_update,
            "max_clients": self.max_clients,
            "round_interval_s": self.round_interval_s,
            "horizon_s": self.horizon_s,
            "minimum_training_windows": self.minimum_windows,
            "feature_schema_hash": self._feature_schema_hash,
            "final_model_version": self.model_version,
            "final_model_hash": model_hash(self.current_tensors),
            "rounds": self._round_count,
            "skipped_rounds": self._skipped_rounds,
            "vehicle_round_participants": self._vehicle_round_participants,
            "rsu_round_participants": self._rsu_round_participants,
            "remaining_unmatured_windows": sum(map(len, self.pending.values())),
            "currently_active_trainable_clients": [client.client_id for client in eligible],
            "client_final_status": statuses,
        }
