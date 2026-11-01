from pathlib import Path
import numpy as np
import pytest

import trafma.live_federation as live
from trafma.features import FEATURE_DIM, FeatureContext, FeatureWindow
from trafma.live_federation import LiveFederation, _Client


ROOT = Path(__file__).resolve().parents[1]


def _learner(tmp_path):
    learner = LiveFederation(ROOT / "configs" / "learning.yaml")
    learner.run_id = "live-test"
    learner.run_dir = tmp_path
    learner.log_file = (tmp_path / "online_federation.jsonl").open("w", encoding="utf-8")
    return learner


def _training_windows(client_id, client_type, approach, count):
    features = [[0.0] * FEATURE_DIM for _ in range(10)]
    return [
        FeatureWindow(
            dataset_id="live_sumo", run_id="live-test", split="train", scenario="live",
            client_id=client_id, client_type=client_type, approach=approach, end_t=float(index),
            zone_m=100.0, features=features, current_queue_m=None, queue_m=0.0,
            congestion_class=0,
        )
        for index in range(count)
    ]


def test_vehicle_requires_matured_windows_before_becoming_trainable(tmp_path):
    learner = _learner(tmp_path)
    client = _Client("veh_0001", "vehicle", "N", True)

    assert learner._eligibility(client)[0] == "inference_only"
    client.valid_observations = 10
    assert learner._eligibility(client)[1] == "waiting_for_matured_t_plus_30_label"
    client.matured_labels = 1
    assert learner._eligibility(client)[1] == "fewer_than_16_matured_training_windows"
    client.training_windows.extend(_training_windows(client.client_id, "vehicle", "N", 16))
    assert learner._eligibility(client) == ("trainable", "eligible")

    learner.rsu_self_update = False
    rsu = _Client("rsu_N", "rsu", "N", True)
    rsu.valid_observations = 10
    rsu.matured_labels = 1
    rsu.training_windows.extend(_training_windows(rsu.client_id, "rsu", "N", 16))
    assert learner._eligibility(rsu) == ("aggregator_only", "rsu_self_update_disabled")

    learner.close()


def test_live_schedule_rejects_misaligned_simulation_report_steps(tmp_path):
    learner = LiveFederation(ROOT / "configs" / "learning.yaml")

    with pytest.raises(live.LiveFederationError, match="align with report steps"):
        learner.start(
            "live-test",
            tmp_path,
            {
                "labels": {"horizon_s": 30.0, "history_len": 10,
                           "congestion_thresholds": [0.10, 0.35]},
                "run": {"step_s": 0.3, "report_step_s": 1.0},
            },
            tmp_path / "topology.json",
        )


def test_live_round_respects_client_cap_and_reserves_rsu_slots(tmp_path):
    learner = _learner(tmp_path)
    learner.max_clients = 6
    clients = [
        _Client(f"rsu_{approach}", "rsu", approach, True)
        for approach in ("N", "E", "S", "W")
    ] + [
        _Client(f"veh_{index:02}", "vehicle", "N", True)
        for index in range(5)
    ]

    learner.round_id = 1
    participants = learner._select_participants(clients)
    assert len(participants) == 6
    assert sum(client.client_type == "rsu" for client in participants) == 4
    assert [client.client_id for client in participants] == [
        client.client_id for client in learner._select_participants(clients)
    ]

    learner.rsu_self_update = False
    participants_without_rsu_updates = learner._select_participants(clients)
    assert len(participants_without_rsu_updates) == 2
    assert all(client.client_type == "vehicle" for client in participants_without_rsu_updates)
    learner.close()


def test_labels_wait_until_maturity_and_departed_vehicle_is_not_trained(tmp_path, monkeypatch):
    learner = _learner(tmp_path)
    client = _Client("veh_0001", "vehicle", "N", True)
    learner.clients[client.client_id] = client
    learner.contexts[client.client_id] = FeatureContext(
        zone_m=100.0, arm_length_m=300.0, incoming_edge_length_m=300.0,
        vehicle_radius_m=100.0, vehicle_length_m=5.0, min_gap_m=2.5,
        max_speed_mps=30.0, max_accel_mps2=3.0, max_decel_mps2=4.0,
        halting_speed_mps=0.1, signal_cycle_s=120.0,
    )
    monkeypatch.setattr(
        live, "adapt_observation",
        lambda *_: np.zeros(FEATURE_DIM, dtype=np.float32),
    )

    for now in range(10):
        learner.observe(client.client_id, float(now), {"scope": "incoming_local"})

    assert client.valid_observations == 10
    assert client.valid_windows == 1
    assert len(learner.pending[39.0]) == 1
    assert learner._eligibility(client)[0] == "inference_only"

    learner.release_matured_labels(38.0, {})
    assert len(learner.pending[39.0]) == 1
    learner.on_vehicle_arrival(client.client_id, 38.5)
    learner.release_matured_labels(39.0, {"N": {"zone_queue_m": 2.0}})

    assert client.matured_labels == 1
    assert client.training_windows == []
    assert learner._eligibility(client)[0] == "departed_inference_only"
    with pytest.raises(live.LiveFederationError, match="departed client"):
        learner.observe(client.client_id, 40.0, {"scope": "incoming_local"})
    learner.close()
