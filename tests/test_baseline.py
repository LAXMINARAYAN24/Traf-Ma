import csv

import pytest

from trafma.baseline import _Metrics, _class_for, _metrics_from_predictions, load_learning_config


def test_baseline_metrics_match_hand_computed_errors_and_class_counts():
    metrics = _Metrics()
    rows = [
        (0.0, 0.0, 0, 0),
        (0.0, 20.0, 0, 1),
        (20.0, 20.0, 1, 1),
        (20.0, 0.0, 2, 2),
        (20.0, 20.0, 2, 2),
    ]
    for row in rows:
        metrics.add(*row)

    result = metrics.result()
    assert result["n"] == 5
    assert result["mae_m"] == pytest.approx(8.0)
    assert result["rmse_m"] == pytest.approx((800.0 / 5) ** 0.5)
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["balanced_accuracy"] == pytest.approx(5.0 / 6.0)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 2]]


def test_queue_class_mapping_uses_frozen_project_thresholds():
    thresholds = [0.10, 0.35]
    assert _class_for(14.9, 150.0, thresholds) == 0
    assert _class_for(15.0, 150.0, thresholds) == 1
    assert _class_for(52.4, 150.0, thresholds) == 1
    assert _class_for(52.5, 150.0, thresholds) == 2


def test_learning_config_points_to_installed_dependency_workflow():
    config = load_learning_config("configs/learning.yaml")
    assert config["dataset"] == "data/phase2_v2_dev"
    assert config["baseline"]["method"] == "pooled_signal_aware_ridge"
    assert config["baseline"]["minimum_client_windows"] == 16


def test_reported_metrics_are_recomputed_from_saved_prediction_rows(tmp_path):
    path = tmp_path / "predictions.csv"
    fields = [
        "split", "run_id", "scenario", "client_id", "client_type", "approach", "end_t",
        "true_queue_m", "true_class", "persistence_pred_queue_m", "persistence_pred_class",
        "signal_aware_pred_queue_m", "signal_aware_pred_class", "majority_pred_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "split": "test", "run_id": "run_a", "scenario": "free", "client_id": "run_a/rsu_N",
            "client_type": "rsu", "approach": "N", "end_t": 10, "true_queue_m": 20,
            "true_class": 1, "persistence_pred_queue_m": 10, "persistence_pred_class": 0,
            "signal_aware_pred_queue_m": 30, "signal_aware_pred_class": 2, "majority_pred_class": 0,
        })
        writer.writerow({
            "split": "test", "run_id": "run_a", "scenario": "free", "client_id": "run_a/veh_0",
            "client_type": "vehicle", "approach": "N", "end_t": 10, "true_queue_m": 20,
            "true_class": 1, "persistence_pred_queue_m": "", "persistence_pred_class": "",
            "signal_aware_pred_queue_m": 20, "signal_aware_pred_class": 1, "majority_pred_class": 0,
        })

    metrics = _metrics_from_predictions(path)["test"]
    assert metrics["signal_aware_ridge"]["overall"]["n"] == 2
    assert metrics["signal_aware_ridge"]["overall"]["mae_m"] == pytest.approx(5.0)
    assert metrics["persistence"]["overall"]["n"] == 1
    assert metrics["persistence"]["overall"]["mae_m"] == pytest.approx(10.0)
    assert metrics["majority_class"]["overall"]["accuracy"] == 0
    assert metrics["majority_class"]["overall"]["macro_f1"] == pytest.approx(0.0)
