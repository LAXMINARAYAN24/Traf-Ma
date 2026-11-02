import csv

import pytest

from trafma.phase2_report import (
    _client_label_distribution_analysis,
    _experiment_label,
    _matched_baseline_metrics,
    _paired_statistics_section,
    ReportError,
)
from trafma.local_evaluation import _distribution, _select_representative_clients


def test_matched_baseline_uses_only_requested_rows_and_reports_partial_persistence(tmp_path):
    path = tmp_path / "predictions.csv"
    fields = [
        "split", "run_id", "client_id", "end_t", "true_queue_m", "true_class",
        "persistence_pred_queue_m", "persistence_pred_class",
        "signal_aware_pred_queue_m", "signal_aware_pred_class", "majority_pred_class",
    ]
    rows = [
        {
            "split": "test", "run_id": "run_a", "client_id": "run_a/rsu_N", "end_t": 30,
            "true_queue_m": 20, "true_class": 1, "persistence_pred_queue_m": 10,
            "persistence_pred_class": 0, "signal_aware_pred_queue_m": 30,
            "signal_aware_pred_class": 2, "majority_pred_class": 0,
        },
        {
            "split": "test", "run_id": "run_a", "client_id": "run_a/veh_0", "end_t": 30,
            "true_queue_m": 20, "true_class": 1, "persistence_pred_queue_m": "",
            "persistence_pred_class": "", "signal_aware_pred_queue_m": 20,
            "signal_aware_pred_class": 1, "majority_pred_class": 0,
        },
        {
            "split": "test", "run_id": "run_b", "client_id": "run_b/rsu_E", "end_t": 30,
            "true_queue_m": 0, "true_class": 0, "persistence_pred_queue_m": 0,
            "persistence_pred_class": 0, "signal_aware_pred_queue_m": 0,
            "signal_aware_pred_class": 0, "majority_pred_class": 0,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = _matched_baseline_metrics(
        path, {("run_a", "run_a/rsu_N", 30.0), ("run_a", "run_a/veh_0", 30.0)}
    )

    assert metrics["signal_aware_ridge"]["n"] == 2
    assert metrics["signal_aware_ridge"]["mae_m"] == 5.0
    assert metrics["persistence"]["n"] == 1
    assert metrics["persistence"]["mae_m"] == 10.0
    assert metrics["majority_class"]["n"] == 2


def test_report_labels_identify_hierarchical_and_synthetic_iid_experiments():
    assert _experiment_label({
        "evaluation_mode": "representative_stratified_local_only",
    }) == "Independent local-only GRU (representative stratified)"
    assert _experiment_label({
        "evaluation_mode": "exhaustive_local_only",
    }) == "Independent local-only GRU (exhaustive)"
    assert _experiment_label({
        "method": "hierarchical_sample_weighted_fedavg",
        "hierarchical": True,
        "rsu_self_update": True,
    }) == "Hierarchical FedAvg (RSU self-update on)"
    assert _experiment_label({
        "method": "hierarchical_sample_weighted_fedavg",
        "hierarchical": True,
        "rsu_self_update": False,
    }) == "Hierarchical FedAvg (RSU self-update off)"
    assert _experiment_label({"method": "flat_synthetic_iid_fedavg", "iid_control": True}) \
        == "Synthetic IID FedAvg"


def test_report_includes_validated_run_bootstrap_statistics():
    summary = {
        "dataset_id": "phase2_v2",
        "test_run_ids": ["run_a"],
        "bootstrap": {
            "unit": "held-out test simulation run; overlapping windows are nested within each run",
            "replicates": 1000,
            "seed": 12,
        },
        "run_level_metrics": {
            "central_gru": {
                name: {"mean": 1.0, "ci95_low": 0.5, "ci95_high": 1.5}
                for name in ("mae_m", "rmse_m", "macro_f1", "balanced_accuracy")
            },
        },
        "paired_differences": [{
            "left_method": "central_gru",
            "right_method": "signal_aware_ridge",
            "metric": "mae_m",
            "mean_difference": -1.0,
            "ci95_low": -2.0,
            "ci95_high": 0.0,
            "runs": 1,
            "significant_at_0_05": False,
        }],
    }

    markdown, html = _paired_statistics_section(summary, "phase2_v2", {"run_a"})

    assert any("1.0000 [0.5000, 1.5000]" in line for line in markdown)
    assert any("not significant" in line for line in markdown)
    assert "Paired run-level uncertainty" in html
    assert "not significant" in html


def test_report_rejects_statistics_from_different_test_runs():
    summary = {
        "dataset_id": "phase2_v2",
        "test_run_ids": ["other_run"],
        "bootstrap": {
            "unit": "held-out test simulation run; overlapping windows are nested within each run",
            "replicates": 1000,
            "seed": 12,
        },
    }

    with pytest.raises(ReportError, match="held-out runs"):
        _paired_statistics_section(summary, "phase2_v2", {"run_a"})


def test_local_evaluation_distribution_reports_median_iqr_and_worst_decile():
    rows = [
        {"status": "completed", "mae_m": 1.0},
        {"status": "completed", "mae_m": 2.0},
        {"status": "completed", "mae_m": 10.0},
        {"status": "failed", "mae_m": 99.0},
    ]

    result = _distribution(rows)

    assert result["clients"] == 3
    assert result["median_mae_m"] == 2.0
    assert result["iqr_mae_m"] == 4.5
    assert result["worst_decile_mae_m"] == 8.4


def test_representative_local_selection_is_deterministic_and_covers_dimensions():
    clients = []
    for client_type in ("rsu", "vehicle"):
        for approach in ("N", "E", "S", "W"):
            for scenario in ("free", "heavy"):
                for windows in (16, 40, 100, 200):
                    client_id = f"{client_type}/{approach}/{scenario}/{windows}"
                    clients.append((client_id, {
                        "type": client_type,
                        "approach": approach,
                        "scenario": scenario,
                        "windows": windows,
                        "trainable": True,
                        "split": "train",
                    }))

    first = _select_representative_clients(clients, 32, seed=7)
    second = _select_representative_clients(clients, 32, seed=7)

    assert [client_id for client_id, _ in first] == [client_id for client_id, _ in second]
    assert len(first) == 32
    assert {client["type"] for _, client in first} == {"rsu", "vehicle"}
    assert {client["approach"] for _, client in first} == {"N", "E", "S", "W"}
    assert {client["scenario"] for _, client in first} == {"free", "heavy"}
    assert {client["windows"] for _, client in first} >= {16, 40, 100, 200}


def test_client_distribution_report_compares_matched_natural_and_iid_partitions():
    dataset_manifest = {
        "clients": {
            "train_a": {
                "split": "train", "trainable": True, "type": "vehicle",
                "approach": "N", "scenario": "balanced", "windows": 10,
                "label_hist": [10, 0, 0],
            },
            "train_b": {
                "split": "train", "trainable": True, "type": "vehicle",
                "approach": "S", "scenario": "balanced", "windows": 10,
                "label_hist": [0, 0, 10],
            },
            "test_client": {
                "split": "test", "trainable": True, "type": "vehicle",
                "approach": "N", "scenario": "balanced", "windows": 20,
                "label_hist": [20, 0, 0],
            },
        }
    }
    natural = {
        "manifest": {
            "method": "flat_sample_weighted_fedavg",
            "seed": 12,
            "cohort_seed": 34,
            "selected_client_ids": ["train_a", "train_b"],
        }
    }
    iid = {
        "manifest": {
            "method": "flat_synthetic_iid_fedavg",
            "iid_control": True,
            "seed": 12,
            "cohort_seed": 34,
            "source_trajectory_client_ids": ["train_a", "train_b"],
            "selected_client_label_counts": {
                "iid_0": {"0": 5, "1": 0, "2": 5},
                "iid_1": {"0": 5, "1": 0, "2": 5},
            },
        }
    }

    result = _client_label_distribution_analysis(dataset_manifest, [natural, iid])

    assert result["natural_train_client_count"] == 2
    pair = result["matched_natural_vs_iid_cohorts"][0]
    assert pair["natural_summary"]["windows"] == pair["iid_summary"]["windows"] == 20
    assert pair["natural_summary"]["median_jsd_bits"] > pair["iid_summary"]["median_jsd_bits"]
