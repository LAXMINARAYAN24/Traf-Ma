import csv

from trafma.phase2_report import _experiment_label, _matched_baseline_metrics


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
    assert _experiment_label({"method": "hierarchical_sample_weighted_fedavg", "hierarchical": True}) \
        == "Hierarchical FedAvg"
    assert _experiment_label({"method": "flat_synthetic_iid_fedavg", "iid_control": True}) \
        == "Synthetic IID FedAvg"
