import csv
import json

from trafma.phase2_statistics import run_paired_statistics


def _write_predictions(path, method, test_rows):
    fields = [
        "split", "run_id", "scenario", "client_id", "client_type", "approach", "end_t",
        "true_queue_m", "true_class", "pred_queue_m", "pred_class",
        "signal_aware_pred_queue_m", "signal_aware_pred_class",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in test_rows:
            writer.writerow({
                **row,
                "pred_queue_m": row["true_queue_m"] + (1 if method == "central" else 3),
                "pred_class": row["true_class"],
                "signal_aware_pred_queue_m": row["true_queue_m"] + 2,
                "signal_aware_pred_class": row["true_class"],
            })


def test_paired_statistics_bootstraps_runs_and_checks_five_seeds(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text(json.dumps({"dataset_id": "phase2_v2"}), encoding="utf-8")
    baseline_rows = [
        {
            "split": "test", "run_id": f"run_{run}", "scenario": "free",
            "client_id": f"run_{run}/rsu_N", "client_type": "rsu", "approach": "N",
            "end_t": 10, "true_queue_m": float(run * 10), "true_class": run % 3,
        }
        for run in range(1, 4)
    ]
    _write_predictions(baseline / "predictions.csv", "baseline", baseline_rows)

    experiments = []
    for seed in range(5):
        run_dir = tmp_path / f"central_{seed}"
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text(json.dumps({
            "dataset_id": "phase2_v2",
            "method": "central_pooled_offline_gru",
            "seed": 1000 + seed,
        }), encoding="utf-8")
        _write_predictions(run_dir / "predictions.csv", "central", baseline_rows)
        experiments.append(run_dir)

    output = run_paired_statistics(
        baseline, experiments, output=tmp_path / "paired",
        bootstrap_replicates=200, bootstrap_seed=33,
    )

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["test_run_count"] == 3
    assert summary["test_rows_per_seed"] == 3
    assert summary["training_seeds_by_method"] == {"central_gru": [1000, 1001, 1002, 1003, 1004]}
    assert summary["bootstrap"]["unit"].startswith("held-out test simulation run")
    rows = list(csv.DictReader((output / "run_metrics.csv").open(encoding="utf-8")))
    assert len(rows) == 15
    paired_rows = list(csv.DictReader((output / "paired_differences.csv").open(encoding="utf-8")))
    central_vs_ridge = next(
        row for row in paired_rows
        if {row["left_method"], row["right_method"]} == {"central_gru", "signal_aware_ridge"}
        and row["metric"] == "mae_m"
    )
    assert float(central_vs_ridge["mean_difference"]) == -1.0
    assert central_vs_ridge["significant_at_0_05"] == "True"
    assert "not statistically distinguishable" in (output / "summary.md").read_text(encoding="utf-8") \
        or "significant (95% CI excludes 0)" in (output / "summary.md").read_text(encoding="utf-8")
