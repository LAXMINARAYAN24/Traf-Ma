"""Generate Phase 2 comparison reports from saved dataset and experiment artifacts."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

from .baseline import _Metrics


class ReportError(ValueError):
    """Raised when a requested report input is missing or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_test_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row["split"] == "test"]


def _row_key(row: dict) -> tuple[str, str, float]:
    return row["run_id"], row["client_id"], round(float(row["end_t"]), 6)


def _experiment_label(manifest: dict) -> str:
    if manifest.get("evaluation_mode") == "representative_stratified_local_only":
        return "Independent local-only GRU (representative stratified)"
    if manifest.get("evaluation_mode") == "exhaustive_local_only":
        return "Independent local-only GRU (exhaustive)"
    if manifest.get("method") == "local_offline_trajectory_gru":
        return f"Local GRU ({manifest.get('local_training_client')})"
    if manifest.get("method") == "central_pooled_offline_gru":
        return "Central GRU"
    if manifest.get("hierarchical"):
        if manifest.get("rsu_self_update") is False:
            return "Hierarchical FedAvg (RSU self-update off)"
        return "Hierarchical FedAvg (RSU self-update on)"
    if manifest.get("iid_control"):
        return "Synthetic IID FedAvg"
    if manifest.get("transient_participation"):
        return "Transient FedAvg"
    return "Flat FedAvg (natural)"


def _paired_statistics_section(summary: dict, dataset_id: str, test_runs: set[str]) -> tuple[list[str], str]:
    if summary.get("dataset_id") != dataset_id:
        raise ReportError("paired statistics dataset does not match the report dataset")
    if set(summary.get("test_run_ids", [])) != test_runs:
        raise ReportError("paired statistics held-out runs do not match the report test runs")
    if summary.get("bootstrap", {}).get("unit") != (
        "held-out test simulation run; overlapping windows are nested within each run"
    ):
        raise ReportError("paired statistics must bootstrap held-out simulation runs")

    labels = {
        "central_gru": "Central GRU",
        "fedavg_iid": "Synthetic IID FedAvg",
        "fedavg_natural": "Flat FedAvg (natural)",
        "hierarchical_rsu_self_update_false": "Hierarchical FedAvg (RSU self-update off)",
        "hierarchical_rsu_self_update_true": "Hierarchical FedAvg (RSU self-update on)",
        "signal_aware_ridge": "Signal-aware Ridge",
    }
    run_metrics = summary.get("run_level_metrics", {})
    markdown = [
        "## Paired run-level uncertainty",
        "",
        f"Run-bootstrap 95% percentile confidence intervals use "
        f"{summary['bootstrap']['replicates']} resamples over {len(test_runs)} held-out simulation runs "
        f"(seed {summary['bootstrap']['seed']}); overlapping windows are not resampled independently.",
        "",
        "| Method | MAE (m), mean [95% CI] | RMSE (m), mean [95% CI] | Macro-F1, mean [95% CI] | Balanced accuracy, mean [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for method, metrics in sorted(run_metrics.items()):
        cells = []
        for metric in ("mae_m", "rmse_m", "macro_f1", "balanced_accuracy"):
            result = metrics[metric]
            cells.append(
                f"{result['mean']:.4f} [{result['ci95_low']:.4f}, {result['ci95_high']:.4f}]"
            )
        markdown.append(f"| {labels.get(method, method)} | " + " | ".join(cells) + " |")
    markdown.extend([
        "",
        "Paired differences are left minus right, averaged over training seeds within each held-out run. "
        "An interval containing zero is not statistically distinguishable at the 0.05 level by this bootstrap.",
        "",
        "| Left | Right | Metric | Difference, mean [95% CI] | Test runs | Conclusion |",
        "|---|---|---|---|---:|---|",
    ])
    paired_rows = []
    for item in summary.get("paired_differences", []):
        left = labels.get(item["left_method"], item["left_method"])
        right = labels.get(item["right_method"], item["right_method"])
        significant = bool(item["significant_at_0_05"])
        conclusion = "significant" if significant else "not significant"
        interval = (
            f"{item['mean_difference']:.4f} "
            f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}]"
        )
        markdown.append(
            f"| {left} | {right} | {item['metric']} | {interval} | {item['runs']} | {conclusion} |"
        )
        paired_rows.append(
            "<tr>" + "".join(
                f"<td>{html.escape(str(value))}</td>"
                for value in (left, right, item["metric"], interval, item["runs"], conclusion)
            ) + "</tr>"
        )
    html_section = (
        "<h2>Paired run-level uncertainty</h2>"
        f"<p>Run-bootstrap 95% percentile intervals use {summary['bootstrap']['replicates']} "
        f"resamples over {len(test_runs)} held-out runs, not overlapping windows.</p>"
        "<table><thead><tr><th>Left</th><th>Right</th><th>Metric</th>"
        "<th>Difference, mean [95% CI]</th><th>Runs</th><th>Conclusion</th></tr></thead><tbody>"
        + "".join(paired_rows)
        + "</tbody></table>"
    )
    return markdown, html_section


def _matched_baseline_metrics(predictions_path: Path, test_keys: set[tuple]) -> dict:
    ridge = _Metrics()
    persistence = _Metrics()
    majority = _Metrics()
    with predictions_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] != "test" or _row_key(row) not in test_keys:
                continue
            queue = float(row["true_queue_m"])
            label = int(row["true_class"])
            ridge.add(
                queue,
                float(row["signal_aware_pred_queue_m"]),
                label,
                int(row["signal_aware_pred_class"]),
            )
            majority.add(
                queue, queue, label, int(row["majority_pred_class"])
            )
            if row["persistence_pred_queue_m"]:
                persistence.add(
                    queue,
                    float(row["persistence_pred_queue_m"]),
                    label,
                    int(row["persistence_pred_class"]),
                )
    majority_result = majority.result()
    majority_result.pop("mae_m")
    majority_result.pop("rmse_m")
    return {
        "signal_aware_ridge": ridge.result() if ridge.n else None,
        "persistence": persistence.result() if persistence.n else None,
        "majority_class": majority_result if majority.n else None,
    }


def _jensen_shannon(left: list[int], right: list[int]) -> float:
    left_total = sum(left)
    right_total = sum(right)
    if not left_total or not right_total:
        return 0.0
    p = [value / left_total for value in left]
    q = [value / right_total for value in right]
    midpoint = [(a + b) / 2 for a, b in zip(p, q)]
    divergence = 0.0
    for distribution in (p, q):
        for value, mean in zip(distribution, midpoint):
            if value:
                divergence += 0.5 * value * math.log2(value / mean)
    return divergence


def _training_label_skew(dataset_manifest: dict) -> dict:
    clients = [
        client for client in dataset_manifest.get("clients", {}).values()
        if client.get("split") == "train"
    ]
    global_hist = [sum(client.get("label_hist", [0, 0, 0])[index] for client in clients)
                   for index in range(3)]
    result = {}
    for dimension in ("approach", "type", "scenario"):
        grouped = defaultdict(lambda: [0, 0, 0])
        for client in clients:
            values = client.get("label_hist", [0, 0, 0])
            totals = grouped[str(client[dimension])]
            for index in range(3):
                totals[index] += values[index]
        result[dimension] = {
            key: {
                "jensen_shannon_divergence_bits": _jensen_shannon(hist, global_hist),
                "label_hist": hist,
                "clients": sum(str(client[dimension]) == key for client in clients),
            }
            for key, hist in sorted(grouped.items())
        }
    return {"global_train_label_hist": global_hist, "groups": result}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _client_distribution_summary(clients: list[dict]) -> dict:
    windows = [float(client["windows"]) for client in clients]
    divergences = [float(client["jsd_bits"]) for client in clients]
    if not clients:
        return {"clients": 0, "windows": 0, "median_windows": None, "iqr_windows": None,
                "p90_windows": None, "median_jsd_bits": None, "iqr_jsd_bits": None,
                "p90_jsd_bits": None, "label_hist": [0, 0, 0]}
    return {
        "clients": len(clients),
        "windows": sum(int(client["windows"]) for client in clients),
        "median_windows": _percentile(windows, 0.5),
        "iqr_windows": _percentile(windows, 0.75) - _percentile(windows, 0.25),
        "p90_windows": _percentile(windows, 0.9),
        "median_jsd_bits": _percentile(divergences, 0.5),
        "iqr_jsd_bits": _percentile(divergences, 0.75) - _percentile(divergences, 0.25),
        "p90_jsd_bits": _percentile(divergences, 0.9),
        "label_hist": [
            sum(int(client["label_hist"][index]) for client in clients)
            for index in range(3)
        ],
    }


def _manifest_histogram(histogram: dict) -> list[int]:
    return [int(histogram.get(str(index), histogram.get(index, 0))) for index in range(3)]


def _client_label_distribution_analysis(dataset_manifest: dict, experiments: list[dict]) -> dict:
    dataset_clients = dataset_manifest.get("clients", {})
    eligible = [
        (client_id, client) for client_id, client in dataset_clients.items()
        if client.get("split") == "train" and client.get("trainable")
    ]
    global_hist = [
        sum(int(client.get("label_hist", [0, 0, 0])[index]) for _, client in eligible)
        for index in range(3)
    ]
    natural_clients = []
    for client_id, client in sorted(eligible):
        histogram = [int(value) for value in client.get("label_hist", [0, 0, 0])]
        natural_clients.append({
            "client_id": client_id,
            "client_type": client["type"],
            "approach": client["approach"],
            "scenario": client["scenario"],
            "windows": int(client["windows"]),
            "label_hist": histogram,
            "jsd_bits": _jensen_shannon(histogram, global_hist),
        })

    grouped = {}
    for dimension, key in (
        ("client_type", "client_type"),
        ("approach", "approach"),
        ("scenario", "scenario"),
    ):
        groups: dict[str, list[dict]] = defaultdict(list)
        for client in natural_clients:
            groups[client[key]].append(client)
        grouped[dimension] = {
            name: _client_distribution_summary(values)
            for name, values in sorted(groups.items())
        }

    natural_runs = [
        item for item in experiments
        if item["manifest"].get("method") == "flat_sample_weighted_fedavg"
        and not item["manifest"].get("hierarchical")
        and not item["manifest"].get("iid_control")
        and not item["manifest"].get("transient_participation")
    ]
    iid_runs = [item for item in experiments if item["manifest"].get("iid_control")]
    iid_comparisons = []
    for iid_item in iid_runs:
        iid_manifest = iid_item["manifest"]
        source_ids = iid_manifest.get("source_trajectory_client_ids") or []
        natural_item = next((
            item for item in natural_runs
            if item["manifest"].get("seed") == iid_manifest.get("seed")
            and item["manifest"].get("cohort_seed") == iid_manifest.get("cohort_seed")
            and set(item["manifest"].get("selected_client_ids", [])) == set(source_ids)
        ), None)
        if natural_item is None or not source_ids:
            continue

        natural_manifest = natural_item["manifest"]
        natural_counts = natural_manifest.get("selected_client_label_counts", {})
        iid_counts = iid_manifest.get("selected_client_label_counts", {})
        if not natural_counts:
            natural_counts = {
                client_id: {
                    str(index): int(value)
                    for index, value in enumerate(dataset_clients[client_id]["label_hist"])
                }
                for client_id in source_ids if client_id in dataset_clients
            }
        if set(natural_counts) != set(source_ids) or not iid_counts:
            continue
        natural_entries = []
        for client_id in sorted(source_ids):
            histogram = _manifest_histogram(natural_counts[client_id])
            metadata = dataset_clients[client_id]
            natural_entries.append({
                "client_id": client_id,
                "client_type": metadata["type"],
                "approach": metadata["approach"],
                "scenario": metadata["scenario"],
                "windows": sum(histogram),
                "label_hist": histogram,
            })
        iid_entries = []
        for client_id, counts in sorted(iid_counts.items()):
            histogram = _manifest_histogram(counts)
            iid_entries.append({
                "client_id": client_id,
                "client_type": "synthetic_iid",
                "approach": "mixed",
                "scenario": "mixed",
                "windows": sum(histogram),
                "label_hist": histogram,
            })
        natural_pool = [
            sum(client["label_hist"][index] for client in natural_entries)
            for index in range(3)
        ]
        iid_pool = [
            sum(client["label_hist"][index] for client in iid_entries)
            for index in range(3)
        ]
        if natural_pool != iid_pool:
            raise ReportError(
                f"natural and IID training cohorts differ for seed {iid_manifest.get('seed')}"
            )
        for client in natural_entries:
            client["jsd_bits"] = _jensen_shannon(client["label_hist"], natural_pool)
        for client in iid_entries:
            client["jsd_bits"] = _jensen_shannon(client["label_hist"], iid_pool)
        iid_comparisons.append({
            "training_seed": iid_manifest.get("seed"),
            "cohort_seed": iid_manifest.get("cohort_seed"),
            "source_client_ids": sorted(source_ids),
            "pooled_label_hist": natural_pool,
            "natural_selected_clients": natural_entries,
            "iid_partitions": iid_entries,
            "natural_summary": _client_distribution_summary(natural_entries),
            "iid_summary": _client_distribution_summary(iid_entries),
        })

    return {
        "divergence_metric": "Jensen-Shannon divergence in bits (base 2) from the corresponding pooled class distribution",
        "natural_train_client_count": len(natural_clients),
        "global_train_label_hist": global_hist,
        "natural_train_clients": natural_clients,
        "natural_group_summaries": grouped,
        "matched_natural_vs_iid_cohorts": iid_comparisons,
    }


def _write_plots(dataset_dir: Path, experiments: list[dict], output_dir: Path,
                 matched_baseline: dict, label_skew: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ReportError(f"Phase 2 report plots require installed matplotlib: {exc}") from exc

    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    paths = []
    colors = {"queue_mae_m": "#2563eb", "macro_f1": "#16a34a"}
    comparison = []
    for metric_name, result in matched_baseline.items():
        if result is not None:
            comparison.append((metric_name.replace("_", " ").title(), result))
    for item in experiments:
        if item["manifest"].get("method") != "local_offline_trajectory_gru":
            comparison.append((item["label"], item["metrics"]["test"]["overall"]))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, key, title in (
        (axes[0], "mae_m", "Queue MAE (m)"),
        (axes[1], "macro_f1", "Congestion macro-F1"),
    ):
        selected = [(label, metrics) for label, metrics in comparison
                    if key in metrics and metrics[key] is not None]
        labels = [label for label, _ in selected]
        values = [metrics[key] for _, metrics in selected]
        bars = axis.bar(range(len(labels)), values, color="#457b9d")
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Matched held-out test rows; overlapping windows are not independent")
    figure.tight_layout()
    path = plots_dir / "matched_metrics.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, dimension in zip(axes, ("approach", "type", "scenario")):
        groups = label_skew["groups"][dimension]
        names = list(groups)
        values = [groups[name]["jensen_shannon_divergence_bits"] for name in names]
        axis.bar(names, values, color="#e09f3e")
        axis.set_title(f"Train label skew by {dimension}")
        axis.set_ylabel("Jensen-Shannon divergence (bits)")
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = plots_dir / "label_skew_js.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    matrix_items = [
        item for item in experiments
        if item["manifest"].get("method") != "local_offline_trajectory_gru"
    ]
    figure, axes = plt.subplots(1, max(1, len(matrix_items)), figsize=(4.2 * max(1, len(matrix_items)), 4))
    if len(matrix_items) == 1:
        axes = [axes]
    for axis, item in zip(axes, matrix_items):
        matrix = np.asarray(item["metrics"]["test"]["overall"]["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(item["label"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(3), ["Free", "Moderate", "Heavy"])
        axis.set_yticks(range(3), ["Free", "Moderate", "Heavy"])
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center")
    figure.suptitle("Held-out confusion matrices")
    figure.tight_layout()
    path = plots_dir / "confusion_matrices.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    figure, axis = plt.subplots(figsize=(8, 5))
    for item in experiments:
        method = item["manifest"].get("method", "")
        if method == "central_pooled_offline_gru":
            history_path = item["path"] / "training_history.csv"
            if history_path.is_file():
                with history_path.open(newline="", encoding="utf-8") as file:
                    history = list(csv.DictReader(file))
                axis.plot(
                    [int(row["epoch"]) for row in history],
                    [float(row["validation_loss"]) for row in history],
                    marker="o", label=f"{item['label']} validation",
                )
        elif "rounds" in item["manifest"]:
            rounds_path = item["path"] / "rounds.json"
            if rounds_path.is_file():
                rounds = json.loads(rounds_path.read_text(encoding="utf-8"))
                axis.plot(
                    [row["round_id"] for row in rounds],
                    [row["validation_loss"] for row in rounds],
                    marker=".", label=f"{item['label']} validation",
                )
    axis.set_xlabel("Epoch / communication round")
    axis.set_ylabel("Validation multitask loss")
    axis.set_title("Validation-only training curves")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = plots_dir / "validation_curves.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    federated = [item for item in experiments if "rounds.json" in {
        file.name for file in item["path"].iterdir()
    }]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for item in federated:
        rounds = json.loads((item["path"] / "rounds.json").read_text(encoding="utf-8"))
        x = [row["round_id"] for row in rounds]
        label = item["label"]
        axes[0, 0].plot(
            x, [row.get("connected_clients", row["eligible_clients"]) for row in rounds],
            marker=".", label=label,
        )
        axes[0, 1].plot(x, [row["serialized_bytes"] for row in rounds], marker=".", label=label)
        axes[1, 0].plot(x, [row["local_training_time_s"] for row in rounds], marker=".", label=label)
        axes[1, 1].plot(x, [row["aggregation_time_s"] for row in rounds], marker=".", label=label)
    for axis, title, ylabel in (
        (axes[0, 0], "Participation", "Connected clients"),
        (axes[0, 1], "Serialized updates", "Bytes (JSON model payload)"),
        (axes[1, 0], "Client compute", "Local training wall seconds"),
        (axes[1, 1], "Coordinator/edge aggregation", "Aggregation wall seconds"),
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Round")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    path = plots_dir / "federation_costs.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    dataset_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    labels_by_split_type = {}
    for client in dataset_manifest.get("clients", {}).values():
        key = (client["split"], client["type"])
        totals = labels_by_split_type.setdefault(key, [0, 0, 0])
        for index, count in enumerate(client.get("label_hist", [0, 0, 0])):
            totals[index] += count
    figure, axis = plt.subplots(figsize=(9, 5))
    names = [f"{split}\n{client_type}" for split, client_type in labels_by_split_type]
    bottoms = np.zeros(len(names))
    for class_id, class_name in enumerate(("Free", "Moderate", "Heavy")):
        values = [labels_by_split_type[key][class_id] for key in labels_by_split_type]
        axis.bar(names, values, bottom=bottoms, label=class_name)
        bottoms += np.asarray(values)
    axis.set_title("Label distributions by split and client type")
    axis.set_ylabel("Saved overlapping-window count")
    axis.legend()
    figure.tight_layout()
    path = plots_dir / "label_distributions.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    audit_path = dataset_dir / "feature_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        feature_names = sorted(set().union(
            *(set(values) for values in audit["availability_fraction"].values())
        ))
        figure, axis = plt.subplots(figsize=(12, 5))
        width = 0.38
        x = np.arange(len(feature_names))
        for index, client_type in enumerate(("rsu", "vehicle")):
            values = [audit["availability_fraction"][client_type].get(name, 0.0)
                      for name in feature_names]
            axis.bar(x + (index - 0.5) * width, values, width, label=client_type)
        axis.set_xticks(x, feature_names, rotation=55, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Available fraction")
        axis.set_title("Feature availability from saved dataset audit")
        axis.legend()
        figure.tight_layout()
        path = plots_dir / "feature_availability.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path.name)

    central = next((item for item in experiments
                    if item["manifest"].get("method") == "central_pooled_offline_gru"), None)
    if central:
        rows = _load_test_rows(central["path"] / "predictions.csv")
        if len(rows) > 2500:
            rows = random.Random(20261031).sample(rows, 2500)
        truth = [float(row["true_queue_m"]) for row in rows]
        predicted = [float(row["pred_queue_m"]) for row in rows]
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.scatter(truth, predicted, s=8, alpha=0.25)
        axis.plot([0, 150], [0, 150], color="black", linestyle="--")
        axis.set_xlim(0, 150)
        axis.set_ylim(0, 150)
        axis.set_xlabel("True future queue (m)")
        axis.set_ylabel("Predicted future queue (m)")
        axis.set_title(f"Central GRU predicted vs true; n={len(rows)}")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = plots_dir / "central_queue_scatter.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path.name)

    approach_items = [item for item in experiments
                      if item["manifest"].get("method") != "local_offline_trajectory_gru"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    approaches = ("N", "E", "S", "W")
    for item in approach_items:
        rows = item["metrics"]["test"].get("by_approach", {})
        axes[0].plot(
            approaches, [rows.get(approach, {}).get("mae_m") for approach in approaches],
            marker="o", label=item["label"],
        )
        axes[1].plot(
            approaches, [rows.get(approach, {}).get("macro_f1") for approach in approaches],
            marker="o", label=item["label"],
        )
    axes[0].set_title("Queue MAE by approach")
    axes[0].set_ylabel("MAE (m)")
    axes[1].set_title("Congestion macro-F1 by approach")
    for axis in axes:
        axis.set_xlabel("Approach")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Held-out error breakdown by approach")
    figure.tight_layout()
    path = plots_dir / "approach_breakdown.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path.name)

    federated_items = [item for item in experiments
                       if (item["path"] / "client_rounds.csv").is_file()]
    if federated_items:
        figure, axes = plt.subplots(
            len(federated_items), 1, figsize=(12, max(4, 2.8 * len(federated_items)))
        )
        if len(federated_items) == 1:
            axes = [axes]
        for axis, item in zip(axes, federated_items):
            with (item["path"] / "client_rounds.csv").open(newline="", encoding="utf-8") as file:
                first_round = {}
                for row in csv.DictReader(file):
                    if int(row["round_id"]) == 1:
                        first_round[row["client_id"]] = int(row["sample_count"])
            names = list(first_round)
            axis.barh(names, [first_round[name] for name in names], color="#457b9d")
            axis.set_title(f"{item['label']}: distinct windows per replay client")
            axis.set_xlabel("Distinct selected training windows")
            axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        path = plots_dir / "client_sample_counts.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(path.name)

    return paths


def run_phase2_report(dataset: Union[str, Path], baseline_run: Union[str, Path],
                      experiment_runs: list[Union[str, Path]],
                      output: Optional[Union[str, Path]] = None,
                      statistics_run: Optional[Union[str, Path]] = None) -> Path:
    """Build Markdown/HTML and plots using only immutable saved experiment outputs."""
    dataset_dir = Path(dataset)
    baseline_dir = Path(baseline_run)
    if not (dataset_dir / "dataset_manifest.json").is_file():
        raise ReportError(f"dataset manifest not found: {dataset_dir}")
    baseline_manifest_path = baseline_dir / "manifest.json"
    baseline_predictions = baseline_dir / "predictions.csv"
    if not baseline_manifest_path.is_file() or not baseline_predictions.is_file():
        raise ReportError(f"baseline experiment artifacts are incomplete: {baseline_dir}")
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    if not experiment_runs:
        raise ReportError("at least one learning experiment path is required")
    experiments = []
    for raw_path in experiment_runs:
        path = Path(raw_path)
        required = ("manifest.json", "metrics.json", "predictions.csv")
        if any(not (path / name).is_file() for name in required):
            raise ReportError(f"learning experiment artifacts are incomplete: {path}")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        label = _experiment_label(manifest)
        experiments.append({"path": path, "manifest": manifest, "metrics": metrics, "label": label})

    global_items = [
        item for item in experiments
        if item["manifest"].get("method") != "local_offline_trajectory_gru"
    ]
    if not global_items:
        raise ReportError("at least one pooled or federated experiment is required for matched comparison")
    key_sets = [
        {_row_key(row) for row in _load_test_rows(item["path"] / "predictions.csv")}
        for item in global_items
    ]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ReportError("global-model experiments must use identical held-out test rows")
    matched_baseline = _matched_baseline_metrics(baseline_predictions, key_sets[0])
    ridge = matched_baseline["signal_aware_ridge"]
    if ridge is None or ridge["n"] != len(key_sets[0]):
        raise ReportError("signal-aware Ridge baseline does not cover the global-model test rows")

    dataset_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    statistics_path = Path(statistics_run) if statistics_run is not None else None
    statistics_markdown = []
    statistics_html = ""
    if statistics_path is not None:
        statistics_summary_path = statistics_path / "summary.json"
        if not statistics_summary_path.is_file():
            raise ReportError(f"paired statistics summary not found: {statistics_summary_path}")
        statistics_summary = json.loads(statistics_summary_path.read_text(encoding="utf-8"))
        test_runs = {key[0] for key in key_sets[0]}
        statistics_markdown, statistics_html = _paired_statistics_section(
            statistics_summary,
            str(dataset_manifest.get("dataset_id", dataset_dir.name)),
            test_runs,
        )

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output) if output else Path("runs/phase2") / f"report_phase2_{timestamp}"
    if output_dir.exists():
        raise ReportError(f"report output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    label_skew = _training_label_skew(dataset_manifest)
    client_distributions = _client_label_distribution_analysis(dataset_manifest, experiments)
    plot_names = _write_plots(dataset_dir, experiments, output_dir, matched_baseline, label_skew)
    (output_dir / "label_skew.json").write_text(
        json.dumps(label_skew, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "client_label_distributions.json").write_text(
        json.dumps(client_distributions, indent=2, sort_keys=True), encoding="utf-8"
    )

    comparison = [
        ("Matched signal-aware Ridge", ridge),
    ]
    if matched_baseline["majority_class"] is not None:
        majority = matched_baseline["majority_class"]
        comparison.append(("Matched majority-class diagnostic", majority))
    if matched_baseline["persistence"] is not None:
        comparison.append(("Matched RSU persistence", matched_baseline["persistence"]))
    for item in global_items:
        comparison.append((item["label"], item["metrics"]["test"]["overall"]))
    local_items = [
        item for item in experiments
        if item["manifest"].get("method") == "local_offline_trajectory_gru"
    ]

    markdown = [
        "# TRAF-MA Phase 2 saved-artifact report",
        "",
        f"- Dataset: `{dataset_dir}` ({json.loads((dataset_dir / 'dataset_manifest.json').read_text(encoding='utf-8')).get('dataset_id', dataset_dir.name)}).",
        f"- Baseline artifact: `{baseline_dir}`; method `{baseline_manifest.get('method')}`.",
        f"- Matched global-model test rows: {len(key_sets[0])}; run-level split; overlapping windows are not independent replicates.",
        "- Flat/HFL/IID are offline trajectory replays. Synthetic IID pooling is diagnostic and not a local deployment.",
        "- Scores below are descriptive development results; no test-set tuning was performed.",
        "",
        "| Method | N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    markdown.extend(
        f"| {name} | {metrics['n']} | {metrics.get('mae_m', 'n/a')} "
        f"| {metrics.get('rmse_m', 'n/a')} "
        f"| {metrics['macro_f1']:.4f} | {metrics['balanced_accuracy']:.4f} |"
        for name, metrics in comparison
    )
    markdown.extend(["", "## Local-only stratum results", ""])
    if local_items:
        markdown.extend([
            "| Local model | Client | Test N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for item in local_items:
            result = item["metrics"]["test"]["overall"]
            markdown.append(
                f"| {item['label']} | `{item['manifest'].get('local_training_client')}` | {result['n']} "
                f"| {result['mae_m']:.4f} | {result['rmse_m']:.4f} | {result['macro_f1']:.4f} "
                f"| {result['balanced_accuracy']:.4f} |"
            )
        markdown.append("")
    for item in global_items:
        markdown.extend([
            f"## {item['label']} breakdown",
            "",
        ])
        if statistics_markdown:
            markdown.extend(statistics_markdown)
            markdown.append("")
        for dimension, title in (
            ("by_scenario", "Scenario"),
            ("by_approach", "Approach"),
            ("by_client_type", "Client type"),
        ):
            groups = item["metrics"]["test"].get(dimension, {})
            if not groups:
                continue
            markdown.extend([
                f"### {title}",
                "",
                "| Group | N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            markdown.extend(
                f"| {group} | {metrics['n']} | {metrics['mae_m']:.4f} | {metrics['rmse_m']:.4f} "
                f"| {metrics['macro_f1']:.4f} | {metrics['balanced_accuracy']:.4f} |"
                for group, metrics in sorted(groups.items())
            )
            markdown.append("")
    markdown.extend([
        "## Interpretation limits",
        "",
        "- Central, flat, IID and hierarchical metrics use identical sampled test rows; their training cohort sizes differ and the federated participant cohorts are not matched across methods.",
        "- Local-only results use held-out rows matched on the selected client's type and approach, not the global-model test cohort; do not compare the local aggregate as if rows were identical.",
        "- Moderate-class recall, class support, and confusion matrices are retained in each `metrics.json`; a zero moderate recall is a material negative result, not an omitted metric.",
        "- Communication bytes are the implementation's JSON model-payload sizes, not measured network traffic. Wall-clock times are from the recorded CPU environment.",
        "- The optional live online federation path is smoke-tested separately; this offline report does not measure live learning performance, radio/network behavior, a formal privacy guarantee, or attack/trust behavior.",
        "",
        "## Generated figures",
        "",
    ])
    markdown.extend(["## Training label-distribution divergence", "",
                     "Jensen-Shannon divergence (base 2, bits) compares each training subgroup's pooled class distribution to all training clients. It is descriptive and uses overlapping-window histograms.", ""])
    for dimension, groups in label_skew["groups"].items():
        markdown.extend([
            f"### {dimension.title()}",
            "",
            "| Group | Clients | Windows by class [free, moderate, heavy] | JS divergence (bits) |",
            "|---|---:|---|---:|",
        ])
        markdown.extend(
            f"| {key} | {value['clients']} | {value['label_hist']} | "
            f"{value['jensen_shannon_divergence_bits']:.6f} |"
            for key, value in groups.items()
        )
        markdown.append("")
    markdown.extend([
        "## Per-client label and sample distributions",
        "",
        f"Per-client records for all {client_distributions['natural_train_client_count']} eligible natural training clients are saved in `client_label_distributions.json`. The declared divergence is base-2 Jensen-Shannon divergence in bits from the applicable pooled class distribution. `p90` is the worst-decile threshold, not a maximum.",
        "",
    ])
    for dimension, groups in client_distributions["natural_group_summaries"].items():
        markdown.extend([
            f"### Natural clients by {dimension.replace('_', ' ')}",
            "",
            "| Group | Clients | Windows | Median windows (IQR; p90) | Median JSD (IQR; p90), bits | Label histogram [free, moderate, heavy] |",
            "|---|---:|---:|---|---|---|",
        ])
        for name, summary in groups.items():
            markdown.append(
                f"| {name} | {summary['clients']} | {summary['windows']} "
                f"| {summary['median_windows']:.1f} ({summary['iqr_windows']:.1f}; {summary['p90_windows']:.1f}) "
                f"| {summary['median_jsd_bits']:.6f} ({summary['iqr_jsd_bits']:.6f}; {summary['p90_jsd_bits']:.6f}) "
                f"| {summary['label_hist']} |"
            )
        markdown.append("")
    comparisons = client_distributions["matched_natural_vs_iid_cohorts"]
    if comparisons:
        markdown.extend([
            "### Matched selected natural clients versus synthetic IID partitions",
            "",
            "These rows compare the same selected natural training windows before and after the synthetic IID partition. They describe class-distribution homogenization; they do not establish an accuracy benefit.",
            "",
            "| Training seed | Natural clients | IID partitions | Natural median JSD (p90) | IID median JSD (p90) |",
            "|---:|---:|---:|---:|---:|",
        ])
        markdown.extend(
            f"| {item['training_seed']} | {item['natural_summary']['clients']} "
            f"| {item['iid_summary']['clients']} "
            f"| {item['natural_summary']['median_jsd_bits']:.6f} ({item['natural_summary']['p90_jsd_bits']:.6f}) "
            f"| {item['iid_summary']['median_jsd_bits']:.6f} ({item['iid_summary']['p90_jsd_bits']:.6f}) |"
            for item in comparisons
        )
        markdown.append("")
    else:
        markdown.extend([
            "No matched natural/IID training-cohort artifacts were supplied, so no IID divergence comparison is reported.",
            "",
        ])
    markdown.extend(f"- `plots/{name}`" for name in plot_names)
    markdown.extend(["", "## Input experiment artifacts", ""])
    markdown.extend(
        f"- `{item['path']}` — `{item['manifest'].get('experiment_id')}`; "
        f"{item['manifest'].get('selected_train_windows', item['manifest'].get('selected_window_counts', {}).get('train'))} "
        f"selected training windows."
        for item in experiments
    )
    markdown.append("")
    markdown_path = output_dir / "report.md"
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")

    html_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            name, metrics["n"],
            f"{metrics['mae_m']:.4f}" if metrics.get("mae_m") is not None else "n/a",
            f"{metrics['rmse_m']:.4f}" if metrics.get("rmse_m") is not None else "n/a",
            f"{metrics['macro_f1']:.4f}", f"{metrics['balanced_accuracy']:.4f}",
        )) + "</tr>"
        for name, metrics in comparison
    )
    figure_html = "".join(
        f'<figure><img src="plots/{html.escape(name)}" alt="{html.escape(name)}">'
        f"<figcaption>{html.escape(name)}</figcaption></figure>"
        for name in plot_names
    )
    html_content = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>TRAF-MA Phase 2 report</title>"
        "<style>body{font:16px sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.4rem}figure{display:inline-block}"
        "img{max-width:100%;height:auto} .warning{background:#fff4d6;padding:1rem}</style></head><body>"
        "<h1>TRAF-MA Phase 2 saved-artifact report</h1>"
        f"<p>Dataset: <code>{html.escape(str(dataset_dir))}</code>; matched test rows: {len(key_sets[0])}.</p>"
        "<p class=\"warning\">Development results use overlapping windows; they are descriptive and not independent-run confidence estimates.</p>"
        "<table><thead><tr><th>Method</th><th>N</th><th>Queue MAE (m)</th><th>Queue RMSE (m)</th>"
        "<th>Macro-F1</th><th>Balanced accuracy</th></tr></thead><tbody>"
        f"{html_rows}</tbody></table>{statistics_html}<h2>Figures</h2>{figure_html}"
        f"<h2>Interpretation limits</h2><pre>{html.escape(chr(10).join(markdown[markdown.index('## Interpretation limits') + 2:]))}</pre>"
        "</body></html>"
    )
    (output_dir / "report.html").write_text(html_content, encoding="utf-8")
    input_hashes = {
        "dataset_manifest_sha256": _sha256(dataset_dir / "dataset_manifest.json"),
        "baseline_manifest_sha256": _sha256(baseline_manifest_path),
        "baseline_predictions_sha256": _sha256(baseline_predictions),
        "experiments": {
            str(item["path"]): {
                name: _sha256(item["path"] / name)
                for name in ("manifest.json", "metrics.json", "predictions.csv")
            }
            for item in experiments
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps({
        "dataset": str(dataset_dir),
        "baseline_run": str(baseline_dir),
        "experiment_runs": [str(item["path"]) for item in experiments],
        "statistics_run": str(statistics_path) if statistics_path is not None else None,
        "matched_test_rows": len(key_sets[0]),
        "input_sha256": input_hashes,
        "statistics_summary_sha256": (
            _sha256(statistics_path / "summary.json") if statistics_path is not None else None
        ),
        "plot_files": plot_names,
        "report_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }, indent=2, sort_keys=True), encoding="utf-8")
    return output_dir
