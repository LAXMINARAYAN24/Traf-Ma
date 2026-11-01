"""Paired, run-level uncertainty summaries for saved Phase 2 experiment predictions."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

from .baseline import _Metrics
from .phase2_report import _row_key
from .training import RESERVED_TRAINING_SEEDS

METRICS = ("mae_m", "rmse_m", "macro_f1", "balanced_accuracy")


class StatisticsError(ValueError):
    """Raised when saved experiments cannot support a paired run-level analysis."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _experiment_key(manifest: dict) -> str:
    method = manifest.get("method")
    if method == "central_pooled_offline_gru":
        return "central_gru"
    if method == "flat_synthetic_iid_fedavg" or manifest.get("iid_control"):
        return "fedavg_iid"
    if method == "hierarchical_sample_weighted_fedavg" or manifest.get("hierarchical"):
        return "hierarchical_rsu_self_update_" + str(bool(manifest.get("rsu_self_update"))).lower()
    if manifest.get("transient_participation"):
        return "fedavg_transient"
    if method == "flat_sample_weighted_fedavg":
        return "fedavg_natural"
    raise StatisticsError(f"unsupported learned experiment method: {method!r}")


def _prediction_rows(path: Path) -> dict[tuple[str, str, float], dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("split") != "test":
                continue
            key = _row_key(row)
            if key in rows:
                raise StatisticsError(f"duplicate test prediction key in {path}: {key}")
            rows[key] = row
    if not rows:
        raise StatisticsError(f"no saved test predictions in {path}")
    return rows


def _metrics_from_rows(rows: list[dict], method: str) -> dict:
    accumulator = _Metrics()
    for row in rows:
        if method == "signal_aware_ridge":
            predicted_queue = float(row["signal_aware_pred_queue_m"])
            predicted_class = int(row["signal_aware_pred_class"])
        else:
            predicted_queue = float(row["pred_queue_m"])
            predicted_class = int(row["pred_class"])
        accumulator.add(
            float(row["true_queue_m"]),
            predicted_queue,
            int(row["true_class"]),
            predicted_class,
        )
    return accumulator.result()


def _percentile(sorted_values: list[float], percentile: float) -> float:
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_mean(values: list[float], rng: random.Random, replicates: int) -> dict:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "runs": 0}
    draws = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    )
    return {
        "mean": sum(values) / len(values),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
        "runs": len(values),
    }


def _paired_bootstrap(left: list[float], right: list[float],
                      rng: random.Random, replicates: int) -> dict:
    if len(left) != len(right) or not left:
        raise StatisticsError("paired bootstrap requires matching nonempty run-level values")
    differences = [a - b for a, b in zip(left, right)]
    draws = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(replicates)
    )
    low, high = _percentile(draws, 0.025), _percentile(draws, 0.975)
    negative = sum(value <= 0 for value in draws) / replicates
    positive = sum(value >= 0 for value in draws) / replicates
    return {
        "mean_difference": sum(differences) / len(differences),
        "ci95_low": low,
        "ci95_high": high,
        "runs": len(differences),
        "significant_at_0_05": not (low <= 0 <= high),
        "bootstrap_two_sided_p": min(1.0, 2.0 * min(negative, positive)),
    }


def _read_ridge_rows(path: Path, keys: set[tuple]) -> dict[tuple, dict]:
    matched = {}
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("split") == "test" and _row_key(row) in keys:
                key = _row_key(row)
                if key in matched:
                    raise StatisticsError(f"duplicate baseline test prediction key: {key}")
                matched[key] = row
    if set(matched) != keys:
        raise StatisticsError(
            f"baseline covers {len(matched)} of {len(keys)} paired test rows"
        )
    return matched


def run_paired_statistics(
    baseline_run: Union[str, Path],
    experiment_runs: list[Union[str, Path]],
    output: Optional[Union[str, Path]] = None,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20261031,
) -> Path:
    """Summarize repeated methods, resampling test runs rather than overlapping windows."""
    baseline_dir = Path(baseline_run)
    if isinstance(bootstrap_replicates, bool) or bootstrap_replicates < 100:
        raise StatisticsError("bootstrap_replicates must be an integer >= 100")
    if not experiment_runs:
        raise StatisticsError("at least one seeded learned experiment is required")
    grouped: dict[str, list[dict]] = defaultdict(list)
    dataset_ids = set()
    all_test_keys = None
    test_runs = None
    for raw_path in experiment_runs:
        path = Path(raw_path)
        for filename in ("manifest.json", "predictions.csv"):
            if not (path / filename).is_file():
                raise StatisticsError(f"missing {filename}: {path}")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        method = _experiment_key(manifest)
        if manifest.get("method") == "local_offline_trajectory_gru":
            raise StatisticsError("local-client benchmark rows require client-stratified analysis")
        dataset_ids.add(manifest.get("dataset_id"))
        rows = _prediction_rows(path / "predictions.csv")
        keys = set(rows)
        if all_test_keys is None:
            all_test_keys = keys
            test_runs = sorted({key[0] for key in keys})
        elif keys != all_test_keys:
            raise StatisticsError(
                f"test row identities differ across learned runs: {path} "
                f"({len(keys)} rows versus {len(all_test_keys)} expected)"
            )
        seed = manifest.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise StatisticsError(f"experiment manifest has no integer training seed: {path}")
        if seed in RESERVED_TRAINING_SEEDS:
            raise StatisticsError(f"reserved training seed {seed} cannot be analyzed: {path}")
        grouped[method].append({
            "path": path,
            "seed": seed,
            "manifest": manifest,
            "rows": rows,
        })
    if len(dataset_ids) != 1 or None in dataset_ids:
        raise StatisticsError(f"learned experiments do not share one dataset: {sorted(dataset_ids)}")
    if not test_runs:
        raise StatisticsError("no test run IDs are represented in experiment predictions")
    for method, runs in grouped.items():
        seeds = [item["seed"] for item in runs]
        if len(set(seeds)) != len(seeds):
            raise StatisticsError(f"duplicate training seeds for {method}: {seeds}")
        if len(seeds) < 5:
            raise StatisticsError(f"{method} has only {len(seeds)} training seeds; at least five required")

    baseline_manifest_path = baseline_dir / "manifest.json"
    baseline_predictions_path = baseline_dir / "predictions.csv"
    if not baseline_manifest_path.is_file() or not baseline_predictions_path.is_file():
        raise StatisticsError(f"baseline artifact is incomplete: {baseline_dir}")
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    if baseline_manifest.get("dataset_id") not in dataset_ids:
        raise StatisticsError("baseline dataset does not match learned experiment dataset")
    assert all_test_keys is not None
    ridge_rows = _read_ridge_rows(baseline_predictions_path, all_test_keys)

    by_seed = {}
    by_run = {}
    prediction_rows_by_method_run: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for method, runs in sorted(grouped.items()):
        prediction_rows_by_method_run[method] = {}
        for item in sorted(runs, key=lambda value: value["seed"]):
            rows = list(item["rows"].values())
            overall = _metrics_from_rows(rows, method)
            by_seed[f"{method}:{item['seed']}"] = {
                "method": method,
                "seed": item["seed"],
                "n": overall["n"],
                **{name: overall[name] for name in METRICS},
                "test_runs": len({row["run_id"] for row in rows}),
            }
            grouped_by_run: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                grouped_by_run[row["run_id"]].append(row)
            prediction_rows_by_method_run[method][str(item["seed"])] = dict(grouped_by_run)
            for run_id, run_rows in sorted(grouped_by_run.items()):
                result = _metrics_from_rows(run_rows, method)
                by_run[f"{method}:{item['seed']}:{run_id}"] = {
                    "method": method,
                    "seed": item["seed"],
                    "run_id": run_id,
                    **{name: result[name] for name in METRICS},
                    "n": result["n"],
                }

    ridge_by_run: dict[str, list[dict]] = defaultdict(list)
    for row in ridge_rows.values():
        ridge_by_run[row["run_id"]].append(row)
    ridge_run_metrics = {
        run_id: _metrics_from_rows(rows, "signal_aware_ridge")
        for run_id, rows in ridge_by_run.items()
    }
    if set(ridge_run_metrics) != set(test_runs):
        raise StatisticsError("matched Ridge baseline and learned predictions do not cover the same test runs")

    method_run_values: dict[str, dict[str, dict[str, list[float]]]] = {}
    method_run_counts: dict[str, dict[str, list[int]]] = {}
    for method, seeds in prediction_rows_by_method_run.items():
        metric_by_run = {name: defaultdict(list) for name in METRICS}
        count_by_run = defaultdict(list)
        for seed, run_map in seeds.items():
            if set(run_map) != set(test_runs):
                raise StatisticsError(
                    f"{method} seed {seed} covers {len(run_map)} of {len(test_runs)} test runs"
                )
            for run_id in test_runs:
                result = _metrics_from_rows(run_map[run_id], method)
                count_by_run[run_id].append(result["n"])
                for metric in METRICS:
                    metric_by_run[metric][run_id].append(result[metric])
        method_run_values[method] = {
            metric: {
                run_id: [sum(metric_by_run[metric][run_id]) / len(metric_by_run[metric][run_id])]
                for run_id in test_runs
            }
            for metric in METRICS
        }
        method_run_counts[method] = {
            run_id: count_by_run[run_id] for run_id in test_runs
        }

    mean_by_run = {
        "signal_aware_ridge": {
            metric: {run_id: [ridge_run_metrics[run_id][metric]] for run_id in test_runs}
            for metric in METRICS
        }
    }
    mean_by_run.update(method_run_values)
    methods = sorted(mean_by_run)
    rng = random.Random(bootstrap_seed)
    run_summary = {}
    for method in methods:
        run_summary[method] = {}
        for metric in METRICS:
            values = [mean_by_run[method][metric][run_id][0] for run_id in test_runs]
            run_summary[method][metric] = _bootstrap_mean(values, rng, bootstrap_replicates)

    paired = []
    for method_index, left_method in enumerate(methods):
        for right_method in methods[method_index + 1:]:
            for metric in METRICS:
                left = [mean_by_run[left_method][metric][run_id][0] for run_id in test_runs]
                right = [mean_by_run[right_method][metric][run_id][0] for run_id in test_runs]
                comparison = _paired_bootstrap(left, right, rng, bootstrap_replicates)
                paired.append({
                    "left_method": left_method,
                    "right_method": right_method,
                    "metric": metric,
                    "difference_definition": "left minus right",
                    **comparison,
                })

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output) if output else Path("runs/phase2") / f"paired_statistics_{timestamp}"
    if output_dir.exists():
        raise StatisticsError(f"statistics output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(json.dumps({
        "dataset_id": next(iter(dataset_ids)),
        "test_run_ids": test_runs,
        "test_run_count": len(test_runs),
        "test_rows_per_seed": len(all_test_keys),
        "training_seeds_by_method": {
            method: sorted(item["seed"] for item in runs) for method, runs in sorted(grouped.items())
        },
        "bootstrap": {
            "unit": "held-out test simulation run; overlapping windows are nested within each run",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence_interval": "percentile 95%",
        },
        "run_level_metrics": run_summary,
        "paired_differences": paired,
        "baseline_manifest_sha256": _sha256(baseline_manifest_path),
        "learned_runs": [
            {"path": str(item["path"]), "manifest_sha256": _sha256(item["path"] / "manifest.json"),
             "predictions_sha256": _sha256(item["path"] / "predictions.csv")}
            for method in sorted(grouped) for item in sorted(grouped[method], key=lambda value: value["seed"])
        ],
    }, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "seed_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["method", "seed", "n", "test_runs", *METRICS]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(by_seed.values())
    with (output_dir / "run_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["method", "seed", "run_id", "n", *METRICS]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(by_run.values())
    with (output_dir / "paired_differences.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ["left_method", "right_method", "metric", "difference_definition",
                  "mean_difference", "ci95_low", "ci95_high", "runs",
                  "significant_at_0_05", "bootstrap_two_sided_p"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)

    markdown = [
        "# Paired Phase 2 evaluation",
        "",
        f"- Dataset: `{next(iter(dataset_ids))}`.",
        f"- Fixed test rows per training seed: {len(all_test_keys)} across {len(test_runs)} held-out simulation runs.",
        f"- Training seeds by method: `{json.dumps({method: sorted(item['seed'] for item in runs) for method, runs in sorted(grouped.items())}, sort_keys=True)}`.",
        "- Test rows are identical across all learned methods and training seeds; bootstrap resampling is over held-out runs, not overlapping windows.",
        f"- 95% percentile bootstrap intervals use {bootstrap_replicates} run resamples (seed {bootstrap_seed}).",
        "",
        "| Method | Test runs | Run-macro MAE (m), 95% CI | Run-macro RMSE (m), 95% CI | Run-macro macro-F1, 95% CI | Run-macro balanced accuracy, 95% CI |",
        "|---|---:|---|---|---|---|",
    ]
    for method in methods:
        summary = run_summary[method]
        count_runs = summary["mae_m"]["runs"]
        cells = []
        for metric in METRICS:
            item = summary[metric]
            cells.append(
                "n/a" if item["mean"] is None else
                f"{item['mean']:.4f} [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}]"
            )
        markdown.append(f"| {method} | {count_runs} | " + " | ".join(cells) + " |")
    markdown.extend([
        "",
        "## Paired differences",
        "",
        "Each difference is the left method minus the right method, paired by the same held-out run and averaged over training seeds. A 95% interval containing zero is reported as not statistically distinguishable by this run-bootstrap analysis.",
        "",
        "| Left | Right | Metric | Difference, 95% CI | Runs | Conclusion |",
        "|---|---|---|---|---:|---|",
    ])
    for item in paired:
        conclusion = "significant (95% CI excludes 0)" if item["significant_at_0_05"] else "not significant (95% CI includes 0)"
        markdown.append(
            f"| {item['left_method']} | {item['right_method']} | {item['metric']} | "
            f"{item['mean_difference']:.4f} [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | "
            f"{item['runs']} | {conclusion} |"
        )
    markdown.extend([
        "",
        "## Artifacts",
        "",
        "- `seed_metrics.csv`: overall saved-prediction metrics per method and training seed.",
        "- `run_metrics.csv`: metrics by method, training seed and held-out run.",
        "- `paired_differences.csv`: run-paired bootstrap comparisons.",
        "- `summary.json`: test identities, input hashes, run-level confidence intervals and all paired results.",
    ])
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return output_dir
