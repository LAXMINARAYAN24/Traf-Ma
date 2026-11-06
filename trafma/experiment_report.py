"""Generate reusable, image-rich reports from saved TRAF-MA experiment runs."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import random
import shutil
from pathlib import Path
from typing import Union


class ExperimentReportError(ValueError):
    """Raised for incomplete or incompatible saved experiment artifacts."""


def _hash(path: Path) -> str:
    """Handles  hash for the values provided.

    Args:
        path: Value used by this function.

    Returns:
        The result produced by this function.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label(manifest: dict) -> str:
    """Handles  label for the values provided.

    Args:
        manifest: Value used by this function.

    Returns:
        The result produced by this function.
    """
    method = manifest.get("method", "")
    if method == "central_pooled_offline_gru":
        return "Central GRU"
    if manifest.get("hierarchical") and manifest.get("rsu_self_update") is False:
        return "Hierarchical FedAvg (RSU self-update off)"
    if manifest.get("hierarchical"):
        return "Hierarchical FedAvg (RSU self-update on)"
    if method == "flat_sample_weighted_fedavg":
        return "Flat FedAvg"
    return str(method or "Experiment")


def _short_label(run: dict) -> str:
    """Handles  short label for the values provided.

    Args:
        run: Value used by this function.

    Returns:
        The result produced by this function.
    """
    return {
        "Central GRU": "Central",
        "Flat FedAvg": "Flat FedAvg",
        "Hierarchical FedAvg (RSU self-update off)": "HFL (RSU off)",
        "Hierarchical FedAvg (RSU self-update on)": "HFL (RSU on)",
    }.get(run["label"], run["label"])


def _read_run(path: Path) -> dict:
    """Handles  read run for the values provided.

    Args:
        path: Value used by this function.

    Returns:
        The result produced by this function.
    """
    needed = ("manifest.json", "metrics.json", "predictions.csv")
    missing = [name for name in needed if not (path / name).is_file()]
    if missing:
        raise ExperimentReportError(f"incomplete experiment artifacts in {path}: missing {', '.join(missing)}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    with (path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "test"]
    if not rows:
        raise ExperimentReportError(f"no test prediction rows in {path / 'predictions.csv'}")
    return {"path": path, "manifest": manifest, "metrics": metrics,
            "rows": rows, "label": _label(manifest)}


def _row_key(row: dict) -> tuple:
    """Handles  row key for the values provided.

    Args:
        row: Value used by this function.

    Returns:
        The result produced by this function.
    """
    return row["run_id"], row["client_id"], round(float(row["end_t"]), 6)


def _create_plots(runs: list[dict], plots_dir: Path, prefix: str = "") -> list[str]:
    """Handles  create plots for the values provided.

    Args:
        runs: Value used by this function.
        plots_dir: Value used by this function.
        prefix: Value used by this function.

    Returns:
        The result produced by this function.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ExperimentReportError(f"report graphs require matplotlib: {exc}") from exc

    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    suffix = f"_{prefix}" if prefix else ""

    # Aggregate predictive quality.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = [_short_label(run) for run in runs]
    for axis, metric, title, color in (
        (axes[0], "mae_m", "Queue MAE (m)", "#2563eb"),
        (axes[1], "macro_f1", "Congestion macro-F1", "#16a34a"),
    ):
        values = [run["metrics"]["test"]["overall"].get(metric, 0.0) for run in runs]
        bars = axis.bar(range(len(labels)), values, color=color)
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=.25)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Held-out test performance; overlapping windows are not independent")
    fig.tight_layout()
    name = f"overall_metrics{suffix}.png"
    fig.savefig(plots_dir / name, dpi=160)
    plt.close(fig)
    outputs.append(name)

    # Type-stratified model performance.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, client_type, metric, title in (
        (axes[0, 0], "vehicle", "macro_f1", "Vehicle macro-F1"),
        (axes[0, 1], "rsu", "macro_f1", "RSU macro-F1"),
        (axes[1, 0], "vehicle", "mae_m", "Vehicle queue MAE (m)"),
        (axes[1, 1], "rsu", "mae_m", "RSU queue MAE (m)"),
    ):
        vals = [run["metrics"]["test"].get("by_client_type", {}).get(client_type, {}).get(metric)
                for run in runs]
        bars = axis.bar(range(len(labels)), [v or 0 for v in vals], color="#457b9d")
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=.25)
        for bar, value in zip(bars, vals):
            if value is not None:
                axis.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                          f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    name = f"metrics_by_type{suffix}.png"
    fig.savefig(plots_dir / name, dpi=160)
    plt.close(fig)
    outputs.append(name)

    # Row-normalized confusion matrices, common 0–100% color scale.
    if len(runs) == 1:
        fig, single_axis = plt.subplots(1, 1, figsize=(6, 5))
        flat_axes = [single_axis]
        right, bottom, top, wspace, hspace = .84, .13, .88, 0, 0
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
        flat_axes = list(axes.flat)
        right, bottom, top, wspace, hspace = .89, .1, .9, .38, .4
    image = None
    for axis, run in zip(flat_axes, runs):
        matrix = np.asarray(run["metrics"]["test"]["overall"]["confusion_matrix"], dtype=float)
        totals = matrix.sum(axis=1, keepdims=True)
        percentages = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0) * 100
        image = axis.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
        axis.set_title(_short_label(run))
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(range(3), ("Free", "Moderate", "Heavy"), rotation=25, ha="right")
        axis.set_yticks(range(3), ("Free", "Moderate", "Heavy"))
        for (row, col), value in np.ndenumerate(percentages):
            axis.text(col, row, f"{value:.1f}%", ha="center", va="center",
                      color="white" if value >= 55 else "#17212b", fontsize=8)
    for axis in flat_axes[len(runs):]:
        axis.set_visible(False)
    cbar_axis = fig.add_axes([.91 if len(runs) > 1 else .88, .18, .025, .64])
    fig.colorbar(image, cax=cbar_axis, label="Recall within true class (%)")
    fig.suptitle("Row-normalized held-out confusion matrices (shared 0–100% scale)")
    fig.subplots_adjust(right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    name = f"confusion_matrices{suffix}.png"
    fig.savefig(plots_dir / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    outputs.append(name)

    # Validation-only training curves.
    fig, axis = plt.subplots(figsize=(8, 4.8))
    curve_count = 0
    for run in runs:
        history_path = run["path"] / "training_history.csv"
        rounds_path = run["path"] / "rounds.json"
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            if history:
                axis.plot([int(row["epoch"]) for row in history],
                          [float(row["validation_loss"]) for row in history],
                          label=f"{run['label']} (epoch)")
                curve_count += 1
        elif rounds_path.is_file():
            history = json.loads(rounds_path.read_text(encoding="utf-8"))
            if history:
                axis.plot([row["round_id"] for row in history],
                          [float(row["validation_loss"]) for row in history],
                          label=f"{run['label']} (round)")
                curve_count += 1
    axis.set_xlabel("Epoch or communication round")
    axis.set_ylabel("Validation multitask loss")
    axis.set_title("Validation loss during training")
    axis.grid(alpha=.25)
    if curve_count:
        axis.legend(fontsize=8)
    fig.tight_layout()
    name = f"validation_curves{suffix}.png"
    fig.savefig(plots_dir / name, dpi=160)
    plt.close(fig)
    outputs.append(name)

    # Queue forecast scatter for each run (deterministic sample for legibility).
    for run in runs:
        rows = run["rows"]
        if len(rows) > 2500:
            rows = random.Random(20261105).sample(rows, 2500)
        truth = [float(row["true_queue_m"]) for row in rows]
        pred = [float(row["pred_queue_m"]) for row in rows]
        fig, axis = plt.subplots(figsize=(5.4, 5.1))
        axis.scatter(truth, pred, s=9, alpha=.3, color="#2563eb")
        axis.plot([0, 150], [0, 150], "k--", linewidth=1)
        axis.set(xlim=(0, 150), ylim=(0, 150), xlabel="True future queue (m)",
                 ylabel="Predicted future queue (m)", title=f"{run['label']}: queue forecast")
        axis.grid(alpha=.2)
        fig.tight_layout()
        name = f"queue_scatter_{run['path'].name}.png"
        fig.savefig(plots_dir / name, dpi=160)
        plt.close(fig)
        outputs.append(name)
    return outputs


def _row_metrics(run: dict, client_type: str | None = None) -> dict:
    """Handles  row metrics for the values provided.

    Args:
        run: Value used by this function.
        client_type: Value used by this function.

    Returns:
        The result produced by this function.
    """
    metrics = run["metrics"]["test"]
    return (metrics.get("by_client_type", {}).get(client_type)
            if client_type else metrics["overall"])


def _training_windows(run: dict) -> str:
    """Handles  training windows for the values provided.

    Args:
        run: Value used by this function.

    Returns:
        The result produced by this function.
    """
    manifest = run["manifest"]
    if manifest.get("selected_train_windows") is not None:
        return f"{manifest['selected_train_windows']:,}"
    counts = manifest.get("selected_window_counts_by_type", {}).get("train", {})
    if counts:
        return f"{sum(counts.values()):,} ({', '.join(f'{k} {v:,}' for k, v in sorted(counts.items()))})"
    return "not recorded"


def _duration(run: dict) -> str:
    """Handles  duration for the values provided.

    Args:
        run: Value used by this function.

    Returns:
        The result produced by this function.
    """
    value = run["manifest"].get("wall_duration_s")
    if value is None:
        return "not recorded"
    return f"{value / 60:.1f} min"


def _rounds_or_epochs(run: dict) -> str:
    """Handles  rounds or epochs for the values provided.

    Args:
        run: Value used by this function.

    Returns:
        The result produced by this function.
    """
    manifest = run["manifest"]
    if manifest.get("method") == "central_pooled_offline_gru":
        return f"{manifest.get('epochs_run', 'n/a')} epochs"
    if manifest.get("rounds") is not None:
        return f"{len(manifest['rounds'])} rounds"
    return "n/a"


def _metric_table(runs: list[dict], detailed: bool = False) -> str:
    """Handles  metric table for the values provided.

    Args:
        runs: Value used by this function.
        detailed: Value used by this function.

    Returns:
        The result produced by this function.
    """
    lines = ["| Method | Train windows | Test N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        metric = _row_metrics(run)
        lines.append(
            f"| {run['label']} | {_training_windows(run)} | {metric['n']:,} | "
            f"{metric['mae_m']:.3f} | {metric['rmse_m']:.3f} | "
            f"{metric['macro_f1']:.4f} | {metric['balanced_accuracy']:.4f} |"
        )
    if detailed:
        lines.extend(["", "| Method | Type | Test N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for run in runs:
            for client_type in ("vehicle", "rsu"):
                metric = _row_metrics(run, client_type)
                if metric:
                    lines.append(
                        f"| {run['label']} | {client_type} | {metric['n']:,} | "
                        f"{metric['mae_m']:.3f} | {metric['rmse_m']:.3f} | "
                        f"{metric['macro_f1']:.4f} | {metric['balanced_accuracy']:.4f} |"
                    )
    return "\n".join(lines)


def _write_report(runs: list[dict], directory: Path, title: str, plots: list[str], detailed: bool) -> None:
    """Handles  write report for the values provided.

    Args:
        runs: Value used by this function.
        directory: Value used by this function.
        title: Value used by this function.
        plots: Value used by this function.
        detailed: Value used by this function.

    Returns:
        None.
    """
    md = [f"# {title}", "",
          "This report uses only the saved experiment run artifacts supplied to this command. "
          "Scores are descriptive; overlapping sliding windows are not independent samples.", "",
          "## Results", "", _metric_table(runs, detailed=detailed), "", "## Figures", ""]
    md.extend(f"![{Path(name).stem}](plots/{name})" for name in plots)
    md.extend(["", "## Run provenance", ""])
    for run in runs:
        manifest = run["manifest"]
        md.extend([
            f"### {run['label']}", "",
            f"- Run folder: `{run['path']}`",
            f"- Duration: {_duration(run)}; training: {_rounds_or_epochs(run)}.",
            f"- Training window count: {_training_windows(run)}.",
            f"- Test windows: {len(run['rows']):,}.",
            f"- Run manifest SHA-256: `{_hash(run['path'] / 'manifest.json')}`.",
            "",
        ])
    if len(runs) > 1:
        md.extend(["## Limits", "",
                   "Training budgets vary by method where specified in each run manifest. Results use one "
                   "training seed and held-out windows nested within four SUMO simulation runs; they do not "
                   "provide run-level confidence intervals or isolate architecture effects when budgets differ.", ""])
    (directory / "report.md").write_text("\n".join(md), encoding="utf-8")

    style = ("body{font:16px/1.55 system-ui,Segoe UI,Arial,sans-serif;max-width:1200px;margin:2rem auto;"
             "padding:0 1rem;color:#17212b}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;"
             "margin:1rem 0 2rem;font-size:.93rem}th,td{border:1px solid #cbd5df;padding:.55rem;text-align:left}"
             "th{background:#edf3f8}figure{margin:1.5rem 0}img{max-width:100%;height:auto}"
             ".note{background:#f2f6fa;padding:1rem;border-left:4px solid #547994}code{overflow-wrap:anywhere}")
    figures = "".join(
        f'<figure><img src="plots/{html.escape(name)}" alt="{html.escape(Path(name).stem)}">'
        f"<figcaption>{html.escape(Path(name).stem.replace('_', ' ').title())}</figcaption></figure>"
        for name in plots
    )
    html_tables = []
    for block in _metric_table(runs, detailed=detailed).split("\n\n"):
        table_lines = [line for line in block.splitlines() if line.startswith("|")]
        if not table_lines:
            continue
        header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
        rows = [line for line in table_lines[2:]]
        table_html = ["<table><thead><tr>" + "".join(
            f"<th>{html.escape(cell)}</th>" for cell in header
        ) + "</tr></thead><tbody>"]
        for line in rows:
            cols = [cell.strip() for cell in line.strip("|").split("|")]
            table_html.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cols) + "</tr>")
        table_html.append("</tbody></table>")
        html_tables.append("".join(table_html))
    provenance = "".join(
        f"<li><strong>{html.escape(run['label'])}</strong>: <code>{html.escape(str(run['path']))}</code>; "
        f"{_duration(run)}, {_rounds_or_epochs(run)}, {_training_windows(run)} training windows; "
        f"manifest SHA-256 <code>{_hash(run['path'] / 'manifest.json')}</code></li>"
        for run in runs
    )
    html_content = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>{style}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><p class=\"note\">This report uses only the saved experiment run artifacts "
        "supplied to this command. Scores are descriptive; overlapping sliding windows are not independent samples.</p>"
        f"<h2>Results</h2>{''.join(html_tables)}<h2>Figures</h2>{figures}"
        f"<h2>Run provenance</h2><ul>{provenance}</ul>"
        "<h2>Limits</h2><p>Results use one training seed. Held-out windows are nested within four SUMO simulation runs; "
        "they do not provide run-level confidence intervals. Different training budgets do not isolate architecture effects.</p>"
        "</body></html>"
    )
    (directory / "report.html").write_text(html_content, encoding="utf-8")


def generate_experiment_reports(experiments: list[Union[str, Path]],
                               output: Union[str, Path], overwrite: bool = False) -> Path:
    """Write one report per run and a cross-run comparison with the same HTML theme."""
    runs = [_read_run(Path(path)) for path in experiments]
    if not runs:
        raise ExperimentReportError("provide at least one experiment run")
    keys = [{_row_key(row) for row in run["rows"]} for run in runs]
    if len(runs) > 1 and any(run_keys != keys[0] for run_keys in keys[1:]):
        raise ExperimentReportError("all experiment runs must have identical held-out test rows for comparison")
    output_dir = Path(output)
    if output_dir.exists() and not overwrite:
        raise ExperimentReportError(f"report output already exists: {output_dir}; pass --overwrite to update it")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("report.md", "report.html"):
        previous = output_dir / filename
        backup = output_dir / f"{previous.stem}.before_graphs{previous.suffix}"
        if overwrite and previous.is_file() and not backup.exists():
            shutil.copy2(previous, backup)
    combined_plots = _create_plots(runs, output_dir / "plots")
    _write_report(runs, output_dir, "TRAF-MA latest-run comparison report", combined_plots, detailed=True)

    individual_root = output_dir / "individual"
    for run in runs:
        one_dir = individual_root / run["path"].name
        one_dir.mkdir(parents=True, exist_ok=True)
        plots = _create_plots([run], one_dir / "plots", prefix="single")
        _write_report([run], one_dir, f"TRAF-MA run report: {run['label']}", plots, detailed=False)

    (output_dir / "manifest.json").write_text(json.dumps({
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_runs": [str(run["path"]) for run in runs],
        "matched_test_rows": len(keys[0]),
        "plot_files": combined_plots,
        "input_sha256": {
            str(run["path"]): {
                name: _hash(run["path"] / name)
                for name in ("manifest.json", "metrics.json", "predictions.csv")
            } for run in runs
        },
    }, indent=2, sort_keys=True), encoding="utf-8")
    return output_dir
