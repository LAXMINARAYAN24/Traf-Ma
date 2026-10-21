"""P2.1 driver: run a grid of SUMO scenarios x seeds, then label them and write the dataset manifest."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from .config import ConfigError, config_from_dict, deep_merge
from .dataset import build_dataset_manifest, format_dataset_report


def _run_one(cfg_dict: dict) -> dict:
    from .runner import run_simulation
    cfg = config_from_dict(cfg_dict)
    res = run_simulation(cfg, gui=False, quiet=True)
    return {"run": cfg.run.run_id, "status": res.status}


def load_collect_config(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"collect config not found: {p}")
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    for k in ("scenarios", "splits"):
        if k not in d or not d[k]:
            raise ConfigError(f"collect config needs '{k}'")
    seen = {}
    for sp, seeds in d["splits"].items():
        for s in seeds:
            if s in seen:
                raise ConfigError(f"seed {s} used in both '{seen[s]}' and '{sp}' splits - splits must be disjoint")
            seen[s] = sp
    return d


def collect(path, resume=False, jobs=None, out=print) -> dict:
    d = load_collect_config(path)
    ds_id = d.get("dataset_id", "dataset")
    ds_dir = Path(d.get("output_dir", "data")) / ds_id
    runs_dir = ds_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs or int(d.get("jobs", 1))
    base = d.get("base", {})
    index, todo = [], []
    for sc in d["scenarios"]:
        for split, seeds in d["splits"].items():
            for seed in seeds:
                rid = f"{sc['name']}_s{seed}"
                over = {"run": {"seed": seed, "vehicles": sc["vehicles"], "run_id": rid,
                                "output_dir": str(runs_dir), "message_trace": "exceptions"},
                        "demand": {"profile": sc.get("profile", "balanced")}}
                if "approach_weights" in sc:
                    over["demand"]["approach_weights"] = sc["approach_weights"]
                if "duration_s" in sc:
                    over["run"]["duration_s"] = sc["duration_s"]
                cd = deep_merge(deep_merge(base, over), sc.get("overrides", {}))
                config_from_dict(cd)  # validate early
                index.append({"run": rid, "scenario": sc["name"], "seed": seed, "split": split})
                done = (runs_dir / rid / "manifest.json").is_file() and resume
                if not done:
                    todo.append(cd)
    out(f"[collect] {len(index)} runs ({len(todo)} to simulate) -> {ds_dir}")
    if jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for r in ex.map(_run_one, todo):
                out(f"  done {r['run']}: {r['status']}")
    else:
        for cd in todo:
            r = _run_one(cd)
            out(f"  done {r['run']}: {r['status']}")
    base_full = deep_merge(base, {})
    man = build_dataset_manifest(ds_dir, index, base_full, d["splits"])
    rep = format_dataset_report(man)
    (ds_dir / "dataset_report.txt").write_text(rep, encoding="utf-8")
    out(rep)
    return man
