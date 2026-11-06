# TRAF-MA: SUMO simulation, datasets and Phase 2 learning

TRAF-MA simulates a four-arm signalised intersection with vehicle agents, four roadside units (RSUs), a lossy message path and an edge coordinator. It generates labelled datasets and trains a GRU congestion predictor, centrally and with three-tier hierarchical federated learning (HFL).

Traffic drives on the **left by default** (Indian norm). Right-hand traffic is one setting away.

## 1. Architecture at a glance

```text
vehicle agents (N)  --V2I-->  RSU-N / RSU-E / RSU-S / RSU-W  --I2C-->  edge coordinator
 (own view, 100 m)            (one per road, 150 m zone)               (validate, dedup, fuse)
```

```text
Federated learning (three tiers)

tier 1   vehicle clients + RSU sensor learners      train locally, send model updates
tier 2   RSU edge aggregators (N, E, S, W)          average their own group
tier 3   global coordinator                         averages the four edge models
```

```text
Dataset flow

SUMO run -> agent logs + evaluator labels -> windows (10 steps -> label 30 s ahead)
         -> data/<dataset>/{train,val,test}/<run>/ -> central GRU, FedAvg, baselines -> reports
```

More detail: [docs/ARCHITECTURE_AS_BUILT.md](docs/ARCHITECTURE_AS_BUILT.md).

## 2. Quick start (Windows PowerShell or Linux/macOS)

Run these from the repository root. Use the activation command for your shell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # includes SUMO through the eclipse-sumo wheel
python -m trafma doctor           # starts, steps and closes a minimal SUMO process
```

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m trafma doctor
```

The following sequence covers the project milestones available on this branch. The retained v1 and 350v250 collection configs use separate dataset IDs. Collect each dataset once, then reuse its split for matched model comparisons.

```bash
python -m trafma doctor
python -m trafma simulate --config configs/phase1.yaml --vehicles 20 --seed 42 --gui
python -m trafma simulate --vehicles 20 --seed 42             # headless
python -m trafma gate --run runs/<run-id> --reproduce --pytest
python -m trafma report --run runs/<run-id>
python -m trafma replay --run runs/<run-id> --every 10
python -m trafma collect --config configs/data.yaml           # v1: 24 runs
python -m trafma baseline --config configs/learning.yaml
python -m trafma train --method central --config configs/learning.yaml
python -m trafma train --method local --client asymmetric_ew_s100/rsu_E --config configs/learning.yaml
python -m trafma fedavg --config configs/learning.yaml        # flat
python -m trafma fedavg --iid-control --config configs/learning.yaml
python -m trafma fedavg --transient-participation --config configs/learning.yaml
python -m trafma fedavg --hierarchical --config configs/learning.yaml
python -m trafma fedavg --hierarchical --rsu-self-update false --config configs/learning.yaml
python -m trafma phase2-report --help
python -m trafma phase2-stats --help
python -m trafma phase2-local-eval --help
python -m trafma simulate --config configs/phase1.yaml --vehicles 20 --seed 42 --duration 180 --online-federation --learning-config configs/learning.yaml
python -m pytest
```

For the resumable full-v2 five-seed sweep and repeatable live demo, use `scripts/run_phase2_full_v2_sweep.ps1` and `scripts/demo_phase2_live.ps1` in Windows PowerShell.

The existing 350v250 experiment ID is `runs/phase2_350v250_20261105_223954`. In the current workspace, replay the saved training-split SUMO run with its existing run ID:

```powershell
$datasetRun = "data/phase2_350clients_1000windows_feasibility/train/heavy_long_hold_s6101"
python -m trafma replay --run $datasetRun --every 10
python -m trafma inspect --run $datasetRun --rsu N
```

To open a fresh SUMO GUI demo, keep it under that same experiment folder and give the new simulation a unique child run ID. This preserves the existing 350v250 ID and avoids overwriting its training artifacts:

```powershell
$comparisonRoot = "runs/phase2_350v250_20261105_223954"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$guiId = "gui_n20_s42_$stamp"
python -m trafma simulate --vehicles 20 --seed 42 --gui --run-id $guiId --output-dir "$comparisonRoot/demo"
$guiRun = "$comparisonRoot/demo/$guiId"
python -m trafma gate --run $guiRun --reproduce --pytest
python -m trafma report --run $guiRun
python -m trafma replay --run $guiRun --every 10
python -m trafma inspect --run $guiRun --vehicle veh_0005
```

The GUI command creates a new simulation run because a saved log replay cannot reopen SUMO's GUI. If you repeat it, the timestamp gives it a new child ID while keeping the same 350v250 parent folder.

Run the full software test suite with:

```bash
python -m pytest
```

The GUI has been verified under a virtual display; a real desktop GUI check remains open. For model/device guidance, see [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

## 3. Driving side

`network.drive_side` is `left` (default) or `right`. Layout, signal plan, demand, agents, messages and labels are otherwise identical.

```bash
python -m trafma simulate --vehicles 20 --seed 42                                  # left-hand
python -m trafma simulate --vehicles 20 --seed 42 --set network.drive_side=right   # original behaviour
```

- The network is built with netconvert `--lefthand true`: each lane keeps to the left of its road.
- Turn names are the driver's own. N to E is a left turn for both settings. With left-hand traffic the left turn is the short one and the right turn crosses oncoming traffic. Split phasing means no two approaches are ever green together, so neither conflicts.
- RSUs stand on the kerb side of each road (left of travel for left-hand traffic).
- `netgen.py` checks after building that lanes are on the requested side and the arms are still N, E, S, W, and stops otherwise.
- `drive_side=right` reproduces the previous implementation exactly (trace hash `e2c4145f...` for 20 vehicles, seed 42). Left-hand gives `aa69daee...`.
- Older `phase2_v1`, `phase2_v2_dev` and `phase2_v2` data, where present, were generated right-hand. Treat them as legacy records and do not mix them with left-hand runs.

## 4. Datasets and the train / validation / test split

`python -m trafma collect --config <yaml>` writes each simulation run into its train, validation or test folder.

```text
data/phase2_350clients_1000windows_feasibility/
  train/<scenario>_s<seed>/
  val/<scenario>_s<seed>/
  test/<scenario>_s<seed>/
  splits.json
  dataset_manifest.json
  dataset_report.txt
  feature_audit.json
  sample_budget.md
```

- The split is by whole run (disjoint seeds), so a window never crosses splits.
- If a collect config has no `splits:` section the default is train seeds 1, 2, 3; val seed 4; test seeds 5, 6.
- Only the names `train`, `val` and `test` are allowed, and a seed may appear in one split only.
- Old datasets that kept runs in `<dataset>/runs/` still load.

| Dataset | Current config | Runs and split |
|---|---|---|
| `phase2_v1` | `configs/data.yaml` | 24 runs; 3 train / 1 validation / 2 test seeds |
| `phase2_350clients_1000windows_feasibility` | `configs/data_350clients_1000windows_feasibility.yaml` | 3 runs; one train, one validation, one test run; each simulates 500 vehicles for 1,800 s |

The 350v250 dataset is present locally at `data/phase2_350clients_1000windows_feasibility/`; generated data is not included in the source-only GitHub branch. On a fresh clone, collect it once with the dedicated config before training. Do not collect it again for each model: the four methods intentionally reuse the same train/validation/test split for a matched comparison. To create independent data, use a **new `dataset_id`** and point the learning configs at that new dataset directory. `--resume` is for continuing the same collection, not for creating an independent dataset.

Available collection commands:

```powershell
python -m trafma collect --config configs/data.yaml            # legacy v1; writes data/phase2_v1
python -m trafma collect --config configs/data_350clients_1000windows_feasibility.yaml  # only if its dataset is absent
```

This branch keeps the v1 and dedicated 350v250 collection configs. The v2 configs and saved v1/v2 manifests were removed from this branch; the 350v250 workflow uses the dedicated configs below.

## 5. Historical compact 1,500-sample experiment

This is an earlier small-budget reference. Full accounting of vehicles, observations, rounds and windows is in [docs/COMPACT_1500.md](docs/COMPACT_1500.md). These commands require the matching compact dataset and configs.

```bash
python -m trafma collect  --config configs/data_compact_1500.yaml
python -m trafma train    --method central --config configs/learning_compact_1500.yaml
python -m trafma fedavg   --hierarchical   --config configs/learning_compact_1500.yaml
python -m trafma baseline --config configs/learning_compact_1500.yaml
```

| Item | Value |
|---|---|
| Simulation | 4 scenarios (60, 220, 380, 260 vehicles), 900 s, 16 runs, 3,680 vehicles scheduled |
| Training / validation / test windows | 1500 / 300 / 300 |
| Central GRU | 104 epochs (minimum 50, early stopped, best at 89) |
| Hierarchical FL | 20 clients x 75 windows, 100 rounds, 5 local epochs |

| Method | Test macro-F1 | Test queue MAE (m) |
|---|---:|---:|
| Central GRU | 0.772 | 20.6 |
| Signal-aware Ridge (matched rows) | 0.562 | 27.1 |
| Hierarchical FedAvg | 0.545 | 36.7 |

These are single-seed results on 300 test windows: descriptive, not a final benchmark.

## 6. Previous 350-client, 20-window compact comparison (2026-11-05)

The latest compact comparison uses **350 vehicle clients with 20 windows each** (7,000 vehicle windows total), plus a four-RSU self-update ablation. The phrase “7,000 examples per client” would be inaccurate: 7,000 is the aggregate vehicle-window budget. In the RSU self-update-on run, each of the four RSUs contributes 1,750 sensor windows, adding 7,000 RSU windows.

All four model runs used the same held-out cohort of 14,000 windows (7,000 vehicle and 7,000 RSU) for validation and test. The central run trained on 7,000 windows of each type. FL configurations used 100 rounds and five local epochs. Those runs and their package have since been removed from the current working tree; the historical scores below are retained from their saved report and status record.

| Method | Train windows | Test macro-F1 | Test balanced accuracy | Test queue MAE (m) | Test queue RMSE (m) |
|---|---:|---:|---:|---:|---:|
| Central GRU | 14,000 | 0.8357 | 0.8292 | 14.905 | 24.578 |
| Flat FedAvg, vehicle only | 7,000 | 0.2669 | 0.2991 | 57.391 | 64.154 |
| Hierarchical FedAvg, RSU self-update off | 7,000 | 0.2524 | 0.2843 | 57.782 | 64.718 |
| Hierarchical FedAvg, RSU self-update on | 14,000 | 0.7078 | 0.6925 | 27.352 | 42.185 |

RSU self-updates strongly affect the result: in the self-update-on run, test macro-F1 / queue MAE are 0.8880 / 6.995 m for RSU rows and 0.5282 / 47.709 m for vehicle rows. The aggregate score therefore hides a substantial gap between sensor types. Flat and hierarchical FedAvg with RSU self-updates off are nearly tied, consistent with the same updates being averaged once; hierarchy alone is not an accuracy gain here.

These are **one-seed descriptive results**, with overlapping one-second windows from four held-out SUMO runs; they are not independent-window confidence estimates or a multi-seed benchmark. The RSU-on comparison also doubles the FL training-window budget relative to the vehicle-only FL runs, so it does not isolate the effect of hierarchy or RSU participation from added training data. This is the earlier 20-window baseline; its old run folders and matched report are no longer in the current `runs/` folder. The historical metrics are retained here and in `Explain Project Implementation.md`; the new 350v250 results follow.

## 7. Current 350v250 comparison and repeatable demo

The current comparison uses 350 real vehicle-trajectory clients with up to 250 distinct windows each. A separate feasibility dataset supports the longer trajectories. Its train split had 361 clients eligible for 250 windows; 350 were selected. No client reached 1,000 windows. Each of the four RSUs had 1,761 eligible sensor windows, and the self-update-on run uses up to 1,750 per RSU.

| Method | Vehicle windows | RSU windows | Total train windows | Test macro-F1 | Test queue MAE (m) |
|---|---:|---:|---:|---:|---:|
| Central GRU | 87,500 | 7,000 | 94,500 | 0.8733 | 9.281 |
| Flat FedAvg | 87,500 | 0 | 87,500 | 0.6784 | 38.241 |
| Hierarchical FedAvg, RSU self-update off | 87,500 | 0 | 87,500 | 0.6781 | 38.277 |
| Hierarchical FedAvg, RSU self-update on | 87,500 | 7,000 | 94,500 | 0.8344 | 15.113 |

Each run evaluates 7,000 vehicle and 7,000 RSU test windows. The four methods share the same held-out identities within this new comparison. The previous 20-window comparison used a different dataset split and seed, so these results do not isolate the effect of increasing the window cap. The new scores are one-seed descriptive results on overlapping windows.

To run all four methods with visible terminal progress, separate output folders, saved config snapshots and one report containing only these runs, use the PowerShell runner:

```powershell
.\scripts\run_phase2_350v250.ps1
```

The script requires the existing `data/phase2_350clients_1000windows_feasibility/` dataset and stops if it is missing. It keeps the existing parent run ID and creates a timestamped repeat folder under `runs/phase2_350v250_20261105_223954/repeats/`. That child contains `central/`, `flat/`, `hierarchical_rsu_off/`, `hierarchical_rsu_on/`, `configs/` and `report/`. It does not collect or overwrite dataset files. Use `-Device cpu` to force CPU central training, `-Device auto` to select CUDA when available, or leave the default `cuda` for the tested RTX 4050 environment.

The individual commands are available when you want a standalone method run, but their source configs use separate default output roots. Use the timestamped runner above when all four outputs and config snapshots must stay under the existing `phase2_350v250_20261105_223954` ID.

```powershell
python -m trafma train --method central --config configs/learning_350v250_central.yaml --device cuda
python -m trafma fedavg --config configs/learning_350v250_flat.yaml --threads 0
python -m trafma fedavg --hierarchical --config configs/learning_350v250_hierarchical.yaml --rsu-self-update false --threads 0
python -m trafma fedavg --hierarchical --config configs/learning_350v250_hierarchical.yaml --rsu-self-update true --threads 0
```

Save the four paths printed by those commands and pass only those paths to `python -m trafma run-report --experiment <run-dir> --experiment <run-dir> --output <new-report-dir>`. The canonical completed run and report in the current workspace are under `runs/phase2_350v250_20261105_223954/`. Generated dataset/run outputs are intentionally not committed; the runner reproduces them from the configs.

## 8. Training: epochs, CPU and GPU

```bash
python -m trafma train --method central --config configs/learning_350v250_central.yaml --device cuda
python -m trafma train --method central --config configs/learning_350v250_central.yaml --device cpu
python -m trafma fedavg --config configs/learning_350v250_flat.yaml --threads 0
python -m trafma fedavg --hierarchical --config configs/learning_350v250_hierarchical.yaml --rsu-self-update false --threads 0
python -m trafma fedavg --hierarchical --config configs/learning_350v250_hierarchical.yaml --rsu-self-update true --threads 0
```

- The 350v250 central config allows up to 150 epochs, sets `min_epochs: 1`, and uses early-stopping patience 15. The completed run stopped after 26 epochs and retained the best validation checkpoint.
- `--device auto|cpu|cuda`: auto uses a GPU when PyTorch can see one.
- `--threads N`: CPU threads; 0 selects all logical cores. FedAvg client training is CPU multi-threaded in this implementation.
- CUDA was exercised for the completed central 350v250 run on an RTX 4050. That timing does not promise a speedup on other hardware.

Details: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

## 9. Command reference

```bash
python -m trafma simulate --vehicles 20 --seed 42 --run-id <unique-id> --output-dir runs/<demo-folder> [--gui]
python -m trafma gate --run <simulation-run-folder> --reproduce --pytest
python -m trafma report --run <simulation-run-folder>
python -m trafma replay --run <simulation-run-folder> --every 10
python -m trafma inspect --run <simulation-run-folder> --vehicle veh_0005
python -m trafma features-audit --dataset data/phase2_350clients_1000windows_feasibility
python -m trafma baseline --config configs/learning_350v250_central.yaml
python -m trafma train --method central --config configs/learning_350v250_central.yaml --device cuda
python -m trafma train --method local --client <eligible-client-id> --config configs/learning_350v250_central.yaml
python -m trafma fedavg --config configs/learning_350v250_flat.yaml --threads 0
python -m trafma fedavg --iid-control --config configs/learning_350v250_flat.yaml --threads 0
python -m trafma fedavg --transient-participation --config configs/learning_350v250_flat.yaml --threads 0
python -m trafma fedavg --hierarchical --config configs/learning_350v250_hierarchical.yaml --rsu-self-update false --threads 0
python -m trafma fedavg --hierarchical --config configs/learning_350v250_hierarchical.yaml --rsu-self-update true --threads 0
python -m trafma run-report --experiment <run-dir> --experiment <run-dir> --output <new-report-dir>
python -m trafma sample-budget --dataset data/phase2_350clients_1000windows_feasibility --experiment <run-dir>
python -m trafma phase2-report --help
python -m trafma phase2-stats  --help
python -m trafma phase2-local-eval --help
```

The full project includes the legacy phase1, learning and v2 configs used by the command examples. If one is missing from a trimmed working copy, restore it from the repository before running that workflow. Any simulation config value can be overridden, for example `--set agents.rsu_zone_m=120 --set comms.v2i.loss_prob=0.1`. In `--gui`, vehicles are green inside the RSU zone, red when halted, yellow when approaching and grey in the junction.

## 10. Run artefacts

Each simulation run folder holds:

| File or folder | Contents |
|---|---|
| `manifest.json` | Config, seed, versions, file hashes, counts, status |
| `scenario/` | Network, routes, signal program, SUMO logs, `topology.json` (includes `drive_side`) |
| `agents/rsu_*.jsonl`, `agents/veh_*.jsonl` | Agent-local views only |
| `messages.jsonl` | Send, deliver and reject trace |
| `trace.jsonl` | Per-step vehicle state; its SHA-256 is the reproducibility fingerprint |
| `evaluator/` | Ground truth, never visible to agents or models |
| `labels.jsonl`, `windows.jsonl` | Dataset runs only: label and window index |

The saved feasibility-dataset training run `data/phase2_350clients_1000windows_feasibility/train/heavy_long_hold_s6101/` can be replayed without SUMO. New GUI simulations need a new run ID; keep them beneath `runs/phase2_350v250_20261105_223954/` to retain the existing top-level experiment ID.

## 11. Defaults and limits

- 300 m arms, one lane per direction, 150 m RSU zone, 1 s step, signal 25 s green, 3 s yellow, 2 s all-red (120 s cycle).
- Congestion rule: queue / zone below 0.10 is free, below 0.35 moderate, otherwise heavy. History 10 steps, horizon 30 s.
- Sensors are noise-free. Attacks, trust and the dashboard belong to Phase 3.
- The `heavy` and `asymmetric` scenarios are over-saturated on purpose: vehicles still queued at 900 s are reported as unfinished.
- GUI mode was verified under a virtual display only.

## 12. Documentation map

| Document | Purpose |
|---|---|
| [docs/ARCHITECTURE_AS_BUILT.md](docs/ARCHITECTURE_AS_BUILT.md) | What the code does today |
| [docs/COMPACT_1500.md](docs/COMPACT_1500.md) | Sample accounting for the 1,500-window experiment |
| [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Epochs, early stopping, CPU, GPU, threads |
| [docs/STATUS.md](docs/STATUS.md) | Evidence ledger |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Design decisions |
| [docs/DATA_CONTRACTS.md](docs/DATA_CONTRACTS.md) | Message, label and feature contracts |
| [AGENTS.md](AGENTS.md) | Rules for coding agents |

`docs/ARCHITECTURE.md` is the original plan and still says "camera" where the code says RSU.
