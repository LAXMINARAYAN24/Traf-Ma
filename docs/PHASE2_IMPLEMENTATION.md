# Phase 2 implementation status

Updated: 2026-10-31. This document describes implemented behavior and saved evidence, not planned features.

## Current milestone

The implemented P2.2-P2.7 development slice provides:

- Feature availability audits over all generated v1, v2_dev and full-v2 per-agent records.
- `trafma.features/1`, a deterministic 22-value timestep adapter with explicit masks.
- Ten-step, client-local feature windows joined offline to the existing `end_t + 30 s` labels.
- RSU persistence, a train-only majority-class diagnostic and pooled signal-aware Ridge reference.
- A two-head GRU, explicit trajectory-client local training and a pooled offline central reference.
- Flat FedAvg, synthetic IID control, transient offline participation replay, and hierarchical FedAvg.
- Four persistent RSU sensor learners; each trains on its own approach's training-run sensor windows only and is counted once in its RSU group.
- Saved predictions, checkpoints, metrics, round/client logs, reproducibility manifests and a report generator producing Markdown, HTML and plots.

Phase 2 is not fully frozen: full-v2 model training, paired multi-seed comparisons, a broad multi-client local-only benchmark, and a live online participation demonstration remain open. The existing transient-participation experiment is offline replay, not an online connection.

## Data boundary and label contract

Features are derived only from one agent's own `agents/rsu_*.jsonl` or `agents/veh_*.jsonl` record plus that run's static config and topology. The feature adapter does not import the evaluator or read another agent's log. Future labels are joined from the dataset's saved `windows.jsonl` only in the offline learning/evaluation path; labels are not part of model inputs.

The input is ten consecutive one-second records from one client and run. Vehicle inputs require all ten records to have `scope=incoming_local`. Windows must end at a saved label exactly 30 seconds in the future. Splits remain whole-run splits. Queue labels are metres for the evaluator's contiguous halted-vehicle chain; class thresholds remain 0.10/0.35 of the 150 m zone.

## Feature schema `trafma.features/1`

Tensor order is `[10, 22]`: ten ordered values, ten matching availability masks, then two client-type bits. The schema and its SHA-256 are defined in `trafma/features.py` and saved in audit/model artifacts.

| Values | RSU source | Vehicle source | Normalization |
|---|---|---|---|
| Local mean speed | `mean_speed_mps` | Own speed plus the same-approach neighbour mean weighted by `neighbor_count`; self counted once | Divide by configured max speed; clip to `[0,1]` |
| Local mean acceleration | `mean_accel_mps2` | Own `accel_mps2` proxy | Divide by twice the larger configured accel/decel reference; clip to `[-1,1]` |
| Distance to stop line | `mean_dist_to_stop_m` | Own `dist_to_stop_line_m` proxy | Divide by configured arm length; clip to `[0,1]` |
| Density ratio | `vehicle_count / zone capacity` | `(neighbor_count + 1) / clipped visible-segment capacity` | Clip to `[0,1]` |
| Halted-count ratio | `halting_count / zone capacity` | `(neighbor_halting_count + own halted indicator) / clipped visible-segment capacity` | Clip to `[0,1]` |
| Waiting time | `mean_waiting_time_s` | Own `waiting_time_s` proxy | Divide by 300 s; clip to `[0,1]` |
| Signal timer | `signal_remaining_s` | `signal_remaining_s` only when observed | Divide by configured signal-cycle duration; clip to `[0,1]` |
| Signal state | Agent-local R/Y/G state | Agent-local R/Y/G state | Fixed red/yellow/green one-hot |

Masks are generated from source availability, not from normalized values. Missing values are encoded as zero only alongside mask zero; a measured zero has mask one. Missing signal state yields three zero one-hot values and three zero masks. Vehicle signal fields remain unavailable when not broadcast or outside their allowed scope. Vehicle acceleration, distance and waiting time remain explicitly local proxies, not RSU-equivalent means.

RSU capacity uses the configured target-zone length divided by vehicle length plus minimum gap. Vehicle capacity uses the one-lane segment clipped to the incoming edge using `lane_pos_m` and the configured radius; it is not a fixed full-length denominator. Local queue length is excluded from the 22 model inputs and is used only by the explicit persistence baseline.

## Development audit evidence

Command:

```powershell
python -m trafma features-audit --dataset data\phase2_v2_dev
```

The saved audit is `data/phase2_v2_dev/feature_audit.json`. It scanned all 48 development runs: 172,800 RSU records and 1,620,921 vehicle records, including 1,372,607 incoming-local vehicle records. The audit passed with no feature errors, evaluator-label keys, mixed broadcast settings or overlapping split runs.

Availability fractions among rows eligible for feature adaptation:

| Feature | RSU | Vehicle |
|---|---:|---:|
| Mean speed, acceleration, distance and waiting time | 0.8048 | 1.0000 |
| Density and halted-count ratios | 1.0000 | 1.0000 |
| Signal state | 1.0000 | 0.6121 |
| Signal remaining time | 1.0000 | 0.0000 |

Vehicle signal availability is expected for the default non-broadcast dataset; these figures are development data measurements, not general deployment estimates. Full scans of v1, v2_dev and full v2 now pass with schema SHA-256 `46196acbcc933a9b60f0a3151cf3a5822350c6988b7894a50f5722ac80f4abe2`.

## Baselines

Command:

```powershell
python -m trafma baseline --config configs\learning.yaml
```

The baseline config fixes Ridge `alpha=1.0`, minimum client training history at 16 windows, and the development dataset. Ridge is fitted only on eligible training clients' windows and the saved signal-aware coefficients are in the experiment checkpoint. Its input is the final timestep's local 22-value vector. It is an offline pooled reference, not a federated or private model. Predicted classes are derived from predicted queue metres using the fixed 0.10/0.35 thresholds.

Persistence predicts using an RSU's current locally sensed `queue_length_m`. It is unavailable for vehicle rows because vehicle logs do not contain a local queue-length measurement. The test comparison below uses the exact same 55,104 held-out RSU windows for both methods:

| Split / client rows | Method | N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| Test / RSU | Persistence | 55,104 | 28.9377 | 48.9294 | 0.5211 | 0.5195 |
| Test / RSU | Signal-aware Ridge | 55,104 | 19.6867 | 27.7702 | 0.5909 | 0.6686 |
| Test / vehicle | Signal-aware Ridge | 420,753 | 37.4736 | 45.6219 | 0.4653 | 0.5265 |

These are saved overlapping-window results, not independent samples or run-level confidence intervals. The complete validation/test breakdowns and confusion matrices are in `metrics.json`; per-row outputs are in `predictions.csv`. Metrics are regenerated by reading those saved prediction rows. The frozen config was not tuned on test data.

Experiment artifacts:

```text
runs/phase2/baseline_phase2_v2_dev_20261031T104426Z/
  manifest.json
  resolved_config.yaml
  summary.md
  metrics.json
  metrics.csv
  predictions.csv
  training_history.csv
  checkpoints/signal_aware_ridge.json
```

The manifest records dataset-manifest and config hashes, disjoint run IDs, feature-schema hash, code revision/worktree state, Python/platform/CPU, dataset simulator versions, installed package versions, seed and wall-clock timing.

## Commands currently implemented

- `python -m trafma features-audit --dataset <dataset-dir> [--output <json-path>]`
- `python -m trafma baseline [--config configs/learning.yaml] [--dataset <dataset-dir>]`
- `python -m trafma train --method central [--config configs/learning.yaml]`
- `python -m trafma train --method local --client <training-client-id> [--config configs/learning.yaml]`
- `python -m trafma fedavg [--config configs/learning.yaml]`
- `python -m trafma fedavg --iid-control [--config configs/learning.yaml]`
- `python -m trafma fedavg --transient-participation [--config configs/learning.yaml]`
- `python -m trafma fedavg --hierarchical [--config configs/learning.yaml]`
- `python -m trafma phase2-report --dataset <dataset-dir> --baseline <baseline-run> --experiment <run> [--experiment <run> ...]`

## Local and central training

`train --method local` requires one explicit eligible training-client ID. It uses only that trajectory for training and evaluates on held-out runs matched by client type and approach. `train --method central` is explicitly pooled/offline. Checkpoint selection uses validation loss; test predictions are recomputed from the reloaded checkpoint. These are smoke runs, not a population-wide local-model benchmark.

## Federation contract and RSU decision

The user approved D4: `rsu_N`, `rsu_E`, `rsu_S`, and `rsu_W` are persistent learning identities. Each trains on its own RSU sensor observations pooled across training runs; no child vehicle histories are used as RSU training examples. The hierarchical smoke caps each RSU learner at 2,000 distinct training windows using a seeded reservoir sample. Vehicle trajectory clients remain separate and route to their original parent RSU group. `max_clients` includes the four RSUs in hierarchical mode.

All clients start each round from the same global checkpoint. Updates carry model weights and allowlisted scalar metadata, including distinct-window sample count; they do not carry feature histories, labels or evaluator truth. Validators reject duplicates, wrong/late rounds, base/schema/signature mismatches, non-finite updates, invalid counts and disallowed metadata. Group aggregation includes an RSU's self-update once alongside its child updates, preserves group mass, omits empty groups and reproduces flat FedAvg on identical fixtures.

## Development comparison (sampled matched rows)

The saved comparison uses a deterministic 10,000-row `phase2_v2_dev` test sample. Central, natural flat, IID, transient and hierarchical global-model test row identities match exactly. Reported windows overlap and are not independent replications. Training budgets/cohorts differ, so these scores do not establish that hierarchy or FL improves accuracy.

| Method | Training windows | Test N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Train-only majority class | train-derived class 2 | 10,000 | n/a | n/a | 0.2086 | 0.3333 |
| Signal-aware Ridge, matched | pooled train only | 10,000 | 35.1568 | 43.7588 | 0.4869 | 0.5436 |
| Central pooled GRU | 30,000 | 10,000 | 34.5355 | 43.2208 | 0.5654 | 0.5716 |
| Natural flat FedAvg | 1,789 | 10,000 | 61.9613 | 66.5890 | 0.2349 | 0.3458 |
| Synthetic IID FedAvg | 1,789 | 10,000 | 61.8780 | 66.5071 | 0.2358 | 0.3463 |
| Transient flat replay, 75% vehicle participation | 1,789 | 10,000 | 62.0214 | 66.6712 | 0.2339 | 0.3453 |
| Hierarchical FedAvg with persistent RSUs | 9,804 | 10,000 | 49.5341 | 56.6077 | 0.3916 | 0.4350 |

Moderate-class recall is zero in all three flat-FL variants and in the hierarchical smoke. Keep this failure visible in confusion matrices; do not make a robustness or accuracy claim from these runs. The selected flat replay cohort was all vehicle clients; hierarchical mode used 12 vehicle clients balanced across approaches plus four persistent RSUs. The local-only smoke used `asymmetric_ew_s100/rsu_E` (861 training windows) and a 10,000-row held-out RSU/E stratum, so it is not the global model's test cohort.

The 75% transient experiment chooses 12 of 16 vehicle clients each round, logs client arrivals/departures and remains an offline replay. It does not show real SUMO clients training online or establish radio/network performance.

## Saved development artifacts

- Baseline including majority diagnostic: `runs/phase2/baseline_phase2_v2_dev_20261031T123905Z/`
- Local GRU: `runs/phase2/local_gru_phase2_v2_dev_20261031T120045Z/`
- Central GRU: `runs/phase2/central_gru_phase2_v2_dev_20261031T115518Z/`
- Natural flat FedAvg: `runs/phase2/flat_fedavg_phase2_v2_dev_20261031T120523Z/`
- Synthetic IID: `runs/phase2/flat_fedavg_iid_phase2_v2_dev_20261031T122319Z/`
- Transient participation replay: `runs/phase2/flat_fedavg_phase2_v2_dev_20261031T122549Z/`
- Hierarchical FedAvg: `runs/phase2/hierarchical_fedavg_phase2_v2_dev_20261031T121932Z/`
- Markdown/HTML and plots: `runs/phase2/report_phase2_20261031T123953Z/`

Trust/security, attack, consensus, quarantine/recovery and dashboard functionality remain Phase 3 and are not included here.
