# Status and decisions

Updated: 2026-11-06. "Verified" is backed by a run/test/artifact; outstanding Phase 2 items are explicitly listed.

## Completed 350-vehicle / RSU-stratified compact comparison (2026-11-05)

- Four runs completed: central GRU (`runs/phase2_compact_350_central_7000each/central_gru_phase2_compact_1500_seed20261031_20261104T183425Z/`, 14,000 train windows, 83 epochs, 3.3 min); flat FedAvg (`runs/phase2_compact_350_flat_vehicle_only_7000/flat_fedavg_phase2_compact_1500_seed20261031_20261104T191151Z/`, 350 vehicle clients x 20 windows = 7,000, 100 rounds, 29.2 min); hierarchical RSU self-update off (`runs/phase2_compact_350_vehicle_only_7000/hierarchical_fedavg_phase2_compact_1500_seed20261031_20261104T183804Z/`, 350 vehicle clients x 20 windows = 7,000, 100 rounds, 32.9 min); hierarchical RSU self-update on (`runs/phase2_compact_350_vehicle_rsu_7000each/hierarchical_fedavg_phase2_compact_1500_seed20261031_20261104T194135Z/`, 350 vehicle clients x 20 plus four RSU groups x 1,750 = 14,000, 100 rounds, 40.3 min).
- Every run evaluated the same 14,000 held-out rows: 7,000 vehicle and 7,000 RSU windows for validation and test. Per-type sample counts are recorded in manifests.
- Combined report: `runs/phase2_compact_350_7000_report/report.md` and `report.html`. Matched Ridge / central / flat / hierarchical-off / hierarchical-on test macro-F1: 0.5335 / 0.8357 / 0.2669 / 0.2524 / 0.7078. Queue MAE (m): 29.905 / 14.905 / 57.391 / 57.782 / 27.352. These are descriptive results for one training seed with overlapping windows, not run-level confidence estimates.
- Dataset capacity supports the sampling targets: train 205,929 vehicle / 27,552 RSU windows; validation 106,786 / 13,776; test 111,067 / 13,776. The flat fixed-budget selection check was tightened to require each selected vehicle to meet the 20-window cap. RSU budget validation now initializes the eligibility threshold before validating the RSU group cap.
- The four experiments and matched report are complete. The test suite was not rerun for these sampler/configuration changes.
- A separate latest-run-only report, `runs/phase2_compact_350_7000_report_latest_only/report.md` and `report.html`, uses only the four current model artifacts and excludes the earlier baseline artifact used in the first combined report.
- The latest-run-only report now includes overall/type-stratified performance graphs, row-normalized confusion matrices on a shared 0–100% scale, validation curves and per-run queue scatter plots. Individual HTML/Markdown reports and their plot assets are in `runs/phase2_compact_350_7000_report_latest_only/individual/`.
- Future run reports use `python -m trafma run-report --experiment <saved-run-folder> --output <new-report-folder>`; repeat `--experiment` to create a matched comparison plus one individual report per run. Report generation verifies matching held-out row identities and records input hashes.
- The final RSU self-update-on run has a Git-ready package at `artifacts/phase2_compact_350_hfl_rsu_on_7000each/`: its full run artifacts, 14,000 selected feature windows for each train/validation/test split cohort (7,000 per type), config, and individual report/graphs. The full raw dataset is intentionally not copied; selected feature tensors and their provenance hashes are included. Package size is below 50 MiB.

## 350-client / 1,000-window feasibility pilot (2026-11-05)

- User requested 350 real vehicle-trajectory clients with 1,000 distinct windows each and 1,750 windows from each of four RSU learners. The existing compact dataset had 1,764 eligible vehicle clients but no client above 347 windows.
- An additive feasibility dataset was collected at `data/phase2_350clients_1000windows_feasibility/` using `configs/data_350clients_1000windows_feasibility.yaml`: three separate 1,800-second, 500-vehicle runs (train/validation/test), with departures scheduled during the first 600 seconds to sustain congestion. Each completed with 500 arrivals, zero collisions and zero teleports; whole-run split and label integrity checks passed.
- The pilot train split has 499 trainable vehicle trajectories, maximum 412 windows; zero clients reach 1,000. 361 clients reach 250 windows, enough for a fixed 350-client x 250-window comparison. Each RSU has 1,761 windows, sufficient for the requested 1,750 cap.
- Therefore the requested 350 x 1,000 training budget is not available from current one-trajectory-per-client data. At the time of this feasibility audit, no central/flat/hierarchical model run had started. Proceeding requires either a different simulation/client design or a lower per-client window budget; do not duplicate/relabel windows to fill the cap.
- Terminal progress bars for central GRU epochs and flat/hierarchical FedAvg rounds were added in `trafma/progress.py`; tests have not been run for that code change.
- An initial exploratory CUDA central-GRU attempt on the feasibility dataset was intentionally interrupted at epoch 28/150 before artifacts were written. A complete four-method comparison subsequently finished under `runs/phase2_350v250_20261105_223954/`: central GRU (94,500 windows, CUDA, 26 epochs, 4.5 min), flat FedAvg (87,500, 100 rounds, 136.3 min), hierarchical FedAvg with RSU self-update off (87,500, 100 rounds, 131.6 min), and hierarchical FedAvg with RSU self-update on (94,500, 100 rounds, 148.6 min). Each FL client uses five local epochs; FedAvg is CPU multi-threaded. The matched report is `runs/phase2_350v250_20261105_223954/report/report.md` and `.html`. Test macro-F1 was 0.8733 / 0.6784 / 0.6781 / 0.8344 respectively; queue MAE was 9.281 / 38.241 / 38.277 / 15.113 m. These are one-seed descriptive results on overlapping windows and use a different dataset split from the prior 20-window comparison, so they do not isolate the window-cap effect. The exact run configs are snapshotted in the run folder.

## Expanded compact 100-vehicle experiment and RSU self-update ablation

- `configs/learning_compact_100vehicles.yaml`: 100 vehicle clients plus four RSU sensor learners; 75 windows each, 7,800 windows total, 100 rounds, five local epochs. Central GRU uses the same 7,800-window budget.
- Separate RSU self-update-off run: `runs/phase2_compact_100clients_rsu_off/hierarchical_fedavg_phase2_compact_1500_seed20261031_20261104T103945Z/`, selected 100 vehicle clients and 7,500 windows; four RSUs aggregate vehicle updates but contribute no sensor-only model updates. Config: `configs/learning_compact_100vehicles_rsu_off.yaml`.
- Combined saved-artifact comparison report: `runs/phase2_compact_100clients/report_100clients_rsu_ablation/`. It includes baseline, central GRU, HFL self-update on and off, all scored on the same 300 test rows.
- Descriptive test results: central GRU MAE 18.514 m / RMSE 28.281 m / macro-F1 0.7637 / balanced accuracy 0.7464; HFL self-update on 61.479 m / 68.148 m / 0.3423 / 0.3609; HFL self-update off 69.576 m / 76.363 m / 0.2976 / 0.3382. One training seed and 300 overlapping held-out windows; these are not run-level confidence estimates.
- The original 20-client/1,500-window compact config/report remains preserved. The dataset manifest lists 1,205 eligible vehicle clients with at least 75 windows (N 333, S 359, W 232, E 281).

## Where the project is

- **Phase 1 (P1.1-P1.6): functionally complete.** One gate item is still open: `simulate --gui` has only been verified under a virtual display, not on a real desktop.
- **P2.1 (data and labels): done.** The v1, v2 development and full v2 datasets are present locally and their manifests pass the listed integrity checks. The signal-broadcast dataset has a config only and has not been generated.
- **P2.2 (baselines):** persistence and train-only pooled signal-aware Ridge are implemented; v2_dev and full-v2 baseline artifacts are saved. All five full-v2 learned methods have five seeds, and paired run-level statistics are saved.
- **P2.3 (features/model):** `trafma.features/1` and the two-head GRU are implemented. The full v1, v2_dev and v2 feature scans pass with the same schema hash.
- **P2.4-P2.5 (federation):** flat FedAvg, four-group hierarchical FedAvg, persistent RSU sensor learners and synthetic IID control are implemented and have saved v2_dev runs. D4 is resolved below.
- **P2.6-P2.7:** natural non-IID and transient offline participation experiments plus a saved-artifact Markdown/HTML report and plots are implemented. Per-client natural/IID distributions, paired run-level statistics, an optional live SUMO federation path, a repeatable Windows smoke-demo script, and exhaustive plus representative resumable local-only evaluators are implemented. Full-v2 learned-method evaluation and the paired report are complete; exhaustive population local evaluation remains optional and is not a Phase 2 completion gate.

## Update 2026-11-04: left-hand traffic, split folders, 50-epoch GRU, compact 1,500-sample budget

All items below are backed by saved artifacts and the test suite (177 tests passing on Linux, SUMO 1.28.0, CPU PyTorch).

| Item | Evidence |
|---|---|
| Left-hand traffic by default (`network.drive_side`) | `tests/test_netgen.py` (lane side, RSU kerb, turn labels, compass layout); TraCI check of every vehicle at 1.6 m left of its road axis; stand-alone 20-vehicle demo regenerated, gate PASS including reproducibility (generated logs excluded from this branch) |
| Right-hand mode equals the old implementation | 20 vehicles, seed 42, `drive_side=right` gives trace hash `e2c4145faae5...`, the hash recorded above |
| Dataset split folders `data/<id>/{train,val,test}/<run>/` | `tests/test_dataset.py`; `phase2_compact_1500` generated, 0 split overlap, 0 collisions, 0 teleports, feature audit PASS with the same schema hash |
| Central GRU with at least 50 epochs and early stopping | `runs/phase2_compact/central_gru_*`: 104 epochs, best epoch 89, minimum 50 enforced |
| CPU threads and `device auto` | Implemented and recorded in manifests. Multi-thread speed-up not measured (1-core sandbox). **GPU path not tested** |
| Compact 1,500-sample experiment | `docs/COMPACT_1500.md`, `data/phase2_compact_1500/sample_budget.md`, `runs/phase2_compact/` |

Compact results (300 held-out test windows, one training seed, descriptive only): central GRU macro-F1 0.772 and queue MAE 20.6 m; matched signal-aware Ridge 0.562 and 27.1 m; hierarchical FedAvg (100 rounds x 5 local epochs, 20 clients x 75 windows) 0.545 and 36.7 m. Hierarchical FedAvg did not beat the pooled model or the Ridge baseline on this budget. A 30-round setting was under-trained (validation macro-F1 0.235); 100 rounds was chosen on validation results only.

Known caveats of this update:

- The committed `phase2_v1`, `phase2_v2_dev` and `phase2_v2` manifests and all earlier `runs/phase2/` results are right-hand and 2-epoch-central results. They are not comparable with the left-hand compact results and were not regenerated.
- The long crossing turn is now the right turn, so heavy-load vehicle counts and trip times differ slightly from right-hand runs.
- The default `epochs` in `configs/learning.yaml` changed from 2 to 150 (minimum 50). Earlier central/local GRU numbers in this file used 2 epochs.
- Vehicle clients in the fixed HFL budget need at least 84 consecutive in-scope steps, so they come mostly from congested runs.

## Previously verified implementation checks

| Check | Result |
|---|---|
| Current full suite (`.venv\Scripts\python.exe -m pytest -q`, Windows) | 165 passed in 41.46 s |
| Current `python -m trafma doctor` (Windows, Python 3.12.7) | READY; SUMO 1.27.1, TraCI/sumolib 1.28.0 |
| Pre-Phase-2 `python -m pytest` | 126 passed |
| New focused feature/baseline tests | 8 passed |
| `simulate --vehicles 20 --seed 42` trace SHA-256 | `e2c4145f...` recorded for the earlier 20-vehicle demo; regenerate with `python -m trafma simulate --vehicles 20 --seed 42` (raw fixture excluded from this branch). |
| `trafma gate --run <new run> --reproduce` | PASS (deterministic re-run identical) |
| `trafma collect --config configs/data.yaml` | `dataset_report.txt` identical to the committed one |
| Real SUMO run of the 4 new v2 scenarios (1 train, 1 val, 1 test seed each, 16 runs) | all ran to the end of the 900 s (over-saturated scenarios finish with unfinished vehicles reported, not as completed), 0 collisions, 0 teleports, integrity block clean, new RSU fields present in logs |

Not verified: GUI on a real desktop and exhaustive population-wide local-client results. The representative local-only mode is the bounded comparison path. The live path is smoke-tested only and does not establish performance or radio/network behavior.

## Data and feature re-audit (2026-10-31)

The generated run folders are present locally under `data/` (even though the run contents may be git-ignored). Manifest checks were run for all three datasets:

| Dataset | Runs | Run-folder bytes | Manifest integrity |
|---|---:|---:|---|
| `phase2_v1` | 24 | 734,585,503 | 0 split overlaps; 0 horizon/window violations; no evaluator keys in agent logs; 0 collisions/teleports |
| `phase2_v2_dev` | 48 | 1,419,515,781 | 0 split overlaps; 0 horizon/window violations; no evaluator keys in agent logs; 0 collisions/teleports |
| `phase2_v2` | 160 | 4,778,347,553 | 0 split overlaps; 0 horizon/window violations; no evaluator keys in agent logs; 0 collisions/teleports |

These are dataset-manifest integrity checks, not feature-tensor or model checks. `phase2_v2` manifests report 115 runs ending at the configured horizon with unfinished vehicles (`sim_timeout`) and 45 `completed`; this is the simulator's end-of-horizon status, not proof all scheduled vehicles arrived. The full broadcast ablation is not present.

All manifests record Windows 11, Python 3.12.7, SUMO 1.27.1, TraCI/sumolib 1.28.0 and TRAF-MA 0.1.0 for the generated datasets. This differs from the Linux/SUMO 1.28.0 environment in the historical test table; record and account for that version difference in experiments rather than assuming the run environment matches the pinned requirements.

A raw-log coverage sample was also checked for the eight `phase2_v2_dev` seed-100 scenarios:

- All 28,800 sampled RSU timesteps had signal, signal-time, vehicle-count, density and halted-count values. RSU mean speed/acceleration/distance were available on 23,184 rows (80.5%); empty zones account for unavailable means.
- A convenience sample of the first three sorted vehicle logs per scenario (24 logs, 2,561 rows) had own speed, acceleration and waiting time on every row. Of these, 1,857 rows were `incoming_local`; signal state was available for 1,439/1,857 (77.5%), and signal time was unavailable on all sampled rows because broadcast was disabled. This is a sample, not a population-wide feature audit.
- Dataset manifests contain labels and aggregate client statistics, not feature tensors. Full feature scans passed on all datasets: v1 has 86,400 RSU and 852,184 vehicle records (729,274 incoming-local); v2_dev has 172,800 RSU and 1,620,921 vehicle records (1,372,607 incoming-local); full v2 has 576,000 RSU and 5,471,540 vehicle records (4,654,139 incoming-local). All 232 runs passed feature, missingness, scope, label-leak and split checks with schema SHA-256 `46196acbcc933a9b60f0a3151cf3a5822350c6988b7894a50f5722ac80f4abe2`. v2_dev availability is recorded in its audit; it is not assumed to generalize beyond the audited data.

## Phase 2 learning and federation evidence (2026-10-31)

The CPU smoke runs use `phase2_v2_dev`, seed `20261031`, a fixed 10,000-window sampled test set and overlapping one-second windows. Global-model test identities (`run_id`, `client_id`, `end_t`) were verified identical across central, flat, IID, transient and hierarchical runs. These are descriptive development results, not run-independent confidence estimates or a final full-v2 benchmark.

| Method | Training windows | Test N | Queue MAE (m) | Queue RMSE (m) | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Train-only majority class (class 2) | train-derived | 10,000 | n/a | n/a | 0.2086 | 0.3333 |
| Signal-aware Ridge, matched subset | train-only pooled | 10,000 | 35.1568 | 43.7588 | 0.4869 | 0.5436 |
| Central pooled GRU | 30,000 | 10,000 | 34.5355 | 43.2208 | 0.5654 | 0.5716 |
| Flat FedAvg, natural replay | 1,789 | 10,000 | 61.9613 | 66.5890 | 0.2349 | 0.3458 |
| Flat FedAvg, synthetic IID | 1,789 | 10,000 | 61.8780 | 66.5071 | 0.2358 | 0.3463 |
| Flat FedAvg, transient 75% participation | 1,789 | 10,000 | 62.0214 | 66.6712 | 0.2339 | 0.3453 |
| Hierarchical FedAvg, persistent RSU learners | 9,804 | 10,000 | 49.5341 | 56.6077 | 0.3916 | 0.4350 |

The hierarchical training-window total is recorded in its run manifest; it includes a deterministic cap of 2,000 sensor windows per persistent RSU plus selected vehicle windows. The local-only smoke used training client `asymmetric_ew_s100/rsu_E` (861 training windows) and a held-out RSU/E stratum of 10,000 rows; its results are in that local run and are not directly comparable to the global 10,000-row cohort. Moderate-class recall was zero in flat, IID, transient and hierarchical smoke runs; this negative result is retained. Different training cohorts and budgets prevent attributing the higher hierarchical score to hierarchy itself.

Artifacts:

- Local GRU: `runs/phase2/local_gru_phase2_v2_dev_20261031T120045Z/`
- Central GRU: `runs/phase2/central_gru_phase2_v2_dev_20261031T115518Z/`
- Flat natural FedAvg: `runs/phase2/flat_fedavg_phase2_v2_dev_20261031T120523Z/`
- Synthetic IID: `runs/phase2/flat_fedavg_iid_phase2_v2_dev_20261031T122319Z/`
- Transient replay: `runs/phase2/flat_fedavg_phase2_v2_dev_20261031T122549Z/`
- Hierarchical FedAvg: `runs/phase2/hierarchical_fedavg_phase2_v2_dev_20261031T121932Z/`
- Baseline with the majority-class diagnostic: `runs/phase2/baseline_phase2_v2_dev_20261031T123905Z/`
- Latest matched report: `runs/phase2/report_phase2_20261031T123953Z/`

Current verification: the full suite passed 165 tests in 41.46 s; report integration tests, the regenerated full-v2 report, and a one-client local-evaluator smoke/resume run are verified. The all-eligible-client local-only benchmark remains open.

### Full-v2 baseline and live-federation smoke (2026-10-31)

The completed baseline artifact is `runs/phase2/baseline_phase2_v2_20261031T164408Z/`. Its signal-aware Ridge test predictions cover 1,187,485 overlapping windows: MAE 35.1973 m, RMSE 43.8793 m, macro-F1 0.4891, balanced accuracy 0.5489. The train-only majority diagnostic has macro-F1 0.2092 / balanced accuracy 0.3333. Persistence is available for 137,760 RSU rows only (MAE 28.1297 m, RMSE 48.0169 m, macro-F1 0.5266); it is not a matched comparison to Ridge's all-client population. These are descriptive window metrics, not run-bootstrap estimates.

The full-v2 sweep completed for seeds 20261031-20261035 across central GRU, natural flat FedAvg, synthetic IID FedAvg, and hierarchical FedAvg with RSU self-updates on and off. The paired-statistics artifact `runs/phase2/stats_phase2_v2_20261031T182654Z/` verifies 10,240 identical held-out test rows per seed across 40 runs; it reports per-seed and per-run metrics, run-bootstrap 95% intervals (10,000 resamples), and paired differences. Its run-level estimates and interpretation are summarized below.

The final matched comparison report is `runs/phase2/report_phase2_v2_final_20261101T0012Z/`. Detailed breakdowns and plots use one representative seed per method; its paired uncertainty section uses all five seeds. The full-v2 per-client distribution file includes all 17,529 eligible training clients; the matched natural-vs-IID cohort contains 16 source clients.

No long-running training jobs are intended to remain active. For local reproduction, `.\scripts\run_phase2_full_v2_sweep.ps1` skips completed runs and regenerates paired statistics. Seeds 400-404 remain unused. The all-eligible-client local-only evaluation remains open and is not included in the report.

The repeatable script was run successfully at `runs/phase2/phase2_live_demo_20261031_225007_34344/`: 180 simulated seconds, three accepted rounds, model version 3, 12 vehicle client-round participations, four RSU self-updates per round, and a valid manifest hash for `online_federation.jsonl`. No departed vehicle participated in a later round. The run is a smoke test, not a performance claim.

### Full-v2 paired evaluation and final report (2026-11-01)

All five learned methods have five training seeds (20261031-20261035) and identical 10,240-row test cohorts spanning 40 held-out runs. The 10,000-replicate percentile bootstrap resamples runs, not overlapping windows; each run's method score is averaged over its five training seeds.

| Method | Run-macro MAE (m), mean [95% CI] | Run-macro RMSE (m), mean [95% CI] | Macro-F1, mean [95% CI] | Balanced accuracy, mean [95% CI] |
|---|---|---|---|---|
| Signal-aware Ridge | 29.1487 [25.6708, 32.4885] | 36.4090 [32.6259, 40.1145] | 0.4191 [0.3979, 0.4402] | 0.5188 [0.4980, 0.5403] |
| Central GRU | 29.6892 [26.1924, 33.1270] | 36.5848 [32.4769, 40.4018] | 0.4890 [0.4573, 0.5193] | 0.5332 [0.5137, 0.5502] |
| Natural flat FedAvg | 61.7315 [59.5552, 63.8186] | 65.2688 [63.7148, 66.7525] | 0.1490 [0.1348, 0.1625] | 0.3063 [0.2895, 0.3226] |
| Synthetic IID FedAvg | 60.0319 [58.0437, 61.9253] | 63.6031 [62.1511, 64.9735] | 0.1611 [0.1348, 0.1872] | 0.2794 [0.2414, 0.3153] |
| Hierarchical FedAvg (RSU self-update on) | 45.0405 [41.1465, 48.7605] | 52.0662 [48.6697, 55.4376] | 0.2867 [0.2705, 0.3027] | 0.3808 [0.3692, 0.3928] |
| Hierarchical FedAvg (RSU self-update off) | 61.3314 [59.1904, 63.3396] | 64.8682 [63.3144, 66.3355] | 0.1546 [0.1404, 0.1678] | 0.3083 [0.2921, 0.3238] |

Central GRU vs Ridge was not significant for RMSE (difference +0.1758 m, 95% CI [-0.3866, 0.7455]) or balanced accuracy (+0.0144, [-0.0101, 0.0378]); central GRU had higher MAE by 0.5405 m and higher macro-F1 by 0.0699. IID vs natural FedAvg was not significant for macro-F1 (+0.0121, [-0.0017, 0.0247]); IID vs hierarchical self-update-off was not significant for macro-F1 (+0.0065, [-0.0071, 0.0198]). Other reported paired intervals excluded zero. The pairwise tests are unadjusted for multiple comparisons, so interpret them as exploratory.

For the matched natural/IID training cohort (16 clients), median/p90 per-client JSD from the pooled label distribution was 0.081104/0.316248 bits for natural clients and 0.000002/0.000012 for IID partitions. The selected IID control homogenized class proportions; this does not demonstrate an accuracy gain. Per-client label/sample counts cover all 17,529 eligible natural training clients.

Artifacts: paired results at `runs/phase2/stats_phase2_v2_20261031T182654Z/`; final Markdown/HTML report and plots at `runs/phase2/report_phase2_v2_final_20261101T0012Z/`. Detailed method breakdowns use one representative seed; paired uncertainty uses all five seeds. The all-client local-only model evaluation is not included.

### Broad local-only evaluator

The evaluator is implemented as `python -m trafma phase2-local-eval`. Exhaustive mode enumerates the 17,529 eligible training clients (320 RSU and 17,209 vehicle) and remains an optional resumable research benchmark. Representative mode selects a deterministic stratified cohort and records its seed, strata and exact client IDs. Both modes record ineligible train clients with explicit reasons and retain one model artifact per completed client under `models/`; `summary.json` and `metrics.csv` provide client-type, approach and scenario distributions. The previously verified capped CPU run at `runs/phase2/local_population_phase2_v2/` remains unchanged and is not relabeled as representative.

The representative selection invariants were verified on the full-v2 manifest for 256 clients with seed `20261101`: both RSU and vehicle types, all four approaches, all eight scenarios and all four training-window bands were present. The live smoke demo at `runs/phase2/phase2_live_demo_20261101_120508_37396/` completed three rounds with 12 vehicle and 12 RSU client-round participations; each round accepted all four RSU self-updates, produced four edge groups and advanced the global model version. Simulated time and wall-clock timings were recorded separately. This remains demonstration evidence, not a real-time or network-performance claim.

## Implemented Phase 1 changes recorded on 2026-10-30

1. RSU summaries carry `mean_accel_mps2` and `mean_dist_to_stop_m`; message schema version 1.1 (1.0 logs stay valid).
2. `agents.signal_broadcast` flag (default false) and `signal_remaining_s` on vehicle reports.
3. `trafma inspect` command.
4. `configs/data_v2.yaml` (160 runs), `configs/data_v2_dev.yaml` (48 runs) and `configs/data_v2_broadcast.yaml` (broadcast ablation config); `configs/data.yaml` and `data/phase2_v1/` unchanged.
5. `requirements.txt` pinned to SUMO/traci/sumolib 1.28.0; Phase 2 packages are installed in the local development environment.
6. `.gitignore` no longer ignores `docs/`; this file and `ARCHITECTURE_AS_BUILT.md` added; README updated.

## Decisions

| ID | Decision | State |
|---|---|---|
| D1 | Queue is the contiguous halted chain in metres; congestion thresholds are 0.10 / 0.35. Freeze these for comparisons; never tune on test results | Implemented in code/config; development results use held-out sampled windows |
| D2 | Signal state + time-to-change broadcast to vehicles is a config flag, default off; compare settings only in a matched ablation | Flag implemented; ablation dataset not generated |
| D3 | The signal stays an environment controller, not an agent | Current implementation fact |
| D4 | RSU learning clients: four persistent identities (`rsu_N/E/S/W`) each train only on that approach's own sensor windows pooled across training runs; each self-update is counted once alongside vehicle child updates | Approved by the user for P2.5; implemented with 2,000 sampled training windows per RSU and group-mass FedAvg |
| D5 | Independent local-only benchmark (B1) for every eligible training client, including all four RSUs and every vehicle client with at least 16 windows; list ineligible clients and counts | Required by the user's Phase 2 completion scope; evaluator and full results remain open |
| D6 | v2 uses 10/5/5 train/validation/test seeds per scenario; v2_dev uses 3/1/2 | Configs and generated datasets verified; seed reservation for Phase 3 is a plan, not an implemented calibration set |
| D7 | Per-client window cap or stride | Flat natural replay is uncapped for the selected cohort; hierarchical persistent RSU learners use a deterministic 2,000-window cap, recorded in each run manifest. |
| D8 | Sensor noise | Deferred to Phase 3 |
| D9 | Candidate 22-value feature mapping | Implemented and audited on v1, v2_dev and full v2; schema hash is stable across all three. |

## Open items to close Phase 1

- Run `simulate --gui` on a real desktop and keep screenshots or a short recording.
- Save a presentation demo with real congestion (the N=20 demo's largest queue is about 3 vehicles), e.g. `python -m trafma simulate --vehicles 100 --seed 42 --run-id demo_n100_s42 --output-dir examples`.
- Run `doctor` and `pytest` once on the team's Windows machines and record versions and steps here.

## Known data limitations

- The moderate class remains smaller than free/heavy in v2: the full-v2 training RSU label histogram is `[163078, 53682, 58760]`; vehicle histogram is `[792575, 294898, 1037692]`. These are overlapping-window counts, not independent run replicates.
- Original full-test development baseline: `runs/phase2/baseline_phase2_v2_dev_20261031T104426Z/`. On matched test RSU windows (55,104), persistence scored MAE 28.94 m / RMSE 48.93 m / macro-F1 0.5211; pooled signal-aware Ridge scored 19.69 m / 27.77 m / 0.5909. Ridge on vehicle test rows (420,753) scored MAE 37.47 m / RMSE 45.62 m / macro-F1 0.4653.
- A later baseline run with a train-only majority-class diagnostic is `runs/phase2/baseline_phase2_v2_dev_20261031T123905Z/`; its training class histogram is `[281368, 104797, 314928]` and its majority-class test macro-F1/balanced accuracy are 0.2086/0.3333 on the shared 10,000-row cohort. Comparisons in this section use overlapping-window descriptive scores, not independent simulation replicates or final full-v2 results.
- On v1, vehicle windows over-represent heavy congestion (about 51% heavy vs about 24% for RSUs) because queued vehicles linger and produce many windows.
- Windows have stride 1 and overlap heavily; evaluate at run level, not window level.
