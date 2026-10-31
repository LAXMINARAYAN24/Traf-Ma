# Status and decisions

Updated: 2026-10-31. "Verified" is backed by a run/test/artifact; outstanding Phase 2 items are explicitly listed.

## Where the project is

- **Phase 1 (P1.1-P1.6): functionally complete.** One gate item is still open: `simulate --gui` has only been verified under a virtual display, not on a real desktop.
- **P2.1 (data and labels): done.** The v1, v2 development and full v2 datasets are present locally and their manifests pass the listed integrity checks. The signal-broadcast dataset has a config only and has not been generated.
- **P2.2 (baselines):** persistence and train-only pooled signal-aware Ridge are implemented; an explicit local GRU and pooled central GRU have saved development results.
- **P2.3 (features/model):** `trafma.features/1` and the two-head GRU are implemented. The full v1, v2_dev and v2 feature scans pass with the same schema hash.
- **P2.4-P2.5 (federation):** flat FedAvg, four-group hierarchical FedAvg, persistent RSU sensor learners and synthetic IID control are implemented and have saved v2_dev runs. D4 is resolved below.
- **P2.6-P2.7:** natural non-IID and transient offline participation experiments plus a saved-artifact Markdown/HTML report and plots are implemented. The online live-participation demo, full-v2 model benchmark, paired multi-seed evaluation and final report freeze remain open.

## Previously verified implementation checks (Linux, Python 3.12, SUMO/traci/sumolib 1.28.0, one core)

| Check | Result |
|---|---|
| Current full suite (`.venv\Scripts\python.exe -m pytest -q`, Windows) | 150 passed in 36.77 s |
| `python -m trafma doctor` (Windows, Python 3.12.7) | READY; SUMO 1.27.1, TraCI/sumolib 1.28.0 |
| Pre-Phase-2 `python -m pytest` | 126 passed |
| New focused feature/baseline tests | 8 passed |
| `simulate --vehicles 20 --seed 42` trace SHA-256 | `e2c4145f...`, identical to `examples/demo_n20_s42`, with and without `agents.signal_broadcast` |
| `trafma gate --run <new run> --reproduce` | PASS (deterministic re-run identical) |
| `trafma collect --config configs/data.yaml` | `dataset_report.txt` identical to the committed one |
| Real SUMO run of the 4 new v2 scenarios (1 train, 1 val, 1 test seed each, 16 runs) | all ran to the end of the 900 s (over-saturated scenarios finish with unfinished vehicles reported, not as completed), 0 collisions, 0 teleports, integrity block clean, new RSU fields present in logs |

Not verified: GUI on a real desktop, live online federated participation, full-v2 model training, paired multi-seed comparisons and the final baseline freeze.

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

Verification after this implementation: the full suite passes, 150 tests. Full model learning on v1/full v2, paired multi-seed comparisons, broader local-client evaluation, and the live online demonstration remain open.

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
| D5 | Vehicle independent-local baseline (B1) on a stratified sample (about 40 per approach) | Proposed for P2.2/P2.3 |
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
