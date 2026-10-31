# Status and decisions

Updated: 2026-10-31. Everything marked "verified" is supported by a run or artifact audit; "proposed" items remain undecided.

## Where the project is

- **Phase 1 (P1.1-P1.6): functionally complete.** One gate item is still open: `simulate --gui` has only been verified under a virtual display, not on a real desktop.
- **P2.1 (data and labels): done.** The v1, v2 development and full v2 datasets are present locally and their manifests pass the listed integrity checks. The signal-broadcast dataset has a config only and has not been generated.
- **P2.2 onward (baselines, feature encoder, GRU, FedAvg, hierarchical FedAvg): not started.** No model has been trained and no learning result exists.

## Previously verified implementation checks (Linux, Python 3.12, SUMO/traci/sumolib 1.28.0, one core)

| Check | Result |
|---|---|
| `python -m trafma doctor` | READY |
| `python -m pytest` | 126 passed (the 102 original tests, one assertion updated, plus 24 new) |
| `simulate --vehicles 20 --seed 42` trace SHA-256 | `e2c4145f...`, identical to `examples/demo_n20_s42`, with and without `agents.signal_broadcast` |
| `trafma gate --run <new run> --reproduce` | PASS (deterministic re-run identical) |
| `trafma collect --config configs/data.yaml` | `dataset_report.txt` identical to the committed one |
| Real SUMO run of the 4 new v2 scenarios (1 train, 1 val, 1 test seed each, 16 runs) | all ran to the end of the 900 s (over-saturated scenarios finish with unfinished vehicles reported, not as completed), 0 collisions, 0 teleports, integrity block clean, new RSU fields present in logs |

Not verified: GUI on a real desktop, Windows `doctor`/test validation and any model training.

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
- Feature/model code is still absent. Dataset manifests contain labels and aggregate client statistics, not feature tensors. A full scan of RSU logs found acceleration/distance keys in all rows, with non-null values on 79.1% of v1, 80.5% of v2_dev and 79.6% of v2 RSU records; the rest are empty-zone means. This disproves the old note that v1 data lacks these fields. The 22-value feature schema remains a candidate; density denominators, masks, record compatibility and proxy semantics must be tested before it is frozen.

## Implemented Phase 1 changes recorded on 2026-10-30

1. RSU summaries carry `mean_accel_mps2` and `mean_dist_to_stop_m`; message schema version 1.1 (1.0 logs stay valid).
2. `agents.signal_broadcast` flag (default false) and `signal_remaining_s` on vehicle reports.
3. `trafma inspect` command.
4. `configs/data_v2.yaml` (160 runs), `configs/data_v2_dev.yaml` (48 runs) and `configs/data_v2_broadcast.yaml` (broadcast ablation config); `configs/data.yaml` and `data/phase2_v1/` unchanged.
5. `requirements.txt` pinned to SUMO/traci/sumolib 1.28.0; `requirements-phase2.txt` added (not yet used by code).
6. `.gitignore` no longer ignores `docs/`; this file and `ARCHITECTURE_AS_BUILT.md` added; README updated.

## Decisions

| ID | Decision | State |
|---|---|---|
| D1 | Queue is the contiguous halted chain in metres; congestion thresholds are 0.10 / 0.35. Freeze these for comparisons; never tune on test results | Implemented in code/config; no model evaluation yet |
| D2 | Signal state + time-to-change broadcast to vehicles is a config flag, default off; compare settings only in a matched ablation | Flag implemented; ablation dataset not generated |
| D3 | The signal stays an environment controller, not an agent | Current implementation fact |
| D4 | RSU learning clients: 4 persistent clients (union of each RSU's windows over training runs), not 48 per-run clients | Proposed for P2.4 |
| D5 | Vehicle independent-local baseline (B1) on a stratified sample (about 40 per approach) | Proposed for P2.2/P2.3 |
| D6 | v2 uses 10/5/5 train/validation/test seeds per scenario; v2_dev uses 3/1/2 | Configs and generated datasets verified; seed reservation for Phase 3 is a plan, not an implemented calibration set |
| D7 | Per-client window cap or stride | Main benchmark uncapped; capped variant reported separately (proposed) |
| D8 | Sensor noise | Deferred to Phase 3 |

## Open items to close Phase 1

- Run `simulate --gui` on a real desktop and keep screenshots or a short recording.
- Save a presentation demo with real congestion (the N=20 demo's largest queue is about 3 vehicles), e.g. `python -m trafma simulate --vehicles 100 --seed 42 --run-id demo_n100_s42 --output-dir examples`.
- Run `doctor` and `pytest` once on the team's Windows machines and record versions and steps here.

## Known data limitations

- The moderate class remains smaller than free/heavy in v2: the full-v2 training RSU label histogram is `[163078, 53682, 58760]`; vehicle histogram is `[792575, 294898, 1037692]`. These are overlapping-window counts, not independent run replicates.
- Evaluate persistence against a signal-aware non-neural reference; the 120 s signal cycle may make persistence weak, but no model baseline has been measured yet.
- On v1, vehicle windows over-represent heavy congestion (about 51% heavy vs about 24% for RSUs) because queued vehicles linger and produce many windows.
- Windows have stride 1 and overlap heavily; evaluate at run level, not window level.
