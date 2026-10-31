# Status and decisions

Updated: 2026-10-30. Everything below marked "verified" was run; nothing marked "proposed" is a decision the team has taken.

## Where the project is

- **Phase 1 (P1.1-P1.6): functionally complete.** One gate item is still open: `simulate --gui` has only been verified under a virtual display, not on a real desktop.
- **P2.1 (data and labels): done**, and extended by the v2 configs (not yet generated at full size).
- **P2.2 onward (baselines, feature encoder, GRU, FedAvg, hierarchical FedAvg): not started.** No model has been trained and no learning result exists.

## Verified in this session (Linux, Python 3.12, SUMO/traci/sumolib 1.28.0, one core)

| Check | Result |
|---|---|
| `python -m trafma doctor` | READY |
| `python -m pytest` | 126 passed (the 102 original tests, one assertion updated, plus 24 new) |
| `simulate --vehicles 20 --seed 42` trace SHA-256 | `e2c4145f...`, identical to `examples/demo_n20_s42`, with and without `agents.signal_broadcast` |
| `trafma gate --run <new run> --reproduce` | PASS (deterministic re-run identical) |
| `trafma collect --config configs/data.yaml` | `dataset_report.txt` identical to the committed one |
| Real SUMO run of the 4 new v2 scenarios (1 train, 1 val, 1 test seed each, 16 runs) | all ran to the end of the 900 s (over-saturated scenarios finish with unfinished vehicles reported, not as completed), 0 collisions, 0 teleports, integrity block clean, new RSU fields present in logs |

Not verified: GUI on a real desktop, Windows, the full 160-run v2 collection, any model training.

## Changes made in this session

1. RSU summaries carry `mean_accel_mps2` and `mean_dist_to_stop_m`; message schema version 1.1 (1.0 logs stay valid).
2. `agents.signal_broadcast` flag (default false) and `signal_remaining_s` on vehicle reports.
3. `trafma inspect` command.
4. `configs/data_v2.yaml` (160 runs, estimated 4-5 GB), `configs/data_v2_dev.yaml` (48 runs, estimated 1.3 GB) and `configs/data_v2_broadcast.yaml` (same size as the full set); `configs/data.yaml` and `data/phase2_v1/` unchanged.
5. `requirements.txt` pinned to SUMO/traci/sumolib 1.28.0; `requirements-phase2.txt` added (not yet used by code).
6. `.gitignore` no longer ignores `docs/`; this file and `ARCHITECTURE_AS_BUILT.md` added; README updated.

## Decisions

| ID | Decision | State |
|---|---|---|
| D1 | Keep the repo's queue definition (halted chain in metres) and congestion thresholds 0.10 / 0.35; freeze them before any training and never tune on test results | Proposed, implemented as-is |
| D2 | Signal state + time-to-change broadcast to vehicles is a config flag, default off; run both settings and report the difference | Flag implemented; running the ablation is proposed |
| D3 | The signal stays an environment controller, not an agent (plan D06) | Proposed; team to confirm |
| D4 | RSU learning clients: 4 persistent clients (union of each RSU's windows over training runs), not 48 per-run clients | Proposed for P2.4 |
| D5 | Vehicle independent-local baseline (B1) on a stratified sample (about 40 per approach) | Proposed for P2.2/P2.3 |
| D6 | Dataset size: full v2 follows the plan's 10/5/5 seeds per scenario; a 3/1/2 dev grid is used while developing; seeds 400-404 reserved for the Phase 3 calibration set | Configs implemented, full collection not yet run. Disk is the constraint: about 27 MB per run on average (measured on v1) |
| D7 | Per-client window cap or stride | Main benchmark uncapped; capped variant reported separately (proposed) |
| D8 | Sensor noise | Deferred to Phase 3 |

## Open items to close Phase 1

- Run `simulate --gui` on a real desktop and keep screenshots or a short recording.
- Save a presentation demo with real congestion (the N=20 demo's largest queue is about 3 vehicles), e.g. `python -m trafma simulate --vehicles 100 --seed 42 --run-id demo_n100_s42 --output-dir examples`.
- Run `doctor` and `pytest` once on the team's Windows machines and record versions and steps here.

## Known data limitations (measured on v1)

- "Moderate" is the thin class (about 19% of RSU train windows) and mostly occurs during build-up or discharge.
- Persistence (current queue as the forecast) is a weak baseline because the 120 s signal cycle drives the queue; Phase 2 needs a signal-aware non-neural reference as well.
- Vehicle windows over-represent congestion (about 51% heavy vs about 24% for RSUs) because queued vehicles linger and produce many windows.
- Windows have stride 1 and overlap heavily; evaluate at run level, not window level.
