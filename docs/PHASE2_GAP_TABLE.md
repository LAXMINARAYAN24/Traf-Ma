# Phase 2 remaining-work gap table

Assessment date: 2026-10-31. The first matrix below is the required pre-change inventory. It records the repository state before implementation work and remains unchanged as an audit trail. The post-change evidence and current status are recorded at the end of this file; a smoke run is not a completed benchmark.

## Baseline checks

| Check | Result |
|---|---|
| `python -m trafma doctor` | READY; Python 3.12.7, SUMO 1.27.1, TraCI 0.0.0, sumolib 1.27.1; minimal simulation stepped and closed |
| `python -m pytest` using PATH Python | Could not start: PATH Python 3.12.7 does not have pytest installed |
| `.\.venv\Scripts\python.exe -m pytest` | 150 passed in 38.27 s |
| Full-v2 dataset | `data/phase2_v2/` contains its manifest, report, feature audit, and 160 run folders |
| Full-v2 feature audit | `data/phase2_v2/feature_audit.json`: passed; 576,000 RSU and 5,471,540 vehicle records; schema SHA-256 `46196acbcc933a9b60f0a3151cf3a5822350c6988b7894a50f5722ac80f4abe2` |
| Full-v2 training eligibility | Manifest lists 17,529 train-split eligible clients: 320 RSU-approach/run clients and 17,209 vehicle trajectory clients (minimum 16 windows) |
| Signal-broadcast ablation | `configs/data_v2_broadcast.yaml` exists; generated `data/phase2_v2_bcast/` was not present |
| RSU self-update switch | Existing federation always includes persistent RSU self-updates; `rsu_self_update` is not a configuration option and no false-mode ablation artifact exists |

## Guide section 8 deliverables

| # | Deliverable | Status | Current evidence and remaining gap |
|---:|---|---|---|
| 1 | Updated source code and tests | PARTIAL | Features, GRU, baseline, flat/hierarchical aggregation, and 150 tests exist in `trafma/` and `tests/`. Missing configurable `rsu_self_update`, its false ablation, paired-seed runner/inference, broad local-client evaluation, and live SUMO federation. |
| 2 | `docs/PHASE2_IMPLEMENTATION.md` architecture, flow, assumptions, commands | PARTIAL | Current document describes offline P2.2–P2.7 and D4, but not full-v2 learning, paired confidence/significance tests, per-client local results, switch ablation, or live-runner behavior. |
| 3 | `docs/STATUS.md` verified/proposed/not implemented | PARTIAL | Existing status honestly lists full-v2 model training, paired seeds, broad local evaluation, and online demo as open; update it only after evidence is produced. |
| 4 | Development audit and final full-dataset audit | DONE | `data/phase2_v2_dev/feature_audit.json` and `data/phase2_v2/feature_audit.json` pass; full-v2 audit records all 160 runs and the frozen schema hash. |
| 5 | Baseline results and signal-aware reference | PARTIAL | Ridge, persistence, and train-only majority artifacts exist for v2_dev, e.g. `runs/phase2/baseline_phase2_v2_dev_20261031T123905Z/`; no full-v2 baseline artifact exists. |
| 6 | Local GRU and pooled-central results | PARTIAL | One local smoke and pooled-central GRU artifacts exist for v2_dev; no full-v2 benchmark, every-eligible-client local evaluation, paired seeds, or run-level bootstrap intervals. |
| 7 | Flat FedAvg and hierarchical FedAvg results | PARTIAL | v2_dev flat/IID/hierarchical artifacts and hand-calculated equivalence tests exist. No full-v2 benchmark or paired five-seed results; hierarchical RSU self-update cannot be disabled. |
| 8 | IID versus natural non-IID comparison | PARTIAL | v2_dev IID and natural replay runs plus grouped JS summaries exist. Per-client distributions, explicit IID-partition divergence comparison, full-v2 results, and run-level inference are missing. |
| 9 | Transient participation replay and live demonstration | PARTIAL | `--transient-participation` is deterministic offline replay only. `trafma/runner.py` has no live learning client, matured-label eligibility, simulated-time round scheduler, or online federation log. |
| 10 | Final comparison with scenario, approach, and client-type breakdowns | PARTIAL | The v2_dev saved report has a matched sampled-row comparison and grouped breakdowns. No full-v2 / five-training-seed paired report, run-level bootstrap CI, paired significance test, or all-client local-vs-FL comparison. |
| 11 | Saved checkpoints, resolved configs, manifests, predictions, and round logs | PARTIAL | These artifacts exist for v2_dev experiments. Full-v2 results and paired-seed/result summaries do not exist; reports currently use JSON/CSV outputs but no five-seed statistical bundle. |
| 12 | Plots and generated Markdown/HTML report | PARTIAL | `runs/phase2/report_phase2_20261031T123953Z/` contains the v2_dev report and plots. Full-v2 paired confidence intervals, local-client distributions, online events, and switch-ablation plots are absent. |
| 13 | Repeatable demo script and setup instructions | MISSING | README documents offline CLI commands, but there is no dedicated repeatable Phase 2 live-demo script or run-through instructions. |
| 14 | Explicit limitations | PARTIAL | Existing docs state offline replay, overlapping-window, privacy, and simulation limitations. Add actual full-v2, paired/local, and live-runner evidence/limits when implemented; do not imply real radio or deployment performance. |

## Additional contract checks

| Item | Status | Evidence |
|---|---|---|
| Full-v2 model benchmark | MISSING | Full-v2 feature audit exists, but `runs/phase2/` contains only v2_dev model/baseline/report artifacts. |
| Paired multi-seed training/evaluation | MISSING | Current learning config has one training seed (`20261031`); no per-seed run summary or run-level bootstrap/paired test implementation. Keep dataset split seeds 400–404 unused. |
| Local evaluation for every eligible training client | MISSING | Current `train --method local --client ...` accepts one client. Full-v2 manifest reports 17,529 eligible clients, so evaluation needs a streaming/batched runner and explicit resource assessment. |
| `rsu_self_update: true|false` and false ablation | MISSING | `configs/learning.yaml` and hierarchical runner have no switch; the implementation always creates RSU learners. Headline must use true; false must remove RSU learner updates without reassigning or double-counting child windows. |
| Signal-broadcast ablation dataset | MISSING | Config is present; no generated broadcast dataset is present. The guide marks this comparison optional and requires matched scenarios/seeds and separate datasets. |
| Live online federation | MISSING | Current simulation runner has no learning round protocol; offline `fedavg` is saved-trajectory replay. |

## File-level implementation plan

1. Add the self-update configuration and hierarchy tests first; preserve true as the headline default and verify both true/false aggregation against flat FedAvg on identical updates.
2. Add full-v2 config overrides and run the already-implemented audit/model/baseline pipeline into new immutable run IDs; never overwrite v2_dev artifacts.
3. Implement a paired evaluation orchestrator that freezes the full-v2 test run IDs and uses five nonreserved model/randomization seeds. Write per-seed and per-test-run predictions/metrics, run-bootstrap 95% intervals, and paired per-run differences/tests.
4. Implement a bounded-memory local-client evaluator over all eligible train clients. Record every eligible/ineligible client and counts; report distributions by type/approach/scenario and comparisons on explicitly matched test rows.
5. Extend the non-IID report with per-client class/sample counts and JS divergence for the natural and synthetic-IID partitions.
6. Add a live SUMO driver/learning scheduler that observes only client-local records, releases supervision only after its 30-second maturity time, evaluates 10-step/16-window eligibility, and invokes the existing aggregation protocol between simulation steps at configured simulated-time boundaries. Keep the runner optional so ordinary Phase 1 simulation is unchanged.
7. Regenerate a final saved-artifact report, add a repeatable Windows demo script/instructions, then reconcile README, `docs/PHASE2_IMPLEMENTATION.md`, `docs/STATUS.md`, decisions, data contracts, and as-built architecture to the measured evidence.

## Dependency and experiment plan

- **Dependencies:** none planned. Use the existing `.venv` and installed PyTorch, NumPy, PyYAML, matplotlib, SUMO/TraCI. Verify package versions in every new run manifest. Do not install or add requirements unless a chosen existing command proves a dependency is missing.
- **Ordering:** implement/test `rsu_self_update` before new hierarchical experiments; verify existing full-v2 audit/manifests before full-v2 model runs; build paired evaluation and run-level statistics before interpreting multi-seed results; implement live scheduling only after the offline client protocol and matured-label contract are tested.
- **Full-v2 experiments:** use existing v2 train/validation/test run splits and current frozen learning defaults; any selection/tuning must be validation-only. Produce baseline, local, central, flat natural, flat synthetic-IID, hierarchical `rsu_self_update=true`, and hierarchical `false` artifacts. Match the fixed test run IDs and keep the signal-broadcast dataset separate.
- **Multi-seed experiments:** vary model initialization, data-loader shuffle and participant/client order while retaining the same data split and test run IDs; use five training seeds other than reserved 400–404. Compute uncertainty by resampling test runs, and compare methods with paired run-level differences rather than treating overlapping windows as independent observations.
- **Live demonstration:** use separate online artifact output. Record simulated time and wall time, vehicle arrivals/departures, valid history/window counts, matured labels, inference-only reasons, round membership/rejections, RSU self-update mode, and group/global sample mass. Do not fabricate active vehicle training if no live vehicle qualifies.

## Primary risks

- The full-v2 dataset has 17,529 eligible training clients; broad local training/evaluation may be CPU- and storage-intensive. Bound memory, persist progress/resumable per-client outputs, and report incomplete work rather than silently sampling.
- Window overlap can make apparently precise confidence intervals invalid; all uncertainty and paired tests must resample the 40 held-out simulation runs, not individual rows.
- Vehicle trajectories may depart before any `t+30 s` label matures. The live demo may honestly produce zero eligible vehicle updates; RSU participation must not be described as vehicle FL.
- Train-time event loops and label release are a leakage risk. Keep future labels unavailable to feature construction/inference and expose only matured targets to the owning local training cache.
- The global v2 result folders are git-ignored generated data. Keep new experiment identifiers immutable, document paths and hashes, and do not stage or commit automatically.
- Full-v2 training settings must be justified from validation artifacts only. Do not select epochs, seeds, client cohorts, thresholds, or report rows based on test outcomes.

## Post-change status (2026-11-01)

| Deliverable | Current status | Verified evidence and remaining work |
|---|---|---|
| Baselines | DONE | Full-v2 baseline saved at `runs/phase2/baseline_phase2_v2_20261031T164408Z/`; signal-aware Ridge test predictions cover 1,187,485 windows. Matched run-bootstrap comparison is in the paired statistics bundle. |
| `rsu_self_update` decision | DONE | Default-true config, false-mode aggregation equivalence test, v2_dev false-mode smoke and full-v2 five-seed artifacts for both on/off modes. |
| Paired run-level statistics | DONE | `runs/phase2/stats_phase2_v2_20261031T182654Z/`: five seeds for every learned method, 10,240 identical test rows across 40 runs, 10,000 run-bootstrap replicates, per-seed/per-run metrics and paired differences. |
| Non-IID client distributions | DONE | Full-v2 report stores distributions for all 17,529 eligible clients and matched natural/IID JSD analysis for the 16-client FL cohort. Median/p90 JSD: natural 0.081104/0.316248 bits; IID 0.000002/0.000012. |
| Live SUMO federation | IMPLEMENTED; SMOKE VERIFIED | Optional runner/CLI hook, matured-label scheduler, client eligibility/departure handling, JSONL events, checkpointing and round timing are verified in `runs/phase2/phase2_live_demo_20261031_225007_34344/` (180 simulated seconds; three accepted rounds). No departed vehicle participated in a later round. This is not a performance benchmark. |
| Repeatable Windows demo | DONE | `scripts/demo_phase2_live.ps1` runs the live smoke with a timestamped output ID. |
| Full-v2 learned methods and final report | DONE | Central, natural flat, IID, hierarchical self-update on and off each have seeds 20261031-20261035. Paired statistics are at `runs/phase2/stats_phase2_v2_20261031T182654Z/`; final Markdown/HTML report and plots are at `runs/phase2/report_phase2_v2_final_20261101T0012Z/`. |
| Broad local-only model evaluation of every eligible client | IMPLEMENTED; RUN OPEN | `phase2-local-eval` enumerates all 17,529 eligible clients, records ineligible reasons, supports resumable per-client artifacts and writes grouped median/IQR/worst-decile summaries. The population run and its metrics remain open. |
