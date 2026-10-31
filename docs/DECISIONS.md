# Decisions and source reconciliation

Date: 2026-09-28. “Fixed” comes from the current user request. “Default” is an implementation choice that can change with documented evidence. No design default below is a measured result.

| ID | Decision | Status and reason |
| --- | --- | --- |
| D01 | Exactly three implementation phases | Fixed: current request supersedes the old two-semester presentation structure |
| D02 | Phase 1: one intersection, four cameras, configurable N vehicles, vehicle → camera → coordinator | Fixed: current request supersedes old 5–10 vehicles plus one signal-agent examples |
| D03 | Phase 2: local prediction first, then hierarchical FL | Default allocation of the remaining work, preserving the requested incremental order |
| D04 | Phase 3: trust, attacks, consensus, recovery, dashboard and final experiments | Default allocation; avoids security complexity before a working clean baseline |
| D05 | Simulated camera sensing through bounded TraCI observations | Default: no video/object-detection requirement exists; four cameras are logical roadside agents |
| D06 | Traffic signal remains a fixed environment controller | Default: no extra signal learning client needed for the fixed first demo |
| D07 | N counts total scheduled vehicles per run; active count varies | Default that removes ambiguity; persistent demand rate can be an additional configuration |
| D08 | Camera sensor/client and camera aggregator are separate logical roles | Necessary to prevent pooled histories and double-counted contributions |
| D09 | Same common GRU, deterministic type-specific adapters, corrected 22-value input | Default: retain temporal model idea while resolving scalar-vs-one-hot dimensional inconsistency |
| D10 | Forecast full approach queue and congestion 30 s ahead from 10 s partial history | Default: future prediction makes ML distinct from reading current simulator state |
| D11 | Per-bin current state and per-approach future predictions are separate | Necessary for legitimate disagreement and region-consistent fusion |
| D12 | One local step schedule, one edge aggregation per global round initially | Default: inspectable HFL; group-mass weighting reproduces flat FedAvg for identical updates |
| D13 | Offline trajectory-client replay before transient online FL | Default: short-lived vehicles may not have enough matured labels for local training |
| D14 | Clean calibration data at trusted camera edges for model-quality evidence | Explicit assumption; independent from training/test runs and evaluated by ablation |
| D15 | Trust factors/weights/EMA begin from chat suggestions, not immutable facts | Report Chapter 3 explicitly leaves design parameters experimental |
| D16 | Reject harmful channel submissions and quarantine persistent corroborated evidence | Corrects unreachable quarantine when only one weighted trust factor is bad |
| D17 | Explicit RECOVERING state with monitored probation | Follows lifecycle diagram; adds guards missing from the chat's simple EMA-only walkthrough |
| D18 | Camera routing/aggregation code and coordinator are trusted | Core threat boundary; camera sensing/self-model can be attacked separately |
| D19 | Coordinator fusion is called operational traffic consensus | No unsupported claim of distributed Byzantine convergence |
| D20 | FastAPI/React dashboard in Phase 3 | Retains stack direction in report; no Streamlit migration required |
| D21 | Data locality, empirical attack robustness and stated limits | No formal privacy guarantee or mathematical security proof implied |
| D22 | New context in docs; original source artifacts preserved | Follows pasted Markdown guide without altering the evidence trail |
| D23 | Project-specific skill sources in skills; discoverable copies in personal Codex skills | Default skill-creator location; skills are narrowly scoped to this project |

## Choices to validate during their milestone

- P1: installed SUMO version, real edge/lane/signal IDs, lane geometry, route mix and demand rates. The environment check found no SUMO commands on PATH.
- P2: actual client sample counts, label class balance, calibration representativeness, viable training cost and useful prediction horizon. Tune defaults using validation data only.
- P3: evidence thresholds, alert persistence, recovery timing and cost/false-positive tradeoffs. Learned trust MLP is an optional comparison after the interpretable baseline, not a replacement for physical validity checks.

These choices do not prevent starting P1.1. Save resolved values and reasons when they become implementation facts.

## Final-source updates still needed after implementation

Update final report/slides/diagrams to four camera agents and configurable N, corrected feature dimensions, the chosen HFL hierarchy, measured results and actual threat assumptions. The original documents are reference material; edit submission copies during the final reporting milestone. Audit any claimed literature gap against the cited papers rather than repeating the chat's wording.
