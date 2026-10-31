# Decisions and source reconciliation

Date: 2026-10-31. This log separates as-built facts from design proposals. Current code/tests and generated artifacts outrank this log; see `docs/ARCHITECTURE_AS_BUILT.md` and `docs/STATUS.md`.

| ID | Decision | Status and reason |
| --- | --- | --- |
| D01 | Exactly three implementation phases | Fixed: current request supersedes the old two-semester presentation structure |
| D02 | Phase 1 as built: one intersection, four persistent roadside agents `rsu_N/E/S/W`, configurable N vehicles, vehicle → parent RSU → coordinator | Implemented. Camera wording in older docs is conceptual and does not change current IDs. |
| D03 | Phase 2: local prediction first, then hierarchical FL | Default allocation of the remaining work, preserving the requested incremental order |
| D04 | Phase 3: trust, attacks, consensus, recovery, dashboard and final experiments | Default allocation; avoids security complexity before a working clean baseline |
| D05 | Simulated roadside sensing through bounded TraCI observations | Implemented; no video/object-detection pipeline |
| D06 | Traffic signal remains a fixed environment controller | Implemented; it is not an agent |
| D07 | N counts total scheduled vehicles per run; active count varies | Default that removes ambiguity; persistent demand rate can be an additional configuration |
| D08 | RSU's own sensor evidence is distinct from relayed vehicle evidence | Implemented. D4 gives each RSU an independent sensor-only learning partition; its update is not trained on child histories. |
| D09 | Common 22-value feature adapter and GRU | `trafma.features/1` and the two-head GRU are implemented; feature audits pass on v1, v2_dev and full v2. |
| D10 | Forecast approach queue and congestion 30 s ahead from a 10-step history | Label/window contract and GRU heads are implemented; development evaluation is descriptive and held-out. |
| D11 | Per-bin current state and per-approach future predictions are separate | Design proposal; per-bin coverage/prediction messages are not implemented. |
| D12 | One local step schedule, one edge aggregation per global round initially | Implemented for offline replay; each update starts from one frozen global model version and group mass is preserved. |
| D13 | Offline trajectory-client replay before transient online FL | Implemented as the repeatable benchmark and deterministic transient-participation replay; a live online participation demo remains open. |
| D14 | Clean calibration data at trusted RSU edges for model-quality evidence | Proposed; no calibration dataset or trust implementation exists |
| D15 | Trust factors/weights/EMA begin from report/chat suggestions | Experimental proposals, not immutable facts |
| D16 | Reject harmful channel submissions and quarantine persistent corroborated evidence | Phase 3 proposal; not implemented |
| D17 | Explicit RECOVERING state with monitored probation | Phase 3 proposal; not implemented |
| D18 | Routing/aggregation code and coordinator are trusted in the threat model | Proposed trust boundary; not yet implemented as a security guarantee |
| D19 | Coordinator fusion is operational traffic-state fusion, not distributed Byzantine consensus | Naming constraint for future implementation |
| D20 | FastAPI/React dashboard in Phase 3 | Planned stack; no dashboard implementation |
| D21 | Data locality and empirical attack evaluation | Do not imply formal privacy or mathematical security guarantees |
| D22 | New context in docs; original source artifacts preserved | Follows pasted Markdown guide without altering the evidence trail |
| D23 | Project-specific skill sources in skills; discoverable copies in personal Codex skills | Default skill-creator location; skills are narrowly scoped to this project |

## Choices to validate during their milestone

- P1: real-desktop GUI evidence remains open. Generated run manifests record Windows SUMO 1.27.1; the historical Linux test environment used SUMO 1.28.0.
- P2: feature audit/schema freeze, client identity and RSU learner role, calibration representativeness, viable training cost and useful prediction horizon. Use existing manifest counts as measured data; tune defaults using validation data only.
- P3: evidence thresholds, alert persistence, recovery timing and cost/false-positive tradeoffs. Learned trust MLP is an optional comparison after the interpretable baseline, not a replacement for physical validity checks.

P1 is implemented. D4 is resolved: the user approved four persistent sensor-only RSU learners (`rsu_N/E/S/W`), each formed by pooling that approach's own training-run sensor windows. RSUs also aggregate their assigned vehicle updates, and each RSU learner update is counted once. v1, v2_dev and v2 feature audits pass; the GRU, local/central training, flat FedAvg, hierarchical FedAvg, IID control and transient offline participation replay are implemented. The live participation demo and full-v2 model benchmark remain open.

## Final-source updates still needed after implementation

Update final report/slides/diagrams to match the as-built RSU identifiers and configurable N, the feature schema only after validation, the approved HFL client policy, measured results and actual threat assumptions. The original documents are reference material; edit submission copies during the final reporting milestone. Audit any claimed literature gap against the cited papers rather than repeating the chat's wording.
