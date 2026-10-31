# TRAF-MA product requirements

> This is a requirements/design document, not the as-built record. Executable code/tests and generated artifacts define current behavior; use [ARCHITECTURE_AS_BUILT.md](ARCHITECTURE_AS_BUILT.md) and [STATUS.md](STATUS.md) for verified implementation facts. In particular, the implemented roadside IDs are `rsu_N/E/S/W`, not camera IDs.

## Product and problem

TRAF-MA is a simulated traffic intelligence system that combines partial vehicle and roadside observations to predict traffic and, in later phases, study unreliable reports and model updates. The current implementation calls its four roadside agents RSUs. Honest agents can see different traffic conditions; disagreement alone must not establish maliciousness.

The final outcome is a repeatable local demonstration and a defensible research evaluation. Intended users are the student implementation team, project supervisor and examiners.

## Fixed first demo

One signalised four-way SUMO intersection; four logical roadside agents implemented as `rsu_N/E/S/W`, one per approach; configurable N vehicle agents; vehicle reports go through the associated RSU to one global coordinator. RSUs are simulated sensor agents, not actual video-processing cameras. The traffic signal is an environment object with a fixed schedule, not a fifth agent.

N means the total number of vehicle agents scheduled for one run, not a constant simultaneous population. Log scheduled, departed, active, arrived and not-yet-inserted counts separately. A later sustained-demand configuration can use departure rate and duration without redefining N.

## Requirements by phase

| ID | Requirement | Phase | Acceptance evidence |
| --- | --- | --- | --- |
| R01 | Reproducible single-intersection SUMO world | 1 | Scenario files, seed/config manifest and successful headless/GUI runs |
| R02 | Four local RSUs and configurable vehicle population | 1 | Registry and observation-scope tests |
| R03 | Hierarchical communication | 1 | Trace shows each vehicle report routed via its parent RSU once |
| R04 | Local collection and heterogeneous feature representation | 1 raw; 2 model encoding | Per-agent storage, coverage/mask fixtures |
| R05 | Congestion classification and queue prediction at a future horizon | 2 | Held-out metrics against persistence and local/central baselines |
| R06 | Non-IID data and common local models | 2 | Per-client distribution summaries and compatible model signatures |
| R07 | Hierarchical FL with central model coordinator | 2 | Two-stage aggregation tests, checkpoints and round logs |
| R08 | Context-aware trust and anomaly detection | 3 | Evidence traces and honest-disagreement controls |
| R09 | Trust-aware aggregation and traffic-state consensus | 3 | Effective weight audit, unknown-state fallback and attack comparisons |
| R10 | False-information and model-poisoning experiments | 3 | Separate reproducible attack hooks and schedules |
| R11 | Quarantine and gradual trust recovery | 3 | Recorded lifecycle, no quarantined contribution and relapse test |
| R12 | Global traffic state and alerts | Basic state in 1; trusted state in 3 | Per-approach state, confidence/coverage and distinct incident/anomaly events |
| R13 | Live dashboard and replay | 3 | Four-approach RSU view, learning/trust history, accessible status and stale-data handling |
| R14 | Performance/security/end-to-end evaluation | Every gate; final in 3 | Reproduction instructions, per-run metrics, plots and limitations |
| R15 | Literature, architecture and methodology traceability | Starts before 1; maintained throughout | Source matrix, decisions and final report alignment |

## User journeys

1. Choose N, demand profile and seed; start a clean run; observe vehicles and parent-RSU ownership; inspect its summary.
2. Generate a dataset; train local and federated predictors; compare held-out congestion/queue predictions and cost.
3. Run a saved attack scenario; see evidence and reduced influence; stop the attack; see monitored recovery.
4. Replay the same run and export evidence for the project report.

## Success criteria

All three phase gates pass; no observation leakage or duplicate contributions; controlled attacks and honest disagreement are evaluated separately; metrics are reproducible and include uncertainty. Improved prediction and robustness are hypotheses to measure. A valid negative result still satisfies research evaluation; it does not justify claiming a successful defence.

## Core scope limits

No real camera hardware/video detection, physical deployment, adaptive signal control, city-scale road network, weather/drone agents, blockchain, cryptographic secure aggregation, differential privacy, learned trust MLP, or formal Byzantine convergence proof in the core build. These remain extensions after the complete three-phase system. Attack experiments operate only within the local simulator.

Data locality reduces raw-data exchange. Model updates and semantic summaries can still leak information. The proposal's phrase “formal security analysis” will be fulfilled as an explicit threat model, assumptions, invariants and empirical robustness analysis; a mathematical proof would require a separate specified theorem and is not implied by experiments.
