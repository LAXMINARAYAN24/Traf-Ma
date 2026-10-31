# TRAF-MA agent instructions

## Read before implementation

Read `docs/STATUS.md`, `docs/ARCHITECTURE_AS_BUILT.md`, `docs/PRD.md`, `docs/IMPLEMENTATION_PLAN.md` and the relevant design docs. Before changing messages, features, labels or model signatures, verify actual code and saved data, then read `docs/DATA_CONTRACTS.md`. Trust work additionally reads `docs/TRUST_AND_SECURITY.md`; experiments read `docs/EVALUATION.md`.

## Scope and priorities

- The current implementation is the baseline: one intersection, four persistent roadside agents `rsu_N/E/S/W`, configurable N vehicles, and vehicle → parent RSU → global coordinator. Do not rename RSUs to cameras to satisfy stale diagrams.
- Follow Phase 1, then Phase 2 local prediction before hierarchical FL, then Phase 3 trust/security/dashboard. Do not silently add later-phase features to an earlier milestone.
- Treat the implementation and verified artifacts present when work begins as the baseline. Preserve their behavior and outputs by default; implement new work additively and keep new experiment outputs separate.
- Change baseline behavior only when the user explicitly requests it. State the intended difference, update affected contracts/decisions, and retain a reproducible baseline for comparison.
- When sources conflict, use this order: executable code and its tests; generated manifests/logs; `docs/ARCHITECTURE_AS_BUILT.md`; current `docs/STATUS.md`; then design plans/contracts and historical source material. Flag stale docs and reconcile them before relying on them.
- Some planning docs specify a signal agent, camera IDs, stopped-vehicle-count labels or different congestion thresholds. Those are not current implementation facts.
- Preserve original `Architecture/`, `PPT/`, `References/`, `Report/` files and the chat export. Maintain new implementation documents in `docs/`.
- Latest user instructions take precedence. Otherwise record material design changes in `docs/DECISIONS.md` and update dependent contracts together. Routine implementation decisions do not need a new approval step.

## Engineering invariants

- Python application, SUMO/TraCI simulation, PyTorch common predictor, custom inspectable FedAvg; FastAPI/React dashboard in Phase 3.
- Start with one process and simulated agent/message isolation. Agents are software objects, not language-model agents or separate machines.
- Only the SUMO adapter can read the simulator. Agent observation interfaces enforce coverage. Separate the evaluator/label oracle from live inference and trust.
- Preserve source identity, region, time, schema/model version and provenance in all messages. Reject duplicate, stale, wrong-round and incompatible updates.
- Raw local histories stay in per-agent storage. Upstream messages contain permitted compact summaries or model updates. Do not describe data locality as a formal privacy guarantee.
- Shared model tensors must have identical names/shapes/semantics. Do not average heterogeneous private encoders or duplicate a camera's child data as camera training data.
- Keep current observations, future predictions, model parameters, trust and consensus as separate concepts and records.
- Physical units, missing-value masks, region scope and prediction horizons are explicit. Missing evidence is not evidence of wrongdoing.
- Use configuration for N, seeds, timings, demand, model/trust parameters and attack schedules. Log the resolved configuration and software versions for every run.
- Preserve stable agent IDs through a run. Quarantine removes influence, not the vehicle from SUMO. Continue bounded evidence monitoring for recovery.
- Never put attack labels, future labels, or private evaluator truth into detector/model inputs.

## Work tracking and verification

- Complete one milestone and its meaningful checks at a time; update `docs/STATUS.md` with artifacts and evidence.
- Do not invent accuracy, robustness, installation status or passing checks. Record negative results and inconclusive experiments.
- Gate progression on reproducibility and correctness. Model improvement is a research question, not a promised result.
- Use CPU smoke checks first. Add dependencies as their milestone needs them, pin tested versions, and document verified Windows commands.
- The project has an executable implementation. Treat `docs/STATUS.md` as the evidence ledger, verify current behavior before changing it, and replace stale planned commands with commands verified in the current environment.
- Dataset manifests contain labels and aggregate client statistics, not model feature tensors. Verify candidate features against the actual per-agent logs; distinguish observed results from a sample audit and from unimplemented proposals.

## Skills

Use the narrowly relevant TRAF-MA skill listed in `docs/SKILLS.md`; its source is in `skills/<name>/SKILL.md`. If it is not auto-discovered, read that file explicitly. Do not load every skill for a small change.
