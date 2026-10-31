---
name: trafma-traffic-prediction
description: Prepare TRAF-MA local traffic datasets, heterogeneous encoders, future labels and congestion or queue forecasting baselines and models.
---

# TRAF-MA local prediction

Resolve the TRAF-MA root from the current workspace using `AGENTS.md` and `pyproject.toml`; do not assume an old absolute path. Read AGENTS.md, docs/STATUS.md, docs/DATA_CONTRACTS.md, Phase 2 in docs/IMPLEMENTATION_PLAN.md and the baseline sections of docs/EVALUATION.md. Request the location if these files are unavailable.

Keep the current simulation, dataset, and recorded metrics as the baseline. Do not overwrite or silently regenerate baseline data; place new datasets, models, and results in separate versioned locations.

Use current RSU/vehicle log fields and explicit availability masks. The 22-value common schema is a proposal, not an implemented or validated contract. Inspect manifests and representative raw logs first: manifests contain labels and client aggregates, while model inputs must be built from each agent's own log. Verify every feature's source, unit, missingness, normalization and observation boundary before freezing the schema and hash.

Build samples from one agent/run/approach history. Current labels are the future contiguous halted-queue length in metres and free/moderate/heavy classes at queue/zone ratios 0.10/0.35, 30 seconds after the input window. They are not stopped-vehicle counts or the older 0.2/0.5 proposal. Enforce whole-run splits before windowing; train-only fitting of learned normalizers; no future/attack/evaluator truth in inference inputs.

Separate observed current queues from forecasts. Implement persistence and local/central references before presenting neural-model gains. Use the documented small common GRU initially; assess class support and per-approach/type errors, not accuracy alone.

Short trajectories may not train: record eligibility and skipped samples. Use declared offline trajectory-client replay, never merge unrelated identities. Keep each RSU's own sensor records distinct from received vehicle histories. Whether RSUs become persistent learning clients remains proposed; do not assume it is approved.

Verify shape/units/masks, future alignment, split leakage and tiny-data learnability before scaling experiments. Save dataset/split/model hashes, configuration, losses and held-out metrics; record negative results. Update STATUS with actual evidence.
