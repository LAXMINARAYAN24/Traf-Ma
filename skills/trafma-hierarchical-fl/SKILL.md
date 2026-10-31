---
name: trafma-hierarchical-fl
description: Implement or review TRAF-MA federated rounds, vehicle-to-RSU edge aggregation, global aggregation and model distribution with correct participation and weighting.
---

# TRAF-MA hierarchical federation

Resolve the TRAF-MA root from the current workspace using `AGENTS.md` and `pyproject.toml`; do not assume an old absolute path. Read AGENTS.md, docs/STATUS.md, docs/ARCHITECTURE.md, docs/DATA_CONTRACTS.md and Phase 2 in docs/IMPLEMENTATION_PLAN.md. For trust weighting also read docs/TRUST_AND_SECURITY.md. Request the project location if unavailable.

Keep this later-phase work separate from the current baseline. Do not change Phase 1 behavior or baseline artifacts as a side effect; make any intended baseline change explicit and preserve a reproducible comparison point.

Current code has RSU sensor/relay roles, but no learning clients. Before implementation, resolve client identity and whether RSUs train independently or only aggregate; the persistent four-RSU client idea remains proposed in STATUS D4. If RSU self-training is approved, keep its learner separate from the aggregator role and count its update once. Never train it on pooled child histories. Federate only tensors with identical signature, semantics and base version. n_i counts distinct windows, not epochs times windows.

Freeze membership and base model per round. Reject duplicates, stale/wrong-round/incompatible/nonfinite updates. New arrivals join at the next boundary. Empty groups contribute no mass; an all-empty round retains the prior model.

Use sample mass at the edge and total accepted group mass globally. With one edge step and identical updates, hierarchical and flat averages must match within floating-point tolerance. Do not describe this algebraic regrouping as an accuracy advance.

In Phase 3 compute effective client mass once from sample count, frozen trust and eligibility/probation. Sum that mass at both levels; never apply RSU trust to descendants again. Individual update validation occurs before aggregation at the trusted edge. Declare that trust boundary.

Test unequal groups, missing clients, rejected tensors and zero mass using hand-calculated fixtures. If RSU learner clients are approved, also test the RSU self-update is counted once. Save accepted identities, masses, model versions, bytes and timings. Distinguish offline replay clients from online connected vehicles. Update STATUS only after the requested checks.
