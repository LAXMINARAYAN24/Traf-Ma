---
name: trafma-sumo-agents
description: Build or debug TRAF-MA's SUMO intersection, bounded vehicle and RSU observations, configurable population, lifecycle and hierarchical messages.
---

# TRAF-MA SUMO and agents

Resolve the TRAF-MA root from the current workspace using `AGENTS.md` and `pyproject.toml`; do not assume an old absolute path. Read AGENTS.md, docs/STATUS.md, Phase 1 in docs/IMPLEMENTATION_PLAN.md, docs/ARCHITECTURE.md and docs/DATA_CONTRACTS.md. If the project is unavailable, request its location.

Preserve the existing simulation behavior and verified traces as the baseline. Keep changes additive or opt-in by default, and write new runs to separate output locations; change baseline behavior only when explicitly requested.

The current roadside IDs are `rsu_N/E/S/W`; do not rename them to camera IDs to match old plans. Start by checking actual SUMO executables and Python connectivity. Do not infer installation from a Python package or claim a GUI/headless run succeeded without executing it. Pin the tested versions and document Windows setup commands.

One SUMO adapter owns simulation stepping and lifecycle. Supply only coverage-filtered observations to software agents. Keep full-world evaluator access separate. Four fixed RSUs cover approaches north/south/east/west; N means scheduled vehicles, while active vehicles follow departure/arrival events.

Map parent RSUs by incoming approach and retain ownership through traversal. Distinguish RSU sensor output from relayed child reports. Preserve origin, time, region and provenance on relay; duplicate paths do not create new observations.

Verify all four approaches, zero/one/multiple vehicles, insertion delays, stale/malformed messages, out-of-coverage observations and reliable cleanup. Save config/seed/version, lifecycle trace and per-agent logs. Compare seeded traces without wall-clock fields for reproducibility.

Phase 1 ends at stable simulation, routing and basic summaries. Do not introduce training or attacks merely because their later interfaces are known. Advance STATUS only with the relevant milestone evidence.
