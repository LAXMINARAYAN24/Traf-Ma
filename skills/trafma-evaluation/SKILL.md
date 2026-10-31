---
name: trafma-evaluation
description: Design, run or audit TRAF-MA prediction and security experiments, reproducibility artifacts, metrics, ablations and final demonstration evidence.
---

# TRAF-MA evaluation

Resolve the TRAF-MA root from the current workspace using `AGENTS.md` and `pyproject.toml`; do not assume an old absolute path. Read AGENTS.md, docs/STATUS.md, docs/EVALUATION.md and relevant data/trust contracts. For literature comparisons also read docs/SOURCE_REVIEW.md and the cited original paper. Request the location if unavailable.

Treat existing implementation, datasets, and recorded runs as the baseline: do not overwrite or relabel them. Start with smoke runs and invariant checks before experiment grids. Pair seeds, scenarios, attack schedules and initial weights across methods. Hold out whole runs, fit/tune on training/validation only, and preserve a separate clean trust calibration set.

Compare persistence, local, offline central reference, flat FL and hierarchy fairly. State offline pooled/IID exceptions to operational locality. HFL with one edge aggregation is equivalent to flat averaging on identical client updates; verify that before discussing gains.

Report class support, macro-F1, queue errors in vehicles, region/type breakdowns and run-level uncertainty. Security metrics need explicit eligible-agent/round denominators, actual attacker fractions, undetected-case censoring, attacker influence, false quarantine and recovery endpoints.

Include clean non-IID disagreement, real physical incidents and network loss as controls. Ablate reputation weighting, channel rejection, calibration evidence and recovery so benefits are not attributed to the wrong mechanism.

Measure serialized communication, wall-clock compute and simulated time separately. Do not claim radio/network performance or byte savings from logical messages alone. Never reuse a paper's numbers as project measurements.

Save immutable run/config/data/model/calibration IDs and all artifacts needed to reproduce plots. Record negative and inconclusive results, threat boundaries and limits. Update STATUS only with measured evidence, and keep a saved replay for the final demonstration.
