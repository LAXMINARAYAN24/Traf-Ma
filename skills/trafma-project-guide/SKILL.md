---
name: trafma-project-guide
description: Resume or plan TRAF-MA work, reconcile its project sources, and maintain milestone, requirements or literature traceability. Use for this traffic project rather than unrelated planning.
---

# TRAF-MA project guide

Resolve the TRAF-MA root from the current workspace using `AGENTS.md` and `pyproject.toml`; do not assume an old absolute path. Verify `docs/IMPLEMENTATION_PLAN.md`, `docs/STATUS.md` and `docs/ARCHITECTURE_AS_BUILT.md` exist. Read them and `AGENTS.md` before selecting work. If unavailable, ask for the project location rather than applying this architecture to another project.

Follow the latest user request and the three-phase gate sequence. The current Phase 1 implementation has one intersection, four persistent RSUs (`rsu_N/E/S/W`) and configurable N vehicles; do not relabel them as camera IDs or use stale signal-agent requirements.

Treat the implementation and verified artifacts present at the start of the task as the baseline. Preserve existing behavior and outputs by default; keep new work additive and new experiment outputs separate. Change baseline behavior only when explicitly requested, and retain a reproducible baseline for comparison.

Resolve conflicts in this order: executable code and tests, generated manifests/logs, `docs/ARCHITECTURE_AS_BUILT.md`, `docs/STATUS.md`, then design plans/contracts and historical material. Read the actual relevant source and data before changing contracts. Keep inferred defaults separate from confirmed decisions and measured results. Do not reproduce unsupported numerical/novelty claims.

Work on the next incomplete milestone unless the user directs otherwise. Explain its working outcome, dependencies and acceptance evidence. Record progress in STATUS with verified artifact paths and the next concrete action; a written module is not a passed gate. New dependencies and technical choices should serve that milestone.

For Phase 2, inspect dataset manifests and at least a representative sample of the underlying agent logs before deciding model inputs. Manifests do not contain feature tensors. Mark a feature schema as proposed until its source, units, missingness, normalization and boundary behavior are verified. Preserve explicitly proposed decisions as unresolved; do not promote them silently.

Keep originals in Architecture/PPT/Report/References and the chat intact. Update maintained contracts together when scope changes. Preserve optional extensions as optional rather than silently making them prerequisites.
