---
name: trafma-trust-security
description: Implement or review TRAF-MA's context-aware trust, controlled false-report or model-poisoning simulations, traffic fusion, quarantine and monitored recovery.
---

# TRAF-MA trust and security

Resolve the TRAF-MA root from the current workspace using `AGENTS.md` and `pyproject.toml`; do not assume an old absolute path. Read AGENTS.md, docs/STATUS.md, docs/TRUST_AND_SECURITY.md, docs/DATA_CONTRACTS.md and docs/EVALUATION.md. Request the project location if unavailable. Follow the Phase 3 prerequisite of a working clean simulation/FL baseline.

Keep the clean implementation and its verified artifacts as the baseline. Attack and trust behavior must be opt-in and must not alter clean baseline runs or overwrite their outputs.

Keep attack hooks isolated to the selected outgoing report/update. Evaluator truth, attack flags and future labels cannot enter detectors. A changed physical SUMO scene is an incident experiment, not a false-message attack. Preserve threat assumptions, including trusted routing/aggregation code.

Compare evidence only across matching region, coverage, time and metric semantics. Exclude self and relayed copies. Use frozen previous trust for peer references; missing evidence remains unavailable, not perfect or malicious. Non-IID update distance and network loss alone do not prove wrongdoing.

Keep evidence, reliability, EMA reputation, channel rejection and lifecycle state distinct. A weighted score can remain high under a single bad channel; implement the documented persistent-evidence guard rather than relying on an unreachable low-score threshold. Show which mechanism caused mitigation.

Quarantine means zero operational influence with bounded shadow evidence still allowed. Recovery must clear the offending channel, satisfy streak/trust guards and probation; silence cannot restore reputation. Keep identity/reputation stable through reconnection.

Fuse current state per coverage bin, forecasts per matching target/horizon. Never sum duplicate sensor estimates of the same queue. Report partial/unknown coverage and all-zero-weight fallbacks.

Verify honest disagreement, real incidents, empty peer sets, repeated poisoning with otherwise good scores, network loss, quarantine exclusion and recovery relapse. Log factor availability, weights and transition reasons; evaluate false quarantine as carefully as detection.
