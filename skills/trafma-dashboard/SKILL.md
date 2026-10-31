---
name: trafma-dashboard
description: Build or review the TRAF-MA FastAPI and React dashboard for four-approach traffic, predictions, FL activity, trust, alerts and saved-run replay.
---

# TRAF-MA dashboard

Resolve the TRAF-MA root from the current workspace using `AGENTS.md` and `pyproject.toml`; do not assume an old absolute path. Read AGENTS.md, docs/STATUS.md, docs/DESIGN_SYSTEM.md, docs/ARCHITECTURE.md and state/event contracts in docs/DATA_CONTRACTS.md. Request the project location if unavailable.

Preserve the current simulation/backend behavior and recorded runs as the baseline. Implement dashboard work without changing baseline outputs; keep live and replay data clearly separated.

Use authoritative snapshot/events from the backend, not browser-invented traffic or trust. Keep simulation stepping outside the UI. Initial snapshot plus reconnect/resync and deduplicated events must work; bound chart buffers.

Present current observation, trusted fused state and future prediction distinctly. Show target time/horizon, units, approach, coverage and freshness. Missing/stale is never zero/normal. Label live versus replay and measured versus demonstration-only data.

Preserve the four-approach RSU hierarchy and origin IDs (`rsu_N/E/S/W`). Agent detail shows trust-factor availability, last evidence and transition reasons. Separate traffic incident, agent anomaly, quarantine, recovery and connection events. Local dashboard alerts do not require external messaging.

Follow visual tokens and accessible status text/icons; verify keyboard use, focus, readable contrast, responsive stacking and empty/disconnected/error states. Do not rely on colors alone or expose algorithm internals in the main user flow.

Verify clean-to-attack-to-recovery display against saved backend events, including reconnect and replay. Update STATUS with what was actually exercised; do not call a static mockup a working live dashboard.
