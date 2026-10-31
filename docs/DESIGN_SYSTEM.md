# Dashboard design system

Implementation starts in P3.5. Phase 1 uses SUMO GUI and a simple run summary.

## Purpose and layout

A desktop-first research dashboard helps a viewer answer: what is happening at each approach, what is predicted, which agents contributed, and why influence changed.

- Header: run ID, seed, simulation time, live/replay state, connection and model version.
- Main row: four-approach intersection view and observed/current fused/future traffic panels.
- Agent table: agent ID/type/parent, trust, lifecycle status, last evidence, eligibility and last-seen time.
- Learning panel: FL round, eligible/skipped clients, group mass, loss/metrics and measured latency.
- Event/history panel: separate agent anomalies, traffic incidents, quarantine/recovery and connectivity events.

Preserve north/south/east/west labels and units. Show horizon and target time beside every prediction. Show coverage/freshness beside fused state. Unknown is “Unavailable”; stale data is marked with its age; neither becomes zero or green.

## Visual tokens

Proposed CSS tokens: background `#F5F7FA`; surface `#FFFFFF`; text `#172033`; muted text `#526174`; border `#CBD5E1`; primary `#1D4ED8`; success `#166534`; warning `#92400E`; danger `#B91C1C`; recovering `#6D28D9`.

Use system sans-serif, 16 px body, 14 px dense tables, 24 px panel/page title where appropriate. Spacing scale 4/8/12/16/24/32 px; 8 px card radius; minimal shadows. Test rendered text contrast rather than assuming every foreground/background combination is accessible.

## Status and interaction

Every trust status has text plus icon/shape; color is supplementary. TRUSTED=green, MONITOR=blue, SUSPICIOUS=amber, QUARANTINED=red, RECOVERING=purple. Agent anomaly and traffic incident use separate labels and filters.

Keyboard-accessible controls, visible focus, labelled inputs, reduced-motion support, and textual/table alternatives for charts. Agent details reveal factor scores, availability, reason and relevant time/region. Do not expose algorithm notation in the default view; keep it in an optional evidence/details panel.

Responsive layout: wide screens show map and metrics side by side; narrow screens stack panels and make tables scroll without clipping key status. Loading, empty, disconnected, stale, failed and partial-coverage states are explicit.

## Data and control behavior

Poll an initial snapshot, then subscribe to versioned events. Reconnection requests a fresh snapshot and deduplicates event IDs. Cap chart buffers; replay uses a saved run and visibly says “Replay.” UI refresh begins at 1 Hz independent of FL rounds. The browser cannot be the only copy of state.

Core controls: choose saved configuration/run, start/stop local simulation, pause/replay, select agent, filter events, export metrics. Attack controls are clearly labelled “Simulation experiment” and show the active schedule. Local notifications stay in the event panel; no external messaging service is required.
