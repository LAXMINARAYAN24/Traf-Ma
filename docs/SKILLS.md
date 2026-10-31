# TRAF-MA reusable skills

Skills capture project-specific workflows; project documents own the evolving decisions. This avoids copying the whole architecture into every future request. Source copies are versionable under `skills/`; discoverable personal copies use `C:/Users/sahul/.codex/skills/`.

Installed on 2026-09-28: all seven skills below. Their source files passed the skill-creator validator. Fresh-chat automatic discovery has not been exercised in this planning session.

| Skill | Use when | Source |
| --- | --- | --- |
| trafma-project-guide | Resume work, choose next milestone, reconcile sources or update literature/requirements | [SKILL.md](../skills/trafma-project-guide/SKILL.md) |
| trafma-sumo-agents | Build/debug intersection, local sensing, N vehicles or camera routing | [SKILL.md](../skills/trafma-sumo-agents/SKILL.md) |
| trafma-traffic-prediction | Prepare agent datasets/features/labels or local forecasting models | [SKILL.md](../skills/trafma-traffic-prediction/SKILL.md) |
| trafma-hierarchical-fl | Implement/review local training rounds and two-stage model aggregation | [SKILL.md](../skills/trafma-hierarchical-fl/SKILL.md) |
| trafma-trust-security | Implement/review trust evidence, simulated attacks, fusion, quarantine or recovery | [SKILL.md](../skills/trafma-trust-security/SKILL.md) |
| trafma-dashboard | Build/review TRAF-MA live monitoring and replay UI | [SKILL.md](../skills/trafma-dashboard/SKILL.md) |
| trafma-evaluation | Run/review research experiments, security metrics and final demo evidence | [SKILL.md](../skills/trafma-evaluation/SKILL.md) |

Example request: `Use $trafma-sumo-agents to implement P1.2 after reading docs/STATUS.md.` Automatic selection is allowed. If a newly added skill is not shown in the current chat, explicitly read its source path or start a fresh chat in this workspace. Do not claim a fresh chat was tested merely because files were installed.

Only load the skill relevant to the current task. Planning skills do not authorize moving to a later implementation phase or publishing/sending anything externally. Keep source and installed copies synchronized when changing a skill; keep detailed parameters in the project documents rather than duplicating them in the skill.
