# TRAF-MA — SUMO simulation layer

Complete SUMO side of TRAF-MA: a four-arm signalised intersection, N vehicle agents, **four RSUs (one per road)**,
a lossy/delayed V2I → I2C message path, an edge coordinator, full run logging, and labelled data generation.

```
vehicle agents (N)  --V2I-->  RSU-N / RSU-E / RSU-S / RSU-W  --I2C-->  edge coordinator  -->  global per-approach state
   (own view, 100 m)          (one per road, 150 m zone)               (validate, dedup, keep
                              senses its road + relays its children     direct vs relayed evidence apart)
```

This implements **Phase 1 (P1.1–P1.6)** of `TRAF_MA_imp_plan.pdf` and the SUMO-side **data generation of Phase 2 (P2.1)**.
Per your instruction, the plan's "camera agents" are **RSUs**: one per road, placed beside the road at its stop line.
It does *not* include model training / FedAvg / trust / attacks / dashboard (Phases 2.3+ and 3).

## Quick start (Windows PowerShell or Linux/macOS)

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # installs SUMO itself via the eclipse-sumo wheel
python -m trafma doctor           # P1.1: starts, steps and closes a minimal SUMO process
python -m trafma simulate --config configs/phase1.yaml --vehicles 20 --seed 42 --gui
python -m trafma simulate --vehicles 20 --seed 42             # headless
python -m trafma gate   --run runs/<run-id> --reproduce --pytest   # Phase 1 gate, machine-checked
python -m trafma report --run runs/<run-id>                   # layout.png + queues.png
python -m trafma replay --run runs/<run-id> --every 10        # replay from logs, no SUMO needed
python -m trafma collect --config configs/data.yaml           # P2.1 labelled dataset (24 runs, ~4 min)
python -m pytest                                              # 102 tests
```
`--gui` colours vehicles: **green** = inside its RSU zone, **red** = halted, **yellow** = approaching, **grey** = in junction / leaving.
RSUs show as blue squares, RSU sensing zones as cyan strips on each incoming arm.
Any config value can be overridden: `--set agents.rsu_zone_m=120 --set comms.v2i.loss_prob=0.1`.

## What was built, per plan milestone

| Milestone | Module | Evidence |
|---|---|---|
| P1.1 environment | `doctor.py`, `requirements.txt` | `trafma doctor` → READY; versions in every `manifest.json` |
| P1.2 intersection | `netgen.py` | 4 in + 4 out roads, 12 explicit connections, split-phase signal (25 s G / 3 s Y / 2 s all-red, 120 s cycle); no two approaches ever green together (test) |
| P1.3 reproducible traffic | `demand.py` | seeded; same seed ⇒ byte-identical routes **and** identical per-step trace hash; N, balanced/asymmetric/custom demand, turn mix; invalid N rejected |
| P1.4 agents + observation boundary | `agents.py`, `sensors.py` | 4 RSUs exist from t=0 to end; vehicles created on SUMO departure, retired on arrival; ownership = origin road, persists to arrival; out-of-coverage fields are `None` + listed in `unavailable` |
| P1.5 messages + coordinator | `messages.py`, `channel.py`, `coordinator.py` | typed envelope (schema/version/seq/time/hops/evidence); duplicate, stale, out-of-order, malformed, forged-provenance all rejected and counted; RSU-sensed vs relayed-child evidence never summed |
| P1.6 logs + demo | `runlog.py`, `runner.py`, `replay.py`, `gate.py` | manifest, per-agent logs, message trace, per-step trace, global snapshots, evaluator truth, console summary, replay, gate |
| P2.1 data + labels | `dataset.py`, `collect.py` | scenarios × seeds, 30 s-ahead labels, windows, coverage audit, label histograms, disjoint split manifest, measured non-IID skew |

## The RSUs (one per road)

* `rsu_N/E/S/W`, each beside its road at the stop line (8 m lateral offset), alive for the whole run.
* **Senses its own road only:** vehicles on its incoming edge within `agents.rsu_zone_m` (150 m) of the stop line, vehicles on its outgoing edge within 100 m, and the signal head of its road. Reports count, halted count, queue length (contiguous halted chain), mean speed, density, waiting times, signal state/time-to-change.
* **Relays its children:** every vehicle that departed on its road reports to it (V2I); the RSU validates, de-duplicates, appends itself to the hop path (`[veh, rsu_X]`), and forwards a bundle to the coordinator (I2C).
* A vehicle keeps its parent RSU **through the junction and onto the outgoing road until arrival**; once off its incoming road all neighbour/leader/signal fields are masked.

## Run artefacts (`runs/<run-id>/`)

`manifest.json` (config, seed, versions, file hashes, counts, per-report fate, status) · `summary.txt` · `scenario/` (net, routes, signal program, `scenario.sumocfg`, SUMO logs, tripinfo) ·
`agents/rsu_*.jsonl`, `agents/veh_*.jsonl` (agent-local views only) · `messages.jsonl` (send/deliver/reject trace) · `trace.jsonl` (deterministic per-step vehicle state; its SHA-256 is the reproducibility fingerprint) ·
`global_snapshots.jsonl`, `global_snapshot_final.json` · `evaluator/truth.jsonl`, `evaluator/events.jsonl` (ground truth, **never** in agent logs).
A finished example is in `examples/demo_n20_s42/` (incl. `gate.txt`, `report/*.png`).

## Defaults are editable defaults, not findings

300 m arms, 150 m RSU zone, 100 m vehicle radius, 1 s step, N=20, seed 42, 600 s, congestion rule `queue/zone < 0.10 free, < 0.35 moderate, else heavy`, horizon 30 s, history 10 steps, ≥16 windows to be trainable.
Straight/left/right all use one lane per arm (left/right turns share the lane).

## Deliberate limits and things to know

* **Sensors are noise-free.** Inside its zone an RSU reads exactly what the evaluator computes (the clean baseline shows RSU = truth). Falsification/poisoning belongs to Phase 3 hooks.
* **No separate signal agent.** The proposal's "traffic-signal agent" is folded into each RSU (it reports its road's signal state). Say if you want a fifth agent that owns the controller.
* **Vehicles inside the junction box** belong to no RSU zone (they are in `in_junction` scope and only report themselves).
* `agents.v2i_range_m` defaults to 400 m (covers the whole network), so no range drops happen unless you lower it.
* `heavy` (380 veh) and `asymmetric` (260 veh) in `configs/data.yaml` are intentionally over-saturated: many vehicles are still queued at 900 s and are reported as unfinished, never as "completed".
* GUI mode was verified to launch, run and close TraCI cleanly under a virtual display and the overlay renders; I could not click through it interactively.
* The first TraCI connect prints `Retrying in 1 seconds` — normal.

See `docs/ARCHITECTURE.md` for message schema, scope rules and the Phase 2/3 hand-off points.
