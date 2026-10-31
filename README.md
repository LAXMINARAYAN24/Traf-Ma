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
Phase 2 now includes audited features, baselines, GRU training, offline flat and hierarchical FedAvg, replay controls, and saved-artifact reporting. Full-v2 model training, paired multi-seed evaluation, broad local-client evaluation and live online federation remain open; trust, attacks and the full dashboard are Phase 3.

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
python -m trafma collect --config configs/data.yaml           # P2.1 labelled dataset v1 (24 runs, ~4 min)
python -m trafma collect --config configs/data_v2_dev.yaml    # P2.1 dataset v2, dev size (48 runs, est. ~1.3 GB)
python -m trafma collect --config configs/data_v2.yaml        # P2.1 dataset v2, full (160 runs, est. 4-5 GB; use --resume)
python -m trafma features-audit --dataset data/phase2_v2_dev  # audit agent-local features and split integrity
python -m trafma baseline --config configs/learning.yaml      # fit/evaluate clean non-neural baselines
python -m trafma train --method central --config configs/learning.yaml
python -m trafma train --method local --client asymmetric_ew_s100/rsu_E --config configs/learning.yaml
python -m trafma fedavg --config configs/learning.yaml        # flat natural client cohort
python -m trafma fedavg --iid-control --config configs/learning.yaml
python -m trafma fedavg --transient-participation --config configs/learning.yaml
python -m trafma fedavg --hierarchical --config configs/learning.yaml
python -m trafma phase2-report --help                         # report saved baseline/experiment runs
python -m trafma inspect --run runs/<run-id> --vehicle veh_0005   # one agent's local view + the path its reports took
python -m trafma inspect --run runs/<run-id> --rsu E              # one RSU's view, signal and detected vehicles
python -m pytest                                              # run all tests
```
`--gui` colours vehicles: **green** = inside its RSU zone, **red** = halted, **yellow** = approaching, **grey** = in junction / leaving.
RSUs show as blue squares, RSU sensing zones as cyan strips on each incoming arm.
Any config value can be overridden: `--set agents.rsu_zone_m=120 --set comms.v2i.loss_prob=0.1`.
`--set agents.signal_broadcast=true` makes each RSU broadcast its road's signal state and time-to-change to its vehicles (default off).
Pin: `requirements.txt` fixes SUMO/TraCI/sumolib to 1.28.0.

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
| P2 prep (schema 1.1) | `sensors.py`, `messages.py`, `configs/data_v2*.yaml` | RSU also reports `mean_accel_mps2` and `mean_dist_to_stop_m`; optional signal broadcast to vehicles; v2 dataset adds 2 demand levels, an E/W-heavy and a single-heavy-arm scenario; full grid has 10/5/5 seeds per scenario, dev grid 3/1/2 |
| P1.6 inspector | `inspector.py` | `trafma inspect` prints one agent's local view and the vehicle → RSU → coordinator path of its reports, from saved logs |
| P2.2-P2.7 development slice | `features.py`, `baseline.py`, `training.py`, `federation.py`, `federated_training.py`, `phase2_report.py` | Full v1/v2_dev/full-v2 feature audits; local/central GRU; flat and hierarchical FedAvg, IID and transient replay; saved Markdown/HTML report and plots. Full-v2 learning, paired seeds and online participation remain open. |

## The RSUs (one per road)

* `rsu_N/E/S/W`, each beside its road at the stop line (8 m lateral offset), alive for the whole run.
* **Senses its own road only:** vehicles on its incoming edge within `agents.rsu_zone_m` (150 m) of the stop line, vehicles on its outgoing edge within 100 m, and the signal head of its road. Reports count, halted count, queue length (contiguous halted chain), mean speed, mean acceleration, mean distance to the stop line, density, waiting times, signal state/time-to-change.
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
`agents.signal_broadcast` defaults to `false`: vehicles read the signal state only within 100 m and never see the timer (original behaviour).

## Deliberate limits and things to know

* **Message schema is 1.1** (adds the RSU fields above and `signal_remaining_s` on vehicle reports; the validator checks the major version only, so 1.0 logs stay readable). Sensor additions never change traffic: the trace hash is identical with and without them.
* **Sensors are noise-free.** Inside its zone an RSU reads exactly what the evaluator computes (the clean baseline shows RSU = truth). Falsification/poisoning belongs to Phase 3 hooks.
* **No separate signal agent.** The proposal's "traffic-signal agent" is folded into each RSU (it reports its road's signal state). Say if you want a fifth agent that owns the controller.
* **Vehicles inside the junction box** belong to no RSU zone (they are in `in_junction` scope and only report themselves).
* `agents.v2i_range_m` defaults to 400 m (covers the whole network), so no range drops happen unless you lower it.
* `heavy` (380 veh) and `asymmetric` (260 veh) in `configs/data.yaml` are intentionally over-saturated: many vehicles are still queued at 900 s and are reported as unfinished, never as "completed".
* GUI mode was verified to launch, run and close TraCI cleanly under a virtual display and the overlay renders; I could not click through it interactively.
* The first TraCI connect prints `Retrying in 1 seconds` — normal.

See `docs/ARCHITECTURE_AS_BUILT.md` for the message schema, scope rules, label contract and Phase 2 hand-off points, and `docs/STATUS.md` for status and decisions. `docs/ARCHITECTURE.md` is the original plan's architecture (it still says "camera" where this repo says RSU).
