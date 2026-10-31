# TRAF-MA as built (SUMO layer, message schema 1.1)

This describes what the code in `trafma/` does today. `ARCHITECTURE.md` in this folder is the original plan's
architecture (it says "camera"; this repo uses **RSU**, one per road). Where the two differ, this file is the
truth about the code and the differences are listed in section 8.

## 1. Components

| Component | Code | Owns | Must not do |
|---|---|---|---|
| SUMO adapter | `world.py` | start / step / close, one `WorldSnapshot` per step | give agents arbitrary world queries |
| Observation boundary | `sensors.py` | the ONLY functions turning world truth into agent-visible data | leak other approaches or evaluator truth |
| Vehicle agent | `agents.py` | its own report sequence; parent RSU fixed at departure | read the world, other vehicles' logs, the signal timer (unless broadcast is on) |
| RSU agent x4 | `agents.py` | its own road's sensing; validation, dedup and relay of its children | observe another approach |
| Signal | `netgen.py` | fixed program, split phasing 25 s G / 3 s Y / 2 s all-red per arm (120 s cycle) | it is an environment object, not an agent |
| Coordinator | `coordinator.py` | validate, dedup, direct vs relayed evidence, per-approach snapshot | sum RSU-sensed and vehicle-reported evidence |
| Evaluator | `evaluator.py` | ground truth and labels | be imported by agent code paths |
| Channel | `channel.py` | seeded loss / delay / duplication on V2I and I2C (default: perfect link) | |

## 2. Message envelope (`trafma.msg`, version 1.1)

`schema, version, msg_id, msg_type, origin_id, sender_id, receiver_id, seq, sim_time, hops, evidence, payload, reserved`

- `msg_id` must equal `<origin_id>#<seq>`. `hops` is the provenance path: `[veh]` at the vehicle, `[veh, rsu_X]` after the RSU.
- `evidence` is `direct` (RSU's own sensing), `relayed_child` (a vehicle report forwarded by its RSU) or `relay` (the bundle wrapper).
- `reserved` = `{"trust": "unavailable", "signature": "unavailable"}` until Phase 3.
- The validator checks only the **major** version, so version 1.0 logs and messages remain valid.
- Rejections are counted by reason: `invalid, duplicate, stale (older than comms.stale_after_s = 5 s), out_of_order, bad_provenance`.
- A value that is `None` must be listed in the payload's `unavailable` list, and vice versa. Unknown is never zero.

### `vehicle_report` payload
`scope, edge_id, speed_mps, accel_mps2, lane_pos_m, waiting_time_s, dist_to_stop_line_m, signal_state, leader_gap_m, leader_speed_mps, neighbor_count, neighbor_ahead_count, neighbor_behind_count, neighbor_halting_count, neighbor_mean_speed_mps, signal_remaining_s` (new in 1.1)

### `rsu_summary` payload
`approach, zone_m, coverage_m, vehicle_count, halting_count, queue_length_m, queue_vehicle_count, mean_speed_mps, mean_accel_mps2` (new), `mean_dist_to_stop_m` (new), `density_veh_per_km, mean_waiting_time_s, max_waiting_time_s, outbound_count, signal_state, signal_remaining_s, signal_phase_index`

`mean_accel_mps2` and `mean_dist_to_stop_m` are averages over the vehicles inside the RSU's 150 m zone (distance in metres to the stop line, not normalised). With an empty zone both are `None` and listed as unavailable, like `mean_speed_mps`.

## 3. Scope rules

- **RSU:** its own incoming road within `agents.rsu_zone_m` (150 m) of the stop line, its own outgoing road within `agents.rsu_outbound_zone_m` (100 m), and its own signal head including time-to-change.
- **Vehicle:** own kinematics always. While on its own incoming road: neighbours within `agents.vehicle_radius_m` (100 m), leader gap/speed, and the signal state when within 100 m of the stop line. In the junction or after it, every neighbour/signal field is `None` and listed as unavailable. Ownership of its parent RSU persists until arrival.
- **`agents.signal_broadcast`** (default `false`): when `true`, the vehicle's own RSU broadcasts its road's signal state and time-to-change to it anywhere on the incoming approach (so `signal_state` and `signal_remaining_s` are available beyond 100 m). It never extends coverage into the junction or beyond. It changes only what vehicles observe, never traffic: the trace hash is identical with the flag on or off.

## 4. Label contract (as built; supersedes the numbers in the plan's DATA_CONTRACTS)

- Target: the evaluator's queue for the client's own approach zone (150 m before the stop line) at `end_t + 30 s`.
- Queue = contiguous chain of halted vehicles (speed <= 0.1 m/s) that starts within 15 m of the stop line and continues while bumper gaps are <= 10 m, in **metres** to the rear bumper of the last vehicle, capped at the zone length. `halting_count` (all halted vehicles in the zone) is logged separately.
- Congestion class from `queue_m / zone_m`: `< 0.10` free (0), `< 0.35` moderate (1), otherwise heavy (2).
- Window = 10 consecutive 1 s observations. RSU windows need 10 consecutive records. Vehicle windows need all 10 to have `scope = incoming_local`. Windows never cross runs; splits are by whole run (disjoint seeds). A client is trainable with >= 16 windows.
- Labels exist only in `evaluator/` and `labels.jsonl` / `windows.jsonl`; the dataset integrity block verifies no evaluator key appears in any agent log.

## 5. Run artefacts

`runs/<id>/`: `manifest.json`, `summary.txt`, `scenario/`, `agents/rsu_*.jsonl`, `agents/veh_*.jsonl` (agent-local views only), `messages.jsonl` (transport trace; `message_trace: full` keeps every send, `exceptions` keeps only drops/rejections), `trace.jsonl` (deterministic per-step state; its SHA-256 is the reproducibility fingerprint), `global_snapshots.jsonl`, `evaluator/truth.jsonl`, `evaluator/events.jsonl`. Dataset runs additionally have `labels.jsonl` and `windows.jsonl` (window index and label only, **not** feature values).

Dataset folders: `data/<dataset_id>/dataset_manifest.json` and `dataset_report.txt` are committed; `data/<dataset_id>/runs/` is git-ignored and regenerated deterministically by `trafma collect`. `examples/demo_n20_s42/` is one stand-alone demonstration run (not part of any dataset).

## 6. Datasets

| Dataset | Config | Runs | Notes |
|---|---|---|---|
| `phase2_v1` | `configs/data.yaml` | 24 (4 scenarios x seeds 1-6) | original small dataset; stored RSU logs currently include acceleration/distance keys and non-null values for non-empty zones; do not infer raw-log field availability from dataset ID |
| `phase2_v2_dev` | `configs/data_v2_dev.yaml` | 48 (8 scenarios x 6 seeds) | development collection; locally audited run folders occupy about 1.42 GB |
| `phase2_v2` | `configs/data_v2.yaml` | 160 (8 scenarios x 20 seeds) | adds `light_moderate` (N=120), `moderate_heavy` (N=300), `asymmetric_ew`, `single_arm`; seeds 100-109 / 200-204 / 300-304; locally audited run folders occupy about 4.78 GB |
| `phase2_v2_bcast` | `configs/data_v2_broadcast.yaml` | 160 planned | same traffic as v2 with `signal_broadcast: true`, for the vehicle-signal ablation; config exists, generated dataset was absent in the 2026-10-31 audit |

## 7. Phase 2 feature adapter

`trafma.features/1` implements a deterministic 22-value timestep adapter and ten-step window reader. The adapter uses only each client's own log plus its run's static config/topology. Full `phase2_v1`, `phase2_v2_dev` and `phase2_v2` audits pass with one shared feature-schema hash.

The two-head GRU supports explicit single-client local training and a pooled offline central reference. Flat FedAvg trains selected saved-trajectory clients from a common global checkpoint. Hierarchical FedAvg uses four fixed groups: vehicle updates route to their original parent RSU, and the user-approved persistent `rsu_N/E/S/W` learners train separately on their own sensor windows pooled across training runs (never on child histories). The smoke configuration caps each RSU learner at 2,000 distinct training windows. All learning/federation is offline replay; the SUMO live loop is not yet performing online learning.

`trafma fedavg --iid-control` is a synthetic IID diagnostic over the same selected natural replay windows. `--transient-participation` deterministically resamples vehicle participation each round; those arrivals/departures are replay simulation, not live connections. Update payloads contain model tensors and allowlisted scalar metadata, not raw histories or labels. See `docs/PHASE2_IMPLEMENTATION.md` for evidence and limitations.

| Shared scalar | RSU source | Vehicle source |
|---|---|---|
| mean speed | `mean_speed_mps` | own speed combined with `neighbor_mean_speed_mps` weighted by `neighbor_count` |
| mean acceleration | `mean_accel_mps2`, null when the zone is empty | own `accel_mps2` only (proxy) |
| mean distance to stop line | `mean_dist_to_stop_m`, null when the zone is empty | own `dist_to_stop_line_m` only (proxy) |
| density ratio | `vehicle_count` / configured zone capacity | (`neighbor_count` + self) / capacity of the clipped visible segment |
| halted-count ratio | `halting_count` / configured zone capacity | (`neighbor_halting_count` + own halted indicator) / visible capacity; not a contiguous queue measurement |
| mean waiting time | `mean_waiting_time_s` | own `waiting_time_s` only (proxy) |
| remaining signal time | `signal_remaining_s` | `signal_remaining_s`, available only when `signal_broadcast` is on |
| signal R/Y/G | `signal_state` | `signal_state`, within 100 m unless `signal_broadcast` is on |

For vehicle mean speed include self exactly once: `(speed_mps + neighbor_count * neighbor_mean_speed_mps) / (neighbor_count + 1)` when neighbors exist, otherwise use own speed. The visible segment is clipped to the incoming edge using own lane position and configured radius. The adapter emits ten availability masks and a vehicle/RSU one-hot. Missing values are zero only with mask zero; a measured zero has mask one. Fixed normalization and ordered names are versioned in `trafma/features.py`.

Full log audits passed across 24 v1, 48 v2_dev and 160 full-v2 runs: 86,400 / 172,800 / 576,000 RSU records and 852,184 / 1,620,921 / 5,471,540 vehicle records respectively. All share feature schema SHA-256 `46196acbcc933a9b60f0a3151cf3a5822350c6988b7894a50f5722ac80f4abe2`; per-dataset availability and errors are saved in each `feature_audit.json`.

## 8. Differences from the plan documents

| Topic | Plan | Code |
|---|---|---|
| Camera | `cam_north` ... | RSU `rsu_N/E/S/W` |
| Queue | stopped-vehicle count in the zone | halted chain in metres (also `halting_count`) |
| Congestion thresholds | queue/capacity 0.2 / 0.5 | queue_m/zone_m 0.10 / 0.35 |
| Envelope | `run_id`, `parent_camera_id`, `coverage_bin_ids`, `sent_time` | `hops` carries provenance; run id is in the manifest, not the envelope; no coverage bins yet |
| Message types | + prediction, model update, group update, trust evidence, alert | only `vehicle_report`, `rsu_summary`, `relay_bundle` |
| Feature schema | 22-value proposal | `trafma.features/1`, GRU and offline federation implemented; audits pass on v1, v2_dev and v2 |
| Seeds | train 100-109, val 200-204, test 300-304 | v2 follows this; v1 used 1-3 / 4 / 5-6 |
