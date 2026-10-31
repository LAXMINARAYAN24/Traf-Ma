# Data contracts and learning hand-off

## Authority and implementation status

This document records the current data contract plus clearly marked Phase 2 proposals. When sources disagree, follow executable code and tests, then generated run artifacts/manifests, then `docs/ARCHITECTURE_AS_BUILT.md` and `docs/STATUS.md`. The original architecture, plans and research materials are design history, not evidence that a feature exists.

Phase 1 simulation, agent logs and P2.1 labels are implemented. `trafma.features/1`, the two-head GRU, local/central training, flat and hierarchical FedAvg, synthetic IID control, and transient offline participation replay are implemented. Full v1, v2_dev and v2 feature audits pass with one shared feature-schema hash. Phase 2 evaluation artifacts and remaining limits are recorded in `docs/STATUS.md`.

## Current simulation and observation contract

- One signalized four-approach SUMO intersection uses approaches `N`, `E`, `S`, `W`.
- The four persistent roadside agents are `rsu_N`, `rsu_E`, `rsu_S`, `rsu_W`. Vehicle IDs are created from SUMO lifecycle events and retain their originating parent RSU until arrival. “Camera” in planning diagrams means the roadside sensing role; do not rename the implemented RSUs.
- One SUMO adapter owns simulator access. `sensors.py` is the observation boundary. Agent/model input must be derived from that agent's own log plus static, versioned configuration; do not read another agent's private log or evaluator truth.
- An RSU senses its own incoming approach within its configured 150 m zone, its own outgoing zone, and its approach signal. Its own fields are not copies of child vehicle reports.
- A vehicle always has its own speed, acceleration, lane position and waiting time. While on its incoming edge, its neighbor fields describe same-edge vehicles within the configured 100 m radius. The vehicle's `scope` is `incoming_local`, `in_junction` or `out_of_coverage`; only `incoming_local` observations are eligible for current dataset windows.
- With `agents.signal_broadcast: false` (default), vehicles see approach signal state only within 100 m of the stop line and do not receive signal time-to-change. With broadcast enabled, signal state and time-to-change are provided on the incoming approach only. Broadcast changes observations, not the simulated traffic.
- The current topology supports one lane per direction. Do not silently generalize visible-density/capacity formulas to multiple lanes.
- Unknown values are `None` and named in `unavailable`; unknown is not zero. A known empty zone has zero count/density/queue, while its mean speed, acceleration, distance and waiting time are unavailable.
- There are no current coverage-bin fields in the message envelope.

## Current message and run artifacts

Current message schema is 1.1; schema 1.0 artifacts remain readable. The envelope is `schema`, `version`, `msg_id`, `msg_type`, `origin_id`, `sender_id`, `receiver_id`, `seq`, `sim_time`, `hops`, `evidence`, `payload`, `reserved`. Provenance is carried by `hops`; run identity is in the manifest, not a current message field. Current message types are `vehicle_report`, `rsu_summary` and `relay_bundle`. Prediction/model-update/group-update/trust/alert messages are future work, not implemented fields.

Run folders contain agent-local logs, message/transport traces, simulator traces, global snapshots, evaluator truth/events and manifests. Dataset runs additionally contain `labels.jsonl` and `windows.jsonl`; those windows contain client/run/time and target labels, not feature tensors. Dataset manifests contain run/client summaries and integrity checks, not model input tensors. Build model features from permitted agent logs and join the future labels through the offline dataset pipeline.

## Current label and window contract

- Target region: the client's own 150 m incoming approach zone.
- Target time: `end_t + labels.horizon_s`, currently 30 seconds after the final observation in the input history.
- Queue target: `zone_queue_m`, in metres. It is the contiguous chain of halted vehicles (speed at or below 0.1 m/s), beginning within 15 m of the stop line and extending while consecutive bumper gaps are at most 10 m. Its value reaches the rear bumper of the last vehicle and is capped at the zone length.
- `halting_count` and `queue_vehicle_count` are separate count measurements; neither is the queue-length target.
- Congestion class is derived from `zone_queue_m / zone_m`: class 0 free below 0.10; class 1 moderate from 0.10 to below 0.35; class 2 heavy at or above 0.35. These are current configured thresholds, not the older 0.2/0.5 proposal.
- Input windows use 10 consecutive 1-second records from one client, run and approach. RSU windows require consecutive RSU records; vehicle windows require all records to have `scope = incoming_local`. Windows never cross runs. Splits are assigned by whole run.
- Current minimum eligibility is 16 valid windows per client, as used in dataset manifests.
- Labels are produced offline from evaluator truth and are not present in agent input records. Live inference must not access future labels or evaluator truth.

## Generated dataset evidence (checked 2026-10-31)

The following artifacts are present locally. `runs/` is generated data and may be git-ignored; do not infer that it is absent merely because it is not versioned.

| Dataset | Runs | Run-folder bytes | Integrity in manifest |
|---|---:|---:|---|
| `phase2_v1` | 24 | 734,585,503 | 0 split overlaps, 0 window/horizon violations, no evaluator keys in agent logs, 0 collisions/teleports |
| `phase2_v2_dev` | 48 | 1,419,515,781 | 0 split overlaps, 0 window/horizon violations, no evaluator keys in agent logs, 0 collisions/teleports |
| `phase2_v2` | 160 | 4,778,347,553 | 0 split overlaps, 0 window/horizon violations, no evaluator keys in agent logs, 0 collisions/teleports |

These are manifest integrity results, not model-performance results. The full broadcast ablation has a config, but no generated `phase2_v2_bcast` dataset was present in the checked workspace. Many demand runs end at the configured 900-second horizon with unfinished vehicles; `sim_timeout` in their run manifest describes the run horizon, not a completed vehicle population. Check each run's duration, completion counts and integrity before selecting it for an experiment.

The v2 manifests contain 3 train / 1 validation / 2 test seeds in dev and 10 / 5 / 5 in full v2, per scenario. Each dataset manifest contains client/window counts and label histograms; the separate raw-log feature audit also passes for all three datasets.

## Phase 2 development input schema: `trafma.features/1`

The adapter emits `[10,22]` per window: seven scalar values, three signal one-hot values, ten availability-mask values and two agent-type values. The sources and limitations below are implemented for the development dataset; see `docs/PHASE2_IMPLEMENTATION.md` for exact units and normalizers. This schema is not final until the v1 and full-v2 scans also pass.

| Index | Feature value | RSU source | Vehicle source and caveat |
|---:|---|---|---|
| 1 | Local mean speed (m/s) | `mean_speed_mps`; unavailable for an empty zone | `(speed_mps + neighbor_count * neighbor_mean_speed_mps) / (neighbor_count + 1)` when neighbors exist; otherwise own `speed_mps`. This is a local mean, not an approach-wide mean. |
| 2 | Local mean acceleration | `mean_accel_mps2`; null in an empty zone | Own `accel_mps2` only; label as a vehicle proxy, not a neighborhood mean. |
| 3 | Local mean distance to stop line | `mean_dist_to_stop_m`; null in an empty zone | Own `dist_to_stop_line_m` only; a vehicle-position proxy, not a neighborhood mean. |
| 4 | Local density ratio | `vehicle_count / capacity_veh_in_zone`; current manifest records the nominal one-lane 150 m capacity | `(neighbor_count + 1) / visible_segment_capacity`; calculate the clipped segment from static edge length, own `lane_pos_m` and radius, and current one-lane geometry. Verify boundary cases and the denominator in tests. |
| 5 | Local halted-count ratio | `halting_count / capacity_veh_in_zone` | `(neighbor_halting_count + own_halted_indicator) / visible_segment_capacity`, where own halted is derived from own speed and configured threshold. This is a local halted-count ratio, not the contiguous queue target. |
| 6 | Local mean waiting time (s) | `mean_waiting_time_s`; unavailable for an empty zone | Own `waiting_time_s` only; a vehicle proxy. |
| 7 | Time to next displayed signal-state change (s) | `signal_remaining_s` | `signal_remaining_s` only when broadcast is enabled; otherwise unavailable. |
| 8–10 | Signal state one-hot | `signal_state` for this approach | `signal_state` when present; vehicle availability is coverage-dependent. Normalize the current `r`/`y`/`G` state representation explicitly. |
| 11–20 | Masks for values 1–10 | Set from source availability, not from normalized value | Same; repeat signal availability consistently for the three one-hot channels. |
| 21–22 | Client type one-hot | RSU `[0,1]` | Vehicle `[1,0]` |

The nominal capacity in the manifests is computed as zone length divided by vehicle length plus minimum gap. Apply the same geometry to the vehicle's clipped visible segment; do not use a fixed 200 m denominator at approach boundaries. Validate the capacity value against the resolved run configuration rather than hard-coding it.

The v1, v2_dev and full-v2 feature-audit artifacts are saved under each dataset directory. All pass with no feature errors, evaluator-label keys, mixed broadcast settings or split overlap. Detailed missingness is per dataset; the v2_dev measurements are not assumed to be population estimates.

For model preprocessing, represent missing normalized values as zero only alongside mask 0; a measured zero has mask 1. The schema uses fixed normalizers defined in `trafma/features.py`; no scaler is fit on validation or test data.

Full RSU-log audit: all `rsu_*.jsonl` records in generated v1, v2_dev and v2 datasets contain the acceleration and distance field keys. Non-null values occur on 68,326/86,400 (79.1%), 139,073/172,800 (80.5%) and 458,725/576,000 (79.6%) records respectively; null values correspond to empty zones. The older claim that v1 records lack these values is stale for the artifacts currently present. Use actual field presence, `None` and `unavailable` per record; never infer feature availability solely from dataset ID or message schema version.

### Feature audit status and remaining checks

1. Full v1, v2_dev and v2 scans verify the 22-value adapter over every RSU and vehicle record; absent/null fields remain masked unavailable.
2. Tests verify self-weighting in vehicle mean speed, zero-neighbor behavior, incoming-local scope, signal masks and clipped visible capacity.
3. RSU aggregates and vehicle-local proxies are distinguished in code and docs.
4. The window adapter enforces run/client/time continuity and future-label alignment; the feature audit checks evaluator-key exclusion and disjoint run splits.
5. The versioned feature order, units, fixed normalizers, type bits and schema hash are saved in audit and experiment artifacts.
6. All available v1, v2_dev and full v2 feature scans pass with schema hash `46196acbcc933a9b60f0a3151cf3a5822350c6988b7894a50f5722ac80f4abe2`.

## Remaining Phase 2 proposals

- Predict both future queue length in metres and the corresponding three-class congestion label as an explicitly multi-task experiment. The class target is derived from the same queue target; report that dependence.
- Persistence and a pooled signal-aware Ridge non-neural reference are implemented and measured on v2_dev. The Ridge model is offline and pooled; it is not a federated result. Vehicle persistence remains unavailable rather than substituting evaluator truth.
- Local training, pooled-central reference, flat FedAvg and hierarchical FedAvg are implemented. The user approved four persistent RSU learners (`rsu_N/E/S/W`), each trained only on its own approach's sensor windows pooled over training runs; each RSU learner update is counted once alongside vehicle child updates. The hierarchical smoke caps each RSU at 2,000 deterministic training windows.
- Updates carry client/type, round and base-model versions/hash, feature-schema hash, tensor weights/signature, distinct-window sample count and allowlisted scalar training metadata. The validator rejects stale/wrong-round/incompatible/nonfinite/duplicate/late/zero-mass updates with reasons. Raw histories, features, per-example labels and evaluator truth are not update fields.
- Flat and hierarchical aggregators preserve sample mass, reject duplicate self/child contributions, handle empty groups without zero models, and match on identical updates in hand-computed fixtures. Synthetic IID pooling is explicitly evaluator-only; transient participation runs are deterministic offline replay, not live connections.

The live online participation demonstration, full-v2 model benchmark and final multi-seed baseline freeze remain open. Do not introduce trust/security fields as an implicit Phase 2 requirement.
