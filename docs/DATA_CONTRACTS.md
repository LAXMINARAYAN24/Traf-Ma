# Data contracts and learning hand-off

## Authority and implementation status

This document records the current data contract plus clearly marked Phase 2 proposals. When sources disagree, follow executable code and tests, then generated run artifacts/manifests, then `docs/ARCHITECTURE_AS_BUILT.md` and `docs/STATUS.md`. The original architecture, plans and research materials are design history, not evidence that a feature exists.

Phase 1 simulation, agent logs and P2.1 labels are implemented. The feature adapter, model, baselines and federation are not implemented. In particular, the proposed 22-value tensor below is a candidate to validate and freeze during P2.2/P2.3; it is not a current code or data contract.

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

The v2 manifests contain 3 train / 1 validation / 2 test seeds in dev and 10 / 5 / 5 in full v2, per scenario. See each dataset's manifest for actual client/window counts and label histograms. Do not claim the full dataset or feature audit is clean solely from aggregate label summaries.

## Phase 2 candidate input schema — not yet frozen or implemented

The previously proposed common tensor shape is `[batch, 10, 22]`: seven scalar values, three signal one-hot values, ten availability-mask values and two agent-type values. Candidate sources appear in the generated logs, with the noted limitations. Implement no adapter until source, units, masks, normalization and boundary behavior have focused tests.

| Index | Candidate value | RSU source | Vehicle source and caveat |
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

Observed sample audit: for the eight `phase2_v2_dev` seed-100 scenario runs, all 28,800 RSU timesteps had signal, signal-time, count, density and halted-count values; RSU mean speed/acceleration/distance were available on 23,184/28,800 records (80.5%), with the rest corresponding to empty-zone missing means. In a convenience sample of the first three sorted vehicle logs per run (24 files, 2,561 records), all own speed/acceleration/waiting values were present; among the 1,857 `incoming_local` records, distance and neighbor-count fields were present, signal state was available on a subset, and signal time was unavailable throughout because broadcast was off. This is a coverage sample, not a population-wide feature audit.

For model preprocessing, represent missing normalized values as zero only alongside mask 0; a measured zero has mask 1. Use fixed, versioned domain normalizers or fit learned scalers on training data only. Do not fit, tune or select features using test labels.

Full RSU-log audit: all `rsu_*.jsonl` records in generated v1, v2_dev and v2 datasets contain the acceleration and distance field keys. Non-null values occur on 68,326/86,400 (79.1%), 139,073/172,800 (80.5%) and 458,725/576,000 (79.6%) records respectively; null values correspond to empty zones. The older claim that v1 records lack these values is stale for the artifacts currently present. Use actual field presence, `None` and `unavailable` per record; never infer feature availability solely from dataset ID or message schema version.

### Feature audit acceptance before training

1. Verify every mapped field against actual records from every dataset generation; absent or null fields remain masked unavailable.
2. Verify vehicle formulas include self exactly once, handle zero neighbors, respect incoming-local scope and compute clipped visible capacity at road boundaries.
3. Keep RSU aggregate measurements and vehicle-local proxies explicitly distinguished in schema/docs.
4. Prove windows and features contain no evaluator keys, future labels or cross-agent reads; retain the manifest integrity check.
5. Freeze schema order/version, units, normalizers and capacity references in one implementation/config and save its hash in each model/run.
6. Only after these checks may the 22-value candidate become the implemented feature contract. If a candidate cannot be supported by permitted evidence, keep it unavailable or revise the schema before training.

## Phase 2 proposals (not implemented)

- Predict both future queue length in metres and the corresponding three-class congestion label as an explicitly multi-task experiment. The class target is derived from the same queue target; report that dependence.
- Compare persistence and signal-aware non-neural references before GRU/FL claims. A vehicle baseline may use only its local observations; do not substitute evaluator truth for its current input.
- Local training, offline pooled-central reference, flat FedAvg and hierarchical FedAvg are planned. The decision whether the four RSUs are persistent learning clients across runs remains proposed in `docs/STATUS.md`; choose and document client identity/partitioning before implementation.
- Any future model-update protocol must specify round/base-model and feature-schema versions, tensor signatures, finite-value validation, unique client IDs, distinct-window sample mass and rejected-update reasons. It must not carry raw histories, per-example labels or evaluator truth.
- Any future hierarchy must preserve group mass, avoid duplicate RSU self-updates/child sample counts, handle empty groups without zero models, and equal flat FedAvg on identical updates in a hand-computed fixture.

Do not treat these planned message types, model shapes, baselines or federation protocols as implemented until code and tests exist. Do not introduce trust/security fields as an implicit Phase 2 requirement.
