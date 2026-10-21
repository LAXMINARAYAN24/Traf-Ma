# Architecture notes

## Boundaries (what each layer may see)
```
SUMO/TraCI --> world.TraciWorld --> WorldSnapshot (ground truth: evaluator + sensors only)
                                         |
                    sensors.rsu_view / sensors.vehicle_view   <- the ONLY observation boundary
                                         |
              agents (VehicleAgent, RsuAgent) -> Envelope -> Channel(v2i) -> RSU -> Channel(i2c) -> Coordinator
evaluator.world_truth(WorldSnapshot) -> evaluator/truth.jsonl -> labels (offline, dataset.py)
```
Agents never import `world`, `evaluator` or the snapshot. `tests/test_sensors.py` holds the scope fixtures
(RSU cannot see another road; a vehicle cannot read the world; out-of-coverage is masked).

## Message envelope (`trafma.msg` v1.0)
`msg_id = origin#seq`, `msg_type` (`vehicle_report` | `rsu_summary` | `relay_bundle`), `origin_id`, `sender_id`,
`receiver_id`, `seq`, `sim_time`, `hops`, `evidence` (`relayed_child` | `direct` | `relay`), `payload`, and
`reserved: {trust: "unavailable", signature: "unavailable"}` (Phase 3 fills these).
Invariant enforced by the validator: a payload field is `None` **iff** it is listed in `payload.unavailable`.
Rejections: `invalid`, `duplicate`, `stale` (older than `comms.stale_after_s`), `out_of_order`, `bad_provenance`, `not_my_child` (RSU).
The coordinator picks an RSU's slot from the sender's identity, not from payload claims (a test caught the opposite).

## Per-report fate accounting
Each vehicle report ends in exactly one of: `coordinator_accepted`, `lost_v2i`, `dropped_range`, `lost_i2c`,
`rsu_rejected:*`, `coord_rejected:*` (or is still in flight at the end). `manifest.reports_disposition` sums to
`vehicle_reports_sent`; duplicates created by the channel are rejected and never double-counted.

## Labels (P2.1)
`label(client, end_t) = evaluator zone queue / class of the client's own approach at end_t + 30 s`.
RSU windows: 10 consecutive report steps. Vehicle windows: 10 consecutive steps all in `incoming_local` scope.
Splits are by whole run (disjoint seeds); `dataset_manifest.json.integrity` records overlap, window-rule violations and
any evaluator key found in an agent log (all zero/empty in the shipped example).

## Hand-off to later phases
* **P2.3 local predictor:** read `agents/*.jsonl` + `runs/*/windows.jsonl`; the plan's 22-value input adapter is not built here (it is a model-side concern).
* **P3.1 attacks:** wrap `RsuAgent.summarise` / `VehicleAgent.report` (semantic falsification) and the outgoing-update path; the evaluator attack labels go in `evaluator/` so agent logs stay clean.
* **P3.2+ trust:** fill `reserved.trust`; `Coordinator.snapshot` already separates direct vs relayed evidence per approach.
