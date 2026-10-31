# Data contracts and learning specification

All defaults below are proposed implementation version 1, not measured optimal parameters. Serialize the chosen values, units, schema version and hashes in every run. Change a contract and its tests together.

## Spatial and temporal meaning

- Time = SUMO simulation seconds; sequence number is per sender. Wall-clock timestamps are separate diagnostic fields.
- Approach ID is north/south/east/west. The prediction target is the 150 m incoming zone before that approach's stop line. Four approach targets remain distinct in global state.
- Camera sees that approach's configured 150 m zone. Vehicle sees itself and visible vehicles within 100 m constrained to the same incoming approach; do not query unrelated edges and call that local sensing.
- Split each target zone into six 25 m bins for coverage matching. Only fully observable bins produce per-bin consensus statistics. Exact positions/trajectories stay local; upstream scope contains bin IDs and coverage, which is still potentially sensitive.
- Queue length means count of vehicles with speed below 0.1 m/s in the target zone, units vehicles. This is a stopped-vehicle count, not physical queue length in metres or a claim of a contiguous queue.
- Missing/unknown is represented explicitly. Empty zone with known visibility has queue=0; average speed/waiting is unavailable when no vehicles are observed.

## Local observation record

Fields: `run_id`, `agent_id`, `agent_type`, `parent_camera_id`, `sim_time_s`, `approach_id`, `visible_bin_ids`, local measurements, measurement availability, source sensor version. Private records may contain position/vehicle IDs needed for local processing. Those do not enter shared summary messages.

Camera records describe its own sensing, not a copy of vehicle reports. Vehicle records describe its own restricted view. An observation adapter returns only what that agent is permitted to see.

## Common feature schema v1: 22 values

The old diagram's 18 dimensions assumed eight scalar features, but one-hot signal state changes that size. Freeze the corrected schema below before training.

| Order | Value | Vehicle meaning | Camera meaning | Scale |
| --- | --- | --- | --- | --- |
| 1 | mean speed | Visible local traffic mean | Visible approach mean | Divide by configured speed limit |
| 2 | mean acceleration | Visible local traffic mean | Visible approach mean | Signed value divided by configured acceleration scale |
| 3 | mean distance to stop line | Mean for visible vehicles | Mean for visible vehicles | Divide by target-zone length |
| 4 | density ratio | Visible count / visible-zone capacity | Visible count / visible-zone capacity | Ratio |
| 5 | observed queue ratio | Stopped visible count / visible-zone capacity | Same for camera zone | Ratio |
| 6 | mean waiting time | Visible traffic waiting, consistently defined | Same for camera zone | Divide by configured waiting-time scale |
| 7 | remaining signal time | If signal timing broadcast is enabled | If available from controller | Divide by maximum cycle duration |
| 8–10 | red/yellow/green | State for relevant approach movement | Same movement | One-hot |
| 11–20 | availability mask | One bit per value above | Same | 0 or 1 |
| 21–22 | vehicle/camera | [1,0] | [0,1] | One-hot |

Use the through-movement signal as the initial approach reference; signal mapping must be explicit when turning phases differ. Disabled timing broadcasts make feature 7 unavailable. A missing value is filled with normalized zero and mask=0, whereas a measured zero has mask=1. If signal state is unavailable, all three signal values and masks are zero.

Capacity is a fixed geometric reference: sum over lanes of floor(observed lane length / (reference vehicle length + reference minimum gap)), with minimum denominator one. Save geometry and reference vehicle parameters. It is a normalization reference, not an absolute physical guarantee for every fleet mix. Validate raw physical values before clipping normalized features so impossible reports are not concealed. Fit any learned scaler on training data only.

The deterministic adapters preserve these semantics across types. They are not trainable private encoders initially. Future type-specific neural encoders would require an explicit compatible shared head and changed federation contract.

## Window and label

One sample = 10 consecutive 1 s observations from one client in one approach, ending at t. Input tensor shape `[batch, 10, 22]`. Never bridge missing observations, client identity, run boundaries or split boundaries; skip incomplete windows first, adding imputation only as a documented experiment.

Both heads predict the full **same approach zone at t+30 s**, regardless of input coverage:

- Queue target = stopped count in that zone at t+30 s, supervised by the isolated simulator oracle.
- Congestion target derives from future queue/capacity ratio: LOW below 0.2; MEDIUM from 0.2 to below 0.5; HIGH at least 0.5. These are operational labels, not universal traffic standards. Log class balance and retain thresholds throughout a comparison.

The learned task is forecasting a later state from partial history. Reporting the current simulator queue as a prediction is invalid. The two heads are related by design, so report that dependence rather than claiming independent tasks.

Train with cross-entropy plus MSE of queue/capacity; the queue head uses a nonnegative activation and converts back to vehicle counts for MAE/RMSE. Do not silently truncate forecasts at capacity; report extrapolation if geometry underestimates effective capacity. Empty target region has a known zero queue and LOW class.

Oracle labels are available for offline supervised datasets and only after horizon maturity in a live simulation. Inference takes input history only. Trust cannot look ahead or read evaluator attack labels.

## Partitioning

Assign entire simulation runs to train/validation/test by fixed seed lists before windowing. Suggested initial run split: seeds 100–109 train, 200–204 validation, 300–304 test, for each demand scenario. Training/validation/test vehicle IDs are run-qualified and disjoint. Persistent camera names are permitted across runs, but no observations/windows cross runs. Use separate seeds 400–404 for the trusted model-validation calibration set; this set is not the final test set.

Demand profiles: balanced light, balanced moderate, balanced heavy and asymmetric heavy north/south versus light east/west. Calibrate actual departure rates with SUMO; verify the resulting queue distributions rather than assuming the named regime occurred. Class imbalance is reported, not fixed by resampling the test set.

Natural non-IID clients differ by approach, residence time, agent type and coverage. For an IID control, redistribute training windows with a saved deterministic partition only as an offline experiment. Label it synthetic IID rather than physically local observations. Keep the held-out evaluation set fixed.

## Message envelope

All messages: `schema_version`, `run_id`, `message_id`, `message_type`, `sender_id`, `origin_id`, `parent_camera_id`, `sequence_no`, `sim_time_s`, `sent_time`, `approach_id`, `coverage_bin_ids`, `provenance_ids`. Preserve origin and original timestamp through relay. A relay is not an independent observation or extra vote.

| Payload | Required content | Forbidden ambiguity |
| --- | --- | --- |
| ObservationSummary | Per-visible-bin speed/queue/waiting summaries and availability; measurement time and coverage | No unlabelled mixing of current measurements and future predictions |
| Prediction | Input end time, target time, horizon, target approach, three class probabilities, queue estimate, model version | No averaging forecasts for different times/regions |
| ModelUpdate | Round ID, base-model hash, schema hash, tensor signature, weights or delta (one documented convention), distinct-window count and local metrics | No raw training histories; never interpret delta as full weights |
| GroupUpdate | Group model, effective mass, included client IDs/update IDs, round/base hash | No lost provenance or equal weighting of unequal groups |
| TrustEvidence | Factor scores, availability, context overlap, calibration-set version, prior/current trust and rejection reason | No attack truth or unavailable factors silently set to perfect/bad |
| GlobalTrafficState | Per-approach observed/fused current state and future forecast, freshness, coverage, contributing origins, confidence status | No single average queue across disjoint approaches |
| Alert | Event ID, event kind, agent/approach, evidence, severity, start/update/end time and lifecycle transition | “Unreliable” is not proof of malicious intent |

## State and protocol invariants

Deduplicate by run + origin + sequence/message ID. Apply a default 5 s semantic freshness limit; an older report becomes stale, not zero. Same-origin relays count once. Validate probability sum, finite values, type, units, coverage and version. Keep invalid-message diagnostics separate from statistical anomalies.

Training update IDs include round and origin. A client can belong to only one camera group in a round. Training sample count is verified against local replay manifest in experiments and bounded by declared dataset size in the simulator; do not assume adversarial sample counts are inherently trustworthy in deployment.

Raw logs, allowed upstream summaries, oracle truth and attack labels have separate stores and explicit readers. A transport payload audit is required at each phase gate.
