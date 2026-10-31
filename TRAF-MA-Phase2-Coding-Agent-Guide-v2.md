# TRAF-MA Phase 2 — Implementation Contract for the Coding Agent

**Revision:** 2.0  
**Purpose:** Give this document and the Phase 1 repository to the coding agent implementing Phase 2.  
**Primary objective:** Deliver a correct, reproducible local-prediction and hierarchical-federated-learning system with defensible experiments, metrics, plots and a working demonstration.

> **Source-of-truth rule:** Executable code and tests, then generated run artifacts/manifests, define current behavior, identifiers, message schemas, observation scope, labels and data. `docs/ARCHITECTURE_AS_BUILT.md` and `docs/STATUS.md` summarize that evidence. If any plan, contract or diagram conflicts, preserve current behavior and reconcile the stale document; do not make a proposal mandatory without approval. This guide is implementation guidance only where it agrees with those facts.

---

## 0. Mandatory first steps

Before writing code, the coding agent must:

1. Read this guide, `README.md`, `docs/ARCHITECTURE_AS_BUILT.md`, `docs/STATUS.md`, `docs/ARCHITECTURE.md`, all `configs/data_v2*.yaml`, `requirements-phase2.txt`, and the relevant implementation modules.
2. Read the maintained in-repository documents as design/evaluation guidance: `docs/IMPLEMENTATION_PLAN.md`, `docs/DATA_CONTRACTS.md`, `docs/DECISIONS.md`, `docs/EVALUATION.md`, `docs/PRD.md` and `docs/SOURCE_REVIEW.md`. They may contain stale decisions; compare claims with code and generated artifacts, then reconcile them rather than treating plans as current facts.
3. Run the existing environment check and tests before changes:
   ```bash
   python -m trafma doctor
   python -m pytest
   ```
   Record Python/SUMO/TraCI versions and the exact test result.
4. Produce a short file-level implementation plan, dependency plan, experiment plan and risk list. Do not begin by rewriting the simulator.
5. Implement in milestones. At the end of each milestone, report files changed, design choices, tests/results, artefacts produced and the next step.
6. Preserve Phase 1 behaviour and deterministic trace hashes for existing configs/seeds. Add regression tests for any intentional behaviour change.
7. Do not use test data to select features, thresholds, epochs, hyperparameters or checkpoints.
8. Never invent or hard-code results. Every number in the final report must be calculated from saved experiment artefacts.

## 1. Source reconciliation: decisions to preserve

The two documentation bundles contain planning decisions that do not exactly match the checked-in Phase 1 code. The following table is the controlling reconciliation for Phase 2.

| Topic | Some planning docs say | Current implementation says | Phase 2 instruction |
|---|---|---|---|
| Roadside agents | `cam_north`, etc. | `rsu_N`, `rsu_E`, `rsu_S`, `rsu_W` | Keep the existing RSU IDs and logs. Treat each RSU as the logical camera/roadside sensing client and edge group. Do not rename the working system. |
| Queue label | Count of stopped vehicles in a zone | Contiguous halted-vehicle chain length in **metres** | Keep the current evaluator label. All regression errors are in metres. Do not silently switch to vehicle counts. |
| Congestion labels | LOW/MEDIUM/HIGH at 0.2/0.5 | free/moderate/heavy at queue ratio 0.10/0.35 | Preserve current labels and thresholds: class 0 free `<0.10`; class 1 moderate `0.10–<0.35`; class 2 heavy `>=0.35`. |
| Current message envelope | Planned fields such as `run_id`, coverage bins and camera IDs | Schema 1.1 with existing fields and `hops` provenance; run ID in manifest | Extend compatibly. Do not replace the envelope or fabricate coverage-bin fields. |
| Agent naming | Camera and vehicle clients | Vehicle agents plus four RSU agents | Use actual IDs (`veh_*`, `rsu_*`) in code, datasets and plots. |
| Input schema | A proposed 22-value common representation | Feature adapter is not implemented; as-built logs define available fields | Implement the 22-value proposal only through a deterministic adapter from fields that really exist. Do not claim it already exists. |
| Phase boundary | Some old material places trust/dashboard work differently | Current implementation status: learning/FL not yet built; trust/security deferred | Phase 2 is clean prediction + FL + non-IID evaluation. Trust, attacks, consensus, quarantine/recovery and full FastAPI/React dashboard remain Phase 3. |

### 1.1 Existing implementation facts

The current code and `ARCHITECTURE_AS_BUILT.md` define:

- One four-arm signalised SUMO intersection.
- Four persistent RSUs, one per approach, and vehicle agents created/retired with SUMO vehicle lifecycle.
- Vehicle ownership is assigned from its origin/incoming road and remains fixed until arrival.
- `sensors.py` is the observation boundary. Agents must not query arbitrary SUMO/world state.
- Vehicles have their own local kinematics and bounded same-approach neighbour visibility. Signal fields are limited by range unless `agents.signal_broadcast` is enabled.
- RSUs sense their own configured road zone and relay child reports.
- Coordinator validation, deduplication, provenance and per-approach snapshots are already implemented.
- Evaluator truth is isolated in evaluator outputs and must not be imported into live agent/inference code.
- Message schema 1.1 is current; 1.0 logs remain readable.
- Missing values are `None` and listed in `unavailable`; unknown is never zero.
- Existing run artefacts include per-agent logs, message trace, simulator trace, snapshots, evaluator truth/events, manifests and summaries.

**Do not change these contracts to match a diagram.** If a required model feature cannot be derived from a client's allowed log, represent it as unavailable with a mask.

### 1.2 Current queue and label contract — freeze this

The target is the evaluator's queue for the client's own approach zone, 150 m before the stop line, at `end_t + 30 s`.

- Queue length is the contiguous chain of halted vehicles (speed `<= 0.1 m/s`) starting within 15 m of the stop line and continuing while bumper gaps are `<= 10 m`.
- The queue value is the distance in metres to the rear bumper of the last vehicle, capped at the zone length.
- `halting_count` is a separate count of all halted vehicles in the zone; it is not the queue-length label.
- Congestion class is derived from `queue_m / zone_m`: free `<0.10`, moderate `<0.35`, heavy otherwise.
- Input = 10 consecutive 1-second observations from the same agent, same run and same approach context.
- RSU windows require 10 consecutive RSU records.
- Vehicle windows require all ten records to have `scope = incoming_local`.
- Windows never cross run boundaries. Dataset splits are by whole run / disjoint seeds.
- A client is eligible for local training with at least 16 valid windows.

The target is future queue length, not current queue and not stopped-vehicle count. Do not change the target or class thresholds after viewing test results.

---

## 2. Phase 2 scope and definition of done

The reconciled project plan has three phases:

- **Phase 1:** SUMO world, agents, observation boundaries, messages, coordinator, logs and labelled dataset generation. Completed according to the current status, with the real-desktop GUI check still open.
- **Phase 2:** Local prediction, baselines, flat FedAvg, hierarchical FedAvg, non-IID and transient-client participation experiments.
- **Phase 3:** Trust/security, attacks, trust-aware aggregation, traffic-state consensus, quarantine/recovery, full dashboard and final security experiments.

Phase 2 must deliver all of the following:

1. A feature audit and, after it passes, a deterministic adapter from existing RSU/vehicle logs to a versioned common tensor schema.
2. Audited development and final datasets, label/coverage reports and split integrity.
3. Meaningful non-neural baselines.
4. A common GRU predictor with queue-regression and congestion-classification heads.
5. Local-only training and a pooled offline central reference.
6. Flat sample-weighted FedAvg.
7. Two-level hierarchical FedAvg. Freeze RSU learner identity/partitioning before implementation; the proposed four-persistent-RSU-client design in STATUS is not yet a decision.
8. IID control and natural non-IID experiments, with client participation and transient-vehicle behaviour measured.
9. Reproducible evaluation with saved configs, checkpoints, predictions, metrics, round logs, plots and a human-readable report.
10. A clear live SUMO demonstration and a reproducible offline replay benchmark.
11. All Phase 1 tests still passing and new tests covering every learning/aggregation contract.

**Not Phase 2 deliverables:** trust scores, trust-weighted aggregation, poisoning/false-report attacks, quarantine, recovery, incident consensus, full FastAPI/React dashboard. Do not add empty or fake implementations of these.

---

## 3. Phase 2 milestones — implement in this order

The milestone names below follow the three-phase implementation plan. P2.1 is already complete as data/label generation; do not regenerate or redesign it casually.

### P2.1 — Existing dataset generation and labels (already implemented)

Current configs:

| Dataset | Config | Runs | Purpose |
|---|---|---:|---|
| `phase2_v1` | `configs/data.yaml` | 24 | Original small dataset; audit the stored records instead of inferring fields from the dataset name |
| `phase2_v2_dev` | `configs/data_v2_dev.yaml` | 48 | Development; 8 scenarios × 3 train / 1 validation / 2 test seeds |
| `phase2_v2` | `configs/data_v2.yaml` | 160 | Final benchmark; 8 scenarios × 10 train / 5 validation / 5 test seeds |
| `phase2_v2_bcast` | `configs/data_v2_broadcast.yaml` | 160 | Separate signal-broadcast ablation dataset |

The v2 scenarios include free, light_moderate, moderate, moderate_heavy, heavy, asymmetric N/S, asymmetric E/W, and single-arm demand. As audited on 2026-10-31, all 48 development runs and all 160 full-v2 runs are present locally; their run folders occupy 1,419,515,781 and 4,778,347,553 bytes respectively. Their manifests report no split overlap, window/horizon violations, evaluator keys in agent logs, collisions or teleports. Some high-demand runs reach the configured 900-second horizon with unfinished vehicles; do not misreport those as all vehicles arriving. The broadcast config exists, but its dataset has not been generated.

- Start with `phase2_v2_dev`.
- Use `--resume` where supported.
- Verify dataset manifest, report, run IDs, seed splits, label histograms, window counts and integrity block. Inspect raw per-agent logs separately for candidate feature availability; manifests do not contain feature tensors.
- Seeds 400–404 are reserved for the Phase 3 calibration set; do not consume them for Phase 2 model tuning.
- Do not mix broadcast and non-broadcast datasets in one experiment. Compare them as a paired ablation using matched scenarios/seeds.
- Existing windows have stride 1 and overlap heavily. Split by whole run, never randomly by window.

**P2.1 acceptance:** data integrity checks pass, future labels align at `end_t+30`, run-level split IDs are disjoint, and per-client/per-scenario label counts are available.

### P2.2 — Baselines

Implement and evaluate these methods before GRU/FL:

**B0 — Persistence.** Predict future queue using the latest available queue measurement. For RSUs, derive only from that RSU's own permitted local fields. For vehicles, the local view is partial; either implement a clearly defined local halted/queue proxy or mark it unavailable for that row. Report the coverage limitation. Do not substitute evaluator truth as the current observation.

**Signal-aware non-neural reference.** The current status explicitly warns that persistence is weak because the signal cycle drives queue build-up/discharge. Add a transparent signal-aware predictor using only signal state/time-to-change and local observed traffic features actually available to that client. No future signal schedule or evaluator values may be used unless they are explicitly public environment information available to the agent. This is a supplementary reference; document its formula and inputs.

**B1 — Independent local model.** Include the local-only learned model in the final baseline ladder. Each client trains only on its own eligible training windows. Report ineligible clients rather than inventing results. A stratified vehicle-client sample may be used for a computationally controlled ablation, but the main benchmark should state exactly which clients were included.

**B2 — Pooled offline central reference.** Train on the union of permitted training windows. This is a non-private offline reference, not a federated method and not guaranteed to be an upper bound. Use the same train/validation/test run split and comparable training budget.

**Diagnostic:** majority-class predictor and class-frequency table may be included to expose imbalance, but do not present majority class as a meaningful queue predictor.

**Required baseline metrics**

- Queue regression: MAE and RMSE in metres; optional R².
- Congestion: macro-F1, balanced accuracy, accuracy, per-class precision/recall/F1 and confusion matrix.
- Break down by client type (RSU/vehicle), approach, demand scenario and split.
- Always show sample counts and class support.

**Acceptance:** baseline results are reproducible and use the same held-out evaluation rows wherever comparison is claimed.

### P2.3 — Feature adapter and local GRU predictor

#### A. Candidate common 22-value representation (not implemented)

The reference design proposes a 22-value common input. It is not present in current code or stored window files. First validate its inputs using raw logs and focused tests. Keep it a candidate until feature order, units, masks, static geometry references and schema-version fallback are fixed:

| Indices | Feature | Definition |
|---|---|---|
| 1 | Local mean speed | RSU `mean_speed_mps`; vehicle `(speed_mps + neighbor_count * neighbor_mean_speed_mps) / (neighbor_count + 1)` when neighbors exist, otherwise own speed |
| 2 | Local mean acceleration | RSU `mean_accel_mps2` (schema 1.1); vehicle own `accel_mps2` proxy |
| 3 | Local mean distance to stop line | RSU `mean_dist_to_stop_m` (schema 1.1); vehicle own `dist_to_stop_line_m` proxy |
| 4 | Local density ratio | RSU `vehicle_count / capacity_veh_in_zone`; vehicle `(neighbor_count + 1) / visible_segment_capacity`, where `visible_length = max(0, min(in_len, lane_pos + radius) - max(0, lane_pos - radius))` and capacity is visible length divided by vehicle length plus minimum gap. Current topology has one lane. |
| 5 | Local halted-count ratio | RSU `halting_count / capacity_veh_in_zone`; vehicle `(neighbor_halting_count + own_halted_indicator) / visible_segment_capacity`. This is not the contiguous future queue target. Use the same clipped segment capacity as feature 4. |
| 6 | Mean waiting time | RSU `mean_waiting_time_s`; vehicle own `waiting_time_s` approximation |
| 7 | Remaining signal time | RSU `signal_remaining_s`; vehicle value only when actually available |
| 8–10 | Signal state R/Y/G | Fixed one-hot encoding of `signal_state` |
| 11–20 | Availability masks | One mask for each of the ten preceding values; `1` means measured/available, `0` means unavailable |
| 21–22 | Client type | Vehicle `[1,0]`; RSU `[0,1]` |

This candidate yields 22 values per timestep and a proposed model input `[batch, 10, 22]`.

The 2026-10-31 sample audit covered all RSU records in eight seed-100 `phase2_v2_dev` scenarios (28,800 rows) and the first three sorted vehicle logs per scenario (24 files, 2,561 rows). RSU mean speed/acceleration/distance were non-null on 80.5% of sampled rows; signal/timer and count/density fields were present throughout. In the vehicle sample, only 1,857 rows were `incoming_local`; signal state was available on 1,439 of those (77.5%), and signal time was unavailable because broadcast was disabled. Treat these as sample coverage, not population estimates. See `docs/DATA_CONTRACTS.md` and `docs/STATUS.md`.

**Critical feature rules**

- Freeze feature order, units, normalization constants and schema version in one config/module. Do not depend on dictionary ordering.
- Use only fields in the client's own log plus static run/config geometry. No cross-agent log reads.
- Preserve `None`/`unavailable` as missing. A normalized zero with mask 0 means unavailable; a measured zero with mask 1 means a real zero.
- For missing signal state, all R/Y/G values and their availability masks are zero. Do not encode missing as green.
- Vehicle signal fields can be unavailable outside the documented range when `signal_broadcast=false`. With broadcast enabled, signal availability expands only as defined in `ARCHITECTURE_AS_BUILT.md`.
- Vehicle acceleration, stop-line distance and waiting time are own-state proxies, not RSU-equivalent means. State this in the report.
- The stored v1, v2_dev and v2 RSU logs audited on 2026-10-31 contain acceleration/distance keys and non-null values for non-empty zones. Use actual record values and `unavailable`, not dataset ID or message schema, to determine feature masks.
- Capacity is geometric and must be computed for the clipped visible segment; do not use a fixed 200 m vehicle denominator at road boundaries. The current network is one lane per approach.
- Validate raw physical values before normalization/clipping. Save geometry/capacity reference values.
- If any historical record genuinely lacks these fields, mask them unavailable. Do not backfill from evaluator truth or other agents.
- Prefer fixed domain/config normalization constants. If a learned scaler is used, fit on train only and save it with the checkpoint.
- Never fill unavailable observations with an unmasked zero.
- Do not construct windows across runs, agents, approaches or split boundaries. Apply existing vehicle `incoming_local` eligibility.
- The target remains full approach queue at t+30, even though vehicle input coverage is partial. Make this prediction difficulty explicit.

**Potential schema conflict resolution:** Current target is the future contiguous halted chain in metres. Candidate halted-count ratio is a distinct local input, not a proxy for that target. Preserve the target and current 0.10/0.35 class thresholds.

#### B. GRU model

- Use a compact GRU encoder over ten timesteps and a shared representation.
- Two output heads:
  - nonnegative queue-length regression in metres;
  - three-class free/moderate/heavy classification.
- Use the current thresholds 0.10/0.35 for labels.
- Suggested objective: normalized queue regression loss + weighted cross-entropy. Choose weights on validation only and document them.
- If a multi-task model proves unstable, any alternative must be justified, versioned and compared fairly; do not silently remove a head.
- Same model architecture and tensor signature for vehicle and RSU clients. Client-type one-hot and availability masks represent heterogeneity.
- Start with small, declared hyperparameter settings. Use early stopping on validation only.
- Set Python, NumPy and PyTorch seeds; enable deterministic options where supported and record limitations.
- Save architecture, weights, feature schema, normalizer, class mapping, seed, optimizer/config, epoch history and dependency versions.
- Report parameter count, training time and inference latency.

**Training/evaluation rules**

- Local model: train only on that client's training runs/windows.
- Central model: pooled offline training on training windows only; clearly label non-private.
- Select checkpoints using validation only; evaluate test once after configuration freeze.
- Report RSU-only, vehicle-only and combined results.
- Include a tiny-data overfit sanity test to verify the model can learn; it is not a headline result.

**Acceptance:** model shape is correct, loss is finite, checkpoint reload reproduces predictions, no evaluator labels enter model inputs, and held-out results beat or are honestly compared against persistence.

### P2.4 — Flat FedAvg reference

Implement a standalone FL protocol; do not put federation logic inside the SUMO simulation loop.

- All eligible clients start a round from the same global model checkpoint/version.
- Each client trains locally on its own training partition.
- Update contains: client ID/type, round ID, base model version/hash, feature-schema hash, tensor names/shapes, model weights or delta (choose one convention and document), distinct sample count, and training metadata.
- `n_i` is distinct training windows used, not epochs × rows.
- No raw features, trajectories, per-example labels or raw client logs in updates.
- Validate round/version/hash, tensor signature, finite values, sample count, duplicate client submissions and deadline.
- Reject malformed updates with a reason. Rejection alone is not evidence of maliciousness.
- If accepted sample mass is zero, skip the round and retain the prior model.

FedAvg:

\[
\theta_{r+1}=\frac{\sum_i n_i\theta_i}{\sum_i n_i}
\]

Record eligible/accepted/rejected clients, sample mass, serialized bytes, local training time, aggregation time, round duration, model version and skipped rounds.

**Acceptance tests:** hand-computable tensor fixture; sample-weighting (not equal-client weighting); duplicate/late/wrong-version/wrong-shape/NaN/Inf/zero-client handling; raw-data payload audit.

### P2.5 — Hierarchical FedAvg (RSU edge groups + global coordinator)

This is the main architecture demonstration.

- Four fixed groups: `rsu_N`, `rsu_E`, `rsu_S`, `rsu_W`.
- Each vehicle replay client belongs to the RSU corresponding to its original approach, as recorded in the run manifest. It is not reassigned based on later movement.
- The current code has RSU sensor records and message aggregation only; it has no learning clients. Whether an RSU also trains as an independent model client, and whether its client persists across runs, is unresolved (STATUS D4).
- Do not include an RSU self-update until its training data/client identity is explicitly decided. If approved, keep its own sensor records independent and count its update exactly once alongside child updates.
- An RSU must not train on the pooled raw histories of its children. It aggregates model updates only.
- Do not count vehicle samples again as RSU samples.

For group \(c\):

\[
m_c=\sum_{i\in c}n_i,\qquad
\theta_c=\frac{\sum_{i\in c}n_i\theta_i}{m_c}
\]

At global coordinator:

\[
\theta_{r+1}=\frac{\sum_c m_c\theta_c}{\sum_c m_c}
\]

- Do not average four group models equally.
- Group mass is the sum of distinct client sample counts in that group.
- A group with no eligible clients is omitted, not represented by a zero model.
- On identical clients/updates, mass-preserving hierarchical FedAvg must equal flat FedAvg within floating-point tolerance. The initial hierarchy demonstrates the edge/global organization and message flow; do not claim grouping alone improves accuracy.
- One local epoch and one edge aggregation per global round is the initial simple protocol unless experiments justify a change.

**Transient vehicles**

- Offline benchmark: each saved vehicle trajectory is a replay client routed through its original RSU group. Clearly label it offline replay; it does not mean the vehicle remained connected at the intersection.
- Online demo: a vehicle needs 10 valid observations, a matured t+30 label and at least 16 valid training windows to train. Otherwise mark it inference-only / insufficient history.
- Departed clients do not send new live updates. New IDs do not inherit another vehicle's data.
- If all live vehicles are ineligible, show that fact; do not claim active vehicle federation just because RSUs train.

**Acceptance tests:** hierarchical result equals flat FedAvg on the same fixture; no duplicate RSU self-update; no overlapping group membership; no equal-group weighting; empty groups handled; group/global sample mass and provenance logged.

### P2.6 — Non-IID and transient participation experiments

Evaluate the real client heterogeneity rather than merely stating that data is non-IID.

Required comparisons:

- Local-only learning.
- Pooled offline central reference.
- Flat FedAvg with a deterministic synthetic IID training partition (diagnostic control only).
- Flat FedAvg on natural non-IID local data.
- Hierarchical FedAvg on natural non-IID local data.

Measure and report:

- Per-client sample counts and label distributions.
- Distribution divergence/skew across approaches, client types and demand scenarios. Choose and document a metric such as Jensen–Shannon divergence; do not report an unexplained “non-IID score”.
- Eligible, connected, participating, accepted, late and dropped clients per round.
- Client arrivals/departures and minimum-history eligibility.
- Per-round group mass and global mass.
- Held-out metrics by scenario, approach and client type.

The IID partition must be generated offline with a saved deterministic seed and must be labelled synthetic IID; it is not a physically local deployment scenario. Keep the held-out test set fixed across methods.

**Acceptance:** a replay benchmark and an online participation demonstration both exist; the report distinguishes them and does not imply that transient vehicles were continuously connected.

### P2.7 — Evaluation, experiment runner and baseline freeze

Build a reproducible experiment runner that saves an immutable run folder, for example:

```text
runs/phase2/<experiment_id>/
  manifest.json
  resolved_config.yaml
  summary.md
  metrics.json
  metrics.csv
  predictions.csv
  training_history.csv
  rounds.jsonl
  checkpoints/
  plots/
```

Manifest must include:

- dataset ID and manifest/hash;
- train/validation/test run IDs and split seeds;
- code commit/version if available;
- Python, OS, CPU/GPU, SUMO and package versions;
- all random seeds and deterministic settings;
- feature/model schema and normalization version;
- resolved hyperparameters;
- experiment start/end and wall-clock duration.

Required plots:

1. Label distribution by split, scenario, client type and class.
2. Feature missingness/availability by feature and client type.
3. Baseline vs local GRU vs central reference vs IID FedAvg vs non-IID FedAvg vs hierarchical FedAvg.
4. Queue MAE/RMSE and congestion macro-F1/balanced accuracy.
5. Confusion matrices for final classification models.
6. Training/validation loss curves.
7. Per-scenario, per-approach and per-client-type error breakdown.
8. Error against demand level and asymmetry pattern.
9. Per-client sample counts and client participation per round.
10. Communication bytes, local training time, edge aggregation time and global aggregation time.
11. Predicted-vs-true queue plots and representative time series.
12. Optional signal-broadcast ablation, using matched traffic scenarios/seeds and separate datasets.

All plots must be generated from saved result files, have units/legends/sample counts, and distinguish validation from test. No manually fabricated curves.

**Required result metrics**

| Area | Metrics |
|---|---|
| Queue forecast | MAE, RMSE in metres; optional R² |
| Congestion forecast | Macro-F1, balanced accuracy, accuracy, per-class precision/recall/F1, confusion matrix |
| Generalization | Results by scenario, approach, client type and demand/asymmetry |
| Non-IID | Client label/sample distributions and declared divergence measure |
| Participation | Eligible/accepted/late/dropped clients, samples per client, rounds completed/skipped |
| Communication | Serialized bytes per client/round, total bytes, edge vs global traffic |
| Runtime | Local train, edge aggregation, global aggregation, end-to-end round time, inference latency |
| Reproducibility | Seed, dataset hash, config hash, code/dependency versions, run IDs |

Use run-level aggregation/uncertainty where appropriate. Because windows overlap, do not treat every window as an independent simulation replicate. Report per-run or per-seed summaries and confidence intervals where feasible.

---

## 4. Live demonstration requirements

Phase 2 needs a convincing demonstration, but the full FastAPI/React dashboard belongs to Phase 3. Do not divert major time into a full frontend before the learning pipeline is correct.

A SUMO GUI + live terminal status panel is acceptable for Phase 2, provided the saved offline reports and plots are complete.

### Suggested demo sequence

1. Run a sufficiently congested scenario (the Phase 1 status recommends testing around `N=100`, seed 42 for a visually meaningful live demo; use the v2 dataset scenarios for benchmark claims).
2. Show the four existing RSUs and their independent approach observations.
3. Show a vehicle's local input view, including unavailable signal/neighbour fields where applicable.
4. Show the ten-step feature sequence and masks for a selected client. Do not display evaluator labels in the live input panel.
5. Show local predictions: queue metres and free/moderate/heavy probabilities or class. Only show probabilities if the model emits calibrated/meaningful probabilities; otherwise label them as raw model scores.
6. Show eligibility: trained / inference-only / insufficient history.
7. Show a round starting from a named global model version; show client updates, accepted sample counts, and rejected/late updates.
8. Show each RSU edge aggregation and its eligible child vehicle clients. Show an RSU self-update only if that client role is approved and implemented.
9. Show global aggregation and the new model version.
10. Show prediction timeline beside evaluator truth only in a clearly separate offline/evaluator comparison panel after labels mature.
11. End with the benchmark comparison and saved report plots.

### Live/offline honesty rules

- Offline replay clients are benchmark clients, not persistent live vehicles.
- If SUMO pauses while training runs, measure simulated time and wall-clock time separately. Do not claim real-time operation.
- A vehicle without enough matured local windows is inference-only.
- Evaluator truth is never available to live prediction. The simulator may release labels to client-local training only after the 30-second horizon matures; state that this is a simulation-supervision assumption, not an on-road deployment capability.
- The full dashboard, trust display, alerts and security lifecycle are Phase 3.

---

## 5. Recommended code organization

Adapt to the current repository's package style. This is a suggested organization, not a mandate to create empty folders.

```text
trafma/
  features/
    schema.py          # ordered 22-value schema, units, version, normalization
    adapter.py         # RSU/vehicle log -> feature vector + masks
    windows.py         # ten-step windows; run/client/split-safe
    audit.py           # availability and leakage audit
  baselines/
    predictors.py      # persistence, signal-aware and diagnostic baselines
    evaluate.py
  models/
    gru.py             # common two-head GRU
    losses.py
    train.py           # local and pooled-central training
    inference.py
  federated/
    protocol.py        # round/update contracts and validation
    client.py          # client-local training
    aggregation.py     # flat FedAvg
    hierarchy.py       # RSU edge + global mass-preserving aggregation
    checkpoint.py
  evaluation/
    metrics.py
    experiments.py
    plots.py
    report.py
```

- Keep `evaluator.py` and evaluator output isolated from live agents/features/inference.
- Offline supervised training may read a separate labelled dataset interface that joins labels to input windows after feature construction. Do not import evaluator internals into live agent code.
- Keep Phase 2 extras in `requirements-phase2.txt`; Phase 1 must remain runnable without PyTorch.
- No external broker/database is required for the Phase 2 benchmark.

---

## 6. Configuration and CLI expectations

All parameters that affect results must be in versioned YAML/config, not hidden constants:

- dataset ID, split, run IDs and dataset hash;
- feature schema/version, ordered fields, units, normalization and capacity references;
- sequence length 10 and forecast horizon 30 seconds;
- current queue definition and congestion thresholds 0.10/0.35;
- GRU hidden size/layers/dropout, optimizer, learning rate, batch size, epochs, early stopping and loss weights;
- local epochs, client eligibility, round interval/deadline and aggregation convention;
- IID partition seed and non-IID evaluation settings;
- random seeds/deterministic options;
- output/run ID and plot/report settings.

Suggested commands below describe the desired workflow. Reuse the current CLI conventions and document the actual implemented command names; do not claim these commands exist until implemented.

```bash
# Existing Phase 1
python -m trafma doctor
python -m pytest

# Existing dataset collection
python -m trafma collect --config configs/data_v2_dev.yaml --resume

# Proposed Phase 2 workflow (implement equivalents)
python -m trafma features-audit --dataset phase2_v2_dev
python -m trafma baseline --config configs/learning.yaml --dataset phase2_v2_dev
python -m trafma train --config configs/learning.yaml --method local --dataset phase2_v2_dev
python -m trafma train --config configs/learning.yaml --method central --dataset phase2_v2_dev
python -m trafma federate --config configs/learning.yaml --method fedavg --dataset phase2_v2_dev
python -m trafma federate --config configs/learning.yaml --method hierarchical_fedavg --dataset phase2_v2_dev
python -m trafma evaluate --run <run-id>
python -m trafma report --run <run-id>
```

---

## 7. Testing checklist

### Phase 1 regression

- Existing test suite still passes.
- Existing deterministic trace hash remains unchanged for existing configs/seeds.
- Existing schema 1.0/1.1 logs remain readable.
- Agent logs contain no evaluator-only fields.
- Observation boundaries and vehicle ownership remain unchanged.

### Feature/data tests

- Every feature maps to the correct current log field.
- Vehicle and RSU approximations are explicitly distinct.
- `None` and `unavailable` masks are consistent.
- Feature order and normalization are fixed.
- R/Y/G one-hot and missing signal behaviour are correct.
- Ten-step windows do not cross agent/run/approach/split boundaries.
- Vehicle windows require ten `incoming_local` records.
- Train/validation/test run IDs are disjoint.
- No evaluator truth/label enters the feature tensor.
- Broadcast and non-broadcast datasets are never silently mixed.

### Model tests

- Input shape `[batch, 10, 22]` and output heads are correct.
- Loss is finite and gradients flow.
- Queue predictions are nonnegative and measured in metres.
- Congestion class mapping uses 0.10/0.35 thresholds.
- Checkpoint reload reproduces predictions.
- Fixed-seed training is reproducible within documented tolerance.
- Train/validation/test are kept separate.

### Federation tests

- Hand-calculated FedAvg correctness.
- Sample weighting uses distinct windows.
- Hierarchical aggregation equals flat FedAvg for identical updates.
- RSU self-update counted exactly once if the approved client design includes RSU learners.
- Group membership disjoint; no double counting.
- Empty groups/rounds retain previous model or omit group as specified.
- Duplicate, late, malformed, wrong-version, wrong-shape, NaN and Inf updates are rejected.
- No raw local data appears in model update payloads.

### Evaluation/demo tests

- Metrics are computed from saved predictions.
- Plots generate headlessly from saved artefacts.
- Every report includes dataset/config/split/seed identity.
- Demo handles no eligible vehicle clients honestly.
- Live simulation time and wall-clock time are distinguished.
- Offline replay is clearly labelled.
- Report and plots can be regenerated from a run folder.

---

## 8. Final Phase 2 deliverables

Do not mark Phase 2 complete until the team has:

1. Updated source code and tests.
2. `docs/PHASE2_IMPLEMENTATION.md` with actual architecture, data flow, assumptions and commands.
3. Updated `docs/STATUS.md` separating verified, proposed and not implemented items.
4. Development dataset audit and final full-dataset audit.
5. Baseline results and documented signal-aware reference.
6. Local GRU and pooled-central reference results.
7. Flat FedAvg and hierarchical FedAvg results.
8. IID vs natural non-IID comparison.
9. Transient-client participation/replay-vs-live demonstration.
10. Final comparison table with per-scenario, per-approach and per-client-type breakdowns.
11. Saved checkpoints, resolved configs, manifests, predictions and round logs.
12. Plots and a generated HTML/Markdown report.
13. A repeatable demo script and setup instructions.
14. Limitations: one simulated intersection, logical edge roles in one Python process, synthetic SUMO traffic, approximate vehicle features, replay clients, training pauses and no formal privacy guarantee.

### Required final report questions

The report must answer with saved evidence:

1. What is predicted, from which agent-local observations, and at what horizon?
2. How do RSU and vehicle observations differ? How is missingness represented?
3. How do persistence and signal-aware baselines perform?
4. Does the GRU improve over baselines on held-out runs?
5. How does local-only compare with pooled central training?
6. How does flat FedAvg compare with local-only and central?
7. What changes between IID control and natural non-IID data?
8. Does hierarchical FedAvg reproduce flat FedAvg mathematically on matched updates?
9. What are communication and wall-clock costs?
10. Which scenarios, approaches and client types are hardest?
11. What happens when vehicles arrive, depart or fail the minimum-history rule?
12. What is live versus offline replay?
13. What remains for Phase 3?

Do not claim that FL must outperform centralized training. The pooled model is a reference, not guaranteed to be an upper bound. Do not claim hierarchical averaging alone improves accuracy: with mass-preserving two-level aggregation over identical updates, it should match flat FedAvg. Its value in this phase is the edge/global organization and realistic participation structure.

---

## 9. Ready-to-paste coding-agent prompt

> You are implementing Phase 2 of TRAF-MA in the supplied existing repository. Read this entire guide and inspect the code before making changes.
>
> **The checked-in Phase 1 implementation is the source of truth for existing behaviour.** Preserve current `rsu_N/E/S/W` identifiers, schema 1.1 compatibility, observation boundaries, queue target in metres, congestion thresholds 0.10/0.35, ten-step windows, 30-second forecast horizon, run-level splits, deterministic traces and all existing tests. Do not rename RSUs to cameras or replace current labels merely to match stale planning documents. Treat the reference docs as intended design only where they do not conflict with the running code.
>
> Implement Phase 2 in this order: (1) re-audit the generated v2 data and build baselines; (2) validate then implement a deterministic common feature adapter and masks from actual agent logs; (3) train/evaluate the common two-head GRU locally and as a pooled offline reference; (4) implement flat sample-weighted FedAvg; (5) freeze RSU learner identity/partitioning before implementing mass-preserving hierarchical FedAvg; (6) evaluate IID control, natural non-IID data and transient-client participation; (7) produce a reproducible experiment runner, metrics, plots, report and live/replay demonstration.
>
> Keep evaluator truth and labels out of live inference and agent features. Do not transmit raw local histories in federation updates. A vehicle replay client is not a persistent live vehicle. The current code has no learning clients; the four-persistent-RSU-client design is still proposed. Record the approved client identity/partition decision before implementing it. If RSU self-updates are approved, count each exactly once; do not double-count child samples or equally average unequal groups. Test that hierarchical FedAvg equals flat FedAvg on identical updates and sample masses.
>
> Use `phase2_v2_dev` while developing; generate the full `phase2_v2` benchmark only after the pipeline is stable. Keep signal-broadcast data separate for a matched ablation. Do not tune on test. Save immutable run manifests, resolved configs, dataset hashes, seeds, checkpoints, predictions, metrics, round logs, plots and reports.
>
> Trust scoring, attacks, trust-weighted aggregation, consensus, quarantine/recovery and the full FastAPI/React dashboard are Phase 3. Do not implement placeholders or claim these are part of Phase 2.
>
> Work milestone by milestone. After each milestone, report changed files, design decisions, tests/results, artefacts, limitations and next step. Do not claim a feature or result is complete without executable evidence.

---

## 10. Stop-and-clarify conditions

Pause and ask the team before proceeding if:

- A required feature cannot be derived from the current agent logs without violating observation boundaries.
- Someone proposes changing queue units, queue definition, congestion thresholds, label horizon or split seeds.
- Someone proposes renaming RSUs or changing the message envelope.
- The feature dimension is changed from 22 or its order/normalization changes after training starts.
- A model update contains raw local examples or evaluator-only information.
- The proposed hierarchical aggregation cannot be shown equivalent to flat FedAvg on a fixed fixture.
- A result improves only after test-set tuning or selective omission of difficult scenarios.
- A live demo is being described as online FL when it actually uses offline replay clients.

**Completion standard:** TRAF-MA Phase 2 is complete when the system can reproduce its learning results from saved artefacts, demonstrate local-to-edge-to-global federated learning, quantify the trade-offs honestly, and pass all regression and new correctness tests—not merely when a GRU trains or a dashboard displays numbers.
