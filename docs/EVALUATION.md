# Evaluation, security analysis and final demonstration

## Reproducibility contract

Every run stores run ID, source revision if available, seed, resolved config, SUMO/Python/library versions, hardware, schema/model/calibration hashes, demand/routes, split manifest, participant counts, attack schedule (evaluator only), checkpoints, predictions, trust/effective weights, events and metrics. Keep simulated time and wall-clock time distinct. Version raw data and transformations; never overwrite a reported experiment silently.

Use at least five paired seeds for final comparisons if compute permits; begin with one-seed smoke experiments. Compare methods on identical traffic/attack schedules and initial model weights. Report mean and standard deviation over run-level metrics; confidence intervals may use run-level bootstrap. Do not treat overlapping windows as independent replications.

## Baseline ladder

| ID | Method | Why it is included |
| --- | --- | --- |
| B0 | Persistence: future queue/current visible queue and corresponding congestion rule | Checks whether the forecast model adds value beyond current measurements; report partial-coverage limitation |
| B1 | Independent local model | Measures learning without collaboration |
| B2 | Offline pooled central model | Reference using the same training examples/budget; not guaranteed to be an upper bound |
| B3 | Flat FedAvg on synthetic IID training partition | Diagnostic control for client distribution |
| B4 | Flat FedAvg on natural non-IID local data | Isolates heterogeneity |
| B5 | Hierarchical FedAvg without trust | Direct deployment baseline for the chosen hierarchy |
| B6 | Hierarchical FL plus trust, channel gates and recovery | Full proposed method |

The pooled model and synthetic IID partition are evaluator-only exceptions to operational data locality. Label them accordingly. For B0, also show a camera-only full-coverage persistence reference; do not secretly give partial vehicle predictors the full oracle current queue. Use the same coverage strata when comparing models.

**Local-only evaluation with unseen vehicle identities:** a test-run vehicle has no matching training-run private model. For the main whole-run holdout, evaluate each trained local model on the held-out approach/type cohort matching that training client; evaluate the central/FL models on those exact same rows. Report per-client model metrics and their macro-average, plus unique-row pooled metrics for the global models. This measures cross-run generalization of locally trained models, not a new vehicle learning from its future test labels. Do not average local predictions into an unlabelled ensemble. If testing personalization separately, define a new support/query protocol with causal label maturity and a gap at least history plus horizon; keep those results distinct from the whole-run benchmark.

B4/B5 should yield matching results on identical updates with one aggregation step per round. This is a correctness expectation, not an opportunity to claim hierarchical learning necessarily improves accuracy.

## Experiment matrix

| ID | Scenario | Methods | Essential observation |
| --- | --- | --- | --- |
| E01 | Light/moderate/heavy clean traffic | B0–B5 | Prediction and cost by demand/approach/type |
| E02 | Natural asymmetric non-IID data | B1–B6 | Accuracy differences and honest-agent trust behavior |
| E03 | False semantic reports only | B5 vs B6 | Fused-state error, detection and false quarantine; no direct model attack |
| E04 | Sign-flip updates only | B5 vs B6 | Prediction degradation, attacker effective model mass and latency |
| E05 | Combined attacks | B5 vs B6 | Interaction of message and learning defences |
| E06 | Honest partial views with no attackers | B5 vs B6 | False flags and quarantine under disagreement |
| E07 | Genuine stalled vehicle / demand surge | B5 vs B6 | Incident detection versus blaming honest observers |
| E08 | Attack stops; later relapse | B6 | Quarantine delay, shadow monitoring, recovery time and relapse response |
| E09 | Delay/dropout without attackers | B5 vs B6 | Availability effects and network-only false accusations |
| E10 | N scaling: 20, 50, 100 initially | B5 vs B6 | Runtime/memory/bytes and actual eligible population |

Sweep 10/20/40% attacker fractions in E03–E05 after smoke checks. Vary group concentration as well as global attacker fraction. Record exact number and fraction of compromised eligible clients; four-camera sensor attacks should be separate named runs because 1/4 is already 25% of infrastructure clients.

Start with clean baseline, then one attack type at a time, then combined/stress tests. Do not launch a huge full-factorial grid before checking labels and metrics.

## Metrics and denominators

| Area | Metrics | Definition/qualification |
| --- | --- | --- |
| Congestion forecast | Macro-F1, balanced accuracy, accuracy, class confusion matrix | Held-out t+30 labels; show class support |
| Queue forecast | MAE, RMSE in vehicles | Same region/horizon; report per-approach/type and aggregate |
| Current consensus | Queue MAE, speed error, valid coverage fraction | Compare fused current state with evaluator truth; distinguish from prediction error |
| Detection | Precision, recall/TPR, FPR, F1 | Eligible agent-round with active attack vs clean, plus per-attack-episode detection |
| False quarantine | Benign quarantine transitions / benign eligible agent-rounds; fraction of benign agents ever quarantined | State both denominators; don't call benign disagreement malicious |
| Detection latency | First sustained flag/quarantine time minus attack start | Report seconds and rounds; undetected cases censored, not assigned zero |
| Attacker influence | Sum of attacker effective weights / sum of all effective weights | Separate model and consensus channels; all-zero denominator is unavailable |
| Recovery | Time from attack stop to RECOVERING / full eligibility / TRUSTED | Separate endpoints; show failures to recover |
| Incidents | Episode precision/recall and alert delay | Controlled physical incident labels, separate from attack labels |
| System | Wall-clock round/inference latency, simulation speed ratio, peak memory | State hardware and simulator-pausing behavior |
| Communication | Serialized upstream/downstream model/semantic bytes, edge/global separately | Include broadcasts, metadata and failed/retried messages where applicable |

Measure a raw-data-transfer counterfactual using a declared encoding, sampling rate and horizon. A small telemetry table may use fewer bytes than model updates; never promise FL communication savings without measurement. Logical in-process serialization size is not real radio/5G performance.

## Ablations

After B6 works: remove C/relevance, remove M, remove history smoothing, remove persistence/probation, and compare reputation weighting alone against channel gates alone versus their combination. Keep runs/budget matched. This exposes whether benefits come from contextual trust, reference validation or deterministic rejection. Add a no-clean-calibration-data variant with M unavailable and report the resulting limitation, not a guaranteed root-free defence.

Median/trimmed mean can be optional extra baselines after primary results. Do not imply they are interchangeable with trust under all client distributions.

## Critical correctness fixtures

- Agent cannot receive out-of-zone data; missing and measured zero differ.
- Future target alignment and whole-run split boundaries are correct.
- Model shape/hash/version mismatch is rejected before aggregation.
- Unequal group sizes, self-camera updates and dropped clients match a hand-calculated global average.
- Duplicate relays neither increase trust support nor model mass.
- Empty evidence, empty groups and all quarantined clients yield explicit fallback/unknown state.
- Same-round trust uses a frozen previous snapshot; order of evaluating agents does not change results.
- A single bad trust factor can trigger guarded channel policy when justified, even if total EMA never reaches 0.20.
- A quarantined update cannot change the model; recovery shadow evidence cannot accidentally count as an accepted contribution.
- Honest non-overlapping regions, signal transitions and connectivity loss do not automatically establish malicious behavior.

Use unit fixtures for those invariants and integration tests for SUMO lifecycle, saved-run replay and phase demonstrations. The final measured experiments are separate from software tests.

## Final demo script

Use one saved deterministic configuration and a prerecorded replay fallback. Show four approaches, agent hierarchy and configurable population first. Display a local future prediction and its later outcome, then a completed HFL round and its contributors.

After a clean warm-up long enough to establish trust, activate a false-report attack. Show original scope, reported state, evidence, reduced contribution and any quarantine. End the attack; show bounded shadow monitoring and guarded recovery. In a separate isolated run, repeat with model poisoning and the affected FL weights/loss. Finish with honest disagreement and a real traffic incident to demonstrate that the system distinguishes those cases.

Use an event-driven or sufficiently long fixed schedule so there are enough clean/attack/recovery rounds; do not hard-code an unrealistically short recovery clip and imply the policy worked. Replay recorded evidence if live computation exceeds presentation time.

## Final report deliverables

Reproduction guide; final architecture; feature/label/FL/trust contracts; threat model and assumptions; baselines and matched experiment table; prediction curves; confusion matrices; trust/effective-weight plots; incident/recovery timeline; resource/communication table; limitations and negative results. Paper metrics remain external results and never become TRAF-MA measurements.
