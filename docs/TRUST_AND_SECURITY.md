# Trust, attacks and recovery methodology

Implement in Phase 3 after the clean Phase 2 baseline. Values here define a reproducible starting method, not established optimal security thresholds. Tune on calibration/validation runs and freeze before test experiments.

## Threat model

Attackers can falsify their own outgoing semantic observations/predictions or alter their own local model update. They retain a registered identity. They cannot edit SUMO truth, another agent's history, the evaluator, global coordinator, or camera routing/aggregation code. A compromised camera sensor/self-model is an optional higher-impact experiment; a compromised camera aggregator, Sybil identities, collusion, coordinator compromise and cryptographic attacks are outside the core claim.

Thus this is empirical resilience under stated assumptions, not a general Byzantine-security proof. Model/semantic exchange is data minimization, not formal privacy. A clean, representative calibration dataset at the trusted edge is an explicit assumption and must be included in limitations and ablations.

## Two attack hooks

1. **False information:** intercept the selected agent's outgoing semantic message after local measurement. Replace its reported queue/speed/congestion with configurable biased values. First use values inside allowed ranges to test consistency detection; impossible values are a separate validation case. Do not alter simulator truth or automatically poison the training data. This primarily attacks traffic-state fusion.
2. **Model poisoning:** intercept the outgoing delta after legitimate local training. For sign flip, transmit `delta_attack = -lambda * delta_clean`, starting lambda=5. Gaussian perturbation can be a second case. Keep semantic messages honest for the isolated poisoning experiment. This primarily attacks global prediction quality.

Store attack identity, seed, channel, magnitude, start/end time and affected rounds in evaluator-only records. Core fractions: 0%, 10%, 20%, 40% of eligible vehicle clients; record exact integer attackers and actual fraction when rounding is necessary. Include how attackers distribute across camera groups.

## Evidence cycle and availability

Run semantic validation on each report. Once per learning/trust round, summarize fresh evidence, compute reliability and update historical trust once. Freeze all prior trust scores before contextual comparisons; do not mutate neighbors' trust while scoring the same round.

For each factor preserve score, availability, reason, scope and timestamp. With no valid evidence, retain trust and mark unavailable. Exclude unavailable factors from the weighted sum and renormalize available weights. Do not substitute stale model-quality evidence indefinitely or make a silent client automatically innocent/guilty.

### C: contextual consistency

Compare current observations only in common fully observable bins and within the freshness bound. Use the same metric/units. Exclude the agent itself and any relayed copy of the same source. Camera observations generated independently by the camera sensor are permitted references; a camera summary derived from children is not new evidence.

```text
G_ij = coverage_overlap_ij * exp(-abs(time_i-time_j)/5)
reference_i = sum(T_previous_j * G_ij * z_j) / sum(T_previous_j * G_ij)
C_i = 1 - mean(clamp(abs(z_i-reference_i)/configured_metric_range, 0, 1))
```

Spatial compatibility is enforced by shared approach/bin identity rather than proximity across unrelated roads. Use per-metric masks and matching coverage. Zero denominator or no comparable metrics means C unavailable. Do not compare one vehicle's partial queue against an entire approach's queue. Forecast agreement, if added, uses the same target approach/time and remains separate from current observation consistency.

### P: physical and delayed-prediction consistency

Range plausibility checks finite values, nonnegative queue/waiting/counts, sensor geometry and configured physical envelopes with tolerance. Count failed checks divided by applicable checks and set P_range=1-failure_fraction. Verify before clipping.

Prediction consistency is evaluated only when an earlier forecast's horizon matures and a valid independent current regional estimate is available. Normalize queue/probability error against the matching target; define P_prediction=1-clipped_mean_error. Never compare a t+30 prediction with a t observation. Use the mean of available P subfactors; when neither is available, P is unavailable. The detector cannot use the private evaluation oracle. Independence and confidence of the reference must be logged.

### S: temporal plausibility

Compare successive same-origin, same-scope reports. Estimate allowed changes from clean calibration traffic, conditioned on signal transition and observed flow/coverage. Set S=1-clipped normalized excess change. A changed field of view resets that comparison. Initial implementation uses per-metric clean 99th-percentile change envelopes; save them in calibration artifacts. A sudden change supported by independent overlapping observations is an incident candidate, not automatically an agent anomaly.

### M: model-update quality

Edge validates a candidate `theta_base + delta_i` on a clean calibration set from separate runs. Use the same normalized two-head loss as training, reporting results by approach/type as well as overall. Compare candidate loss with the same base model on exactly those examples.

```text
degradation = max(0, (loss_candidate-loss_base) / max(loss_base, epsilon))
M_i = 1 - min(1, degradation / delta_M)
```

Starting delta_M=0.20, validated through clean non-IID controls. No eligible candidate or no representative calibration data means M unavailable. Honest update direction/norm difference alone is not a poisoning verdict. Preserve calibration access and compute cost in the experimental record.

### N: delivery reliability

N=0.5*delivery_ratio + 0.3*eligible_round_availability + 0.2*latency_score. Compute only over expected messages/eligible rounds while the client is active. Latency score = 1-min(delay/1 s,1), measured in the declared simulated delivery clock. Zero expected messages means unavailable. Network loss can reduce usable contribution but cannot by itself trigger a malicious-agent quarantine.

## Reliability and historical trust

Starting weights from the earlier discussion:

```text
R_i = weighted_available_mean(C=0.30, M=0.25, P=0.20, N=0.10, S=0.15)
T_i_new = 0.8*T_i_previous + 0.2*R_i
T_i_initial = 0.5
```

Clip computed scores to [0,1] only after recording invalid inputs. Trust initializes as MONITOR, not TRUSTED. Scores and state transitions are different records.

**Necessary correction to the earlier chat:** a convex score can remain high when only one channel is malicious. If M=0 and every other factor=1, R=0.75; a trust threshold of 0.20 can never quarantine that consistently poisoned client. Therefore use explicit channel rejection and repeated-evidence transitions in addition to the shared reputation score. Evaluate their separate effect in ablations; do not attribute their benefit solely to T.

## Eligibility, lifecycle and recovery

At each trust round compute an evidence flag, without attack labels:

- `bad_model`: candidate degradation above delta_M on a sufficiently representative calibration set.
- `bad_report`: repeated invalid physical payload, or C<0.3 with an independent overlapping reference plus S<0.3 or a physical contradiction. A single disagreement without support is not enough.
- A traffic incident corroborated by distinct original sensors is exempt from the temporal-change contradiction, while impossible fields still fail validation.

A harmful/invalid message is excluded from its current channel immediately, with a reason. One weak statistical flag is recorded as monitoring evidence; it is not proof of intent. After three consecutive eligible rounds with the same corroborated bad-channel flag, quarantine the client. Also quarantine after T<0.20 for three consecutive evidence-bearing rounds with non-network behavioral evidence. All counters and thresholds are configuration.

Normal displayed score bands: TRUSTED >=0.75; MONITOR [0.40,0.75); SUSPICIOUS [0.20,0.40); below 0.20 remains SUSPICIOUS until the persistence guard fires. A fresh strong flag can display SUSPICIOUS even with higher T. Always show the transition reason rather than inferring status from color/score alone.

QUARANTINED overrides score bands and sets both model and consensus eligibility to zero. Continue bounded shadow observations and candidate validation; reject their operational influence. Bad counters do not improve just because the agent disconnects.

Recovery policy:

1. At least three consecutive observed eligible rounds with R>0.70 and no bad flag make a quarantined client eligible for RECOVERING. For poisoning quarantine, recovery must include clean model candidate checks; semantic silence/honesty alone cannot clear it.
2. RECOVERING needs T>=0.40 before any influence returns. Cap its ordinary effective weight with a multiplier 0.25, 0.50, 0.75 over the next three consecutive clean rounds.
3. After those three rounds, return to the appropriate normal score band. TRUSTED additionally requires T>=0.75. Missing evidence pauses recovery; it does not advance counters.
4. A new corroborated bad-channel flag during recovery resets the recovery streak and returns the client to QUARANTINED. Record the relapse.

This explicitly adds persistence and probation beyond the historical chat's simple EMA example. Test false quarantine and on-off relapse rather than presuming this policy is sufficient.

## Trust-aware model aggregation

Effective model mass for an accepted client: `a_i = n_i * T_i^2 * model_eligibility_i * recovery_multiplier_i`. Use the same frozen a_i at edge and global aggregation; group mass is the sum of client masses. If all masses are zero, retain the previous model and emit an insufficient-contribution event.

Reject stale/wrong-version/NaN models regardless of trust. Trust does not waive protocol validity. Quarantine affects individual clients; a camera's sensor reputation must not erase all its children's mass.

## Traffic-state fusion and incident alerts

“Consensus” here means coordinator-based semantic fusion, not a proven distributed agreement protocol. For each approach/bin/current-time bucket, compute per-metric trust × freshness weighted means over eligible independent sources with adequate overlap. Queue estimates for the same bin are averaged, not summed across sensors. Sum fused bin queues across disjoint bins to get approach queue only with adequate coverage. Missing bins produce partial coverage/unknown full total, not a false zero. Global state retains four approach estimates; an intersection total is a sum of disjoint fully covered approach estimates.

Forecast fusion is a separate product: same target approach, target time and horizon only; weighted class probability vectors are normalized; queue forecasts are nonnegative. Correlated sensors are not claimed independent statistical evidence merely because IDs differ. Preserve direct/relay provenance.

Traffic incidents require sustained regional evidence: start with queue ratio rising by >0.20 or normalized speed falling by >0.30 relative to the prior 30 s window, persisting 5 s and supported by at least two independent overlapping origins with T>=0.75. If support is insufficient, show unconfirmed evidence. These are calibration defaults; include congestion from an ordinary red phase as a false-incident control. A genuine controlled stopped-vehicle incident changes SUMO physics, whereas a false report changes messages only.

Alerts: separate `agent_anomaly`, `quarantine`, `recovery`, `traffic_incident`, `stale_data` and `insufficient_contribution`. Deduplicate by event kind + origin/region; update an active event instead of emitting it each second. Store first/last time, evidence and resolution. Dashboard notifications are local in-app events; external email/SMS is not required.
