# Evaluation Notes and Limitations

This document separates **saved notebook outputs**, **illustrative inputs**, and **unverified production claims**. Values below were inspected from committed notebook outputs; they were not independently rerun for this documentation update.

## Experiment groups

| Notebook | Evaluation evidence | Caveat |
| --- | --- | --- |
| Skill Parsing and Productionalization | 12,000-row synthetic dataset; chronological split; logistic regression and Isolation Forest; precision/recall, ROC-AUC/PR-AUC, calibration, skill segments, latency and fallback simulations | Local data file is not committed; temporal split may include synthetic match-ID inconsistencies; latency is not end-to-end service latency |
| Existing vs. New Model | Synthetic 100,000-player population; reviewed-only vs. representative-label training; ROC-AUC 0.9852 vs. 0.9849; Brier 0.0328 vs. 0.0097; review capacity 200 | Illustrates label-selection bias; synthetic labels and assumptions limit external validity |
| Adversarial Adaptation | Constructed before/after scenario: ROC-AUC 0.998 to 0.742; fixed-threshold recall 0.985 to 0.120 | Scenario-based drift demonstration, not observed live adversarial adaptation |
| Graph-Based Coordinated Cheating | Small NetworkX graph, clustering, time-decay examples | Precision@100 65% vs. 82% is entered as assumed counts; **not** a validated graph-model lift |
| Real-Time Production Detection | Arithmetic throughput/backlog scenarios and confusion-matrix calculations | No deployed service, real-time ingestion system, load test or production enforcement |
| Counter-Strike 2 Cheat Detection Dataset | Event-level exploration and small labeled-player example | Some inputs are absent and cells may not run cleanly from checkout |

## Responsible evaluation checklist

- [x] Demonstrate temporal validation in a synthetic experiment
- [x] Compare supervised and anomaly approaches
- [x] Explore class imbalance, thresholds and review capacity
- [x] Examine calibration and selective investigation labels
- [x] Explore skill-segment false-positive risks
- [x] Demonstrate missing-feature and model-unavailable statuses
- [ ] Provide versioned, redistributable data generation and clean-run instructions
- [ ] Independently rerun and snapshot all reported metrics in CI
- [ ] Establish validated end-to-end latency SLOs under load
- [ ] Evaluate graph lift on independent, non-hard-coded labels
- [ ] Deploy a monitored scoring service with rollback and audit trail
- [ ] Validate fairness and enforcement safeguards with real, consented, governed data

## Interpretation

**Discrimination is not calibration.** Similar ROC-AUC can coexist with materially different probability reliability. **Precision at a review budget is not enforcement precision.** A flagged case is not a confirmed cheater. **Simulated P99 is not a production SLA.** Local scoring omits ingestion, network, feature freshness, serialization, and concurrency. **Skill fairness matters.** Review segment-level false positives before selecting thresholds.

No results here establish a production-ready CS2 anti-cheat system.
