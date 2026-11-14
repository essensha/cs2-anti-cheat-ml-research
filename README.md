# Counter-Strike 2 Anti-Cheat ML Research

**Independent applied machine learning research | Synthetic and illustrative data | Not a deployed anti-cheat system**

This repository explores how to design, evaluate, and operate anti-cheat detection in a multiplayer game setting. It examines individual and coordinated behavior, label-selection bias, concept drift, skill-segment fairness, human-review capacity, and failure handling. Counter-Strike 2 provides the *gameplay framing*; this is **not Valve telemetry, VAC integration, or a production service**.

## The player problem

An anti-cheat detector must surface suspicious behavior without treating exceptional legitimate skill as cheating. False positives can damage player trust, and a high offline AUC does not automatically produce a safe enforcement system. The proposed decision process therefore separates **risk scoring**, **investigative review**, and **any enforcement action**. Model scores are investigative signals, not proof.

## What is implemented vs. proposed

| Component | Status | Evidence |
| --- | --- | --- |
| Synthetic gameplay-feature exploration and temporal holdout | Notebook experiments | [Skill parsing and productionalization](Anti_Cheating_Skill_Parsing_and_Productionalization.ipynb) |
| Logistic regression and Isolation Forest comparison | Notebook experiments | [Skill parsing and productionalization](Anti_Cheating_Skill_Parsing_and_Productionalization.ipynb) |
| Probability calibration, threshold and review-budget analysis | Notebook experiments | [Existing vs. new model](Anti_Cheating_(Exisiting_vs_New_Model).ipynb) |
| Concept drift and simulated model-health decisions | Notebook experiments | [Adversarial adaptation](Adversarial_Adaptation_and_Concept_Drift.ipynb) |
| Player-interaction graphs and graph-informed scoring | Small synthetic examples | [Graph-based coordinated cheating](Graph_Based_Coordinated_Cheating_(Nodes_and_Edges).ipynb) |
| Event throughput, queue backlogs, windowing and review capacity | Arithmetic / illustrative simulations | [Real-time production detection](Real_Time_Production_Detection.ipynb) |
| Real-time deployed scoring API, streaming infrastructure, anti-cheat enforcement | **Not implemented or verified** | [Architecture proposal](docs/ARCHITECTURE.md) |

## Notebook guide

1. [Counter-Strike 2 Cheat Detection Dataset](Counter_Strike_2_Cheat_Detection_Dataset.ipynb) — exploratory event-level features and a small labeled-player example. Some cells reference local files not included here.
2. [Anti-Cheating: Existing vs. New Model](Anti_Cheating_(Exisiting_vs_New_Model).ipynb) — synthetic populations, reaction-time baseline, multifeature classifiers, investigation-label bias and calibration.
3. [Anti-Cheating: Skill Parsing and Productionalization](Anti_Cheating_Skill_Parsing_and_Productionalization.ipynb) — 12,000-row synthetic dataset, temporal validation, logistic regression, Isolation Forest, thresholds, skill segments, scoring fallbacks and simulated latency.
4. [Adversarial Adaptation and Concept Drift](Adversarial_Adaptation_and_Concept_Drift.ipynb) — before/after drift scenario, changing precision/recall, selective review labels and health gates.
5. [Graph-Based Coordinated Cheating](Graph_Based_Coordinated_Cheating_(Nodes_and_Edges).ipynb) — NetworkX graph construction, clustering, recency weighting and illustrative graph-enhanced scoring.
6. [Individual and Coordinated Cheating Simulation](Individual_and_Coordinated_Cheating_Simulation.ipynb) — small reaction-consistency examples and segment-level false-positive calculations.
7. [Real-Time Production Detection](Real_Time_Production_Detection.ipynb) — **simulation of production considerations**, not a running production service; throughput, backlog, windows and review metrics.

The repository currently contains **seven** Jupyter notebooks. Some notebook names use “production” to describe a *design exercise*, not a deployed application.

## Selected results and how to interpret them

The following are **saved notebook outputs** from synthetic or illustrative exercises, not measurements of a live game system. They have **not been independently rerun as part of this documentation update**.

| Exercise | Saved output | Interpretation |
| --- | --- | --- |
| Skill-parsing notebook: scoring latency | Mean **2.608 ms**, P95 **6.1 ms**, P99 **10.059 ms** | Local simulation/benchmark; not an end-to-end service latency SLO or load test |
| Skill-parsing notebook: request status | **94.4% success**, **3.3% insufficient telemetry**, **2.3% model unavailable** | Simulated outcomes over 1,000 requests, not real uptime |
| Existing vs. new model: reviewed-only training | ROC-AUC **0.9852**, Brier **0.0328** | Synthetic label-selection experiment |
| Existing vs. new model: representative training | ROC-AUC **0.9849**, Brier **0.0097** | Synthetic experiment suggests better probability calibration with representative labels |
| Concept drift: before / after | ROC-AUC **0.998 → 0.742**; recall **0.985 → 0.120** | Constructed drift scenario showing a fixed detector degrading |
| Graph example: Precision@100 | **65% vs. 82%** | **Illustrative hard-coded comparison**, not a measured improvement from a trained graph model |
| Real-time detection example | Precision **75%**, recall **60%**, FPR **0.62%** | Calculated from assumed confusion-matrix counts, not live detection results |

**Why these metrics matter:** ROC-AUC/PR-AUC assess ranking under class imbalance; precision and recall at an investigation threshold reflect review cost; Brier score and calibration reveal whether scores can support decisions; false-positive rates by player-skill segment help identify disparate player impact. AUC alone is insufficient.

For further verification, see [evaluation and limitations](docs/EVALUATION_AND_LIMITATIONS.md).

## Proposed system architecture

See the [architecture diagram and operational decision path](docs/ARCHITECTURE.md).

At a high level: game-event telemetry → validated feature windows → candidate detectors → calibrated risk scoring → review queue → human adjudication and delayed labels → monitoring and controlled retraining. Missing telemetry or model failure should **abstain or fall back safely**, not trigger automatic punishment.

## Reproducing the research

The notebooks are exploratory and **not yet a fully reproducible one-command pipeline**. Several read files from absolute paths such as `/srv/synthetic_anticheat_telemetry.csv` or `/srv/Counter Strike cheating.csv`; those source files are not included in this repository. Other cells generate their own synthetic examples. Open a notebook in Jupyter or Colab, inspect its dependencies and required inputs, and run applicable cells in order. Do not assume every notebook will run from a clean checkout.

Typical Python packages used across the notebooks include `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `networkx`. No pinned environment or CI verification is currently provided.

## Responsible use and limitations

- All modeled labels and gameplay scenarios are synthetic, illustrative, or locally supplied; none should be interpreted as verified player cheating.
- Synthetic data can encode unrealistic separability, label bias, and temporal or match-identifier inconsistencies; high model scores may not transfer to real gameplay.
- Some analyses use assumed counts or hand-entered outcomes to illustrate trade-offs. These are identified above.
- A prototype score must **not** automatically ban players. Review capacity, appeal processes, evidence quality, segment fairness and privacy controls are central to any real deployment.
- Proposed streaming, serving, latency SLOs, canary rollout, rollback and production monitoring are architectural considerations, not validated deployments.

**Author:** [Essence Lynn Wilson, PhD](https://github.com/essensha) · Independent research portfolio.

## Clean-checkout reproducible baseline (new)

A standalone, **fully synthetic** experiment is now included alongside the original exploratory notebooks. It generates its own data, applies a chronological train/test split, fits logistic regression and Isolation Forest, and exports metrics plus skill-segment false-positive rates. The generator deliberately uses **new synthetic data**, so its metrics **will not match** the historical notebook outputs above.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python generate_synthetic_data.py
python run_baseline.py
python -m unittest test_baseline.py
```

Outputs are written to `outputs/metrics.json` and `outputs/predictions.csv`; generated data is written to `data/synthetic_anticheat_telemetry.csv`. Both directories are ignored by Git. Use `--rows` and `--seed` on the generator and `--threshold` on the baseline to explore trade-offs.

**Scope:** this is an offline research baseline, not a streaming system or enforcement model. The original notebooks remain unchanged and may still require local files. The commands were executed successfully in a separate Python environment with NumPy 2.3.5, pandas 2.2.3, and scikit-learn 1.8.0; both unit tests passed. This verifies the standalone baseline on that environment, not compatibility with every dependency version or operating system.

### Verified standalone run (seed 42, 12,000 rows)

A separate execution generated **9,667 training rows** and **2,333 temporal test rows** (cutoff day 48). Saved console metrics from that execution:

| Metric | Result |
| --- | ---: |
| Logistic regression ROC-AUC | 0.999994 |
| Logistic regression average precision | 0.999900 |
| Logistic regression Brier score | 0.000647 |
| Logistic regression precision at threshold 0.70 | 1.0000 |
| Logistic regression recall at threshold 0.70 | 0.9787 |
| Isolation Forest ROC-AUC | 0.990523 |
| Isolation Forest average precision | 0.894990 |
| Automated tests | 2 passed |

**Interpretation:** The generated label patterns are deliberately strong and highly separable. Near-perfect metrics, and the observed zero false positives across skill groups in this run, **do not establish real-world detection performance or fairness**. This baseline validates code execution and evaluation wiring, not deployment readiness.
