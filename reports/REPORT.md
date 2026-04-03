# Leakage Simulator — Technical Report

## Overview

A tool that generates "trap datasets" with embedded ML issues, then evaluates whether an LLM (Qwen3-32b via Groq) can detect and reason about those issues. The goal is not prediction accuracy but *diagnostic reasoning quality*.

---

## Architecture

```
leakage-simulator/
├── src/
│   ├── datasets.py     # Trap dataset generators
│   ├── evaluator.py    # Groq LLM caller + scorer
│   ├── runner.py       # Experiment orchestrator (MLflow)
│   └── app.py          # Flask API for dashboard
├── dashboard.html      # Standalone frontend
├── .env.example
├── requirements.txt
└── reports/
    └── REPORT.md       ← this file
```

**Design principle:** minimal scripts, single responsibility, token-efficient prompts.

---

## Trap Datasets

| Dataset | Issue Type | Key Mechanism | Leaky Feature |
|---|---|---|---|
| `target_leakage` | Target leak | `approval_score` computed from label | `approval_score` |
| `temporal_leakage` | Time leak | Rolling avg uses future values | `rolling_30d_avg` |
| `class_imbalance` | Imbalance | 99:1 fraud ratio | — |
| `train_test_contamination` | Preprocessing | Scaler fitted on full data | `feature_a/b_scaled` |
| `overfit_trap` | Overfitting | 150 rows, 52 features (50 noise) | — |

### Why these are "subtle"

- **Target leakage**: The feature name (`approval_score`) sounds like a legitimate domain variable, not an artifact of the target. Naive models trained on it achieve AUC > 0.98.
- **Temporal leakage**: A rolling average looks reasonable; the direction (forward vs backward) is invisible in summary stats.
- **Class imbalance**: With 99% accuracy easily achievable via majority vote, naive reporters are misled.
- **Preprocessing contamination**: The features look scaled — the issue is *when* the scaler was fit, not *that* it was fit.
- **Overfitting**: High dimensionality relative to sample count is visually obvious in `df.shape` but easy to overlook.

---

## LLM Evaluation

### Prompt Design

System prompt specifies exact JSON schema with keys: `detected_issues`, `leaky_features`, `issue_type`, `severity`, `reasoning`, `recommended_fixes`, `confidence`. Temperature = 0.2 for consistency.

Dataset context sent to LLM contains: column names, dtypes, shape, first 5 rows, and `describe()` statistics. No ground truth is revealed.

Token budget per call: ~700 output tokens max. Total experiment cost: ~3,500–5,000 tokens.

### Example Scoring Rubric

| Dimension | Weight | Method |
|---|---|---|
| Issue Detection | 50% | Exact match of `issue_type` to ground truth |
| Feature Precision | 25% | F1 score of named leaky features vs ground truth |
| Reasoning Quality | 25% | Keyword density heuristic over 15 ML-relevant terms |

**Overall score** = `0.5 × detected + 0.25 × feature_f1 + 0.25 × reasoning`

---

## Experiment Tracking

MLflow logs per-dataset metrics and aggregate scores. Run artifacts stored as JSON in `experiments/run_YYYYMMDD_HHMMSS.json`. Dashboard reads the latest run file.

To view MLflow UI:
```bash
mlflow ui --backend-store-uri experiments/mlruns
```

---

## Known Failure Modes

1. **LLM over-detects leakage** — may flag legitimate features if names sound suspicious.
2. **Reasoning score is heuristic** — keyword count is a proxy; a future improvement is using a second LLM call to rate reasoning quality on a rubric.
3. **Class imbalance detection** — requires the LLM to infer class distribution from `describe()` stats, which is non-trivial without seeing `value_counts()`.
4. **Preprocessing contamination** — hardest to detect; the only signal is that features are already scaled, with no split information provided.

---

## Theoretical Background

### Data Leakage
Leakage occurs when information from outside the training window enters the model. Target leakage is post-hoc feature construction; temporal leakage violates the causal ordering of observations.

### Class Imbalance
Accuracy is not a valid metric under imbalance. Correct metrics: Precision-Recall AUC, Matthews Correlation Coefficient (MCC), or F1 with appropriate thresholding.

### Overfitting Mechanics
When p (features) >> n (samples), models memorise training noise. The bias-variance decomposition predicts high variance / low bias in this regime. Regularisation, cross-validation, and feature selection are the standard mitigations.

### LLM Reasoning Evaluation
Chain-of-thought prompting (implicit here via the `reasoning` field) improves structured diagnosis. Self-consistency decoding (sampling k responses and taking majority) would further reduce variance but is omitted to minimise token cost.

---

## Reproducing Results

```bash
git clone <repo>
cd leakage-simulator
pip install -r requirements.txt
# add your LLM_API_KEY to .env (I used free tier groq for this demo)
cd src
python runner.py --seed 42    # run experiment
python app.py                  # start dashboard API
# open dashboard.html in browser
```

---

## References

- *Data Leakage in Machine Learning* — Kaufman et al. (2012)
- *Chain-of-Thought Prompting* — Wei et al. (2022)
- *Self-Consistency Improves CoT Reasoning* — Wang et al. (2022)
- Groq Qwen3-32b documentation: https://console.groq.com
