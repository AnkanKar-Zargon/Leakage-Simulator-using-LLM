# Leakage Simulator

> Generate "trap datasets" with subtle ML issues → test if an LLM can detect them → evaluate reasoning quality.

**Stack:** Python · LLM engine (I used Groq for this demo setup) · MLflow · Flask · Vanilla JS dashboard

---

## Quickstart

```bash
pip install -r requirements.txt
# add LLM_API_KEY in .env
cd src && python runner.py   # run experiment (~30s, ~4k tokens)
python app.py                # start API on :5050
# open dashboard.html in browser
```

---

## Project Structure

```
src/
  datasets.py   — 5 trap dataset generators
  evaluator.py  — Groq LLM caller + scoring
  runner.py     — experiment orchestrator + MLflow logging
  app.py        — Flask API for dashboard
dashboard.html  — standalone frontend
reports/REPORT.md
```

---

## Trap Datasets

| Dataset | Issue |
|---|---|
| `target_leakage` | Feature derived post-hoc from label |
| `temporal_leakage` | Rolling average uses future data |
| `class_imbalance` | 99:1 fraud — accuracy illusion |
| `train_test_contamination` | Scaler fitted before split |
| `overfit_trap` | 150 rows, 52 features (50 noise) |

---

## Example Evaluation Scoring

| Dimension | Weight |
|---|---|
| Issue type correctly detected | 50% |
| Leaky feature identification (F1) | 25% |
| Reasoning quality (keyword depth) | 25% |

---

## MLflow

```bash
mlflow ui --backend-store-uri experiments/mlruns
```

---

See `reports/REPORT.md` for full technical writeup.


---



<img width="1886" height="930" alt="Leakage Simulator Dashboard" src="https://github.com/user-attachments/assets/74d3c184-9f45-46cb-bf7c-55344dd8eb4d" />
<p align="center"> <i><strong>Figure: Leakage Simulator Dashboard</strong></i> </p>
