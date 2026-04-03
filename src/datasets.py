"""
Trap Dataset Generator
Generates datasets with subtle data leakage, class imbalance, and time dependencies.
Each dataset type targets a specific ML pitfall.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import json


@dataclass
class DatasetResult:
    name: str
    df: pd.DataFrame
    target_col: str
    issue_type: str
    ground_truth: Dict[str, Any]  # What issues actually exist
    description: str


def _set_seed(seed: int = 42):
    np.random.seed(seed)


# ── 1. TARGET LEAKAGE ────────────────────────────────────────────────────────

def make_target_leakage(n: int = 500, seed: int = 42) -> DatasetResult:
    """Subtle target leakage: a feature is derived from the target post-hoc."""
    _set_seed(seed)
    age = np.random.randint(18, 70, n)
    income = np.random.normal(50000, 15000, n).clip(10000)
    target = (income > 55000).astype(int)

    # Leaky feature: approval_score is computed AFTER knowing the target
    approval_score = target * np.random.uniform(70, 100, n) + \
                     (1 - target) * np.random.uniform(20, 50, n) + \
                     np.random.normal(0, 3, n)

    df = pd.DataFrame({
        "age": age,
        "income": income.round(2),
        "approval_score": approval_score.round(2),  # ← LEAKY
        "credit_history_years": np.random.randint(0, 20, n),
        "loan_approved": target,
    })

    return DatasetResult(
        name="target_leakage",
        df=df,
        target_col="loan_approved",
        issue_type="target_leakage",
        ground_truth={
            "has_leakage": True,
            "leaky_feature": "approval_score",
            "mechanism": "approval_score is derived from loan_approved after the fact",
            "expected_model_auc": ">0.98 (inflated)",
        },
        description=(
            "Loan approval dataset. `approval_score` is computed post-hoc from "
            "the target variable — a classic target leakage pattern."
        ),
    )


# ── 2. TEMPORAL LEAKAGE ──────────────────────────────────────────────────────

def make_temporal_leakage(n: int = 600, seed: int = 42) -> DatasetResult:
    """Future data bleeds into past via aggregation without time-aware split."""
    _set_seed(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    sales = np.random.normal(1000, 200, n).clip(100)
    trend = np.linspace(0, 500, n)
    sales += trend
    target = (sales > np.median(sales)).astype(int)

    # Leaky feature: rolling 30-day FUTURE average
    s = pd.Series(sales)
    future_avg = s[::-1].rolling(30, min_periods=1).mean()[::-1].values

    df = pd.DataFrame({
        "date": dates,
        "daily_sales": sales.round(2),
        "rolling_30d_avg": future_avg.round(2),   # ← uses future data
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "high_sales_day": target,
    })

    return DatasetResult(
        name="temporal_leakage",
        df=df,
        target_col="high_sales_day",
        issue_type="temporal_leakage",
        ground_truth={
            "has_leakage": True,
            "leaky_feature": "rolling_30d_avg",
            "mechanism": "rolling average computed using future observations",
            "correct_split": "must split by date; never shuffle time-series",
        },
        description=(
            "Daily sales forecasting. `rolling_30d_avg` uses future values — "
            "will cause severe overfitting when data is shuffled before splitting."
        ),
    )


# ── 3. CLASS IMBALANCE + ACCURACY ILLUSION ───────────────────────────────────

def make_class_imbalance(n: int = 1000, seed: int = 42) -> DatasetResult:
    """99:1 imbalance — naive accuracy is misleading."""
    _set_seed(seed)
    n_fraud = max(1, int(n * 0.01))
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        "amount": np.random.normal(200, 80, n_legit).clip(1),
        "hour": np.random.randint(0, 24, n_legit),
        "merchant_risk": np.random.uniform(0, 0.3, n_legit),
        "is_fraud": 0,
    })
    fraud = pd.DataFrame({
        "amount": np.random.normal(800, 300, n_fraud).clip(1),
        "hour": np.random.choice([0, 1, 2, 23], n_fraud),
        "merchant_risk": np.random.uniform(0.6, 1.0, n_fraud),
        "is_fraud": 1,
    })
    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=seed)

    return DatasetResult(
        name="class_imbalance",
        df=df,
        target_col="is_fraud",
        issue_type="class_imbalance",
        ground_truth={
            "imbalance_ratio": f"{n_legit}:{n_fraud}",
            "naive_accuracy_ceiling": f"~{100*n_legit//n}%",
            "correct_metric": "F1, precision-recall AUC, or MCC",
            "common_mistake": "reporting accuracy on imbalanced data",
        },
        description=(
            "Fraud detection with ~1% fraud rate. "
            "A model predicting 'not fraud' always achieves 99% accuracy."
        ),
    )


# ── 4. TRAIN-TEST CONTAMINATION ──────────────────────────────────────────────

def make_train_test_contamination(n: int = 400, seed: int = 42) -> DatasetResult:
    """Preprocessing (scaling/encoding) fitted on full data before split."""
    _set_seed(seed)
    x1 = np.random.normal(0, 10, n)
    x2 = np.random.normal(100, 50, n)
    noise = np.random.normal(0, 1, n)
    raw_score = 0.3 * x1 + 0.1 * x2 + noise
    target = (raw_score > np.median(raw_score)).astype(int)

    # Contaminated: features are already standardised using ALL data
    x1_scaled = (x1 - x1.mean()) / x1.std()
    x2_scaled = (x2 - x2.mean()) / x2.std()

    df = pd.DataFrame({
        "feature_a_scaled": x1_scaled.round(6),   # ← fitted on full dataset
        "feature_b_scaled": x2_scaled.round(6),   # ← same issue
        "category": np.random.choice(["A", "B", "C"], n),
        "label": target,
    })

    return DatasetResult(
        name="train_test_contamination",
        df=df,
        target_col="label",
        issue_type="preprocessing_leakage",
        ground_truth={
            "has_leakage": True,
            "leaky_features": ["feature_a_scaled", "feature_b_scaled"],
            "mechanism": "StandardScaler fitted on full dataset before train/test split",
            "fix": "fit scaler only on training fold inside a Pipeline",
        },
        description=(
            "Binary classification where feature scaling used the entire dataset — "
            "test statistics leak into training."
        ),
    )


# ── 5. OVERFITTING VIA LOW SIGNAL ────────────────────────────────────────────

def make_overfit_trap(n: int = 150, seed: int = 42) -> DatasetResult:
    """Very few samples, many noise features — model memorises training data."""
    _set_seed(seed)
    n_signal = 2
    n_noise = 50
    signal = np.random.normal(0, 1, (n, n_signal))
    noise = np.random.normal(0, 1, (n, n_noise))
    target = (signal[:, 0] - signal[:, 1] > 0).astype(int)

    cols = {f"signal_{i}": signal[:, i] for i in range(n_signal)}
    cols.update({f"noise_{i:02d}": noise[:, i] for i in range(n_noise)})
    cols["target"] = target
    df = pd.DataFrame(cols)

    return DatasetResult(
        name="overfit_trap",
        df=df,
        target_col="target",
        issue_type="overfitting",
        ground_truth={
            "n_samples": n,
            "n_signal_features": n_signal,
            "n_noise_features": n_noise,
            "p_to_n_ratio": f"{n_noise + n_signal}/{n}",
            "expected_train_acc": ">95%",
            "expected_test_acc": "~55-65%",
            "fix": "regularisation, feature selection, more data",
        },
        description=(
            "150 samples, 52 features (50 pure noise). "
            "High train accuracy will not generalise."
        ),
    )


# ── REGISTRY ─────────────────────────────────────────────────────────────────

GENERATORS = {
    "target_leakage": make_target_leakage,
    "temporal_leakage": make_temporal_leakage,
    "class_imbalance": make_class_imbalance,
    "train_test_contamination": make_train_test_contamination,
    "overfit_trap": make_overfit_trap,
}


def generate_all(seed: int = 42) -> Dict[str, DatasetResult]:
    return {name: fn(seed=seed) for name, fn in GENERATORS.items()}


def dataset_to_prompt_context(result: DatasetResult) -> str:
    """Dataset summary for the LLM prompt — includes signals for each pitfall type."""
    import pandas as pd
    df = result.df
    target = result.target_col

    dtypes = df.dtypes.to_string()
    head = df.head(5).to_string(index=False)
    stats = df.describe(include="all").round(4).to_string()

    # Target distribution — critical for imbalance detection
    target_counts = df[target].value_counts().to_string()
    target_mean = df[target].mean()

    # Correlation of numeric features with target — exposes target leakage
    numeric = df.select_dtypes(include="number")
    if target in numeric.columns and len(numeric.columns) > 1:
        corr = numeric.corr()[target].drop(target).round(4).sort_values(key=abs, ascending=False)
        corr_str = corr.to_string()
    else:
        corr_str = "n/a"

    # Feature-to-row ratio — overfitting signal
    n_rows, n_cols = df.shape
    n_features = n_cols - 1
    p_n_ratio = round(n_features / n_rows, 4)

    # Date columns present?
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    return f"""DATASET: {result.name}
DESCRIPTION: {result.description}
TARGET: {target}
SHAPE: {n_rows} rows x {n_cols} columns
FEATURE-TO-ROW RATIO (p/n): {p_n_ratio}  [>0.2 = overfitting risk]
DATE COLUMNS: {date_cols if date_cols else "none"}

COLUMN DTYPES:
{dtypes}

TARGET DISTRIBUTION ({target}):
{target_counts}
Target mean: {round(target_mean, 4)}  [near 0 or 1 = class imbalance]

FEATURE CORRELATIONS WITH TARGET (numeric only):
{corr_str}
[High correlation (>0.7 or <-0.7) may indicate target leakage]

SAMPLE (first 5 rows):
{head}

DESCRIPTIVE STATISTICS:
{stats}
"""


if __name__ == "__main__":
    for name, result in generate_all().items():
        print(f"\n{'─'*60}")
        print(f"[{result.issue_type.upper()}] {result.name}")
        print(f"Shape: {result.df.shape} | Target: {result.target_col}")
        print(f"Ground truth: {json.dumps(result.ground_truth, indent=2)}")