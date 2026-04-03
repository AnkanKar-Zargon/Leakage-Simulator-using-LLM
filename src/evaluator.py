"""
LLM Evaluator
Sends dataset context to LLM (here I used Qwen3-32b as default) and evaluates detection + reasoning quality.
"""

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

SRC_DIR = __import__('pathlib').Path(__file__).parent
load_dotenv(SRC_DIR.parent / ".env")

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("API_KEY", "")
        if not key:
            raise EnvironmentError("API_KEY not set in .env")
        _client = Groq(api_key=key)
    return _client


MODEL = os.getenv("MODEL", "qwen/qwen3-32b")

SYSTEM_PROMPT = """\
You are a senior ML engineer and data scientist specialising in diagnosing dataset quality issues.

Your job: analyse a dataset and detect ALL of the following issues if present:

1. TARGET LEAKAGE — a feature is computed from or strongly determined by the target variable.
   Signs: a feature correlates suspiciously with the target (e.g. approval_score for loan_approved),
   or its name implies it is derived from the outcome.

2. TEMPORAL LEAKAGE — a feature uses future data. Signs: rolling/lagging aggregates computed
   without respecting time order, or the dataset has a date column but aggregates look symmetric.

3. CLASS IMBALANCE — the target is heavily skewed. Signs: target mean close to 0 or 1,
   target std very low, or count of minority class is tiny. Always check describe() stats for the target.

4. PREPROCESSING LEAKAGE — features are already scaled/encoded using the full dataset before split.
   Signs: features with suspiciously perfect mean=0, std=1, or names ending in _scaled, _encoded, _norm.

5. OVERFITTING RISK — far more features than samples. Signs: p/n ratio > 0.2, many noise-like
   feature names (noise_00, noise_01...), or very few rows relative to columns.

Think step by step. Be specific — name the exact feature(s) causing each issue and explain WHY.

Respond ONLY with valid JSON, no markdown fences, matching this exact schema:
{
  "detected_issues": ["list of issue names found"],
  "leaky_features": ["exact feature column names suspected, or []"],
  "issue_type": "target_leakage | temporal_leakage | class_imbalance | preprocessing_leakage | overfitting | none",
  "severity": "low | medium | high | critical",
  "reasoning": "Step-by-step diagnosis. Name features. Cite statistics from the data.",
  "recommended_fixes": ["concrete actionable fixes"],
  "confidence": 0.0
}
If multiple issues exist, set issue_type to the most severe one. confidence is 0.0-1.0.\
"""

USER_TEMPLATE = """\
Analyse this dataset carefully for ML issues. Use ALL the statistics provided.

{context}

Key things to check:
- Does any feature have a suspiciously high correlation with the target? (target leakage)
- Are there scaled features (mean≈0, std≈1) already applied? (preprocessing leakage)
- What is the target column distribution — is it heavily imbalanced?
- Is there a date/time column? Could any aggregate feature use future values?
- How does the number of features compare to number of rows? (overfitting risk)

Name exact column names in your answer.\
"""


@dataclass
class EvalResult:
    dataset_name: str
    raw_response: str
    parsed: dict = field(default_factory=dict)
    latency_s: float = 0.0
    tokens_used: int = 0
    parse_error: Optional[str] = None


def evaluate(dataset_name: str, context: str) -> EvalResult:
    client = _get_client()
    prompt = USER_TEMPLATE.format(context=context)

    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,   # lower = more deterministic diagnosis
        max_tokens=900,
    )
    latency = round(time.time() - t0, 3)

    raw = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 0

    result = EvalResult(
        dataset_name=dataset_name,
        raw_response=raw,
        latency_s=latency,
        tokens_used=tokens,
    )

    clean = raw.strip("`")
    if clean.startswith("json"):
        clean = clean[4:]
    clean = clean.strip()

    try:
        result.parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        result.parse_error = str(e)

    return result


# ── SCORING ──────────────────────────────────────────────────────────────────

ISSUE_TYPE_MAP = {
    "target_leakage": "target_leakage",
    "temporal_leakage": "temporal_leakage",
    "class_imbalance": "class_imbalance",
    "train_test_contamination": "preprocessing_leakage",
    "overfit_trap": "overfitting",
}


def score(eval_result: EvalResult, ground_truth: dict) -> dict:
    p = eval_result.parsed
    if not p:
        return {"issue_detected": 0, "feature_precision": 0,
                "reasoning_score": 0, "overall": 0, "notes": "parse_failed"}

    dataset_name = eval_result.dataset_name
    expected_canonical = ISSUE_TYPE_MAP.get(dataset_name, "")
    predicted_type = p.get("issue_type", "none")
    issue_detected = int(predicted_type == expected_canonical)

    # Also credit if the issue appears in detected_issues list even if primary type differs
    if not issue_detected:
        detected_list = [d.lower().replace(" ", "_") for d in p.get("detected_issues", [])]
        if any(expected_canonical in d or d in expected_canonical for d in detected_list):
            issue_detected = 1

    leaky_truth = set()
    for key in ("leaky_feature", "leaky_features"):
        val = ground_truth.get(key)
        if val:
            leaky_truth.update([val] if isinstance(val, str) else val)

    predicted_leaky = set(p.get("leaky_features", []))
    if leaky_truth:
        tp = len(leaky_truth & predicted_leaky)
        fp = len(predicted_leaky - leaky_truth)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / len(leaky_truth) if leaky_truth else 0.0
        feat_score = (2 * precision * recall / (precision + recall)
                      if (precision + recall) else 0.0)
    else:
        feat_score = 1.0 if not predicted_leaky else 0.5

    reasoning = p.get("reasoning", "").lower()
    keywords = [
        "leakage", "overfitting", "imbalance", "temporal", "split",
        "train", "test", "future", "scale", "contamination", "fraud",
        "precision", "recall", "f1", "pipeline", "shuffle", "target",
        "derived", "correlated", "ratio", "minority",
    ]
    kw_hits = sum(1 for k in keywords if k in reasoning)
    reasoning_score = min(1.0, kw_hits / 7)

    overall = round(0.5 * issue_detected + 0.25 * feat_score + 0.25 * reasoning_score, 4)

    return {
        "issue_detected": issue_detected,
        "feature_precision": round(feat_score, 4),
        "reasoning_score": round(reasoning_score, 4),
        "overall": overall,
        "predicted_issue_type": predicted_type,
        "expected_issue_type": expected_canonical,
        "confidence_reported": p.get("confidence", None),
    }