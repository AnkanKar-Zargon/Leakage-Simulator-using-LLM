"""
Experiment Runner
Orchestrates dataset generation → LLM evaluation → scoring → MLflow logging.
Results saved to experiments/ as JSON for dashboard consumption.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import mlflow
from dotenv import load_dotenv

# Always resolve paths relative to this file so subprocess invocation works
SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
load_dotenv(ROOT_DIR / ".env")

from datasets import generate_all, dataset_to_prompt_context
from evaluator import evaluate, score

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT_DIR = ROOT_DIR / os.getenv("EXPERIMENT_DIR", "experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"file://{EXPERIMENT_DIR.absolute()}/mlruns")
mlflow.set_experiment("leakage_simulator")


def run_experiment(seed: int = 42, datasets: list[str] | None = None) -> dict:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"Starting experiment run: {run_id}")

    all_datasets = generate_all(seed=seed)
    if datasets:
        all_datasets = {k: v for k, v in all_datasets.items() if k in datasets}

    results = []
    total_tokens = 0

    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("seed", seed)
        mlflow.log_param("model", os.getenv("LLM_MODEL", "qwen/qwen3-32b"))
        mlflow.log_param("n_datasets", len(all_datasets))

        for name, dataset in all_datasets.items():
            log.info(f"  → Evaluating: {name}")
            context = dataset_to_prompt_context(dataset)

            try:
                eval_result = evaluate(name, context)
            except Exception as e:
                log.error(f"    LLM call failed for {name}: {e}")
                results.append({"dataset": name, "error": str(e), "scores": {}})
                continue

            scores = score(eval_result, dataset.ground_truth)
            total_tokens += eval_result.tokens_used

            entry = {
                "dataset": name,
                "issue_type": dataset.issue_type,
                "description": dataset.description,
                "ground_truth": dataset.ground_truth,
                "llm_response": eval_result.parsed,
                "parse_error": eval_result.parse_error,
                "scores": scores,
                "latency_s": eval_result.latency_s,
                "tokens_used": eval_result.tokens_used,
            }
            results.append(entry)

            prefix = name[:20]
            mlflow.log_metric(f"{prefix}_overall", scores.get("overall", 0))
            mlflow.log_metric(f"{prefix}_detected", scores.get("issue_detected", 0))
            mlflow.log_metric(f"{prefix}_latency", eval_result.latency_s)

            log.info(
                f"    Scores → overall={scores.get('overall'):.3f}  "
                f"detected={scores.get('issue_detected')}  "
                f"latency={eval_result.latency_s}s"
            )
            time.sleep(0.5)

        valid = [r for r in results if r.get("scores")]
        if valid:
            avg_overall = sum(r["scores"].get("overall", 0) for r in valid) / len(valid)
            avg_detected = sum(r["scores"].get("issue_detected", 0) for r in valid) / len(valid)
            mlflow.log_metric("avg_overall_score", round(avg_overall, 4))
            mlflow.log_metric("avg_detection_rate", round(avg_detected, 4))
            mlflow.log_metric("total_tokens", total_tokens)

        mlflow_run_id = mlflow.active_run().info.run_id

    out_path = EXPERIMENT_DIR / f"run_{run_id}.json"
    payload = {
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "timestamp": run_id,
        "seed": seed,
        "total_tokens": total_tokens,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(f"Results saved → {out_path}")
    log.info(f"Total tokens used: {total_tokens}")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run leakage detection experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="*", help="Subset of dataset names to run")
    args = parser.parse_args()
    run_experiment(seed=args.seed, datasets=args.datasets)