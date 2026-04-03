"""
Dashboard API
Flask backend serving experiment results to the frontend dashboard.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from dotenv import load_dotenv

SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent
load_dotenv(ROOT_DIR / ".env")

app = Flask(__name__)
EXPERIMENT_DIR = ROOT_DIR / os.getenv("EXPERIMENT_DIR", "experiments")
EXPERIMENT_DIR.mkdir(exist_ok=True)


def _load_latest() -> dict | None:
    files = sorted(EXPERIMENT_DIR.glob("run_*.json"), reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text())


def _load_all_runs() -> list[dict]:
    files = sorted(EXPERIMENT_DIR.glob("run_*.json"), reverse=True)
    runs = []
    for f in files[:10]:
        try:
            data = json.loads(f.read_text())
            runs.append({
                "run_id": data["run_id"],
                "timestamp": data.get("timestamp", ""),
                "total_tokens": data.get("total_tokens", 0),
                "n_results": len(data.get("results", [])),
                "avg_score": _avg_score(data.get("results", [])),
            })
        except Exception:
            continue
    return runs


def _avg_score(results: list) -> float:
    valid = [r for r in results if r.get("scores")]
    if not valid:
        return 0.0
    return round(sum(r["scores"].get("overall", 0) for r in valid) / len(valid), 4)


@app.route("/")
def index():
    return send_file(ROOT_DIR / "dashboard.html")


@app.route("/api/status")
def status():
    return jsonify({"status": "ok", "experiment_dir": str(EXPERIMENT_DIR)})


@app.route("/api/runs")
def list_runs():
    return jsonify(_load_all_runs())


@app.route("/api/latest")
def latest():
    data = _load_latest()
    if not data:
        return jsonify({"error": "No experiments run yet. POST /api/run to start."}), 404
    return jsonify(data)


@app.route("/api/run", methods=["POST"])
def trigger_run():
    body = request.get_json(silent=True) or {}
    seed = int(body.get("seed", 42))
    datasets = body.get("datasets")

    cmd = [sys.executable, str(SRC_DIR / "runner.py"), "--seed", str(seed)]
    if datasets:
        cmd += ["--datasets"] + datasets

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(SRC_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return jsonify({"status": "started", "pid": proc.pid, "seed": seed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/<name>")
def dataset_detail(name: str):
    data = _load_latest()
    if not data:
        return jsonify({"error": "no data"}), 404
    for r in data.get("results", []):
        if r["dataset"] == name:
            return jsonify(r)
    return jsonify({"error": "dataset not found"}), 404


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    print(f"Dashboard API running → http://localhost:{port}")
    app.run(debug=True, port=port)