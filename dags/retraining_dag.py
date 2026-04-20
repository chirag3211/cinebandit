"""
DAG 3: Retraining Pipeline
Trigger: triggered by drift_detection_dag OR manually via Airflow UI

Tasks:
  1. prepare_training_data — validate data is fresh and baseline exists
  2. train_model           — run train_mlflow.py with current best alpha
  3. evaluate_model        — compare challenger vs champion CTR
  4. promote_or_reject     — promote to production if better, else archive
  5. reload_api_model      — hot-reload the new model into the FastAPI server
  6. notify_completion     — log final status to MLflow

This is the core continuous learning loop.
"""

from datetime import datetime, timedelta
import logging
import os
import subprocess
import json
import requests
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# ── Default args ──────────────────────────────────────────────────────────────

default_args = {
    "owner":            "cinebandit",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

# ── Config ────────────────────────────────────────────────────────────────────

API_URL       = os.environ.get("API_URL", "http://api:8000")
MLFLOW_URI    = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
OUTPUTS_DIR   = Path("/opt/airflow/outputs")
DATA_DIR      = Path("/opt/airflow/data/raw/ml-100k")
BASELINE_PATH = Path("/opt/airflow/data/baselines/genre_baseline.json")

# Best alpha from hyperparameter tuning — update this after running tune_alpha
BEST_ALPHA    = float(os.environ.get("CINEBANDIT_ALPHA", "0.1"))
MODEL_NAME    = "CineBandit"


# ── Task functions ────────────────────────────────────────────────────────────

def prepare_training_data(**context) -> dict:
    """
    Validate all prerequisites before training:
      - Data files exist
      - Baseline statistics exist
      - Output directory is writable
    """
    errors = []

    for fname in ["u.data", "u.item", "u.user"]:
        if not (DATA_DIR / fname).exists():
            errors.append(f"Missing data file: {DATA_DIR / fname}")

    if not BASELINE_PATH.exists():
        logging.warning(
            "[retrain] Baseline not found — will be created during training"
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if errors:
        raise FileNotFoundError("\n".join(errors))

    trigger_reason = context.get("dag_run") and context["dag_run"].conf.get(
        "reason", "manual"
    )
    logging.info(f"[retrain] Trigger reason: {trigger_reason}")
    logging.info("[retrain] Prerequisites validated ✅")

    context["ti"].xcom_push(key="trigger_reason", value=trigger_reason)
    return {"status": "ready", "trigger_reason": trigger_reason}


def train_model(**context) -> dict:
    """
    Run the MLflow-instrumented training script.
    Uses subprocess so it runs in the same Python environment as the DAG.
    """
    import sys

    trigger_reason = context["ti"].xcom_pull(
        key="trigger_reason", task_ids="prepare_training_data"
    ) or "manual"

    logging.info(f"[retrain] Starting training with alpha={BEST_ALPHA} ...")

    cmd = [
        sys.executable,
        "/opt/airflow/train_mlflow.py",
        "--alpha", str(BEST_ALPHA),
        "--no-promote",   # We handle promotion in promote_or_reject task
    ]

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW_URI
    env["PYTHONPATH"] = "/opt/airflow"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd="/opt/airflow"
    )

    logging.info(f"[retrain] Training stdout:\n{result.stdout[-3000:]}")

    if result.returncode != 0:
        logging.error(f"[retrain] Training failed:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Training script failed with exit code {result.returncode}")

    # Parse run_id and test_ctr from stdout
    run_id = None
    test_ctr = None
    for line in result.stdout.split("\n"):
        if "Run ID" in line:
            run_id = line.split(":")[-1].strip()
        if "Test CTR" in line:
            try:
                test_ctr = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    if not run_id:
        raise RuntimeError("Could not parse run_id from training output")

    logging.info(f"[retrain] Training complete: run_id={run_id} test_ctr={test_ctr}")

    context["ti"].xcom_push(key="run_id",   value=run_id)
    context["ti"].xcom_push(key="test_ctr", value=test_ctr)

    return {"run_id": run_id, "test_ctr": test_ctr}


def promote_or_reject(**context) -> str:
    """
    Compare challenger (new model) vs champion (current production model).
    Returns task_id for branching.
    """
    import mlflow

    run_id   = context["ti"].xcom_pull(key="run_id",   task_ids="train_model")
    test_ctr = context["ti"].xcom_pull(key="test_ctr", task_ids="train_model")

    if not run_id:
        logging.error("[retrain] No run_id found — skipping promotion")
        return "reject_model"

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    # Get current champion CTR
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "production")
        champion_run = client.get_run(mv.run_id)
        champion_ctr = float(champion_run.data.metrics.get("test_ctr", 0.0))
        champion_version = mv.version
    except Exception:
        champion_ctr = 0.0
        champion_version = None

    logging.info(
        f"[retrain] Challenger CTR={test_ctr:.4f} vs "
        f"Champion CTR={champion_ctr:.4f} (v{champion_version})"
    )

    context["ti"].xcom_push(key="champion_ctr",     value=champion_ctr)
    context["ti"].xcom_push(key="champion_version", value=champion_version)

    if test_ctr is not None and test_ctr > champion_ctr:
        logging.info("[retrain] ✅ Challenger wins — promoting to production")
        return "promote_model"
    else:
        logging.info("[retrain] ❌ Challenger does not beat champion — rejecting")
        return "reject_model"


def promote_model(**context) -> None:
    """Register and promote the challenger model to production."""
    import mlflow

    run_id   = context["ti"].xcom_pull(key="run_id",   task_ids="train_model")
    test_ctr = context["ti"].xcom_pull(key="test_ctr", task_ids="train_model")

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    # Register model
    model_uri = f"runs:/{run_id}/model"
    try:
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        version = mv.version
    except Exception as e:
        logging.warning(f"[retrain] Model registration failed: {e}")
        # Try to get existing version
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        run_versions = [v for v in versions if v.run_id == run_id]
        if run_versions:
            version = run_versions[0].version
        else:
            raise

    # Set production alias
    client.set_registered_model_alias(
        name=MODEL_NAME, alias="production", version=version
    )

    logging.info(
        f"[retrain] ✅ Model v{version} promoted to production "
        f"(CTR={test_ctr:.4f})"
    )
    context["ti"].xcom_push(key="promoted_version", value=version)


def reject_model(**context) -> None:
    """Log rejection reason — no model changes made."""
    test_ctr      = context["ti"].xcom_pull(key="test_ctr",      task_ids="train_model")
    champion_ctr  = context["ti"].xcom_pull(key="champion_ctr",  task_ids="promote_or_reject")
    logging.info(
        f"[retrain] Model rejected. "
        f"Challenger CTR={test_ctr} ≤ Champion CTR={champion_ctr}. "
        f"Production model unchanged."
    )


def reload_api_model(**context) -> None:
    """
    Hot-reload the new production model into the FastAPI server.
    Calls POST /model/reload — no restart needed.
    """
    try:
        resp = requests.post(f"{API_URL}/model/reload", timeout=30)
        resp.raise_for_status()
        result = resp.json()
        logging.info(
            f"[retrain] API model reloaded: timesteps={result.get('timesteps')}"
        )
    except Exception as e:
        logging.warning(
            f"[retrain] Could not reload API model: {e}. "
            "Manual reload may be required."
        )


def notify_completion(**context) -> None:
    """Log final retraining status to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)

        test_ctr     = context["ti"].xcom_pull(key="test_ctr",      task_ids="train_model") or 0
        champion_ctr = context["ti"].xcom_pull(key="champion_ctr",  task_ids="promote_or_reject") or 0
        run_id       = context["ti"].xcom_pull(key="run_id",        task_ids="train_model")
        promoted     = context["ti"].xcom_pull(key="promoted_version", task_ids="promote_model")

        with mlflow.start_run(run_name="retraining_pipeline"):
            mlflow.set_tag("dag", "retraining")
            mlflow.set_tag("trigger_run_id", run_id or "unknown")
            mlflow.set_tag("promoted", str(promoted is not None))
            mlflow.log_metrics({
                "challenger_ctr": test_ctr,
                "champion_ctr":   champion_ctr,
                "promoted":       1.0 if promoted else 0.0,
            })
        logging.info("[retrain] Retraining pipeline complete ✅")
    except Exception as e:
        logging.warning(f"[retrain] MLflow notification failed: {e}")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="cinebandit_retraining",
    description="Model retraining — triggered by drift detection or manually",
    default_args=default_args,
    schedule_interval=None,    # Only triggered, never scheduled
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["cinebandit", "training", "mlops"],
) as dag:

    t1 = PythonOperator(
        task_id="prepare_training_data",
        python_callable=prepare_training_data,
    )

    t2 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        execution_timeout=timedelta(hours=2),
    )

    t3 = BranchPythonOperator(
        task_id="promote_or_reject",
        python_callable=promote_or_reject,
    )

    t4_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    t4_reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model,
    )

    t5 = PythonOperator(
        task_id="reload_api_model",
        python_callable=reload_api_model,
        trigger_rule="none_failed_min_one_success",
    )

    t6 = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_completion,
        trigger_rule="none_failed_min_one_success",
    )

    # Pipeline
    t1 >> t2 >> t3 >> [t4_promote, t4_reject] >> t5 >> t6