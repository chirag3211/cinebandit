"""
DAG 2: Drift Detection Pipeline
Schedule: every 6 hours

Tasks:
  1. load_recent_interactions  — load last N interactions from logs
  2. compute_drift_score       — KL divergence vs baseline
  3. update_api_drift_score    — POST drift score to FastAPI /drift/update
  4. decide_retraining         — branch: trigger retraining DAG if alert

This DAG is the continuous monitoring heartbeat of CineBandit.
"""

from datetime import datetime, timedelta
import logging
import json
import os
import requests
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator

# ── Default args ──────────────────────────────────────────────────────────────

default_args = {
    "owner":            "cinebandit",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
}

# ── Config ────────────────────────────────────────────────────────────────────

API_URL           = os.environ.get("API_URL", "http://api:8000")
BASELINE_PATH     = Path("/opt/airflow/data/baselines/genre_baseline.json")
DATA_DIR          = Path("/opt/airflow/data/raw/ml-100k")
DRIFT_THRESHOLD   = 0.3
# Use last 5000 interactions for drift detection window
DETECTION_WINDOW  = 5000


# ── Task functions ────────────────────────────────────────────────────────────

def load_recent_interactions(**context) -> dict:
    """
    Load the most recent N interactions.
    In production this would query a live interactions database.
    For this project we use the last N rows of u.data (sorted by timestamp).
    """
    import sys
    sys.path.insert(0, "/opt/airflow")
    import pandas as pd
    from pathlib import Path as P

    ratings_path = P("/opt/airflow/data/raw/ml-100k/u.data")
    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {ratings_path}. "
            "Run the ingestion DAG first."
        )

    ratings = pd.read_csv(
        ratings_path, sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    ratings["reward"] = (ratings["rating"] >= 4).astype(int)
    ratings = ratings.sort_values("timestamp")

    # Take last DETECTION_WINDOW interactions as "recent"
    recent = ratings.tail(DETECTION_WINDOW).reset_index(drop=True)

    logging.info(
        f"[drift] Loaded {len(recent):,} recent interactions "
        f"(window={DETECTION_WINDOW})"
    )

    context["ti"].xcom_push(key="n_recent", value=len(recent))
    return {"n_recent": len(recent)}


def compute_drift_score(**context) -> dict:
    """
    Compute KL divergence between recent interactions and baseline.
    Pushes drift_score and alert flag to XCom.
    """
    import sys
    sys.path.insert(0, "/opt/airflow")
    import pandas as pd
    from pathlib import Path as P
    from src.drift.detector import detect_drift
    from src.data.loader import load_movies

    # Reload recent ratings
    ratings_path = P("/opt/airflow/data/raw/ml-100k/u.data")
    ratings = pd.read_csv(
        ratings_path, sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    ratings["reward"] = (ratings["rating"] >= 4).astype(int)
    recent = ratings.sort_values("timestamp").tail(DETECTION_WINDOW).reset_index(drop=True)

    movies = load_movies(P("/opt/airflow/data/raw/ml-100k"))

    drift_score, alert, report = detect_drift(
        current_ratings=recent,
        movies=movies,
        baseline_path=BASELINE_PATH,
        threshold=DRIFT_THRESHOLD,
    )

    logging.info(
        f"[drift] Score={drift_score:.4f} | "
        f"Alert={'YES' if alert else 'NO'} | "
        f"Threshold={DRIFT_THRESHOLD}"
    )

    if alert:
        logging.warning(
            f"[drift] 🚨 DRIFT DETECTED! KL={drift_score:.4f} > {DRIFT_THRESHOLD}"
        )
        top_shifts = report.get("top_genre_shifts", {})
        logging.warning(f"[drift] Top genre shifts: {top_shifts}")

    context["ti"].xcom_push(key="drift_score", value=drift_score)
    context["ti"].xcom_push(key="drift_alert", value=alert)
    context["ti"].xcom_push(key="drift_report", value=json.dumps(report))

    return {"drift_score": drift_score, "alert": alert}


def update_api_drift_score(**context) -> None:
    """Push the drift score to the FastAPI /drift/update endpoint."""
    drift_score = context["ti"].xcom_pull(
        key="drift_score", task_ids="compute_drift_score"
    ) or 0.0

    try:
        resp = requests.post(
            f"{API_URL}/drift/update",
            params={"score": drift_score},
            timeout=10
        )
        resp.raise_for_status()
        logging.info(
            f"[drift] API drift score updated: {drift_score:.4f} → "
            f"{resp.json()}"
        )
    except Exception as e:
        logging.warning(f"[drift] Could not update API drift score: {e}")

    # Also log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        with mlflow.start_run(run_name="drift_detection"):
            mlflow.set_tag("dag", "drift_detection")
            mlflow.set_tag("run_date", str(datetime.now().date()))
            mlflow.log_metric("drift_score", drift_score)
            mlflow.log_metric(
                "drift_alert",
                1.0 if context["ti"].xcom_pull(
                    key="drift_alert", task_ids="compute_drift_score"
                ) else 0.0
            )
        logging.info("[drift] Drift score logged to MLflow")
    except Exception as e:
        logging.warning(f"[drift] MLflow logging failed (non-fatal): {e}")


def decide_retraining(**context) -> str:
    """
    Branch operator: trigger retraining DAG if drift alert, else skip.
    Returns the task_id of the next task to run.
    """
    alert = context["ti"].xcom_pull(
        key="drift_alert", task_ids="compute_drift_score"
    )
    drift_score = context["ti"].xcom_pull(
        key="drift_score", task_ids="compute_drift_score"
    ) or 0.0

    if alert:
        logging.info(
            f"[drift] Drift score {drift_score:.4f} > {DRIFT_THRESHOLD} "
            f"→ triggering retraining"
        )
        return "trigger_retraining"
    else:
        logging.info(
            f"[drift] Drift score {drift_score:.4f} ≤ {DRIFT_THRESHOLD} "
            f"→ no retraining needed"
        )
        return "no_retraining_needed"


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="cinebandit_drift_detection",
    description="Drift detection — runs every 6 hours, triggers retraining if needed",
    default_args=default_args,
    schedule_interval="0 */6 * * *",   # every 6 hours
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["cinebandit", "drift", "monitoring"],
) as dag:

    t1 = PythonOperator(
        task_id="load_recent_interactions",
        python_callable=load_recent_interactions,
    )

    t2 = PythonOperator(
        task_id="compute_drift_score",
        python_callable=compute_drift_score,
    )

    t3 = PythonOperator(
        task_id="update_api_drift_score",
        python_callable=update_api_drift_score,
    )

    t4 = BranchPythonOperator(
        task_id="decide_retraining",
        python_callable=decide_retraining,
    )

    t5 = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="cinebandit_retraining",
        conf={"reason": "drift_detected", "trigger": "automatic"},
        wait_for_completion=False,
    )

    t6 = EmptyOperator(task_id="no_retraining_needed")

    # Pipeline
    t1 >> t2 >> t3 >> t4 >> [t5, t6]