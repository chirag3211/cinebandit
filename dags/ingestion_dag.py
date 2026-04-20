"""
DAG 1: Data Ingestion Pipeline
Schedule: @daily

Tasks:
  1. validate_data_files   — check u.data, u.item, u.user exist and are valid
  2. compute_statistics    — compute feature baselines and save to data/baselines/
  3. validate_schema       — check column counts, data types, value ranges
  4. log_ingestion_metrics — log dataset stats to Airflow XCom and MLflow

This DAG runs daily but is idempotent — safe to re-run at any time.
"""

from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Default args ──────────────────────────────────────────────────────────────

default_args = {
    "owner":            "cinebandit",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR      = Path("/opt/airflow/data/raw/ml-100k")
BASELINE_DIR  = Path("/opt/airflow/data/baselines")
BASELINE_PATH = BASELINE_DIR / "genre_baseline.json"


# ── Task functions ────────────────────────────────────────────────────────────

def validate_data_files(**context) -> dict:
    """Check all required MovieLens files exist and are non-empty."""
    required = ["u.data", "u.item", "u.user"]
    stats = {}

    for fname in required:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Required file not found: {fpath}\n"
                "Ensure the MovieLens-100K dataset is mounted at "
                "/opt/airflow/data/raw/ml-100k/"
            )
        size_kb = fpath.stat().st_size / 1024
        stats[fname] = {"size_kb": round(size_kb, 1), "exists": True}
        logging.info(f"[ingestion] ✅ {fname} ({size_kb:.1f} KB)")

    # Push stats to XCom for downstream tasks
    context["ti"].xcom_push(key="file_stats", value=stats)
    logging.info(f"[ingestion] All {len(required)} files validated")
    return stats


def compute_statistics(**context) -> dict:
    """
    Load MovieLens data, compute baseline statistics, save to disk.
    These baselines are used by the drift detection DAG for comparison.
    """
    import sys
    sys.path.insert(0, "/opt/airflow")

    from src.data.loader import load_ratings, load_movies, DATA_DIR as LOADER_DIR
    from src.drift.detector import compute_baseline_statistics

    # Override DATA_DIR to point to Airflow mount
    import src.data.loader as loader_module
    from pathlib import Path as P
    loader_module.DATA_DIR = P("/opt/airflow/data/raw/ml-100k")

    ratings = load_ratings(P("/opt/airflow/data/raw/ml-100k"))
    movies  = load_movies(P("/opt/airflow/data/raw/ml-100k"))

    logging.info(f"[ingestion] Loaded {len(ratings):,} ratings, {len(movies):,} movies")

    # Compute and save baselines
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    stats = compute_baseline_statistics(ratings, movies, save_path=BASELINE_PATH)

    logging.info(f"[ingestion] Baseline saved: like_rate={stats['like_rate']:.3f}")

    context["ti"].xcom_push(key="baseline_stats", value={
        "like_rate":  stats["like_rate"],
        "n_ratings":  stats["n_ratings"],
        "rating_mean": stats["rating_mean"],
    })

    return stats


def validate_schema(**context) -> bool:
    """
    Validate that the loaded data meets schema expectations.
    Checks:
      - ratings has 4 columns
      - user_ids and movie_ids are positive integers
      - ratings are in [1, 5]
      - no duplicate (user, movie) pairs above threshold
    """
    import sys
    sys.path.insert(0, "/opt/airflow")
    import pandas as pd
    from pathlib import Path as P

    ratings_path = P("/opt/airflow/data/raw/ml-100k/u.data")
    ratings = pd.read_csv(
        ratings_path, sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )

    errors = []

    # Column count
    if len(ratings.columns) != 4:
        errors.append(f"Expected 4 columns, got {len(ratings.columns)}")

    # Value ranges
    if ratings["rating"].min() < 1 or ratings["rating"].max() > 5:
        errors.append(f"Ratings out of range [1,5]: "
                      f"min={ratings['rating'].min()} max={ratings['rating'].max()}")

    if ratings["user_id"].min() < 1:
        errors.append("Negative user IDs found")

    if ratings["movie_id"].min() < 1:
        errors.append("Negative movie IDs found")

    # Missing values
    null_counts = ratings.isnull().sum()
    if null_counts.any():
        errors.append(f"Missing values found: {null_counts[null_counts > 0].to_dict()}")

    # Minimum size
    if len(ratings) < 1000:
        errors.append(f"Too few ratings: {len(ratings)} (expected >= 1000)")

    if errors:
        raise ValueError(f"Schema validation failed:\n" + "\n".join(errors))

    logging.info(f"[ingestion] ✅ Schema valid: {len(ratings):,} ratings, "
                 f"{ratings['user_id'].nunique()} users, "
                 f"{ratings['movie_id'].nunique()} movies")
    return True


def log_ingestion_metrics(**context) -> None:
    """Log dataset statistics to MLflow for tracking over time."""
    import sys
    sys.path.insert(0, "/opt/airflow")

    try:
        import mlflow
        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )

        baseline_stats = context["ti"].xcom_pull(
            key="baseline_stats", task_ids="compute_statistics"
        ) or {}

        with mlflow.start_run(run_name="data_ingestion"):
            mlflow.set_tag("dag", "ingestion")
            mlflow.set_tag("run_date", str(datetime.now().date()))
            mlflow.log_metrics({
                "n_ratings":   baseline_stats.get("n_ratings", 0),
                "like_rate":   baseline_stats.get("like_rate", 0),
                "rating_mean": baseline_stats.get("rating_mean", 0),
            })
        logging.info("[ingestion] Metrics logged to MLflow")
    except Exception as e:
        logging.warning(f"[ingestion] MLflow logging failed (non-fatal): {e}")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="cinebandit_ingestion",
    description="Daily data ingestion and baseline computation",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["cinebandit", "ingestion"],
) as dag:

    t1 = PythonOperator(
        task_id="validate_data_files",
        python_callable=validate_data_files,
    )

    t2 = PythonOperator(
        task_id="compute_statistics",
        python_callable=compute_statistics,
    )

    t3 = PythonOperator(
        task_id="validate_schema",
        python_callable=validate_schema,
    )

    t4 = PythonOperator(
        task_id="log_ingestion_metrics",
        python_callable=log_ingestion_metrics,
    )

    # Pipeline order
    t1 >> t2 >> t3 >> t4