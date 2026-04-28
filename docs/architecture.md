# CineBandit — System Architecture

**DA5402 · MLOps | Chirag · DA25M008**

---

## Architecture Overview

CineBandit is deployed as a set of 8 Docker containers (plus a one-shot init container) connected through a shared Docker network (`cinebandit_cinebandit`). Each container has a single responsibility. The system is divided into four logical layers.

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER LAYER                                                         │
│                                                                     │
│   ┌───────────────────────────────────────┐                         │
│   │  Streamlit Frontend  :8501            │                         │
│   │  Recommendations · Feedback · Monitor │                         │
│   └─────────────────┬─────────────────────┘                         │
└─────────────────────┼───────────────────────────────────────────────┘
                      │ HTTP
┌─────────────────────┼───────────────────────────────────────────────┐
│  SERVING LAYER      │                                               │
│                     ▼                                               │
│   ┌──────────────────────────────┐   ┌───────────────────────────┐  │
│   │  FastAPI  :8000              │──▶│  PostgreSQL  :5432        │  │
│   │  /recommend  /feedback       │   │  Feedback log · state     │  │
│   │  /health     /metrics        │   └───────────────────────────┘  │
│   └──────┬──────────────┬────────┘                                  │
└──────────┼──────────────┼──────────────────────────────────────────-┘
           │ model calls  │ metrics
┌──────────┼──────────────┼──────────────────────────────────────────-┐
│  ML CORE │              │                                           │
│          ▼              ▼                                           │
│   ┌─────────────┐   ┌──────────────────────────────────────────┐   │
│   │  LinUCB     │   │  Prometheus  :9090  →  Grafana  :3000    │   │
│   │  Model      │   │  Scrapes /metrics every 15 s             │   │
│   │  (in-memory │   └──────────────────────────────────────────┘   │
│   │   in API)   │                                                   │
│   └─────────────┘                                                   │
└────────────────────────────────────────────────────────────────────-┘
┌────────────────────────────────────────────────────────────────────-┐
│  MLOPS LAYER                                                        │
│                                                                     │
│  ┌──────────────┐  ┌────────────────────────────────┐  ┌─────────┐ │
│  │  MLflow :5000│  │  Airflow Scheduler + Web :8080 │  │   DVC   │ │
│  │  Experiments │  │  ingestion · drift · retrain   │  │ Pipeline│ │
│  │  Registry    │  └────────────────────────────────┘  └─────────┘ │
│  └──────────────┘                                                   │
└────────────────────────────────────────────────────────────────────-┘
```

---

## Container Inventory

| Container | Image | Port | Role |
|---|---|---|---|
| `cinebandit_frontend` | `cinebandit-frontend` | 8501 | Streamlit UI |
| `cinebandit_api` | `cinebandit-api` | 8000 | FastAPI inference server |
| `cinebandit_postgres` | `postgres:15-alpine` | 5432 | Feedback & state persistence |
| `cinebandit_mlflow` | `ghcr.io/mlflow/mlflow:v2.22.0` | 5000 | Experiment tracking & model registry |
| `cinebandit_airflow_webserver` | `cinebandit-airflow-webserver` | 8080 | Airflow UI |
| `cinebandit_airflow_scheduler` | `cinebandit-airflow-scheduler` | — | DAG scheduler |
| `cinebandit_prometheus` | `prom/prometheus:v2.51.0` | 9090 | Metrics collection |
| `cinebandit_grafana` | `grafana/grafana:10.4.0` | 3000 | Metrics visualisation |

The `cinebandit_airflow_init` container is a one-shot initialisation job that sets up the Airflow metadata database and exits.

---

## Block Descriptions

### Streamlit Frontend (`cinebandit_frontend`)

The primary user interface. It communicates with the FastAPI server over HTTP and provides three views: a Recommendations screen where users browse suggestions and submit feedback, a Pipeline Console showing Airflow and MLflow status, and a Monitoring Dashboard displaying live Prometheus metrics.

### FastAPI (`cinebandit_api`)

The core inference server, implemented in `src/api/main.py` with routes split across `recommend.py`, `feedback.py`, and `system.py`. It loads the LinUCB model into memory on startup (copied from the host via `docker cp`), handles recommendation requests in under 25 ms (p95), and performs in-process model updates on every feedback call using the Sherman-Morrison formula. It also exposes a Prometheus-compatible `/metrics` endpoint.

### LinUCB Model (in-process within the API)

The recommendation engine. Implemented in `src/bandit/linucb.py`, it maintains one `LinUCBArm` per movie (1,682 arms). Each arm holds an inverse matrix `A_inv` and reward vector `b`. On a `/recommend` call it computes UCB scores for all candidate arms and returns the top-N. On a `/feedback` call it performs a rank-1 Sherman-Morrison update — no matrix inversion required, O(d²) per update.

### PostgreSQL (`cinebandit_postgres`)

Stores user feedback events and any persistent system state. The API writes a row for every `/feedback` call. Used by Airflow's drift detection DAG to read recent feedback and compare genre distributions against the stored baseline.

### MLflow (`cinebandit_mlflow`)

Tracks every training run via `train_mlflow.py`. Logs hyperparameters (`alpha`), metrics (`train_ctr`, `test_ctr`, `genre_diversity`), evaluation plots, and the serialised model pickle as artifacts. Maintains a Model Registry where the best model is promoted to the `production` alias and made available to the API.

### Airflow (`cinebandit_airflow_scheduler` + `cinebandit_airflow_webserver`)

Orchestrates three DAGs defined in the `dags/` directory:

- **`cinebandit_ingestion`** — loads and validates the MovieLens dataset, writes the genre baseline to `data/baselines/genre_baseline.json`.
- **`cinebandit_drift_detection`** — computes KL divergence between the current genre distribution of liked movies and the baseline. Triggers the retraining DAG if the divergence exceeds the threshold (0.3).
- **`cinebandit_retraining`** — re-runs the DVC pipeline and registers the new model in MLflow.

### Prometheus (`cinebandit_prometheus`)

Scrapes the API's `/metrics` endpoint every 15 seconds. Collects the following metrics exposed by the API: `cinebandit_recommendation_requests_total`, `cinebandit_feedback_total`, `cinebandit_rolling_ctr`, `cinebandit_drift_score`, `cinebandit_model_timesteps`, `cinebandit_recommendation_latency_seconds` (histogram), and `cinebandit_arms_seen_total`.

### Grafana (`cinebandit_grafana`)

Reads from Prometheus via the provisioned datasource (`http://prometheus:9090`). The `CineBandit — Live Monitoring` dashboard (UID `cinebandit-main`) contains 11 panels: Rolling CTR, Model Timestep, Drift Score, Arms Seen, Requests/s, Latency (p50/p95/p99), Feedback Breakdown, Cumulative Reward, Retraining Runs, API Errors, and Model Alpha.

### DVC Pipeline

Version-controls data and models. The three-stage pipeline (`prepare → train → evaluate`) is defined in `dvc.yaml` with parameters in `params.yaml`. Stages are cached by input hash — changing only `alpha` causes only the `train` and `evaluate` stages to re-run, while `prepare` is skipped. Metrics are tracked in `metrics/train_metrics.json` and `metrics/eval_metrics.json`.

---

## Data Flow

### Recommendation request

```
User → Streamlit → POST /recommend → LinUCB (score all arms) → return top-N → Streamlit renders list
```

### Feedback loop

```
User clicks "Like" → Streamlit → POST /feedback → LinUCB (Sherman-Morrison update) → Prometheus gauge updated → PostgreSQL row written
```

### Drift response

```
Airflow drift_detection DAG → reads recent feedback from DB → computes KL divergence
  → if KL > 0.3: trigger retraining DAG → DVC repro → MLflow run logged → model registered → API reloads model
```

### Monitoring

```
FastAPI /metrics → Prometheus scrape (every 15s) → Grafana query → dashboard panels refresh
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| LinUCB over collaborative filtering | Online updates without full retraining; handles cold-start via UCB bonus |
| Sherman-Morrison incremental update | O(d²) per feedback call vs O(d³) for full inversion — enables synchronous in-request updates |
| In-process model (not a separate container) | Eliminates network hop on the hot path; p95 latency < 25 ms vs target 200 ms |
| Replay evaluation | Unbiased offline evaluation using only interactions where the logged action matches the bandit's recommendation |
| KL divergence for drift | Sensitive to distribution shape changes, not just mean shifts; appropriate for genre proportion drift |
| DVC caching | Changing `alpha` does not re-run data preparation — saves ~90 s on every hyperparameter sweep |