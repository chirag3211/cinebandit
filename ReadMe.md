# 🎬 CineBandit — Adaptive Movie Recommendation System
Chirag - DA25M008

> An adaptive movie recommendation system powered by **LinUCB Contextual Bandits**, built with a production-grade **MLOps stack**. The system learns from user feedback in real time, detects preference drift automatically, and retrains itself without human intervention.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running Locally (without Docker)](#running-locally-without-docker)
- [Running with Docker](#running-with-docker)
- [ML Pipeline](#ml-pipeline)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Airflow DAGs](#airflow-dags)
- [DVC Pipeline](#dvc-pipeline)
- [Monitoring](#monitoring)
- [API Reference](#api-reference)
- [Unit Tests](#unit-tests)
- [Service URLs](#service-urls)

---

## Overview

CineBandit recommends movies to users by balancing **exploration** (trying new movies) and **exploitation** (recommending known good ones) using the LinUCB algorithm. Unlike static recommendation models, CineBandit:

- Updates model parameters **incrementally** on every user interaction (no batch retraining needed for real-time adaptation)
- Detects **distribution drift** in user preferences via KL divergence
- **Automatically retrains** when drift is detected via Airflow
- Tracks every experiment in **MLflow** and promotes the best model to production
- Exposes **12 Prometheus metrics** visualised in a Grafana dashboard

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Docker Compose                         │
│                                                              │
│  ┌─────────────┐   REST    ┌──────────────┐                  │
│  │  Streamlit  │──────────▶│   FastAPI    │                  │
│  │  :8501      │◀──────────│   :8000      │                  │
│  └─────────────┘           └──────┬───────┘                  │
│                                   │                          │
│              ┌────────────────────▼──────────────────┐       │
│              │         LinUCB Bandit Model            │       │
│              │    (incremental updates via feedback)  │       │
│              └────────────────────┬───────────────────┘       │
│                                   │                          │
│  ┌──────────────────┐   ┌────────▼───────────────────┐       │
│  │    Airflow       │   │     MLflow Tracking        │       │
│  │  ┌────────────┐  │   │   Experiments + Registry   │       │
│  │  │Ingestion   │  │   └─────────────────────────────┘       │
│  │  │Drift Check │  │                                         │
│  │  │Retraining  │  │   ┌──────────────┐  ┌───────────────┐  │
│  │  └────────────┘  │   │  Prometheus  │  │    Grafana    │  │
│  └──────────────────┘   │  :9090       │  │    :3000      │  │
│                          └──────────────┘  └───────────────┘  │
│  ┌───────────────────────────────────────────────────────┐    │
│  │              PostgreSQL :5432                          │    │
│  │         (Airflow + MLflow backend)                    │    │
│  └───────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology | Version |
|---|---|---|
| ML Algorithm | LinUCB (custom implementation) | — |
| Dataset | MovieLens-100K | — |
| API Framework | FastAPI | 0.135.2 |
| Frontend | Streamlit | 1.55.0 |
| Experiment Tracking | MLflow | 2.22.0 |
| Pipeline Orchestration | Apache Airflow | 2.10.0 |
| Data Versioning | DVC | 3.51.2 |
| Containerisation | Docker + Compose | Latest |
| Metrics | Prometheus | v2.51.0 |
| Visualisation | Grafana | 10.4.0 |
| Database | PostgreSQL | 15 |
| Testing | pytest | 9.0.3 |
| Language | Python | 3.12 |

---

## Project Structure

```
cinebandit/
├── docker-compose.yml          # All 9 services
├── Makefile                    # Convenience commands
├── MLproject                   # MLflow Projects definition
├── python_env.yaml             # MLflow Projects environment
├── params.yaml                 # All hyperparameters (DVC tracked)
├── dvc.yaml                    # DVC pipeline definition
├── requirements.txt            # pip dependencies
├── environment.yml             # conda dependencies
├── train.py                    # Train LinUCB (no MLflow)
├── train_mlflow.py             # Train LinUCB with MLflow tracking
├── evaluate.py                 # Evaluate saved model
│
├── data/
│   ├── raw/ml-100k/            # MovieLens dataset (DVC tracked)
│   ├── processed/              # Feature-engineered parquet files
│   └── baselines/              # Genre distribution baselines
│
├── outputs/                    # Trained model pickles
├── metrics/                    # DVC pipeline metrics
├── plots/                      # DVC pipeline evaluation plots
│
├── src/
│   ├── bandit/
│   │   ├── linucb.py           # LinUCB implementation (Sherman-Morrison)
│   │   └── simulator.py        # Offline replay evaluation engine
│   ├── data/
│   │   └── loader.py           # MovieLens loader + context vector builder
│   ├── drift/
│   │   └── detector.py         # KL divergence drift detection
│   ├── evaluation/
│   │   └── metrics.py          # CTR, diversity, coverage, plots
│   ├── pipeline/
│   │   ├── prepare.py          # DVC Stage 1: data preparation
│   │   ├── train_stage.py      # DVC Stage 2: training
│   │   └── evaluate_stage.py   # DVC Stage 3: evaluation
│   ├── api/
│   │   ├── main.py             # FastAPI app factory
│   │   ├── core/
│   │   │   ├── model_store.py  # Thread-safe model singleton
│   │   │   ├── metrics.py      # Prometheus metric definitions
│   │   │   └── schemas.py      # Pydantic request/response models
│   │   └── routes/
│   │       ├── recommend.py    # POST /recommend
│   │       ├── feedback.py     # POST /feedback
│   │       └── system.py       # /health, /ready, /drift, /snapshot
│   └── frontend/
│       └── app.py              # Streamlit 3-screen UI
│
├── dags/
│   ├── ingestion_dag.py        # Daily data ingestion + validation
│   ├── drift_detection_dag.py  # 6-hourly drift monitoring
│   └── retraining_dag.py       # On-demand model retraining
│
├── docker/
│   ├── Dockerfile.api          # FastAPI container
│   ├── Dockerfile.frontend     # Streamlit container
│   ├── airflow-init.sh         # Airflow DB init script
│   ├── requirements.api.txt    # API Docker dependencies
│   └── requirements.frontend.txt
│
├── prometheus/
│   └── prometheus.yml          # Scrape config
├── grafana/
│   ├── dashboards/cinebandit.json     # Auto-provisioned dashboard
│   └── provisioning/                  # Grafana provisioning configs
│
├── tests/
│   ├── test_bandit.py          # 22 tests: LinUCBArm, LinUCB
│   ├── test_data.py            # 16 tests: loader, context vectors
│   ├── test_drift.py           # 26 tests: KL divergence, drift detection
│   └── test_api.py             # 38 tests: all FastAPI endpoints
│
└── docs/
    ├── HLD.docx                # High-level design document
    ├── LLD.docx                # Low-level design + API specs
    ├── TestPlan.docx           # Test plan + test cases + report
    ├── UserManual.docx         # Non-technical user guide
    └── Architecture.png        # System architecture diagram
```

---

## Prerequisites

- **OS**: Linux / macOS / Windows with WSL2
- **Python**: 3.12
- **conda**: Miniconda or Anaconda
- **Docker Desktop**: with WSL2 backend enabled (Windows)
- **Git**: for version control
- **DVC**: installed via pip

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/chirag3211/cinebandit.git
cd cinebandit

conda env create -f environment.yml
conda activate cinebandit
```

### 2. Download the dataset

```bash
mkdir -p data/raw
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip -P data/raw/
unzip data/raw/ml-100k.zip -d data/raw/
```

### 3. Train the model

```bash
python train.py --alpha 0.1
```

Expected output:
```
Train CTR: 0.6828  |  Test CTR: 0.6334  |  Arm coverage: 100%
```

### 4. Start all services

```bash
make up
```

This starts all 9 Docker services and automatically loads the trained model into the API.

### 5. Open the UI

Navigate to **http://localhost:8501** in your browser.

---

## Running Locally (without Docker)

Use this for development and testing without spinning up the full stack.

```bash
conda activate cinebandit

# Train
python train.py --alpha 0.1

# Evaluate saved model
python evaluate.py --model outputs/linucb_alpha0.1.pkl

# Run API locally with hot-reload
MODEL_PATH=outputs/linucb_alpha0.1.pkl uvicorn src.api.main:app --reload --port 8000

# Run tests
pytest tests/ -v

# Run Streamlit (in a separate terminal, with API running)
API_URL=http://localhost:8000 streamlit run src/frontend/app.py
```

---

## Running with Docker

```bash
# Start all 9 services
make up

# Stop all services
make down

# Stop and delete all data volumes (WARNING: deletes MLflow runs, Airflow history)
make down-volumes

# Rebuild Docker images (after code changes)
make build

# View logs from all services
make logs

# View API logs only
make logs-api

# Check service status
make ps

# Manually reload model into API after retraining
make reload
```

**Makefile targets summary:**

| Command | Description |
|---|---|
| `make up` | Start all services + auto-load model |
| `make down` | Stop all services |
| `make build` | Rebuild images |
| `make logs` | Tail all logs |
| `make train` | Train locally (alpha=0.1) |
| `make train-mlflow` | Train with MLflow tracking |
| `make tune-mlflow` | Alpha hyperparameter sweep |
| `make evaluate` | Evaluate saved model |
| `make test` | Run all pytest tests |
| `make reload` | Hot-reload model into API |

---

## ML Pipeline

### Training

```bash
# Standard training (no MLflow)
python train.py --alpha 0.1

# With MLflow tracking + model registry promotion
MLFLOW_TRACKING_URI=http://localhost:5000 python train_mlflow.py --alpha 0.1

# Alpha hyperparameter sweep (compares 5 values, promotes best)
MLFLOW_TRACKING_URI=http://localhost:5000 python train_mlflow.py --tune-alpha --steps 30000
```

### Evaluation results (alpha=0.1)

| Metric | Train | Test |
|---|---|---|
| CTR | 0.6828 | 0.6334 |
| Cumulative Reward | 7,389 | 1,548 |
| Genre Diversity | 3.595 bits | 3.621 bits |
| Arm Coverage | 100% | 100% |
| Learning Improvement | +10.9% (Q1→Q4) | — |
| p95 Latency | — | < 25ms |

### Algorithm

LinUCB (Disjoint variant) with a **60-dimensional context vector**:

```
Context = [user_genre_profile (19) | movie_genre (19) | interaction (19) | year | age | gender]
UCB score = theta^T * x + alpha * sqrt(x^T * A_inv * x)
```

Updates use the **Sherman-Morrison formula** for O(d²) incremental matrix inverse updates — no full retraining needed per feedback.

---

## MLflow Experiment Tracking

```bash
# Open MLflow UI
open http://localhost:5000

# Run a tracked training experiment
make train-mlflow
```

**What gets tracked per run:**

| Category | Items |
|---|---|
| Parameters | algorithm, alpha, pool_size, context_dim, seed, n_train, n_test |
| Metrics | train_ctr, test_ctr, genre_diversity, arm_coverage, match_rate, train_time_s |
| Step metrics | cumulative_ctr (logged every 50 steps) |
| Artifacts | model pickle, train evaluation plot, test evaluation plot |
| Registry | Model registered as `CineBandit`, promoted to `production` alias if CTR improves |

**Model promotion logic:**

New model (challenger) is compared against the current `production` model. If `challenger_ctr > champion_ctr`, the challenger is promoted automatically. This happens inside the Airflow retraining DAG without any manual intervention.

---

## Airflow DAGs

```bash
# Open Airflow UI
open http://localhost:8080  # admin / admin

# Trigger ingestion DAG (creates baseline statistics)
docker exec cinebandit_airflow_scheduler airflow dags trigger cinebandit_ingestion

# Trigger drift detection
docker exec cinebandit_airflow_scheduler airflow dags trigger cinebandit_drift_detection

# Trigger retraining
docker exec cinebandit_airflow_scheduler airflow dags trigger cinebandit_retraining
```

**DAG overview:**

| DAG | Schedule | Tasks | Purpose |
|---|---|---|---|
| `cinebandit_ingestion` | @daily | 4 | Load data → validate schema → compute baselines → log to MLflow |
| `cinebandit_drift_detection` | Every 6h | 5 | Compute KL divergence → update API → trigger retraining if alert |
| `cinebandit_retraining` | On-demand | 6 | Train → compare vs champion → promote → reload API |

---

## DVC Pipeline

```bash
# Initialise DVC (first time only)
git init
dvc init
dvc config core.autostage true
mkdir -p ~/dvc-store
dvc remote add -d localstore ~/dvc-store

# Track the raw dataset
dvc add data/raw/ml-100k

# View the pipeline DAG
dvc dag

# Run the full pipeline
dvc repro

# Show metrics
dvc metrics show

# Compare metrics between git commits
dvc metrics diff

# Change a hyperparameter and see incremental re-run
# (only affected stages re-run — prepare is skipped if data unchanged)
sed -i 's/alpha: 0.1/alpha: 0.5/' params.yaml
dvc repro

# Rollback to a previous experiment
git checkout <commit-hash>
dvc checkout
```

**Pipeline stages:**

```
prepare → train → evaluate
```

- `prepare`: raw data → processed parquet + baseline statistics
- `train`: processed data + params.yaml → model pickle + train metrics
- `evaluate`: model + test data → eval metrics + evaluation plots

---

## Monitoring

### Prometheus

```bash
# Open Prometheus
open http://localhost:9090

# Example queries
cinebandit_recommendation_requests_total
cinebandit_rolling_ctr
cinebandit_drift_score
histogram_quantile(0.95, rate(cinebandit_recommendation_latency_seconds_bucket[5m]))
```

**Instrumented metrics (12 total):**

| Metric | Type | Description |
|---|---|---|
| `cinebandit_recommendation_requests_total` | Counter | Total requests by status |
| `cinebandit_recommendation_latency_seconds` | Histogram | Latency (p50/p95/p99) |
| `cinebandit_feedback_total` | Counter | Feedback by reaction type |
| `cinebandit_rolling_ctr` | Gauge | CTR over last 100 events |
| `cinebandit_drift_score` | Gauge | KL divergence score |
| `cinebandit_model_timesteps` | Gauge | Model update count |
| `cinebandit_model_alpha` | Gauge | Exploration parameter |
| `cinebandit_arms_seen_total` | Gauge | Unique movies updated |
| `cinebandit_cumulative_reward_total` | Counter | Total reward since startup |
| `cinebandit_retrain_runs_total` | Counter | Retraining triggers |
| `cinebandit_model_load_time_seconds` | Gauge | Startup model load time |
| `cinebandit_api_errors_total` | Counter | Errors by endpoint and type |

**Alert thresholds:**
- Drift score > 0.3 → triggers automatic retraining
- Error rate > 5% → alert
- p95 latency > 200ms → warning

### Grafana

```bash
# Open Grafana
open http://localhost:3000  # admin / admin
```

The CineBandit dashboard (11 panels) is auto-provisioned on startup — no manual configuration needed.

---

## API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness probe — always 200 if server is up |
| GET | `/ready` | Readiness probe — 200 only if model is loaded |
| POST | `/recommend` | Get top-N movie recommendations for a user |
| POST | `/feedback` | Submit user feedback (like/dislike/skip) |
| GET | `/model/info` | Current model metadata |
| GET | `/drift/status` | Current drift score and alert state |
| POST | `/retrain/trigger` | Manually trigger retraining pipeline |
| POST | `/model/reload` | Hot-reload model without restart |
| GET | `/snapshot` | Quick metrics snapshot for dashboard |
| GET | `/metrics` | Prometheus metrics endpoint |

Full interactive API documentation: **http://localhost:8000/docs**

### Quick examples

```bash
# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "n": 5}'

# Send feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 50, "reaction": "like"}'

# Check model status
curl http://localhost:8000/ready
curl http://localhost:8000/snapshot
```

---

## Unit Tests

```bash
# Run all 102 tests
pytest tests/ -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing

# Run individual test suites
pytest tests/test_bandit.py -v   # 22 tests: LinUCBArm, LinUCB
pytest tests/test_data.py -v     # 16 tests: loader, context vectors, profiles
pytest tests/test_drift.py -v    # 26 tests: KL divergence, drift detection
pytest tests/test_api.py -v      # 38 tests: all FastAPI endpoints
```

**Test results: 102 passed, 0 failed**

---

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Streamlit UI | http://localhost:8501 | — |
| FastAPI (Swagger) | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Airflow | http://localhost:8080 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

## Troubleshooting

**Model not loaded after `make up`:**
```bash
make reload
```

**Airflow DAGs not appearing:**
```bash
docker logs cinebandit_airflow_scheduler 2>&1 | grep -i "dag\|error" | tail -20
```

**Prometheus not scraping metrics:**
```bash
curl -sL http://localhost:8000/metrics | grep "^cinebandit"
```

**Database connection errors:**
```bash
docker compose restart postgres
sleep 10
make up
```

**Full reset (WARNING: deletes all data):**
```bash
make down
docker volume rm $(docker volume ls -q | grep cinebandit)
make up
```
