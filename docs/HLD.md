# CineBandit
## Adaptive Movie Recommendation System
### High-Level Design Document
*Version 1.0 | April 2026*

---

## 1. Executive Summary

CineBandit is an adaptive movie recommendation system built on Contextual Bandit learning, specifically the LinUCB (Linear Upper Confidence Bound) algorithm. The system continuously learns from user feedback in real time, adapting recommendations as preferences evolve. It is designed and deployed using a comprehensive MLOps stack, ensuring full automation of the ML lifecycle from data ingestion through deployment, monitoring, and automated retraining.

The system addresses the core challenge of recommendation systems: balancing exploration of new movies (discovering what a user might like) with exploitation of known preferences (recommending movies the user is likely to enjoy). Unlike static recommendation models that require offline batch retraining, CineBandit updates its parameters incrementally with every user interaction, making it genuinely adaptive.

---

## 2. Problem Statement

Modern digital platforms serving large movie catalogs face a fundamental challenge: how to surface relevant content to users from thousands of options, while continuously adapting to evolving tastes. Static models trained offline cannot respond to preference drift — the gradual shift in what a user enjoys over time.

CineBandit solves this with a production-grade adaptive recommendation system that:

- Learns user preferences in real time from explicit feedback (like/dislike/skip)
- Detects when the user population's preferences have shifted (distribution drift)
- Automatically retrains when drift is detected, without human intervention
- Operates entirely on-premises — no cloud dependency
- Provides full observability through Prometheus metrics and Grafana dashboards

### 2.1 Success Metrics

| Metric | Type | Target | Achieved |
|--------|------|--------|----------|
| Click-Through Rate (CTR) | ML Metric | > 0.55 | 0.683 (train), 0.633 (test) |
| Recommendation Latency p95 | Business Metric | < 200ms | < 25ms |
| Arm Coverage | ML Metric | 100% | 100% (1,682 movies) |
| Learning Improvement | ML Metric | > 5% | +10.9% (Q1 → Q4) |
| Genre Diversity | ML Metric | > 3.0 bits | 3.62 bits (max 4.25) |
| API Availability | Business Metric | > 99% | Health + Ready endpoints |
| Drift Detection | Business Metric | KL < 0.3 | Automated threshold alerting |

---

## 3. Architecture Overview

CineBandit follows a microservices architecture with 9 containerised services orchestrated via Docker Compose. The system is divided into three functional layers:

#### Layer 1 — Data & ML
- Raw data ingestion and validation (Airflow DAG 1)
- Feature engineering and context vector construction (`src/data/loader.py`)
- LinUCB bandit training with offline replay evaluation (`src/bandit/`)
- Drift detection via KL divergence (`src/drift/detector.py`)

#### Layer 2 — Serving
- FastAPI REST server exposing `/recommend`, `/feedback`, `/health`, `/ready`, `/metrics`
- Thread-safe model store with hot-reload capability
- Prometheus instrumentation on all key metrics

#### Layer 3 — Operations
- MLflow experiment tracking and model registry
- Airflow pipeline orchestration (3 DAGs)
- Prometheus metrics collection and Grafana dashboards
- DVC data and model versioning

### 3.1 Service Topology

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| postgres | postgres:15-alpine | 5432 | Shared DB backend for Airflow and MLflow |
| mlflow | ghcr.io/mlflow/mlflow:v2.22.0 | 5000 | Experiment tracking + model registry |
| api | cinebandit-api (custom) | 8000 | FastAPI model serving + Prometheus metrics |
| frontend | cinebandit-frontend (custom) | 8501 | Streamlit 3-screen UI |
| airflow-webserver | apache/airflow:2.10.0 | 8080 | Airflow pipeline UI and API |
| airflow-scheduler | apache/airflow:2.10.0 | — | DAG execution engine |
| airflow-init | apache/airflow:2.10.0 | — | One-shot DB initialisation |
| prometheus | prom/prometheus:v2.51.0 | 9090 | Metrics scraping and storage |
| grafana | grafana/grafana:10.4.0 | 3000 | Real-time monitoring dashboards |

---

## 4. ML Algorithm — LinUCB

### 4.1 Algorithm Selection Rationale

The Contextual Bandit approach, specifically LinUCB, was chosen over traditional collaborative filtering or deep learning approaches for the following reasons:

- **Incremental learning:** parameters update in O(d²) per interaction using Sherman-Morrison formula — no batch retraining required for real-time adaptation
- **Exploration-exploitation balance:** the UCB formula explicitly balances trying new movies (explore) with recommending known good ones (exploit), controlled by the alpha hyperparameter
- **Interpretability:** θᵀ × x is a linear dot product — the model's reasoning is fully transparent and defensible in a viva
- **CPU efficiency:** no GPU required; inference takes < 25ms on commodity hardware
- **Industry precedent:** Contextual Bandits are used in production at Netflix, Google, and Amazon for exactly this use case

### 4.2 Context Vector Design

Each (user, movie) pair is represented as a 60-dimensional context vector:

| Component | Dimensions | Description |
|-----------|-----------|-------------|
| User genre preference | 19 | Normalised mean genre vector of liked movies (L2 norm = 1) |
| Movie genre vector | 19 | One-hot genre encoding from MovieLens metadata |
| Interaction term | 19 | Element-wise product: user_profile * movie_genre (captures alignment) |
| Movie release year (normalised) | 1 | Year scaled to [0,1] range |
| User age (normalised) | 1 | Age scaled to [0,1] range |
| User gender (binary) | 1 | Male=1.0, Female=0.0 |

---

## 5. Data Pipeline

### 5.1 Dataset

MovieLens-100K: 100,000 ratings from 943 users on 1,682 movies. Temporal split: 80% train (80,000 ratings), 20% test (20,000 ratings), ordered by timestamp to prevent data leakage.

### 5.2 Airflow DAGs

| DAG | Schedule | Tasks | Purpose |
|-----|----------|-------|---------|
| cinebandit_ingestion | @daily | 4 | Load data, validate schema, compute baselines, log to MLflow |
| cinebandit_drift_detection | Every 6 hours | 5 | Compute KL divergence, update API drift score, trigger retraining if alert |
| cinebandit_retraining | On-demand only | 6 | Train model, compare vs champion, promote if better, reload API |

---

## 6. MLOps Stack

### 6.1 Experiment Tracking — MLflow

Every training run logs the following to MLflow:

- **Parameters:** algorithm, alpha, pool_size, context_dim, seed, n_train, n_test
- **Metrics:** train_ctr, test_ctr, genre_diversity, arm_coverage, match_rate, train_time_s
- **Metrics (step-by-step):** cumulative_ctr logged every 50 steps for learning curve
- **Artifacts:** model pickle, train evaluation plot, test evaluation plot
- **Model registry:** registered as 'CineBandit', promoted via 'production' alias if CTR improves

### 6.2 Data & Model Versioning — DVC

DVC manages the full pipeline as a DAG of reproducible stages:

- **prepare:** raw data → processed parquet files + baseline statistics
- **train:** processed data + params.yaml → model pickle + train metrics
- **evaluate:** model + test data → eval metrics + evaluation plots

Changing any parameter in `params.yaml` triggers only affected downstream stages. `git checkout` + `dvc checkout` provides complete rollback to any previous experiment.

### 6.3 Monitoring — Prometheus + Grafana

| Metric | Type | Alert Threshold |
|--------|------|----------------|
| cinebandit_recommendation_requests_total | Counter | — |
| cinebandit_recommendation_latency_seconds | Histogram | p95 > 200ms |
| cinebandit_feedback_total{reaction} | Counter | — |
| cinebandit_rolling_ctr | Gauge | — |
| cinebandit_drift_score | Gauge | > 0.3 triggers retraining |
| cinebandit_model_timesteps | Gauge | — |
| cinebandit_model_alpha | Gauge | — |
| cinebandit_arms_seen_total | Gauge | — |
| cinebandit_cumulative_reward_total | Counter | — |
| cinebandit_retrain_runs_total | Counter | — |
| cinebandit_model_load_time_seconds | Gauge | — |
| cinebandit_api_errors_total{endpoint,error_type} | Counter | Rate > 5% |

---

## 7. Design Principles

### 7.1 Loose Coupling

The Streamlit frontend and FastAPI backend are completely independent Docker containers communicating exclusively via REST API calls. The `API_URL` is a configurable environment variable — the frontend has zero awareness of the model implementation. This is enforced architecturally by the Docker network.

### 7.2 Reproducibility

Every experiment is reproducible via a specific Git commit hash + DVC lock file + MLflow run ID. The combination uniquely identifies: the code version, the data version, the hyperparameters, and all output metrics and artifacts.

### 7.3 Continuous Learning

Unlike traditional ML systems with periodic batch retraining, CineBandit implements a true continuous learning loop: every `/feedback` call triggers an incremental LinUCB parameter update (Sherman-Morrison, O(d²)) without any retraining pipeline. Periodic full retraining (triggered by drift detection) serves to recalibrate the model when distribution shift is detected.

---

## 8. Technology Stack

| Category | Technology | Version | Role |
|----------|-----------|---------|------|
| ML Algorithm | LinUCB (custom) | — | Contextual bandit recommendation |
| Data | MovieLens-100K | — | 100k ratings, 1682 movies, 943 users |
| API Framework | FastAPI | 0.135.2 | REST model serving |
| Frontend | Streamlit | 1.55.0 | 3-screen interactive UI |
| Experiment Tracking | MLflow | 2.22.0 | Runs, metrics, model registry |
| Pipeline Orchestration | Apache Airflow | 2.10.0 | 3 automated DAGs |
| Data Versioning | DVC | 3.51.2 | Pipeline + artifact versioning |
| Containerisation | Docker + Compose | Latest | 9-service orchestration |
| Metrics | Prometheus | v2.51.0 | 12 custom metrics |
| Visualisation | Grafana | 10.4.0 | 11-panel NRT dashboard |
| Database | PostgreSQL | 15 | Airflow + MLflow backend |
| Testing | pytest | 9.0.3 | 102 unit tests |
| Language | Python | 3.12.13 | All components |
