cinebandit/
│
├── docker-compose.yml                  # Brings up all 9 services with one command
├── Makefile                            # Convenience commands (make up, make train, etc.)
├── requirements.txt                    # pip dependencies (use this for venv)
├── environment.yml                     # conda dependencies (use this for conda)
├── .dockerignore                       # Files excluded from Docker build context
├── train.py                            # CLI: train LinUCB model
├── evaluate.py                         # CLI: evaluate saved model
│
├── data/                               # All data (download once, git-ignored)
│   └── raw/
│       └── ml-100k/                    # Downloaded automatically by loader.py
│           ├── u.data                  # 100k ratings (user, movie, rating, timestamp)
│           ├── u.item                  # Movie metadata (title, year, genres)
│           └── u.user                  # User demographics (age, gender, occupation)
│
├── outputs/                            # Trained model artifacts (git-ignored)
│   └── linucb_alpha0.1.pkl             # Saved LinUCB model (your best alpha)
│
├── src/                                # All application source code
│   ├── __init__.py
│   │
│   ├── bandit/                         # Core ML — LinUCB implementation
│   │   ├── __init__.py
│   │   ├── linucb.py                   # LinUCBArm + LinUCB classes
│   │   └── simulator.py                # Offline replay evaluation engine
│   │
│   ├── data/                           # Data loading and feature engineering
│   │   ├── __init__.py
│   │   └── loader.py                   # MovieLens loader, context vector builder
│   │
│   ├── evaluation/                     # Metrics and plotting
│   │   ├── __init__.py
│   │   └── metrics.py                  # CTR, diversity, coverage, plots
│   │
│   ├── api/                            # FastAPI service (model serving)
│   │   ├── __init__.py
│   │   ├── main.py                     # App factory, lifespan, middleware
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── model_store.py          # Thread-safe model singleton
│   │   │   ├── metrics.py              # Prometheus metric definitions
│   │   │   └── schemas.py              # Pydantic request/response models
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── recommend.py            # POST /recommend
│   │       ├── feedback.py             # POST /feedback
│   │       └── system.py              # GET /health, /ready, /model/info, /drift/status, etc.
│   │
│   └── frontend/                       # Streamlit UI (Week 5 — not yet built)
│       └── app.py
│
├── dags/                               # Airflow DAGs (Week 3 — not yet built)
│   ├── ingestion_dag.py
│   ├── drift_detection_dag.py
│   └── retraining_dag.py
│
├── tests/                              # Unit and integration tests (Week 5)
│   ├── test_bandit.py
│   ├── test_api.py
│   └── test_drift.py
│
├── docker/                             # Dockerfiles and per-service requirements
│   ├── Dockerfile.api                  # FastAPI container
│   ├── Dockerfile.frontend             # Streamlit container
│   ├── init-postgres.sh                # Creates airflow + mlflow databases on first boot
│   ├── requirements.api.txt            # API-only dependencies for Docker layer caching
│   └── requirements.frontend.txt      # Frontend-only dependencies
│
├── prometheus/
│   └── prometheus.yml                  # Scrape config (targets: api:8000/metrics)
│
├── grafana/
│   ├── dashboards/
│   │   └── cinebandit.json             # Auto-provisioned dashboard (11 panels)
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yml          # Auto-adds Prometheus datasource
│       └── dashboards/
│           └── dashboards.yml          # Tells Grafana where to find dashboard JSON
│
└── docs/                               # Documentation (Week 5 — not yet built)
    ├── HLD.md
    ├── LLD.md
    ├── architecture.png
    ├── test_plan.md
    └── user_manual.md