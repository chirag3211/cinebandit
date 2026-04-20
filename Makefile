# CineBandit — Makefile
# Convenience commands for development and deployment.
#
# Usage:
#   make up          → start all services
#   make down        → stop all services
#   make logs        → tail logs from all services
#   make train       → train the LinUCB model locally
#   make test        → run unit tests
#   make clean       → remove containers and volumes

.PHONY: up down logs build restart train evaluate test clean ps help

# ── Docker Compose ────────────────────────────────────────────────────────────

up:
	@echo "Starting CineBandit services..."
	docker compose up -d
	@echo ""
	@echo "Waiting for API to be ready..."
	@sleep 15
	@if [ -f outputs/linucb_alpha0.1.pkl ]; then 		docker cp outputs/linucb_alpha0.1.pkl cinebandit_api:/app/outputs/linucb_alpha0.1.pkl && 	docker exec -u root cinebandit_api chown 50000:0 /app/outputs/linucb_alpha0.1.pkl &&	curl -s -X POST http://localhost:8000/model/reload > /dev/null && 		echo "Model loaded into API ✅"; 	else 		echo "⚠️  No model found in outputs/ — run 'make train' first then 'make reload'"; 	fi
	@echo ""
	@echo "Services available at:"
	@echo "  API          → http://localhost:8000/docs"
	@echo "  Frontend     → http://localhost:8501"
	@echo "  MLflow       → http://localhost:5000"
	@echo "  Airflow      → http://localhost:8080  (admin/admin)"
	@echo "  Prometheus   → http://localhost:9090"
	@echo "  Grafana      → http://localhost:3000  (admin/admin)"

reload:
	@echo "Copying model and reloading API..."
	docker cp outputs/linucb_alpha0.1.pkl cinebandit_api:/app/outputs/linucb_alpha0.1.pkl
	docker exec -u root cinebandit_api chown 50000:0 /app/outputs/linucb_alpha0.1.pkl
	curl -s -X POST http://localhost:8000/model/reload
	@echo ""
	@echo "Model reloaded ✅"

down:
	docker compose down

down-volumes:
	@echo "WARNING: This will delete all persistent data (models, MLflow runs, Airflow history)"
	docker compose down -v

build:
	docker compose build --no-cache

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-airflow:
	docker compose logs -f airflow-webserver airflow-scheduler

ps:
	docker compose ps

# ── Local development (no Docker) ─────────────────────────────────────────────

install:
	pip install -r requirements.txt

train:
	python train.py --alpha 0.1

train-mlflow:
	MLFLOW_TRACKING_URI=http://localhost:5000 python train_mlflow.py --alpha 0.1

tune-mlflow:
	MLFLOW_TRACKING_URI=http://localhost:5000 python train_mlflow.py --tune-alpha --steps 30000

tune:
	python train.py --tune-alpha --steps 30000

evaluate:
	python evaluate.py

api:
	MODEL_PATH=outputs/linucb_alpha0.1.pkl uvicorn src.api.main:app --reload --port 8000

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v

test-api:
	python -m pytest tests/test_api.py -v

test-bandit:
	python -m pytest tests/test_bandit.py -v

# ── Utilities ─────────────────────────────────────────────────────────────────

clean:
	docker compose down
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "CineBandit Makefile commands:"
	@echo ""
	@echo "  Docker:"
	@echo "    make up            Start all services (detached)"
	@echo "    make down          Stop all services"
	@echo "    make down-volumes  Stop and delete all data volumes"
	@echo "    make build         Rebuild Docker images"
	@echo "    make logs          Tail logs from all services"
	@echo "    make logs-api      Tail API logs only"
	@echo "    make ps            Show service status"
	@echo ""
	@echo "  Local dev:"
	@echo "    make install       Install Python dependencies"
	@echo "    make train         Train LinUCB model"
	@echo "    make tune          Run alpha hyperparameter sweep"
	@echo "    make evaluate      Evaluate saved model"
	@echo "    make api           Run FastAPI locally with hot-reload"
	@echo ""
	@echo "  Testing:"
	@echo "    make test          Run all tests"
	@echo "    make test-api      Run API tests only"
	@echo "    make test-bandit   Run bandit tests only"