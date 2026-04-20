#!/bin/bash
set -e

echo "[airflow-init] Starting..."

# Install postgresql-client if psql not available
if ! command -v psql &> /dev/null; then
    echo "[airflow-init] Installing postgresql-client..."
    apt-get update -qq && apt-get install -y -qq postgresql-client
fi

echo "[airflow-init] Creating databases if they don't exist..."

PGPASSWORD=postgres psql -h postgres -U postgres -d postgres \
    -tc "SELECT 1 FROM pg_database WHERE datname='airflow'" | grep -q 1 \
    || PGPASSWORD=postgres psql -h postgres -U postgres -d postgres \
       -c "CREATE DATABASE airflow"
echo "[airflow-init] airflow database ready"

PGPASSWORD=postgres psql -h postgres -U postgres -d postgres \
    -tc "SELECT 1 FROM pg_database WHERE datname='mlflow'" | grep -q 1 \
    || PGPASSWORD=postgres psql -h postgres -U postgres -d postgres \
       -c "CREATE DATABASE mlflow"
echo "[airflow-init] mlflow database ready"

# Install project Python dependencies into the Airflow environment
# These are needed by train_mlflow.py and the DAG tasks
# echo "[airflow-init] Installing project dependencies..."

# su -s /bin/bash airflow -c "
# /home/airflow/.local/bin/python -m pip install --quiet \
#     mlflow==3.10.1 \
#     scikit-learn \
#     pandas \
#     numpy \
#     matplotlib \
#     seaborn \
#     tqdm \
#     psycopg2-binary \
#     requests
# "

echo "[airflow-init] Dependencies installed"

# Fix permissions for airflow user (uid=50000)
echo "[airflow-init] Fixing directory permissions..."

mkdir -p /opt/airflow/logs/scheduler
chown -R 50000:0 /opt/airflow/logs
chmod -R 775     /opt/airflow/logs

mkdir -p /opt/airflow/outputs
chown -R 50000:0 /opt/airflow/outputs
chmod -R 775     /opt/airflow/outputs

chmod -R 755 /opt/airflow/data/raw  2>/dev/null || true
mkdir -p /opt/airflow/data/baselines
chown -R 50000:0 /opt/airflow/data/baselines
chmod -R 775     /opt/airflow/data/baselines

chmod -R 755 /opt/airflow/src  2>/dev/null || true
chmod -R 755 /opt/airflow/dags 2>/dev/null || true

echo "[airflow-init] Permissions fixed"

echo "[airflow-init] Running airflow db migrate..."
su -s /bin/bash airflow -c "airflow db migrate"

echo "[airflow-init] Creating admin user..."
su -s /bin/bash airflow -c "
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@cinebandit.local 2>/dev/null \
    || echo 'Admin user already exists, skipping.'
"

echo "[airflow-init] Done ✅"