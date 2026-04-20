#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'Creating airflow database...';
    CREATE DATABASE airflow;
    SELECT 'Creating mlflow database...';
    CREATE DATABASE mlflow;
EOSQL
echo "[init-postgres] Databases created."