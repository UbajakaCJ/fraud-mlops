#!/bin/bash
# Creates the frauddb database alongside the default airflow DB

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE frauddb;
    GRANT ALL PRIVILEGES ON DATABASE frauddb TO $POSTGRES_USER;
    CREATE DATABASE mlflowdb;
    GRANT ALL PRIVILEGES ON DATABASE mlflowdb TO $POSTGRES_USER;
EOSQL
