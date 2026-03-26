# FROM python:3.11-slim
 
# RUN pip install --no-cache-dir mlflow==2.13.2 psycopg2-binary boto3
 
# EXPOSE 5000
 
# # Use shell form so env vars expand correctly
# CMD mlflow server \
#     --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
#     --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}" \
#     --host 0.0.0.0 \
#     --port 5000 \
#     --workers 2



FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.13.2 psycopg2-binary boto3

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5000

CMD mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "${MLFLOW_ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 2

