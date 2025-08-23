FROM python:3.10-slim

RUN pip install --no-cache-dir mlflow==2.14.1 psycopg2-binary==2.9.9 gunicorn==22.0.0

# Create artifact dir
RUN mkdir -p /mlflow/artifacts
WORKDIR /mlflow

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --artifacts-destination ${MLFLOW_ARTIFACT_ROOT}
