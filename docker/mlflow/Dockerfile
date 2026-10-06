FROM python:3.12-slim

ARG MLFLOW_VERSION=3.16.1
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" "psycopg[binary]" psycopg2-binary

EXPOSE 5000