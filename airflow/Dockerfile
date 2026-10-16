FROM pet-breed-training:local

USER root

ENV AIRFLOW_HOME=/opt/airflow

RUN python -m venv /opt/airflow-venv \
    && /opt/airflow-venv/bin/pip install --no-cache-dir \
       "apache-airflow==2.10.5" \
       "apache-airflow-providers-docker" \
       --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt" \
    && mkdir -p /opt/airflow \
    && chown -R appuser:appuser /opt/airflow /opt/airflow-venv

ENV PATH="/app/.venv/bin:/opt/airflow-venv/bin:$PATH"

USER appuser

WORKDIR /workspace

ENTRYPOINT ["/opt/airflow-venv/bin/airflow"]
CMD ["standalone"]
