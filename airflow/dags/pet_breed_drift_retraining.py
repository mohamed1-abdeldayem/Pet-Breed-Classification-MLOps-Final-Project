import json
import os
import urllib.request
from datetime import UTC, datetime

from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from airflow import DAG

PROMETHEUS_URL = "http://prometheus:9090"
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "").strip()

if not PROJECT_ROOT:
    raise ValueError("PROJECT_ROOT environment variable is required")

PROJECT_MOUNT = [
    Mount(
        source=PROJECT_ROOT,
        target="/workspace",
        type="bind",
    )
]

TRAINING_IMAGE = "pet-breed-training:local"


def detect_drift(**context):
    url = f"{PROMETHEUS_URL}/api/v1/alerts"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise AirflowException(
            f"Could not read Prometheus alerts: {exc}"
        ) from exc

    if payload.get("status") != "success":
        raise AirflowException(
            f"Prometheus API returned status: {payload.get('status')!r}"
        )
    alerts = payload.get("data", {}).get("alerts", [])
    firing_alerts = [
        alert
        for alert in alerts
        if alert.get("labels", {}).get("alertname")
        == "PetBreedPredictionDrift"
        and alert.get("state") == "firing"
    ]

    if not firing_alerts:
        return {
            "branch": "no_drift",
            "active_at": "",
            "reason": "No firing drift alert",
        }

    alert = max(firing_alerts, key=lambda item: item.get("activeAt", ""))

    active_at = alert.get("activeAt", "")
    if not active_at:
        raise AirflowException(
            "Drift alert is firing but activeAt is missing"
        )

    last_passed = Variable.get(
        "pet_breed_last_drift_episode",
        default_var="",
    )
    last_failed_gate = Variable.get(
        "pet_breed_last_failed_gate_episode",
        default_var="",
    )

    if active_at == last_passed:
        reason = "This alert episode already passed the quality gate"
    elif active_at == last_failed_gate:
        reason = "This alert episode already failed the quality gate"
    else:
        return {
            "branch": "train_candidate",
            "active_at": active_at,
            "reason": "New firing drift episode",
        }

    return {
        "branch": "no_drift",
        "active_at": active_at,
        "reason": reason,
    }


def choose_branch(**context):
    result = context["ti"].xcom_pull(task_ids="detect_drift")
    return result["branch"]



def record_episode(**context):
    ti = context["ti"]
    result = ti.xcom_pull(task_ids="detect_drift") or {}
    active_at = result.get("active_at", "")

    if not active_at or result.get("branch") == "no_drift":
        print("No new drift episode to record.")
        return

    dag_run = ti.get_dagrun()
    train_ti = dag_run.get_task_instance("train_candidate")
    gate_ti = dag_run.get_task_instance("quality_gate")

    if train_ti.state != "success":
        raise AirflowException(
            f"Candidate training did not succeed: {train_ti.state}"
        )

    if gate_ti.state != "success":
        raise AirflowException(
            f"Quality gate task execution failed: {gate_ti.state}"
        )

    gate_output = ti.xcom_pull(task_ids="quality_gate")

    try:
        output_lines = gate_output.strip().splitlines()
        gate_result = json.loads(output_lines[-1])
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise AirflowException(
            "Could not parse quality gate JSON output"
        ) from exc

    status = gate_result.get("status")

    if status == "passed":
        Variable.set("pet_breed_last_drift_episode", active_at)
        Variable.set("pet_breed_last_quality_pass_episode", active_at)
        print(
            "Quality gate passed. Candidate remains unpromoted; "
            "production model was not changed."
        )
        return

    if status == "failed":
        Variable.set(
            "pet_breed_last_failed_gate_episode",
            active_at,
        )
        raise AirflowException(
            "Quality gate rejected the candidate. "
            "Candidate was not promoted."
        )

    raise AirflowException(
        f"Unexpected quality gate status: {status!r}"
    )

with DAG(
    dag_id="pet_breed_drift_retraining",
    description="Detect prediction drift, train an isolated candidate, and run a quality gate",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": 1,
    },
    tags=["pet-breed", "drift", "candidate"],
) as dag:

    detect_drift_task = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
    )

    route_drift = BranchPythonOperator(
        task_id="route_drift",
        python_callable=choose_branch,
    )

    no_drift = EmptyOperator(
        task_id="no_drift",
    )

    train_candidate = DockerOperator(
        task_id="train_candidate",
        image=TRAINING_IMAGE,
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=PROJECT_MOUNT,
        working_dir="/workspace",
        entrypoint=["python"],
        command=[
            "/workspace/training/train.py",
            "--backbone", "resnet50",
            "--lr", "0.0003",
            "--batch-size", "64",
            "--epochs", "8",
            "--seed", "42",
            "--out-dir", "models/candidates/{{ ts_nodash }}",
            "--model-version", "candidate-{{ ts_nodash }}",
        ],
        environment={
            "PET_LABEL_MAP": "/workspace/configs/label_map.json",
        },
    )

    quality_gate = DockerOperator(
        task_id="quality_gate",
        do_xcom_push=True,
        image=TRAINING_IMAGE,
        docker_url="unix://var/run/docker.sock",
        api_version="auto",
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=PROJECT_MOUNT,
        working_dir="/workspace",
        environment={"PYTHONPATH": "/workspace/src"},
        entrypoint=["python"],
        command=[
            "-m",
            "pet_breed_mlops.quality_gate_runner",
            "--baseline",
            "reports/baseline.json",
            "--candidate",
            "models/candidates/{{ ts_nodash }}/metrics.json",
            "--tolerance",
            "0.01",
            "--result",
            "models/candidates/{{ ts_nodash }}/quality_gate_result.json",
        ],
    )

    record_result = PythonOperator(
        task_id="record_result",
        python_callable=record_episode,
        trigger_rule="all_done",
    )

    finish = EmptyOperator(
        task_id="finish",
        trigger_rule="none_failed_min_one_success",
    )

    detect_drift_task >> route_drift
    route_drift >> [train_candidate, no_drift]
    train_candidate >> quality_gate >> record_result
    [no_drift, record_result] >> finish
