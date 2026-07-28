from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "water-pressure",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="water_pressure_lstm_inference",
    default_args=default_args,
    description="Predict water pressure and save alert result every 15 minutes.",
    schedule_interval="*/15 * * * *",
    start_date=datetime(2026, 7, 2),
    catchup=False,
    max_active_runs=1,
    tags=["water-pressure", "lstm", "inference"],
) as dag:
    run_prediction = BashOperator(
        task_id="run_lstm_prediction",
        bash_command="cd /opt/airflow && python /opt/airflow/scripts/predict_single_logger.py",
    )

    run_prediction
