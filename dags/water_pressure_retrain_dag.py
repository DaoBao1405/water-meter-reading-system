from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "water-pressure",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="water_pressure_lstm_retrain",
    default_args=default_args,
    description="Retrain water pressure LSTM model from SQL Server once per day.",
    schedule_interval="0 1 * * *",
    start_date=datetime(2026, 7, 2),
    catchup=False,
    max_active_runs=1,
    tags=["water-pressure", "lstm", "retrain"],
) as dag:
    run_retrain = BashOperator(
        task_id="run_lstm_retrain",
        bash_command="cd /opt/airflow && python /opt/airflow/scripts/train_single_logger.py",
    )

    run_retrain
