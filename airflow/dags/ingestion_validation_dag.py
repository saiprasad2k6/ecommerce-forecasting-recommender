from datetime import datetime

from airflow.operators.bash import BashOperator

from airflow import DAG

with DAG(
    dag_id="olist_ingestion_validation",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:
    download = BashOperator(task_id="download_sample", bash_command="python scripts/download_olist_data.py")
    validate = BashOperator(task_id="validate_data", bash_command="python scripts/validate_olist_data.py")
    download >> validate
