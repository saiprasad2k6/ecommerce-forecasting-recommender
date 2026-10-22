from datetime import datetime

from airflow.operators.bash import BashOperator

from airflow import DAG

with DAG(
    dag_id="olist_training",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:
    train = BashOperator(task_id="train_models", bash_command="python scripts/train_models.py")
