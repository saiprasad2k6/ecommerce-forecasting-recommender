"""Load the real Airflow DAG and exercise commands without running data jobs."""

from pathlib import Path
import subprocess
import sys
import unittest


class DataPipelineTests(unittest.TestCase):
    def test_dag_dependencies_parameters_and_commands(self):
        root = Path(__file__).resolve().parents[1]
        # Airflow configuration/logging is isolated from the user's Airflow home.
        check = r'''
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path.cwd()
with tempfile.TemporaryDirectory() as directory:
    temporary = Path(directory)
    os.environ['AIRFLOW_HOME'] = str(temporary / 'airflow')
    os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
    os.environ['OLIST_PROJECT_ROOT'] = str(root)
    executable = temporary / 'python with spaces'
    executable.write_text('#!' + sys.executable + '\n'
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'Path(os.environ["TEST_RECORD_FILE"]).write_text(json.dumps(sys.argv[1:]))\n')
    executable.chmod(0o755)
    os.environ['OLIST_PYTHON'] = str(executable)

    from airflow.dag_processing.dagbag import DagBag
    bag = DagBag(dag_folder=str(root / 'pipelines/data_pipeline.py'))
    assert not bag.import_errors, bag.import_errors
    dag = bag.dags['olist_end_to_end_pipeline']
    expected = ['download_olist_data', 'validate_olist_data', 'clean_olist_data',
        'prepare_curated_data', 'build_forecasting_features', 'build_recommendation_features',
        'run_forecasting_baselines', 'run_recommendation_baselines']
    assert [task.task_id for task in dag.topological_sort()] == expected
    assert dag.max_active_runs == 1 and dag.max_active_tasks == 1
    assert dag.schedule is None and not dag.catchup
    for index, task_id in enumerate(expected):
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == ({expected[index + 1]} if index + 1 < len(expected) else set())
        assert task.trigger_rule == 'all_success'
        assert task.append_env and not task.do_xcom_push
        assert task.skip_on_exit_code == []
        assert Path(task.cwd) == root

    # Invalid periods must be rejected before a pipeline run is accepted.
    try:
        dag.params.get_param('test_days').resolve(0)
    except Exception as error:
        assert type(error).__name__ == 'ParamValidationError', type(error)
    else:
        raise AssertionError('Nonpositive test_days accepted')

    params = dag.params.validate()
    marker = temporary / 'must_not_be_created'
    customer = 'customer"; touch "' + str(marker) + '"; echo "'
    params.update(validation_days=14, test_days=7, top_k=3,
                  recommendation_customer_id=customer)
    records = {}
    for task in dag.topological_sort():
        task.render_template_fields({'params': params})
        record_file = temporary / (task.task_id + '.json')
        env = {**os.environ, **task.env, 'TEST_RECORD_FILE': str(record_file)}
        subprocess.run(['bash', '-c', task.bash_command], cwd=task.cwd, env=env,
                       check=True, capture_output=True, text=True)
        args = json.loads(record_file.read_text())
        assert Path(args[0]).is_file(), args[0]
        records[task.task_id] = args
    assert not marker.exists(), 'Customer parameter was interpreted as shell code'
    feature_args = records['build_forecasting_features']
    assert feature_args[feature_args.index('--validation-days') + 1] == '14'
    assert feature_args[feature_args.index('--test-days') + 1] == '7'
    rec_args = records['build_recommendation_features']
    assert rec_args[rec_args.index('--customer-id') + 1] == customer
    for task_id in ['build_recommendation_features', 'run_recommendation_baselines']:
        args = records[task_id]
        assert args[args.index('--top-k') + 1] == '3'
    for task_id in ['run_forecasting_baselines', 'run_recommendation_baselines']:
        args = records[task_id]
        assert args[args.index('--tracking-uri') + 1] == f'sqlite:///{root / "data/baselines/mlflow.db"}'
    training_bag = DagBag(dag_folder=str(root / 'pipelines/training_pipeline.py'))
    assert not training_bag.import_errors, training_bag.import_errors
    training = training_bag.dags['olist_production_training']
    training_order = ['validate_training_inputs', 'evaluate_forecasting_baselines',
        'evaluate_recommendation_baselines', 'train_forecasting_model', 'train_recommendation_model', 'compare_models']
    assert [task.task_id for task in training.topological_sort()] == training_order
    assert training.max_active_runs == 1 and training.max_active_tasks == 1
    assert training.schedule is None
    training_params = training.params.validate()
    training_params.update(max_train_rows=1000, forecast_estimators=5, ranking_estimators=5)
    for task in training.topological_sort():
        task.render_template_fields({'params': training_params, 'run_id': 'manual__example'})
        record_file = temporary / (task.task_id + '.json')
        subprocess.run(['bash', '-c', task.bash_command], cwd=task.cwd,
                       env={**os.environ, **task.env, 'TEST_RECORD_FILE': str(record_file)},
                       check=True, capture_output=True, text=True)
        args = json.loads(record_file.read_text())
        assert Path(args[0]).is_file()
        if task.task_id != 'validate_training_inputs':
            assert args[args.index('--comparison-id') + 1] == 'manual__example'
        if task.task_id == 'train_forecasting_model':
            assert args[args.index('--max-train-rows') + 1] == '1000'
    print('Both DAGs, dependencies, parameter validation and shell commands verified')
'''
        result = subprocess.run([sys.executable, "-c", check], cwd=root,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
