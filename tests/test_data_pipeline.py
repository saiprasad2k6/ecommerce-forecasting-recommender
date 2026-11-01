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
    dag = bag.dags['olist_data_preparation']
    expected = ['download_olist_data', 'validate_olist_data', 'clean_olist_data',
        'prepare_curated_data', 'build_forecasting_features', 'build_recommendation_features']
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
        dag.params.get_param('test_start').resolve('invalid-date')
    except Exception as error:
        assert type(error).__name__ == 'ParamValidationError', type(error)
    else:
        raise AssertionError('Invalid test_start accepted')

    params = dag.params.validate()
    marker = temporary / 'must_not_be_created'
    customer = 'customer"; touch "' + str(marker) + '"; echo "'
    params.update(validation_start='2018-06-01', test_start='2018-07-01', top_k=3,
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
    assert feature_args[feature_args.index('--validation-start') + 1] == '2018-06-01'
    assert feature_args[feature_args.index('--test-start') + 1] == '2018-07-01'
    assert feature_args[feature_args.index('--monitoring-start') + 1] == '2018-08-01'
    rec_args = records['build_recommendation_features']
    assert rec_args[rec_args.index('--customer-id') + 1] == customer
    assert rec_args[rec_args.index('--top-k') + 1] == '3'
    training_bag = DagBag(dag_folder=str(root / 'pipelines/training_pipeline.py'))
    assert not training_bag.import_errors, training_bag.import_errors
    training = training_bag.dags['olist_model_training']
    model_jobs = json.loads((root / 'configs/production_models.json').read_text())
    training_order = ['validate_training_inputs', 'evaluate_forecasting_baselines', 'evaluate_recommendation_baselines']
    training_order += [f'train_{task}_{model}' for task, models in model_jobs.items() for model in models]
    training_order += ['compare_models']
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
            assert args[args.index('--pipeline-run-id') + 1] == 'manual__example'
        if task.task_id in ['evaluate_recommendation_baselines', 'train_recommendation_lightgbm_ranker']:
            assert args[args.index('--validation-cutoff') + 1] == '2018-06-01'
            assert args[args.index('--test-cutoff') + 1] == '2018-07-01'
            assert args[args.index('--test-end') + 1] == '2018-08-01'
        if task.task_id == 'train_forecasting_lightgbm':
            assert args[args.index('--max-train-rows') + 1] == '1000'
    expanded_root = temporary / 'project with more models'
    (expanded_root / 'configs').mkdir(parents=True)
    model_jobs['forecasting']['another_model'] = {
        'script': 'forecasting/forecasting_another_model.py', 'parameters': {'max-train-rows': 'max_train_rows'},
    }
    (expanded_root / 'configs/production_models.json').write_text(json.dumps(model_jobs))
    (expanded_root / 'configs/settings.yaml').write_text((root / 'configs/settings.yaml').read_text())
    os.environ['OLIST_PROJECT_ROOT'] = str(expanded_root)
    expanded_bag = DagBag(dag_folder=str(root / 'pipelines/training_pipeline.py'))
    assert not expanded_bag.import_errors, expanded_bag.import_errors
    expanded = expanded_bag.dags['olist_model_training']
    expanded_order = [task.task_id for task in expanded.topological_sort()]
    assert 'train_forecasting_another_model' in expanded_order
    assert expanded_order.index('train_forecasting_another_model') < expanded_order.index('compare_models')
    assert expanded.get_task('compare_models').upstream_task_ids == {expanded_order[-2]}
    os.environ['OLIST_PROJECT_ROOT'] = str(root)
    inference_bag = DagBag(dag_folder=str(root / 'pipelines/batch_inference.py'))
    assert not inference_bag.import_errors, inference_bag.import_errors
    inference = inference_bag.dags['olist_batch_inference']
    assert [task.task_id for task in inference.topological_sort()] == ['predict_forecasting', 'predict_recommendation']
    assert inference.max_active_tasks == 1 and inference.schedule is None
    for task in inference.topological_sort():
        task.render_template_fields({'params': inference.params.validate(), 'run_id': 'inference__example'})
        record_file = temporary / (task.task_id + '.json')
        subprocess.run(['bash', '-c', task.bash_command], cwd=task.cwd,
            env={**os.environ, **task.env, 'TEST_RECORD_FILE': str(record_file)},
            check=True, capture_output=True, text=True)
        args = json.loads(record_file.read_text())
        assert Path(args[0]).is_file()
        assert args[args.index('--pipeline-run-id') + 1] == 'inference__example'
        assert args[args.index('--models-dir') + 1] == 'models'
        assert args[args.index('--as-of') + 1] == '2018-08-01'
        if task.task_id == 'predict_forecasting':
            assert args[args.index('--input-path') + 1] == 'data/features/forecasting/inference.parquet'
    print('Data, training and inference DAGs, model registration and shell commands verified')
'''
        result = subprocess.run([sys.executable, "-c", check], cwd=root,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
