"""Parse real DAGs and verify release boundaries, cleanup and shell quoting."""
from pathlib import Path
import subprocess
import sys
import unittest


class DataPipelineTests(unittest.TestCase):
    def test_dag_dependencies_parameters_and_commands(self):
        root = Path(__file__).resolve().parents[1]
        check = r'''
import json, os, subprocess, sys, tempfile
from pathlib import Path
root = Path.cwd()
with tempfile.TemporaryDirectory() as directory:
    temporary = Path(directory)
    os.environ['AIRFLOW_HOME'] = str(temporary / 'airflow')
    os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'
    os.environ['OLIST_PROJECT_ROOT'] = str(root)
    executable = temporary / 'python with spaces'
    executable.write_text('#!' + sys.executable + '\n'
        'import json, os, sys\nfrom pathlib import Path\n'
        'Path(os.environ["TEST_RECORD_FILE"]).write_text(json.dumps(sys.argv[1:]))\n')
    executable.chmod(0o755)
    os.environ['OLIST_PYTHON'] = str(executable)
    from airflow.dag_processing.dagbag import DagBag
    bag = DagBag(dag_folder=str(root / 'pipelines'))
    assert not bag.import_errors, bag.import_errors
    data = bag.dags['olist_data_preparation']
    preparation = ['begin_preparation', 'download_olist_data', 'validate_olist_data', 'publish_raw_version',
                   'clean_olist_data', 'prepare_curated_data', 'build_forecasting_features',
                   'build_recommendation_features', 'publish_prepared_version']
    training = bag.dags['olist_model_training']
    registered = json.loads((root / 'configs/production_models.json').read_text())
    model_tasks = ['select_prepared_version', 'evaluate_forecasting_baselines', 'evaluate_recommendation_baselines']
    model_tasks += [f'train_{task}_{model}' for task, models in registered.items() for model in models]
    model_tasks += ['compare_models', 'preserve_run_snapshot', 'backup_mlflow']
    for dag, order in [(data, preparation), (training, model_tasks)]:
        assert dag.max_active_runs == 1 and dag.max_active_tasks == 1
        assert dag.schedule is None and not dag.catchup
        assert [t.task_id for t in dag.topological_sort() if t.task_id not in ['release_workspace','workflow_failed']] == order
        cleanup = dag.get_task('release_workspace')
        watcher = dag.get_task('workflow_failed')
        assert cleanup.trigger_rule == 'all_done'
        assert watcher.trigger_rule == 'one_failed'
        assert cleanup.upstream_task_ids == set(order)
        assert watcher.upstream_task_ids == set(order) | {'release_workspace'}
        assert dag.leaves == [watcher], 'Cleanup success must never mask a failed business task'
        for first, second in zip(order, order[1:]):
            assert second in dag.get_task(first).downstream_task_ids
    assert 'activate_models' not in training.task_ids
    try:
        training.params.validate()
    except Exception:
        pass
    else:
        raise AssertionError('Training accepted missing prepared_version')
    training.params.get_param('prepared_version').resolve('a' * 64)
    params = training.params.validate()
    params.update(max_train_rows=1000, forecast_estimators=5, release=True)
    training_run_id = 'training__example"; touch "' + str(temporary / 'training_injected') + '"; echo "'
    for task in training.topological_sort():
        if task.task_id == 'workflow_failed': continue
        task.render_template_fields({'params': params, 'run_id': training_run_id})
        record = temporary / (task.task_id + '.json')
        subprocess.run(['bash', '-c', task.bash_command], cwd=task.cwd,
            env={**os.environ, **task.env, 'TEST_RECORD_FILE': str(record)}, check=True, capture_output=True, text=True)
        args = json.loads(record.read_text())
        assert Path(args[0]).is_file()
        if task.task_id == 'backup_mlflow':
            assert args[1:] == ['backup-mlflow']
        else:
            assert args[args.index('--pipeline-run-id') + 1] == training_run_id
        if task.task_id.startswith(('evaluate_', 'train_')):
            model_task = args[args.index('--task') + 1]
            model = args[args.index('--model') + 1]
            assert model_task in registered
            assert model == 'baselines' or model in registered[model_task]
        if task.task_id == 'release_workspace':
            assert args[1] == 'release-workspace'
            assert args[args.index('--kind') + 1] == 'train'
        if task.task_id == 'select_prepared_version':
            assert args[args.index('--prepared-version') + 1] == 'a' * 64
            assert '--development' not in args
            assert json.loads(task.env['OLIST_PARAMETERS'])['max_train_rows'] == 1000
    assert not (temporary / 'training_injected').exists()
    marker = temporary / 'must_not_exist'
    customer = 'customer"; touch "' + str(marker) + '"; echo "'
    params = data.params.validate()
    params.update(recommendation_customer_id=customer, top_k=3)
    for task in data.topological_sort():
        if task.task_id == 'workflow_failed': continue
        task.render_template_fields({'params': params, 'run_id': 'preparation__example'})
        record = temporary / (task.task_id + '.json')
        subprocess.run(['bash','-c',task.bash_command],cwd=task.cwd,
            env={**os.environ, **task.env, 'TEST_RECORD_FILE':str(record)}, check=True,capture_output=True,text=True)
        args = json.loads(record.read_text())
        if task.task_id == 'build_recommendation_features':
            assert args[args.index('--customer-id') + 1] == customer
            assert 'execute-preparation' in args
            assert '--' in args
    assert not marker.exists(), 'Trigger parameter was interpreted as shell code'
    activation = bag.dags['olist_model_activation']
    activation.params.get_param('training_run_id').resolve('training__example')
    task = activation.get_task('activate_models')
    task.render_template_fields({'params': activation.params.validate()})
    assert task.env['TRAINING_RUN_ID'] == 'training__example'
    assert bag.dags['olist_batch_inference'].task_ids == ['predict_forecasting','predict_recommendation']
    # Registering another model must extend the same Airflow workflow.
    other = temporary / 'project with spaces'
    (other / 'configs').mkdir(parents=True)
    registered['forecasting']['challenger'] = {'script':'forecasting/challenger.py','parameters':{}}
    (other / 'configs/production_models.json').write_text(json.dumps(registered))
    (other / 'configs/settings.yaml').write_text((root / 'configs/settings.yaml').read_text())
    os.environ['OLIST_PROJECT_ROOT'] = str(other)
    expanded = DagBag(dag_folder=str(root / 'pipelines/training_pipeline.py'))
    assert not expanded.import_errors, expanded.import_errors
    dag = expanded.dags['olist_model_training']
    assert 'train_forecasting_challenger' in dag.task_ids
    assert 'train_forecasting_challenger' in dag.get_task('release_workspace').upstream_task_ids
    dag.params.get_param('prepared_version').resolve('b' * 64)
    params = dag.params.validate()
    params['release'] = False
    task = dag.get_task('select_prepared_version')
    task.render_template_fields({'params': params, 'run_id': 'development__example'})
    record = temporary / 'development.json'
    subprocess.run(['bash', '-c', task.bash_command], cwd=task.cwd,
        env={**os.environ, **task.env, 'TEST_RECORD_FILE': str(record)},
        check=True, capture_output=True, text=True)
    args = json.loads(record.read_text())
    assert args[1] == 'begin-training'
    assert args.count('--development') == 1
    assert args[args.index('--prepared-version') + 1] == 'b' * 64
'''
        result = subprocess.run([sys.executable, "-c", check], cwd=root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
