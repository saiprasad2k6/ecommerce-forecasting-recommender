"""HTTP inference contracts using real approved baseline bundles and historical fixtures."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app, create_app
from api.service import InferencePaths, InferenceService
from src.models.common import publish_model
from src.models.forecasting.forecasting_baseline import BaselineForecaster
from src.models.recommendation.recommendation_baseline import BaselineRecommender
from src.versioning import snapshots
from test_production_models import forecast_fixture, warehouse_fixture


class InferenceApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patcher = patch.object(snapshots, 'ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        frame, _ = forecast_fixture(self.root / 'features')
        self.date = str(frame.purchase_date.min().date())
        self.product = str(frame.iloc[0].product_id)
        self.state = str(frame.iloc[0].customer_state)
        self.features = self.root / 'features/train.parquet'
        self.warehouse = self.root / 'olist.duckdb'
        warehouse_fixture(self.warehouse)
        models = self.root / 'models'
        for task, filename, model in (
            ('forecasting', 'forecast_model.pkl', BaselineForecaster('last_day')),
            ('recommendation', 'recommender.pkl', BaselineRecommender('purchase_history')),
        ):
            publish_model(model, models / filename, False, metadata={
                'task': task, 'approved_for_inference': True, 'model_name': task + '_baseline',
                'model_role': 'baseline', 'model_run_id': task + '-run',
                'training_pipeline_run_id': 'train-001', 'candidate_limit': 5, 'k': 3,
            })
        self.service = InferenceService(InferencePaths(models, self.features, self.warehouse))
        self.client = TestClient(create_app(self.service))

    def forecast_payload(self):
        return {'as_of': self.date, 'items': [{'product_id': self.product, 'customer_state': self.state}]}

    def test_forecast_returns_real_model_prediction_and_provenance(self):
        response = self.client.post('/v1/inference/forecast', json=self.forecast_payload())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body['model']['training_pipeline_run_id'], 'train-001')
        self.assertEqual(body['model']['model_role'], 'baseline')
        from datetime import date, timedelta
        self.assertEqual(body['predictions'][0]['target_date'], str(date.fromisoformat(self.date) + timedelta(days=1)))
        import pandas as pd
        frame = pd.read_parquet(self.features)
        expected = frame.loc[(frame.product_id == self.product) & (frame.customer_state == self.state)
                             & (frame.purchase_date == pd.Timestamp(self.date)), 'demand'].iloc[0]
        self.assertEqual(body['predictions'][0]['prediction'], expected)
        self.assertTrue(body['request_id'])
        self.assertFalse((self.root / 'data/predictions').exists())

    def test_forecast_missing_pairs_returns_404_without_partial_result(self):
        payload = self.forecast_payload()
        payload['items'].append({'product_id': 'missing', 'customer_state': 'SP'})
        response = self.client.post('/v1/inference/forecast', json=payload)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()['error']['code'], 'features_not_found')

    def test_invalid_requests_use_documented_error_shape(self):
        payload = self.forecast_payload()
        bad = [dict(payload, as_of='not-a-date'), dict(payload, items=[]),
               dict(payload, items=payload['items'] * 2), dict(payload, arbitrary_path='/tmp/file'),
               dict(payload, items=[{'product_id':'p1','customer_state':'XX'}])]
        for body in bad:
            with self.subTest(body=body):
                response = self.client.post('/v1/inference/forecast', json=body)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()['error']['code'], 'validation_error')

    def test_recommendations_respect_rank_k_and_model(self):
        response = self.client.post('/v1/inference/recommendations', json={
            'as_of': '2018-08-01', 'customer_unique_id':'c1', 'k':2,
        })
        self.assertEqual(response.status_code, 200, response.text)
        body=response.json()
        self.assertFalse(body['cold_start'])
        self.assertEqual(body['k'], 2)
        self.assertEqual([row['rank'] for row in body['recommendations']], [1,2])
        self.assertEqual(body['recommendations'][0]['product_id'], 'p1')
        self.assertEqual(body['model']['model_run_id'], 'recommendation-run')

    def test_unknown_customer_and_default_k(self):
        response = self.client.post('/v1/inference/recommendations', json={
            'as_of':'2018-08-01','customer_unique_id':'brand-new-customer',
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()['cold_start'])
        self.assertEqual(response.json()['k'], 3)
        self.assertEqual(len(response.json()['recommendations']), 3)

    def test_k_must_match_active_candidate_limit_and_be_an_integer(self):
        for k in [0,6,101,True,'2']:
            response = self.client.post('/v1/inference/recommendations',json={
                'as_of':'2018-08-01','customer_unique_id':'c1','k':k,
            })
            self.assertEqual(response.status_code,422,response.text)

    def test_history_before_cutoff_is_required(self):
        response = self.client.post('/v1/inference/recommendations',json={
            'as_of':'1900-01-01','customer_unique_id':'c1',
        })
        self.assertEqual(response.status_code,404,response.text)
        self.assertEqual(response.json()['error']['code'],'history_not_found')

    def test_liveness_and_readiness_are_distinct(self):
        self.assertEqual(self.client.get('/health').json(),{'status':'ok'})
        self.assertEqual(len(self.client.get('/v1/models').json()['models']),2)
        (self.service.paths.models / 'forecast_model.pkl').unlink()
        self.assertEqual(self.client.get('/health').status_code,200)
        self.assertEqual(self.client.get('/v1/models').status_code,503)
        self.assertEqual(self.client.post('/v1/inference/forecast',json=self.forecast_payload()).status_code,503)

    def test_busy_workspace_returns_503(self):
        directory = self.root / '.dvc'
        directory.mkdir(exist_ok=True)
        (directory / 'workspace.json').write_text(json.dumps({'owner':'train:running'}))
        response=self.client.post('/v1/inference/forecast',json=self.forecast_payload())
        self.assertEqual(response.status_code,503,response.text)
        self.assertEqual(self.client.get('/health').status_code,200)

    def test_latest_activation_is_loaded_on_each_request(self):
        path = self.service.paths.models / 'forecast_model.pkl'
        model = BaselineForecaster('moving_average_7')
        publish_model(model,path,True,metadata={
            'task':'forecasting','approved_for_inference':True,'model_name':'replacement',
            'model_role':'candidate','model_run_id':'new-model','training_pipeline_run_id':'train-002',
        })
        response=self.client.post('/v1/inference/forecast',json=self.forecast_payload())
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['model']['model_run_id'],'new-model')

    def test_swagger_and_exported_contract_match_application(self):
        self.assertEqual(self.client.get('/docs').status_code,200)
        schema=self.client.get('/openapi.json').json()
        self.assertEqual(schema['info']['version'],'1.0.0')
        self.assertEqual(set(schema['paths']),{'/health','/v1/models','/v1/inference/forecast','/v1/inference/recommendations'})
        root=Path(__file__).resolve().parents[1]
        self.assertEqual(json.loads((root/'documentation/inference-openapi.json').read_text()),app.openapi())
