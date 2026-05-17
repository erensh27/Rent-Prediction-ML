import pytest
from app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    rv = client.get('/health')
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['status'] == 'ok'

def test_predict_api(client):
    payload = {
        "BHK": 2,
        "Size": 1000,
        "Bathroom": 2,
        "City": "Mumbai",
        "Area Type": "Super Area",
        "Furnishing Status": "Semi-Furnished",
        "Tenant Preferred": "Bachelors/Family",
        "Point of Contact": "Contact Owner"
    }
    rv = client.post('/api/predict',
                     data=json.dumps(payload),
                     content_type='application/json')
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'predicted_rent' in data
    assert data['currency'] == 'INR'

def test_predict_api_missing_fields(client):
    payload = {
        "BHK": 2,
        "Size": 1000
    }
    rv = client.post('/api/predict',
                     data=json.dumps(payload),
                     content_type='application/json')
    assert rv.status_code == 400
    data = rv.get_json()
    assert 'error' in data
