import requests

def test_predict_endpoint():
    response = requests.post("http://127.0.0.1:5000/predict", json={"temp": 25})
    assert response.status_code == 200