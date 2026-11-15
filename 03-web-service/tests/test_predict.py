import predict

RIDE = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40,
}


def test_prepare_features():
    features = predict.prepare_features(RIDE)

    expected = {
        "PU_DO": "10_50",
        "trip_distance": 40,
    }
    assert features == expected


def test_predict_returns_positive_number():
    features = predict.prepare_features(RIDE)
    duration = predict.predict(features)

    assert isinstance(duration, float)
    assert duration > 0


def test_predict_endpoint():
    client = predict.app.test_client()
    response = client.post("/predict", json=RIDE)

    assert response.status_code == 200
    assert "duration" in response.get_json()