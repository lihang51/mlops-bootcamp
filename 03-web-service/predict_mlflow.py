import pickle

import mlflow
from flask import Flask, request, jsonify

RUN_ID = "2a793dc693044d40806f3f946470f960"
mlflow.set_tracking_uri(
    "sqlite:////workspaces/mlops-bootcamp/02-experiment-tracking/mlflow.db"
)

model = mlflow.pyfunc.load_model(f"runs:/{RUN_ID}/models_mlflow")

dv_path = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID, artifact_path="preprocessor/preprocessor.b"
)
with open(dv_path, "rb") as f_in:
    dv = pickle.load(f_in)


def prepare_features(ride):
    features = {}
    features["PU_DO"] = f"{ride['PULocationID']}_{ride['DOLocationID']}"
    features["trip_distance"] = ride["trip_distance"]
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])


app = Flask("duration-prediction")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {"duration": pred, "model_version": RUN_ID}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)