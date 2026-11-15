import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40,
}

url = "https://ride-duration.onrender.com/predict"   # paste your URL + /predict
response = requests.post(url, json=ride)
print(response.json())