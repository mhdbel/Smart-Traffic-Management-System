import requests
import joblib
import pandas as pd

class TrafficAgent:
    def __init__(self, model_path="../models/traffic_model.pkl"):
        self.model = joblib.load(model_path)

    def fetch_traffic_data(self, origin, destination):
        url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key=YOUR_API_KEY"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching traffic data: {response.status_code}")
            return None

    def predict_congestion(self, input_data):
        prediction = self.model.predict(input_data)
        return prediction

    def reason_and_act(self, origin, destination):
        traffic_data = self.fetch_traffic_data(origin, destination)
        if not traffic_data:
            return "Unable to fetch traffic data."

        features = {
            "temp": traffic_data.get("temperature", 25),
            "rain_1h": traffic_data.get("precipitation", 0),
            "clouds_all": traffic_data.get("cloud_coverage", 50),
        }

        prediction = self.predict_congestion([features])
        if prediction > threshold:  # Define a threshold for high congestion
            return "High traffic predicted. Consider alternative routes."
        else:
            return "Traffic is normal. Safe to proceed."
