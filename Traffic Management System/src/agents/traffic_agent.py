import os
import requests
import joblib
import pandas as pd

class TrafficAgent:
    """
    TrafficAgent handles traffic data fetching, congestion prediction, and action reasoning.
    """

    def __init__(self, model_path="../models/traffic_model.pkl", threshold=0.7):
        self.model = joblib.load(model_path)
        self.threshold = threshold
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    def fetch_traffic_data(self, origin, destination):
        """
        Fetch traffic data between origin and destination using Google Maps API.
        """
        if not self.api_key:
            raise ValueError("Google Maps API key not set in environment variables.")
        url = (
            f"https://maps.googleapis.com/maps/api/directions/json?"
            f"origin={origin}&destination={destination}&key={self.api_key}"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching traffic data: {e}")
            return None

    def extract_features(self, traffic_data):
        """
        Extracts relevant features for congestion prediction.
        NOTE: Google Maps Directions API does not provide weather data.
        You need to integrate a weather API if you want weather features.
        Below is a placeholder for features.
        """
        # Placeholder values; replace with actual integration as needed
        features = {
            "temp": 25,  # Example static value
            "rain_1h": 0,
            "clouds_all": 50,
        }
        return features

    def predict_congestion(self, input_features):
        """
        Predicts congestion level using the loaded model.
        """
        # Convert input_features to DataFrame if model expects it
        if isinstance(input_features, dict):
            input_data = pd.DataFrame([input_features])
        else:
            input_data = input_features
        prediction = self.model.predict(input_data)
        return prediction[0] if hasattr(prediction, "__getitem__") else prediction

    def reason_and_act(self, origin, destination):
        """
        Combines fetching and prediction to recommend actions.
        """
        traffic_data = self.fetch_traffic_data(origin, destination)
        if not traffic_data:
            return "Unable to fetch traffic data."

        features = self.extract_features(traffic_data)
        prediction = self.predict_congestion(features)
        if prediction > self.threshold:
            return "High traffic predicted. Consider alternative routes."
        else:
            return "Traffic is normal. Safe to proceed."
