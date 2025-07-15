# scrape_traffic_data.py
import requests

# Replace with your Google Maps API key
API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"
BASE_URL = "https://maps.googleapis.com/maps/api/directions/json "

def get_traffic_data(origin, destination):
    params = {
        "origin": origin,
        "destination": destination,
        "key": API_KEY,
        "departure_time": "now"  # For real-time traffic conditions
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching traffic  {response.status_code}")
        return None

# Example usage
traffic_data = get_traffic_data("Rabat", "Casablanca")
with open("data/raw_data/traffic_data.json", "w") as f:
    json.dump(traffic_data, f)