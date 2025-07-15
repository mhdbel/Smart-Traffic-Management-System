# scrape_weather_data.py
import requests

# Replace with your OpenWeatherMap API key
API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather "

def get_weather_data(city):
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"  # Use metric units for temperature in Celsius
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching weather  {response.status_code}")
        return None

# Example usage
weather_data = get_weather_data("Rabat")
with open("data/raw_data/weather_data.json", "w") as f:
    json.dump(weather_data, f)