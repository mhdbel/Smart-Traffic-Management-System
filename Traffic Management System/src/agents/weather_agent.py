import requests
import os

class WeatherAgent:
    """
    Agent for fetching and analyzing weather data.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY not set in environment variables.")

    def fetch_weather_data(self, city: str) -> dict | None:
        """
        Fetches current weather data for a given city from OpenWeatherMap.
        """
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None

    def analyze_weather_impact(self, weather_data: dict) -> str:
        """
        Analyzes weather data for potential traffic impact.
        """
        if not weather_data:
            return "Unable to retrieve weather data."

        weather = weather_data.get("weather", [{}])[0].get("main", "")
        rain = weather_data.get("rain", {}).get("1h", 0)
        temp = weather_data.get("main", {}).get("temp", 25)

        if weather.lower() in ["rain", "thunderstorm"] or rain > 0:
            return f"Rain detected ({rain}mm). Expect slower traffic and possible delays."
        elif temp < 0:
            return "Freezing temperatures detected. Potential for icy roads and slow traffic."
        elif weather.lower() in ["snow"]:
            return "Snow detected. Expect hazardous driving conditions and delays."
        else:
            return "Weather conditions normal. No significant impact on traffic expected."
