import requests

class WeatherAgent:
    def fetch_weather_data(self, city):
        url = f" https://api.openweathermap.org/data/2.5/weather?q={city}&appid=YOUR_API_KEY&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching weather  {response.status_code}")
            return None

    def analyze_weather_impact(self, weather_data):
        weather_main = weather_data.get("weather", [{}])[0].get("main", "Clear")
        temp = weather_data.get("main", {}).get("temp", 25)
        rain = weather_data.get("rain", {}).get("1h", 0)

        if weather_main in ["Rain", "Thunderstorm"]:
            return "Adverse weather conditions detected. Expect delays."
        elif temp > 35:
            return "High temperatures may cause discomfort. Plan accordingly."
        else:
            return "Weather conditions are favorable."
