import requests

class EventAgent:
    def fetch_event_data(self, city):
        url = f" https://www.eventbriteapi.com/v3/events/search/?location.address={city}&token=YOUR_API_KEY"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching event data: {response.status_code}")
            return None

    def analyze_event_impact(self, event_data):
        events = event_data.get("events", [])
        if len(events) > 0:
            return "Upcoming events detected. Expect increased traffic."
        else:
            return "No significant events in the area."
