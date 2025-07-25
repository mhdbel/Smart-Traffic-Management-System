import os
import requests

class EventAgent:
    """
    Agent for fetching and analyzing public events for traffic impact.
    """

    def __init__(self):
        self.token = os.getenv("EVENTBRITE_API_KEY")
        if not self.token:
            raise ValueError("EVENTBRITE_API_KEY not set in environment variables.")

    def fetch_event_data(self, city: str) -> dict | None:
        """
        Fetches event data from Eventbrite API for a given city.
        """
        url = f"https://www.eventbriteapi.com/v3/events/search/?location.address={city}&token={self.token}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching event data: {e}")
            return None

    def analyze_event_impact(self, event_data: dict) -> str:
        """
        Analyzes events for likely traffic impact.
        """
        events = event_data.get("events", []) if event_data else []
        if events:
            # You can enhance this by counting events, checking expected attendees, etc.
            return f"{len(events)} upcoming events detected. Expect increased traffic."
        else:
            return "No significant events in the area."
