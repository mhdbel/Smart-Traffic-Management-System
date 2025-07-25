import requests

class RoutingAgent:
    """
    Agent responsible for calculating optimal routes and alternatives.
    """

    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_route(self, origin, destination, avoid=None):
        """
        Fetches route data from Google Maps Directions API.
        Optionally, can avoid certain features like 'tolls', 'highways', or 'ferries'.
        """
        url = (
            f"https://maps.googleapis.com/maps/api/directions/json?"
            f"origin={origin}&destination={destination}&key={self.api_key}"
        )
        if avoid:
            url += f"&avoid={avoid}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching route: {e}")
            return None

    def get_alternative_routes(self, origin, destination):
        """
        Fetches alternative routes from Google Maps Directions API.
        """
        url = (
            f"https://maps.googleapis.com/maps/api/directions/json?"
            f"origin={origin}&destination={destination}&alternatives=true&key={self.api_key}"
        )
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching alternative routes: {e}")
            return None

    def analyze_routes(self, route_data):
        """
        Analyzes route data and returns summary statistics.
        """
        if not route_data or "routes" not in route_data:
            return "No route data available."

        routes = route_data["routes"]
        summaries = []
        for route in routes:
            summary = {
                "summary": route.get("summary", ""),
                "distance": route["legs"][0]["distance"]["text"] if route["legs"] else "",
                "duration": route["legs"][0]["duration"]["text"] if route["legs"] else "",
            }
            summaries.append(summary)
        return summaries
