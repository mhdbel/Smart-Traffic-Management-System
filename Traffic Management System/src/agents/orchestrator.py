from agents.traffic_agent import TrafficAgent
from agents.weather_agent import WeatherAgent
from agents.event_agent import EventAgent
from agents.routing_agent import RoutingAgent

class Orchestrator:
    def __init__(self):
        self.traffic_agent = TrafficAgent()
        self.weather_agent = WeatherAgent()
        self.event_agent = EventAgent()
        self.routing_agent = RoutingAgent()

    def handle_user_query(self, origin, destination):
        # Step 1: Analyze traffic
        traffic_response = self.traffic_agent.reason_and_act(origin, destination)

        # Step 2: Check weather
        weather_data = self.weather_agent.fetch_weather_data(destination)
        weather_response = self.weather_agent.analyze_weather_impact(weather_data)

        # Step 3: Identify events
        event_data = self.event_agent.fetch_event_data(destination)
        event_response = self.event_agent.analyze_event_impact(event_data)

        # Step 4: Suggest routes
        routes = self.routing_agent.suggest_routes(origin, destination, traffic_data)

        # Combine responses
        return {
            "traffic": traffic_response,
            "weather": weather_response,
            "events": event_response,
            "routes": routes,
        }
