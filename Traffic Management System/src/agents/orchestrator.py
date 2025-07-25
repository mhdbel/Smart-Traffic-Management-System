class Orchestrator:
    """
    Coordinates the interaction between all agents in the system.
    """

    def __init__(self, traffic_agent, event_agent, weather_agent, routing_agent):
        self.traffic_agent = traffic_agent
        self.event_agent = event_agent
        self.weather_agent = weather_agent
        self.routing_agent = routing_agent

    def run(self, origin, destination, city):
        """
        Runs the full analysis and recommendation cycle.
        """
        # Step 1: Get basic traffic prediction
        traffic_advice = self.traffic_agent.reason_and_act(origin, destination)

        # Step 2: Check for events
        event_data = self.event_agent.fetch_event_data(city)
        event_advice = self.event_agent.analyze_event_impact(event_data) if event_data else "Event data unavailable."

        # Step 3: Get weather impact
        weather_data = self.weather_agent.fetch_weather_data(city)
        weather_advice = self.weather_agent.analyze_weather_impact(weather_data) if weather_data else "Weather data unavailable."

        # Step 4: Get alternative routes
        alternatives = self.routing_agent.get_alternative_routes(origin, destination)
        route_summaries = self.routing_agent.analyze_routes(alternatives) if alternatives else "Route data unavailable."

        # Aggregate advisories
        result = {
            "traffic": traffic_advice,
            "event": event_advice,
            "weather": weather_advice,
            "routes": route_summaries
        }
        return result
