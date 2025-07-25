class RoutingAgent:
    def suggest_routes(self, origin, destination, traffic_data):
        url = f" https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&avoid=tolls&key=YOUR_API_KEY"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error suggesting routes: {response.status_code}")
            return None
