# scrape_event_data.py
import requests
from bs4 import BeautifulSoup

def get_event_data(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        events = []
        for event in soup.find_all("div", class_="event-item"):
            title = event.find("h2").text.strip()
            date = event.find("span", class_="date").text.strip()
            location = event.find("span", class_="location").text.strip()
            events.append({"title": title, "date": date, "location": location})
        return events
    else:
        print(f"Error scraping event  {response.status_code}")
        return None

# Example usage
url = "https://www.guichet.ma/events "
events = get_event_data(url)
with open("data/raw_data/event_data.json", "w") as f:
    json.dump(events, f)