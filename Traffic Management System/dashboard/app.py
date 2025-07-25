# app.py
from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# Load data
df = pd.read_csv("../data/processed_data/processed_traffic_data.csv")

# Create visualization
fig = px.line(df, x="date_time", y="traffic_volume", title="Traffic Volume Over Time")

app.layout = html.Div([
    html.H1("City Traffic Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
