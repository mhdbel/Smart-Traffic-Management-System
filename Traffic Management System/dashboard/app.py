# dashboard/app.py
"""
Traffic Dashboard - Interactive visualization using Dash/Plotly.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed_data"
DEFAULT_DATA_FILE = "processed_traffic_data.csv"

DATA_PATH = os.getenv(
    "TRAFFIC_DATA_PATH",
    str(DATA_DIR / DEFAULT_DATA_FILE)
)

# Required columns
REQUIRED_COLUMNS = ['date_time', 'traffic_volume']
OPTIONAL_COLUMNS = ['location', 'weather', 'temperature', 'event_impact']

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:5000")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_traffic_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load traffic data from CSV file.
    
    Args:
        path: Optional path to data file
        
    Returns:
        Validated DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid
    """
    data_path = Path(path or DATA_PATH)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        df = validate_dataframe(df)
        logger.info(f"Loaded {len(df)} records from {data_path}")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("Data file is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing data file: {e}")


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the traffic dataframe."""
    # Check required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Parse dates
    if not pd.api.types.is_datetime64_any_dtype(df['date_time']):
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
    
    # Remove invalid dates
    invalid_dates = df['date_time'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"Removed {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=['date_time'])
    
    # Validate traffic_volume
    if not pd.api.types.is_numeric_dtype(df['traffic_volume']):
        df['traffic_volume'] = pd.to_numeric(df['traffic_volume'], errors='coerce')
    
    # Remove invalid values
    df = df[df['traffic_volume'] >= 0]
    df = df.dropna(subset=['traffic_volume'])
    
    # Sort by date
    df = df.sort_values('date_time').reset_index(drop=True)
    
    return df


# =============================================================================
# CHART CREATION
# =============================================================================

def create_main_chart(df: pd.DataFrame) -> go.Figure:
    """Create the main traffic volume chart."""
    fig = go.Figure()
    
    # Main line
    fig.add_trace(go.Scatter(
        x=df['date_time'],
        y=df['traffic_volume'],
        mode='lines',
        name='Traffic Volume',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)',
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Volume: %{y:,.0f}<extra></extra>'
    ))
    
    # Moving average
    if len(df) >= 7:
        ma = df['traffic_volume'].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df['date_time'],
            y=ma,
            mode='lines',
            name='7-Period Moving Avg',
            line=dict(color='#e74c3c', width=2, dash='dash'),
        ))
    
    fig.update_layout(
        title=dict(text='📊 Traffic Volume Over Time', x=0.5, font=dict(size=20)),
        xaxis_title='Date & Time',
        yaxis_title='Traffic Volume',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig


def create_hourly_chart(df: pd.DataFrame) -> go.Figure:
    """Create hourly pattern chart."""
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    hourly = df.groupby('hour')['traffic_volume'].mean().reset_index()
    
    fig = px.bar(
        hourly,
        x='hour',
        y='traffic_volume',
        title='🕐 Average Traffic by Hour',
        labels={'hour': 'Hour of Day', 'traffic_volume': 'Avg Volume'},
        color='traffic_volume',
        color_continuous_scale='Blues',
    )
    
    fig.update_layout(template='plotly_white', showlegend=False, coloraxis_showscale=False)
    
    return fig


def create_daily_chart(df: pd.DataFrame) -> go.Figure:
    """Create daily pattern chart."""
    df = df.copy()
    df['day_name'] = df['date_time'].dt.day_name()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily = df.groupby('day_name')['traffic_volume'].mean().reindex(day_order).reset_index()
    
    fig = px.bar(
        daily,
        x='day_name',
        y='traffic_volume',
        title='📅 Average Traffic by Day of Week',
        labels={'day_name': 'Day', 'traffic_volume': 'Avg Volume'},
        color='traffic_volume',
        color_continuous_scale='Greens',
    )
    
    fig.update_layout(template='plotly_white', showlegend=False, coloraxis_showscale=False)
    
    return fig


def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create hour-by-day heatmap."""
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    df['day_name'] = df['date_time'].dt.day_name()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    pivot = df.pivot_table(
        values='traffic_volume',
        index='day_name',
        columns='hour',
        aggfunc='mean'
    ).reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        hovertemplate='Day: %{y}<br>Hour: %{x}<br>Volume: %{z:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='🗓️ Traffic Patterns Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        template='plotly_white',
    )
    
    return fig


# =============================================================================
# LAYOUT COMPONENTS
# =============================================================================

def create_kpi_cards(df: pd.DataFrame) -> html.Div:
    """Create KPI summary cards."""
    avg_volume = df['traffic_volume'].mean()
    max_volume = df['traffic_volume'].max()
    min_volume = df['traffic_volume'].min()
    total_records = len(df)
    
    # Find peak hour
    df_copy = df.copy()
    df_copy['hour'] = df_copy['date_time'].dt.hour
    peak_hour = df_copy.groupby('hour')['traffic_volume'].mean().idxmax()
    
    return html.Div([
        _kpi_card("📊 Average", f"{avg_volume:,.0f}", "vehicles/period"),
        _kpi_card("📈 Peak", f"{max_volume:,.0f}", "max recorded"),
        _kpi_card("📉 Minimum", f"{min_volume:,.0f}", "min recorded"),
        _kpi_card("🕐 Peak Hour", f"{peak_hour}:00", "busiest time"),
        _kpi_card("📋 Records", f"{total_records:,}", "data points"),
    ], className="kpi-row")


def _kpi_card(title: str, value: str, subtitle: str) -> html.Div:
    """Create a single KPI card."""
    return html.Div([
        html.P(title, className="kpi-title"),
        html.H2(value, className="kpi-value"),
        html.P(subtitle, className="kpi-subtitle"),
    ], className="kpi-card")


def create_filters(df: pd.DataFrame) -> html.Div:
    """Create filter controls."""
    min_date = df['date_time'].min().date()
    max_date = df['date_time'].max().date()
    
    locations = []
    if 'location' in df.columns:
        locations = [{'label': loc, 'value': loc} for loc in df['location'].unique()]
    
    return html.Div([
        html.Div([
            html.Label("📅 Date Range"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=min_date,
                max_date_allowed=max_date,
                start_date=min_date,
                end_date=max_date,
                display_format='YYYY-MM-DD',
            ),
        ], className="filter-item"),
        
        html.Div([
            html.Label("⏱️ Aggregation"),
            dcc.Dropdown(
                id='aggregation',
                options=[
                    {'label': 'Raw Data', 'value': ''},
                    {'label': 'Hourly', 'value': 'H'},
                    {'label': 'Daily', 'value': 'D'},
                    {'label': 'Weekly', 'value': 'W'},
                ],
                value='D',
                clearable=False,
            ),
        ], className="filter-item"),
        
        html.Div([
            html.Label("📍 Location"),
            dcc.Dropdown(
                id='location-filter',
                options=locations,
                value=None,
                multi=True,
                placeholder="All locations",
            ),
        ], className="filter-item") if locations else html.Div(),
        
    ], className="filters")


def create_error_layout(title: str, message: str) -> html.Div:
    """Create error state layout."""
    return html.Div([
        html.Div([
            html.H1("⚠️ Dashboard Error"),
            html.H2(title, style={'color': '#e74c3c'}),
            html.P(message),
            html.Button("Reload Page", id="reload-btn", className="reload-button",
                       **{'data-onclick': 'location.reload()'}),
        ], className="error-container"),
    ], className="dashboard")


def create_layout(df: pd.DataFrame) -> html.Div:
    """Create the main dashboard layout."""
    return html.Div([
        # Header
        html.Div([
            html.H1("🚦 City Traffic Dashboard"),
            html.Div(id='live-indicator'),
        ], className="header"),
        
        # Filters
        create_filters(df),
        
        # KPIs
        html.Div(id='kpi-section', children=create_kpi_cards(df)),
        
        # Main Chart
        html.Div([
            dcc.Graph(id='main-chart', figure=create_main_chart(df)),
        ], className="chart-container"),
        
        # Secondary Charts Row
        html.Div([
            html.Div([
                dcc.Graph(id='hourly-chart', figure=create_hourly_chart(df)),
            ], className="half-chart"),
            html.Div([
                dcc.Graph(id='daily-chart', figure=create_daily_chart(df)),
            ], className="half-chart"),
        ], className="chart-row"),
        
        # Heatmap
        html.Div([
            dcc.Graph(id='heatmap', figure=create_heatmap(df)),
        ], className="chart-container"),
        
        # Data store
        dcc.Store(id='traffic-data', data=df.to_json(date_format='iso', orient='records')),
        
        # Auto-refresh interval
        dcc.Interval(id='refresh-interval', interval=5*60*1000, n_intervals=0),
        
        # Footer
        html.Div([
            html.P(f"Data range: {df['date_time'].min().strftime('%Y-%m-%d')} to {df['date_time'].max().strftime('%Y-%m-%d')}"),
            html.P(f"Total records: {len(df):,}"),
        ], className="footer"),
        
    ], className="dashboard")


# =============================================================================
# CALLBACKS
# =============================================================================

def register_callbacks(app: Dash):
    """Register all dashboard callbacks."""
    
    @app.callback(
        Output('main-chart', 'figure'),
        Output('kpi-section', 'children'),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date'),
        Input('aggregation', 'value'),
        Input('location-filter', 'value'),
        State('traffic-data', 'data'),
    )
    def update_dashboard(start_date, end_date, aggregation, locations, data):
        """Update dashboard based on filters."""
        if not data:
            raise PreventUpdate
        
        # Load data
        df = pd.read_json(data, orient='records')
        df['date_time'] = pd.to_datetime(df['date_time'])
        
        # Apply filters
        if start_date and end_date:
            mask = (df['date_time'].dt.date >= pd.to_datetime(start_date).date()) & \
                   (df['date_time'].dt.date <= pd.to_datetime(end_date).date())
            df = df[mask]
        
        if locations and 'location' in df.columns:
            df = df[df['location'].isin(locations)]
        
        # Aggregate if specified
        if aggregation:
            df = df.set_index('date_time').resample(aggregation).agg({
                'traffic_volume': 'mean'
            }).reset_index()
            df = df.dropna()
        
        if df.empty:
            raise PreventUpdate
        
        return create_main_chart(df), create_kpi_cards(df)
    
    @app.callback(
        Output('live-indicator', 'children'),
        Input('refresh-interval', 'n_intervals'),
    )
    def update_live_indicator(n):
        """Update the live status indicator."""
        return html.Span([
            html.Span("🟢 ", style={'fontSize': '10px'}),
            html.Span(
                f"Last updated: {datetime.now().strftime('%H:%M:%S')}",
                style={'fontSize': '11px', 'color': '#7f8c8d'}
            ),
        ])


# =============================================================================
# APP FACTORY
# =============================================================================

def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title="Traffic Dashboard",
        update_title="Loading...",
    )
    
    try:
        df = load_traffic_data()
        app.layout = create_layout(df)
        register_callbacks(app)
        logger.info("Dashboard initialized successfully")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        app.layout = create_error_layout(
            "Data Not Found",
            str(e)
        )
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        app.layout = create_error_layout(
            "Data Error",
            str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        app.layout = create_error_layout(
            "Initialization Error",
            "An unexpected error occurred while loading the dashboard."
        )
    
    return app


# =============================================================================
# MAIN
# =============================================================================

# Create app instance
app = create_app()
server = app.server  # For WSGI servers like gunicorn

if __name__ == '__main__':
    debug_mode = os.getenv("DASH_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 8050))
    host = os.getenv("HOST", "127.0.0.1")
    
    if debug_mode:
        logger.warning("Running in DEBUG mode - do not use in production!")
    
    logger.info(f"Starting dashboard on http://{host}:{port}")
    
    app.run_server(
        host=host,
        port=port,
        debug=debug_mode,
    )
