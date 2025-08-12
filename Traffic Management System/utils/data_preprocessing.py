# data_preprocessing.py
import pandas as pd

def clean_data(df):
    """
    Cleans the input DataFrame by handling missing values and encoding categorical variables.
    """
    df.fillna(df.median(numeric_only=True), inplace=True)

    if 'weather_main' in df.columns:
        df = pd.get_dummies(df, columns=['weather_main'], drop_first=True)

    return df

def feature_engineer(df):
    """
    Adds derived features to enhance model performance.
    Assumes df contains columns like 'timestamp', 'traffic_volume', 'weather_main', etc.
    """
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    if 'traffic_volume' in df.columns:
        df['traffic_level'] = pd.cut(df['traffic_volume'],
                                     bins=[0, 200, 400, 600, 800, float('inf')],
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    if 'weather_main' in df.columns:
        df['severe_weather'] = df['weather_main'].isin(['Rain', 'Snow', 'Storm']).astype(int)

    return df
