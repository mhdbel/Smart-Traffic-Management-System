# data_preprocessing.py
import pandas as pd

def preprocess_data(raw_file, processed_file):
    df = pd.read_csv(raw_file)
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['weather_main'], drop_first=True)
    # Save processed data
    df.to_csv(processed_file, index=False)

# Example usage
preprocess_data("data/raw_data/traffic_data.csv", "data/processed_data/processed_traffic_data.csv")