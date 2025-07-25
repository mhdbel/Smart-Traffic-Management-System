# data_preprocessing.py
import pandas as pd

def clean_data(df):
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['weather_main'], drop_first=True)
    return df
