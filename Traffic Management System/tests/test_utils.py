import pytest
from utils.data_preprocessing import clean_data

def test_clean_data():
    df = clean_data(pd.DataFrame({"temp": [15, None]}))
    assert df["temp"].isnull().sum() == 0