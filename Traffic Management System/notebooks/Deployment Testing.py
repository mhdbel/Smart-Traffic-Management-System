# Deployment Testing
import joblib
import pandas as pd

# Load model
model = joblib.load("../models/traffic_model.pkl")

# Test prediction
test_data = pd.DataFrame({
    "temp": [25],
    "rain_1h": [0],
    "snow_1h": [0],
    "clouds_all": [50]
})
prediction = model.predict(test_data)
print(f"Predicted Traffic Volume: {prediction[0]}")
