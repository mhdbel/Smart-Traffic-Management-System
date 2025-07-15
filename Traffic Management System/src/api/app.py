# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("../models/traffic_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)
    return jsonify({'predicted_traffic_volume': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)