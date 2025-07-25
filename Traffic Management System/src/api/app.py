# app.py
from flask import Flask, request, jsonify
from agents.orchestrator import Orchestrator

app = Flask(__name__)
orchestrator = Orchestrator()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    origin = data.get("origin")
    destination = data.get("destination")

    # Handle user query using the orchestrator
    response = orchestrator.handle_user_query(origin, destination)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
