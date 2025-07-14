# Smart-Traffic-Management-System

This project predicts traffic congestion in Rabat, Morocco, using historical data, real-time GPS inputs, and event schedules.

## Features
- Real-time traffic predictions.
- Event impact analysis.
- Mobile app for drivers.
- City management dashboard.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Traffic-Congestion-Prediction-Rabat.git

# System Architecture

- **Data Collection**: Fetch real-time GPS data and event schedules.
- **Preprocessing**: Clean and merge datasets.
- **Modeling**: Train a Random Forest model for traffic prediction.
- **Deployment**: Serve predictions via API and mobile app.

# Deployment Guide

1. Build Docker image: `docker build -t traffic-prediction .`
2. Run container: `docker run -p 5000:5000 traffic-prediction`

# User Guide

## Mobile App
- View real-time traffic predictions.
- Get optimal route suggestions.

## Dashboard
- Visualize traffic trends.
- Analyze event impacts.

Requirements:
1- backend API (Flask/FastAPI):
Flask==2.3.2
joblib==1.3.2
pandas==2.0.3
scikit-learn==1.3.0
requests==2.31.0

2- Dashboard:
dash==2.14.0
plotly==5.17.0
pandas==2.0.3
flask==2.3.2


---

### **10. `LICENSE`**
```text
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
