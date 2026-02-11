# 🌤️ Weather Prediction ML Project

A machine learning project that predicts weather conditions using historical data and multiple ML algorithms. The project includes data collection, preprocessing, model training, and a Flask web application for real-time predictions.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)
- [Web Application](#web-application)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project leverages machine learning to predict weather conditions based on historical meteorological data. It compares multiple regression algorithms to find the best performing model and provides an easy-to-use web interface for making predictions.

**Location**: Delhi, India (Lat: 28.6139, Long: 77.2090)

## ✨ Features

- **Automated Data Collection**: Fetches historical weather data programmatically
- **Comprehensive Preprocessing**: Feature engineering, scaling, and train-test split
- **Multiple ML Algorithms**: Trains and compares various regression models
- **Model Persistence**: Saves the best performing model for deployment
- **Web Interface**: Flask-based web app for real-time predictions
- **Visualization**: Performance metrics and comparison plots
- **Modular Design**: Clean separation of concerns with dedicated modules

## 📁 Project Structure

```
weather-prediction/
│
├── src/
│   ├── data_collector.py       # Weather data collection module
│   ├── data_preprocessor.py    # Data preprocessing and feature engineering
│   └── model_trainer.py        # Model training and evaluation
│
├── models/                      # Saved trained models
│
├── web_app/
│   └── app.py                  # Flask web application
│
├── train_model.py              # Main training pipeline
├── requirements.txt            # Project dependencies
│
├── weather_data.csv            # Raw weather data
├── X_train.csv                 # Training features
├── X_test.csv                  # Testing features
├── y_train.csv                 # Training labels
├── y_test.csv                  # Testing labels
├── feature_cols.pkl            # Feature column names
└── scaler.pkl                  # Fitted StandardScaler object
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/weather-prediction.git
   cd weather-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Training the Models

Run the complete training pipeline:

```bash
python train_model.py
```

This script will:
1. Collect 2 years of historical weather data
2. Preprocess and engineer features
3. Train multiple ML models
4. Select and save the best performing model
5. Generate performance visualizations

### Running the Web Application

After training, launch the Flask web app:

```bash
python web_app/app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

### Making Predictions

You can make predictions through:

1. **Web Interface**: Enter weather parameters in the form
2. **Python Script**: Import and use the trained model directly

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare your input data
# Make prediction
prediction = model.predict(scaled_data)
```

## 🤖 Models

The project trains and compares multiple regression algorithms:

- **Linear Regression**: Baseline linear model
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Boosted ensemble method
- **XGBoost**: Optimized gradient boosting

Each model is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## 📊 Dataset

The dataset includes historical weather data with the following features:

- Temperature (°C)
- Humidity (%)
- Pressure (hPa)
- Wind Speed (m/s)
- Cloud Coverage (%)
- Precipitation (mm)
- Time-based features (hour, day, month, season)
- Cyclical encodings for temporal features

**Data Source**: Historical weather API for Delhi, India  
**Time Period**: 2 years of hourly data  
**Total Samples**: ~17,500 records

## 🌐 Web Application

The Flask web application provides:

- **User-friendly Interface**: Simple form for inputting weather parameters
- **Real-time Predictions**: Instant weather forecasting
- **Model Information**: Display of model performance metrics
- **Responsive Design**: Works on desktop and mobile devices

### API Endpoint

```http
POST /predict
Content-Type: application/json

{
  "temperature": 25.5,
  "humidity": 65,
  "pressure": 1013,
  "wind_speed": 3.5,
  ...
}
```

## 📈 Results

Model performance comparison (example):

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | 2.34 | 3.12 | 0.85 |
| Ridge | 2.31 | 3.09 | 0.86 |
| Lasso | 2.35 | 3.14 | 0.85 |
| Random Forest | 1.89 | 2.56 | 0.91 |
| Gradient Boosting | 1.76 | 2.41 | 0.93 |
| **XGBoost** | **1.68** | **2.32** | **0.94** |

*Note: Update these values with your actual results*

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **pandas & numpy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **Flask**: Web application framework
- **matplotlib & seaborn**: Data visualization
- **joblib**: Model serialization

## 🔮 Future Enhancements

- [ ] Add deep learning models (LSTM, GRU)
- [ ] Implement multi-step ahead forecasting
- [ ] Include more weather parameters (UV index, visibility)
- [ ] Add location-based predictions for multiple cities
- [ ] Create a RESTful API with authentication
- [ ] Deploy to cloud platforms (AWS, Heroku, etc.)
- [ ] Add real-time weather data integration
- [ ] Implement model monitoring and retraining pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

