import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# API Configuration (using Open-Meteo - free, no API key needed)
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Model Configuration
FEATURES = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 
            'pressure_msl', 'wind_speed_10m', 'cloud_cover']
TARGET = 'temperature_2m_next_day'