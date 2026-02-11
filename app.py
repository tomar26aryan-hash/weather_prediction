from flask import Flask, render_template, request, jsonify
import sys, os, requests, datetime

# allow importing your src.predictor package (adjust path if needed)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predictor import WeatherPredictor

app = Flask(__name__, template_folder='templates')
predictor = None

def initialize_predictor():
    global predictor
    try:
        predictor = WeatherPredictor()
        return True
    except Exception as e:
        print("Error loading predictor:", e)
        predictor = None
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()
        lat = float(body.get('latitude'))
        lon = float(body.get('longitude'))

        if predictor is None:
            return jsonify({"error": "Model not initialized"}), 500

        # 1) ML predictor result (expected to include predicted_temperature, maybe current_temperature, prediction_time, forecast_for)
        ai_result = predictor.predict(lat, lon) or {}

        # 2) Fetch live 'current' from Open-Meteo
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m",
            "timezone": "auto"
        }
        weather_res = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)

        current = {}
        if weather_res.status_code == 200:
            jsonr = weather_res.json()
            current = jsonr.get("current", {}) or {}

        # 3) Build final response: prefer top-level ai_result keys if present, otherwise use current values.
        # Ensure numeric types (or None)
        def as_num(x):
            try:
                return float(x)
            except Exception:
                return None

        current_temp = ai_result.get("current_temperature") or as_num(current.get("temperature_2m")) or as_num(current.get("temperature"))
        predicted_temp = ai_result.get("predicted_temperature") or ai_result.get("predicted_temp") or None

        feels_like = ai_result.get("feels_like") or as_num(current.get("apparent_temperature")) or current_temp
        humidity = ai_result.get("humidity") or as_num(current.get("relative_humidity_2m")) or as_num(current.get("humidity"))
        precipitation = ai_result.get("precipitation") or as_num(current.get("precipitation"))
        wind_speed = ai_result.get("wind_speed") or as_num(current.get("wind_speed_10m")) or as_num(current.get("wind_speed"))
        wind_direction = ai_result.get("wind_direction") or as_num(current.get("wind_direction_10m"))
        cloud_cover = ai_result.get("cloud_cover") or as_num(current.get("cloud_cover"))
        pressure = ai_result.get("pressure") or as_num(current.get("pressure_msl")) or as_num(current.get("surface_pressure"))
        wind_gusts = ai_result.get("wind_gusts") or as_num(current.get("wind_gusts_10m"))

        # Build response dict expected by frontend
        response = {
            "current_temperature": None if current_temp is None else round(current_temp, 2),
            "predicted_temperature": None if predicted_temp is None else round(as_num(predicted_temp), 2),
            "feels_like": None if feels_like is None else round(as_num(feels_like), 2),
            "humidity": None if humidity is None else round(as_num(humidity), 2),
            "precipitation": None if precipitation is None else round(as_num(precipitation), 3),
            "wind_speed": None if wind_speed is None else round(as_num(wind_speed), 2),
            "wind_direction": None if wind_direction is None else round(as_num(wind_direction), 1),
            "cloud_cover": None if cloud_cover is None else round(as_num(cloud_cover), 1),
            "pressure": None if pressure is None else round(as_num(pressure), 1),
            "wind_gusts": None if wind_gusts is None else round(as_num(wind_gusts), 2),

            # keep original ai_result metadata if present
            "prediction_time": ai_result.get("prediction_time") or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_for": ai_result.get("forecast_for") or (datetime.datetime.now() + datetime.timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S"),

            # include raw current for debugging/UI if needed
            "current_weather": current
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/search_location', methods=['GET'])
def search_location():
    try:
        query = request.args.get('query', '').strip()
        if not query or len(query) < 2:
            return jsonify([])

        # use Open-Meteo geocoding (fast + no key)
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": query, "count": 8, "language": "en", "format": "json"}
        r = requests.get(url, params=params, timeout=6)
        j = r.json()
        results = j.get("results") or []
        out = []
        for loc in results:
            out.append({
                "name": f"{loc.get('name')}, {loc.get('country')}" if loc.get('country') else loc.get('name'),
                "latitude": float(loc.get("latitude")),
                "longitude": float(loc.get("longitude"))
            })
        return jsonify(out)
    except Exception as e:
        return jsonify([])


@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_loaded": predictor is not None})

if __name__ == '__main__':
    print("Loading predictor...")
    if initialize_predictor():
        print("Predictor loaded.")
    else:
        print("Predictor failed to load.")
    app.run(debug=True, host='0.0.0.0', port=5000)
