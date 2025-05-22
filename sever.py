import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- First Model (Crop Prediction) ---
crop_data = pd.read_csv('C:/users/asus/Downloads/Crop_recommendation.csv')
X_crop = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
label_encoder = LabelEncoder()
y_crop = label_encoder.fit_transform(crop_data['label'])
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)
crop_model = RandomForestRegressor(n_estimators=100, random_state=42)
crop_model.fit(X_train_crop, y_train_crop)

# --- Second Model (Water Requirement Prediction from Synthetic Data) ---
water_data = pd.DataFrame({
    'temperature': np.random.uniform(20, 40, 100),
    'humidity': np.random.uniform(40, 90, 100),
    'ph': np.random.uniform(5.5, 8.5, 100),
    'rainfall': np.random.uniform(50, 300, 100),
    'crop_type': np.random.choice(['wheat', 'rice', 'maize', 'soybean', 'cotton'], 100),
    'water_requirement': np.random.uniform(100, 800, 100)
})
water_data = pd.get_dummies(water_data, columns=['crop_type']).drop(columns='rainfall')
X_water = water_data.drop(columns='water_requirement')
y_water = water_data['water_requirement']
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(
    X_water, y_water, test_size=0.2, random_state=42
)
water_model = RandomForestRegressor(n_estimators=100, random_state=42)
water_model.fit(X_train_water, y_train_water)

# --- Third Model (Water Requirement based on Irrigation Days) ---
irrigation_data = {
    "Crop": ["Rice", "Groundnut", "Sorghum", "Maize", "Sugarcane", "Ragi", "Cotton", "Pulses"],
    "Duration in days": [135, 105, 100, 110, 365, 100, 165, 65],
    "Water requirement (mm)": [1250, 550, 350, 500, 2000, 350, 550, 350],
    "Number of irrigations": [18, 10, 6, 8, 24, 6, 11, 4]
}
df_irrig = pd.DataFrame(irrigation_data)
X_irrig = df_irrig[["Duration in days", "Number of irrigations"]]
y_irrig = df_irrig["Water requirement (mm)"]
irrig_model = RandomForestRegressor(n_estimators=100, random_state=42)
irrig_model.fit(X_irrig, y_irrig)

# --- API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        features = pd.DataFrame([{
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K']),
            'temperature': float(data['temperature']),
            'humidity': float(data['humidity']),
            'ph': float(data['ph']),
            'rainfall': float(data['rainfall'])
        }])
        pred = crop_model.predict(features)
        crop_label = label_encoder.inverse_transform([int(round(pred[0]))])[0]
        return jsonify({'predicted_crop': crop_label}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predictwater', methods=['POST'])
def predict_water_requirement():
    try:
        data = request.get_json()
        temp = float(data['temperature'])
        hum = float(data['humidity'])
        ph_val = float(data['ph'])
        crop = data['crop'].strip().lower()

        valid = ['wheat', 'rice', 'maize', 'soybean', 'cotton']
        if crop not in valid:
            return jsonify({'error': f"Invalid crop type. Valid: {', '.join(valid)}"}), 400

        types = [f"crop_type_{c}" for c in valid]
        dummy = [1 if t == f"crop_type_{crop}" else 0 for t in types]
        input_vec = [[temp, hum, ph_val] + dummy]
        pred = water_model.predict(input_vec)
        return jsonify({'water_requirement': float(pred[0])}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_water', methods=['POST'])
def predict_irrigation_water():
    try:
        data = request.get_json()
        duration = data.get("duration")
        irrigations = data.get("irrigations")
        if duration is None or irrigations is None:
            return jsonify({"error": "Provide both 'duration' and 'irrigations'."}), 400

        df_in = pd.DataFrame([[duration, irrigations]],
                             columns=["Duration in days", "Number of irrigations"])
        pred = irrig_model.predict(df_in)[0]
        return jsonify({"predicted_water_requirement": f"{pred:.2f} mm"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --- Server Runner for Windows using Waitress ---
if __name__ == '__main__':
    from waitress import serve
    port = int(os.environ.get('PORT', 5000))
    serve(app, host='0.0.0.0', port=port)
