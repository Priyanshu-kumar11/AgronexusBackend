from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- First Model (Crop Prediction) ---
# Read the data
crop_data = pd.read_csv('C:/users/asus/Downloads/Crop_recommendation.csv')

# Separate features and target
X_crop = crop_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Encode the target column (crop types)
label_encoder = LabelEncoder()
y_crop = label_encoder.fit_transform(crop_data['label'])

# Split the dataset into training and testing sets
X_train_crop, X_test_crop, y_train_crop, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model for crop prediction
crop_model = RandomForestRegressor(n_estimators=100, random_state=42)
crop_model.fit(X_train_crop, y_train_crop)

# --- Second Model (Water Requirement Prediction) ---
# Generate synthetic data for water requirement prediction
water_data = pd.DataFrame({
    'temperature': np.random.uniform(20, 40, 100),
    'humidity': np.random.uniform(40, 90, 100),
    'ph': np.random.uniform(5.5, 8.5, 100),
    'rainfall': np.random.uniform(50, 300, 100),
    'crop_type': np.random.choice(['wheat', 'rice', 'maize', 'soybean', 'cotton'], 100),
    'water_requirement': np.random.uniform(100, 800, 100)
})

# One-hot encoding for 'crop_type'
water_data = pd.get_dummies(water_data, columns=['crop_type'])

# Remove the 'rainfall' column
water_data = water_data.drop(columns='rainfall')

# Separate features and target variable
X_water = water_data.drop(columns='water_requirement')
y_water = water_data['water_requirement']

# Split the data into training and test sets
X_train_water, X_test_water, y_train_water, y_test_water = train_test_split(X_water, y_water, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model for water requirement prediction
water_model = RandomForestRegressor(n_estimators=100, random_state=42)
water_model.fit(X_train_water, y_train_water)

# Evaluate the water model
y_pred_water = water_model.predict(X_test_water)
mse_water = mean_squared_error(y_test_water, y_pred_water)
print("Mean Squared Error for Water Requirement Model:", mse_water)

# Save the water model
joblib.dump(water_model, 'water_requirement_prediction_model.joblib')

# --- API Endpoints ---

# Crop prediction API
@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        input_data = request.json  # Expecting a JSON payload
        N = float(input_data['N'])
        P = float(input_data['P'])
        K = float(input_data['K'])
        temperature = float(input_data['temperature'])
        humidity = float(input_data['humidity'])
        ph = float(input_data['ph'])
        rainfall = float(input_data['rainfall'])

        # Prepare the input for prediction
        input_features = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })

        # Make crop prediction
        prediction = crop_model.predict(input_features)
        predicted_label = label_encoder.inverse_transform([int(round(prediction[0]))])[0]

        # Return the prediction as JSON
        return jsonify({'predicted_crop': predicted_label}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Water requirement prediction API
@app.route('/predictwater', methods=['POST'])
def predict_water_requirement():
    try:
        # Get input values from the JSON request
        data = request.get_json()
        
        # Extract input data
        temp = float(data['temperature'])
        hum = float(data['humidity'])
        ph_value = float(data['ph'])
        crop = data['crop'].strip().lower()

        # Prepare input for water prediction with dummy encoding for crop types
        crop_types = ['crop_type_wheat', 'crop_type_rice', 'crop_type_maize', 'crop_type_soybean', 'crop_type_cotton']
        crop_data = [1 if f"crop_type_{crop}" == ct else 0 for ct in crop_types]
        
        input_data = [[temp, hum, ph_value] + crop_data]
        
        # Predict water requirement
        prediction = water_model.predict(input_data)
        
        # Return the prediction as a JSON response
        return jsonify({'water_requirement': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
