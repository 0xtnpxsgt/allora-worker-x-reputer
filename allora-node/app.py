from flask import Flask, Response
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import asyncio
import functools
import json  # Needed for manually creating JSON responses
from app_config import DATABASE_PATH

# Suppress TensorFlow logging (suppress INFO and WARNING messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# Constants from environment variables
API_PORT = int(os.environ.get('API_PORT', 8000))
LOOK_BACK = int(os.environ.get('LOOK_BACK', 10))  # Default to 10 if not set
PREDICTION_STEPS = int(os.environ.get('PREDICTION_STEPS', 10))  # Default to 10 if not set

# HTTP Response Codes
HTTP_RESPONSE_CODE_200 = 200
HTTP_RESPONSE_CODE_404 = 404
HTTP_RESPONSE_CODE_500 = 500

# Load model and scaler
def load_model_and_scaler(token_name, prediction_horizon):
    model_path = f'models/{token_name.lower()}_model_{prediction_horizon}m.keras'
    scaler_path = f'models/{token_name.lower()}_scaler_{prediction_horizon}m.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Cache predictions to improve performance
@functools.lru_cache(maxsize=128)
def cached_prediction(token_name, prediction_horizon):
    model, scaler = load_model_and_scaler(token_name, prediction_horizon)
    
    if model is None or scaler is None:
        return None
    
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT price FROM prices 
            WHERE token=?
            ORDER BY block_height DESC 
            LIMIT ?
        """, (token_name, LOOK_BACK))
        result = cursor.fetchall()
    
    if not result or len(result) == 0:
        return None
    
    # Reverse the result to chronological order
    prices = np.array([x[0] for x in reversed(result)]).reshape(-1, 1)
    
    # Preprocess data
    scaled_data = scaler.transform(prices)
    
    # Prepare the data in the format expected by the model
    recent_data = scaled_data.reshape(1, LOOK_BACK, 1)
    predictions = []

    # Make predictions for the next N minutes
    for _ in range(PREDICTION_STEPS):
        pred = model.predict(recent_data)
        predictions.append(pred[0][0])
        recent_data = np.append(recent_data[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    # Inverse scaling to get actual prices
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Get the mean forecast for the next N minutes
    mean_forecast = np.mean(predictions)
    return mean_forecast

@app.route('/', methods=['GET'])
async def health():
    return "Hello, World, I'm alive!"

@app.route('/inference/<token>', methods=['GET'])
async def get_inference(token):
    if not token:
        response = json.dumps({"error": "Token is required"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    token_name = f"{token}USD".lower()

    try:
        loop = asyncio.get_event_loop()
        mean_forecast = await loop.run_in_executor(None, cached_prediction, token_name, PREDICTION_STEPS)

        if mean_forecast is None:
            response = json.dumps({"error": "No data found or model unavailable for the specified token"})
            return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')

        print(f"{token} inference: {mean_forecast}")
        return Response(str(mean_forecast), status=HTTP_RESPONSE_CODE_200, mimetype='text/plain')

    except Exception as e:
        response = json.dumps({"error": str(e)})
        return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

@app.route('/truth/<token>/<block_height>', methods=['GET'])
async def get_price(token, block_height):
    # Directly interact with SQLite database
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT block_height, price 
            FROM prices 
            WHERE token=? AND block_height <= ? 
            ORDER BY ABS(block_height - ?) 
            LIMIT 1
        """, (token.lower(), block_height, block_height))
        result = cursor.fetchone()

    if result:
        # Only return the price in the "body" field as per your desired format
        response = json.dumps(result[1])  # This will only return the price, not the entire object
        return Response(response, status=HTTP_RESPONSE_CODE_200, mimetype='application/json')
    else:
        response = json.dumps({'error': 'No price data found for the specified token and block_height'})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=API_PORT)
