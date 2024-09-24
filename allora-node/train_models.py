import os
import numpy as np
import sqlite3
from sklearn.preprocessing import MinMaxScaler
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from app_config import DATABASE_PATH

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Fetch data from the database
def load_data(token_name):
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT price, block_height FROM prices 
            WHERE token=?
            ORDER BY block_height ASC
        """, (token_name,))
        result = cursor.fetchall()
    return np.array(result)

# Prepare data for TFT
def prepare_data_for_tft(data, look_back, prediction_horizon):
    df = pd.DataFrame(data, columns=['price', 'time_idx'])
    df['group_id'] = "crypto"
    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="price",
        group_ids=["group_id"],
        min_encoder_length=look_back,
        max_encoder_length=look_back,
        min_prediction_length=prediction_horizon,
        max_prediction_length=prediction_horizon,
        time_varying_unknown_reals=["price"],
        target_normalizer=MinMaxScaler()
    )
    return dataset

# Create TFT model
def create_tft_model(dataset, hidden_size=64, attention_head_size=4, dropout=0.3, learning_rate=1e-3):
    model = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4
    )
    return model

# Train and save the TFT model
def train_and_save_model(token_name, look_back, prediction_horizon):
    try:
        print(f"Training model for {token_name} with a {prediction_horizon}-minute horizon.")

        data = load_data(token_name)
        dataset = prepare_data_for_tft(data, look_back, prediction_horizon)

        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        model = create_tft_model(dataset)

        early_stop_callback = EarlyStopping(monitor="val_loss", patience=3)
        trainer = Trainer(
            max_epochs=20,
            gpus=1,  # Use GPU if available
            callbacks=[early_stop_callback]
        )

        trainer.fit(model, train_dataloaders=dataloader)

        # Save the model
        model_path = f'models/{token_name.lower()}_tft_model_{prediction_horizon}m.pth'
        torch.save(model.state_dict(), model_path)

        print(f"Model for {token_name} ({prediction_horizon}-minute prediction) saved to {model_path}")
    
    except Exception as e:
        print(f"Error occurred while training model for {token_name}: {e}")

# Training for different time horizons
time_horizons = {
    '10m': (10, 10),       
    '20m': (10, 20),      
    '24h': (1440, 1440)    
}

for token in ['ETH', 'ARB', 'BTC', 'SOL', 'BNB']:
    for horizon_name, (look_back, prediction_horizon) in time_horizons.items():
        train_and_save_model(f"{token}USD".lower(), look_back, prediction_horizon)
