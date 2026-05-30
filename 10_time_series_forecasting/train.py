import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
DATA_PATH = "data/airline-passengers.csv"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/lstm_airline.pth"
METADATA_PATH = "models/metadata.pkl"


def download_dataset():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading Airline Passengers dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH


def load_data(filepath, seq_length=12):
    df = pd.read_csv(filepath)
    months = pd.to_datetime(df["Month"], format="%Y-%m")
    data = df["Passengers"].values.astype(float).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)

    features, targets = [], []
    for i in range(len(data_normalized) - seq_length):
        features.append(data_normalized[i : i + seq_length])
        targets.append(data_normalized[i + seq_length])

    X = torch.tensor(np.array(features), dtype=torch.float32)
    y = torch.tensor(np.array(targets), dtype=torch.float32)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler, months, data.reshape(-1)


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        return self.linear(lstm_out[:, -1, :])


def regression_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.where(y_true == 0, 1.0, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    filepath = download_dataset()
    seq_length = 12
    X_train, y_train, X_test, y_test, scaler, months, raw_series = load_data(filepath, seq_length)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model = LSTMForecaster().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 200
    print(f"Starting LSTM training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        y_pred = model(X_train)
        loss = loss_function(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d} loss: {loss.item():10.8f}")

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).cpu().numpy()
        test_pred = model(X_test).cpu().numpy()

    y_train_real = scaler.inverse_transform(y_train.cpu().numpy()).reshape(-1)
    y_test_real = scaler.inverse_transform(y_test.cpu().numpy()).reshape(-1)
    train_pred_real = scaler.inverse_transform(train_pred).reshape(-1)
    test_pred_real = scaler.inverse_transform(test_pred).reshape(-1)

    train_metrics = regression_metrics(y_train_real, train_pred_real)
    test_metrics = regression_metrics(y_test_real, test_pred_real)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "seq_length": seq_length,
        "dataset_path": filepath,
        "num_points": int(len(raw_series)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "first_month": months.iloc[0].strftime("%Y-%m"),
        "last_observed_month": months.iloc[-1].strftime("%Y-%m"),
        "last_observed_value": float(raw_series[-1]),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    joblib.dump(metadata, METADATA_PATH)

    print(f"Training completed. Model saved to {MODEL_PATH}")
    print(
        "Train metrics - "
        f"MAE: {train_metrics['mae']:.2f}, "
        f"RMSE: {train_metrics['rmse']:.2f}, "
        f"MAPE: {train_metrics['mape']:.2f}%"
    )
    print(
        "Test metrics - "
        f"MAE: {test_metrics['mae']:.2f}, "
        f"RMSE: {test_metrics['rmse']:.2f}, "
        f"MAPE: {test_metrics['mape']:.2f}%"
    )


if __name__ == "__main__":
    train()
