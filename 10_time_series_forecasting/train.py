import os
import urllib.request
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def download_dataset():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    os.makedirs('data', exist_ok=True)
    filepath = "data/airline-passengers.csv"
    if not os.path.exists(filepath):
        print("Downloading Airline Passengers dataset...")
        urllib.request.urlretrieve(url, filepath)
    return filepath

def load_data(filepath, seq_length=4):
    df = pd.read_csv(filepath)
    # The dataset has columns 'Month' and 'Passengers'
    data = df['Passengers'].values.astype(float).reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_normalized) - seq_length):
        X.append(data_normalized[i:i+seq_length])
        y.append(data_normalized[i+seq_length])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    
    # 80-20 Train-Test split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    filepath = download_dataset()
    seq_length = 12 # 12 months (1 year) lookback
    X_train, y_train, X_test, y_test, scaler = load_data(filepath, seq_length)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    model = LSTMForecaster().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 150
    print(f"Starting LSTM training for {epochs} epochs...")
    for i in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        y_pred = model(X_train)
        single_loss = loss_function(y_pred, y_train)
        single_loss.backward()
        optimizer.step()
        
        if i % 30 == 0:
            print(f'Epoch {i:3} loss: {single_loss.item():10.8f}')

    os.makedirs('models', exist_ok=True)
    save_path = "models/lstm_airline.pth"
    torch.save(model.state_dict(), save_path)
    
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"Training completed. Model saved to {save_path}")

if __name__ == '__main__':
    train()
