import torch
import numpy as np
import pandas as pd
from train import LSTMForecaster
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class TimeSeriesPredictor:
    def __init__(self, model_path='models/lstm_airline.pth', scaler_path='models/scaler.pkl', data_path='data/airline-passengers.csv'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seq_length = 12
        
        try:
            self.model = LSTMForecaster()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.scaler = joblib.load(scaler_path)
            
            # Load raw data to plot historical context
            df = pd.read_csv(data_path)
            self.raw_data = df['Passengers'].values.astype(float)
            print("Model and data loaded successfully.")
        except Exception as e:
            print(f"Failed to load model. Please run train.py first. Error: {e}")
            self.model = None

    def predict(self, future_steps=12):
        if self.model is None:
            return None, "Error: Model weights not found. Train the model first."
            
        # Get the last sequence of data
        test_inputs = self.raw_data[-self.seq_length:].tolist()
        
        # Predict iteratively
        self.model.eval()
        for i in range(future_steps):
            seq = np.array(test_inputs[-self.seq_length:]).reshape(-1, 1)
            seq = self.scaler.transform(seq)
            seq = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred = self.model(seq).item()
                # Inverse transform
                pred_real = self.scaler.inverse_transform(np.array([[pred]]))[0][0]
                test_inputs.append(pred_real)
                
        actual_predictions = test_inputs[self.seq_length:]
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.title('Airline Passengers Forecasting')
        plt.xlabel('Months')
        plt.ylabel('Passengers')
        
        historical_len = len(self.raw_data)
        plt.plot(np.arange(historical_len), self.raw_data, label='Historical Data', color='blue')
        
        pred_x = np.arange(historical_len, historical_len + future_steps)
        plt.plot(pred_x, actual_predictions, label='Forecast', color='red', linestyle='dashed')
        
        plt.legend()
        plt.grid(True)
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        
        return img, f"Predicted next {future_steps} months successfully."

if __name__ == '__main__':
    predictor = TimeSeriesPredictor()
    img, txt = predictor.predict(12)
    if img:
        img.save("forecast_test.png")
