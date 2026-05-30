from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from train import LSTMForecaster


class TimeSeriesPredictor:
    def __init__(
        self,
        model_path="models/lstm_airline.pth",
        scaler_path="models/scaler.pkl",
        metadata_path="models/metadata.pkl",
        data_path="data/airline-passengers.csv",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_length = 12
        self.metadata = {}

        try:
            self.model = LSTMForecaster()
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.scaler = joblib.load(scaler_path)
            self.metadata = joblib.load(metadata_path)
            self.seq_length = int(self.metadata.get("seq_length", self.seq_length))

            df = pd.read_csv(data_path)
            self.months = pd.to_datetime(df["Month"], format="%Y-%m")
            self.raw_data = df["Passengers"].values.astype(float)
            self.fitted_df = self._build_fitted_frame()
            print("Model and data loaded successfully.")
        except Exception as e:
            print(f"Failed to load model. Please run train.py first. Error: {e}")
            self.model = None

    def _predict_next_value(self, recent_values):
        seq = np.array(recent_values).reshape(-1, 1)
        seq = self.scaler.transform(seq)
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(seq_tensor).item()
        return float(self.scaler.inverse_transform(np.array([[pred]]))[0][0])

    def _build_fitted_frame(self):
        fitted_values = [np.nan] * self.seq_length
        self.model.eval()
        for idx in range(self.seq_length, len(self.raw_data)):
            history = self.raw_data[idx - self.seq_length : idx]
            fitted_values.append(self._predict_next_value(history))

        return pd.DataFrame(
            {
                "Month": self.months,
                "Actual": self.raw_data,
                "Fitted": fitted_values,
            }
        )

    def _build_forecast_frame(self, future_steps):
        future_values = []
        context = self.raw_data[-self.seq_length :].tolist()

        for _ in range(future_steps):
            next_value = self._predict_next_value(context[-self.seq_length :])
            future_values.append(next_value)
            context.append(next_value)

        future_months = pd.date_range(
            self.months.iloc[-1] + pd.offsets.MonthBegin(1),
            periods=future_steps,
            freq="MS",
        )
        forecast_df = pd.DataFrame(
            {
                "Month": future_months,
                "Forecast": np.round(future_values, 1),
            }
        )
        forecast_df["Delta vs Last"] = np.round(forecast_df["Forecast"] - self.raw_data[-1], 1)
        return forecast_df

    def _plot(self, forecast_df, history_window):
        history_start = max(0, len(self.raw_data) - history_window)
        history_df = self.fitted_df.iloc[history_start:].copy()

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(history_df["Month"], history_df["Actual"], color="#1f77b4", linewidth=2.3, label="History")
        ax.plot(history_df["Month"], history_df["Fitted"], color="#2ca02c", linewidth=1.8, alpha=0.85, label="Model Fit")
        ax.plot(
            forecast_df["Month"],
            forecast_df["Forecast"],
            color="#d62728",
            linewidth=2.2,
            linestyle="--",
            marker="o",
            label="Forecast",
        )
        ax.axvline(self.months.iloc[-1], color="#666666", linestyle=":", linewidth=1.5)
        ax.text(self.months.iloc[-1], ax.get_ylim()[1] * 0.98, " Forecast Start", color="#666666", va="top")
        ax.set_title("Airline Passenger Forecast", fontsize=14, pad=12)
        ax.set_xlabel("Month")
        ax.set_ylabel("Passengers")
        ax.legend()
        fig.autofmt_xdate()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img

    def predict(self, future_steps=12, history_window=60):
        if self.model is None:
            return None, "模型文件缺失，请先运行 train.py。", None, None

        forecast_df = self._build_forecast_frame(int(future_steps))
        img = self._plot(forecast_df, int(history_window))

        last_actual = float(self.raw_data[-1])
        avg_forecast = float(forecast_df["Forecast"].mean())
        peak_row = forecast_df.loc[forecast_df["Forecast"].idxmax()]
        trend = "上升" if forecast_df["Forecast"].iloc[-1] >= forecast_df["Forecast"].iloc[0] else "下降"

        metrics = self.metadata.get("test_metrics", {})
        summary = (
            f"最近月份：{self.metadata.get('last_observed_month', 'N/A')}，旅客数：{last_actual:.0f}\n"
            f"预测范围：未来 {future_steps} 个月，平均值：{avg_forecast:.1f}，趋势：{trend}\n"
            f"峰值月份：{peak_row['Month'].strftime('%Y-%m')}，预测值：{peak_row['Forecast']:.1f}\n"
            f"测试集 MAE：{metrics.get('mae', float('nan')):.2f}，RMSE：{metrics.get('rmse', float('nan')):.2f}"
        )

        display_df = forecast_df.copy()
        display_df["Month"] = display_df["Month"].dt.strftime("%Y-%m")
        display_df = display_df.rename(
            columns={
                "Month": "月份",
                "Forecast": "预测值",
                "Delta vs Last": "较最近值变化",
            }
        )
        return img, summary, display_df, forecast_df


if __name__ == "__main__":
    predictor = TimeSeriesPredictor()
    img, txt, table, _ = predictor.predict(12)
    if img:
        img.save("forecast_test.png")
        print(txt)
        print(table.head())
