import os

os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

import gradio as gr
from inference import TimeSeriesPredictor

predictor = TimeSeriesPredictor()


def forecast(months):
    img, text = predictor.predict(int(months))
    return img, text


with gr.Blocks(title="10 Time Series Forecasting") as demo:
    gr.Markdown("# 10 Time Series Forecasting (LSTM)")
    gr.Markdown("Based on historical Airline Passengers data, predict the passenger volume for the next N months.")

    with gr.Row():
        with gr.Column():
            months_input = gr.Slider(minimum=1, maximum=60, step=1, value=12, label="Months to Forecast")
            submit_btn = gr.Button("Forecast")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Forecast Plot")
            text_output = gr.Textbox(label="Status")

    submit_btn.click(fn=forecast, inputs=months_input, outputs=[image_output, text_output])

    if predictor.model is None:
        gr.Markdown("### ⚠️ Warning: Model weights not found. Please run `python train.py` first.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
