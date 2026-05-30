import os

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import gradio as gr

from inference import TimeSeriesPredictor


predictor = TimeSeriesPredictor()


def forecast(months, history_window):
    img, summary, table, raw_df = predictor.predict(int(months), int(history_window))
    if raw_df is None:
        return img, summary, table, "模型不可用"

    peak_row = raw_df.loc[raw_df["Forecast"].idxmax()]
    stats_md = (
        f"峰值：**{peak_row['Month'].strftime('%Y-%m')} / {peak_row['Forecast']:.1f}**  \n"
        f"首月：**{raw_df['Forecast'].iloc[0]:.1f}**  \n"
        f"末月：**{raw_df['Forecast'].iloc[-1]:.1f}**"
    )
    return img, summary, table, stats_md


DESCRIPTION = """
# 10 时间序列预测
基于 LSTM，对航空旅客月度数据进行预测。
"""


with gr.Blocks(title="10 时间序列预测", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            months_input = gr.Slider(minimum=1, maximum=36, step=1, value=12, label="预测月数")
            history_input = gr.Slider(minimum=24, maximum=120, step=12, value=72, label="显示历史")
            submit_btn = gr.Button("开始预测", variant="primary")
            gr.Examples(
                examples=[[6, 48], [12, 72], [24, 96]],
                inputs=[months_input, history_input],
                label="快捷示例",
            )
        with gr.Column(scale=2):
            image_output = gr.Image(type="pil", label="预测图")

    with gr.Row():
        summary_output = gr.Textbox(label="结果摘要", lines=4)
        stats_output = gr.Markdown(label="关键结果")

    forecast_table = gr.Dataframe(
        headers=["月份", "预测值", "较最近值变化"],
        datatype=["str", "number", "number"],
        label="预测明细",
        wrap=True,
    )

    submit_btn.click(
        fn=forecast,
        inputs=[months_input, history_input],
        outputs=[image_output, summary_output, forecast_table, stats_output],
    )

    if predictor.model is None:
        gr.Markdown("### 缺少模型文件，请先在该目录运行 `python train.py`。")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
