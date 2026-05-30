import os
from pathlib import Path

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import gradio as gr

from inference import SpeechRecognizer


sr = SpeechRecognizer()
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "results" / "08_speech_recognition"
SAMPLE_FILES = sorted(SAMPLES_DIR.glob("*.wav"))
SAMPLE_CHOICES = [sample.name for sample in SAMPLE_FILES]
SAMPLE_MAP = {sample.name: str(sample) for sample in SAMPLE_FILES}


def recognize(audio):
    if audio is None:
        return "请先录音、上传音频，或从下方示例列表中选择一个文件。"
    return sr.predict(audio)


def load_sample(sample_name):
    if not sample_name:
        return None
    return SAMPLE_MAP.get(sample_name)


with gr.Blocks(title="08 Speech Recognition") as demo:
    gr.Markdown("# 08 语音识别")
    gr.Markdown("上传、录制，或直接选择 `docs/results/08_speech_recognition/` 目录中的中文示例音频进行转写。")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="输入音频")
            sample_selector = gr.Dropdown(
                choices=SAMPLE_CHOICES,
                label="示例音频",
                value=SAMPLE_CHOICES[0] if SAMPLE_CHOICES else None,
                interactive=True,
            )
            load_btn = gr.Button("加载示例音频")
            submit_btn = gr.Button("开始识别")
        with gr.Column():
            text_output = gr.Textbox(label="识别结果", lines=6)

    load_btn.click(fn=load_sample, inputs=sample_selector, outputs=audio_input)
    submit_btn.click(fn=recognize, inputs=audio_input, outputs=text_output)

    if not SAMPLE_CHOICES:
        gr.Markdown("### 未找到示例音频\n请将 `.wav` 文件放入 `docs/results/08_speech_recognition/` 目录。")

    if sr.model is None:
        gr.Markdown(
            "### 模型未加载\n请先运行 `python train.py` 下载中文 ASR 模型，然后重新启动界面。"
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False, allowed_paths=[str(SAMPLES_DIR)])
