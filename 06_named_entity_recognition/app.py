import os

os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

import gradio as gr
from inference import NERModel

ner = NERModel()


def extract_entities(text):
    if not text.strip():
        return "请输入中文文本。"
    return ner.predict(text)


with gr.Blocks(title="06 中文命名实体识别") as demo:
    gr.Markdown("# 06 中文命名实体识别")
    gr.Markdown("输入中文句子，提取人名、机构、地点、日期、金额等实体。当前演示使用中文预训练 NER 模型，本地缓存后即可离线推理。")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                lines=5,
                placeholder="例如：小米集团今天在北京发布了新手机，雷军表示今年研发投入将超过300亿元。",
                label="输入文本",
            )
            submit_btn = gr.Button("抽取实体")
        with gr.Column():
            text_output = gr.Textbox(lines=8, label="识别结果")

    submit_btn.click(fn=extract_entities, inputs=text_input, outputs=text_output)

    if ner.nlp is None:
        gr.Markdown("### Warning: 未找到中文 NER 模型，请先在当前目录运行 `python train.py`。")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
