import os

os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import gradio as gr
from inference import Summarizer

summarizer = Summarizer()


def generate_summary(text):
    if not text.strip():
        return '请输入需要摘要的中文文章。'
    return summarizer.summarize(text)


with gr.Blocks(title='07 中文生成式摘要') as demo:
    gr.Markdown('# 07 中文生成式摘要')
    gr.Markdown('输入较长中文文本，系统会生成新的中文摘要。训练阶段会在线下载 LCSTS 小子集，并微调一个更适合中文摘要的 Randeng-T5 多任务模型。')

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=10, placeholder='请粘贴中文文章，例如新闻、公告、项目说明等...', label='原文')
            submit_btn = gr.Button('生成摘要')
        with gr.Column():
            text_output = gr.Textbox(lines=5, label='摘要结果')

    submit_btn.click(fn=generate_summary, inputs=text_input, outputs=text_output)

    if summarizer.model is None:
        gr.Markdown('### Warning: 未找到生成式摘要模型，请先在当前目录运行 `python train.py`。')


if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=6006, share=False)
