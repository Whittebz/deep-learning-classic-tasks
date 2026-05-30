import os

os.environ['NO_PROXY'] = '127.0.0.1,localhost'
os.environ['no_proxy'] = '127.0.0.1,localhost'

import gradio as gr

from inference import SentimentClassifier

MODEL_PATH = os.environ.get(
    'SENTIMENT_MODEL_PATH',
    '/root/autodl-tmp/04_sentiment_analysis/models/bert_chinese_sentiment',
)

clf = SentimentClassifier(model_path=MODEL_PATH)


def predict(text):
    if not text.strip():
        return {'请输入一些中文评论文本': 1.0}
    return clf.predict(text)


with gr.Blocks(title='04 文本分类 - 中文情感分析') as demo:
    gr.Markdown('# 04 文本分类（中文情感分析）')
    gr.Markdown('输入一段中文评论文本，模型将判断其情感倾向为积极或消极。训练会在首次运行时自动下载并解压完整 ChnSentiCorp 数据集。')

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=5, placeholder='这家餐厅味道不错，下次还会再来。', label='输入文本')
            submit_btn = gr.Button('开始分类')
        with gr.Column():
            label_output = gr.Label(num_top_classes=2, label='情感概率')

    submit_btn.click(fn=predict, inputs=text_input, outputs=label_output)

    if clf.model is None:
        gr.Markdown(
            '### 警告：未找到模型权重，请先运行 `python train.py` 完成训练。'
            f' 当前查找路径：`{MODEL_PATH}`'
        )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=6006, share=False)
