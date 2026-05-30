import os

os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS') or '1'

import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

MODEL_PATH = os.path.join('models', 't5_zh_summarization')
SUMMARY_PROMPT_PREFIX = '生成式摘要任务：【'
SUMMARY_PROMPT_SUFFIX = '】这篇文章的摘要是什么？'


class Summarizer:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, use_safetensors=True)
            self.model.to(self.device)
            self.model.eval()
            print('Model loaded successfully.')
        except Exception as e:
            print(f'Failed to load model from {model_path}. Please run train.py first. Error: {e}')

    def summarize(self, text):
        text = text.strip()
        if self.model is None or self.tokenizer is None:
            return 'Error: Model weights not found. Train the model first.'
        if not text:
            return '请输入需要摘要的中文文本。'

        prompt = SUMMARY_PROMPT_PREFIX + text + SUMMARY_PROMPT_SUFFIX
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=384,
            truncation=True,
        )
        inputs.pop('token_type_ids', None)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=96,
                min_length=16,
                num_beams=4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


if __name__ == '__main__':
    s = Summarizer()
    article = '人工智能正在加速进入教育、医疗和制造业。越来越多企业开始使用大模型提升效率，但同时也面临数据安全、成本控制和落地场景有限的问题。专家认为，未来竞争的关键不只是模型能力，还包括行业知识沉淀和工程化部署水平。'
    print(s.summarize(article))
