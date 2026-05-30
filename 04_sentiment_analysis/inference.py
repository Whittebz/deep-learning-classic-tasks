from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_CANDIDATES = [
    Path('/root/autodl-tmp/04_sentiment_analysis/models/bert_chinese_sentiment'),
    Path(__file__).resolve().parent / 'models' / 'bert_chinese_sentiment',
]


class SentimentClassifier:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        resolved_model_path = self._resolve_model_path(model_path)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(resolved_model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f'Model loaded successfully from {resolved_model_path}.')
        except Exception as e:
            print(f'Failed to load model from {resolved_model_path}. Please run train.py first. Error: {e}')
            self.model = None

    @staticmethod
    def _resolve_model_path(model_path):
        if model_path:
            return model_path

        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate.exists():
                return str(candidate)

        return str(DEFAULT_MODEL_CANDIDATES[0])

    def predict(self, text):
        if self.model is None:
            return {'错误': '模型权重不存在，请先运行 train.py 训练模型。'}

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]

        return {
            '消极': float(probabilities[0]),
            '积极': float(probabilities[1]),
        }


if __name__ == '__main__':
    clf = SentimentClassifier()
    print(clf.predict('这家店的服务很好，体验非常满意。'))
