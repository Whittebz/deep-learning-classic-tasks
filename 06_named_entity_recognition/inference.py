import os

os.environ['OMP_NUM_THREADS'] = os.environ.get('OMP_NUM_THREADS') or '1'

import torch
from transformers import AutoModelForTokenClassification, BertTokenizerFast, pipeline

MODEL_PATH = os.path.join("models", "bert_base_chinese_ner")
LABEL_NAMES = {
    "PERSON": "人名",
    "ORG": "组织",
    "GPE": "地缘政治实体",
    "LOC": "地点",
    "FAC": "设施",
    "NORP": "族群/宗教/政治团体",
    "DATE": "日期",
    "TIME": "时间",
    "EVENT": "事件",
    "WORK_OF_ART": "作品",
    "LAW": "法律",
    "LANGUAGE": "语言",
    "PRODUCT": "产品",
    "ORDINAL": "序数",
    "CARDINAL": "数值",
    "MONEY": "金额",
    "PERCENT": "百分比",
    "QUANTITY": "数量",
}


class NERModel:
    def __init__(self, model_path=MODEL_PATH):
        self.device = 0 if torch.cuda.is_available() else -1
        self.model = None
        self.tokenizer = None
        self.nlp = None

        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.nlp = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                aggregation_strategy="simple",
            )
            print("Chinese NER model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model from {model_path}. Please run train.py first. Error: {e}")

    def _merge_entities(self, entities, text):
        merged = []
        for entity in entities:
            label = entity.get("entity_group", "UNKNOWN")
            start = int(entity.get("start", -1))
            end = int(entity.get("end", -1))
            score = float(entity.get("score", 0.0))
            word = text[start:end] if start >= 0 and end >= 0 else str(entity.get("word", "")).replace(" ", "")
            if not word:
                continue

            if merged:
                prev = merged[-1]
                between = text[prev["end"]:start]
                can_join = (
                    prev["entity_group"] == label
                    and start >= prev["end"]
                    and between.strip() == ""
                )
                if can_join:
                    prev["end"] = end
                    prev["word"] = text[prev["start"]:end]
                    prev["score_sum"] += score
                    prev["score_count"] += 1
                    prev["score"] = prev["score_sum"] / prev["score_count"]
                    continue

            merged.append(
                {
                    "entity_group": label,
                    "start": start,
                    "end": end,
                    "word": word,
                    "score": score,
                    "score_sum": score,
                    "score_count": 1,
                }
            )

        for item in merged:
            item.pop("score_sum", None)
            item.pop("score_count", None)
        return merged

    def _format_entity(self, entity):
        label = entity.get("entity_group", "UNKNOWN")
        display_label = LABEL_NAMES.get(label, label)
        word = str(entity.get("word", "")).replace(" ", "").strip()
        score = float(entity.get("score", 0.0))
        start = entity.get("start", "-")
        end = entity.get("end", "-")
        return f"实体：{word} | 类型：{display_label}({label}) | 置信度：{score:.4f} | 位置：[{start}, {end})"

    def predict(self, text):
        text = text.strip()
        if self.nlp is None:
            return "错误：未找到中文 NER 模型，请先运行 `python train.py`。"
        if not text:
            return "请输入中文文本。"

        results = self.nlp(text)
        merged_results = self._merge_entities(results, text)
        output = [self._format_entity(entity) for entity in merged_results if str(entity.get("word", "")).strip()]

        if not output:
            return "未识别到明确实体。建议输入包含人名、机构名、地点、日期的完整中文句子。"
        return "\n".join(output)


if __name__ == "__main__":
    ner = NERModel()
    text = "马英九今天上午在台北会见了鸿海集团负责人，并讨论了两岸产业合作。"
    print(ner.predict(text))
