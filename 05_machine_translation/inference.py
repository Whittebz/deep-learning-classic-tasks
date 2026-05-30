import json
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    def __init__(self, model_path="models/translation_en_fr"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_path = model_path
        self.input_prefix = ""
        self.max_input_length = 128
        self.max_target_length = 128

        config_path = os.path.join(model_path, "task_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                self.input_prefix = config.get("input_prefix", "")
                self.max_input_length = config.get("max_input_length", 128)
                self.max_target_length = config.get("max_target_length", 128)
            except Exception as e:
                print(f"Failed to read task config from {config_path}: {e}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model from {model_path}. Please run train.py first. Error: {e}")
            self.model = None

    def translate(self, text):
        if self.model is None:
            return "Error: Model weights not found. Train the model first."

        source_text = f"{self.input_prefix}{text}"
        inputs = self.tokenizer(
            source_text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_target_length,
                num_beams=4,
                early_stopping=True,
            )

        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text


if __name__ == "__main__":
    t = Translator()
    print(t.translate("Deep learning is a subset of machine learning."))
