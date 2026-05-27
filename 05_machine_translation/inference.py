import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator:
    def __init__(self, model_path='models/translation_en_fr'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
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
            
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
            
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

if __name__ == "__main__":
    t = Translator()
    print(t.translate("Deep learning is a subset of machine learning."))
