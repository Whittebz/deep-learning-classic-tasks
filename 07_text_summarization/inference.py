import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    def __init__(self, model_path='models/t5_summarization'):
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

    def summarize(self, text):
        if self.model is None:
            return "Error: Model weights not found. Train the model first."
            
        # T5 requires the "summarize: " prefix
        input_text = "summarize: " + text
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_length=150, 
                min_length=30, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary

if __name__ == "__main__":
    s = Summarizer()
    article = """New York (CNN) -- When Liana Barrientos was 23 years old, she got married in Westchester County, New York. 
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband. 
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes 
    only within two weeks of each other."""
    print(s.summarize(article))
