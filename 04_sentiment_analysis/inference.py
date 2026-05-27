import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentClassifier:
    def __init__(self, model_path='models/distilbert_imdb'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model from {model_path}. Please run train.py first. Error: {e}")
            self.model = None

    def predict(self, text):
        if self.model is None:
            return {"Error": "Model weights not found. Train the model first."}
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
            
        # IMDB: 0 is Negative, 1 is Positive
        results = {
            "Negative": float(probabilities[0]),
            "Positive": float(probabilities[1])
        }
        
        return results

if __name__ == "__main__":
    clf = SentimentClassifier()
    print(clf.predict("I really loved this movie, it was fantastic!"))
