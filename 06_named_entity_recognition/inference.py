import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class NERModel:
    def __init__(self, model_path='models/distilbert_ner'):
        self.device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Using huggingface pipeline for NER simplifies token aggregation
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer, device=self.device, aggregation_strategy="simple")
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model from {model_path}. Please run train.py first. Error: {e}")
            self.nlp = None

    def predict(self, text):
        if self.nlp is None:
            return "Error: Model weights not found. Train the model first."
            
        results = self.nlp(text)
        
        # Format results
        output = []
        for entity in results:
            output.append(f"Entity: '{entity['word']}' | Type: {entity['entity_group']} | Score: {entity['score']:.4f}")
            
        if not output:
            return "No named entities found."
        return "\n".join(output)

if __name__ == "__main__":
    ner = NERModel()
    text = "Apple is looking at buying U.K. startup for $1 billion, said Tim Cook in London."
    print(ner.predict(text))
