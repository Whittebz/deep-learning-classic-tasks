import torch
import torchaudio
import os
from train import M5

class SpeechRecognizer:
    def __init__(self, model_path='models/m5_speech_commands.pth', labels_path='models/speech_labels.txt'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Load labels
        try:
            with open(labels_path, 'r') as f:
                self.labels = f.read().splitlines()
        except Exception:
            self.labels = []
            
        # Load model
        try:
            self.model = M5(n_input=1, n_output=max(35, len(self.labels)))
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model. Please run train.py first. Error: {e}")
            self.model = None

    def predict(self, audio_path):
        if self.model is None or not self.labels:
            return {"Error": 1.0}
            
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16000Hz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            
        # Force mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(waveform)
            probabilities = torch.exp(output[0])
            
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        results = {self.labels[top3_indices[i]]: float(top3_prob[i]) for i in range(3)}
        
        return results

if __name__ == "__main__":
    sr = SpeechRecognizer()
    # Provide a dummy test path if running directly
    print("Initialize SpeechRecognizer...")
