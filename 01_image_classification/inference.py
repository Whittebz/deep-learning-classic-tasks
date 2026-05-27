import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ImageClassifier:
    def __init__(self, model_path='models/resnet18_cifar10.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Initialize ResNet18 structure
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
        # Load weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model weights from {model_path}. Please run train.py first. Error: {e}")
            self.model = None

        # CIFAR-10 inference transforms
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def predict(self, image_input):
        if self.model is None:
            return {"Error": "Model weights not found. Train the model first."}
            
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            # Assume PIL Image
            image = image_input.convert('RGB')
            
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        results = {self.classes[top3_indices[i]]: float(top3_prob[i]) for i in range(3)}
        
        return results

if __name__ == '__main__':
    print("Testing ImageClassifier initialization...")
    classifier = ImageClassifier()
