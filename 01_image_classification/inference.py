import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class ImageClassifier:
    def __init__(self, model_path='models/resnet18_cifar100.pth', data_root='./data'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self._load_cifar100_labels(data_root)

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 100)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model weights from {model_path}. Please run train.py first. Error: {e}")
            self.model = None

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    def _load_cifar100_labels(self, data_root):
        meta_path = os.path.join(data_root, 'cifar-100-python', 'meta')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f, encoding='latin1')
            return meta.get('fine_label_names', [f'class_{i}' for i in range(100)])
        return [f'class_{i}' for i in range(100)]

    def predict(self, image_input):
        if self.model is None:
            return {"Error": "Model weights not found. Train the model first."}

        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top3_prob, top3_indices = torch.topk(probabilities, 3)
        results = {self.classes[top3_indices[i]]: float(top3_prob[i]) for i in range(3)}

        return results


if __name__ == '__main__':
    print("Testing ImageClassifier initialization...")
    classifier = ImageClassifier()
