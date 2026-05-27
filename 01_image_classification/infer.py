import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys
from PIL import Image
from train import SimpleCNN

def load_model():
    model = SimpleCNN()
    weights_path = 'weights/model.pth'
    if not os.path.exists(weights_path):
        print(f"Error: Trained model weights not found at {weights_path}.")
        print("Please run 'python train.py' first to train the model.")
        sys.exit(1)
    
    # Load model on CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def infer_sample_dataset():
    model, device = load_model()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    print("Loading test dataset for demo...")
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
    
    # Get a batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        
    print("\n--- MNIST Inference Demo ---")
    for i in range(5):
        print(f"Sample {i+1}: True Label = {labels[i].item()} | Predicted Label = {predictions[i].item()} | Confidence = {confidences[i].item()*100:.2f}%")

def infer_image(image_path):
    model, device = load_model()
    
    # Open and convert to grayscale 28x28
    try:
        img = Image.open(image_path).convert('L').resize((28, 28))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Add batch dimension [1, 1, 28, 28]
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    print(f"\nImage: {image_path}")
    print(f"Predicted Digit: {prediction.item()}")
    print(f"Confidence: {confidence.item()*100:.2f}%")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        infer_image(sys.argv[1])
    else:
        infer_sample_dataset()
