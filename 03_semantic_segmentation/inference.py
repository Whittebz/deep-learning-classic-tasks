import torch
from torchvision import models
import torchvision.transforms.v2 as T
from PIL import Image
import numpy as np

class SegmentationModel:
    def __init__(self, model_path='models/fcn_resnet50_pet.pth'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model = models.segmentation.fcn_resnet50(weights=None, num_classes=3)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model weights. Error: {e}")
            self.model = None

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Color palette: Background (black), Pet (red), Boundary (green)
        self.colors = np.array([
            [0, 0, 0],       # 0: Background
            [255, 0, 0],     # 1: Pet
            [0, 255, 0]      # 2: Boundary
        ], dtype=np.uint8)

    def predict(self, image_input):
        if self.model is None:
            return None, "Error: Model weights not found. Train the model first."
            
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
            
        # Resize original image to match mask size for overlay
        orig_resized = image.resize((256, 256))
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)['out'][0]
            
        predicted_mask = torch.argmax(output, dim=0).cpu().numpy()
        
        # Convert mask to colored image
        color_mask = self.colors[predicted_mask]
        mask_img = Image.fromarray(color_mask)
        
        # Blend original and mask
        blended = Image.blend(orig_resized, mask_img, alpha=0.5)
        
        return blended, "Segmentation successful."

if __name__ == '__main__':
    seg = SegmentationModel()
