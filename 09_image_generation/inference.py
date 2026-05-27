import torch
from train import Generator
import torchvision.utils as vutils
from PIL import Image
from io import BytesIO

class ImageGenerator:
    def __init__(self, model_path='models/dcgan_generator.pth'):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        try:
            self.model = Generator(nz=100)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model from {model_path}. Error: {e}")
            self.model = None

    def generate(self, num_images=16):
        if self.model is None:
            return None, "Error: Model weights not found. Train the model first."
            
        noise = torch.randn(num_images, 100, 1, 1, device=self.device)
        with torch.no_grad():
            fake_images = self.model(noise).cpu()
            
        # Denormalize [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2.0
        
        # Make a grid
        grid = vutils.make_grid(fake_images, padding=2, normalize=False, nrow=4)
        
        # Convert to PIL
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(ndarr)
        
        return im, f"Successfully generated {num_images} images."

if __name__ == '__main__':
    g = ImageGenerator()
    img, _ = g.generate(16)
    if img:
        img.save("test_generation.png")
