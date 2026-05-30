import math
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image

from train import DiffusionConfig, DiffusionUNet, GaussianDiffusion


class ImageGenerator:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.diffusion = None

        task_dir = Path(__file__).resolve().parent
        if model_path is None:
            model_path = task_dir / "models" / "mnist_diffusion.pth"
        else:
            model_path = Path(model_path)

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            config = DiffusionConfig(**checkpoint["config"])
            self.model = DiffusionUNet(image_channels=config.channels).to(self.device)
            state_dict = checkpoint.get("ema_state_dict") or checkpoint["model_state_dict"]
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.diffusion = GaussianDiffusion(config, self.device)
            print(f"Model loaded successfully from {model_path}.")
        except Exception as error:
            print(f"Failed to load model from {model_path}. Error: {error}")

    def generate(self, num_images=16):
        if self.model is None or self.diffusion is None:
            return None, "Error: Model weights not found. Train the model first."

        num_images = int(num_images)
        with torch.no_grad():
            images = self.diffusion.sample(self.model, num_images).cpu()

        images = (images + 1.0) / 2.0
        nrow = max(1, int(math.sqrt(num_images)))
        grid = vutils.make_grid(images, padding=2, normalize=False, nrow=nrow)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        if ndarr.ndim == 3 and ndarr.shape[-1] == 1:
            image = Image.fromarray(ndarr[:, :, 0], mode="L")
        else:
            image = Image.fromarray(ndarr)

        steps = self.diffusion.config.timesteps
        return image, f"Successfully generated {num_images} images with {steps} denoising steps."


if __name__ == "__main__":
    generator = ImageGenerator()
    image, _ = generator.generate(16)
    if image:
        image.save("test_generation.png")
