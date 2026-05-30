import os

import numpy as np
import torch
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights
import torchvision.transforms.v2 as T


class SegmentationModel:
    def __init__(self, model_path="models/fcn_resnet50_pet_binary.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.image_size = 320
        self.class_names = ["background", "pet"]

        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.overlay_colors = np.array(
            [
                [0, 0, 0],
                [255, 80, 80],
            ],
            dtype=np.uint8,
        )

        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.image_size = checkpoint.get("config", {}).get("image_size", self.image_size)
                self.class_names = checkpoint.get("class_names", self.class_names)
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            self.model = models.segmentation.fcn_resnet50(
                weights=None,
                weights_backbone=ResNet50_Weights.DEFAULT,
                num_classes=2,
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model weights from {model_path}. Error: {e}")
            self.model = None

    def predict(self, image_input):
        if self.model is None:
            return None, None, "Error: Model weights not found. Train the model first."

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        resized_image = image.resize((self.image_size, self.image_size))
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)["out"][0]
            probabilities = torch.softmax(logits, dim=0)
            predicted_mask = torch.argmax(probabilities, dim=0).cpu().numpy().astype(np.uint8)
            pet_confidence = probabilities[1].mean().item()

        binary_mask = (predicted_mask * 255).astype(np.uint8)
        mask_image = Image.fromarray(binary_mask, mode="L")
        color_mask = Image.fromarray(self.overlay_colors[predicted_mask])
        overlay = Image.blend(resized_image, color_mask, alpha=0.45)

        pet_ratio = float((predicted_mask == 1).mean())
        status = f"Segmentation successful. Pet foreground ratio: {pet_ratio:.1%}. Mean pet confidence: {pet_confidence:.3f}."
        return overlay, mask_image, status


if __name__ == "__main__":
    seg = SegmentationModel()
