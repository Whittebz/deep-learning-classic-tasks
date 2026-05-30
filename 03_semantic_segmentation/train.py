import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models
from torchvision.models import ResNet50_Weights
import torchvision.transforms.v2 as T


@dataclass
class TrainConfig:
    data_root: str = "./data"
    image_size: int = 320
    batch_size: int = 8
    num_workers: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 8
    train_samples: int = 1800
    val_samples: int = 200
    seed: int = 42
    save_path: str = "models/fcn_resnet50_pet_binary.pth"


class PetSegmentationDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int):
        self.dataset = datasets.OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=True,
        )
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToImage(),
            T.ToDtype(torch.long, scale=False),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        mask = np.array(mask, dtype=np.uint8)
        # Oxford-IIIT Pet labels: 1=pet, 2=background, 3=border.
        # For a usable binary project, merge border into pet and keep background separate.
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        binary_mask[(mask == 1) | (mask == 3)] = 1

        image = self.image_transform(image)
        mask = self.mask_transform(Image.fromarray(binary_mask)).squeeze(0)
        return image, mask


def build_loaders(config: TrainConfig):
    print("Preparing Oxford-IIIT Pet segmentation dataset...")
    full_dataset = PetSegmentationDataset(root=config.data_root, split="trainval", image_size=config.image_size)

    requested = config.train_samples + config.val_samples
    usable_samples = min(len(full_dataset), requested)
    train_len = min(config.train_samples, max(usable_samples - 1, 1))
    val_len = usable_samples - train_len
    if val_len == 0:
        val_len = 1
        train_len = max(usable_samples - 1, 1)

    generator = torch.Generator().manual_seed(config.seed)
    subset, _ = random_split(full_dataset, [usable_samples, len(full_dataset) - usable_samples], generator=generator)
    train_dataset, val_dataset = random_split(subset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    return train_loader, val_loader


def build_model(num_classes: int = 2):
    return models.segmentation.fcn_resnet50(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
    )


def compute_iou(predictions: torch.Tensor, targets: torch.Tensor):
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    intersection = torch.logical_and(predictions == 1, targets == 1).sum().item()
    union = torch.logical_or(predictions == 1, targets == 1).sum().item()
    if union == 0:
        return 1.0
    return intersection / union


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    batches = 0
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)["out"]
            loss = criterion(logits, masks)
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item()
            total_iou += compute_iou(preds.cpu(), masks.cpu())
            batches += 1
    return total_loss / max(batches, 1), total_iou / max(batches, 1)


def train(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_loader, val_loader = build_loaders(config)
    model = build_model(num_classes=2).to(device)

    class_weights = torch.tensor([1.0, 2.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_iou = -1.0
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    print(f"Starting training for {config.epochs} epochs...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)["out"]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 20 == 0 or step == len(train_loader):
                avg_loss = running_loss / min(20, step if step < 20 else 20)
                print(f"Epoch {epoch}/{config.epochs} Step {step}/{len(train_loader)} Loss: {avg_loss:.4f}")
                running_loss = 0.0

        scheduler.step()
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{config.epochs} Validation Loss: {val_loss:.4f} | Pet IoU: {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "num_classes": 2,
                "class_names": ["background", "pet"],
                "best_val_iou": best_val_iou,
            }
            torch.save(checkpoint, config.save_path)
            print(f"Saved improved checkpoint to {config.save_path}")

    metadata_path = os.path.splitext(config.save_path)[0] + ".json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_goal": "Segment pet foreground from background environment.",
                "class_mapping": {"0": "background", "1": "pet"},
                "best_val_iou": best_val_iou,
                "config": asdict(config),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Training completed. Best validation IoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task 03 semantic segmentation model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=1800)
    parser.add_argument("--val-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=320)
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    train(config)
