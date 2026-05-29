import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_dataset(batch_size=8):
    print("Downloading and preparing Oxford-IIIT Pet dataset for Segmentation...")
    # Transformations for semantic segmentation are tricky because image and mask
    # must be transformed in sync. We will use v2 transforms which handle this.
    import torchvision.transforms.v2 as T
    
    # We will resize to a smaller resolution for speed
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
        T.ToImage(),
        T.ToDtype(torch.long, scale=False)
    ])
    
    # Wrap in a custom dataset to apply sync transforms if needed, or simply apply separately
    # The torchvision OxfordIIITPet returns (image, mask) if target_types='segmentation'
    class PetSegmentationDataset(torch.utils.data.Dataset):
        def __init__(self, root, split):
            self.dataset = datasets.OxfordIIITPet(root=root, split=split, target_types='segmentation', download=True)
            self.transform = transform
            self.target_transform = target_transform
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            img, mask = self.dataset[idx]
            # Convert mask to 0-indexed (classes are 1: foreground, 2: background, 3: not classified)
            # We map: 1 -> 1, 2 -> 0, 3 -> 2
            import numpy as np
            mask = np.array(mask)
            mask[mask == 2] = 0
            mask[mask == 3] = 2
            from PIL import Image
            mask = Image.fromarray(mask)
            
            img = self.transform(img)
            mask = self.target_transform(mask).squeeze(0) # [H, W]
            return img, mask

    # We use a very small subset just to make sure it runs fast
    full_dataset = PetSegmentationDataset(root='./data', split='trainval')
    indices = torch.randperm(len(full_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(full_dataset, indices[:200]) # Mini set
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    train_loader = get_dataset(batch_size=4)

    # Use FCN with ResNet50 backbone and custom number of classes for Oxford-IIIT Pet
    # Oxford pet has 3 classes: Background(0), Foreground/Pet(1), Boundary(2)
    model = models.segmentation.fcn_resnet50(weights=None, num_classes=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 2
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch {epoch} [{i+1}/{len(train_loader)}] Loss: {running_loss/10:.4f}")
                running_loss = 0.0

    os.makedirs('models', exist_ok=True)
    save_path = 'models/fcn_resnet50_pet.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Model weights saved to {save_path}")

if __name__ == '__main__':
    train()
