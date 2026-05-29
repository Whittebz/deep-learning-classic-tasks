import os
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

LOCAL_CIFAR100_ARCHIVE = '/root/autodl-pub/cifar-100/cifar-100-python.tar.gz'
LOCAL_DATA_ROOT = './data'


def prepare_local_cifar100(data_root=LOCAL_DATA_ROOT, archive_path=LOCAL_CIFAR100_ARCHIVE):
    extracted_dir = os.path.join(data_root, 'cifar-100-python')
    train_file = os.path.join(extracted_dir, 'train')
    test_file = os.path.join(extracted_dir, 'test')

    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"Using existing CIFAR-100 files in {extracted_dir}")
        return data_root

    if not os.path.exists(archive_path):
        raise FileNotFoundError(
            f"CIFAR-100 archive not found: {archive_path}. "
            "Please place cifar-100-python.tar.gz there."
        )

    os.makedirs(data_root, exist_ok=True)
    print(f"Extracting local CIFAR-100 archive: {archive_path}")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=data_root)

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        raise RuntimeError("CIFAR-100 extraction completed, but expected files are missing.")

    return data_root


def load_data(batch_size=64):
    data_root = prepare_local_cifar100()
    print("Preparing CIFAR-100 dataset from local files...")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100(root=data_root, train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=data_root, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader, train_dataset.classes


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, classes = load_data(batch_size=128)

    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 2
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch} Evaluation Accuracy: {accuracy:.2f}%")

    os.makedirs('models', exist_ok=True)
    save_path = 'models/resnet18_cifar100.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Model weights saved to {save_path}")


if __name__ == '__main__':
    train()
