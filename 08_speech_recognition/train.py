import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2).squeeze(1)


def _cleanup_corrupted_archive(root_dir):
    for name in os.listdir(root_dir):
        if name.startswith("speech_commands_v0.02.tar.gz"):
            path = os.path.join(root_dir, name)
            if os.path.isfile(path):
                print(f"Removing corrupted download artifact: {path}")
                os.remove(path)


def get_dataset():
    print("Downloading SpeechCommands dataset...")
    data_root = "./data"
    os.makedirs(data_root, exist_ok=True)

    try:
        train_set = torchaudio.datasets.SPEECHCOMMANDS(data_root, download=True)
    except RuntimeError as e:
        if "invalid hash value" in str(e):
            _cleanup_corrupted_archive(data_root)
            print("Retrying SpeechCommands download after cleanup...")
            train_set = torchaudio.datasets.SPEECHCOMMANDS(data_root, download=True)
        else:
            raise

    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

    indices = torch.randperm(len(train_set)).tolist()
    train_subset = torch.utils.data.Subset(train_set, indices[:1000])
    test_subset = torch.utils.data.Subset(train_set, indices[-200:])

    return train_subset, test_subset, labels


def label_to_index(word, labels):
    return torch.tensor(labels.index(word))


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.0)
    return batch.permute(0, 2, 1)


def collate_fn(batch, labels):
    tensors, targets = [], []
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label, labels)]

    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)
    return tensors, targets


def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    train_set, test_set, labels = get_dataset()

    os.makedirs("models", exist_ok=True)
    with open("models/speech_labels.txt", "w") as f:
        f.write("\n".join(labels))

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, labels),
        num_workers=2,
        pin_memory=True,
    )

    model = M5(n_input=1, n_output=len(labels))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    epochs = 2
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output.squeeze(), target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    save_path = "models/m5_speech_commands.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Model saved to {save_path}")


if __name__ == "__main__":
    train()
