import copy
import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / max(half, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DiffusionUNet(nn.Module):
    def __init__(self, image_channels=1, base_channels=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        self.down1a = ResBlock(base_channels, base_channels, time_dim)
        self.down1b = ResBlock(base_channels, base_channels, time_dim)
        self.downsample1 = Downsample(base_channels)

        self.down2a = ResBlock(base_channels, base_channels * 2, time_dim)
        self.down2b = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = Downsample(base_channels * 2)

        self.mid1 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.mid2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)

        self.upsample1 = Upsample(base_channels * 2)
        self.up1a = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        self.up1b = ResBlock(base_channels * 2, base_channels, time_dim)

        self.upsample2 = Upsample(base_channels)
        self.up2a = ResBlock(base_channels * 2, base_channels, time_dim)
        self.up2b = ResBlock(base_channels, base_channels, time_dim)

        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x, timesteps):
        time_emb = self.time_mlp(timestep_embedding(timesteps, self.time_dim))

        x0 = self.input_conv(x)
        x1 = self.down1a(x0, time_emb)
        x1 = self.down1b(x1, time_emb)
        x2 = self.downsample1(x1)

        x3 = self.down2a(x2, time_emb)
        x3 = self.down2b(x3, time_emb)
        x4 = self.downsample2(x3)

        h = self.mid1(x4, time_emb)
        h = self.mid2(h, time_emb)

        h = self.upsample1(h)
        h = torch.cat([h, x3], dim=1)
        h = self.up1a(h, time_emb)
        h = self.up1b(h, time_emb)

        h = self.upsample2(h)
        h = torch.cat([h, x1], dim=1)
        h = self.up2a(h, time_emb)
        h = self.up2b(h, time_emb)

        return self.output_conv(F.silu(self.output_norm(h)))


@dataclass
class DiffusionConfig:
    image_size: int = 28
    channels: int = 1
    timesteps: int = 100
    beta_schedule: str = "cosine"
    batch_size: int = 128
    learning_rate: float = 2e-4
    epochs: int = 20
    ema_decay: float = 0.995


class GaussianDiffusion:
    def __init__(self, config: DiffusionConfig, device):
        self.config = config
        self.device = device

        betas = self._make_beta_schedule(config).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    @staticmethod
    def _make_beta_schedule(config: DiffusionConfig):
        if config.beta_schedule == "cosine":
            steps = config.timesteps + 1
            t = torch.linspace(0, config.timesteps, steps, dtype=torch.float32) / config.timesteps
            alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            return betas.clamp(1e-4, 0.999)
        return torch.linspace(1e-4, 0.02, config.timesteps, dtype=torch.float32)

    def q_sample(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x, t):
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        predicted_noise = model(x, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t.clamp(min=1e-20)) * noise

    def sample(self, model, num_images):
        model.eval()
        x = torch.randn(num_images, self.config.channels, self.config.image_size, self.config.image_size, device=self.device)
        with torch.no_grad():
            for step in reversed(range(self.config.timesteps)):
                t = torch.full((num_images,), step, device=self.device, dtype=torch.long)
                x = self.p_sample(model, x, t)
        return x.clamp(-1.0, 1.0)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters()):
                shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)
            for shadow_buffer, model_buffer in zip(self.shadow.buffers(), model.buffers()):
                shadow_buffer.copy_(model_buffer)


def save_samples(diffusion, model, epoch, output_dir):
    samples = diffusion.sample(model, num_images=16).cpu()
    samples = (samples + 1.0) / 2.0
    grid = vutils.make_grid(samples, nrow=4, padding=2)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"epoch_{epoch:03d}.png")
    vutils.save_image(grid, path)
    print(f"Saved samples to {path}")


def train():
    config = DiffusionConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = DiffusionUNet(image_channels=config.channels).to(device)
    ema = EMA(model, config.ema_decay)
    diffusion = GaussianDiffusion(config, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    print(f"Starting diffusion training for {config.epochs} epochs...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (images, _) in enumerate(dataloader, start=1):
            images = images.to(device)
            timesteps = torch.randint(0, config.timesteps, (images.size(0),), device=device)
            noise = torch.randn_like(images)
            noisy_images = diffusion.q_sample(images, timesteps, noise)
            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item()
            if step % 100 == 0:
                avg_loss = running_loss / 100
                print(f"Epoch [{epoch}/{config.epochs}] Step [{step}/{len(dataloader)}] Loss: {avg_loss:.4f}")
                running_loss = 0.0

        save_samples(diffusion, ema.shadow, epoch, output_dir="samples")

    os.makedirs("models", exist_ok=True)
    save_path = "models/mnist_diffusion.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.shadow.state_dict(),
            "config": asdict(config),
        },
        save_path,
    )
    print(f"Training completed. Model weights saved to {save_path}")


if __name__ == "__main__":
    train()
