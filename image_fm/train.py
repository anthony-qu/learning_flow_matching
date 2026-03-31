from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .data import get_dataloader
from .model import UNet


def train(
    nb_epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    base_channels: int = 32,
    time_dim: int = 128,
    checkpoint_dir: str = "checkpoints",
    device: str | None = None,
    max_samples: int | None = None,
    # --- custom data ---
    dataloader: torch.utils.data.DataLoader | None = None,
    in_channels: int = 1,
) -> tuple[UNet, list[float]]:
    """
    Train a Flow Matching U-Net.

    By default trains on MNIST (grayscale, 28x28).
    Pass a custom `dataloader` + `in_channels` to train on your own images instead.

    Returns the trained model and a flat list of per-iteration losses.
    Checkpoints are saved to checkpoint_dir/ckpt_epochXXX.pt after every epoch.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Training on {device}")

    if dataloader is None:
        dataloader = get_dataloader(batch_size=batch_size, max_samples=max_samples)
        in_channels = 1  # MNIST is always grayscale
    model = UNet(in_channels=in_channels, base_channels=base_channels, time_dim=time_dim).to(device).train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_losses: list[float] = []

    for epoch in range(1, nb_epochs + 1):
        epoch_losses: list[float] = []

        with tqdm(dataloader, desc=f"Epoch {epoch}/{nb_epochs}") as pbar:
            for batch in pbar:
                # MNIST returns (image, label); PhotoDataset returns just image
                x1 = batch[0] if isinstance(batch, (list, tuple)) else batch
                x1 = x1.to(device)                    # [B, 1, 28, 28] in [-1, 1]
                x0 = torch.randn_like(x1)              # Gaussian noise, same shape
                t  = torch.rand(x1.shape[0], 1, device=device)  # [B, 1], uniform in [0,1]

                # Linear interpolation: x_t = (1-t)*x0 + t*x1
                t_img = t[:, :, None, None]            # [B, 1, 1, 1] for spatial broadcast
                x_t = (1 - t_img) * x0 + t_img * x1

                # Straight-line conditional velocity: v = x1 - x0
                v_target = x1 - x0

                v_pred = model(x_t, t)                 # [B, 1, 28, 28]
                loss = F.mse_loss(v_pred, v_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = sum(epoch_losses) / len(epoch_losses)
        all_losses.extend(epoch_losses)
        print(f"Epoch {epoch:>3d} | avg loss: {avg:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "losses": all_losses,
                "config": {
                    "in_channels": in_channels,
                    "base_channels": base_channels,
                    "time_dim": time_dim,
                },
            },
            ckpt_dir / f"ckpt_epoch{epoch:03d}.pt",
        )

    return model, all_losses


def load_checkpoint(path: str, device: str | None = None) -> tuple[UNet, list[float], int]:
    """Load a saved checkpoint. Returns (model, losses, epoch)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {"in_channels": 1, "base_channels": 32, "time_dim": 128})
    model = UNet(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["losses"], ckpt["epoch"]
