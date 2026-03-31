import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_loss(losses: list[float], smooth_window: int = 100) -> None:
    """Plot raw and smoothed training loss."""
    fig, ax = plt.subplots(figsize=(11, 3), dpi=100)
    ax.plot(losses, alpha=0.25, color="tab:blue", label="Loss")
    if len(losses) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses, kernel, mode="valid")
        ax.plot(
            np.arange(smooth_window - 1, len(losses)),
            smoothed,
            color="tab:blue",
            label=f"Loss (MA {smooth_window})",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_samples(images: torch.Tensor, nrow: int = 4, title: str = "Generated Samples") -> None:
    """
    Display a grid of generated images.
    images: [N, 1, H, W] tensor in [-1, 1]
    """
    imgs = (images.cpu().float() + 1) / 2   # [-1,1] -> [0,1]
    n = imgs.shape[0]
    ncols = nrow
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2), dpi=100)
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_denoising_trajectory(
    model: torch.nn.Module,
    nb_steps: int = 8,
) -> None:
    """
    Run one sample through the ODE and show snapshots at each step.
    Useful for visually confirming that noise gradually becomes a digit.
    """
    device = next(model.parameters()).device
    model.eval()

    x = torch.randn(1, 1, 28, 28, device=device)
    t_steps = torch.linspace(0, 1, nb_steps + 1, device=device)
    snapshots = [(0.0, x.squeeze().cpu().numpy())]

    with torch.inference_mode():
        for i in range(nb_steps):
            t  = t_steps[i]
            dt = t_steps[i + 1] - t
            t_batch = t.expand(1, 1)
            x = x + model(x, t_batch) * dt
            snapshots.append((t_steps[i + 1].item(), x.clamp(-1, 1).squeeze().cpu().numpy()))

    fig, axes = plt.subplots(1, len(snapshots), figsize=(len(snapshots) * 2, 2.5), dpi=100)
    for ax, (t_val, img) in zip(axes, snapshots):
        ax.imshow((img + 1) / 2, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={t_val:.2f}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Denoising Trajectory (one sample)", y=1.02)
    plt.tight_layout()
    plt.show()
