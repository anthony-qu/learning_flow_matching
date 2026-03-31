import torch
from .model import UNet


@torch.inference_mode()
def sample(
    model: UNet,
    n_samples: int = 16,
    nb_steps: int = 100,
    image_size: int = 28,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Generate images by integrating the learned velocity field with Euler's method.

    ODE:  dx/dt = model(x_t, t)
    Step: x_{t+dt} = x_t + model(x_t, t) * dt

    image_size must match what the model was trained on (28 for MNIST, 64/128 for photos).
    in_channels is read automatically from model.in_channels.

    Returns: [n_samples, C, image_size, image_size] tensor in [-1, 1]
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    in_channels = getattr(model, 'in_channels', 1)
    x = torch.randn(n_samples, in_channels, image_size, image_size, device=device)
    t_steps = torch.linspace(0, 1, nb_steps + 1, device=device)

    for i in range(nb_steps):
        t  = t_steps[i]
        dt = t_steps[i + 1] - t
        t_batch = t.expand(n_samples, 1)        # [B, 1]
        v = model(x, t_batch)
        x = x + v * dt

    return x.clamp(-1, 1)
