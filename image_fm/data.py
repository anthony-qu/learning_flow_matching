from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataloader(batch_size: int = 128, data_dir: str = "data", max_samples: int | None = None) -> DataLoader:
    """Download MNIST and return a DataLoader with images normalized to [-1, 1].

    Set max_samples to use only the first N images (e.g. 5000 for quick experiments).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                   # [0, 1]
        transforms.Normalize((0.5,), (0.5,)),    # [-1, 1]
    ])
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
