from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image


class PhotoDataset(Dataset):
    """
    Loads JPG/JPEG/PNG images from a directory.
    - Resizes so the shorter side = image_size, then center-crops to image_size x image_size
    - Optionally converts to grayscale
    - Normalizes to [-1, 1]
    """

    EXTENSIONS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")

    def __init__(self, photo_dir: str, image_size: int = 64, grayscale: bool = False) -> None:
        paths = []
        for ext in self.EXTENSIONS:
            paths.extend(Path(photo_dir).glob(ext))
        if not paths:
            raise FileNotFoundError(
                f"No jpg/jpeg/png images found in '{photo_dir}'.\n"
                f"Put your images there and re-run."
            )
        self.paths = sorted(paths)
        self.in_channels = 1 if grayscale else 3

        pipeline = [
            transforms.Resize(image_size),            # shorter side → image_size
            transforms.CenterCrop(image_size),         # square crop
        ]
        if grayscale:
            pipeline.append(transforms.Grayscale(num_output_channels=1))
        pipeline += [
            transforms.ToTensor(),                     # [0, 1]
            transforms.Normalize(
                mean=[0.5] * self.in_channels,
                std=[0.5]  * self.in_channels,
            ),                                         # [-1, 1]
        ]
        self.transform = transforms.Compose(pipeline)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def get_photo_dataloader(
    photo_dir: str = "data/photos",
    image_size: int = 128,
    grayscale: bool = False,
    batch_size: int = 2,
    max_samples: int | None = None,
) -> tuple[DataLoader, int]:
    """
    Build a DataLoader from your own photos.

    Returns:
        dataloader  — ready to pass into train()
        in_channels — 1 if grayscale, 3 if RGB (pass this to UNet)

    Args:
        photo_dir:   folder containing your JPG/PNG files
        image_size:  all images are resized + center-cropped to (image_size x image_size)
        grayscale:   True to convert to single-channel
        batch_size:  keep small (16–32) if you have few photos
        max_samples: cap the dataset size for quick experiments
    """
    dataset = PhotoDataset(photo_dir, image_size=image_size, grayscale=grayscale)
    in_channels = dataset.in_channels
    print(f"Found {len(dataset)} images in '{photo_dir}' → {in_channels}ch, {image_size}x{image_size}")

    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
        print(f"Subsetting to {len(dataset)} samples")

    n = len(dataset)
    effective_batch = min(batch_size, n)
    if effective_batch < batch_size:
        print(f"Warning: only {n} images — reducing batch_size from {batch_size} to {effective_batch}")

    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch,
        shuffle=True,
        drop_last=False,   # keep the last incomplete batch when dataset is small
        num_workers=2,
        pin_memory=True,
    )
    return dataloader, in_channels
