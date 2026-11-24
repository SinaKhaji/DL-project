import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from PIL import Image

import random

from pathlib import Path
from dataclasses import dataclass
from typing import Literal, NamedTuple, Sequence, overload


# parents[1] is the same as .parent.parent
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1] # the DL-project folder
DATA_DIR: Path = PROJECT_ROOT / "data" # the DIV2K folders are in here
UINT8_MAX: int = torch.iinfo(torch.uint8).max # 255, the largest value of an 8-bit pixel


# File paths of one high-resolution (HR) image and its 4x smaller
# low-resolution (LR) version
@dataclass(frozen = True)
class ImagePathPair:
    hr: Path
    lr: Path


# The same pair after both files have been opened as PIL images
@dataclass(frozen = True)
class ImagePair:
    hr: Image.Image
    lr: Image.Image


# Opens an image file and makes sure it has 3 color channels (red, green, blue)
def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image: # "with" closes the file when we are done
        return image.convert("RGB")


# Opens both files of a pair and checks that the HR image is exactly
# lr_to_hr_scale times larger than the LR image
def load_image_pair(
    image_path_pair: ImagePathPair,
    lr_to_hr_scale: int = 4
) -> ImagePair:
    hr_image: Image.Image = load_rgb_image(image_path_pair.hr)
    lr_image: Image.Image = load_rgb_image(image_path_pair.lr)

    # Every LR pixel must match a 4x4 block of HR pixels, or the images don't line up
    expected_hr_width: int = lr_image.width * lr_to_hr_scale
    expected_hr_height: int = lr_image.height * lr_to_hr_scale
    if hr_image.width != expected_hr_width or hr_image.height != expected_hr_height:
        raise ValueError(
            f"{image_path_pair.hr.name} is {hr_image.width}x{hr_image.height}, but it "
            f"should be {expected_hr_width}x{expected_hr_height} "
            f"({lr_to_hr_scale} times {image_path_pair.lr.name})."
        )
    return ImagePair(hr = hr_image, lr = lr_image)


# Turns 0-255 integers (uint8) into 0-1 floats, the format the model works with
def uint8_to_float_image(image: torch.Tensor) -> torch.Tensor:
    float_image: torch.Tensor = image.to(torch.float32)
    return float_image / UINT8_MAX


# One sample as the DataLoader sees it: what the model gets and what it should produce
class DatasetSample(NamedTuple):
    lr: torch.Tensor # model input
    hr: torch.Tensor # target


class SRWholeImageDataset(Dataset):
    """Whole images for validation, read once and kept in memory as uint8."""

    def __init__(self, image_path_pairs: Sequence[ImagePathPair]) -> None:
        super().__init__()
        # Each image is a uint8 tensor [3, H, W]; H and W differ per image
        lr_images: list[torch.Tensor] = list()
        hr_images: list[torch.Tensor] = list()

        # Read every image once; pil_to_tensor keeps the pixels as 0-255 integers (uint8)
        for image_path_pair in image_path_pairs:
            image_pair: ImagePair = load_image_pair(image_path_pair)
            lr_image: torch.Tensor = pil_to_tensor(image_pair.lr)
            hr_image: torch.Tensor = pil_to_tensor(image_pair.hr)
            lr_images.append(lr_image)
            hr_images.append(hr_image)

        # A tuple of separate tensors, not one big tensor, because the image sizes differ
        self.lr_images: tuple[torch.Tensor, ...] = tuple(lr_images)
        self.hr_images: tuple[torch.Tensor, ...] = tuple(hr_images)

    # Number of samples: one per image
    def __len__(self) -> int:
        return len(self.lr_images)

    # Called by the DataLoader for each sample: turns 0-255 integers into 0-1 floats
    def __getitem__(self, index: int) -> DatasetSample:
        lr_image: torch.Tensor = uint8_to_float_image(self.lr_images[index])
        hr_image: torch.Tensor = uint8_to_float_image(self.hr_images[index])
        return DatasetSample(lr = lr_image, hr = hr_image)


# Cuts one random LR patch and the HR patch that shows exactly the same spot
def random_crop_image_pair(
    image_pair: ImagePair,
    lr_patch_size: int = 40,
    lr_to_hr_scale: int = 4,
) -> ImagePair:
    # Reject sizes that make no sense
    if lr_patch_size <= 0:
        raise ValueError("lr_patch_size must be positive.")
    if lr_to_hr_scale <= 0:
        raise ValueError("lr_to_hr_scale must be positive.")

    # load_image_pair already checked that HR is exactly lr_to_hr_scale times larger
    lr_image: Image.Image = image_pair.lr
    hr_image: Image.Image = image_pair.hr

    # The patch has to fit inside the LR image
    if lr_patch_size > lr_image.width or lr_patch_size > lr_image.height:
        raise ValueError(
            f"LR patch size {lr_patch_size} exceeds LR image dimensions "
            f"{lr_image.width}x{lr_image.height}."
        )

    hr_patch_size: int = lr_patch_size * lr_to_hr_scale # 40 * 4 = 160

    # Largest start position that keeps the whole patch inside the LR image
    max_lr_patch_start_x: int = lr_image.width - lr_patch_size
    max_lr_patch_start_y: int = lr_image.height - lr_patch_size

    # Random start position (x, y) of the LR patch
    lr_patch_start_x: int = random.randint(0, max_lr_patch_start_x)
    lr_patch_start_y: int = random.randint(0, max_lr_patch_start_y)

    # The same spot in the HR image: every LR pixel matches a 4x4 block of HR pixels
    hr_patch_start_x: int = lr_patch_start_x * lr_to_hr_scale
    hr_patch_start_y: int = lr_patch_start_y * lr_to_hr_scale

    # A PIL crop box is (start_x, start_y, end_x, end_y)
    lr_patch_box: tuple[int, int, int, int] = (
        lr_patch_start_x,
        lr_patch_start_y,
        lr_patch_start_x + lr_patch_size,
        lr_patch_start_y + lr_patch_size,
    )
    hr_patch_box: tuple[int, int, int, int] = (
        hr_patch_start_x,
        hr_patch_start_y,
        hr_patch_start_x + hr_patch_size,
        hr_patch_start_y + hr_patch_size,
    )
    lr_patch: Image.Image = lr_image.crop(lr_patch_box)
    hr_patch: Image.Image = hr_image.crop(hr_patch_box)
    return ImagePair(hr = hr_patch, lr = lr_patch)


class SRPatchDataset(Dataset):
    """Random patches stored as uint8 and returned as float tensors in [0, 1]."""

    def __init__(
        self,
        image_path_pairs: Sequence[ImagePathPair],
        lr_patch_size: int = 40,
        patches_per_image: int = 20,
        lr_to_hr_scale: int = 4,
    ) -> None:
        super().__init__()
        # Check the arguments before starting the slow work
        if not image_path_pairs:
            raise ValueError("image_path_pairs must not be empty.")
        if lr_patch_size <= 0:
            raise ValueError("lr_patch_size must be positive.")
        if patches_per_image <= 0:
            raise ValueError("patches_per_image must be positive.")
        if lr_to_hr_scale <= 0:
            raise ValueError("lr_to_hr_scale must be positive.")

        # For example 800 images * 20 patches = 16000 patches
        patch_count: int = len(image_path_pairs) * patches_per_image
        hr_patch_size: int = lr_patch_size * lr_to_hr_scale

        # Reserve memory for all patches at once, as 0-255 integers.
        # Shape: [patch, channel, height, width]
        self.lr_patches: torch.Tensor = torch.empty(
            (patch_count, 3, lr_patch_size, lr_patch_size),
            dtype = torch.uint8,
        )
        self.hr_patches: torch.Tensor = torch.empty(
            (patch_count, 3, hr_patch_size, hr_patch_size),
            dtype = torch.uint8,
        )
        
        # Open each image only once and cut several random patches from it
        for image_index, image_path_pair in enumerate(image_path_pairs):
            image_pair: ImagePair = load_image_pair(
                image_path_pair,
                lr_to_hr_scale = lr_to_hr_scale,
            )
            for patch_index in range(patches_per_image):
                patch_pair: ImagePair = random_crop_image_pair(
                    image_pair,
                    lr_patch_size = lr_patch_size,
                    lr_to_hr_scale = lr_to_hr_scale,
                )
                lr_patch: torch.Tensor = pil_to_tensor(patch_pair.lr)
                hr_patch: torch.Tensor = pil_to_tensor(patch_pair.hr)
                # Where this patch goes in the big tensors
                tensor_index: int = image_index * patches_per_image + patch_index
                self.lr_patches[tensor_index] = lr_patch
                self.hr_patches[tensor_index] = hr_patch

    # Number of samples: one per patch
    def __len__(self) -> int:
        return len(self.lr_patches)

    # Called by the DataLoader for each sample: turns 0-255 integers into 0-1 floats
    def __getitem__(self, index: int) -> DatasetSample:
        lr_patch: torch.Tensor = uint8_to_float_image(self.lr_patches[index])
        hr_patch: torch.Tensor = uint8_to_float_image(self.hr_patches[index])
        return DatasetSample(lr = lr_patch, hr = hr_patch)

    # Average red, green and blue value of all LR patches
    # (EDSR subtracts these from its input)
    def get_red_mean(self) -> float:
        return self.get_channel_mean(0)

    def get_green_mean(self) -> float:
        return self.get_channel_mean(1)

    def get_blue_mean(self) -> float:
        return self.get_channel_mean(2)

    # All three averages as (red, green, blue)
    def get_means(self) -> tuple[float, float, float]:
        return (self.get_red_mean(), self.get_green_mean(), self.get_blue_mean())

    # Average of one color channel (0 = red, 1 = green, 2 = blue), between 0 and 1
    def get_channel_mean(self, channel_index: int) -> float:
        # This channel of every patch: [patch, height, width]
        channel: torch.Tensor = self.lr_patches[:, channel_index]
        # Sum as float64: adding millions of numbers in float32 would lose precision
        channel_sum: float = channel.sum(dtype = torch.float64).item()
        pixel_count: int = channel.numel()
        mean_pixel_value: float = channel_sum / pixel_count # between 0 and 255
        return mean_pixel_value / UINT8_MAX


# Finds every HR image of a partition ("train" or "valid") and its matching LR file
def load_pairs(partition: Literal["train", "valid"]) -> tuple[ImagePathPair, ...]:
    hr_dir: Path = DATA_DIR / f"DIV2K_{partition}_HR"
    lr_dir: Path = DATA_DIR / f"DIV2K_{partition}_LR_bicubic" / "X4"
    # Stop with a clear message if a dataset folder is missing
    if not hr_dir.is_dir():
        raise FileNotFoundError(f"HR image directory does not exist: {hr_dir}")
    if not lr_dir.is_dir():
        raise FileNotFoundError(f"LR image directory does not exist: {lr_dir}")

    # Sorted, so the order is the same in every run
    hr_paths: list[Path] = sorted(hr_dir.glob("*.png"))
    image_path_pairs: list[ImagePathPair] = list()
    for hr_path in hr_paths:
        lr_path: Path = lr_dir / f"{hr_path.stem}x4.png" # e.g. 0001.png -> 0001x4.png
        if not lr_path.is_file():
            raise FileNotFoundError(
                f"LR image for {hr_path.name} does not exist: {lr_path}."
            )
        image_path_pair: ImagePathPair = ImagePathPair(hr = hr_path, lr = lr_path)
        image_path_pairs.append(image_path_pair)

    if not image_path_pairs:
        raise ValueError(f"No HR image pairs found for partition {partition!r}")

    print(f"Found {len(image_path_pairs)} {partition} image pairs")
    return tuple(image_path_pairs)


# The two @overload versions only tell type checkers (like VS Code) which class each
# partition returns; the function that actually runs is the third one
@overload
def load_dataset(partition: Literal["train"] = "train") -> SRPatchDataset: ...


@overload
def load_dataset(partition: Literal["valid"]) -> SRWholeImageDataset: ...


# Training gets random patches, validation gets whole images
def load_dataset(
    partition: Literal["train", "valid"] = "train",
) -> SRPatchDataset | SRWholeImageDataset:
    image_path_pairs: tuple[ImagePathPair, ...] = load_pairs(partition)
    if partition == "train":
        return SRPatchDataset(image_path_pairs)
    if partition == "valid":
        return SRWholeImageDataset(image_path_pairs)
    raise ValueError(f"Unsupported dataset partition: {partition!r}")
