from functools import lru_cache

import torch
from torchmetrics.functional.image import structural_similarity_index_measure

import lpips


# LPIPS uses a neural network (AlexNet). Building it is slow, so it is built only once per
# device (CPU or GPU) and then reused: lru_cache remembers what the function returned.
@lru_cache(maxsize = None)
def _get_lpips_alex_metric(device: torch.device) -> lpips.LPIPS:
    """Build the LPIPS network once per device, then reuse it on every later call."""
    lpips_alex_metric: lpips.LPIPS = lpips.LPIPS(net = "alex")
    return lpips_alex_metric.to(device) # on the same device as the images


def _lpips_scores(
    predicted_images: torch.Tensor,
    ground_truth_images: torch.Tensor,
) -> torch.Tensor:
    """Return one LPIPS score per predicted and ground-truth image pair."""
    lpips_alex_metric: lpips.LPIPS = _get_lpips_alex_metric(predicted_images.device)

    # Lower is better: 0 means the two images look the same to the network
    scores: torch.Tensor = lpips_alex_metric(
        predicted_images,
        ground_truth_images,
        normalize = True, # our images are in [0, 1]; LPIPS rescales them to [-1, 1]
    )
    # One score per image: (B, 1, 1, 1) -> (B,)
    return scores.flatten()


def _remove_border(images: torch.Tensor, border_size: int) -> torch.Tensor:
    """Cut border_size pixels from every side of the images."""
    if border_size < 0:
        raise ValueError("border_size must not be negative.")
    if border_size == 0:
        return images # images[..., 0:-0] would be empty, so 0 is handled separately
    # (B, C, H, W) -> (B, C, H - 2 * border_size, W - 2 * border_size)
    # border_size:-border_size selects rows from the first row
    # after the top border to the first row before the bottom border.
    return images[..., border_size:-border_size, border_size:-border_size]


def _psnr_scores(
    predicted_images: torch.Tensor,
    ground_truth_images: torch.Tensor,
) -> torch.Tensor:
    """Return one PSNR score per predicted and ground-truth image pair."""
    # Each tensor has shape (B, C, H, W), for example (4, 3, 140, 140):
    # 4 images, 3 color channels, 140 x 140 pixels after removing a 10-pixel border.
    # Subtraction creates one tensor of squared-error operands per image:
    # (4, 3, 140, 140) -> (4, 3, 140, 140).
    squared_errors: torch.Tensor = (predicted_images - ground_truth_images) ** 2

    # Mean over C, H, and W leaves one MSE value per batch item:
    # (4, 3, 140, 140) -> (4,).
    mean_squared_errors: torch.Tensor = squared_errors.mean(dim = (1, 2, 3))

    # Avoid log(1 / 0) for identical images. Example:
    # tensor([0.0, 0.01]) -> tensor([eps, 0.01]).
    minimum_mse: float = torch.finfo(mean_squared_errors.dtype).eps
    mean_squared_errors = mean_squared_errors.clamp(min = minimum_mse)

    # Convert each MSE to decibels, preserving shape (B,):
    # tensor([eps, 0.01]) -> a tensor containing the PSNR values. Higher is better.
    return 10 * torch.log10(1.0 / mean_squared_errors)


def _ssim_scores(
    predicted_images: torch.Tensor,
    ground_truth_images: torch.Tensor,
) -> torch.Tensor:
    """Return one SSIM score per predicted and ground-truth image pair."""
    # Compares local brightness, contrast and structure; 1.0 means identical images
    return structural_similarity_index_measure(
        predicted_images,
        ground_truth_images,
        data_range = 1.0, # pixel values go from 0 to 1
        reduction = "none", # one score per image instead of one average for the batch
    )


@torch.no_grad() # we only measure here, so PyTorch does not need to track gradients
def evaluate_image_metrics(
    predicted_images: torch.Tensor,
    ground_truth_images: torch.Tensor,
    border_size: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return LPIPS, PSNR, and SSIM score tensors for a batch of image pairs.

    All three metrics compare the RGB images with border_size pixels removed on every
    side. This is how the EDSR paper evaluates on DIV2K, which is where the A+, SRCNN and
    VDSR baseline numbers come from: border_size = 6 + the upscaling factor 4 = 10.
    """
    # The model's output can go slightly below 0 or above 1; real images cannot
    predicted_images = predicted_images.clamp(min = 0, max = 1)

    # Pixels near the border have fewer neighbors to learn from, so they are not scored
    predicted_without_border: torch.Tensor = _remove_border(
        predicted_images,
        border_size,
    )
    ground_truth_without_border: torch.Tensor = _remove_border(
        ground_truth_images,
        border_size,
    )

    lpips_scores: torch.Tensor = _lpips_scores(
        predicted_without_border,
        ground_truth_without_border,
    )
    psnr_scores: torch.Tensor = _psnr_scores(
        predicted_without_border,
        ground_truth_without_border,
    )
    ssim_scores: torch.Tensor = _ssim_scores(
        predicted_without_border,
        ground_truth_without_border,
    )
    return lpips_scores, psnr_scores, ssim_scores
