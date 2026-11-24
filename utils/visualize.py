from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from utils.data import DatasetSample
from utils.metrics import evaluate_image_metrics
from utils.model import DEVICE, EDSR, upscale_bicubic


# Cuts a square patch out of a (1, 3, H, W) image
def cut_patch(
    image: torch.Tensor,
    patch_start_x: int,
    patch_start_y: int,
    patch_size: int,
) -> torch.Tensor:
    patch_end_x: int = patch_start_x + patch_size
    patch_end_y: int = patch_start_y + patch_size
    # Rows (y, height) come before columns (x, width) in a tensor
    return image[..., patch_start_y:patch_end_y, patch_start_x:patch_end_x]


# Turns a (1, 3, H, W) tensor into an (H, W, 3) NumPy array, the format matplotlib shows
def to_picture(image: torch.Tensor) -> np.ndarray:
    single_image: torch.Tensor = image[0] # (1, 3, H, W) -> (3, H, W)
    channels_last: torch.Tensor = single_image.permute(1, 2, 0) # -> (H, W, 3)
    valid_pixels: torch.Tensor = channels_last.clamp(min = 0, max = 1)
    return valid_pixels.cpu().numpy() # matplotlib needs NumPy arrays on the CPU


# Title of a patch: its name and its scores compared with the HR patch
def make_patch_title(
    name: str,
    predicted_patch: torch.Tensor,
    high_resolution_patch: torch.Tensor,
) -> str:
    # Type hints only: they declare the types of the three results unpacked below
    lpips_scores: torch.Tensor
    psnr_scores: torch.Tensor
    ssim_scores: torch.Tensor
    lpips_scores, psnr_scores, ssim_scores = evaluate_image_metrics(
        predicted_patch,
        high_resolution_patch,
        border_size = 0, # the patch is already cut out, so score all of it
    )
    psnr_score: float = psnr_scores.item()
    ssim_score: float = ssim_scores.item()
    lpips_score: float = lpips_scores.item()
    psnr_and_ssim_line: str = f"PSNR {psnr_score:.2f} dB, SSIM {ssim_score:.4f}"
    lpips_line: str = f"LPIPS {lpips_score:.4f}"
    return f"{name}\n{psnr_and_ssim_line}\n{lpips_line}" # \n starts a new line


# Shows one patch in one panel of the figure
def show_patch(axis: Axes, patch: torch.Tensor, title: str) -> None:
    picture: np.ndarray = to_picture(patch)
    # "nearest" shows the real pixels, without smoothing them
    axis.imshow(picture, interpolation = "nearest")
    axis.set_title(title)


# Saves a figure like Figure 6 of the EDSR paper: the whole HR image with a yellow box,
# then the boxed patch from bicubic, EDSR and the HR ground truth, with their scores.
# The project asks for patches because on a whole image the differences are too small
# to see.
@torch.no_grad() # only predicting, so PyTorch does not need to track gradients
def save_patch_comparison(
    model: EDSR,
    sample: DatasetSample,
    patch_start_x: int,
    patch_start_y: int,
    output_path: Path,
    patch_size: int = 160,
) -> None:
    model = model.to(DEVICE)
    model.eval() # evaluation mode, as in evaluate()

    # The model works on batches, so add a batch dimension: (3, H, W) -> (1, 3, H, W)
    low_resolution_image: torch.Tensor = sample.lr.unsqueeze(0).to(DEVICE)
    high_resolution_image: torch.Tensor = sample.hr.unsqueeze(0).to(DEVICE)

    # Enlarge the whole LR image 4x in both ways
    bicubic_image: torch.Tensor = upscale_bicubic(low_resolution_image)
    edsr_image: torch.Tensor = model(low_resolution_image)

    # Cut the same patch out of the three images
    bicubic_patch: torch.Tensor = cut_patch(
        bicubic_image,
        patch_start_x,
        patch_start_y,
        patch_size,
    )
    edsr_patch: torch.Tensor = cut_patch(
        edsr_image,
        patch_start_x,
        patch_start_y,
        patch_size,
    )
    high_resolution_patch: torch.Tensor = cut_patch(
        high_resolution_image,
        patch_start_x,
        patch_start_y,
        patch_size,
    )

    # One row of 4 panels
    figure: Figure
    axes: np.ndarray
    figure, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (16, 5))

    # Panel 1: the whole HR image, with a yellow box around the patch
    whole_image_axis: Axes = axes[0]
    whole_picture: np.ndarray = to_picture(high_resolution_image)
    whole_image_axis.imshow(whole_picture)
    patch_outline: Rectangle = Rectangle(
        xy = (patch_start_x, patch_start_y), # top-left corner
        width = patch_size,
        height = patch_size,
        fill = False,
        edgecolor = "yellow",
        linewidth = 2,
    )
    whole_image_axis.add_patch(patch_outline)
    whole_image_axis.set_title("Whole HR image")

    # Panels 2 to 4: the same patch from bicubic, EDSR and the HR image
    bicubic_title: str = make_patch_title("Bicubic", bicubic_patch, high_resolution_patch)
    show_patch(axes[1], bicubic_patch, bicubic_title)
    edsr_title: str = make_patch_title("EDSR", edsr_patch, high_resolution_patch)
    show_patch(axes[2], edsr_patch, edsr_title)
    show_patch(axes[3], high_resolution_patch, "HR (ground truth)")

    # Hide the axis ticks: they only add noise to pictures
    axis: Axes
    for axis in axes:
        axis.axis("off")

    figure.tight_layout() # keep the titles from overlapping
    output_path.parent.mkdir(exist_ok = True) # create the folder if needed
    figure.savefig(output_path, dpi = 150)
    plt.close(figure) # free the memory the figure uses
    print(f"Saved the patch comparison to {output_path}")
