import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.data import (
    PROJECT_ROOT,
    DatasetSample,
    SRPatchDataset,
    SRWholeImageDataset,
    load_dataset,
)
from utils.model import DEVICE, EDSR, evaluate, train, upscale_bicubic
from utils.metrics import evaluate_image_metrics
from utils.visualize import save_patch_comparison

# Fixed seeds: every run gets the same random patches, starting weights and shuffle order
random.seed(1)
torch.manual_seed(1)

# The trained weights are saved here, so the model can be reused without training again
TRAINED_MODEL_PATH: Path = PROJECT_ROOT / "checkpoints" / "edsr_x4.pt"
# Saved after every epoch, so training can continue after a crash. On Colab, keep the
# project folder on Google Drive: files on Colab's own disk disappear with the session.
CHECKPOINT_PATH: Path = PROJECT_ROOT / "checkpoints" / "edsr_x4_checkpoint.pt"
# Figures for the report are saved here
FIGURES_DIR: Path = PROJECT_ROOT / "figures"


# Simplest baseline: enlarge every LR image 4x with bicubic interpolation (no learning).
# EDSR is only useful if it scores better than this.
def bicubic_baseline(
    validation_loader: DataLoader[DatasetSample],
) -> None:
    # One tensor of scores per batch; they are joined at the end
    lpips_batches: list[torch.Tensor] = list()
    psnr_batches: list[torch.Tensor] = list()
    ssim_batches: list[torch.Tensor] = list()
    # Types of the loop variables (they can't be written in the for line itself)
    low_resolution_images: torch.Tensor
    high_resolution_images: torch.Tensor
    for low_resolution_images, high_resolution_images in validation_loader:
        # Same device as the EDSR evaluation: the GPU if there is one
        low_resolution_images = low_resolution_images.to(DEVICE)
        high_resolution_images = high_resolution_images.to(DEVICE)

        # Enlarge 4x: (B, 3, H, W) -> (B, 3, 4 * H, 4 * W). The result can go slightly
        # below 0 or above 1; evaluate_image_metrics clamps it
        bicubic_images: torch.Tensor = upscale_bicubic(low_resolution_images)
        # Type hints only: they declare the types of the three results unpacked below
        lpips_scores: torch.Tensor
        psnr_scores: torch.Tensor
        ssim_scores: torch.Tensor
        lpips_scores, psnr_scores, ssim_scores = evaluate_image_metrics(
            bicubic_images,
            high_resolution_images,
        )
        lpips_batches.append(lpips_scores)
        psnr_batches.append(psnr_scores)
        ssim_batches.append(ssim_scores)
    # torch.cat joins the batches into one tensor with a score per image
    all_psnr_scores: torch.Tensor = torch.cat(psnr_batches)
    all_ssim_scores: torch.Tensor = torch.cat(ssim_batches)
    all_lpips_scores: torch.Tensor = torch.cat(lpips_batches)
    # .mean() averages over the images, .item() turns the result into a Python float
    mean_psnr: float = all_psnr_scores.mean().item()
    mean_ssim: float = all_ssim_scores.mean().item()
    mean_lpips: float = all_lpips_scores.mean().item()
    print(
        f"Bicubic x4 -> PSNR: {mean_psnr:.2f}, "
        f"SSIM: {mean_ssim:.4f}, "
        f"LPIPS: {mean_lpips:.4f}"
    )


# This part runs only when the file is started directly (python main.py), not on import
if __name__ == "__main__":
    # Training data: random patches. One batch has lr of shape (16, 3, 40, 40),
    # the model input, and hr of shape (16, 3, 160, 160), the target
    train_set: SRPatchDataset = load_dataset("train")
    # Average color (red, green, blue) of the training patches; EDSR subtracts it first
    train_mean: tuple[float, float, float] = train_set.get_means()
    train_loader: DataLoader[DatasetSample] = DataLoader(
        train_set,
        batch_size = 16,
        shuffle = True, # a new random order in every epoch
    )

    # Validation images have different sizes, so use batch_size = 1.
    valid_set: SRWholeImageDataset = load_dataset("valid")
    valid_loader: DataLoader[DatasetSample] = DataLoader(
        valid_set,
        batch_size = 1,
        shuffle = False, # the same order every time
    )

    # The paper's big EDSR: 32 residual blocks, 256 feature maps, residual scaling 0.1
    model: EDSR = EDSR(
        num_residual_blocks = 32,
        num_features = 256,
        scale_factor = 4,
        residual_scale = 0.1,
        mean = train_mean,
    )
    # The paper counts minibatch updates, not epochs. The official EDSR code trains for
    # 300 epochs of 1000 batches of 16 (300,000 updates): the same epoch size as ours.
    # num_epochs is the total: after a crash, run main.py again and training continues
    # from the last checkpoint until num_epochs epochs are done. This run uses one epoch.
    model = train(
        model,
        train_loader,
        checkpoint_path = CHECKPOINT_PATH,
        num_epochs = 1,
    )

    # Save before evaluating, so the training is not lost if the evaluation fails.
    # state_dict() holds the weights and the mean buffer. To load them later:
    # model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location = DEVICE))
    # Create the checkpoints folder; exist_ok = True means no error if it already exists
    TRAINED_MODEL_PATH.parent.mkdir(exist_ok = True)
    torch.save(model.state_dict(), TRAINED_MODEL_PATH)
    print(f"Saved the trained model to {TRAINED_MODEL_PATH}")

    evaluate(model, valid_loader)
    bicubic_baseline(valid_loader)

    # Qualitative example for the report: whiskers of the cat in 0869.png, the DIV2K
    # image the EDSR paper also shows in its Figure 6. Validation images start at
    # 0801.png (index 0), so 0869.png is index 68.
    cat_sample: DatasetSample = valid_set[68]
    save_patch_comparison(
        model,
        cat_sample,
        patch_start_x = 360,
        patch_start_y = 800,
        output_path = FIGURES_DIR / "patch_comparison_0869.png",
    )
