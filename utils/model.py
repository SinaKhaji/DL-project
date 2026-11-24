# Enhanced Deep Residual Networks for Single Image Super-Resolution.
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils.metrics import evaluate_image_metrics
from utils.data import DatasetSample


# Where the model runs: the GPU if there is one (for example on Colab), otherwise the CPU
DEVICE: torch.device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# One residual block: conv -> ReLU -> conv, then the block's input is added back.
# Unlike ResNet there is no batch normalization: the EDSR paper found that removing it
# gives better super-resolution results and uses less memory.
class ResBlock(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        # 3x3 kernels with padding 1 keep the height and width unchanged
        self.first_conv: nn.Conv2d = nn.Conv2d(
            in_channels = num_channels,
            out_channels = num_channels,
            kernel_size = 3,
            padding = 1,
        )
        # inplace = True reuses the input's memory instead of making a new tensor
        self.activation: nn.ReLU = nn.ReLU(inplace = True)
        self.second_conv: nn.Conv2d = nn.Conv2d(
            in_channels = num_channels,
            out_channels = num_channels,
            kernel_size = 3,
            padding = 1,
        )
        self.residual_scale: float = residual_scale

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = self.first_conv(input_tensor)
        residual = self.activation(residual)
        residual = self.second_conv(residual)
        # Scaling the residual down (by 0.1) keeps training stable in a wide network
        residual = residual * self.residual_scale
        # Skip connection: the block only has to learn a correction to its input
        return residual + input_tensor


# Makes the feature maps 4 times larger (height and width) in two x2 steps
class Upsampler(nn.Module):
    def __init__(
        self,
        num_channels: int = 64,
        scale_factor: int = 4,
    ) -> None:
        super().__init__()
        if scale_factor != 4:
            raise ValueError("Upsampler only supports a scale factor of 4.")

        # In each x2 step, a convolution makes 4 times more channels (2 x 2 = 4), then
        # PixelShuffle(2) rearranges each group of 4 channels into a 2x2 block of pixels:
        # (B, 4 * C, H, W) -> (B, C, 2 * H, 2 * W)
        self.upsampler: nn.Sequential = nn.Sequential(
            # First x2 step
            nn.Conv2d(
                in_channels = num_channels,
                out_channels = num_channels * 4,
                kernel_size = 3,
                padding = 1,
            ),
            nn.PixelShuffle(2),
            # Second x2 step
            nn.Conv2d(
                in_channels = num_channels,
                out_channels = num_channels * 4,
                kernel_size = 3,
                padding = 1,
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.upsampler(input_tensor)


# We focus on the single-scale because the project description
# only narrows the scope to X4
class EDSR(nn.Module):
    def __init__(
        self,
        num_residual_blocks: int = 32,
        num_features: int = 256,
        scale_factor: int = 4,
        residual_scale: float = 0.1,
        mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()
        if scale_factor != 4:
            raise ValueError("EDSR only supports a scale factor of 4.")

        # The average color (red, green, blue) of the training images, as a (1, 3, 1, 1)
        # tensor so it can be subtracted from a whole (B, 3, H, W) batch
        mean_color: torch.Tensor = torch.tensor(mean) # shape (3,)
        mean_color = mean_color.view(1, 3, 1, 1) # shape (1, 3, 1, 1)
        # Stored as a buffer, not a plain attribute, so it moves with model.to(device)
        # and is saved with the weights, but the optimizer never changes it.
        self.register_buffer("mean", mean_color)

        # Head: one convolution from 3 color channels to num_features feature maps
        self.head: nn.Conv2d = nn.Conv2d(
            in_channels = 3,
            out_channels = num_features,
            kernel_size = 3,
            padding = 1,
        )

        # Based on the paper it should be 48 but the project says 40,
        # the actual code runs with 40.
        # Shape flow (x4, 40x40 input patch, num_features = 256):
        #   input       (B,    3,  40,  40)   RGB image
        #   head        (B,  256,  40,  40)   3 -> 256 channels
        #   body        (B,  256,  40,  40)   every ResBlock keeps 256 channels
        #   upsampler   (B, 1024,  40,  40) -> PixelShuffle(2) -> (B, 256,  80,  80)
        #               (B, 1024,  80,  80) -> PixelShuffle(2) -> (B, 256, 160, 160)
        #   final conv  (B,    3, 160, 160)   256 -> 3 channels, the SR image

        # Body: num_residual_blocks residual blocks, then one more convolution
        body_layers: list[nn.Module] = list()
        for _ in range(num_residual_blocks):
            residual_block: ResBlock = ResBlock(
                num_channels = num_features,
                residual_scale = residual_scale,
            )
            body_layers.append(residual_block)
        
        body_last_conv: nn.Conv2d = nn.Conv2d(
            in_channels = num_features,
            out_channels = num_features,
            kernel_size = 3,
            padding = 1,
        )
        body_layers.append(body_last_conv)
        # The * unpacks the list: nn.Sequential(block_1, block_2, ..., body_last_conv)
        self.body: nn.Sequential = nn.Sequential(*body_layers)

        # Tail: make the feature maps 4 times larger, then go back to 3 color channels
        upsampler: Upsampler = Upsampler(
            num_channels = num_features,
            scale_factor = scale_factor,
        )
        output_conv: nn.Conv2d = nn.Conv2d(
            in_channels = num_features,
            out_channels = 3,
            kernel_size = 3,
            padding = 1,
        )
        self.tail: nn.Sequential = nn.Sequential(upsampler, output_conv)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Subtract the average color so the input is centered around 0
        normalized_input: torch.Tensor = input_tensor - self.mean
        features: torch.Tensor = self.head(normalized_input)
        # Global skip connection: the body only learns what to add to the head's features
        features = features + self.body(features)
        super_resolution: torch.Tensor = self.tail(features)
        # Add the average color back, so the output is a normal image again
        return super_resolution + self.mean


# The bicubic baseline: enlarges images 4x with bicubic interpolation, no learning.
# With antialias = True, PyTorch's bicubic gives almost the same pixels as PIL and
# MATLAB, which the EDSR paper's bicubic numbers come from: 26.66 dB on the DIV2K
# validation set, like the paper (26.79 dB without it)
def upscale_bicubic(low_resolution_images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        low_resolution_images,
        scale_factor = 4,
        mode = "bicubic",
        antialias = True,
    )


# Saves everything needed to continue training later: the model, the optimizer and the
# scheduler, plus the number of the epoch that just finished. The file is first written
# under a temporary name and then renamed, so if Colab stops in the middle of writing,
# the previous checkpoint is still complete.
def save_checkpoint(
    checkpoint_path: Path,
    finished_epoch: int,
    model: EDSR,
    optimizer: Adam,
    scheduler: StepLR,
) -> None:
    checkpoint: dict[str, Any] = {
        "finished_epoch": finished_epoch,
        "model": model.state_dict(), # the weights and the mean buffer
        "optimizer": optimizer.state_dict(), # Adam's running averages of the gradients
        "scheduler": scheduler.state_dict(), # how many batches were done so far
    }
    temporary_path: Path = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path) # rename; replaces the previous checkpoint


# Loads a checkpoint into the model, optimizer and scheduler and returns the epoch to
# continue with. Without a checkpoint file, training starts at epoch 1.
def load_checkpoint(
    checkpoint_path: Path,
    model: EDSR,
    optimizer: Adam,
    scheduler: StepLR,
) -> int:
    if not checkpoint_path.exists():
        print(f"No checkpoint at {checkpoint_path}, starting at epoch 1")
        return 1
    # map_location = DEVICE puts every saved tensor on the device the model is on
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location = DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    finished_epoch: int = checkpoint["finished_epoch"]
    print(f"Loaded the checkpoint: epoch {finished_epoch} was the last one finished")
    return finished_epoch + 1


# Trains the model on the training patches until num_epochs epochs are done in total.
# After a crash, call it again with the same checkpoint_path: it continues after the
# last saved epoch.
def train(
    model: EDSR,
    train_loader: DataLoader[DatasetSample],
    checkpoint_path: Path,
    num_epochs: int = 100,
) -> EDSR:
    # Move the weights and the mean buffer to the GPU, if there is one
    model = model.to(DEVICE)
    print(f"Training on {DEVICE}")

    # Settings from the EDSR paper: Adam (β1 = 0.9, β2 = 0.999, ε = 1e-8, which are
    # PyTorch's defaults), minibatch size 16, initial learning rate 1e-4,
    # halved every 2×10^5 minibatch updates.
    optimizer: Adam = Adam(model.parameters(), lr = 1e-4)
    # scheduler.step() runs after every batch, so step_size counts batches, not epochs.
    # 1000 batches per epoch (16000 patches / 16), so the first halving is after epoch 200
    scheduler: StepLR = StepLR(
        optimizer,
        step_size = 200_000,
        gamma = 0.5, # multiply the learning rate by 0.5
    )

    # Continue where the last run stopped, if a checkpoint was saved
    first_epoch: int = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    if first_epoch > num_epochs:
        print("Nothing to train: all epochs are already done")
    checkpoint_path.parent.mkdir(parents = True, exist_ok = True) # create missing folders

    for epoch in range(first_epoch, num_epochs + 1):
        # Training mode. EDSR has no dropout or batch norm, so this changes nothing here,
        # but it is good practice (evaluate() switches to model.eval())
        model.train()
        epoch_loss_total: float = 0.0
        # Types of the loop variables (they can't be written in the for line itself)
        low_resolution_images: torch.Tensor
        high_resolution_images: torch.Tensor
        for low_resolution_images, high_resolution_images in train_loader:
            # The batch has to be on the same device as the model
            low_resolution_images = low_resolution_images.to(DEVICE)
            high_resolution_images = high_resolution_images.to(DEVICE)

            super_resolution_images: torch.Tensor = model(low_resolution_images)
            # L1 loss: the average absolute pixel difference (the paper uses L1, not L2)
            batch_loss: torch.Tensor = F.l1_loss(
                super_resolution_images,
                high_resolution_images,
            )
            optimizer.zero_grad() # remove the gradients of the previous batch
            batch_loss.backward() # compute new gradients
            optimizer.step() # update the weights
            scheduler.step() # count one more minibatch update
            epoch_loss_total += batch_loss.item() # .item() gives a Python float
        # len(train_loader) is the number of batches, so this is the average batch loss
        mean_epoch_loss: float = epoch_loss_total / len(train_loader)
        print(f"epoch {epoch} | L1 loss {mean_epoch_loss:.4f}")
        # Save after every epoch: a crash then loses at most the epoch that was running
        save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler)
    return model


# Measures PSNR, SSIM and LPIPS on every image of the loader and prints the averages
@torch.no_grad() # only measuring, so PyTorch does not need to track gradients
def evaluate(
    model: EDSR,
    loader: DataLoader[DatasetSample],
) -> None:
    model = model.to(DEVICE)
    model.eval() # evaluation mode, the opposite of model.train()
    # One tensor of scores per batch; they are joined at the end
    lpips_batches: list[torch.Tensor] = list()
    psnr_batches: list[torch.Tensor] = list()
    ssim_batches: list[torch.Tensor] = list()
    # Types of the loop variables (they can't be written in the for line itself)
    low_resolution_images: torch.Tensor
    high_resolution_images: torch.Tensor
    for low_resolution_images, high_resolution_images in loader:
        # The images have to be on the same device as the model
        low_resolution_images = low_resolution_images.to(DEVICE)
        high_resolution_images = high_resolution_images.to(DEVICE)

        super_resolution_images: torch.Tensor = model(low_resolution_images)
        # Type hints only: they declare the types of the three results unpacked below
        lpips_scores: torch.Tensor
        psnr_scores: torch.Tensor
        ssim_scores: torch.Tensor
        lpips_scores, psnr_scores, ssim_scores = evaluate_image_metrics(
            super_resolution_images,
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
        f"EDSR -> PSNR: {mean_psnr:.2f}, "
        f"SSIM: {mean_ssim:.4f}, "
        f"LPIPS: {mean_lpips:.4f}"
    )
