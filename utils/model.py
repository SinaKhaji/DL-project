# Enhanced Deep Residual Networks for Single Image Super-Resolution.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import eval_metrics

class ResBlock(nn.Module):
    def __init__(self, in_features=64, residual_scale = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, in_features, 3, padding=1)
        self.scale = residual_scale

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.scale # In each residual block, constant scaling layers are placed after the last convolution layers.
        return out + x

class Upsampler(nn.Module):
    def __init__(self, in_features=64, scale=4):
        super().__init__()
        layers = []

        # normally should be like the below but because of the scope of the project (X4), we only use it with scale=4.
        for _ in range(int(math.log2(scale))):
            layers.append(nn.Conv2d(in_features, in_features * (2 ** 2), kernel_size=3, padding=1)) # (B,64,32,32) --> (B,256,32,32)
            layers.append(nn.PixelShuffle(2)) # scale 2: (B,256,32,32) --> (B,64,64,64)

        self.upsampler = nn.Sequential(*layers)

    def forward(self, x):
        x = self.upsampler(x)
        return x

# we focus on the single-scale because the project description only narrows the scope to X4
class EDSR(nn.Module):
    def __init__(self, n_blocks=32, number_features=256, scale=4, residual_scale=0.1, mean=(0.5, 0.5, 0.5)):
        super().__init__()
        # Stored as a buffer, not a plain attribute, so it moves with model.to(device)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))

        self.head = nn.Conv2d(3, number_features, 3, padding=1)

        # Based on the paper it should be 48 but the project says 40, the actual code runs with 40.
        # Shape flow (x4, 48x48 input patch, number_features=256):
        #   input       (B,    3,  48,  48)   RGB image
        #   head        (B,  256,  48,  48)   3 -> 256 channels
        #   body        (B,  256,  48,  48)   every ResBlock keeps 256 channels
        #   upsampler   (B, 1024,  48,  48) -> PixelShuffle(2) -> (B, 256,  96,  96)
        #               (B, 1024,  96,  96) -> PixelShuffle(2) -> (B, 256, 192, 192)
        #   final conv  (B,    3, 192, 192)   256 -> 3 channels, the SR image
        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(in_features=number_features, residual_scale=residual_scale))
        blocks.append(nn.Conv2d(number_features, number_features, 3, padding=1))
        self.body = nn.Sequential(*blocks)

        self.tail = nn.Sequential(Upsampler(number_features, scale), nn.Conv2d(number_features, 3, 3, padding=1))

    def forward(self, x):
        out = x - self.mean
        out = self.head(out)
        out = out + self.body(out)
        out = self.tail(out)
        return  out + self.mean # target hr is in normal image space so we add back the mean


def train(model, train_loader, epochs=100):
    # Adam optimizer (β1=0.9, β2=0.999, ε=1e-8), minibatch size=16, initial learning rate=1e-4, halved every 2×10^5 minibatch updates.
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=200_000, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        for lr, hr in train_loader:
            new_lr = model(lr)
            loss = F.l1_loss(new_lr, hr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"epoch {epoch} | L1 loss {total / len(train_loader):.4f}")
    return model

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    lpips_all, psnr_all, ssim_all = [], [], []
    for lr, hr in loader:
        sr = model(lr)
        lp, psnr, ssim = eval_metrics(sr, hr)
        lpips_all.append(lp)
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    mean_psnr = torch.cat(psnr_all).mean().item()
    mean_ssim = torch.cat(ssim_all).mean().item()
    mean_lpips = torch.cat(lpips_all).mean().item()
    print(f"EDSR -> PSNR: {mean_psnr:.2f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")
