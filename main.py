import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data import load_dataset
from utils.model import EDSR, evaluate, train
from utils.metrics import eval_metrics

random.seed(42)
torch.manual_seed(42)

def bicubic_baseline(loader):
    lpips_all, psnr_all, ssim_all = [], [], []
    for lr, hr in loader:
        sr = F.interpolate(lr, scale_factor=4, mode="bicubic")
        lp, psnr, ssim = eval_metrics(sr, hr)
        lpips_all.append(lp)
        psnr_all.append(psnr)
        ssim_all.append(ssim)
    mean_psnr = torch.cat(psnr_all).mean().item()
    mean_ssim = torch.cat(ssim_all).mean().item()
    mean_lpips = torch.cat(lpips_all).mean().item()
    print(f"Bicubic x4 -> PSNR: {mean_psnr:.2f}, SSIM: {mean_ssim:.4f}, LPIPS: {mean_lpips:.4f}")

if __name__ == "__main__":
    # x, torch.Size([16, 3, 40, 40])
    # y, torch.Size([16, 3, 160, 160])
    train_set = load_dataset("train")
    train_mean = train_set.get_mean()
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    # batch_size=1, validation images have different sizes, so they can't be into one batch
    valid_set = load_dataset("valid")
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

    model = EDSR(n_blocks=32, number_features=256, scale=4, residual_scale=0.1, mean=train_mean)
    # 300 epochs in the paper
    model = train(model, train_loader, epochs=1)

    evaluate(model, valid_loader)
    bicubic_baseline(valid_loader)
