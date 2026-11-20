import torch
import lpips as lp
from torchmetrics.functional.image import structural_similarity_index_measure

loss_fn_alex = lp.LPIPS(net='alex')


def _lpips_eval(img1, img2):
    return loss_fn_alex(img1, img2, normalize=True).flatten()

def _psnr_eval(img1, img2):
    mse = ((img1 - img2) ** 2).mean(dim=(1, 2, 3))
    return 10 * torch.log10(1.0 / mse)

def _ssim_eval(img1, img2):
    return structural_similarity_index_measure(img1, img2, data_range=1.0, reduction="none")

@torch.no_grad()
def eval_metrics(img1, img2):
    img1 = img1.clamp(min=0, max=1)
    lpips = _lpips_eval(img1, img2)
    psnr = _psnr_eval(img1, img2)
    ssim = _ssim_eval(img1, img2)
    return lpips, psnr, ssim
