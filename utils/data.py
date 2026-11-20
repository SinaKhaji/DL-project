import torch
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class SRDataset(Dataset):
    def __init__(self, image_pairs):
        self.image_pairs = image_pairs
        self.mean = None

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, i):
        hr, lr = self.image_pairs[i]
        return to_tensor(lr), to_tensor(hr)

    def get_mean(self):
        if self.mean is None:
            total = torch.zeros(3, dtype=torch.float64)
            count = 0
            for _, lr in self.image_pairs:
                img = to_tensor(lr).double()
                total += img.sum(dim=(1, 2)) # adds up all the pixels separately for each channel
                count += img.shape[1] * img.shape[2] # number of pixels in that image
            self.mean = tuple((total / count).tolist())
        return self.mean

def load_pairs(split):
    hr_dir = DATA_DIR / f"DIV2K_{split}_HR"
    lr_dir = DATA_DIR / f"DIV2K_{split}_LR_bicubic" / "X4"
    dataset = []
    for hr in sorted(hr_dir.glob("*.png")):
        lr = lr_dir / f"{hr.stem}x4.png"
        dataset.append((hr, lr))
    print(f"Loaded {len(dataset)} pairs")
    return dataset

def load_image(hr_path, lr_path):
    hr = Image.open(hr_path).convert("RGB")
    lr = Image.open(lr_path).convert("RGB")
    return hr, lr

def crop(hr_img, lr_img, size=40, scale=4):
    x = random.randint(0, lr_img.width - size)
    y = random.randint(0, lr_img.height - size)
    lr_patch = lr_img.crop((x, y, x + size, y + size))
    hr_patch = hr_img.crop((x * scale, y * scale, (x + size) * scale, (y + size) * scale))
    return hr_patch, lr_patch

def create_patches(dataset, patches_for_image=20):
    dataset_patch = []
    for hr_path, lr_path in dataset:
        hr, lr = load_image(hr_path, lr_path)
        for _ in range(patches_for_image):
            dataset_patch.append(crop(hr, lr))
    return dataset_patch

def load_dataset(partition="train"):
    dataset = load_pairs(partition)
    if partition == "train":
        dataset = create_patches(dataset)
    else:
        dataset = [load_image(hr, lr) for hr, lr in dataset]
    return SRDataset(dataset)

# train_dataset = get_pairs("train")
# valid_dataset = get_pairs("valid")
# train_dataset_patch = create_patches(train_dataset)
# print(len(train_dataset_patch))
# print(len(valid_dataset))

# for x, y in train_dataloader:
    # img = x[0].permute(dims=(1, 2, 0))
    # new_img = new_x[0].permute(dims=(1, 2, 0))
    # target_img = y[0].permute(dims=(1, 2, 0))


    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #
    # axes[0].imshow(img)
    # axes[0].set_title("Original")
    # axes[0].axis("off")
    # axes[1].imshow(new_img)
    # axes[1].set_title("Bicubic Upscaled")
    # axes[1].axis("off")
    # axes[2].imshow(target_img)
    # axes[2].set_title("Target")
    # axes[2].axis("off")
    #
    # plt.tight_layout()
    # plt.show()
    # break
