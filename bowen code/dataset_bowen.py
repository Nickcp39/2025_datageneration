# dataset.py
import os
import glob
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class PAImageDataset(Dataset):
    """
    简单的 PA 图像 Dataset
    目录结构假设为：
        root/train/*.png
        root/val/*.png
    图像统一 resize 到 image_size，并归一化到 [-1, 1]
    """

    def __init__(self,
                 root: str,
                 split: str = "train",
                 image_size: int = 256):
        super().__init__()
        assert split in ["train", "val"]
        self.root = root
        self.split = split
        self.image_size = image_size

        pattern = os.path.join(root, split, "*.png")
        self.files: List[str] = sorted(glob.glob(pattern))
        if len(self.files) == 0:
            raise RuntimeError(f"No png images found in {pattern}")

        print(f"[PAImageDataset] split={split}, num_images={len(self.files)}")

    def __len__(self) -> int:
        return len(self.files)

    def _load_img(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L")  # 灰度
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)  # [H, W], 0~255
        return arr

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        img = self._load_img(path)

        # 0~255  ->  [-1,1]
        img = img / 127.5 - 1.0
        # [H,W] -> [1,H,W]
        img = np.expand_dims(img, axis=0)

        x = torch.from_numpy(img)  # float32
        return x


def build_dataloaders(data_root: str,
                      image_size: int = 256,
                      batch_size: int = 8,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:

    train_set = PAImageDataset(data_root, split="train", image_size=image_size)
    val_set = PAImageDataset(data_root, split="val", image_size=image_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
