"""
dataset_gray.py
最小而实用的图像数据集加载器（修正：Resize 强制输出正方形）
- 递归读取 data_root 下的所有图片
- center_crop=True 时先把长方图裁成中心正方形
- 统一 Resize 到 (img_size, img_size)  —— 关键修正！
- 通道：灰度 1ch（默认）或伪 RGB 3ch
- 轻量增强：水平翻转、微小旋转（可关闭）
- normalize: "none" -> [0,1]；"tanh" -> [-1,1]
"""
import os
import glob
import random
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class GrayImageFolder(Dataset):
    def __init__(
        self,
        root: str,
        img_size: int = 256,
        channels: int = 1,                 # 1=灰度, 3=伪RGB
        center_crop: bool = True,          # 是否中心裁成正方形
        aug: bool = True,                  # 是否进行轻量数据增强
        normalize: str = "none",           # "none" 或 "tanh"（[-1,1]）
        extensions: Tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"),
    ):
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.channels = channels
        self.center_crop = center_crop
        self.aug = aug
        assert normalize in ("none", "tanh"), "normalize must be 'none' or 'tanh'"
        self.normalize = normalize

        # 收集所有图片路径
        files: List[str] = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
        self.files = sorted(files)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found under: {root}")

        # 基础几何变换：中心裁剪（可选）+ Resize(强制正方形)
        geo: List[T.transforms] = []
        if self.center_crop:
            geo.append(T.Lambda(center_square))
        geo.append(T.Resize((self.img_size, self.img_size),  # <-- 修正点：强制 (H,W) = (S,S)
                            interpolation=T.InterpolationMode.BICUBIC,
                            antialias=True))

        # 通道与张量
        to_tensor = [T.ToTensor()]  # [0,1]

        # 合并
        self.base_transform = T.Compose(geo + to_tensor)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]

        # 读图并按通道处理
        if self.channels == 1:
            img = Image.open(path).convert("L")   # 灰度
        elif self.channels == 3:
            img = Image.open(path).convert("RGB") # 伪RGB
        else:
            raise ValueError("channels must be 1 or 3")

        x = self.base_transform(img)  # [C,H,W], range=[0,1]

        # 轻量增强（尽量不破坏结构）
        if self.aug:
            # 水平翻转 50%
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            # 微小旋转 10% 概率（-5°~5°），保持尺寸不变
            if random.random() < 0.1:
                angle = random.uniform(-5, 5)
                x = TF.rotate(x, angle,
                              interpolation=T.InterpolationMode.BILINEAR,
                              expand=False)

        # 归一化
        if self.normalize == "tanh":
            x = x * 2.0 - 1.0  # [-1,1]

        # 保底：确保通道数匹配
        if self.channels == 1 and x.size(0) != 1:
            x = x.mean(0, keepdim=True)  # [1,H,W]
        if self.channels == 3 and x.size(0) != 3:
            x = x.expand(3, *x.shape[1:])  # 罕见情况兜底

        return x


def center_square(img: Image.Image) -> Image.Image:
    """把长方图裁成中心正方形。"""
    w, h = img.size
    s = min(w, h)
    l = (w - s) // 2
    t = (h - s) // 2
    return img.crop((l, t, l + s, t + s))


# 便于快速 sanity check
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # 示例一：你的 1000×860 RGB 原始数据（转灰度，中心裁，再强制到 256×256）
    ds1 = GrayImageFolder(root="../data2025_raw1000x860",
                          img_size=256, channels=1,
                          center_crop=True, aug=True, normalize="none")
    # 示例二：你的 512×512 灰度数据（已方形，可关闭 center_crop）
    ds2 = GrayImageFolder(root="../data2025_crop512",
                          img_size=256, channels=1,
                          center_crop=False, aug=True, normalize="none")

    print("found:", len(ds1), len(ds2))
    dl1 = DataLoader(ds1, batch_size=4, shuffle=True)
    x1 = next(iter(dl1))
    print("ds1 batch:", x1.shape, x1.min().item(), x1.max().item())  # [4,1,256,256]

    dl2 = DataLoader(ds2, batch_size=4, shuffle=True)
    x2 = next(iter(dl2))
    print("ds2 batch:", x2.shape, x2.min().item(), x2.max().item())  # [4,1,256,256]
