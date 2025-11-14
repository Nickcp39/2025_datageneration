# dataset.py
from pathlib import Path
from typing import List, Tuple, Optional
import random
import math

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# 支持的图片后缀
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXTS

def _list_images(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if _is_image_file(root) else []
    return sorted([p for p in root.rglob("*") if _is_image_file(p)])

def _to_grayscale_preserve_depth(img: Image.Image) -> np.ndarray:
    """
    转灰度但保留位深：
    - 对单通道整型（8/16-bit）按 dtype 的最大值做到 [0,1]
    - 对 RGB/RGBA 按亮度公式转灰度（float32），再归一到 [0,1]
    返回 float32 的 (H,W) 数组，范围 [0,1]
    """
    arr = np.array(img)
    if arr.ndim == 2:  # 单通道
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            denom = float(info.max)
            x = (arr.astype(np.float32) / max(denom, 1.0))
        else:
            x = arr.astype(np.float32)
            if x.max() > 1.0 or x.min() < 0.0:
                x = (x - x.min()) / max(x.max() - x.min(), 1e-8)
        return np.clip(x, 0.0, 1.0)

    elif arr.ndim == 3:  # 多通道
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]  # 去掉 alpha
        arr_f = arr.astype(np.float32)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            denom = float(info.max)
            arr_f = arr_f / max(denom, 1.0)
        else:
            if arr_f.max() > 1.0 or arr_f.min() < 0.0:
                arr_f = (arr_f - arr_f.min()) / max(arr_f.max() - arr_f.min(), 1e-8)
        r, g, b = arr_f[..., 0], arr_f[..., 1], arr_f[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(gray.astype(np.float32), 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image array shape: {arr.shape}")

def _resize_and_center_crop(x: np.ndarray, out_size: int) -> np.ndarray:
    """
    等比缩放（短边到 out_size）→ 中心裁剪成 out_size×out_size。
    输入 x 为 (H,W) 的 float32 [0,1]
    """
    h, w = x.shape
    if h == out_size and w == out_size:
        return x

    scale = out_size / min(h, w)
    nh = int(round(h * scale))
    nw = int(round(w * scale))

    # 用 PIL 做高质量缩放
    pil = Image.fromarray((x * 255.0).astype(np.uint8))
    pil = pil.resize((nw, nh), resample=Image.BILINEAR)
    x2 = np.array(pil).astype(np.float32) / 255.0

    # 中心裁剪
    top = max((nh - out_size) // 2, 0)
    left = max((nw - out_size) // 2, 0)
    x3 = x2[top: top + out_size, left: left + out_size]

    # 边缘对齐时可能需要补齐
    if x3.shape[0] != out_size or x3.shape[1] != out_size:
        pad_h = out_size - x3.shape[0]
        pad_w = out_size - x3.shape[1]
        x3 = np.pad(x3, ((0, pad_h), (0, pad_w)), mode='edge')
    return np.clip(x3, 0.0, 1.0)

def _light_augment(x: np.ndarray, max_translate: float = 0.02, hflip: bool = True) -> np.ndarray:
    """
    对齐数据的保守增强：
    - 可选水平翻转
    - ≤2% 的微小平移（edge 填充）
    """
    H, W = x.shape
    if hflip and random.random() < 0.5:
        x = np.ascontiguousarray(x[:, ::-1])

    max_dx = int(round(W * max_translate))
    max_dy = int(round(H * max_translate))
    if max_dx > 0 or max_dy > 0:
        tx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        ty = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
        pad_x = np.pad(x, ((abs(ty), abs(ty)), (abs(tx), abs(tx))), mode='edge')
        y0 = abs(ty) - ty
        x0 = abs(tx) - tx
        x = pad_x[y0:y0 + H, x0:x0 + W]
    return x

class GrayImageFolder(Dataset):
    """
    灰度数据集（稳健版）：
    - 接受 8/16-bit PNG/TIFF/JPEG/BMP/GIF
    - 全局线性归一化到 [0,1]，保持图间亮度差异
    - 输出张量 ∈ [-1,1]，shape=[1,H,W]
    """
    def __init__(
        self,
        root: str,
        image_size: int = 512,
        augment: bool = False,
        allow_flip: bool = True,
        max_translate: float = 0.02,
        filelist: Optional[List[str]] = None,
    ) -> None:
        self.root = Path(root)
        if filelist is not None:
            self.files = [Path(p) for p in filelist if _is_image_file(Path(p))]
        else:
            self.files = _list_images(self.root)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found under: {self.root} (extensions: {sorted(SUPPORTED_EXTS)})")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.allow_flip = bool(allow_flip)
        self.max_translate = float(max_translate)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.files[idx]
        try:
            with Image.open(p) as img:
                img.load()
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {p}") from e

        # 读→灰度→缩放裁剪→（可选）轻量增强
        x = _to_grayscale_preserve_depth(img)                # (H,W) in [0,1]
        x = _resize_and_center_crop(x, self.image_size)      # (H,W) in [0,1]
        if self.augment:
            x = _light_augment(x, max_translate=self.max_translate, hflip=self.allow_flip)

        # 转张量到 [-1,1]
        x = torch.from_numpy(x).unsqueeze(0).to(dtype=torch.float32)  # [1,H,W], [0,1]
        x = x * 2.0 - 1.0                                             # [-1,1]
        return x

def build_dataloaders(
    data_root: str,
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    val_ratio: float = 0.05,
    seed: int = 42,
    augment_train: bool = False,
    allow_flip: bool = True,
    max_translate: float = 0.02,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    随机划分 train/val 并构建 DataLoader。
    """
    root = Path(data_root)
    files = _list_images(root)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found under: {root}")

    n = len(files)
    n_val = int(round(n * val_ratio))
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    val_idxs = idxs[:n_val] if n_val > 0 else []
    train_idxs = idxs[n_val:]

    train_files = [str(files[i]) for i in train_idxs]
    val_files   = [str(files[i]) for i in val_idxs] if n_val > 0 else None

    train_ds = GrayImageFolder(
        root=str(root),
        image_size=image_size,
        augment=augment_train,
        allow_flip=allow_flip,
        max_translate=max_translate,
        filelist=train_files,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_dl = None
    if val_files is not None and len(val_files) > 0:
        val_ds = GrayImageFolder(
            root=str(root),
            image_size=image_size,
            augment=False,
            allow_flip=False,
            filelist=val_files,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return train_dl, val_dl

@torch.no_grad()
def save_debug_batch(dl: DataLoader, out_path: str, n: int = 16) -> None:
    """
    保存一个 batch 的可视化网格（不依赖 torchvision）。
    假设输入张量在 [-1,1]。
    """
    it = iter(dl)
    try:
        x = next(it)
    except StopIteration:
        return
    x = x[:n].detach().float().clamp(-1, 1)
    x = (x + 1.0) / 2.0  # [0,1]
    b, c, h, w = x.shape
    if c != 1:
        x = x.mean(dim=1, keepdim=True)

    # 自适应排列 rows/cols
    cols = int(math.sqrt(b))
    if cols * cols < b:
        cols += 1
    rows = (b + cols - 1) // cols

    canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)
    k = 0
    for r in range(rows):
        for cidx in range(cols):
            if k >= b: break
            tile = (x[k, 0].cpu().numpy() * 255.0).astype(np.uint8)
            canvas[r*h:(r+1)*h, cidx*w:(cidx+1)*w] = tile
            k += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)
