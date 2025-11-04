import os, glob, random
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T


def _center_square_tensor(x: torch.Tensor) -> torch.Tensor:
    """x: [C,H,W] -> 中心正方裁剪"""
    _, h, w = x.shape
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    return x[:, top:top + s, left:left + s]


def _array_to_float_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    支持 8/16-bit，输出 [C,H,W]、float32、范围[0,1]。
    不依赖 PIL 的 mode 转换，避免 16-bit 被量化到 8-bit。
    """
    if arr.ndim == 2:  # H,W -> H,W,1
        arr = arr[..., None]

    if arr.dtype == np.uint16:
        arr = arr.astype("float32") / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype("float32") / 255.0
    else:
        arr = arr.astype("float32")

    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C,H,W]
    # 若是 RGBA / LA，丢弃 alpha
    if x.shape[0] == 4:
        x = x[:3]
    elif x.shape[0] == 2:
        x = x[:1]
    return x


def _rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    """RGB -> 灰度（luma），x: [3,H,W] in [0,1]"""
    r, g, b = x[0:1], x[1:2], x[2:3]
    # ITU-R BT.601
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


def _save_debug_image(x: torch.Tensor, out_png: str):
    """
    保存张量到 PNG：
    - 若在 [-1,1] 则先映射到 [0,1]
    - 仅用于可视化，不影响训练
    """
    with torch.no_grad():
        y = x.detach().cpu()
        if y.min() < -0.01 or y.max() > 1.01:
            y = (y + 1.0) / 2.0
        y = y.clamp(0, 1)
        if y.shape[0] == 1:
            pil = TF.to_pil_image(y[0])
        elif y.shape[0] == 3:
            pil = TF.to_pil_image(y)
        else:
            raise ValueError("debug saver only supports C=1 or 3")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        pil.save(out_png)


class GrayImageFolder(Dataset):
    def __init__(
        self,
        root: str,
        img_size: int = 256,
        channels: int = 3,                 # 1=灰度, 3=RGB
        center_crop: bool = False,
        aug: bool = False,                 # 默认关闭增强，先稳
        normalize: str = "tanh",           # 推荐 "tanh" -> [-1,1]，与扩散引擎对齐
        extensions: Tuple[str, ...] = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"),
        save_debug: bool = False,          # 新增：是否保存首样本 debug 图
        debug_outdir: Optional[str] = None # 新增：debug 输出目录
    ):
        super().__init__()
        assert channels in (1, 3)
        assert normalize in ("none", "tanh")
        self.root, self.img_size = root, int(img_size)
        self.channels, self.center_crop, self.aug = int(channels), bool(center_crop), bool(aug)
        self.normalize = normalize
        self.save_debug = bool(save_debug)
        self.debug_outdir = debug_outdir
        self._saved_debug = False

        files: List[str] = []
        for ext in extensions:
            files += glob.glob(os.path.join(root, "**", ext), recursive=True)
        self.files = sorted(files)
        if not self.files:
            raise FileNotFoundError(f"No images found under: {root}")

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]

        # **关键**：直接转 numpy，保留位深（避免 PIL mode 转换丢 16-bit）
        with Image.open(path) as im:
            arr = np.array(im)  # HxW(,C)  uint8/uint16/float
        x = _array_to_float_tensor(arr)  # [C,H,W] in [0,1], 保留16-bit动态范围

        # 通道整理（只在张量域做，避免 PIL 降位）
        c = x.shape[0]
        if self.channels == 1:
            if c == 3:
                x = _rgb_to_luma(x)             # RGB -> 1ch 灰度
            elif c == 1:
                pass
            else:
                # 罕见情况（例如单通道以外的多通道科学图）
                x = x[:1, ...]
        else:  # self.channels == 3
            if c == 1:
                x = x.repeat(3, 1, 1)           # 灰度复制到3通道
            elif c == 3:
                pass
            else:
                x = x[:3, ...]

        # 中心裁剪（可选）
        if self.center_crop:
            x = _center_square_tensor(x)

        # Resize 到 (S,S) —— 直接对 tensor 做几何变换，避免 PIL 模式问题
        x = TF.resize(x, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BILINEAR, antialias=True)

        # 轻量增强（仅水平翻转；默认关闭）
        if self.aug and random.random() < 0.5:
            x = torch.flip(x, dims=[2])

        # 归一化
        if self.normalize == "tanh":
            x = x * 2.0 - 1.0   # [-1,1]
        # else: "none" 保持 [0,1]

        # 首样本保存 debug 图
        if self.save_debug and (not self._saved_debug):
            outdir = self.debug_outdir or os.path.join(os.getcwd(), "runs_local_gray_3060")
            out_png = os.path.join(outdir, "data_debug.png")
            _save_debug_image(x, out_png)
            self._saved_debug = True

        return x
    
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--normalize", type=str, default="tanh", choices=["none", "tanh"])
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--debug_outdir", type=str, default="./runs_local_gray_3060")
    args = parser.parse_args()

    ds = GrayImageFolder(
        root=args.root,
        img_size=args.img_size,
        channels=args.channels,
        center_crop=args.center_crop,
        aug=args.aug,
        normalize=args.normalize,
        save_debug=args.save_debug,
        debug_outdir=args.debug_outdir,
    )
    print(f"[data] found {len(ds)} images under {args.root}")
    dl = DataLoader(ds, batch_size=4, num_workers=0, shuffle=True)
    batch = next(iter(dl))
    print("[data] batch shape:", batch.shape, "| min/max:", float(batch.min()), float(batch.max()))
