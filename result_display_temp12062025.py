# make_grid_preview_flexible.py
#
# Usage:
#   python make_grid_preview_flexible.py
#
# It reads PNGs, selects GRID_M * GRID_N patches, and saves a GRID_M x GRID_N grid image.

import random
from pathlib import Path

from PIL import Image

# ---- config ----
DATA_DIR = Path("data/bowen_patch_256_12052025/train")

# ** 您只需要修改这两个值来自定义网格布局：**
GRID_M = 4               # 网格的行数 (例如：4)
GRID_N = 4               # 网格的列数 (例如：1)
# ----------------------------------------------------

N_SAMPLES = GRID_M * GRID_N  # 总样本数 = 行数 * 列数
OUTPUT_NAME = f"train_grid_{GRID_M}x{GRID_N}.png" # 输出文件名自动更新
PATCH_SIZE = 256          # 补丁的预期尺寸（例如 256x256）

def main():
    png_files = sorted(DATA_DIR.glob("*.png"))
    if len(png_files) < N_SAMPLES:
        raise ValueError(f"Not enough PNGs in {DATA_DIR}. Found {len(png_files)}, but need {N_SAMPLES} for a {GRID_M}x{GRID_N} grid.")

    # for reproducibility you can fix the seed
    random.seed(42)
    selected = random.sample(png_files, N_SAMPLES)

    # load first image to get size
    first = Image.open(selected[0]).convert("L")
    w, h = first.size

    # optionally force to PATCH_SIZE
    if PATCH_SIZE is not None:
        w = h = PATCH_SIZE
        first = first.resize((w, h))

    # create canvas: 宽度是 GRID_N * w，高度是 GRID_M * h
    grid_img = Image.new("L", (GRID_N * w, GRID_M * h))

    # load all images
    imgs = [first] + [
        Image.open(p).convert("L").resize((w, h)) for p in selected[1:]
    ]

    for idx, img in enumerate(imgs):
        # 核心逻辑：计算任意 M x N 网格的坐标
        row = idx // GRID_N  # 确定行号
        col = idx % GRID_N   # 确定列号
        
        x0 = col * w
        y0 = row * h
        grid_img.paste(img, (x0, y0))

    out_path = DATA_DIR.parent / OUTPUT_NAME
    grid_img.save(out_path)
    print(f"Saved grid image to: {out_path}")

if __name__ == "__main__":
    main()