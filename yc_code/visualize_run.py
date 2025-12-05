# visualize_run.py
"""
功能：
- 给定一个路径 --path：
  1) 如果这个路径本身就是一个 run 目录（含 args.json + train_log.json），只处理这一条；
  2) 如果它是一个根目录（比如 runs/），就自动在下面寻找所有包含 train_log.json 的子目录，一一处理。

- 对每个 run 目录：
  - 读取 train_log.json，画：
      loss_full.png               # full 曲线（可用 full_min_step 去掉最开始几千 step）
      loss_late_fromXXXX.png      # 只看 step >= loss_min_step
      loss_late_smooth.png        # 后半段 loss 的 moving average 曲线（PPT 用这张）
      eps_corr_full.png           # eps_corr 曲线（full_min_step 过滤）
      eps_corr_smooth.png         # eps_corr 的 moving average 曲线
  - 从 samples/ 里挑最后几张 png 复制到 visualization/ 里
  - 根据 args.json 写一个 args_summary.txt 方便 PPT 填参数

输出目录结构：
run_dir/
  args.json
  train_log.json
  samples/...
  visualization/
    loss_full.png
    loss_late_from20000.png
    loss_late_smooth.png
    eps_corr_full.png
    eps_corr_smooth.png
    args_summary.txt
    step_*.png / forward_step_*.png (复制过来的样本图)

用法：
1）只处理一个 run：
    python visualize_run.py --path ..\\runs\\T22000_full_256_bs8_T1000_ep100 --full_min_step 2000 --loss_min_step 20000 --smooth_window 100

2）批量处理 runs 下所有实验：
    python visualize_run.py --path ..\\runs --full_min_step 2000 --loss_min_step 20000 --smooth_window 100
"""

import argparse
from pathlib import Path
import json
import shutil
import re

import matplotlib.pyplot as plt
import numpy as np


# -----------------------
# 参数解析
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        type=str,
        required=True,
        help="可以是单个 run 目录（含 args.json/train_log.json），也可以是包含多个 run 的根目录（比如 runs/）",
    )
    ap.add_argument(
        "--loss_min_step",
        type=int,
        default=20000,
        help="画后半段 loss 曲线的起始 step，比如 20000",
    )
    ap.add_argument(
        "--full_min_step",
        type=int,
        default=0,
        help="画 full loss / eps_corr 曲线时，忽略 step < full_min_step 的点，比如 1000 或 2000",
    )
    ap.add_argument(
        "--smooth_window",
        type=int,
        default=100,
        help="moving average 的窗口大小（以记录点数计，不是 step 大小），比如 50 或 100",
    )
    ap.add_argument(
        "--top_k_samples",
        type=int,
        default=6,
        help="从 samples/ 里拷贝多少张最新的 png 到 visualization/",
    )
    return ap.parse_args()


# -----------------------
# 小工具：moving average
# -----------------------
def moving_average(y: np.ndarray, window: int):
    """
    简单 moving average:
    - y: 1D 数组
    - window: 窗口宽度（>=1）
    返回：
    - y_smooth: 长度为 len(y) - window + 1
    """
    window = int(window)
    if window <= 1 or len(y) < window:
        return y, np.arange(len(y))

    kernel = np.ones(window, dtype=np.float32) / float(window)
    y_smooth = np.convolve(y, kernel, mode="valid")
    idx = np.arange(window - 1, len(y))  # 对应的索引
    return y_smooth, idx


# -----------------------
# 工具函数：读日志
# -----------------------
def load_train_log(run_dir: Path):
    log_path = run_dir / "train_log.json"
    if not log_path.is_file():
        raise FileNotFoundError(f"找不到 {log_path}，先确认 train.py 已经跑完并保存了 train_log.json。")

    with log_path.open("r") as f:
        history = json.load(f)

    steps = np.array(history.get("step", []), dtype=np.int64)
    loss = np.array(history.get("loss", []), dtype=np.float32)
    eps_corr = np.array(history.get("eps_corr", []), dtype=np.float32)

    if len(steps) == 0:
        raise ValueError(f"{log_path} 里面没有 step 记录，history 可能是空的。")

    return steps, loss, eps_corr


# -----------------------
# 工具函数：画曲线
# -----------------------
def plot_loss_curves(
    steps: np.ndarray,
    loss: np.ndarray,
    eps_corr: np.ndarray,
    out_dir: Path,
    loss_min_step: int,
    full_min_step: int,
    smooth_window: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------- full 曲线：按 full_min_step 过滤 ----------
    if full_min_step > 0:
        mask_full = steps >= full_min_step
        if mask_full.sum() == 0:
            mask_full = np.ones_like(steps, dtype=bool)
    else:
        mask_full = np.ones_like(steps, dtype=bool)

    steps_full = steps[mask_full]
    loss_full = loss[mask_full]
    eps_full = eps_corr[mask_full]

    # 1) full loss 曲线（已经避开最开始的 step）
    fig, ax1 = plt.subplots()
    ax1.plot(steps_full, loss_full, linewidth=1.0)
    ax1.set_xlabel("global step")
    ax1.set_ylabel("train diffusion loss (MSE)")
    if full_min_step > 0:
        ax1.set_title(f"Training diffusion loss (steps >= {full_min_step})")
    else:
        ax1.set_title("Training diffusion loss (full range)")

    if len(loss_full) > 10:
        hi = float(np.quantile(loss_full, 0.99))
        if hi > 0:
            ax1.set_ylim(0, hi * 1.1)

    fig.tight_layout()
    fig.savefig(out_dir / "loss_full.png", dpi=200)
    plt.close(fig)

    # 2) 后半段 loss（用 loss_min_step）
    mask_late = steps >= loss_min_step
    if mask_late.sum() > 10:
        steps_late = steps[mask_late]
        loss_late = loss[mask_late]

        # 原始后半段曲线
        fig, ax1 = plt.subplots()
        ax1.plot(steps_late, loss_late, linewidth=0.8)
        ax1.set_xlabel("global step")
        ax1.set_ylabel("train diffusion loss (MSE)")
        ax1.set_title(f"Training loss (steps >= {loss_min_step})")
        fig.tight_layout()
        fig.savefig(out_dir / f"loss_late_from{loss_min_step}.png", dpi=200)
        plt.close(fig)

        # moving average（平滑版，PPT 建议用这张）
        loss_smooth, idx = moving_average(loss_late, smooth_window)
        steps_smooth = steps_late[idx]

        fig, ax1 = plt.subplots()
        ax1.plot(steps_smooth, loss_smooth, linewidth=1.5)
        ax1.set_xlabel("global step")
        ax1.set_ylabel("smoothed diffusion loss (MSE)")
        ax1.set_title(
            f"Smoothed training loss (steps >= {loss_min_step}, window={smooth_window})"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"loss_late_smooth.png", dpi=200)
        plt.close(fig)

    # 3) eps_corr 曲线（full_min_step 过滤）
    fig, ax1 = plt.subplots()
    ax1.plot(steps_full, eps_full, linewidth=1.0)
    ax1.set_xlabel("global step")
    ax1.set_ylabel("cosine similarity(eps_pred, eps_true)")
    if full_min_step > 0:
        ax1.set_title(f"Eps prediction correlation (steps >= {full_min_step})")
    else:
        ax1.set_title("Eps prediction correlation (full range)")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "eps_corr_full.png", dpi=200)
    plt.close(fig)

    # 4) eps_corr 的 moving average（同样在 full 区间内）
    if len(eps_full) > smooth_window:
        eps_smooth, idx_eps = moving_average(eps_full, smooth_window)
        steps_eps = steps_full[idx_eps]

        fig, ax1 = plt.subplots()
        ax1.plot(steps_eps, eps_smooth, linewidth=1.5)
        ax1.set_xlabel("global step")
        ax1.set_ylabel("smoothed cosine similarity")
        ax1.set_title(
            f"Smoothed eps_corr (steps >= {full_min_step}, window={smooth_window})"
        )
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "eps_corr_smooth.png", dpi=200)
        plt.close(fig)


# -----------------------
# 工具函数：拷贝样本图
# -----------------------
def copy_sample_images(run_dir: Path, out_dir: Path, top_k: int = 4):
    samples_dir = run_dir / "samples"
    if not samples_dir.is_dir():
        print(f"[warn] {samples_dir} 不存在，跳过样本图复制。")
        return

    step_re = re.compile(r".*?(\d+)\.png$")

    png_files = sorted(samples_dir.glob("*.png"))
    if not png_files:
        print(f"[warn] {samples_dir} 里没有 png 文件。")
        return

    files_with_step = []
    for p in png_files:
        m = step_re.match(p.name)
        if m:
            step_val = int(m.group(1))
        else:
            step_val = -1
        files_with_step.append((step_val, p))

    files_with_step.sort(key=lambda x: x[0])
    to_copy = [p for _, p in files_with_step[-top_k:]]

    for src in to_copy:
        dst = out_dir / src.name
        shutil.copy2(src, dst)
        print(f"[sample] copied {src} -> {dst}")


# -----------------------
# 工具函数：导出 args 摘要
# -----------------------
def dump_args_summary(run_dir: Path, out_dir: Path):
    args_json = run_dir / "args.json"
    if not args_json.is_file():
        print(f"[warn] 找不到 {args_json}，跳过参数摘要。")
        return

    with args_json.open("r") as f:
        args = json.load(f)

    lines = []
    lines.append("=== Diffusion training config (from args.json) ===\n")
    lines.append(f"data_root   : {args.get('data_root')}")
    lines.append(f"image_size  : {args.get('image_size')}")
    lines.append(f"batch_size  : {args.get('batch_size')}")
    lines.append("")
    lines.append(f"timesteps   : {args.get('timesteps')}")
    lines.append(f"schedule    : {args.get('schedule')}")
    lines.append(f"beta_start  : {args.get('beta_start')}")
    lines.append(f"beta_end    : {args.get('beta_end')}")
    lines.append("")
    lines.append(f"lr          : {args.get('lr')}")
    lines.append(f"weight_decay: {args.get('weight_decay')}")
    lines.append(f"grad_clip   : {args.get('grad_clip')}")
    lines.append(f"amp         : {args.get('amp')}")
    lines.append(f"ema_decay   : {args.get('ema_decay')}")
    lines.append("")
    lines.append(f"in_ch       : {args.get('in_ch')}")
    lines.append(f"base        : {args.get('base')}")
    lines.append(f"mult        : {args.get('mult')}")
    lines.append(f"t_dim       : {args.get('t_dim')}")
    lines.append("")
    lines.append(f"epochs      : {args.get('epochs')}")
    lines.append(f"sample_every: {args.get('sample_every')}")
    lines.append(f"save_every  : {args.get('save_every')}")
    lines.append("")

    out_path = out_dir / "args_summary.txt"
    with out_path.open("w") as f:
        f.write("\n".join(lines))

    print(f"[args] wrote summary to {out_path}")


# -----------------------
# 单个 run 的处理逻辑
# -----------------------
def process_single_run(
    run_dir: Path,
    loss_min_step: int,
    full_min_step: int,
    smooth_window: int,
    top_k_samples: int,
):
    print(f"\n==============================")
    print(f"[run ] {run_dir}")
    print(f"==============================")

    vis_dir = run_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        steps, loss, eps_corr = load_train_log(run_dir)
    except Exception as e:
        print(f"[error] 读取 train_log 失败：{e}")
        return

    print(f"[info] 画 loss / eps_corr 曲线到 {vis_dir}")
    plot_loss_curves(
        steps,
        loss,
        eps_corr,
        vis_dir,
        loss_min_step=loss_min_step,
        full_min_step=full_min_step,
        smooth_window=smooth_window,
    )

    print(f"[info] 复制样本图到 {vis_dir}")
    copy_sample_images(run_dir, vis_dir, top_k=top_k_samples)

    print(f"[info] 导出 args 摘要到 {vis_dir}")
    dump_args_summary(run_dir, vis_dir)

    print(f"[done] run {run_dir.name} 可视化完成。输出目录：{vis_dir}")


# -----------------------
# 主函数
# -----------------------
def main():
    args = parse_args()
    root = Path(args.path).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"--path 不是一个有效目录: {root}")

    this_is_run = (root / "train_log.json").is_file()

    if this_is_run:
        process_single_run(
            root,
            loss_min_step=args.loss_min_step,
            full_min_step=args.full_min_step,
            smooth_window=args.smooth_window,
            top_k_samples=args.top_k_samples,
        )
    else:
        run_dirs = []
        for d in root.iterdir():
            if not d.is_dir():
                continue
            if (d / "train_log.json").is_file():
                run_dirs.append(d)

        if not run_dirs:
            raise RuntimeError(f"{root} 下没有找到包含 train_log.json 的 run 目录。")

        print(f"[info] 在 {root} 下找到 {len(run_dirs)} 个 run：")
        for d in run_dirs:
            print(f"  - {d.name}")

        for d in run_dirs:
            process_single_run(
                d,
                loss_min_step=args.loss_min_step,
                full_min_step=args.full_min_step,
                smooth_window=args.smooth_window,
                top_k_samples=args.top_k_samples,
            )


if __name__ == "__main__":
    main()
