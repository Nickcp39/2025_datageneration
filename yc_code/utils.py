# yc_code/utils/asserts.py
# Hard guards for training/sampling consistency.
# 用法（训练开始前）：
# from yc_code.utils.asserts import assert_consistency
# assert_consistency(
#     engine=engine,
#     timesteps=args.timesteps,
#     prediction_type="epsilon",
#     beta_schedule="cosine",        # 或 "ddpm"/"linear" 等，与你的引擎一致
#     image_range="[-1,1]",          # 训练脚本里把 batch 做了 x = x*2-1
#     sample_batch=None              # 可选：传一个 [B,C,H,W] 的小 batch 做数值校验
# )

from __future__ import annotations
from typing import Optional, Tuple, Any, Iterable, Union
import torch
import torch.nn as nn


def _norm_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    return str(x).strip().lower().replace(" ", "").replace("_", "-")


def _range_tuple(name: str) -> Tuple[float, float]:
    name = _norm_str(name or "")
    if name in ("[-1,1]", "(-1,1)", "-1..1", "-1to1"):
        return (-1.0, 1.0)
    if name in ("[0,1]", "(0,1)", "0..1", "0to1"):
        return (0.0, 1.0)
    raise RuntimeError(f"[asserts] unknown image_range spec: {name!r}. Use '[-1,1]' or '[0,1]'.")


def _hasattr(obj: Any, names: Iterable[str]) -> Optional[str]:
    """Return the first existing attribute name or None."""
    for n in names:
        if hasattr(obj, n):
            return n
    return None


def _maybe_get(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    n = _hasattr(obj, names)
    return getattr(obj, n, default) if n else default


def _tensor_close_len(x: torch.Tensor, n: int) -> bool:
    try:
        return x.numel() == n or x.shape[0] == n
    except Exception:
        return False


def _check_image_range(sample: torch.Tensor, expect: Tuple[float, float], tol: float = 5e-3) -> None:
    if not isinstance(sample, torch.Tensor):
        raise RuntimeError("[asserts] sample_batch must be a torch.Tensor")
    if sample.ndim != 4:
        raise RuntimeError(f"[asserts] sample_batch must be [B,C,H,W], got shape={tuple(sample.shape)}")
    smin = float(sample.min())
    smax = float(sample.max())
    lo, hi = expect
    if smin < lo - tol or smax > hi + tol:
        raise RuntimeError(
            f"[asserts] image_range mismatch: expected in {lo}..{hi}, "
            f"but sample_batch min/max = {smin:.4f}/{smax:.4f}. "
            "确保 Dataset/训练前处理与设定一致（例如训练前将 [0,1] -> [-1,1]）。"
        )


def assert_consistency(
    *,
    engine: Any,
    timesteps: int,
    prediction_type: str,
    beta_schedule: str,
    image_range: str,
    sample_batch: Optional[torch.Tensor] = None,
) -> None:
    """
    硬护栏：训练/采样一致性检查。任何不一致直接抛错退出。
    - timesteps（T）
    - β schedule（名称或长度）
    - prediction_type（语义：必须为 'epsilon' 等）
    - image_range（输入范围约定）
    - 额外：net/engine 通道对齐、engine 不是 nn.Module、部分可选属性的交叉校验
    """
    # -------- 基础合法性 --------
    if not isinstance(timesteps, int) or timesteps <= 0:
        raise RuntimeError(f"[asserts] timesteps must be positive int, got {timesteps!r}")

    pred = _norm_str(prediction_type)
    if pred not in ("epsilon", "eps", "e", "noise"):
        # 若真要支持 x0/v，可在此扩展；当前项目约定 epsilon
        raise RuntimeError(f"[asserts] prediction_type must be 'epsilon', got {prediction_type!r}")

    sched = _norm_str(beta_schedule)
    if not sched:
        raise RuntimeError(f"[asserts] beta_schedule must be a non-empty string, got {beta_schedule!r}")

    lohi = _range_tuple(image_range)

    # -------- engine 基础属性 --------
    if isinstance(engine, nn.Module):
        raise RuntimeError("[asserts] engine MUST NOT be subclass of nn.Module. 它应为纯控制器对象（非 .to(device)).")

    # 读取 engine 侧可用的元信息（尽量兼容多实现）
    eng_T = _maybe_get(engine, ("timesteps", "T", "num_timesteps"))
    eng_sched_name = _norm_str(_maybe_get(engine, ("beta_schedule", "schedule", "scheduler_name")))
    eng_betas = _maybe_get(engine, ("betas", "beta", "beta_table"))
    eng_pred_type = _norm_str(_maybe_get(engine, ("prediction_type", "pred_type", "objective")))
    eng_channels = _maybe_get(engine, ("channels", "in_channels", "c"))

    net = _maybe_get(engine, ("net", "model"))
    net_in = _maybe_get(net, ("in_ch", "in_channels", "input_channels"))
    net_out = _maybe_get(net, ("out_ch", "out_channels", "output_channels"))

    # -------- T / β 一致性 --------
    if eng_T is not None and int(eng_T) != int(timesteps):
        raise RuntimeError(f"[asserts] timesteps mismatch: engine.T={eng_T} vs arg.T={timesteps}")

    if isinstance(eng_betas, torch.Tensor):
        if not _tensor_close_len(eng_betas, timesteps):
            raise RuntimeError(
                f"[asserts] betas length mismatch: len(engine.betas)={eng_betas.numel()} vs T={timesteps}"
            )

    # schedule 名称一致（若 engine 暴露该字段）
    if eng_sched_name is not None and eng_sched_name != sched:
        raise RuntimeError(
            f"[asserts] beta_schedule mismatch: engine='{eng_sched_name}' vs arg='{sched}'"
        )

    # -------- 预测语义（ε）一致性 --------
    if eng_pred_type is not None:
        # 常见写法：'eps'/'epsilon'/'noise'
        if eng_pred_type not in ("epsilon", "eps", "e", "noise"):
            raise RuntimeError(
                f"[asserts] prediction_type mismatch: engine='{eng_pred_type}' (expect 'epsilon')."
            )

    # -------- 通道一致性（in==out==engine.channels）--------
    if eng_channels is not None and net_in is not None and int(eng_channels) != int(net_in):
        raise RuntimeError(
            f"[asserts] channel mismatch: engine.channels={eng_channels} vs net.in_ch={net_in}"
        )
    if net_in is not None and net_out is not None and int(net_in) != int(net_out):
        raise RuntimeError(
            f"[asserts] UNet in/out mismatch: in_ch={net_in} vs out_ch={net_out}（应当相等以预测 ε）"
        )

    # -------- 图像输入范围 --------
    # 优先用 sample_batch 做数值校验；否则尝试读取 engine 侧约定
    if sample_batch is not None:
        _check_image_range(sample_batch, lohi, tol=5e-3)
    else:
        eng_img_range = _norm_str(_maybe_get(engine, ("image_range", "input_range", "range")))
        if eng_img_range is not None:
            lohi_eng = _range_tuple(eng_img_range)
            if abs(lohi_eng[0] - lohi[0]) > 1e-6 or abs(lohi_eng[1] - lohi[1]) > 1e-6:
                raise RuntimeError(
                    f"[asserts] image_range mismatch: engine={lohi_eng} vs arg={lohi}"
                )

    # -------- 通过 --------
    # 返回 None 即表示成功；如需调试信息，可在此打印一行 OK 摘要
    # print(f"[asserts] OK: T={timesteps}, schedule={sched}, pred=epsilon, range={lohi}, channels={eng_channels}")
