# models/hf_unet2d.py
def _import_unet2dmodel():
    try:
        # 新版
        from diffusers import UNet2DModel  # noqa
        return UNet2DModel
    except Exception:
        # 旧版路径
        from diffusers.models.unets.unet_2d import UNet2DModel  # type: ignore
        return UNet2DModel

UNet2DModel = _import_unet2dmodel()

def build_unet2d(sample_size: int, in_ch: int, out_ch: int, base: int = 64):
    """
    创建一个稳定的 UNet2DModel：
    - 3 层下采样（总下采样因子 8），适配 128/256 的常见尺寸
    - 不使用注意力，避免不同版本 block 名称差异
    """
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_ch,
        out_channels=out_ch,
        layers_per_block=2,
        block_out_channels=(base, base * 2, base * 4),   # 64,128,256
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
