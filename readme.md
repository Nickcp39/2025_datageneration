2025 Data Generation for Diffusion/GAN Training

用于灰度图像（~8k）的生成式模型训练与数据增广的最小可复现管线（支持本地工作站 / AWS EC2）。
本仓库采用自研 Diffusion 引擎（Single Source of Truth），停用一切 HF 依赖分支，训练/采样语义统一为 ε-prediction。

🧱 项目结构
2025_datageneration/
├── yc_code/
│   ├── data/
│   │   └── gray_image_folder.py     # 数据集：输出 [0,1]；中心裁剪/resize/轻增强
│   ├── models/
│   │   └── unet_eps.py              # UNet ε 预测器（in_ch == out_ch）
│   ├── diffusion/
│   │   └── diffusion_engine.py      # 自研引擎（β/ᾱ 表 + DDPM 单步 + DDIM）
│   └── utils/
│       └── asserts.py               # 训练/采样一致性硬护栏
├── train_diffusion.py               # 训练脚本（EMA、周期采样、日志）
├── sample_diffusion.py              # 独立采样脚本（DDPM/DDIM 可切换）
├── requirements.txt                 # 依赖（PyTorch cu121）
├── setup.sh                         # 环境初始化（可选）
├── train.sh                         # 训练启动示例
├── sample.sh                        # 采样启动示例
└── .gitignore                       # 忽略 data、ckpt、logs


⚠️ data2025/（~8,000 图）不随仓库发布，已在 .gitignore 中忽略。

🚀 快速开始
1) 环境准备
conda create -n diffusion python=3.10 -y
conda activate diffusion

# 可选：你自己的 CUDA 驱动 >= 535
bash setup.sh            # 如有
pip install -r requirements.txt

2) 训练

数据集输出固定 [0,1]，训练脚本内部会统一映射到 [-1,1] 再送入引擎。

python train_diffusion.py \
  --data_root ./data2025 \
  --image_size 256 \
  --channels 1 \
  --batch_size 32 \
  --timesteps 1000 \
  --max_steps 12000 \
  --base 64 \
  --time_dim 256 \
  --mid_attn \
  --out_dir .runs/runs_vessel


要点：

默认停用任何 HF 引擎；只有自研 DiffusionEngine 生效；

训练前自动断言一致性（yc_code/utils/asserts.py），不一致即报错退出；

周期性用 EMA 权重做采样预览，采样后会恢复训练权重（不污染训练）。

3) 采样（独立脚本）

独立脚本只走自研引擎；--method 可切换 ddpm|ddim，--steps 为 DDIM 步数。保存前统一反归一化到 [0,1]。

python sample_diffusion.py \
  --ckpt .runs/runs_vessel/ckpts/ckpt_012000.pt \
  --out_dir ./samples_vessel \
  --num 64 \
  --nrow 8 \
  --method ddim \
  --steps 50 \
  --t_ratio 0.4

🧩 数据集规范

文件支持：png/jpg/jpeg/bmp/tif/tiff，8/16-bit 自动识别（不降位）

输出：[C,H,W]、float32、范围 [0,1]

通道对齐：channels ∈ {1,3}，与模型 in_ch/out_ch 保持一致

预处理：可选 center_crop + resize(S,S)（S % 4 == 0）

轻度增强：训练态可开启 水平翻转（默认关闭）

🧪 一致性硬护栏（必须过）

训练开始前会调用：

from yc_code.utils.asserts import assert_consistency
assert_consistency(
    engine=engine,
    timesteps=args.timesteps,
    prediction_type="epsilon",
    beta_schedule="cosine",     # 依你的引擎设置
    image_range="[-1,1]"        # 训练前已从 [0,1] → [-1,1]
)


若发现以下任一不一致将直接抛错：

T（time steps）不一致；

β 表长度/调度名不一致；

prediction_type 非 ε；

image_range 与数据/训练前处理不匹配；

in_ch != out_ch 或引擎/模型通道不一致；

误把引擎当 nn.Module 调 .to(device)。

⚙️ 关键训练开关

--mid_attn：是否启用 bottleneck 自注意力（默认关闭，带上该 flag 表示开启）

--preview_method（训练内部预览）：ddpm|ddim

采样起点：按比例 t_start = int(0.4 * T)（更稳的结构）

🗂️ .gitignore 建议
data2025/
.runs/
outputs/
*.pt
__pycache__/

📌 设计约束（简明）

输出语义唯一：UNet 预测 ε（无 tanh/sigmoid 头）；in_ch == out_ch == channels

引擎唯一：训练/采样都走自研引擎（禁用 HF 隐式 schedule）

数值路径唯一：数据集 [0,1] → 训练前统一 [-1,1]；采样保存前统一映射回 [0,1]

👤 Author

Yanda Cheng (PhD, University at Buffalo)
Biomedical AI · Photoacoustic Imaging · Deep Generative Models
Buffalo, NY · LinkedIn · GitHub

🔖 复用声明

仅供研究使用（research only）。如需用于商业或第三方发布，请先联系作者。

✅ 一键提交示例
git add -A
git commit -m "Add cleaned README and unified self-engine pipeline"
git push


需要我把 train.sh / sample.sh / setup.sh 也按上述参数模板给你生成吗？