# 2025 Data Generation for Diffusion/GAN Training

This repository contains the **training and data generation pipeline** used for generating and augmenting grayscale images (~8,000 samples) for deep generative model research.

---

## 🧱 Project Structure

📁 2025_datageneration/
├── dataset_gray.py # Dataset loading & preprocessing (grayscale)
├── model.py # Model architecture (GAN/Diffusion)
├── train.sh # Training script (bash)
├── sample.sh # Sampling/inference script
├── setup.sh # Environment setup and dependencies
├── utils.py # Helper functions (logging, visualization, etc.)
├── requirements.txt # Python dependencies
└── .gitignore # Ignore data, checkpoints, logs

yaml
Copy code

> ⚠️ Note: The `data2025/` directory (8,000 images) is **not uploaded** to GitHub due to storage limits.  
> It is automatically ignored via `.gitignore`.

---

## 🚀 Quick Start

### 1️⃣ Environment Setup
```bash
conda create -n diffusion python=3.10
conda activate diffusion
bash setup.sh
pip install -r requirements.txt
2️⃣ Training
bash
Copy code
bash train.sh
3️⃣ Sampling
bash
Copy code
bash sample.sh
💡 Features
GPU-accelerated training (PyTorch)

Modular dataset and model structure

Automatic checkpoint saving and resume

.gitignore ensures privacy and clean commits

Designed for AWS EC2 / local workstation compatibility

🧑‍💻 Author
Yanda Cheng (PhD, University at Buffalo)
Biomedical AI, Photoacoustic Imaging, and Deep Generative Models
📍 Buffalo, NY | 🌐 LinkedIn | GitHub

yaml
Copy code

---

### 🧩 2. 添加并推送
```bash
git add README.md
git commit -m "Add project README"
git push