# 深度学习环境配置指南

> 从裸机到可训练的完整 GPU 环境搭建流程。

---

## 目录

1. [硬件选择](#1-硬件选择)
2. [操作系统安装](#2-操作系统安装)
3. [GPU 驱动与 CUDA 安装](#3-gpu-驱动与-cuda-安装)
4. [Python 与 PyTorch 环境](#4-python-与-pytorch-环境)
5. [开发环境配置](#5-开发环境配置)
6. [Docker 环境](#6-docker-环境)
7. [环境验证](#7-环境验证)

---

## 1. 硬件选择

### 1.1 GPU 推荐

| GPU | 显存 | 适合场景 |
|-----|------|---------|
| RTX 4090 24GB | 24 GB | 个人开发、小模型训练 |
| RTX 6000 Ada | 48 GB | 中等模型训练、微调 |
| A100 80GB | 80 GB | 大模型训练 |
| H100 80GB | 80 GB | 大模型训练（FP8 加速） |
| RTX 4060 8GB | 8 GB | 入门学习、推理 |

### 1.2 其他硬件

| 组件 | 建议 |
|------|------|
| CPU | 核数 = GPU 数 × 4-8 |
| 内存 | GPU 显存 × 2-3 |
| 存储 | NVMe SSD 1TB+ |
| 电源 | 按 GPU 功耗 + 余量计算 |

---

## 2. 操作系统安装

推荐 **Ubuntu 22.04 LTS**（深度学习生态最佳支持）。

### 安装后基础配置

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential cmake git curl wget vim htop

# 安装 SSH Server
sudo apt install -y openssh-server
```

---

## 3. GPU 驱动与 CUDA 安装

### 3.1 安装 NVIDIA 驱动

```bash
# 查看推荐驱动版本
ubuntu-drivers devices

# 自动安装推荐驱动
sudo ubuntu-drivers autoinstall

# 重启
sudo reboot

# 验证
nvidia-smi
```

### 3.2 安装 CUDA Toolkit

推荐使用 conda 安装 CUDA（避免系统级冲突）：

```bash
conda install -c nvidia cuda-toolkit
```

或从 NVIDIA 官网下载 runfile 安装：

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

### 3.3 安装 cuDNN

推荐 conda 安装：

```bash
conda install -c nvidia cudnn
```

### 3.4 验证 CUDA 环境

```bash
nvcc --version
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. Python 与 PyTorch 环境

### 4.1 使用 Conda 管理环境

```bash
# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建独立环境
conda create -n dl python=3.10
conda activate dl
```

### 4.2 安装 PyTorch

```bash
# PyTorch 2.x (推荐)
pip install torch torchvision torchaudio

# 特定 CUDA 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4.3 常用深度学习库

```bash
pip install transformers datasets accelerate peft
pip install numpy pandas scikit-learn matplotlib seaborn
pip install jupyter jupyterlab ipywidgets
pip install wandb tensorboard
pip install opencv-python pillow
pip install triton # Flash Attention 2
```

### 4.4 配置示例 (requirements.txt)

```text
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.25.0
peft>=0.7.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
wandb>=0.15.0
tensorboard>=2.13.0
```

---

## 5. 开发环境配置

### 5.1 VS Code

推荐安装以下扩展：
- Python
- Pylance
- Jupyter
- GitHub Copilot
- Remote - SSH
- Markdown All in One

### 5.2 Jupyter Lab

```bash
# 安装
pip install jupyterlab

# 启动
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

### 5.3 远程开发

```bash
# SSH 免密登录
ssh-copy-id user@server

# SSH 配置文件 (~/.ssh/config)
Host dl-server
    HostName 192.168.1.100
    User username
    Port 22
    IdentityFile ~/.ssh/id_rsa
```

### 5.4 Tmux 终端复用

```bash
tmux new -s training    # 创建会话
tmux attach -t training # 重新连接
Ctrl+B D                # 分离会话
```

---

## 6. Docker 环境

### 6.1 安装 Docker + NVIDIA Container Toolkit

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 6.2 使用 PyTorch Docker 镜像

```bash
# 拉取镜像
docker pull pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 启动容器
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
```

---

## 7. 环境验证

### 7.1 GPU 可用性

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 显存容量
total_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f"GPU memory: {total_mem:.1f} GB")
```

### 7.2 性能测试

```python
import torch
import time

# 矩阵乘法基准测试
x = torch.randn(10000, 10000, device='cuda')
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y = x @ x.T
torch.cuda.synchronize()
print(f"Time: {time.time() - start:.2f}s")
```

### 7.3 常见问题排查

| 问题 | 解决方案 |
|------|---------|
| `nvidia-smi` 无法识别 GPU | 重新安装驱动，检查 Secure Boot |
| `torch.cuda.is_available()` 返回 False | 检查 PyTorch 版本与 CUDA 版本匹配 |
| cuDNN 错误 | 确保 cuDNN 版本与 CUDA 版本兼容 |
| 显存不足 (OOM) | 减小 batch size，使用梯度累积，启用混合精度 |
