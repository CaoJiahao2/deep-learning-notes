# 单目深度估计与 3D 视觉基础模型

> Depth Anything v2、Metric3D v2 等基础模型把单目深度估计做到了"零样本可用"。

---

## 目录

1. [任务定义](#1)
2. [评价指标](#2)
3. [从 CNN 到 Transformer](#3)
4. [Depth Anything 系列](#4)
5. [Metric3D v2](#5)
6. [关键技术](#6)
7. [数据集](#7)
8. [应用](#8)
9. [评测与基准](#9)
10. [参考文献](#10)

---

## 1 任务定义 {#1}

**单目深度估计（Monocular Depth Estimation, MDE）** 从单张 RGB 图像预测每个像素的深度。是计算机视觉最基础的问题之一。

### 1.1 Relative Depth vs Metric Depth

| 类型 | 输出 | 单位 | 适用 |
|------|------|------|------|
| Relative Depth（相对深度） | 远近排序 | 无单位 | 3D 重建、相对位姿 |
| Metric Depth（度量深度） | 真实距离 | 米 / 毫米 | 自动驾驶、AR、机器人 |

### 1.2 室内 vs 室外 vs 跨域

- **室内**：NYU、ScanNet，深度范围 0.5–10 m，常见反光、弱纹理；
- **室外**：KITTI、Cityscapes，深度范围 0–200 m，运动模糊；
- **跨域**：从街景到动物、从游戏到医疗。

### 1.3 仿射不变性

单目深度的"尺度-平移模糊"：给定 $(d, a)$，实际深度 $D = a \cdot d + b$，$a, b$ 未知。需要相机内参或真实度量才能恢复。

---

## 2 评价指标 {#2}

设预测深度为 $\hat{d}_i$，真实为 $d_i$，$N$ 为像素数：

- **AbsRel**：$\frac{1}{N} \sum_i \frac{|\hat{d}_i - d_i|}{d_i}$
- **RMSE**：$\sqrt{\frac{1}{N} \sum_i (\hat{d}_i - d_i)^2}$
- **δ₁ / δ₂ / δ₃**：满足 $\max(\hat{d}_i/d_i, d_i/\hat{d}_i) < 1.25$ 等阈值的像素比例；
- **SiLog**：尺度不变对数误差，对尺度-平移模糊鲁棒：

$$\text{SiLog} = \frac{1}{N} \sum_i \left( \log \hat{d}_i - \log d_i \right)^2 - \frac{1}{N^2} \left( \sum_i (\log \hat{d}_i - \log d_i) \right)^2$$

---

## 3 从 CNN 到 Transformer {#3}

### 3.1 早期 CNN

- Eigen et al. 2014：首个端到端深度 CNN；
- Laina et al. 2016：基于 ResNet 的全卷积网络。

### 3.2 DPT（Vision Transformer for Dense Prediction）

**DPT**（Ranftl et al., 2021）把 ViT 用作编码器，渐进式上采样恢复空间分辨率：

```text
ViT blocks → reassemble → fusion → progressive refinement → dense prediction
```

### 3.3 MiDaS

**MiDaS**（Ranftl et al., 2022）通过**混合数据集训练**实现零样本深度估计，是 Depth Anything 的前身。

### 3.4 ZoeDepth

**ZoeDepth**（Bhat et al., 2023）在 MiDaS 基础上加入"度量头"，把相对深度升级为度量深度。

---

## 4 Depth Anything 系列 {#4}

### 4.1 Depth Anything v1（2024）

**核心思路**：用大模型（DPT）当教师，在 6 千万张无标注图像上生成伪深度，作为学生（DINOv2 + DPT）的训练数据。

```text
无标注图像  ──▶ MiDaS 教师  ──▶ 伪深度标签
                                 │
   小标注集 ────────────────┐   │
                          ▼   ▼
                  学生模型（DINOv2 + DPT）
                          │
                          ▼
                   相对深度输出
```

### 4.2 Depth Anything v2（2025）

**主要改进**：

1. **教师换成更大模型**：用合成数据训练的更强教师；
2. **纯合成数据训练**学生：避免"伪标签噪声"；
3. **多尺度特征融合**：细粒度提升；
4. **三个规模**：Small (25M)、Base (105M)、Large (335M)，分别适配不同算力。

效果：在多个 zero-shot benchmark 上达到 SOTA，且对细粒度纹理恢复更好。

### 4.3 Metric Depth 变体

DAV2 还发布了 Metric Depth 版本：通过引入 camera intrinsics，把相对深度转为 metric depth（米）。

### 4.4 工程意义

- 一次训练，三个规模，覆盖边缘到云端；
- 与 Segment Anything 类似，是"基础模型"思路的胜利。

---

## 5 Metric3D v2 {#5}

### 5.1 动机

Metric3D 关注"零样本度量深度"，支持多种相机模型（针孔、广角、鱼眼），并输出 surface normal。

### 5.2 关键技术

- **法向约束**：联合训练 depth + surface normal，提升几何一致性；
- **相机模型适配**：把不同内参的相机图像归一到统一表示；
- **大规模合成 + 真实微调**：用 Hypersim / IRS 等合成数据预训练，再在真实数据上微调。

### 5.3 输出形式

```text
输入: RGB 图像 + 相机内参（可选）
输出: 
  - Metric Depth: H × W 真实距离图
  - Surface Normal: H × W × 3 法向图
  - 置信度图（可选）
```

### 5.4 应用

- 自动驾驶占据预测；
- 机器人抓取（高度图）；
- 3D 重建的初始化深度。

---

## 6 关键技术 {#6}

### 6.1 大规模伪标签

- DPT 教师 → 伪深度；
- 学生模型在大量伪标签上学"相对排序"；
- 微调时再用少量真实数据。

### 6.2 仿射不变损失

直接用 L1/L2 损失，模型会被"尺度"主导。仿射不变损失先做线性对齐：

$$\mathcal{L}_{\text{si}} = \frac{1}{N} \sum_i \left( \hat{d}_i - \alpha d_i - \beta \right)^2$$

其中 $\alpha, \beta$ 在每个 mini-batch 内最小二乘拟合。

### 6.3 多数据集混合训练

不同数据集标注协议不同（有的提供 metric depth，有的不提供），需要做"协议归一化"：

- 统一为 metric depth；
- 用 camera intrinsics 把所有相机模型归一；
- 加权混合采样。

### 6.4 跨域泛化

- **数据增广**：颜色抖动、随机裁剪、随机缩放；
- **多分辨率训练**：从 384p 到 1536p；
- **大规模预训练**：在 ImageNet-22k / SA-1B 等基础模型权重上初始化。

---

## 7 数据集 {#7}

### 7.1 真实数据集

| 数据集 | 规模 | 类型 |
|--------|------|------|
| NYU-Depth-v2 | 1449 对齐对 | 室内 RGB-D |
| ScanNet | 1500+ 场景 | 室内 RGB-D |
| KITTI | 22k 图像 | 自动驾驶 LiDAR |
| Cityscapes | 5k | 街景 |
| DDAD | 110+ 场景 | 自动驾驶 |
| ETH3D | 多种 | 室内外混合 |
| SA-1B | 11M 图像 | SAM 标注 |
| Sintel | 1623 帧 | 合成（电影风格） |

### 7.2 合成数据集

- **Hypersim**（Apple, 2021）：高保真室内合成，77K 图像；
- **IRS**（Large Image+Depth）：大规模合成数据集；
- **TartanAir**：复杂运动 + 多样化场景。

### 7.3 伪标签

Depth Anything v1 用 DPT 在 62M 图像上生成的伪标签，是大规模伪标签的范例。

---

## 8 应用 {#8}

### 8.1 3D 重建

- 单目深度 → 3D 点云（配合相机内参）；
- 与 NeRF / Gaussian Splatting 配合，初始化深度。

### 8.2 具身智能

- 高度图生成（导航 / 抓取）；
- 占据预测（避障）；
- 端到端 VLA 中的几何表征（DP3）。

### 8.3 自动驾驶

- 占据预测；
- 距离估计；
- 3D 检测与跟踪。

### 8.4 AR / VR

- 场景重建；
- 虚实遮挡；
- 平面检测。

### 8.5 图像编辑

- 深度条件生成（ControlNet Depth）；
- 重对焦、背景虚化；
- 3D-aware 图像变换。

---

## 9 评测与基准 {#9}

### 9.1 评测集

- **NYU-Depth-v2**：室内，标准 benchmark；
- **KITTI**：室外，自动驾驶；
- **ETH3D**：室内外，高精度激光；
- **ScanNet**：室内 RGB-D；
- **跨域泛化**：在 SA-1B、Sintel 等未见域测试。

### 9.2 在线评测

- **Depth Anything 在线 demo**：HuggingFace Spaces；
- **Metric3D v2**：GitHub + 模型权重。

### 9.3 评测陷阱

- 训练数据与测试集分布相似时，泛化被高估；
- 仿射对齐后的指标可能掩盖真实尺度误差；
- 不同数据集"metric depth"的物理含义略有差异（深度 vs 视差）。

---

## 10 参考文献 {#10}

[1] L. Yang et al., *Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data*, NeurIPS, 2024.

[2] L. Yang et al., *Depth Anything v2*, 2025.

[3] M. Hu et al., *Metric3D v2: A Versatile Monocular Geometric Foundation Model*, 2025. arXiv:2409.06409.

[4] R. Ranftl et al., *Vision Transformers for Dense Prediction*, ICCV, 2021. arXiv:2103.13413.

[5] R. Ranftl et al., *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer (MiDaS)*, TPAMI, 2022. arXiv:1907.01341.

[6] S. F. Bhat et al., *ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth*, 2023. arXiv:2302.12288.

[7] A. Eigen et al., *Depth Map Prediction from a Single Image using a Multi-Scale Deep Network*, NeurIPS, 2014.

[8] I. Laina et al., *Deeper Depth Prediction with Fully Convolutional Residual Networks*, 3DV, 2016.

[9] A. Geiger et al., *Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR, 2012.

[10] P. K. Nathan Silberman et al., *NYU Depth Dataset v2*, ECCV, 2012.

[11] B. Ummenhofer et al., *Sintel: Photorealistic Video Generation with Depth*, 2012.

[12] M. Roberts et al., *Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding*, ICCV, 2021.