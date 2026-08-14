# 3D 生成：NeRF 与 3D Gaussian Splatting

> 从神经辐射场的体渲染到高斯泼溅的实时表示，从文本到 3D 到动态 4D 场景。

---

## 目录

1. [概述](#1)
2. [3D 表示方法](#2-3d)
3. [NeRF：神经辐射场](#3-nerf)
4. [3D Gaussian Splatting](#4-3d-gaussian-splatting)
5. [文本到 3D 生成](#5-3d)
6. [前馈 3D 重建](#6-3d)
7. [4D 动态场景](#7-4d)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1. 概述

3D 生成与重建旨在从图像、文本或多视角输入中构建三维场景表示。它是计算机视觉的核心问题，也是 AR/VR、机器人仿真、数字孪生、游戏和电影制作的关键技术。

2020 年 NeRF 将神经隐式表示推向主流；2023 年 3D Gaussian Splatting 以实时渲染和高质量革新了显式表示；同时，扩散模型推动了文本到 3D 和前馈重建的快速发展。

核心技术路线：

$$
\underbrace{\text{多视图 3D 重建}}_{\text{NeRF / 3DGS}} \rightarrow \underbrace{\text{文本到 3D}}_{\text{Score Distillation}} \rightarrow \underbrace{\text{大规模前馈重建}}_{\text{原生 3D 扩散}}
$$

---

## 2. 3D 表示方法

### 2.1 表示分类

| 类型 | 方法 | 特点 |
|------|------|------|
| 显式体素 | Voxel Grid | 规则网格，分辨率受显存限制 |
| 显式点云 | Point Cloud | 灵活但缺乏表面连续性 |
| 显式网格 | Mesh | 标准图形学格式，拓扑固定 |
| 隐式神经表示 | NeRF (MLP) | 紧凑、连续、分辨率无限但渲染慢 |
| 混合表示 | Instant NGP, 3DGS | 兼顾质量、速度和可编辑性 |

### 2.2 渲染方程

3D 表示最终通过渲染方程生成图像。体渲染的核心是沿光线积分颜色和密度：

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt
$$

其中 $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$ 是透射率，$\sigma$ 是密度，$\mathbf{c}$ 是视角依赖的颜色。

---

## 3. NeRF：神经辐射场

### 3.1 基本原理

NeRF（Mildenhall et al., 2020）用 MLP 将 3D 位置 $\mathbf{x}=(x,y,z)$ 和视角方向 $\mathbf{d}=(\theta,\phi)$ 映射为密度 $\sigma$ 和颜色 $\mathbf{c}=(r,g,b)$：

$$
(\sigma, \mathbf{c}) = F_\Theta(\mathbf{x}, \mathbf{d})
$$

给定一组已知相机位姿的图像，NeRF 通过最小化渲染图像与真实图像的差异优化 MLP 参数：

$$
\mathcal{L} = \sum_{\mathbf{r}} \left\| C(\mathbf{r}) - C_{\text{gt}}(\mathbf{r}) \right\|^2
$$

### 3.2 位置编码

原始 MLP 难以表示高频几何细节。NeRF 使用位置编码将低维输入映射到高维正弦空间：

$$
\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))
$$

这使得 MLP 能表示高频的颜色和几何变化。位置编码的频率数 $L$ 控制可表示的细节级别。

### 3.3 分层采样

NeRF 采用粗-细两网络策略：

1. **粗网络（Coarse）**：在线性均匀采样的点上渲染，估计密度分布。
2. **细网络（Fine）**：根据粗网络的密度进行重要性采样，在表面附近密集采样。

这提高了渲染效率和质量。

### 3.4 NeRF 的加速演进

| 方法 | 核心改进 |
|------|---------|
| Instant NGP (Müller, 2022) | 多分辨率哈希编码，训练从数天降到数分钟 |
| TensoRF | 张量化体素网格，紧凑高效 |
| DVGO | 显式体素 + 后期密度平滑 |
| Plenoxels | 无 MLP，直接优化稀疏体素 |

Instant NGP 的可训练多分辨率哈希编码是关键突破：每个空间位置在 $L$ 个分辨率层级上查询哈希表中的可学习特征，插值后拼接送入小型 MLP，实现了速度与质量的兼顾。

---

## 4. 3D Gaussian Splatting

### 4.1 核心思想

3D Gaussian Splatting（Kerbl et al., 2023）用一组显式的 3D 高斯椭球体表示场景，每个高斯由以下参数定义：

- 位置 $\boldsymbol{\mu} \in \mathbb{R}^3$
- 协方差矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$（由缩放 $\mathbf{s}$ 和旋转 $\mathbf{q}$ 参数化）
- 不透明度 $\alpha \in [0,1]$
- 球谐系数（视角依赖颜色）$SH \in \mathbb{R}^k$

协方差矩阵分解为旋转和缩放：

$$
\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top
$$

其中 $\mathbf{R}$ 由四元数 $\mathbf{q}$ 构建，$\mathbf{S}$ 是对角缩放矩阵。

### 4.2 可微光栅化

3DGS 的核心创新是基于瓦片（tile-based）的可微光栅化器：

1. 将 3D 高斯投影到 2D 图像平面。
2. 对每个屏幕瓦片，按深度排序高斯。
3. 按 $\alpha$-blending 公式从前到后混合颜色：

$$
C = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

4. 利用 GPU 并行性实现实时渲染（>100 FPS）。

### 4.3 自适应密度控制

3DGS 在训练中动态增删高斯：

- **欠重建区域**（位置梯度大）：克隆或分裂小高斯，增加密度。
- **过密区域**（覆盖过大）：分裂大高斯为两个更小的高斯。
- **低不透明度高斯**：定期剪枝 $\alpha$ 过低的高斯。

这使表示从 SfM 初始化的稀疏点云逐步增长到数百万个高斯，自适应地捕捉场景细节。

### 4.4 3DGS 与 NeRF 对比

| 维度 | NeRF | 3DGS |
|------|------|------|
| 表示 | 隐式 MLP | 显式高斯椭球 |
| 渲染速度 | 慢（逐光线 MLP 查询） | 快（100+ FPS 光栅化） |
| 训练速度 | 数小时到数天 | 数十分钟 |
| 编辑性 | 难（隐式编码在权重中） | 容易（直接操作高斯） |
| 存储 | 小（MLP 权重） | 较大（数百万高斯） |
| 几何质量 | 优秀（连续表面） | 优秀，但可能有伪影 |

### 4.5 3DGS 生态发展

- **Scaffold-GS**：用锚点和神经特征预测高斯参数，减少伪影。
- **Mip-Splatting**：解决 3DGS 的锯齿和模糊问题。
- **Gaussian Grouping**：加入实例分割，支持 3D 编辑。
- **SuGaR**：从 3DGS 提取 Mesh，桥接传统图形管线。
- **Dynamic 3D Gaussians**：4D 场景中跟踪高斯随时间的运动。

---

## 5. 文本到 3D 生成

### 5.1 Score Distillation Sampling（SDS）

DreamFusion（Poole et al., 2022）开创了用预训练 2D 扩散模型监督 3D 生成的方法。核心思想是：从 3D 表示渲染不同视角的图像，用扩散模型的得分函数作为 3D 参数的梯度。

SDS 梯度为：

$$
\nabla_\theta \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t,\epsilon}\left[w(t)\left(\epsilon_\phi(\mathbf{x}_t; t, y) - \epsilon\right) \frac{\partial \mathbf{x}}{\partial \theta}\right]
$$

其中 $\mathbf{x} = g(\theta)$ 是从 3D 参数 $\theta$ 渲染的图像，$\epsilon_\phi$ 是扩散模型预测的噪声，$y$ 是文本提示。直观理解：扩散模型"告诉"3D 渲染结果"应该往哪个方向调整才能更像文本描述"。

### 5.2 代表性方法

| 方法 | 3D 表示 | 特点 |
|------|---------|------|
| DreamFusion | NeRF | SDS 开创性工作 |
| Magic3D | 两阶段（粗 NeRF → 细 Mesh） | 更高质量 |
| Fantasia3D | 分离几何与纹理 | 更好的表面质量 |
| ProlificDreamer | VSD（Variational Score Distillation） | 解决 SDS 过饱和问题 |
| DreamGaussian | 3DGS | 快速生成（~10 秒） |
| GaussianDreamer | 3DGS + 原始点云先验 | 更快收敛 |

### 5.3 SDS 的局限与改进

- **过度平滑和过饱和**：SDS 倾向于生成模式平均的结果，细节模糊。VSD（Wang et al., 2023）用变分分布建模替代单一模式，显著改善。
- **多视角不一致**（Janus 问题）：2D 扩散模型缺乏 3D 感知，可能给不同视角生成不对应的结构（如多面人）。
- **计算成本**：SDS 需要数千次扩散查询，生成一个物体需数十分钟到数小时。

---

## 6. 前馈 3D 重建

### 6.1 从优化到前馈

NeRF 和 3DGS 都是 per-scene 优化：每个场景需要单独训练。前馈方法训练一个神经网络，一次前向传播直接从图像（或文本）预测 3D 表示。

### 6.2 多视图扩散

- **Zero-1-to-3（Liu et al., 2023）**：给定一张输入图像，生成指定相机视角的新视图。通过在大规模数据上学习相机视角条件的扩散模型。
- **Viewset Diffusion**：联合生成多视图一致的图像集。
- **Instant3D**：一次前馈生成 4 个一致视图，再重建为 3D。

### 6.3 原生 3D 扩散模型

直接在 3D 表示上训练扩散模型：

- **Point-E / Shap-E（OpenAI）**：在点云或隐式函数上训练扩散模型，秒级生成。
- **LN3Diff / MeshDiffusion**：在 Mesh 或 triplane 表示上做扩散。
- **Rodin**：在 triplane 特征上做 3D 感知扩散。
- **Hunyuan3D（腾讯, 2024）**：两阶段生成（轮廓 → 细节），高质量文本到 3D。
- **TRELLIS（微软, 2024）**：结构化 Latent 表示上的 3D 扩散，支持生成 Mesh/3DGS/Radiance Field 多种输出。

前馈 3D 生成是 2024-2025 年最活跃的方向，目标是实现秒级、高质量、多表示的 3D 内容创作。

---

## 7. 4D 动态场景

### 7.1 动态 NeRF

- **D-NeRF**：引入时间维度，建模形变场（canonical space → deformed space）。
- **HexPlane**：将 4D 时空体素分解为六个平面，高效表示动态场景。
- **K-Planes**：统一的 $d$-维平面分解，支持静态/动态场景。

### 7.2 动态 3DGS

- **Dynamic 3D Gaussians（Luiten et al., 2024）**：每个高斯有时间相关的位置轨迹，用多项式建模运动。
- **4D Gaussians**：引入时间变形场，将 canonical 高斯变形到每帧。
- **Real-time Photorealistic Dynamic Scene Representation**：4DGS 实现动态场景实时渲染。

### 7.3 文本到 4D

- **4D-fy**：将 SDS 扩展到时间维度，生成文本到 4D 内容。
- **DreamGaussian4D**：基于 3DGS 的快速 4D 生成。
- **AYG (Align-Your-Gaussians)**：文本驱动的 4D 高斯生成。

---

## 8. 实践建议

### 8.1 方法选型

- **新视角合成**：3DGS 是当前首选，训练快、渲染快、质量高。
- **几何重建需 Mesh 输出**：NeRF/3DGS + 表面提取（如 SuGaR、2DGS）。
- **文本到 3D 快速原型**：基于 3DGS 的 SDS 方法（DreamGaussian）。
- **高质量产品级 3D 资产生成**：前馈模型（Hunyuan3D、TRELLIS）+ 人工精修。
- **动态场景**：4D Gaussian Splatting。

### 8.2 3DGS 训练建议

- 输入图像需覆盖场景各角度，避免大面积遮挡。COLMAP/SfM 质量直接影响结果。
- 训练从 SfM 点云初始化，若点云过于稀疏可先用随机初始化补充。
- 调整 densification 阈值和间隔：小场景更激进，大场景更保守。
- 移动端部署时需压缩高斯数量（剪枝 + 量化）。

### 8.3 与其他主题的衔接

- 3DGS 的实时渲染使其适合作为具身智能仿真器的场景表示（见 `embodied-ai/sim2real.md`）。
- 视频生成模型（见 `architectures/video-generation.md`）可提供多视图一致的训练数据。
- 3D 生成的资产可用于游戏、机器人训练数据合成和数字孪生。

---

## 9. 参考文献

[1] B. Mildenhall, P. P. Srinivasan, M. Tancik, et al., *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV, 2020.

[2] T. Müller, A. Evans, C. Schied, A. Keller, *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*, ACM TOG, 2022.

[3] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG, 2023.

[4] B. Poole, A. Jain, J. T. Barron, B. Mildenhall, *DreamFusion: Text-to-3D Using 2D Diffusion*, ICLR, 2023.

[5] Z. Wang, C. Lu, Y. Wang, et al., *ProlificDreamer: High-Fidelity and Diverse 3D Creative Generation with Variational Score Distillation*, NeurIPS, 2023.

[6] R. Liu, R. Wu, K. Vanhoey, et al., *Zero-1-to-3: Zero-Shot One Image to 3D Object*, ICCV, 2023.

[7] J. Luiten, G. Kopanas, B. Leibe, D. Ramanan, *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis*, 3DV, 2024.

[8] Y. Tang, J. Wang, Q. Wang, et al., *TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation*, arXiv preprint, 2024.

[9] X. Long, Y. Guo, C. Liu, et al., *Hunyuan3D 2.0: From Text to 3D Assets in Seconds*, arXiv preprint, 2025.

[10] A. Chen, Z. Xu, A. Geiger, et al., *TensoRF: Tensorial Radiance Fields*, ECCV, 2022.
