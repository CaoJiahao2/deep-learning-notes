# 扩散模型（Diffusion Models）详解

> 从 DDPM 的数学推导到 DDIM、Classifier-Free Guidance 与 Flow Matching 的完整解析。

---

## 目录

1. [核心思想](#1)
2. [前向扩散过程](#2)
3. [反向生成过程与 ELBO](#3-elbo)
4. [简化损失 L_simple](#4-l_simple)
5. [网络与噪声调度](#5)
6. [DDIM：确定性采样](#6-ddim)
7. [Classifier-Free Guidance](#7-classifier-free-guidance)
8. [Latent Diffusion 与 Stable Diffusion](#8-latent-diffusion-stable-diffusion)
9. [Flow Matching / Rectified Flow](#9-flow-matching-rectified-flow)
10. [主流框架与库](#10)
11. [参考文献](#11)

---

## 1. 核心思想

扩散模型（Ho et al., 2020）分两个阶段：

1. **前向过程（加噪）**：逐步向数据 $\mathbf{x}_0$ 添加高斯噪声，直到变成纯噪声 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$。
2. **反向过程（去噪）**：学习一个神经网络逐步"去噪"，从纯噪声恢复数据。

关键洞察：前向过程是**已知、固定的**（由方差调度决定），反向过程才是需要学习的。

---

## 2. 前向扩散过程

### 2.1 单步加噪

给定方差调度 $\{\beta_t\}_{t=1}^T$（通常 $0 < \beta_t < 1$），前向过程定义为马尔可夫链：

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t; \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1}, \beta_t \mathbf{I}\right)$$

即 $\mathbf{x}_t = \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_t$，$\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。

### 2.2 任意时刻的解析形式（重参数化）

定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$。利用高斯分布的可加性，可**一步**从 $\mathbf{x}_0$ 得到 $\mathbf{x}_t$：

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**推导**：由 $\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_t$ 逐层展开并合并独立高斯噪声（两个独立高斯的加权和仍为高斯，方差相加）：

$$\mathbf{x}_t = \sqrt{\prod_{s=1}^{t}\alpha_s}\,\mathbf{x}_0 + \underbrace{\sqrt{1 - \bar{\alpha}_t}}_{\text{累积噪声方差}}\,\boldsymbol{\epsilon}$$

当 $T$ 足够大且 $\bar{\alpha}_T \to 0$ 时，$\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$。

---

## 3. 反向生成过程与 ELBO

### 3.1 反向过程

反向过程 $p_\theta$ 每一步也是高斯（对足够小的 $\beta_t$ 的近似），由神经网络参数化：

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)\right)$$

目标：最大化对数似然 $\log p_\theta(\mathbf{x}_0)$。

### 3.2 ELBO 推导

引入前向过程 $q$，由 Jensen 不等式（或重写为期望形式）：

$$\log p_\theta(\mathbf{x}_0) = \log \int p_\theta(\mathbf{x}_0, \mathbf{x}_{1:T}) \, d\mathbf{x}_{1:T} \ge \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \left[\log \frac{p_\theta(\mathbf{x}_0, \mathbf{x}_{1:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}\right]$$

将联合分布展开并整理，损失（负 ELBO）可分解为三项：

$$\mathcal{L} = \underbrace{D_{KL}\big(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, p(\mathbf{x}_T)\big)}_{\mathcal{L}_T} + \sum_{t=2}^{T} \underbrace{D_{KL}\big(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big)}_{\mathcal{L}_{t-1}} - \underbrace{\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}_{\mathcal{L}_0}$$

- $\mathcal{L}_T$：先验项，无参数（$\mathbf{x}_T$ 趋近标准高斯），可视为常数；
- $\mathcal{L}_0$：重建项（最后一步），离散像素时用独立离散解码器；
- $\mathcal{L}_{t-1}$：**去噪匹配项**，是训练的核心。

### 3.3 可解析的高斯后验

关键：当以 $\mathbf{x}_0$ 为条件时，$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ 的均值和方差可解析。由贝叶斯公式对高斯分布求后验：

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1};\, \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\, \tilde{\sigma}_t^2 \mathbf{I}\right)$$

其中（令 $\tilde\beta_t = \beta_t$，$\bar\alpha_t$ 如上）：

$$\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}\,\mathbf{x}_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}\,\mathbf{x}_t, \qquad \tilde{\sigma}_t^2 = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\,\beta_t$$

### 3.4 两个高斯之间的 KL

两个高斯 $\mathcal{N}(\boldsymbol{\mu}_1, \sigma_1^2 \mathbf{I})$ 与 $\mathcal{N}(\boldsymbol{\mu}_2, \sigma_2^2 \mathbf{I})$ 的 KL 散度为：

$$D_{KL}\big(\mathcal{N}_1 \,\|\, \mathcal{N}_2\big) = \frac{1}{2}\left[d \log\frac{\sigma_2^2}{\sigma_1^2} + \frac{d\,\sigma_1^2 + \|\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1\|^2}{\sigma_2^2} - d\right]$$

因此 $\mathcal{L}_{t-1}$ 等价于使网络均值 $\boldsymbol{\mu}_\theta$ 逼近解析后验均值 $\tilde{\boldsymbol{\mu}}_t$：

$$\mathcal{L}_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\left\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\right\|^2\right] + \text{const}$$

---

## 4. 简化损失 L_simple

将 $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon}$ 代入 $\tilde{\boldsymbol{\mu}}_t$，可消去 $\mathbf{x}_0$，得到 $\tilde{\boldsymbol{\mu}}_t$ 关于 $\mathbf{x}_t$ 与 $\boldsymbol{\epsilon}$ 的表达式。令网络改为**预测噪声**：$\boldsymbol{\mu}_\theta$ 由 $\boldsymbol{\epsilon}_\theta$ 表示，则均值匹配损失等价于（忽略常数系数）：

$$\mathcal{L}_{t-1} \propto \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left(\sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\epsilon},\, t\right)\right\|^2\right]$$

Ho et al. (2020) 进一步去掉各时刻的权重系数，得**简化的训练目标**：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t \sim \mathcal{U}\{1,\dots,T\},\, \mathbf{x}_0 \sim q(\mathbf{x}_0),\, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})} \left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right\|^2\right]$$

这是实际训练中学到的唯一损失：**随机采样时刻 $t$ 与噪声 $\boldsymbol{\epsilon}$，让网络预测注入的噪声**。

### 推理（生成）

采样时从 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，逐步：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

---

## 5. 网络与噪声调度

### 5.1 架构

- **UNet**：多分辨率编码器-解码器 + 跳跃连接，时间 $t$ 通过正弦位置编码 + FiLM/AdaGN 注入。
- **DiT（Diffusion Transformer, 2023）**：用 Transformer 块替代 UNet 残差块，将扩散模型带入"scaling"范式，是 Sora、SD3、Flux 的基础。

### 5.2 方差调度

| 调度 | 形式 | 特点 |
|------|------|------|
| Linear（DDPM） | $\beta_t$ 从 $10^{-4}$ 线性到 $0.02$ | 原始方案 |
| Cosine（2021） | 平滑的余弦形式 | 高分辨率训练更稳 |
| Learned | 网络学习 $\beta_t$ | 灵活但更复杂 |

---

## 6. DDIM：确定性采样

DDIM（Song et al., 2021）将采样过程改写为**非马尔可夫**形式，使反向过程可以**确定性**（$\sigma_t = 0$）：

$$\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\left(\frac{\mathbf{x}_t - \sqrt{1 - \bar\alpha_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar\alpha_t}}\right) + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \sigma_t \mathbf{z}$$

- $\sigma_t = 0$：确定性映射，**DDIM**，可用远少于训练步数的步数采样（如 20–50 步）；
- $\sigma_t = \sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}}$：退化为 DDPM 的随机采样。

DDIM 的代价是通常略降低多样性，但显著加速推理，并支持**潜空间插值**（在 $\mathbf{x}_T$ 上插值得到平滑过渡）。

---

## 7. Classifier-Free Guidance

CFG（Ho & Salimans, 2022）在**训练时**以概率 $p_{\text{uncond}}$ 丢弃条件（如文本），使同一个网络同时建模条件与无条件分布。**推理时**：

$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) + w\left(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing)\right)$$

其中 $w \ge 1$ 为引导强度（$w$ 越大结果越贴合条件、多样性越低）。直观上，网络在有条件与无条件两个方向上做外推，从而放大条件信息的影响。

---

## 8. Latent Diffusion 与 Stable Diffusion

Latent Diffusion（Rombach et al., 2022）的核心：**不在像素空间而在 VAE 潜空间做扩散**。

- 先用预训练 VAE 将图像编码为潜变量 $\mathbf{z} = \mathcal{E}(\mathbf{x})$；
- 在潜空间运行 DDPM/DDIM；
- 用 VAE 解码器 $\mathcal{D}$ 重建图像。

VAE 训练损失为**重构项 + KL 正则**：

$$\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\left[\log p_\theta(\mathbf{x} \mid \mathbf{z})\right] + \beta\, D_{KL}\big(q_\phi(\mathbf{z}\mid\mathbf{x}) \,\|\, p(\mathbf{z})\big)$$

优势：潜空间维度远小于像素空间，训练与推理成本大幅降低；条件（文本）通过 cross-attention 注入。**Stable Diffusion** 即在此框架上加入 CLIP 文本编码器的大规模实现。

---

## 9. Flow Matching / Rectified Flow

Flow Matching（Lipman et al., 2023）是 2024–2025 图像/视频生成（Flux、SD3、视频模型）的主流框架，可视为扩散的推广。

### 9.1 插值路径

定义概率路径：从数据 $\mathbf{x}_0$ 到噪声 $\boldsymbol{\epsilon}$ 的线性插值：

$$\mathbf{x}_t = (1 - t)\,\mathbf{x}_0 + t\,\boldsymbol{\epsilon}, \qquad t \in [0, 1]$$

对 $t$ 求导，目标速度场（velocity）为常数：

$$\dot{\mathbf{x}}_t = \boldsymbol{\epsilon} - \mathbf{x}_0$$

### 9.2 训练目标

用神经网络 $v_\theta$ 回归速度场：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\left\|v_\theta\left((1-t)\mathbf{x}_0 + t\boldsymbol{\epsilon},\, t\right) - \left(\boldsymbol{\epsilon} - \mathbf{x}_0\right)\right\|^2\right]$$

### 9.3 采样

学得 $v_\theta$ 后，用 ODE 求解器（如 Euler 或更高阶）从 $\mathbf{x}_0 = \boldsymbol{\epsilon}$ 积分到 $\mathbf{x}_1$：

$$\frac{d\mathbf{x}_t}{dt} = v_\theta(\mathbf{x}_t, t), \qquad \mathbf{x}_1 = \mathbf{x}_0 + \int_0^1 v_\theta(\mathbf{x}_t, t)\, dt$$

**Rectified Flow** 进一步通过"多次沿直线重新配对"来让轨迹更直，从而能用更少步数精确采样。**SDE→ODE 的视角**统一了扩散与流匹配：扩散可以看作一种特殊的随机微分方程，而蒸馏/少步采样的核心是让 ODE 轨迹更可积分。

---

## 10. 主流框架与库

| 框架/库 | 语言 | 特点 |
|---------|------|------|
| Hugging Face Diffusers | Python | 模块化（UNet/DiT、Scheduler、Pipeline），支持 DDPM/DDIM/LDM，文档完善 |
| torchdiffeq | Python | 常微分方程求解器，用于 Flow Matching / 连续归一化流采样 |
| OpenAI guided-diffusion | Python | 论文官方实现，适合研究 DDPM→DDIM |
| ComfyUI / Automatic1111 | Python | Stable Diffusion 生态、可视化工作流 |
| MMDiT（SD3/Flux） | Python | 多模态 DiT，当前图像生成 SOTA 架构 |

---

## 11. 参考文献

[1] Ho, J., Jain, A. & Abbeel, P. Denoising Diffusion Probabilistic Models. NeurIPS, 2020.

[2] Song, J., et al. Denoising Diffusion Implicit Models. ICLR, 2021.

[3] Dhariwal, P. & Nichol, A. Diffusion Models Beat GANs on Image Synthesis. NeurIPS, 2021.

[4] Ho, J. & Salimans, T. Classifier-Free Diffusion Guidance. NeurIPS Workshop, 2022.

[5] Rombach, R., et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR, 2022.

[6] Peebles, W. & Xie, S. Scalable Diffusion Models with Transformers (DiT). ICCV, 2023.

[7] Lipman, Y., et al. Flow Matching for Generative Modeling. ICLR, 2023.

[8] Liu, X., et al. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. ICLR, 2023.

[9] Kingma, D. & Welling, M. Auto-Encoding Variational Bayes. ICLR, 2014. (VAE)
