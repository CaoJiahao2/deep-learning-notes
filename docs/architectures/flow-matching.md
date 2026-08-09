# Flow Matching 与 Rectified Flow

> 用 ODE 直接回归"速度场"，替代 SDE 噪声预测。SD3、Flux、π0、Wan 2.1 都建立在此基础上。

---

## 目录

1. [概念与动机](#1)
2. [从 Diffusion 到 Continuous Normalizing Flow](#2)
3. [Flow Matching 形式化](#3)
4. [Conditional Flow Matching](#4)
5. [与 Diffusion 的等价关系](#5)
6. [Rectified Flow 与 Reflow](#6)
7. [少步采样与蒸馏](#7)
8. [代表工作](#8)
9. [实践建议](#9)
10. [参考文献](#10)

---

## 1 概念与动机 {#1}

扩散模型（DDPM/DDIM）已是大规模生成的主流，但有两条被广泛讨论的限制：

1. **训练**：依赖噪声调度与 SDE 路径，超参数敏感；
2. **采样**：需要数十到上百步 ODE 求解。

**Flow Matching**（Lipman et al., 2023）与 **Rectified Flow**（Liu et al., 2023）提供了一条更直接的路径：

- 用简单线性插值把"数据 $x_0$" 与"噪声 $x_1$" 连接；
- 让神经网络预测**速度场** $v_\theta(x_t, t)$，而不是"噪声 $\epsilon$"或"分数 $\nabla \log p_t$"；
- 训练目标简单、采样步数少、与 ODE 求解器兼容。

SD3、Flux、π0、Wan 2.1 等 2024–2025 主流生成模型都基于这一思想。

---

## 2 从 Diffusion 到 Continuous Normalizing Flow {#2}

### 2.1 Continuous Normalizing Flow (CNF)

CNF 定义一个连续可微的微分方程：

$$\frac{dx}{dt} = v_\theta(x, t), \quad x(0) = x_0$$

由 $x_0$ 沿 ODE 演化到 $x(1)$。对任意时刻 $t$：

$$x(t) = x_0 + \int_0^t v_\theta(x(s), s) \, ds$$

### 2.2 与生成模型的关系

只要让 $x(0)$ 服从"目标分布"、$x(1)$ 服从"简单分布"（如高斯），这个 ODE 就是一个生成模型。Flow Matching 直接学习 $v_\theta$。

---

## 3 Flow Matching 形式化 {#3}

### 3.1 概率路径与速度场

定义一族概率路径 $p_t(x)$，连接数据分布 $p_0$ 与噪声分布 $p_1$。对应的速度场 $v_t(x)$ 满足：

$$\frac{dx}{dt} = v_t(x), \quad \frac{\partial p_t}{\partial t} = -\nabla \cdot (p_t v_t)$$

### 3.2 Flow Matching 目标

训练 $v_\theta(x, t)$ 逼近真实速度场 $v_t(x)$：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim U(0,1),\, x \sim p_t(x)} \left[\| v_\theta(x, t) - v_t(x) \|^2 \right]$$

这是 Flow Matching 的一般定义。它看起来很简单，但直接计算 $v_t(x)$ 通常很困难。

---

## 4 Conditional Flow Matching {#4}

### 4.1 条件化简化

把整条路径"条件化"在样本对 $(x_0, x_1)$ 上：给定 $x_0 \sim p_0, x_1 \sim p_1$，定义条件路径

$$p_t(x \mid x_0, x_1) = \mathcal{N}(x;\, \mu_t(x_0, x_1),\, \sigma_t^2(x_0, x_1))$$

最简单的选择是**线性插值**（Rectified Flow）：

$$\mu_t(x_0, x_1) = (1-t) x_0 + t x_1, \quad \sigma_t = 0$$

此时条件速度场简化为：

$$v_t(x \mid x_0, x_1) = \frac{d \mu_t}{dt} = x_1 - x_0$$

### 4.2 Conditional Flow Matching 目标

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t,\, x_0,\, x_1,\, x_t \mid x_0, x_1} \left[\| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$

关键结论（Lipman et al.）：**CFM 梯度等于 FM 梯度**。即这两个目标在期望意义上等价，所以可以直接用简单的线性插值训练。

### 4.3 训练循环伪代码

```text
for batch:
    x0 = sample from data distribution       # 真图像
    x1 = sample from N(0, I)                 # 噪声
    t  = sample uniform(0, 1)
    xt = (1 - t) * x0 + t * x1               # 线性插值
    target = x1 - x0                          # 速度场
    loss = || v_theta(xt, t) - target ||^2
    optimizer.step(loss)
```

---

## 5 与 Diffusion 的等价关系 {#5}

### 5.1 速度场与分数函数

在扩散模型里我们学习分数函数 $s_\theta(x_t, t) \approx \nabla_{x_t} \log p_t(x_t)$，它们满足：

$$v_t(x_t) = \frac{\dot{\sigma}_t}{\sigma_t} x_t + \dot{\mu}_t - \dot{\sigma}_t \sigma_t \nabla_{x_t} \log p_t(x_t)$$

其中 $\mu_t, \sigma_t$ 是边缘分布的均值与标准差。给定分数函数，可以"反过来"得到速度场。

### 5.2 DDPM 重新解释

DDPM 训练 $\epsilon_\theta(x_t, t) \approx \epsilon$ 的损失等价于训练分数函数：

$$\mathcal{L}_{\text{diff}} = \mathbb{E}\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right] = c(t) \mathbb{E}\left[\| \nabla_{x_t} \log p_t - \nabla_{x_t} \log p_\theta(x_t) \|^2\right]$$

因此**Diffusion 与 Flow Matching 在期望意义上可以互相重写**，只是参数化方式不同。

### 5.3 实用差异

| 维度 | Diffusion (DDPM) | Flow Matching |
|------|------------------|---------------|
| 训练目标 | 预测噪声 $\epsilon$ | 预测速度场 $v$ |
| 训练稳定性 | 良好，依赖噪声调度 | 通常更好，路径直接 |
| 路径选择 | 灵活 | 通常线性插值 |
| 采样器 | DDPM/SDE/DDIM | ODE（Euler、Heun、Dopri5） |

---

## 6 Rectified Flow 与 Reflow {#6}

### 6.1 Rectified Flow

**Rectified Flow**（Liu et al., 2023）使用**线性插值路径**：

$$\frac{dx_t}{dt} = x_1 - x_0$$

直接回归 $x_1 - x_0$，训练稳定、收敛快。

### 6.2 Reflow

把训练好的 Rectified Flow 模型用作"教师"，把它的 ODE 轨迹重新"拉直"：

- 用教师模型从 $x_1$ 积分到 $x_0$；
- 把 $(x_0, x_1)$ 对重新收集为新的训练数据；
- 重新训练。

经过多轮 Reflow 后，ODE 轨迹几乎是一条直线，可以用极少的步数（如 1–4 步）采样得到高质量结果。

### 6.3 工业意义

Reflow 是少步采样的关键训练技巧，避免 Consistency Model / Progressive Distillation 的复杂流程。

---

## 7 少步采样与蒸馏 {#7}

### 7.1 ODE 求解器

推理时用 ODE 求解器从 $t=1$ 积分到 $t=0$：

| 求解器 | 步数 | 精度 |
|--------|------|------|
| Euler | 多 | 低 |
| Heun | 中 | 高 |
| Dopri5 | 自适应 | 高 |

### 7.2 蒸馏

把多步 ODE 的输出蒸馏成单步模型：

- **Consistency Model**：直接训练"任意时刻收敛到同一终点"的模型；
- **Consistency Distillation**：以教师轨迹为监督；
- **Score Distillation**：用扩散损失蒸馏。

### 7.3 渐进蒸馏

逐步把 $N$ 步模型蒸馏成 $N/2$ 步，重复直到目标步数。SD3-Turbo、Flux-Schnell 等都用了类似技巧。

---

## 8 代表工作 {#8}

### 8.1 学术

- **Lipman et al., Flow Matching for Generative Modeling, ICLR 2023**
- **Liu et al., Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, ICLR 2023**
- **Albergo & Vanden-Eijnden, Building Normalizing Flows with Stochastic Interpolants, NeurIPS 2023**

### 8.2 工业

- **Stable Diffusion 3**（Stability AI）：用 Rectified Flow + MMDiT；
- **Flux**（Black Forest Labs）：Flow Matching + DiT；
- **π0 / π0.5**（Physical Intelligence）：Flow Matching 在动作空间上的应用；
- **Wan 2.1**（Alibaba）：Flow Matching 用于视频生成；
- **Sora**（OpenAI）：技术细节未完全披露，但社区普遍认为使用了 Rectified Flow。

---

## 9 实践建议 {#9}

1. **采样步数**：起步用 20–50 步 Euler / Heun，根据 FID 调节。
2. **CFG**：Classifier-Free Guidance 仍适用，注意 Flow Matching 的 cfg 计算公式略有差异。
3. **蒸馏路径**：先用全步教师，再做 Consistency / Reflow 蒸馏。
4. **架构选择**：DiT（Diffusion Transformer）在 Flow Matching 时代已成为主流，CNN UNet 仍可用于小模型。
5. **数值稳定性**：使用 fp16 / bf16 时注意 ODE 求解器在极端时间步下的稳定性，必要时切换为 fp32 时间步。
6. **训练数据规模**：Flow Matching 与 Diffusion 都需要足够规模的数据；架构选择之外，数据规模往往是决定性因素。

---

## 10 参考文献 {#10}

[1] Y. Lipman et al., *Flow Matching for Generative Modeling*, ICLR, 2023. arXiv:2210.02747.

[2] X. Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR, 2023. arXiv:2209.03003.

[3] M. S. Albergo et al., *Building Normalizing Flows with Stochastic Interpolants*, NeurIPS, 2023. arXiv:2209.14903.

[4] M. S. Albergo et al., *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*, 2023.

[5] P. Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (Stable Diffusion 3)*, 2024. arXiv:2403.06306.

[6] Black Forest Labs, *Flux: A Flow Matching Architecture*, 2024.

[7] Y. Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR, 2021. arXiv:2011.13456.

[8] J. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS, 2020. arXiv:2006.11239.

[9] T. Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models*, NeurIPS, 2022.

[10] Physical Intelligence, *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. arXiv:2410.24164.