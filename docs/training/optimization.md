# 优化器详解

> 从 SGD 到 AdamW 再到 Lion，深度学习优化器的完整演进。

---

## 目录

1. [梯度下降基础](#1)
2. [SGD 与 Momentum](#2-sgd-momentum)
3. [自适应学习率方法](#3)
4. [Adam 及其变体](#4-adam)
5. [学习率调度](#5)
6. [优化器选择指南](#6)
7. [参考文献](#7)

---

## 1. 梯度下降基础

### 1.1 批量梯度下降 (BGD)

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

每次使用全部训练数据计算梯度。优点是收敛稳定，缺点是计算成本高。

### 1.2 随机梯度下降 (SGD)

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t; x_i, y_i)$$

每次只使用一个样本计算梯度。优点是更新快，缺点是方差大、收敛不稳定。

### 1.3 小批量梯度下降 (Mini-batch SGD)

$$\theta_{t+1} = \theta_t - \eta \cdot \frac{1}{m}\sum_{i=1}^{m} \nabla_\theta J(\theta_t; x_i, y_i)$$

现代深度学习的标准做法，在 BGD 和 SGD 之间取得平衡。

---

## 2. SGD 与 Momentum

### 2.1 SGD 的问题

- 在峡谷形（ill-conditioned）损失曲面上震荡
- 收敛速度慢，尤其是在平坦区域
- 容易被局部极小值或鞍点困住

### 2.2 Momentum (动量法)

引入速度变量，累积历史梯度：

$$v_t = \beta v_{t-1} + \nabla_\theta J(\theta_t)$$

$$\theta_{t+1} = \theta_t - \eta v_t$$

其中 $\beta$ 通常设为 0.9。

**物理意义**：球滚下山坡时，动量使其保持方向，不会因小坑洼而停滞。

### 2.3 Nesterov Accelerated Gradient (NAG)

先沿动量方向移动，再计算梯度进行修正：

$$v_t = \beta v_{t-1} + \nabla_\theta J(\theta_t - \eta \beta v_{t-1})$$

$$\theta_{t+1} = \theta_t - \eta v_t$$

NAG 通过"前瞻"梯度计算，在凸优化中具有更好的收敛率保证。

---

## 3. 自适应学习率方法

### 3.1 AdaGrad

对每个参数使用不同的自适应学习率：

$$G_t = G_{t-1} + (\nabla_\theta J_t)^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla_\theta J_t$$

**问题**：$G_t$ 单调递增，导致学习率过早衰减到 0。

### 3.2 RMSprop

使用指数移动平均替代累加和：

$$v_t = \beta v_{t-1} + (1 - \beta)(\nabla_\theta J_t)^2$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \odot \nabla_\theta J_t$$

### 3.3 AdaDelta

在 RMSprop 基础上消除对全局学习率的依赖，同时维护参数更新的移动平均。

---

## 4. Adam 及其变体

### 4.1 Adam

Adam (Kingma & Ba, 2015) 结合了 Momentum 和 RMSprop：

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{（一阶矩估计）}$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{（二阶矩估计）}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{（偏差校正）}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**默认超参数**：$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$

### 4.2 AdamW

Adam 的 Weight Decay 实现存在问题：L2 正则化和 Weight Decay 在 Adam 中不等价。AdamW (Loshchilov & Hutter, 2019) 将权重衰减与梯度更新解耦：

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

这一改进对现代 Transformer 训练至关重要。**现代 DL 训练几乎一律使用 AdamW**。

### 4.3 Adam vs AdamW 核心区别

| 方面 | Adam | AdamW |
|------|------|-------|
| Weight Decay | 混在梯度中 | 与梯度解耦 |
| 泛化性能 | 较差 | 更好 |
| 超参数分离 | $\eta$ 和 $\lambda$ 耦合 | $\eta$ 和 $\lambda$ 独立 |

### 4.4 其他 Adam 变体

| 变体 | 创新点 |
|------|--------|
| AMSGrad | 用历史最大值替代 EMA，保证收敛 |
| AdaBound | Adam 到 SGD 的平滑过渡 |
| RAdam | 修正自适应学习率的方差问题 |
| Adamax | 使用无穷范数替代 L2 范数 |
| NAdam | 结合 Nesterov 动量 |

### 4.5 Lion (2023)

Google 发现的进化优化器，使用符号更新：

$$\theta_{t+1} = \theta_t - \eta \cdot (\text{sign}(\beta_1 m_{t-1} + (1 - \beta_1) g_t) + \lambda \theta_t)$$

- 内存占用比 AdamW 少一半（不需要二阶矩）
- 训练 Vision Transformer 和扩散模型更高效
- 对 batch size 和 learning rate 更敏感

---

### 4.6 前沿优化器（2024–2025）

| 优化器 | 核心思想 | 特点 |
|--------|---------|------|
| **Muon** | 对"类矩阵"参数近似做 $O$（正交化）更新，标量参数用 AdamW | 大模型训练常比 AdamW 收敛更快（如原版 Gemini 2 / 若干开源模型） |
| **Schedule-Free** | 去掉显式学习率调度，用凸组合（interpolation）更新历史 | 减少调度调参负担，PyTorch 官方已实现 `torch.optim.ScheduleFree` |
| **SophiG** | 用矩阵 preconditioner 的幂次近似（Schur-Newton）替代逐元素缩放 | 高精度、内存可控，适合稠密中层 |
| **muP（Maximal Update Parametrization）** | 随宽度缩放时使用不同的学习率标度，使模型达到"最大更新" | 理论支撑：小模型超参可迁移到大模型的必要架构与学习率标度 |

**Muon 更新示意**（设 $M$ 为权重矩阵）：

$$M_{t+1} = M_t - \eta\,\operatorname{NewtonSchulz}(\text{orthog}(\text{muon\_momentum}_t))$$

其中 `orthog` 做正交化、`NewtonSchulz` 近似矩阵平方根，从而在保持秩的同时提供更"全局"的更新方向。

---

## 5. 学习率调度

### 5.1 常见调度策略

| 调度器 | 描述 | 适用场景 |
|--------|------|---------|
| StepLR | 每 N 步将 LR 乘以 $\gamma$ | 简单任务 |
| MultiStepLR | 在预设里程碑处衰减 | 传统 CV 任务 |
| ExponentialLR | 指数衰减 | 持续衰减 |
| CosineAnnealingLR | 余弦衰减 | 现代 DL 标配 |
| Linear Warmup + Cosine | 线性预热 + 余弦衰减 | Transformer 训练 |
| OneCycleLR | 先增后减的单周期策略 | 快速训练 |
| ReduceLROnPlateau | 监控指标，停滞时衰减 | 自适应调整 |

### 5.2 Warmup 的重要性

在训练初期，模型参数是随机初始化的，梯度方差很大。直接使用大学习率会导致训练不稳定。

**Warmup 策略**：在最初的 $N$ 步中，学习率从 0 线性增加到目标值。

$$\eta_t = \eta_{target} \cdot \frac{t}{N_{warmup}}$$

### 5.3 Cosine Annealing with Warmup

当前 Transformer 训练的标准做法：

```python
# 典型配置
lr = 3e-4
warmup_steps = 4000
total_steps = 100000

# 1. Warmup 阶段：线性增加
lr(t) = lr * t / warmup_steps     (t < warmup_steps)

# 2. Cosine 衰减阶段
lr(t) = lr * 0.5 * (1 + cos(π * (t - warmup) / (total - warmup)))
```

---

## 6. 优化器选择指南

### 6.1 按任务推荐

| 任务 | 推荐优化器 | 说明 |
|------|-----------|------|
| Transformer 预训练 | AdamW | 结合 warmup + cosine decay |
| CNN 图像分类 | SGD + Momentum | 经典选择，泛化好 |
| 大模型微调 | AdamW | 默认选择 |
| GAN 训练 | Adam | 通常 $\beta_1=0.5$ |
| 扩散模型 | AdamW / Lion | Lion 更快但需调参 |
| 强化学习 | Adam | 默认选择 |

### 6.2 超参数调优建议

| 超参数 | 建议范围 | 说明 |
|--------|---------|------|
| $\eta$ (学习率) | $10^{-4} \sim 10^{-3}$ (AdamW) | Transformer 推荐 $3 \times 10^{-4}$ |
| $\beta_1$ | 0.9 (默认) | 一阶动量 |
| $\beta_2$ | 0.999 (默认) 或 0.95 (大模型) | 二阶动量 |
| $\epsilon$ | $10^{-8} \sim 10^{-6}$ | 大模型可增大到 $10^{-5}$ |
| Weight Decay | 0.01 ~ 0.1 | AdamW 中与学习率解耦 |

### 6.3 常见陷阱

1. **Adam 的 Weight Decay**：务必使用 AdamW 而非 Adam 的 weight_decay 参数
2. **学习率预热**：Transformer 训练必须使用 warmup，否则极易发散
3. **梯度裁剪**：配合 warmup 使用，裁剪阈值通常设为 1.0
4. **混合精度**：使用 FP16/BF16 训练时，需注意 loss scaling

---

## 7. 参考文献

[1] Kingma, D. P. & Ba, J. Adam: A Method for Stochastic Optimization. ICLR, 2015.

[2] Loshchilov, I. & Hutter, F. Decoupled Weight Decay Regularization. ICLR, 2019.

[3] Chen, X., et al. Symbolic Discovery of Optimization Algorithms. NeurIPS, 2023. (Lion)

[4] Sutskever, I., et al. On the Importance of Initialization and Momentum in Deep Learning. ICML, 2013.

[5] Reddi, S. J., et al. On the Convergence of Adam and Beyond. ICLR, 2018. (AMSGrad)

[6] Liu, L., et al. On the Variance of the Adaptive Learning Rate and Beyond. ICLR, 2020. (RAdam)

[7] Loshchilov, I. & Hutter, F. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR, 2017.

[8] Smith, L. N. Cyclical Learning Rates for Training Neural Networks. WACV, 2017.
