# 正则化技术详解

> 从经典 Dropout 到现代归一化方法，防止过拟合的完整工具箱。

---

## 目录

1. [过拟合与正则化概述](#1-过拟合与正则化概述)
2. [参数范数惩罚](#2-参数范数惩罚)
3. [Dropout 系列](#3-dropout-系列)
4. [归一化方法](#4-归一化方法)
5. [数据增强](#5-数据增强)
6. [早停法 (Early Stopping)](#6-早停法-early-stopping)
7. [标签平滑 (Label Smoothing)](#7-标签平滑-label-smoothing)
8. [正则化策略选择](#8-正则化策略选择)
9. [参考文献](#9-参考文献)

---

## 1. 过拟合与正则化概述

### 1.1 过拟合的表现

- 训练集上表现优异，验证/测试集上表现差
- 模型对训练数据中的噪声也进行了拟合
- 泛化误差远大于训练误差

### 1.2 偏差-方差权衡

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{(\mathbb{E}[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}$$

- **高偏差**：欠拟合，模型过于简单
- **高方差**：过拟合，模型对训练数据过于敏感

正则化的目标是**降低方差**，同时不过度增加偏差。

---

## 2. 参数范数惩罚

### 2.1 L2 正则化 (Weight Decay / Ridge)

在损失函数中添加参数平方和：

$$\tilde{J}(\theta) = J(\theta) + \frac{\lambda}{2}\|\theta\|_2^2$$

梯度更新：

$$\theta_{t+1} = \theta_t - \eta(\nabla_\theta J(\theta_t) + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta\nabla_\theta J(\theta_t)$$

**效果**：使权重趋于较小的值，限制模型复杂度。在 SGD 中等价于 Weight Decay。

### 2.2 L1 正则化 (Lasso)

$$\tilde{J}(\theta) = J(\theta) + \lambda\|\theta\|_1$$

**效果**：产生稀疏解，部分权重变为 0，天然具有特征选择功能。

### 2.3 L2 vs L1 对比

| 特性 | L2 | L1 |
|------|-----|-----|
| 解的性质 | 权重接近 0 但非 0 | 权重可能精确为 0 |
| 梯度 | 与权重成正比 | 常数（符号函数） |
| 特征选择 | 否 | 是 |
| 常用场景 | 深度学习标配 | 稀疏模型 |
| 与优化器交互 | Adam 中需用 AdamW | 不常用 |

### 2.4 Elastic Net

L1 和 L2 的组合：

$$\tilde{J}(\theta) = J(\theta) + \lambda_1\|\theta\|_1 + \frac{\lambda_2}{2}\|\theta\|_2^2$$

---

## 3. Dropout 系列

### 3.1 Dropout

训练时以概率 $p$ 随机丢弃神经元：

$$\tilde{h}_i = h_i \cdot m_i, \quad m_i \sim \text{Bernoulli}(1-p)$$

推理时使用所有权重，但乘以 $(1-p)$ 进行缩放（或训练时除以 $1-p$ 实现 inverted dropout）：

$$h_i^{\text{test}} = (1-p) \cdot h_i$$

**为什么有效**：
- **集成学习视角**：每次训练相当于训练一个不同的子网络，推理时相当于 $2^n$ 个子网络的集成
- **防止共适应**：每个神经元不能依赖特定其他神经元的存在
- **噪声注入**：相当于对隐藏层添加乘性噪声

### 3.2 Dropout 的改进变体

| 变体 | 方法 | 适用场景 |
|------|------|---------|
| DropConnect | 随机丢弃权重而非激活值 | 全连接层 |
| Spatial Dropout | 按通道丢弃整个特征图 | CNN |
| DropBlock | 丢弃连续区域 | CNN |
| Stochastic Depth | 随机丢弃整个残差块 | ResNet |
| DropPath | 随机丢弃路径 | ViT, Swin Transformer |

### 3.3 Dropout 使用建议

- CNN：通常 $p=0.2 \sim 0.5$，在全连接层使用
- Transformer：$p=0.1$ 为默认值，应用于 Attention 和 FFN 输出
- 现代 CNN（ResNet 等）倾向于使用 BatchNorm 替代 Dropout
- ViT 等现代架构仍广泛使用 Dropout / DropPath
- 训练时开启，推理时关闭

---

## 4. 归一化方法

### 4.1 Batch Normalization

对每个 mini-batch 在通道维度上归一化：

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

**为什么 BN 有正则化效果**：
- 每个样本的统计量依赖于 batch 中的其他样本，引入随机性
- 这种隐式噪声类似于 Dropout 的效果

**局限性**：
- 小 batch 时统计量不稳定
- RNN 中难以应用（序列长度可变）
- 训练和推理行为不一致（需维护 running mean/var）

### 4.2 Layer Normalization

在特征维度上归一化（每个样本独立）：

$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$$

**优势**：
- 不依赖 batch size，小 batch 也稳定
- 训练和推理行为一致
- Transformer 的标准选择

### 4.3 其他归一化方法

| 方法 | 归一化维度 | 适用场景 |
|------|-----------|---------|
| InstanceNorm | 单样本单通道 | 风格迁移 (StyleGAN) |
| GroupNorm | 通道分组 | 小 batch 训练、分割 |
| RMSNorm | 减均值改为缩放 | LLaMA 等大模型 |

### 4.4 Pre-Norm vs Post-Norm

Transformer 中归一化层的位置：

**Post-Norm**（原始论文）：
$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Pre-Norm**（现代实践）：
$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

Pre-Norm 训练更稳定，梯度流动更好，是现代大模型的标准选择。

---

## 5. 数据增强

### 5.1 图像增强

| 方法 | 说明 |
|------|------|
| 随机裁剪 | ResNet 的标配 |
| 水平/垂直翻转 | 简单有效 |
| 颜色抖动 | 亮度、对比度、饱和度 |
| 随机擦除 (Cutout) | 随机遮挡区域 |
| Mixup | 两幅图像线性混合 |
| CutMix | 将一幅图像的区域粘贴到另一幅 |
| RandAugment | 自动搜索增强策略 |
| AutoAugment | RL 搜索增强策略 |

### 5.2 Mixup

$$\tilde{x} = \lambda x_i + (1 - \lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1 - \lambda) y_j$$

其中 $\lambda \sim \text{Beta}(\alpha, \alpha)$，通常 $\alpha = 0.2$。

**效果**：鼓励模型在样本之间线性插值，提高鲁棒性和泛化能力。

### 5.3 CutMix

将图像 A 的随机区域替换为图像 B 的对应区域，标签按面积比例混合。

### 5.4 文本增强

| 方法 | 说明 |
|------|------|
| 回译 (Back Translation) | 翻译到中间语言再翻译回来 |
| 同义词替换 | 随机替换同义词 |
| 随机删除/交换 | EDA (Easy Data Augmentation) |
| 对抗训练 | 添加对抗扰动 |

---

## 6. 早停法 (Early Stopping)

### 6.1 原理

在验证集性能不再提升时停止训练，防止过拟合。

### 6.2 实现

```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    train()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

### 6.3 与 L2 正则化的等价性

在一定条件下，早停等价于 L2 正则化：两者都限制了参数空间中的有效搜索半径。

---

## 7. 标签平滑 (Label Smoothing)

### 7.1 动机

One-hot 标签过于自信（100% 概率），可能导致模型过度自信。

### 7.2 方法

将 one-hot 标签 $y$ 替换为平滑标签 $\tilde{y}$：

$$\tilde{y}_i = \begin{cases} 1 - \epsilon + \epsilon/K & \text{if } i = y \\ \epsilon/K & \text{otherwise} \end{cases}$$

其中 $\epsilon$ 通常为 0.1。

**效果**：
- 防止模型对训练标签过度自信
- 提高泛化能力
- 在 Transformer 训练中广泛使用

---

## 8. 正则化策略选择

### 8.1 按架构推荐

| 架构 | 推荐正则化 | 说明 |
|------|-----------|------|
| CNN (ResNet) | BN + Weight Decay + 数据增强 | BN 已有正则化效果 |
| Transformer | Pre-LN + Dropout(0.1) + Label Smoothing(0.1) + Weight Decay | 大模型标配 |
| ViT | DropPath + LayerNorm + 强数据增强 | DropPath 替代 Dropout |
| GAN | SpectralNorm + 数据增强 | 判别器需 Lipshitz 约束 |
| LSTM | Dropout(层间) + BN/LN | 不在循环连接上使用 Dropout |

### 8.2 调参建议

1. 从简单开始：先只用 Weight Decay + BN/LN，逐步添加
2. 密切关注验证集曲线，判断是欠拟合还是过拟合
3. 数据增强几乎总是有效，优先尝试
4. 大模型（>1B）通常不需要强正则化，数据量足够大就是最好的正则化

---

## 9. 参考文献

[1] Srivastava, N., et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 2014.

[2] Ioffe, S. & Szegedy, C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML, 2015.

[3] Ba, J. L., et al. Layer Normalization. arXiv:1607.06450, 2016.

[4] Zhang, H., et al. mixup: Beyond Empirical Risk Minimization. ICLR, 2018.

[5] Yun, S., et al. CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. ICCV, 2019.

[6] Szegedy, C., et al. Rethinking the Inception Architecture for Computer Vision. CVPR, 2016. (Label Smoothing)

[7] Huang, G., et al. Deep Networks with Stochastic Depth. ECCV, 2016.

[8] Wu, Y. & He, K. Group Normalization. ECCV, 2018.

[9] Zhang, B. & Sennrich, R. Root Mean Square Layer Normalization. NeurIPS, 2019. (RMSNorm)
