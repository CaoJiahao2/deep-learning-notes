# 生成对抗网络 (GAN) 详解

> 从 Vanilla GAN 到 StyleGAN 的生成模型演进。

---

## 目录

1. [核心思想](#1)
2. [数学原理](#2)
3. [训练过程与技巧](#3)
4. [经典变体](#4)
5. [条件 GAN 与可控生成](#5-gan)
6. [评估指标](#6)
7. [GAN vs Diffusion Model](#7-gan-vs-diffusion-model)
8. [参考文献](#8)

---

## 1. 核心思想

### 1.1 对抗博弈

GAN (Goodfellow et al., 2014) 由两个网络组成，通过对抗训练相互提升：

- **生成器（Generator）** $G$：将随机噪声 $z$ 映射为逼真的数据样本 $G(z)$
- **判别器（Discriminator）** $D$：判断输入是真实数据还是生成器产生的假数据

类比：**造假者（G）** 试图制造逼真的假画，**鉴定师（D）** 试图区分真画和假画。两者在不断博弈中共同进步。

### 1.2 损失函数

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

- **判别器目标**：最大化 $V(D,G)$，即对真实数据输出高值，对生成数据输出低值
- **生成器目标**：最小化 $V(D,G)$，即让判别器无法区分生成数据

---

## 2. 数学原理

### 2.1 最优判别器

对于固定生成器 $G$，最优判别器为：

$$D_G^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

### 2.2 全局最优解

当 $p_g = p_{data}$ 时，$D_G^*(x) = \frac{1}{2}$，$V(D, G)$ 达到全局最小值 $-\log 4$。

### 2.3 与 Jensen-Shannon 散度的关系

$$\max_D V(D, G) = -\log 4 + 2 \cdot \text{JSD}(p_{data} \parallel p_g)$$

GAN 的优化目标等价于最小化 $p_{data}$ 和 $p_g$ 之间的 JS 散度。

### 2.4 梯度消失问题

当判别器训练得太好时，生成器的梯度趋近于 0：

$$\nabla_{\theta_g} \mathbb{E}_{z}[\log(1 - D(G(z)))] \approx 0$$

**Non-saturating Loss**（实用技巧）：改为最大化 $\log D(G(z))$，提供更强的梯度信号。

$$\max_G \mathbb{E}_{z \sim p_z(z)}[\log D(G(z))]$$

---

## 3. 训练过程与技巧

### 3.1 交替训练

```
for 每个训练迭代：
    # 训练判别器（k 步）
    for step in range(k):
        采样 m 个真实样本 {x₁, ..., xₘ}
        采样 m 个噪声向量 {z₁, ..., zₘ}
        生成假样本 {G(z₁), ..., G(zₘ)}
        更新 D 以最大化：1/m Σ[log D(xᵢ) + log(1 - D(G(zᵢ)))]
    
    # 训练生成器（1 步）
    采样 m 个噪声向量 {z₁, ..., zₘ}
    更新 G 以最大化：1/m Σ log D(G(zᵢ))
```

通常 $k=1$（交替更新一次），某些情况下 $k>1$ 使判别器更强。

### 3.2 训练技巧

| 技巧 | 说明 |
|------|------|
| One-sided Label Smoothing | 将真实标签从 1.0 平滑到 0.9，防止判别器过度自信 |
| Noisy Labels | 偶尔翻转标签，引入随机性 |
| Feature Matching | 让生成器匹配判别器中间层的特征统计 |
| Minibatch Discrimination | 让判别器考虑整个 batch 的统计信息 |
| Spectral Normalization | 对判别器权重做谱归一化，保证 Lipschitz 约束 |
| Gradient Penalty | 对判别器梯度添加惩罚，实现 Lipschitz 约束 |

### 3.3 常见问题

| 问题 | 表现 | 解决方案 |
|------|------|---------|
| Mode Collapse | 生成器只产生少数几种样本 | Minibatch Discrimination, Unrolled GAN |
| 训练不稳定 | 损失震荡剧烈 | WGAN-GP, Spectral Normalization |
| 梯度消失 | 生成器 loss 不下降 | Non-saturating loss, WGAN |
| 判别器过强 | D 的准确率接近 100% | 减少 D 的训练步数，添加噪声 |

---

## 4. 经典变体

### 4.1 DCGAN (2015)

深度卷积 GAN，将 CNN 引入 GAN 架构。

**架构指南**：
- 判别器：用 strided convolution 替代 pooling
- 生成器：用 transposed convolution 上采样
- 使用 BatchNorm（G 和 D 都使用）
- 生成器用 ReLU（输出层用 Tanh），判别器用 LeakyReLU

### 4.2 WGAN / WGAN-GP (2017)

**问题**：JS 散度在 $p_{data}$ 和 $p_g$ 不重叠时梯度消失。

**Wasserstein 距离**：

$$W(p_{data}, p_g) = \inf_{\gamma \sim \Pi(p_{data}, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

直观理解：将分布 $p_{data}$ 的"土"搬运到 $p_g$ 的最小成本。

**Kantorovich-Rubinstein 对偶**：

$$W(p_{data}, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_{data}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]$$

其中 $f$ 是 1-Lipschitz 函数。

**WGAN 损失**：

$$\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

**Lipschitz 约束**：
- WGAN：权重裁剪（Weight Clipping），简单但粗暴
- WGAN-GP：梯度惩罚（Gradient Penalty），更优雅

$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

### 4.3 StyleGAN / StyleGAN2 / StyleGAN3 (2019-2021)

**核心创新**：
- **映射网络**：将噪声 $z$ 映射到中间潜空间 $w$（$\mathcal{W}$ 空间）
- **AdaIN (Adaptive Instance Normalization)**：通过 $w$ 调制每一层的风格
- **噪声注入**：在每层卷积后添加随机噪声，控制随机细节

**StyleGAN2 改进**：
- 移除 AdaIN 中的水滴伪影（通过权重解调替代）
- 路径长度正则化，提高潜空间平滑性

**StyleGAN3 改进**：
- 解决纹理粘连（Texture Sticking）问题
- 实现平移和旋转等变性

### 4.4 BigGAN (2019)

大规模条件 GAN：
- 大 batch size（2048）+ 大通道数
- 使用自注意力模块
- 正交正则化

### 4.5 CycleGAN (2017)

无配对图像到图像翻译：
- 通过**循环一致性损失**实现无监督学习
- $G: X \to Y$ 和 $F: Y \to X$ 两个生成器
- $\mathcal{L}_{cyc} = \mathbb{E}_{x}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y}[\|G(F(y)) - y\|_1]$

### 4.6 Pix2Pix (2017)

配对图像到图像翻译：
- 使用 U-Net 作为生成器
- 使用 PatchGAN 作为判别器
- 结合 L1 损失和对抗损失

---

## 5. 条件 GAN 与可控生成

### 5.1 条件 GAN (cGAN)

将条件信息 $c$（如类别标签）输入生成器和判别器：

$$\min_G \max_D \mathbb{E}_{x \sim p_{data}}[\log D(x|c)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z|c)))]$$

### 5.2 常用条件注入方式

| 方式 | 说明 |
|------|------|
| 拼接 | 将条件向量拼接到输入或中间特征 |
| 条件 BatchNorm | 每类条件使用不同的 $\gamma, \beta$ |
| AdaIN | 通过条件信息调制特征图的均值和方差 |
| 交叉注意力 | 将条件信息通过 Cross-Attention 注入 |

---

## 6. 评估指标

### 6.1 Inception Score (IS)

$$\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g}[D_{KL}(p(y|x) \parallel p(y))]\right)$$

- 好：生成样本类别清晰（$p(y|x)$ 低熵）、类别多样（$p(y)$ 高熵）
- 局限：不考虑真实分布，可能被对抗样本欺骗

### 6.2 Fréchet Inception Distance (FID)

$$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

- 比较真实图像和生成图像在 Inception 特征空间的分布距离
- 越低越好，更符合人类感知
- 对 Mode Collapse 敏感

### 6.3 其他指标

| 指标 | 说明 |
|------|------|
| Precision & Recall | 分别衡量生成质量和多样性 |
| KID (Kernel Inception Distance) | FID 的无偏替代 |
| LPIPS | 基于深度特征的感知相似度 |

---

## 7. GAN vs Diffusion Model

### 7.1 对比

| 维度 | GAN | Diffusion Model |
|------|-----|-----------------|
| 生成质量 | 高（但易 Mode Collapse） | 极高 |
| 多样性 | Mode Collapse 风险 | 好 |
| 训练稳定性 | 不稳定 | 稳定 |
| 推理速度 | 快（单次前向传播） | 慢（多步去噪） |
| 可控性 | 好（StyleGAN 潜空间编辑） | 好（条件控制） |
| 计算成本 | 低 | 高 |

### 7.2 现状

- Diffusion Model 在图像生成质量上已超越 GAN
- GAN 在需要快速推理的场景（实时生成、视频）仍有优势
- GAN 的潜空间编辑能力（StyleGAN）仍被广泛使用

---

## 8. 参考文献

[1] Goodfellow, I., et al. Generative Adversarial Nets. NeurIPS, 2014.

[2] Radford, A., et al. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. ICLR, 2016.

[3] Arjovsky, M., et al. Wasserstein GAN. ICML, 2017.

[4] Gulrajani, I., et al. Improved Training of Wasserstein GANs. NeurIPS, 2017.

[5] Karras, T., et al. A Style-Based Generator Architecture for Generative Adversarial Networks. CVPR, 2019.

[6] Karras, T., et al. Analyzing and Improving the Image Quality of StyleGAN. CVPR, 2020.

[7] Brock, A., et al. Large Scale GAN Training for High Fidelity Natural Image Synthesis. ICLR, 2019.

[8] Zhu, J.-Y., et al. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV, 2017.

[9] Isola, P., et al. Image-to-Image Translation with Conditional Adversarial Networks. CVPR, 2017.

[10] Mirza, M. & Osindero, S. Conditional Generative Adversarial Nets. arXiv:1411.1784, 2014.

[11] Heusel, M., et al. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. NeurIPS, 2017.
