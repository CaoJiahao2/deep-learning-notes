# 卷积神经网络 (CNN) 详解

> 从 LeNet 到 EfficientNet 的 CNN 架构演进全景。

---

## 目录

1. [基本概念](#1-基本概念)
2. [核心组件](#2-核心组件)
3. [经典架构演进](#3-经典架构演进)
4. [现代设计原则](#4-现代设计原则)
5. [CNN 在 CV 中的应用](#5-cnn-在-cv-中的应用)
6. [参考文献](#6-参考文献)

---

## 1. 基本概念

### 1.1 为什么需要 CNN

全连接网络处理图像时面临两个核心问题：
- **参数爆炸**：一张 224×224×3 的图像，第一层 1000 个神经元就需要 1.5 亿个参数
- **空间信息丢失**：全连接层将图像展平为一维向量，破坏了像素间的空间结构

CNN 通过**局部连接**和**权重共享**两大设计原则解决了这些问题。

### 1.2 核心思想

- **局部感受野（Local Receptive Field）**：每个神经元只连接输入的一个局部区域
- **权重共享（Weight Sharing）**：同一卷积核在输入的所有位置共享参数
- **平移等变性（Translation Equivariance）**：输入平移，输出也相应平移

---

## 2. 核心组件

### 2.1 卷积层（Convolutional Layer）

**二维卷积运算**：

$$(I * K)_{i,j} = \sum_{m=0}^{k_h-1}\sum_{n=0}^{k_w-1} I_{i+m, j+n} \cdot K_{m,n}$$

**输出尺寸计算**：

$$H_{out} = \left\lfloor\frac{H_{in} + 2P - k_h}{S}\right\rfloor + 1$$

其中：
- $H_{in}$：输入高度
- $k_h$：卷积核高度
- $P$：填充（Padding）
- $S$：步长（Stride）

**参数数量**：$k_h \times k_w \times C_{in} \times C_{out} + C_{out}$（包含偏置）

**计算量 (FLOPs)**：$H_{out} \times W_{out} \times k_h \times k_w \times C_{in} \times C_{out}$

### 2.2 池化层（Pooling Layer）

| 类型 | 操作 | 特点 |
|------|------|------|
| Max Pooling | 取局部区域最大值 | 保留纹理信息，默认选择 |
| Average Pooling | 取局部区域平均值 | 保留背景信息 |
| Global Average Pooling | 全局平均池化 | 替代全连接层，减少过拟合 |

**作用**：
- 降采样，减少计算量
- 增大感受野
- 提供局部平移不变性

### 2.3 激活函数

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| ReLU | $f(x) = \max(0, x)$ | 简单高效，缓解梯度消失 |
| Leaky ReLU | $f(x) = \max(0.01x, x)$ | 解决 ReLU 的 "Dead Neuron" 问题 |
| GELU | $f(x) = x \cdot \Phi(x)$ | 现代架构首选，平滑近似 |
| SiLU/Swish | $f(x) = x \cdot \sigma(x)$ | EfficientNet 中使用 |

### 2.4 全连接层（Fully Connected Layer）

通常放在 CNN 末端，用于分类或回归输出。现代架构（ResNet 等）倾向于用 Global Average Pooling 替代。

---

## 3. 经典架构演进

### 3.1 LeNet-5 (1998)

```
Input(32×32) → Conv(5×5, 6) → AvgPool(2×2) → Conv(5×5, 16) → AvgPool(2×2)
→ FC(120) → FC(84) → FC(10)
```

**贡献**：开创了 CNN 的基本范式（卷积 → 池化 → 全连接）

### 3.2 AlexNet (2012)

```
Input(224×224×3) → Conv(11×11, 96) → MaxPool → Conv(5×5, 256) → MaxPool
→ Conv(3×3, 384) → Conv(3×3, 384) → Conv(3×3, 256) → MaxPool
→ FC(4096) → FC(4096) → FC(1000)
```

**关键贡献**：
- 首次使用 ReLU 激活函数，加速训练
- 使用 Dropout（0.5）防止过拟合
- 数据增强（随机裁剪、水平翻转）
- 双 GPU 并行训练

### 3.3 VGG (2014)

**设计哲学**：使用小卷积核（3×3）堆叠替代大卷积核，在相同感受野下减少参数。

- 两个 3×3 卷积的感受野 = 一个 5×5 卷积
- 三个 3×3 卷积的感受野 = 一个 7×7 卷积

**VGG16 结构**：
```
2×Conv(3×3, 64)  → MaxPool
2×Conv(3×3, 128) → MaxPool
3×Conv(3×3, 256) → MaxPool
3×Conv(3×3, 512) → MaxPool
3×Conv(3×3, 512) → MaxPool
FC(4096) → FC(4096) → FC(1000)
```

### 3.4 GoogLeNet / Inception (2014)

**Inception Module**：在同一层并行使用不同大小的卷积核（1×1, 3×3, 5×5）和池化，然后拼接。

**1×1 卷积的降维作用**：在 3×3 和 5×5 卷积前使用 1×1 卷积减少通道数，大幅降低计算量。

**Inception v3 的改进**：
- 将大卷积核分解为小卷积核（5×5 → 两个 3×3）
- 进一步分解为 n×1 + 1×n 非对称卷积
- 使用 Batch Normalization
- 使用 Label Smoothing

### 3.5 ResNet (2015) — 里程碑

**核心问题**：随着网络加深，训练误差反而升高（退化问题，不是过拟合）。

**残差学习**：

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

其中 $\mathcal{F}$ 是残差函数，$\mathbf{x}$ 是恒等映射（shortcut connection）。

**两种残差块**：
- **BasicBlock**：两个 3×3 卷积，用于 ResNet-18/34
- **Bottleneck**：1×1 → 3×3 → 1×1，首尾 1×1 分别压缩和恢复通道，用于 ResNet-50/101/152

**ResNet-50 结构**：
```
Conv(7×7, 64) → MaxPool
[3× Bottleneck(64→256)] × 3
[4× Bottleneck(128→512)] × 4
[6× Bottleneck(256→1024)] × 6
[3× Bottleneck(512→2048)] × 3
AvgPool → FC(1000)
```

**为什么 ResNet 有效**：
- 残差连接使梯度可以通过 shortcut 直接传播，解决了深层网络的梯度消失
- 恒等映射使网络至少不会比浅层网络更差
- 等效于浅层网络和深层网络的集成（Unraveled View）

### 3.6 DenseNet (2017)

**密集连接**：每一层都接收前面所有层的特征图：

$$x_l = H_l([x_0, x_1, \ldots, x_{l-1}])$$

**优点**：特征复用、梯度流动更好、参数量更少。缺点是显存占用大。

### 3.7 SENet (2018) — 通道注意力

**Squeeze-and-Excitation Block**：

1. **Squeeze**：Global Average Pooling → 通道描述符
2. **Excitation**：两个 FC 层 → 通道权重
3. **Scale**：将权重乘回原特征图

$$\mathbf{z} = \text{GAP}(\mathbf{X}), \quad \mathbf{s} = \sigma(\mathbf{W}_2\delta(\mathbf{W}_1\mathbf{z})), \quad \tilde{\mathbf{X}}_c = s_c \cdot \mathbf{X}_c$$

### 3.8 EfficientNet (2019) — 模型缩放

**核心思想**：同时缩放深度、宽度和分辨率，而非只调整一个维度。

$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$

约束条件：$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$，$\phi$ 控制总缩放量。

**EfficientNet-B0 的 MBConv Block**：
- 深度可分离卷积 + SE 模块
- 使用 Swish 激活函数
- 使用 DropConnect 正则化

### 3.9 ConvNeXt (2022) — 现代化 CNN

ConvNeXt 借鉴 Vision Transformer 的设计理念"现代化"了 ResNet：

| 改进 | 说明 |
|------|------|
| 训练策略优化 | 使用 AdamW、更长的训练周期、更强的数据增强 |
| 阶段比例调整 | 从 (3,4,6,3) 改为 (3,3,9,3) |
| Patchify Stem | 用 4×4 stride 4 的卷积替代 7×7 stride 2 |
| 深度可分离卷积 | 用分组卷积替代传统卷积 |
| 倒置瓶颈 | 将 ResNet bottleneck 的宽-窄-宽改为窄-宽-窄 |
| 更大的卷积核 | 将 3×3 改为 7×7 |
| 更少的归一化 | 只用 LayerNorm 替代 BatchNorm |

---

## 4. 现代设计原则

### 4.1 卷积核选择

- 堆叠小卷积核（3×3）替代大卷积核，减少参数
- 1×1 卷积用于通道变换和降维
- 深度可分离卷积（Depthwise + Pointwise）大幅减少计算量

### 4.2 深度可分离卷积

将标准卷积分解为两步：

**标准卷积**：$k_h \times k_w \times C_{in} \times C_{out}$

**深度可分离卷积**：
1. Depthwise Conv：$k_h \times k_w \times C_{in}$（每个通道独立卷积）
2. Pointwise Conv：$1 \times 1 \times C_{in} \times C_{out}$（1×1 卷积混合通道）

计算量比值：$\frac{1}{C_{out}} + \frac{1}{k_h \times k_w}$

### 4.3 归一化层选择

| 归一化 | 归一化维度 | 适用场景 |
|--------|-----------|---------|
| BatchNorm | (N, H, W) | CNN 标配，小 batch 不稳定 |
| LayerNorm | (C, H, W) | Transformer 标配 |
| GroupNorm | 分组通道 | 小 batch 训练，分割任务 |
| InstanceNorm | 单通道 | 风格迁移 |

### 4.4 感受野计算

$$RF_l = RF_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} S_i$$

其中 $RF_l$ 是第 $l$ 层的感受野，$k_l$ 是卷积核大小，$S_i$ 是第 $i$ 层的步长。

---

## 5. CNN 在 CV 中的应用

### 5.1 图像分类
- 基准网络：ResNet, EfficientNet, ConvNeXt
- 常用数据集：ImageNet-1K (1.28M), ImageNet-21K (14M)

### 5.2 目标检测
- **Two-Stage**：Faster R-CNN, Mask R-CNN
- **One-Stage**：YOLO, SSD, RetinaNet
- **Anchor-Free**：CenterNet, FCOS
- 常用 Backbone：ResNet + FPN

### 5.3 语义分割
- **FCN**：全卷积网络，首个端到端分割网络
- **U-Net**：编码器-解码器 + 跳跃连接，医学图像标配
- **DeepLab 系列**：空洞卷积 + ASPP
- **SegFormer**：Transformer + CNN 混合

### 5.4 实例分割
- Mask R-CNN：在 Faster R-CNN 上加分割分支
- YOLACT：实时实例分割

---

## 6. 参考文献

[1] LeCun, Y., et al. Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE, 1998.

[2] Krizhevsky, A., et al. ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS, 2012.

[3] Simonyan, K. & Zisserman, A. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR, 2015.

[4] Szegedy, C., et al. Going Deeper with Convolutions. CVPR, 2015.

[5] He, K., et al. Deep Residual Learning for Image Recognition. CVPR, 2016.

[6] Huang, G., et al. Densely Connected Convolutional Networks. CVPR, 2017.

[7] Hu, J., et al. Squeeze-and-Excitation Networks. CVPR, 2018.

[8] Tan, M. & Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.

[9] Liu, Z., et al. A ConvNet for the 2020s. CVPR, 2022.

[10] Howard, A. G., et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861, 2017.
