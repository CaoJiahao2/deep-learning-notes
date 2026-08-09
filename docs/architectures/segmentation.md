# 图像分割详解

> 图像分割将像素级理解从「框」推进到像素级掩码：语义分割为每个像素分配类别，实例分割区分同类不同个体，全景分割统一两者。

---

## 目录

1. [概述](#1)
2. [语义分割](#2)
3. [实例分割](#3)
4. [损失函数](#4)
5. [训练策略与数据](#5)
6. [参考文献](#6)

---

## 1. 概述

### 1.1 三大分割任务

图像分割（Image Segmentation）是计算机视觉中的核心任务之一，旨在将图像划分为具有语义意义的区域。根据任务粒度的不同，分割可分为三个层次：

| 任务 | 定义 | 是否区分同类个体 | 覆盖范围 |
|------|------|:---:|:---:|
| **语义分割（Semantic Segmentation）** | 为每个像素分配类别标签 | 否 | 全图 |
| **实例分割（Instance Segmentation）** | 在检测实例的基础上预测掩码 | 是 | 前景物体 |
| **全景分割（Panoptic Segmentation）** | 语义 + 实例统一，每像素唯一标签 | 是 | 全图 |

- **语义分割**：将图像中所有像素分类为「天空」「道路」「行人」等，但画面中两个行人会被标记为同一类别，无法区分个体。
- **实例分割**：在目标检测的基础上，为每个检测到的实例预测一个二值掩码，能区分同类不同个体（如 Mask R-CNN）。
- **全景分割**：对「stuff」（背景区域，如天空、道路）做语义分割，对「things」（前景物体，如行人、车辆）做实例分割，合并为每个像素唯一的类别与实例 ID。

### 1.2 分割 vs 检测 vs 分类

理解分割与相关任务的区别，有助于精确把握问题的定义：

| 任务 | 输出粒度 | 典型输出 | 关键挑战 |
|------|----------|----------|----------|
| 图像分类 | 图像级 | 单个标签 | 全局语义理解 |
| 目标检测 | 框级 | 边界框 + 类别 | 定位 + 多实例 |
| 语义分割 | 像素级 | 逐像素类别图 | 边界精确定位 |
| 实例分割 | 像素级 + 实例级 | 逐实例掩码 + 类别 | 检测 + 分割联合 |

分割是计算机视觉中粒度最细的任务之一，也是通往场景理解的关键一步。

### 1.3 核心评估指标

**mIoU（Mean Intersection over Union）**，是分割任务最主流的评估指标。对每个类别计算预测区域与真实区域的交并比，再对所有类别求平均：

$$\text{mIoU} = \frac{1}{C}\sum_{c=1}^{C}\frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$

其中 $C$ 为类别数，$\text{TP}_c$、$\text{FP}_c$、$\text{FN}_c$ 分别为第 $c$ 类的真正例、假正例、假负例（像素级别）。

**Dice 系数（F1-Score 的变体）**，常用于医学影像分割，衡量预测掩码与真实掩码的重叠程度：

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

其中 $A$ 为预测区域，$B$ 为真实区域。Dice 系数取值范围为 $[0, 1]$，越接近 1 表示分割越准确。

**全景分割评估指标 PQ（Panoptic Quality）**：将语义分割与实例分割统一评估，分为识别质量（RQ）和分割质量（SQ）两部分：

$$\text{PQ} = \underbrace{\frac{\sum_{(p,g)\in TP}\text{IoU}(p,g)}{|TP|}}_{\text{SQ}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{RQ}}$$

---

## 2. 语义分割

### 2.1 FCN：全卷积网络的端到端分割

**FCN（Fully Convolutional Network）** [Long et al., 2015] 是语义分割的开山之作，核心思想是将分类网络（如 VGG）的全连接层替换为 $1 \times 1$ 卷积，实现端到端的像素级分类。

**关键创新**：

- **全卷积化**：去掉了全连接层，可以接受任意尺寸的输入，输出一张大小一致的分割图。
- **上采样（Upsampling）**：通过转置卷积（Transposed Convolution）或双线性插值将低分辨率特征图恢复到原图分辨率。
- **跳跃连接（Skip Connection）**：融合浅层精细空间信息与深层语义信息，形成 FCN-32s、FCN-16s、FCN-8s 三种变体，数字越小融合的浅层特征越多，边缘越精细。

```
FCN-32s: 输入 → conv → pool×5 → conv7(1×1) → 32×上采样 → 输出
FCN-16s: 输入 → ... → pool4 + (2×上采样 conv7) → 16×上采样 → 输出
FCN-8s:  输入 → ... → pool3 + pool4 + conv7 → 8×上采样 → 输出
```

**局限性**：上采样结果较为粗糙，对小物体和边缘细节的分割效果不佳，但其端到端训练范式为后续所有工作奠定了基础。

### 2.2 U-Net：编码器-解码器 + 跳跃连接

**U-Net** [Ronneberger et al., 2015] 提出了对称的编码器-解码器架构，是医学影像分割的标配方法。

**架构示意**：

```text
编码器（下采样）: 输入 → Conv×2 → MaxPool↓ → Conv×2 → MaxPool↓ → Conv×2 → MaxPool↓ → Conv×2 → MaxPool↓ → Conv×2
                         │跳跃连接          │跳跃连接          │跳跃连接          │跳跃连接
解码器（上采样）: 分割图 ← Conv×2 ← ↑拼接 ← Conv×2 ← ↑拼接 ← Conv×2 ← ↑拼接 ← Conv×2 ← ↑拼接 ← Conv×2
```

**核心设计**：

- **对称结构**：编码器逐步下采样提取高维语义，解码器逐步上采样恢复空间分辨率，形成 U 形对称结构。原始 U-Net 为 4 次下采样，共 5 个层级，每层卷积核数翻倍（64→128→256→512→1024）。
- **跳跃连接**：每个编码器层的输出直接拼接到对应解码器层，将浅层高分辨率细节与深层语义信息融合，极大改善了分割边界的精度。具体而言，第 $i$ 层解码器的输入为上层上采样特征与编码器第 $i$ 层特征的拼接。
- **重叠策略**：使用 Overlap-tile 策略处理大图，边缘区域利用镜像填充来预测，避免边界伪影。
- **损失函数**：原始 U-Net 使用加权交叉熵，为靠近边界的分割区域赋予更大权重，强制模型关注细胞边界：

$$\mathcal{L}_{\text{U-Net}} = -\sum_{\mathbf{x} \in \Omega} w(\mathbf{x}) \log(p_{\ell(\mathbf{x})}(\mathbf{x}))$$

其中 $w(\mathbf{x})$ 为像素权重图，边界像素权重更高，$p_{\ell(\mathbf{x})}(\mathbf{x})$ 为像素 $\mathbf{x}$ 属于真实类别 $\ell(\mathbf{x})$ 的预测概率。

**U-Net 变体**：

- **U-Net++**：引入密集跳跃连接和深监督，在编码器-解码器之间添加嵌套的卷积块，实现更灵活的特征融合。
- **Attention U-Net**：在跳跃连接前加入注意力门控（Attention Gate），让解码器自动学习关注编码器特征中的相关区域，抑制不相关背景。
- **nnU-Net**：自动化 U-Net 配置，根据数据集特性自适应调整网络结构、预处理和后处理，在多个医学分割基准上取得了 SOTA 性能。

U-Net 虽然简洁，但凭借跳跃连接的高效特征融合，在小样本、高分辨率的医学影像场景中表现尤为突出，至今仍是医学分割的基准模型。其设计理念被广泛应用于后续的 SegNet、UNet 3+ 等工作中。

### 2.3 DeepLab 系列：空洞卷积与多尺度上下文

**DeepLab** 系列（v1/v2/v3/v3+）由 Google 提出，核心贡献在于引入空洞卷积和 ASPP 模块来捕获多尺度上下文。

**空洞卷积（Dilated / Atrous Convolution）**：

标准卷积的感受野与卷积核大小成正比。空洞卷积通过在卷积核元素之间插入空洞（即填充 0），在不增加参数量的前提下指数级扩大感受野，同时保持特征图分辨率不变。

对于空洞率为 $d$ 的 $k \times k$ 卷积，其等效感受野为 $k' = k + (k-1)(d-1)$。例如，$3 \times 3$ 空洞卷积在 $d=2$ 时等效于 $5 \times 5$ 卷积的感受野，参数量却仅为 $3 \times 3 = 9$。

输出尺寸公式：

$$H_{out} = \left\lfloor\frac{H_{in} + 2P - d \cdot (k-1) - 1}{S}\right\rfloor + 1$$

其中 $d$ 为空洞率（dilation rate），$d=1$ 退化为标准卷积。DeepLab v2 使用空洞率为 6、12、18、24 的并行空洞卷积来捕获多尺度上下文。

**感受野计算**：对于连续卷积层，感受野递推公式为：

$$r_{i} = r_{i-1} + (k_i - 1) \times \prod_{j=1}^{i-1} s_j$$

其中 $r_i$ 为第 $i$ 层的感受野，$k_i$ 为卷积核大小，$s_j$ 为步长。通过堆叠空洞卷积，DeepLab 可以在不降分辨率的情况下获得极大的感受野，这对准确分割大物体至关重要。

**ASPP（Atrous Spatial Pyramid Pooling）**：

ASPP 模块并行使用多个不同空洞率的空洞卷积（以及全局平均池化分支），在多个尺度上捕获上下文信息，然后拼接融合：

```text
输入特征图
  ├── 1×1 卷积
  ├── 3×3 空洞卷积 (rate=6)
  ├── 3×3 空洞卷积 (rate=12)
  ├── 3×3 空洞卷积 (rate=18)
  └── 全局平均池化 → 1×1 卷积 → 上采样
           ↓
        拼接 → 1×1 卷积 → 输出
```

**各版本演进**：

| 版本 | 关键改进 |
|------|----------|
| DeepLab v1 | 空洞卷积 + 全连接 CRF 后处理 |
| DeepLab v2 | ASPP 模块，多尺度空洞卷积 |
| DeepLab v3 | 改进 ASPP（加入全局池化分支 + BN），去掉 CRF |
| DeepLab v3+ | 编码器-解码器结构，Xception 骨干，深度可分离卷积 |

### 2.4 SegFormer：分层 Transformer 编码 + MLP 解码

**SegFormer** [Xie et al., 2021] 将 Transformer 引入语义分割，提出了一个简洁高效的分层分割框架。

**架构设计**：

- **分层 Transformer 编码器（MiT）**：类似 CNN 的金字塔结构，输出 4 个不同分辨率的特征图（1/4, 1/8, 1/16, 1/32），兼顾局部细节与全局语义。
- **轻量 All-MLP 解码器**：仅用 MLP 层融合多尺度特征，不涉及卷积或注意力。具体步骤：
  1. 各层特征通过 MLP 统一通道维度
  2. 上采样到 1/4 分辨率
  3. 拼接后通过 MLP 融合
  4. 最后经 MLP 输出分割结果

SegFormer 的设计实现了强大的性能与极高的推理效率，后续版本 SegFormer-b5 在 ADE20K 上取得了当时最优的语义分割结果。

### 2.5 关键设计选择

语义分割模型的设计中，有几个贯穿始终的关键选择：

**上采样方式**：

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 双线性插值 | 邻近像素加权平均 | 简单、无参数 | 不可学习，结果模糊 |
| 转置卷积 | 可学习的上采样卷积 | 可学习，灵活性高 | 容易产生棋盘伪影 |
| 像素重排（Pixel Shuffle） | 通道→空间重排 | 高效，无棋盘伪影 | 仅适用于特定倍率 |

**跳跃连接策略**：FCN 使用加法融合（求和），U-Net 使用拼接融合（通道拼接）。拼接保留了更丰富的特征信息，但增加了参数量。加法更轻量，但对特征对齐要求更高。

**多尺度上下文建模**：大物体需要大感受野，小物体需要高分辨率特征。主流方法包括：
- 空洞卷积金字塔（DeepLab ASPP）
- 特征金字塔网络（FPN）
- 多尺度输入融合（Inception 风格）
- 自注意力机制（Transformer 风格）

---

## 3. 实例分割

### 3.1 Mask R-CNN：检测 + 掩码分支

**Mask R-CNN** [He et al., 2017] 在 Faster R-CNN 的基础上增加了一个掩码预测分支，以极小的额外开销将目标检测扩展为实例分割。

**核心组件**：

- **RoIAlign**：Faster R-CNN 中的 RoIPool 在量化过程中引入了空间不对齐问题。RoIAlign 使用双线性插值，避免了量化取整，保存了精确的空间对应关系，对像素级掩码预测至关重要。
- **掩码分支（Mask Branch）**：一个轻量 FCN，在检测分支之外为每个 RoI 预测一个 $28 \times 28$ 的二值掩码。掩码分支与分类、回归分支并行，共享 RoI 特征。
- **多任务损失**：

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}$$

其中 $\mathcal{L}_{\text{mask}}$ 为逐像素 Sigmoid + 二值交叉熵，每个类别独立预测掩码（避免类间竞争）。

Mask R-CNN 是实例分割的标杆方法，其架构简洁、精度高，后续大量工作在此基础上改进。

### 3.2 Mask2Former：掩码分类统一范式

**Mask2Former** [Cheng et al., 2022] 提出掩码分类（Mask Classification）统一架构，可同时处理语义、实例和全景分割。

**核心思想**：与传统「逐像素分类」范式不同，掩码分类将分割定义为学习一组「掩码预测 + 类别预测」的对。每个 query 对应一个可能的掩码，通过 Transformer Decoder 迭代优化。

**关键设计**：

- **掩码注意力（Masked Attention）**：标准交叉注意力在全局操作，计算量大且引入噪声。掩码注意力将注意力限制在预测掩码的高响应区域，既减少了计算开销，又加速了收敛。
- **多尺度特征**：从骨干网络提取多尺度特征图，送入逐层的 Transformer Decoder 进行逐步细化。
- **统一损失**：使用匈牙利匹配进行二部图匹配，结合分类损失（CE）和掩码损失（Dice + BCE）。

Mask2Former 在各大分割基准上均取得了 SOTA 性能，同时保持架构简洁、训练高效。

### 3.3 SAM：Segment Anything Model

**SAM（Segment Anything Model）** [Kirillov et al., 2023] 是 Meta 提出的提示式通用分割基础模型，开创了分割的「基础模型」时代。

**核心设计理念**：

- **提示式分割（Promptable Segmentation）**：用户通过点、框、掩码或文本提示来指定分割目标，模型输出对应的掩码。同一个图像编码器只需提取一次特征，即可支持任意提示的组合，实现零样本分割。
- **模型架构（三部分）**：
  - **图像编码器**：基于 MAE 预训练的 ViT-H（Vision Transformer-Huge），处理 $1024 \times 1024$ 输入，输出 64×64 的特征图。编码开销最大，但只需一次前向。
  - **提示编码器**：点（位置编码 + 类别嵌入）、框（对角点位置编码）、掩码（卷积编码）分别编码后加和，统一为一个稀疏的提示嵌入。
  - **掩码解码器**：轻量 Transformer（2 层），将图像嵌入与提示嵌入进行交叉注意力融合，输出多个候选掩码（通常 3 个）及置信度分数。解码器非常轻量，支持实时交互。
- **多掩码输出**：对于每个提示，SAM 输出 3 个掩码（整体、子部分、子部分），因为在点提示下，一个点可能对应多个有效的分割区域（如衣领 vs 整件衬衫）。

**数据引擎**：SAM 的核心突破在于构建了史上最大的分割数据集 SA-1B。数据引擎分三个阶段迭代：

1. **辅助手动标注**：标注员使用 SAM 初始模型进行交互式标注，SAM 辅助输出候选掩码，大幅提升标注效率。
2. **半自动标注**：SAM 自动标注置信度高的区域，标注员仅修正困难的模糊区域，进一步加速标注。
3. **全自动标注**：在每张图上使用 32×32 网格点作为提示，自动生成大量掩码，经后处理去重后保留高质量掩码。

最终 SA-1B 包含 1100 万张图像和超过 10 亿个高质量掩码，规模远超之前所有分割数据集。

**局限性**：SAM 在自然场景中泛化能力极强，但推理速度较慢（ViT-H 编码器约 0.15s/图），在特定领域（如医学影像、遥感）需要微调才能达到最佳效果，且点提示的交互方式对复杂场景需要多次尝试。

### 3.4 模型对比总结

| 模型 | 年份 | 任务 | 范式 | 骨干 | 核心创新 |
|------|------|------|------|------|----------|
| FCN | 2015 | 语义 | 全卷积 + 上采样 | VGG | 首个端到端分割 |
| U-Net | 2015 | 语义 | 编码器-解码器 | 自设计 | 对称跳跃连接 |
| DeepLab v3+ | 2018 | 语义 | 编码器-解码器 | Xception | 空洞卷积 + ASPP |
| SegFormer | 2021 | 语义 | Transformer | MiT | 分层编码 + MLP 解码 |
| Mask R-CNN | 2017 | 实例 | 检测 + 掩码 | ResNet+FPN | RoIAlign + 掩码分支 |
| Mask2Former | 2022 | 统一 | 掩码分类 | ResNet/Swin | 掩码注意力 |
| SAM | 2023 | 通用 | 提示式 | ViT | 提示式基础模型 |

此表展示了分割任务从 CNN 到 Transformer、从专用模型到通用基础模型的演进趋势。

---

## 4. 损失函数

分割任务的损失函数通常由多个分量组合而成，以应对类别不平衡、难易样本不平衡和边界模糊等问题。

### 4.1 交叉熵损失（Cross Entropy Loss）

逐像素的交叉熵是最基础的分割损失：

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log(p_{i,c})$$

其中 $N$ 为像素总数，$C$ 为类别数，$y_{i,c}$ 为真实标签（one-hot），$p_{i,c}$ 为预测概率。

**加权交叉熵**：为每个类别赋予不同权重 $\alpha_c$，缓解类别不平衡问题：

$$\mathcal{L}_{\text{WCE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{C} \alpha_c \cdot y_{i,c} \log(p_{i,c})$$

### 4.2 Dice Loss

Dice Loss 直接优化 Dice 系数，对小目标和不平衡数据尤为有效：

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_{i} p_i g_i + \epsilon}{\sum_{i} p_i + \sum_{i} g_i + \epsilon}$$

其中 $p_i$ 为预测概率，$g_i$ 为真实标签（0 或 1），$\epsilon$ 为平滑项（通常取 1）防止分母为零。Dice Loss 对前景和背景像素等权重，天然适合前景占比小的场景（如医学影像中的病灶分割）。

### 4.3 Focal Loss

Focal Loss 通过降低易分类样本的权重，使训练聚焦于难分类样本：

$$\mathcal{L}_{\text{Focal}} = -\frac{1}{N}\sum_{i=1}^{N} (1 - p_i)^{\gamma} \log(p_i)$$

其中 $\gamma \geq 0$ 为聚焦参数（通常取 2）。当 $p_i$ 接近 1（易分类样本）时，$(1-p_i)^{\gamma}$ 趋近于 0，大幅降低其损失贡献；当 $p_i$ 较小时，损失接近原始交叉熵。

### 4.4 边界损失

边界区域通常是分割最难的部分。边界敏感损失通过显式建模边界像素来提升分割精度：

- **Boundary Loss**：使用距离图（Signed Distance Map）来衡量预测边界与真实边界的距离，直接优化边界形状。
- **Hausdorff Distance Loss**：近似 Hausdorff 距离，惩罚预测边界与真实边界之间的最大距离。

### 4.5 组合损失

实践中常将多个损失组合使用，以结合各自优势：

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{CE}} + \lambda_2 \mathcal{L}_{\text{Dice}} + \lambda_3 \mathcal{L}_{\text{Boundary}}$$

CE + Dice 的组合是最常见的做法：CE 保证像素级分类精度，Dice 优化全局重叠度，两者互补。$\lambda_i$ 为各损失的权重，通常通过实验调参确定。

### 4.6 Tversky Loss

Tversky Loss 是 Dice Loss 的泛化形式，通过引入 $\alpha$ 和 $\beta$ 参数来分别控制假正例和假负例的惩罚力度：

$$\mathcal{L}_{\text{Tversky}} = 1 - \frac{\sum_{i} p_i g_i}{\sum_{i} p_i g_i + \alpha \sum_{i} p_i (1-g_i) + \beta \sum_{i} (1-p_i) g_i}$$

当 $\alpha = \beta = 0.5$ 时退化为 Dice Loss。设置 $\beta > \alpha$ 可加大对假负例（漏检）的惩罚，这在病灶检测中尤为重要——漏检的代价远高于误检。

### 4.7 Lovász-Softmax Loss

Lovász-Softmax Loss 直接优化 mIoU 指标，将离散的 Jaccard 指数扩展为连续的 Lovász 扩展：

$$\mathcal{L}_{\text{Lovász}} = \frac{1}{C}\sum_{c=1}^{C} \overline{\Delta_{J_c}}(\mathbf{m}(c))$$

其中 $\overline{\Delta_{J_c}}$ 为 Jaccard 损失的 Lovász 扩展，$\mathbf{m}(c)$ 为第 $c$ 类的预测误差向量。Lovász-Softmax 在类别极度不平衡的场景下通常优于 CE + Dice 组合，但其计算开销较大，且在不同批次大小下稳定性不如 CE。

### 4.8 损失选择建议

| 场景 | 推荐损失 | 原因 |
|------|----------|------|
| 通用语义分割 | CE + Dice（各 0.5） | 稳定、普适 |
| 极度不平衡 | CE + Dice + Focal | 多层缓解不平衡 |
| 医学影像 | Dice + Boundary | 边界精度优先 |
| 小目标分割 | Tversky（$\beta > \alpha$） | 降低漏检 |
| 追求 mIoU 极致 | Lovász-Softmax | 直接优化指标 |

---

## 5. 训练策略与数据

### 5.1 编码器预训练

分割模型通常使用在 ImageNet 上预训练的分类网络（如 ResNet、ViT）作为编码器初始化，解码器则随机初始化。预训练编码器提供了强大的特征提取能力，显著加速收敛并提升最终性能。

**预训练策略对比**：

| 预训练方式 | 数据规模 | 标注需求 | 典型方法 | 分割效果 |
|------------|----------|----------|----------|----------|
| ImageNet 监督 | 百万级 | 全标注 | ResNet, ViT | 强基线 |
| 自监督对比学习 | 百万级 | 无标注 | MoCo, SimCLR | 接近监督 |
| 掩码图像建模 | 百万级 | 无标注 | MAE, BEiT | 适合 ViT |
| 自蒸馏 | 百万级 | 无标注 | DINO, DINOv2 | 极强特征 |
| 大规模弱监督 | 十亿级 | 弱监督 | SAM, CLIPSeg | 强大零样本 |

- **自监督预训练**：MoCo（动量对比学习）、SimCLR（强数据增强对比）、DINO（自蒸馏 + ViT）等方法在近年也成为分割模型的流行初始化策略，尤其在标注数据稀缺的场景下优势明显。
- **SAM 等大模型**：使用大规模弱监督数据进行预训练，展示了强大的零样本分割能力。SAM 在 SA-1B 上训练了 11 亿参数模型，可以零样本迁移到多种分割任务。
- **域适应策略**：当预训练域与目标域差异较大时（如自然图像预训练用于医学影像），可使用域适应（Domain Adaptation）或对抗训练来缩小域差距。

### 5.2 数据增强

分割任务对数据增强非常敏感，合理的增强策略可以显著提升泛化性能：

| 增强策略 | 说明 | 适用场景 |
|----------|------|----------|
| 随机裁剪（Random Crop） | 裁剪固定尺寸的子图 | 通用 |
| 随机翻转（Random Flip） | 水平/垂直翻转，同步变换掩码 | 通用 |
| 随机缩放（Random Scale） | 多尺度训练，增强尺度不变性 | 通用 |
| 弹性形变（Elastic Deformation） | 局部扭曲变形，模拟组织形变 | 医学影像 |
| 颜色抖动（Color Jitter） | 亮度、对比度、饱和度、色调随机变化 | 自然图像 |
| Copy-Paste | 将实例掩码复制粘贴到其他图像 | 实例分割 |

### 5.3 常用数据集

| 数据集 | 规模 | 任务类型 | 场景 |
|--------|------|----------|------|
| **Pascal VOC 2012** | 约 1.5 万张，20 类 | 语义分割 | 通用场景 |
| **COCO** | 约 20 万张，80 类 | 实例分割 / 全景分割 | 通用场景 |
| **ADE20K** | 约 2.5 万张，150 类 | 语义分割 / 全景分割 | 室内外场景 |
| **Cityscapes** | 约 5000 张精细标注，30 类 | 语义分割 | 城市场景（自动驾驶） |
| **ISIC** | 约 2.5 万张 | 语义分割 | 皮肤病变分割 |
| **SA-1B** | 1100 万张，10 亿+掩码 | 通用分割 | 通用场景（SAM 数据集） |

### 5.4 训练技巧

- **多尺度训练/测试**：训练时随机缩放输入，测试时多尺度推理并融合结果，常用策略如 0.5x、1.0x、1.5x、2.0x。
- **OHEM（Online Hard Example Mining）**：在训练过程中动态选择损失最大的像素/区域进行反向传播，强制模型关注难例。
- **学习率调度**：通常使用 Poly 学习率策略（$\text{lr} = \text{base\_lr} \times (1 - \frac{\text{iter}}{\text{max\_iter}})^{\text{power}}$，power=0.9），配合 Warmup 线性预热。
- **混合精度训练**：使用 FP16 加速训练，减少显存占用，是分割任务的标准实践。

---

## 6. 参考文献

[1] Long, J., Shelhamer, E., & Darrell, T., *Fully Convolutional Networks for Semantic Segmentation*, CVPR, 2015.

[2] Ronneberger, O., Fischer, P., & Brox, T., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI, 2015.

[3] Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L., *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs*, TPAMI, 2017.

[4] He, K., Gkioxari, G., Dollar, P., & Girshick, R., *Mask R-CNN*, ICCV, 2017.

[5] Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS, 2021.

[6] Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R., *Segment Anything*, ICCV, 2023.

[7] Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R., *Masked-attention Mask Transformer for Universal Image Segmentation*, CVPR, 2022.

[8] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollar, P., *Focal Loss for Dense Object Detection*, ICCV, 2017.

[9] Chen, L. C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*, ECCV, 2018.