# 目标检测详解

> 从两阶段到单阶段：让模型同时回答「是什么」与「在哪里」。

---

## 目录

1. [概述](#1)
2. [两阶段方法](#2)
3. [单阶段方法](#3)
4. [两阶段 vs 单阶段对比](#4-vs)
5. [训练策略](#5)
6. [参考文献](#6)

---

## 1. 概述

### 1.1 什么是目标检测

目标检测（Object Detection）是计算机视觉的核心任务之一，要求模型同时完成两个子任务：

- **分类（Classification）**：判断图像中每个目标属于哪个类别；
- **定位（Localization）**：用边界框（Bounding Box）精确标出每个目标的位置。

与图像分类不同，目标检测的输出不是单个标签，而是不定长的检测结果列表，每项包含类别标签、置信度分数和框坐标。这使其比分类更复杂，也更贴近实际应用。

### 1.2 核心评价指标

**IoU（Intersection over Union）** 衡量预测框与真实框的重叠程度：

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

其中 $A$ 为预测框，$B$ 为真实框。IoU 取值 $[0, 1]$，通常设定阈值（如 0.5）判断预测是否正确。

**mAP（mean Average Precision）** 是目标检测的核心评价指标：

- **AP（Average Precision）**：对单个类别，在不同召回率下计算精确率的平均值，即 PR 曲线下的面积。
- **mAP**：对所有类别的 AP 取平均。
- **mAP@0.5**：IoU 阈值为 0.5 时的 mAP（Pascal VOC 标准）。
- **mAP@0.5:0.95**：在 0.5 到 0.95 之间以 0.05 为步长取 10 个 IoU 阈值，分别计算 mAP 后取平均（COCO 主指标，更严格）。

此外，COCO 还提供 AP$_{S}$、AP$_{M}$、AP$_{L}$（小/中/大目标的 AP），以及 AR（Average Recall）。

### 1.3 方法分类

目标检测方法可分为两大类：

| 类型 | 核心思想 | 代表方法 |
|------|---------|---------|
| 两阶段（Two-Stage） | 先提议区域，再分类精修 | R-CNN 系列 |
| 单阶段（One-Stage） | 跳过区域提议，直接预测 | YOLO、SSD、RetinaNet |

此外，近年出现了基于 Transformer 的端到端方法（如 DETR），将检测建模为集合预测问题。

---

## 2. 两阶段方法

### 2.1 核心思想

两阶段方法将检测拆分为两个步骤：

1. **区域提议（Region Proposal）**：生成可能包含目标的候选区域；
2. **分类与精修（Classification + Refinement）**：对每个候选区域进行类别判断和边界框回归。

这种「先粗筛再精判」的策略精度高，但速度相对较慢。

### 2.2 Faster R-CNN 流水线

Faster R-CNN 是两阶段方法的里程碑，首次用 RPN（Region Proposal Network）实现端到端的区域提议，取代了传统的外部提议算法（如 Selective Search）。

```text
图像 → Backbone(ResNet + FPN) → 多层特征图
                                    │
                   RPN: 每 anchor 预测 物体性 + 框偏移 → 候选框
                                    │
                   RoI Align → 固定尺寸 RoI 特征
                                    │
                   检测头: 全连接/卷积 → 分类 + 框回归
                                    │
                   NMS 去重 → 最终检测结果
```

### 2.3 关键组件

#### Anchor（锚框）

在特征图的每个空间位置，预置多种尺度和宽高比的参考框，称为 Anchor。RPN 不直接预测绝对坐标，而是预测相对于 Anchor 的偏移量，这大大降低了回归难度。

例如，在 FPN 的每一层，每个位置预置 3 种宽高比（$\{1:2, 1:1, 2:1\}$）和 3 种尺度（$\{2^0, 2^{1/3}, 2^{2/3}\}$），共 9 个 Anchor。对于 5 层 FPN 特征图，每张图像可产生约 200k 个 Anchor。

#### RPN 损失

RPN 对每个 Anchor 做两件事：

- **二分类**：判断该 Anchor 是前景（有目标）还是背景（无目标）。正样本定义为与任意真实框 IoU > 0.7 的 Anchor，负样本为 IoU < 0.3 的 Anchor。
- **框回归**：预测 Anchor 到匹配真实框的偏移量 $(\Delta x, \Delta y, \Delta w, \Delta h)$。

RPN 总损失为：

$$L_{\text{RPN}} = \frac{1}{N_{cls}}\sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}}\sum_i p_i^* L_{reg}(t_i, t_i^*)$$

其中 $p_i^*$ 为真实标签（1 表示前景），$L_{cls}$ 为二分类交叉熵，$L_{reg}$ 为 Smooth L1 损失。

#### RoI Align

RoI Pooling 在将候选区域映射到固定尺寸特征时，会进行两次量化取整，导致特征与原始图像像素的错位。RoI Align（Mask R-CNN 提出）使用双线性插值替代量化，消除了这个误差，对小目标检测提升尤为明显。

具体步骤：
1. 将 RoI 均匀划分为 $k \times k$ 个 bin；
2. 在每个 bin 内采样 4 个点（通过双线性插值计算特征值）；
3. 对每个 bin 做 Max/Average Pooling。

#### 检测头

对每个 RoI 特征，检测头输出：
- 多类别分类概率（$C+1$ 类，含背景）；
- 每个类别的边界框精修偏移量。

现代检测头常采用「解耦头」设计，将分类和回归分支分开，彼此独立。

### 2.4 经典演进

| 模型 | 年份 | 贡献 |
|------|------|------|
| R-CNN (Girshick) | 2014 | Selective Search 提候选框 → CNN 特征 → SVM 分类，开创两阶段范式 |
| Fast R-CNN (Girshick) | 2015 | 整图过 CNN + RoI Pooling，端到端训练，速度提升 10 倍 |
| Faster R-CNN (Ren) | 2015 | **RPN** 取代外部提议，近乎实时的两阶段检测 |
| FPN (Lin) | 2017 | 特征金字塔，融合多尺度特征，大幅提升小目标检测 |
| Mask R-CNN (He) | 2017 | RoI Align + 分割头，检测 + 实例分割统一框架 |
| Cascade R-CNN (Cai) | 2018 | 级联多个回归头，逐步提升 IoU 阈值，减少过拟合 |

### 2.5 代码示例

```python
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 加载预训练模型（ResNet-50 + FPN）
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)
model.eval()

# 推理
images = [torch.rand(3, 300, 400)]  # 模拟一批图像
predictions = model(images)
# 输出: list of dict, 每项包含 boxes, labels, scores
print(predictions[0]["boxes"].shape)   # [N, 4]
print(predictions[0]["scores"].shape)  # [N]
```

---

## 3. 单阶段方法

### 3.1 核心思想

单阶段方法跳过区域提议步骤，直接在特征图上一次性预测所有目标的类别和边界框。这种「一步到位」的设计使推理速度极快，适合实时应用。

单阶段方法面临的核心挑战是**极端的正负样本不平衡**——特征图上绝大多数位置是背景，只有极少数位置包含目标。Focal Loss 的提出有效解决了这一问题。

### 3.2 YOLO 系列

YOLO（You Only Look Once）是最具代表性的单阶段检测方法，将检测建模为回归问题。

**YOLOv1 核心流程**：

```text
图像 → 分成 S×S 网格 → 每格预测 B 个框(坐标 + 置信度) + C 类概率
          ↓ 单次前向
        NMS 去重 → 检测框
```

**演进路线**：

| 版本 | 年份 | 关键改进 |
|------|------|---------|
| YOLOv1 | 2016 | 网格预测，极快端到端，开创单阶段范式 |
| YOLOv2 | 2017 | Anchor 机制、BatchNorm、多尺度训练 |
| YOLOv3 | 2018 | FPN 式多尺度预测、Darknet-53 主干 |
| YOLOv4 | 2020 | 大量工程优化（Mish、CIoU、Mosaic） |
| YOLOv5 | 2020 | PyTorch 实现，工程化部署友好 |
| YOLOv8 | 2023 | Anchor-free、解耦头、C2f 模块 |

YOLOv8 的核心改进：
- **Anchor-free**：直接预测框中心与宽高，摆脱 Anchor 的超参数调优；
- **解耦头（Decoupled Head）**：分类和回归分支独立，避免任务冲突；
- **C2f 模块**：改进的 CSP 瓶颈结构，丰富梯度流。

### 3.3 SSD

SSD（Single Shot MultiBox Detector）在多个尺度的特征图上分别进行预测，每个特征图负责检测不同尺度的目标：

- 浅层高分辨率特征图 → 检测小目标；
- 深层低分辨率特征图 → 检测大目标。

这种设计使 SSD 在速度和精度之间取得了较好的平衡，但小目标检测仍然弱于 FPN 方案。

### 3.4 Focal Loss 与 RetinaNet

单阶段方法在训练时，大量候选位置（约 $10^4 \sim 10^5$）几乎全为背景，标准交叉熵损失被简单负样本主导，导致训练不稳定。

**Focal Loss**（RetinaNet 提出）通过降权易分样本来解决这个问题：

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log p_t$$

其中：
- $p_t$：模型对正确类别的预测概率；
- $\alpha_t$：类别权重因子，平衡正负样本（通常 $\alpha \approx 0.25$）；
- $\gamma$：聚焦参数（通常 $\gamma = 2$），$\gamma$ 越大，易分样本的权重越低。

当 $\gamma = 0$ 时 Focal Loss 退化为带权重的交叉熵。当 $\gamma = 2$、$p_t = 0.9$ 时，该样本的损失被缩小 $(1-0.9)^2 = 100$ 倍；而 $p_t = 0.1$ 的难样本几乎不受影响。这是 RetinaNet 首次在精度上追平两阶段方法的关键。

### 3.5 Anchor-free 范式

传统方法依赖预置 Anchor，需要精心设计 Anchor 的尺度、宽高比等超参数。Anchor-free 方法直接预测目标的关键点：

**FCOS（Fully Convolutional One-Stage）**：
- 对特征图每个前景点，直接预测该点到目标框四条边的距离 $(l, t, r, b)$；
- 引入 **Center-ness** 分支，预测该点距离目标中心的程度，用于抑制偏离中心的低质量预测框；
- Center-ness 分数：$\sqrt{\frac{\min(l, r)}{\max(l, r)} \times \frac{\min(t, b)}{\max(t, b)}}$，越接近中心越接近 1。

**CenterNet**：
- 将目标表示为边界框中心点的热图（Heatmap） + 宽高；
- 输入图像通过全卷积网络输出三个分支：中心点热图、中心点偏移、目标宽高；
- 推理时直接在热图上取峰值，无需 NMS 后处理。

### 3.6 代码示例

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")  # n: nano, 最轻量版本

# 单张图像推理
results = model("image.jpg")
for r in results:
    print(r.boxes.xyxy)   # 检测框坐标 [N, 4]
    print(r.boxes.cls)    # 类别索引 [N]
    print(r.boxes.conf)   # 置信度 [N]
```

---

## 4. 两阶段 vs 单阶段对比

| 维度 | 两阶段 | 单阶段 |
|------|--------|--------|
| 速度 | 较慢（5-20 FPS） | 快（30-100+ FPS），可实时 |
| 精度 | 略高，小目标更优 | 已接近（Focal Loss 后），大中目标相当 |
| 模型复杂度 | 较高，多组件 | 较低，端到端 |
| 训练复杂度 | 需平衡 RPN 和检测头 | 需处理正负样本不平衡 |
| 部署 | 较重型 | 轻量，适合端侧和边缘设备 |
| 适用场景 | 精度优先（安防、医疗） | 实时优先（自动驾驶、无人机、移动端） |
| 扩展性 | 易扩展（Mask R-CNN 加分 割头） | 可扩展为多任务（YOLO 系列已支持分割、姿态） |

单阶段方法在 Focal Loss 提出后精度不再明显落后于两阶段，而 YOLOv8 等现代方法已在速度-精度曲线上占据主导地位。DETR 等基于 Transformer 的方法则开启了第三条路线（端到端集合预测）。

---

## 5. 训练策略

### 5.1 数据增强

目标检测对数据增强高度敏感，合理增强可显著提升泛化能力：

| 增强方法 | 说明 |
|---------|------|
| 随机翻转/旋转 | 水平翻转是标配，旋转增强几何多样性 |
| 多尺度训练 | 训练时随机缩放输入，增强尺度鲁棒性 |
| Mosaic | 将 4 张图像拼接为 1 张，丰富小目标上下文 |
| Mixup | 两张图像线性混合，软化标签 |
| 随机仿射/HSV | 仿射变换 + 颜色抖动，模拟真实场景变化 |

Mosaic 增强是 YOLOv4/v5 系列的标配，有效提升了小目标和遮挡场景下的检测能力。

### 5.2 框回归损失

框回归损失从简单的坐标误差逐步演化为更贴合几何直觉的 IoU 系列损失：

| 损失函数 | 公式/思想 | 特点 |
|---------|----------|------|
| Smooth L1 | 分段函数，$|x| < 1$ 时用 $0.5x^2$，否则 $|x| - 0.5$ | 对异常值鲁棒，但与 IoU 不对齐 |
| IoU Loss | $L = 1 - \text{IoU}$ | 直接优化 IoU，尺度不变 |
| GIoU Loss | $L = 1 - \text{IoU} + \frac{|C - A \cup B|}{|C|}$ | 解决无重叠时梯度消失 |
| CIoU Loss | 考虑中心距离 + 宽高比一致性 | YOLOv4 起广泛使用，收敛更快 |

CIoU（Complete IoU）损失公式：

$$L_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

其中 $\rho$ 为预测框与真实框中心的欧氏距离，$c$ 为包围两框的最小矩形对角线长度，$v$ 衡量宽高比一致性，$\alpha$ 为平衡系数。

### 5.3 NMS 与 Soft-NMS

**NMS（Non-Maximum Suppression）** 是检测后处理的标准步骤：
1. 按置信度排序所有检测框；
2. 选择置信度最高的框，移除所有与它 IoU 超过阈值的框；
3. 重复直到无框剩余。

**Soft-NMS** 不直接移除重叠框，而是降低其置信度：

$$s_i = s_i \cdot e^{-\frac{\text{IoU}(M, b_i)^2}{\sigma}}$$

其中 $M$ 为当前最高分框，$b_i$ 为被抑制的框。Soft-NMS 在密集场景下保留更多检测，提升召回率。

### 5.4 采样策略与正负样本分配

- **正负样本比**：训练时通常控制正负样本比例为 1:3 左右，防止背景主导损失；
- **标签分配**：现代方法（如 YOLOv8 的 TaskAlignedAssigner）不再仅依赖固定 IoU 阈值，而是综合考虑分类得分和定位质量来分配正样本；
- **在线难例挖掘（OHEM）**：按损失排序，只保留最难样本反向传播。

### 5.5 常用数据集

| 数据集 | 类别数 | 图像数 | 特点 |
|--------|--------|--------|------|
| Pascal VOC | 20 | ~11k | 经典基准，mAP@0.5 |
| COCO | 80 | ~330k | 主流基准，mAP@0.5:0.95 |
| Objects365 | 365 | ~1.7M | 大规模预训练 |
| Open Images | 600 | ~9M | 超大规模，含层级标签 |

COCO 是目前最广泛使用的检测基准，其 80 个类别包含日常物体、动物、交通工具等，评测体系完善。

---

## 6. 参考文献

[1] Girshick, R., et al. Rich feature hierarchies for accurate object detection and semantic segmentation. CVPR, 2014.

[2] Girshick, R. Fast R-CNN. ICCV, 2015.

[3] Ren, S., et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS, 2015.

[4] Lin, T.-Y., et al. Feature Pyramid Networks for Object Detection. CVPR, 2017.

[5] He, K., et al. Mask R-CNN. ICCV, 2017.

[6] Cai, Z. & Vasconcelos, N. Cascade R-CNN: Delving into High Quality Object Detection. CVPR, 2018.

[7] Redmon, J., et al. You Only Look Once: Unified, Real-Time Object Detection. CVPR, 2016.

[8] Liu, W., et al. SSD: Single Shot MultiBox Detector. ECCV, 2016.

[9] Lin, T.-Y., et al. Focal Loss for Dense Object Detection. ICCV, 2017.

[10] Tian, Z., et al. FCOS: Fully Convolutional One-Stage Object Detection. ICCV, 2019.

[11] Zhou, X., et al. Objects as Points. arXiv:1904.07850, 2019.

[12] Carion, N., et al. End-to-End Object Detection with Transformers. ECCV, 2020.

[13] Bochkovskiy, A., et al. YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv:2004.10934, 2020.

[14] Rezatofighi, H., et al. Generalized Intersection over Union. CVPR, 2019.

[15] Zheng, Z., et al. Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. AAAI, 2020.