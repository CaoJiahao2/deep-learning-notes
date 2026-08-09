# 模型压缩技术详解

> 量化、剪枝、蒸馏与紧凑架构：让大模型跑得动、跑得起、塞得进。

---

## 目录

1. [概述](#1)
2. [定点数基础](#2)
3. [训练后量化 PTQ 与量化感知训练 QAT](#3-ptq-qat)
4. [权重量化方法精讲](#4)
5. [极致低比特](#5)
6. [剪枝 (Pruning)](#6-pruning)
7. [知识蒸馏 (Knowledge Distillation)](#7-knowledge-distillation)
8. [组合策略与选择指南](#8)
9. [参考文献](#9)

---

## 1. 概述

### 1.1 为什么要压缩

大模型在精度上不断突破，但参数量与显存需求同步飙升：一个 7B 模型以 FP16 存储就要 14 GB，推理时还要额外装载 KV cache 与激活。模型压缩的核心目标是**在可接受的精度损失下，显著降低显存占用、推理延迟与单位成本**，从而让模型能够：

- 部署到边缘/移动端（手机、车机、IoT）；
- 在服务器侧提升吞吐、降低单 token 成本；
- 放进更小的 GPU/NPU，省下硬件采购与运维费用。

### 1.2 四大手段

| 手段 | 核心思想 | 主要收益 | 主要代价 |
|------|---------|---------|---------|
| 量化 (Quantization) | 用更低比特表示权重/激活 | 显存↓、带宽↓、速度↑ | 低比特易掉点 |
| 剪枝 (Pruning) | 去除冗余权重/通道 | 计算量↓、显存↓ | 结构化稀疏上限有限 |
| 知识蒸馏 (Knowledge Distillation) | 大模型教小模型 | 小模型逼近大模型 | 需教师、训练成本高 |
| 紧凑架构设计 | 设计高效算子/结构 | 同精度下更小更快 | 需重新设计与训练 |

实际部署中，四者往往组合使用：先用紧凑架构搭骨架，训练时叠加 QAT 与结构化剪枝，最后用蒸馏把大模型的能力迁移过来。

---

## 2. 定点数基础

### 2.1 浮点与定点表示

| 类型 | 比特 | 指数位 | 尾数位 | 动态范围 | 精度 |
|------|------|--------|--------|----------|------|
| FP32 | 32 | 8 | 23 | 大 | 高 |
| FP16 | 16 | 5 | 10 | 中 | 中 |
| BF16 | 16 | 8 | 7 | 大 | 低 |
| INT8 | 8 | - | - | $[-128,127]$ | 等距 |
| INT4 | 4 | - | - | $[-8,7]$ | 等距 |

**BF16 vs FP16**：BF16 牺牲尾数位换取与 FP32 相同的指数位（8 位），动态范围大、训练稳定，但精度低于 FP16；FP16 精度高但易溢出。训练常选 BF16，推理可选 FP16 或更低。

### 2.2 量化基本映射

将浮点张量 $x$ 量化为定点整数 $\hat{x}$：

$$
\hat{x} = \text{round}\left(\frac{x}{s}\right) + z
$$

其中 $s$ 为 scale（缩放因子），$z$ 为 zero-point（零点偏移）。反量化近似为：

$$
x \approx s\,(\hat{x} - z)
$$

- **对称量化**（$z=0$）：原点 0 对应整数 0，范围关于 0 对称，常用于权重（权重分布通常近似零均值）。
- **非对称量化**（$z \ne 0$）：可覆盖任意区间，常用于激活（如 ReLU 后非负分布，最小值不为 0）。

### 2.3 对称量化示例

```text
浮点权重范围 [-2.0, +2.0]，目标 INT8 对称量化

scale  s = max(|x|) / 127 = 2.0 / 127 ≈ 0.01575

x =  1.50  → round(1.50 / 0.01575) = 95    → 反量化 ≈  1.496
x = -0.30  → round(-0.30 / 0.01575) = -19   → 反量化 ≈ -0.299
x =  2.10  → clip 到 127                    → 反量化 ≈  2.000 (饱和)

量化误差来源：round 的舍入 + clip 的饱和
```

### 2.4 量化误差与粒度

量化误差 $e = x - s(\hat{x}-z)$ 主要来自**舍入**与**饱和**两类。降低误差的常用手段是改变量化粒度：

- **per-tensor**：整张量共用一组 $(s,z)$，开销小但易被离群点拖累；
- **per-channel**：每个输出通道单独一组 $(s,z)$，权重常采用，精度更稳但需额外存储 scale。

对于激活，由于范围随输入变化，常用 per-tensor（或 per-token，LLM 场景按 token 维度统计）。

---

## 3. 训练后量化 PTQ 与量化感知训练 QAT

### 3.1 PTQ (Post-Training Quantization)

PTQ 在模型训练完成后直接量化，**无需重训**，流程快、门槛低：

```text
训好的浮点模型 → 用少量校准数据统计激活范围
            → 确定 scale / zero-point
            → 写出量化模型
```

- **校准 (calibration)**：用几百到几千条代表性样本跑前向，记录每层激活的最值/分位数，估计量化范围。
- **优点**：几小时内完成，适合快速验证与大规模落地。
- **缺点**：低比特（< INT8）下精度掉点明显，尤其激活中存在离群值时。

典型方法：MinMax、Percentile、ACIQ、以及针对 LLM 的 SmoothQuant、AWQ、GPTQ（见第 4 节）。

### 3.2 QAT (Quantization-Aware Training)

QAT 在训练中插入**伪量化 (fake quantization)** 节点，前向时模拟量化误差，让模型学会在量化精度下仍保持精度。伪量化前向：

$$
\hat{x} = \text{round-clip}\left(\frac{x}{s}\right)\cdot s
$$

即先量化再反量化回浮点，前向数值落在定点网格上。反向传播时 `round` 不可导，采用**直通估计器 (Straight-Through Estimator, STE)**：梯度直接透传，等价于把量化操作的导数视为 1。

```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, z, qmin, qmax):
        xq = torch.round(x / s).clamp(qmin, qmax)  # 量化到整数网格
        xq = (xq - z) * s                            # 反量化回浮点
        ctx.save_for_backward(x, xq)
        return xq

    @staticmethod
    def backward(ctx, grad_output):
        x, xq = ctx.saved_tensors
        # STE: 仅在未饱和区间透传梯度，饱和处梯度为 0
        grad_x = grad_output * (xq == x).to(grad_output.dtype)
        return grad_x, None, None, None, None
```

- **优点**：精度最高，可稳定支撑 INT8 乃至更低比特。
- **缺点**：需完整训练流程、标注数据与超参调优，成本高。

### 3.3 PTQ vs QAT 对比

| 维度 | PTQ | QAT |
|------|-----|-----|
| 数据需求 | 少量无标注校准集 | 完整训练/标注集 |
| 时间成本 | 分钟~小时级 | 天~周级 |
| 精度 | INT8 接近全精度，<INT8 掉点明显 | 低比特仍可保持高精度 |
| 实现复杂度 | 低 | 高（需插入伪量化、改训练循环） |
| 适用场景 | 快速落地、大模型 LLM | 移动端、极致低比特、对精度敏感任务 |

---

## 4. 权重量化方法精讲

LLM 体量大、重训昂贵，因此面向 LLM 的低比特量化以 **PTQ + 仅量化权重**（或 W8A8）为主流。下面三种方法刻画了三条不同思路。

### 4.1 GPTQ

GPTQ (Frantar et al., 2022) 基于**二阶信息**逐层贪心量化权重。对每层权重矩阵 $W$，按列逐个量化，并用**未量化部分的二阶信息（近似 Hessian $H$）补偿已量化的误差**。直觉上，量化第 $i$ 列后产生的误差会通过未量化列的更新被抵消：

$$
w_{\text{new}} \leftarrow w - \frac{1}{[H^{-1}]_{ii}}\,H^{-1}_{:,i}\,\delta_i
$$

其中 $\delta_i$ 为第 $i$ 列量化引入的误差，$H^{-1}_{:,i}$ 是 Hessian 逆的第 $i$ 列。上式是逐列补偿的简化形式，实际按 block 更新。

- 常用 3-4 bit 权重；
- 精度高、掉点小，但需逐层计算 Hessian 逆，流程略复杂；
- 适合一次性离线量化、长期推理服务。

### 4.2 AWQ

AWQ (Lin et al., 2023) 的核心观察是**并非所有权重同等重要**：当某通道对应的激活幅度大时，该通道的权重对输出更敏感（salient channels）。AWQ 不去算 Hessian，而是通过 **per-channel scaling** 保护这些显著权重：

对权重列 $w$ 与激活 $x$，引入通道缩放因子 $s$（$s>1$ 对显著通道放大）：

$$
Y = (x)\,(w) \;\Rightarrow\; Y = (x \cdot s)\,(w \cdot s^{-1})
$$

放大 $x\cdot s$ 后再量化 $w\cdot s^{-1}$，使显著权重的量化误差被相对缩小（因为 $s^{-1}$ 把权重拉回原值时误差按 $s$ 缩放）。保护误差直觉：显著通道量化误差从 $\Delta w$ 降为约 $\Delta w / s$。

- 仅需少量校准数据估计显著通道；
- 速度快、易于实现，4-bit 掉点极小；
- 不改变模型结构，推理时可还原为标准低比特 kernel。

### 4.3 SmoothQuant

SmoothQuant (Xiao et al., 2022) 直面 LLM 的**迁移难度不对称**：权重分布平缓、易量化；激活有离群值、难量化。思路是用平滑因子 $s$ 把激活的难度**转移到权重**上：

$$
Y = (X)(W) = (X \cdot s^{-1})\cdot(s \cdot W)
$$

选择 $s$ 使 $X\cdot s^{-1}$ 的离群值被压平（激活变易量化），而 $s\cdot W$ 的范围仍可被权重量化吸收（权重难度的余量更大）。$s$ 通常按通道、基于激活与权重的最值比自适应求解。

- 实现 W8A8（权重 INT8、激活 INT8）的稳定量化；
- 无需重训、可融合进推理框架；
- 解决了激活离群值导致 INT8 PTQ 崩坏的痛点。

### 4.4 三者对比

| 方法 | 量化对象 | 典型比特 | 精度保持 | 速度/实现 | 核心思路 |
|------|---------|---------|---------|----------|---------|
| GPTQ | 权重 (W4A16) | 3-4 bit | 高 | 离线较慢、推理快 | 二阶误差补偿 |
| AWQ | 权重 (W4A16) | 4 bit | 高 | 离线快、推理快 | 激活感知显著通道 |
| SmoothQuant | 权重+激活 (W8A8) | 8 bit | 高 | 离线快、推理快 | 难度从激活迁移到权重 |

---

## 5. 极致低比特

进一步压到 1-bit / 1.58-bit，是用"几乎免费"的位运算换取极大吞吐与显存节省，代价是精度恢复更困难、需要专门训练。

### 5.1 BitNet 与 1.58-bit

BitNet (Wang et al., 2023) 将权重**二值化**为 $\{-1,+1\}$，矩阵乘退化为加减与累加，显存与能耗极低；但二值信息量太少，精度恢复难。后续 **BitNet b1.58** 把权重扩展为三值 $\{-1,0,+1\}$（即 1.58-bit，$\log_2 3 \approx 1.58$）：

$$
w_b \in \{-1, 0, +1\}
$$

三值相比二值多了"0"（剪枝语义），表达能力显著提升，在同等规模下可逼近全精度基线。

- 训练时仍需量化感知（二值化/三值化的 STE 与特殊初始化）；
- 推理极快、显存极省，适配自研高效 kernel；
- 精度-效率权衡：比特越低越快，但训练难度与对架构/数据的要求越高。

### 5.2 低比特的权衡

低比特并非"越低越好"：从 16→8 收益最大且风险低；8→4 需 GPTQ/AWQ 这类方法才能保精度；4→1.58 则需要重训与定制架构。选择时要在**显存/带宽节省**与**精度恢复成本**之间权衡，并考虑目标硬件是否有对应低比特 kernel 支持。

---

## 6. 剪枝 (Pruning)

### 6.1 非结构化剪枝

按权重**幅度置零 (magnitude pruning)**：把绝对值小的权重设为 0，得到稀疏矩阵。稀疏度可做到很高（如 90%+），模型存储为稀疏 CSR/COO 即可省显存。

- 优点：稀疏度上限高，理论上几乎不掉点；
- 缺点：非零元素位置随机，通用硬件难以加速（稀疏访存与负载均衡差），需专用稀疏库才能跑出实际加速。

**Lottery Ticket Hypothesis** (Frankle & Carbin, 2018)：一个密集网络中存在可独立训练的稀疏"中奖彩票"子网络，从初始化直接训练该子结构即可达到原网络精度。该假说揭示了初始化与稀疏结构的联系，是现代剪枝研究的重要理论基础。

### 6.2 结构化剪枝

直接以**整通道/整 head**为单位剪枝，剪后仍是稠密结构，硬件可直接加速。典型思路是对 BatchNorm 的 $\gamma$ 参数做 $L_1$ 稀疏正则，$\gamma$ 趋 0 的通道被认为不重要而被整通道移除：

$$
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_i |\gamma_i|
$$

- 优点：推理直接提速、无需特殊稀疏硬件；
- 缺点：稀疏度上限受结构约束（一般到 50% 左右就明显掉点）。

### 6.3 稀疏度与加速

非结构化稀疏只有落到**结构化稀疏模式**才能换来真实加速。NVIDIA Ampere 及之后的 GPU 提供 **2:4 结构化稀疏 (structured sparsity)** tensor core：每连续 4 个非零元素中恰有 2 个为 0 时，可用稀疏 kernel 加速约 2 倍。

```text
非结构化稀疏：  * 0 * 0 0 * * 0   →  硬件难加速（位置随机）
2:4 稀疏：      * 0 * 0  * 0 * 0  →  每块 4 元素恰 2 零，硬件可 2x 加速
```

### 6.4 非结构化 vs 结构化

| 维度 | 非结构化剪枝 | 结构化剪枝 |
|------|-------------|-----------|
| 稀疏上限 | 高（90%+） | 中（~50%） |
| 硬件加速 | 难（需专用稀疏库/2:4） | 直接加速 |
| 实现难度 | 中（需 mask 与稀疏存储） | 低（移除整通道即可） |
| 典型代表 | Deep Compression, Lottery Ticket | 通道剪枝、BN-$\gamma$ 稀疏 |

---

## 7. 知识蒸馏 (Knowledge Distillation)

### 7.1 经典 KD (Hinton et al., 2015)

让学生模型 $s$ 模仿教师 $t$ 的**软标签 (soft labels)**。教师 logits $z_t$ 经温度 $T$ 软化后蒸馏，损失为硬标签交叉熵与软标签 KL 的加权和：

$$
\mathcal{L}_{\text{KD}} = (1-\alpha)\,\text{CE}\bigl(y,\,\sigma(z_s)\bigr) + \alpha\,T^2\,\text{KL}\Bigl(\sigma(z_t/T)\,\|\,\sigma(z_s/T)\Bigr)
$$

- 温度 $T$：放大 logits 的相对差异，使"暗知识 (dark knowledge)"——非正确类的相对概率——被学生学到；$T^2$ 用以补偿温度对 KL 梯度尺度的缩放。
- $\alpha$：权衡硬标签（真值）与软标签（教师）。

### 7.2 任务蒸馏 vs 特征蒸馏

- **任务蒸馏 (logit distillation)**：仅在最终输出层对齐，学生模仿教师的预测分布（即上式）。
- **特征蒸馏 (FitNets, Romero et al., 2014)**：在**中间层隐表示**上对齐，让学生隐层 $h_s$ 经变换后匹配教师隐层 $h_t$：

$$
\mathcal{L}_{\text{FitNet}} = \bigl\|\,\phi(h_s) - h_t\,\bigr\|_2^2
$$

中间层监督提供更密集的梯度信号，常比单纯 logit 蒸馏更稳、更省训练步数。

### 7.3 白盒 vs 黑盒蒸馏

| 维度 | 白盒蒸馏 | 黑盒蒸馏 |
|------|---------|---------|
| 可访问信息 | logits、隐层、梯度 | 仅 API 输出（文本/prob） |
| 适用 | 自有模型 | 第三方闭源模型 |
| 典型做法 | FitNets、logit KD | 数据蒸馏（用强模型生成数据再 SFT） |

LLM 时代常用"**数据蒸馏**"：用强模型（GPT-4 等）生成高质量指令/回答数据，对小模型做 SFT。DistilBERT (Sanh et al., 2019) 则把 BERT (110M) 蒸馏到 66M，保留约 97% 能力、推理快 60%。

### 7.4 KD 损失伪代码

```python
import torch
import torch.nn.functional as F

def kd_loss(logits_s, logits_t, labels, T=4.0, alpha=0.5):
    # 硬标签部分
    loss_hard = F.cross_entropy(logits_s, labels)
    # 软标签部分：KL(教师 || 学生)，梯度尺度用 T^2 补偿
    p_t = F.softmax(logits_t / T, dim=-1)
    log_p_s = F.log_softmax(logits_s / T, dim=-1)
    loss_soft = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
    return (1 - alpha) * loss_hard + alpha * loss_soft
```

---

## 8. 组合策略与选择指南

### 8.1 典型压缩管线

实际部署常组合多种手段，例如：

```text
教师模型 (大)
   │  知识蒸馏
   ▼
学生模型 (紧凑架构)
   │  QAT + 结构化剪枝
   ▼
量化+稀疏的紧凑模型 → 导出 (ONNX/TensorRT) → 部署
```

各手段之间存在**相互作用**：

- 蒸馏 + QAT：蒸馏提供平滑监督，能缓解 QAT 在低比特下的精度塌陷；
- 剪枝 + 量化：先剪枝再量化通常优于反过来，因为剪枝改变了数值范围；
- 边际递减：量化到 INT8 收益最大，再叠加 INT4 剪枝，精度恢复成本急升。

### 8.2 场景选择决策表

| 场景 | 推荐组合 | 预期收益 |
|------|---------|---------|
| 移动端 (手机/手表) | 紧凑架构 + QAT (INT8) + 结构化剪枝 + 蒸馏 | 10x+ 体积与延迟下降 |
| 服务器 LLM 推理 | AWQ/GPTQ (W4A16) 或 SmoothQuant (W8A8) | 2-4x 吞吐、显存减半 |
| 边缘 NPU (车机/IoT) | 紧凑架构 + PTQ (INT8) + 通道剪枝 | 5-10x 加速、低功耗 |
| 极致吞吐 (自研硬件) | BitNet 1.58-bit 训练 + 蒸馏 | 数倍吞吐、能耗骤降 |
| 精度敏感任务 | 仅蒸馏 + INT8 PTQ | 精度近无损、轻量加速 |

### 8.3 选择要点

1. 先量化，再剪枝，最后蒸馏兜底——顺序影响最终精度；
2. LLM 体量大，优先考虑 W4A16 (AWQ/GPTQ)，避免 QAT 的全量重训成本；
3. 硬件支持是关键：2:4 稀疏、INT4 kernel、低比特算子决定了理论收益能否落地；
4. 永远留一条回归基线：压缩后必须用业务指标（而非仅 perplexity/accuracy）回归。

---

## 9. 参考文献

[1] Frantar E. et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR, 2023.

[2] Lin J. et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*, MLSys, 2024.

[3] Xiao G. et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, ICML, 2023.

[4] Wang H. et al., *BitNet: Scaling 1-bit Transformers for Large Language Models*, arXiv:2310.11453, 2023.

[5] Wang H. et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58-bits*, arXiv:2402.17764, 2024.

[6] Hinton G. et al., *Distilling the Knowledge in a Neural Network*, NeurIPS Deep Learning Workshop, 2015.

[7] Romero A. et al., *FitNets: Hints for Thin Deep Nets*, ICLR, 2015.

[8] Frankle J. and Carbin M., *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*, ICLR, 2019.

[9] Sanh V. et al., *DistilBERT: A distilled version of BERT, smaller, faster, cheaper and lighter*, NeurIPS EMC Workshop, 2019.

[10] Han S. et al., *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding*, ICLR, 2016.

[11] Mishra A. et al., *Accelerating Sparse Deep Neural Networks*, arXiv:2104.08378, 2021.

[12] Dettmers T. et al., *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*, NeurIPS, 2022.
