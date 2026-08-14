# 非自回归与扩散式语言模型

> 从并行解码到扩散语言模型，打破自回归逐 Token 生成的推理瓶颈。

---

## 目录

1. [概述](#1)
2. [自回归的瓶颈](#2)
3. [投机解码（Speculative Decoding）](#3-speculative-decoding)
4. [并行解码：Medusa 与 Eagle](#4-medusa-eagle)
5. [掩码语言模型复兴](#5)
6. [扩散式语言模型](#6)
7. [Blockwise 并行生成](#7-blockwise)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1. 概述

主流 LLM 采用自回归（Autoregressive, AR）方式逐 Token 生成：每步生成一个 Token，串行执行 $n$ 步。这一范式简单稳定，但在推理时存在两个根本问题：

- **延迟与长度成正比**：生成 1000 个 Token 需要 1000 次串行前向传播。
- **计算利用率低**：每步只处理一个新 Token，GPU 大量计算资源闲置。

非自回归（Non-Autoregressive, NAR）和半自回归方法旨在通过并行生成多个 Token 来加速推理。2023-2025 年，这一方向因投机解码的成熟、扩散语言模型的出现和掩码生成模型的复兴而重新活跃。

技术路线图：

$$
\underbrace{\text{投机解码}}_{\text{不修改模型}} \rightarrow \underbrace{\text{并行解码头}}_{\text{轻量修改}} \rightarrow \underbrace{\text{掩码/扩散 LM}}_{\text{重新设计}}
$$

---

## 2. 自回归的瓶颈

### 2.1 延迟分析

自回归生成长度为 $n$ 的序列需要 $n$ 次串行前向传播。每次前向传播的延迟包括：

- GPU kernel 启动开销。
- 注意力计算（随上下文增长）。
- 权重读取（内存带宽瓶颈，尤其在小 batch 时）。

在内存带宽受限的推理场景中，每个 Token 都需要将整个模型权重从 HBM 读取一次，导致硬件利用率极低（通常 < 1% MFU）。

### 2.2 理论加速空间

根据内存带宽分析：

$$
\text{单次前向可承受 Token 数} \approx \frac{\text{内存带宽} \times \text{延迟容忍}}{\text{模型参数字节数}}
$$

在典型 A100 + 7B 模型场景下，单次前向理论上可处理数十至数百个 Token，但自回归每次只处理 1 个。这意味着如果能并行生成 $k$ 个 Token，理论加速比可达 $k$ 倍。

---

## 3. 投机解码（Speculative Decoding）

### 3.1 核心思想

投机解码（Leviathan et al., 2023; Chen et al., 2023）用一个小模型（草稿模型）快速生成 $k$ 个候选 Token，再用大模型（目标模型）一次性验证。接受匹配的前缀，拒绝后从拒绝位置重新采样。

算法流程：

1. 草稿模型自回归生成 $k$ 个候选 Token：$\tilde{x}_1, \ldots, \tilde{x}_k$。
2. 目标模型对 $\tilde{x}_1, \ldots, \tilde{x}_k$ 并行前向，获得每个位置的概率分布。
3. 从第一个位置开始，按接受/拒绝准则采样：若草稿 Token 被接受则保留，否则用调整后的分布重新采样并停止。
4. 若全部接受，再从目标模型采样一个额外 Token。

### 3.2 接受准则

为保证输出分布与目标模型完全一致，第 $i$ 个草稿 Token 的接受概率为：

$$
\alpha_i = \min\left(1, \frac{p(\tilde{x}_i | x_{<i})}{q(\tilde{x}_i | x_{<i})}\right)
$$

其中 $p$ 是目标模型分布，$q$ 是草稿模型分布。若拒绝，修正分布为：

$$
p'(x) = \max(0, p(x) - q(x))
$$

### 3.3 特点

- **无损加速**：输出分布严格等同于目标模型，不损失质量。
- **加速比**：取决于草稿模型与目标模型的对齐程度，通常 2-3x。
- **无需重新训练**：只需有一个合适的草稿模型（可用同模型的量化版或小参数版）。
- **自回归方案的优化**：投机解码不改变自回归本质，只是用并行验证减少串行步数。

### 3.4 变体

- **Medusa（Cai et al., 2024）**：在目标模型上添加多个并行解码头，每个头预测未来第 $k$ 个 Token，无需独立草稿模型。
- **EAGLE（Li et al., 2024）**：在隐藏状态上训练轻量自回归草稿头，比投机解码接受率更高。
- **Lookahead Decoding**：用 Jacobi 迭代并行生成候选，无需草稿模型。
- **Prompt Lookup Decoding**：直接从 prompt 中匹配 $n$-gram 作为草稿。

---

## 4. 并行解码：Medusa 与 Eagle

### 4.1 Medusa

Medusa 在 LLM 最后一层隐藏状态上添加 $k$ 个解码头，每个头独立预测未来第 $i$ 个 Token：

$$
\hat{x}_{t+i} = \text{head}_i(h_t)
$$

生成时，$k$ 个头同时产生候选 Token，组合成候选序列树，由主模型并行验证。Medusa 的典型配置是 4-5 个头，实现 2-3x 加速。

Medusa 头的训练只需少量数据（~100K 条），冻结主模型，单独训练解码头，训练成本极低。

### 4.2 EAGLE

EAGLE 在自回归特征上训练一个轻量草稿模型：

1. 利用主模型最后一层隐藏状态和已生成 Token 的嵌入。
2. 草稿模型预测下一个 Token 的隐藏状态（而非 Token 本身）。
3. 从预测的隐藏状态生成候选 Token 树。

EAGLE 利用了隐藏状态中比 Token 概率更丰富的信息，接受率和加速比通常优于标准投机解码和 Medusa。EAGLE-2 进一步引入动态草稿树，根据上下文自适应调整候选数量。

### 4.3 投机解码 vs 并行头

| 维度 | 投机解码 | Medusa/EAGLE |
|------|---------|-------------|
| 草稿模型 | 需要独立小模型 | 内置于主模型 |
| 训练成本 | 零（需已有草稿模型） | 训练额外头（低成本） |
| 部署复杂度 | 维护两个模型 | 单一模型 |
| 加速比 | 2-3x | 2-4x |
| 质量损失 | 无损 | 近似无损 |

---

## 5. 掩码语言模型复兴

### 5.1 从 BERT 到生成式掩码 LM

BERT 类掩码语言模型（MLM）擅长理解但不能直接生成。2024-2025 年，多个工作将掩码预测重新用于生成：

- **MaskGiT（Chang et al., 2022）**：图像生成中用并行掩码解码，每步同时预测所有被掩码位置。
- **GPT 仍是自回归**：文本生成长期由 AR 主导，但掩码方法开始回归。

### 5.2 Mercury

NVIDIA 的 Mercury（2025）代表了掩码生成模型在文本上的复兴：

- 使用类似 MaskGiT 的并行解码：每步同时预测多个被掩码位置。
- 通过置信度调度决定每步解码哪些位置。
- 在保持生成质量的同时，实现显著的延迟降低。

### 5.3 掩码生成的挑战

- **位置独立性问题**：并行预测的 Token 之间缺乏顺序依赖，可能出现不一致。
- **生成顺序敏感**：先确定的 Token 影响后确定的 Token，顺序策略影响质量。
- **采样温度控制**：并行解码下的采样理论不如自回归成熟。

---

## 6. 扩散式语言模型

### 6.1 核心思想

扩散模型在图像生成上取得了巨大成功（见 `architectures/diffusion.md`）。扩散式语言模型（Diffusion LLM）将离散文本 Token 视为离散状态，通过前向加噪和反向去噪生成文本：

- **前向过程**：逐步将文本 Token 替换为随机 Token（或 mask Token），从 $x_0$ 到 $x_T$（全噪声）。
- **反向过程**：训练神经网络从 $x_t$ 预测 $x_0$（或预测被噪声替换的原始 Token），迭代去噪。

### 6.2 离散扩散

与连续扩散不同，文本 Token 是离散的。离散扩散的转移矩阵定义为：

$$
q(x_t | x_{t-1}) = (1 - \beta_t) \delta(x_t, x_{t-1}) + \beta_t \frac{1}{K}
$$

即以概率 $\beta_t$ 将 Token 替换为均匀随机 Token（或 [MASK]）。反向过程学习 $p_\theta(x_{t-1} | x_t)$，恢复被腐蚀的 Token。

### 6.3 LLaDA

LLaDA（Large Language Diffusion with mAsking, 2025）是代表性的扩散式语言模型：

- 用 [MASK] Token 作为腐蚀方式（absorbing state diffusion）。
- 训练目标：随机 mask 一部分 Token，让模型预测被 mask 的 Token：

$$
\mathcal{L} = \mathbb{E}_{t, x_t}\left[\sum_{i: x_i = [\text{MASK}]} -\log p_\theta(x_i^0 | x_t)\right]
$$

- 反向采样：每步将置信度最低的位置 mask 并重新预测，逐步细化。
- LLaDA 在 8B 规模上展示了接近 LLaMA3 的生成质量。

### 6.4 扩散 LM 的优劣势

**优势**：

- 并行生成：所有位置可同时去噪，推理延迟与序列长度解耦。
- 灵活的 infilling：可同时填充任意位置，天然适合文本编辑、补全和重写。
- 双向注意力：不像 AR 模型只能看到左侧上下文，扩散 LM 可使用双向注意力。

**挑战**：

- 采样质量在 10B+ 规模尚未被证明完全匹配自回归模型。
- 推理时需要多步去噪（通常 20-100 步），但每步可并行处理。
- 采样理论和可控生成不如自回归成熟。
- 与 RLHF/DPO 等对齐技术的结合尚在探索中。

### 6.5 其他方法

- **Plaid（Sahoo et al., 2024）**：改进离散扩散的采样效率和质量。
- **MDLM（Molecular Dynamics Language Model）**：连续空间扩散 + 离散化。
- **SEDD（Score Entropy Discrete Diffusion）**：基于得分的离散扩散框架。
- **SunDA**：统一自回归和扩散的混合模型。

---

## 7. Blockwise 并行生成

### 7.1 半自回归

Blockwise 或 Semi-Autoregressive 方法将序列分成块（block），块间串行、块内并行：

- 每个块包含 $k$ 个连续 Token。
- 块内 Token 由专门的解码头并行预测。
- 块之间保持自回归依赖，确保一致性。

### 7.2 Blockwise Parallel Decoding

Blockwise Parallel Decoding（Stern et al., 2018）是早期探索，近年因推理优化需求被重新关注：

- 模型在每步预测一个块（block）而非单个 Token。
- 块内用条件独立假设或轻量依赖建模。
- 用验证机制（类似投机解码）保证输出质量。

### 7.3 混合架构

- **AR + NAR 混合**：用 AR 模型生成大纲/关键 Token，NAR 模型填充细节。
- **草稿-验证范式**：NAR 生成块草稿，AR 验证。
- **多 Token 预测（Multi-Token Prediction, MTP）**：训练模型直接预测多个未来 Token，推理时用于投机解码草稿（DeepSeek-V3 采用）。

---

## 8. 实践建议

### 8.1 加速方案选型

- **零训练成本、即插即用**：投机解码。用同系列小模型（如 Llama-3-8B 草稿 → Llama-3-70B 目标）或量化版作为草稿。
- **可少量训练、追求更高加速**：Medusa 或 EAGLE，在目标模型上训练额外头。
- **愿意重新训练、追求极致延迟**：关注扩散 LM（LLaDA）和掩码生成（Mercury），但需评估质量损失。
- **代码/结构化文本**：投机解码接受率通常更高，因为草稿模型对模式化内容预测准确。

### 8.2 投机解码实践

- 草稿模型与目标模型必须共享同一分词器。
- 草稿长度 $k$ 需根据接受率调整：接受率高时增大 $k$，低时减小。
- 典型配置：草稿模型参数量为目标模型的 1/10 到 1/5。
- vLLM、TensorRT-LLM、TGI 等主流推理框架已内置投机解码支持。

### 8.3 研究方向建议

- 扩散 LM 当前最适合需要 infilling 和双向注意力的场景（代码补全、文本编辑）。
- 关注 MTP（Multi-Token Prediction）作为训练目标——即使推理仍用自回归，MTP 训练也能提升模型规划能力。
- 非自回归方法与长上下文推理的结合是重要方向。

---

## 9. 参考文献

[1] Y. Leviathan, M. Kalman, Y. Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML, 2023.

[2] C. Chen, S. Borgeaud, G. Irving, et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv preprint, 2023.

[3] T. Cai, Y. Li, Z. Geng, et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*, arXiv preprint, 2024.

[4] Y. Li, D. Li, Y. Bai, et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, ICML, 2024.

[5] H. Chang, H. Zhang, L. Jiang, et al., *Muse: Text-To-Image Generation via Masked Generative Transformers*, ICML, 2023.

[6] C. Lu, Z. Zhou, F. Bao, et al., *Advancing Generative AI Through Discrete Diffusion: A Survey*, arXiv preprint, 2024.

[7] N. Z. Wang, E. Wang, J. J. Sun, et al., *LLaDA: Large Language Diffusion Models*, arXiv preprint, 2025.

[8] M. Stern, N. Shazeer, J. Uszkoreit, *Blockwise Parallel Decoding for Deep Autoregressive Models*, EMNLP, 2018.

[9] A. Santilli, S. Severo, S. C. Chan, et al., *Accelerating Inference for Pretrained Language Models by Unified Multi-Path Early Exit*, arXiv preprint, 2023.

[10] DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024.
