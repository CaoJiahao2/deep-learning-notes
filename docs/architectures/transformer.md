# Transformer 架构详解

> 基于论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) 的完整解析。

---

## 目录

1. [背景与动机](#1)
2. [整体架构](#2)
3. [Self-Attention 机制](#3-self-attention)
4. [Multi-Head Attention](#4-multi-head-attention)
5. [Positional Encoding](#5-positional-encoding)
6. [Feed-Forward Network](#6-feed-forward-network)
7. [残差连接与 Layer Normalization](#7-layer-normalization)
8. [Encoder 详解](#8-encoder)
9. [Decoder 详解](#9-decoder)
10. [训练与推理](#10)
11. [Transformer 变体](#11-transformer)
12. [参考文献](#12)

---

## 1. 背景与动机

在 Transformer 出现之前，序列建模主要由 RNN、LSTM 和 GRU 主导。这些模型的核心问题是：

- **顺序计算限制**：RNN 必须按时间步依次计算，无法并行化，导致训练效率低下
- **长程依赖问题**：尽管 LSTM/GRU 引入了门控机制，但处理长序列时梯度消失问题依然存在
- **计算复杂度**：每个时间步的隐藏状态依赖前一步，难以利用 GPU 并行优势

Transformer 的核心创新：**完全摒弃循环结构，仅依赖注意力机制（Self-Attention）进行序列建模**，实现了：

- 训练时完全并行化
- 任意两个位置之间的路径长度恒为 O(1)
- 可扩展性强，支持超大规模预训练

---

## 2. 整体架构

Transformer 采用经典的 **Encoder-Decoder** 结构：

```
输入序列 → [Encoder × N] → [Decoder × N] → 输出序列
```

**Encoder**：由 6 个相同的层堆叠而成，每层包含：
- Multi-Head Self-Attention
- Position-wise Feed-Forward Network
- 两个子层均使用残差连接 + Layer Normalization

**Decoder**：同样由 6 个相同的层堆叠，每层包含：
- Masked Multi-Head Self-Attention（防止看到未来位置）
- Multi-Head Cross-Attention（关注 Encoder 输出）
- Position-wise Feed-Forward Network
- 三个子层均使用残差连接 + Layer Normalization

关键维度参数（论文 base 模型）：
- 模型维度 $d_{model} = 512$
- 注意力头数 $h = 8$
- FFN 中间维度 $d_{ff} = 2048$
- Encoder/Decoder 层数 $N = 6$

---

## 3. Self-Attention 机制

### 3.1 核心思想

Self-Attention 允许序列中的每个位置都关注（attend to）序列中的所有位置，从而捕获全局依赖关系。

### 3.2 Scaled Dot-Product Attention

给定输入序列 $X \in \mathbb{R}^{n \times d_{model}}$，通过三个线性变换生成 Query、Key、Value：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

其中 $W^Q, W^K \in \mathbb{R}^{d_{model} \times d_k}$，$W^V \in \mathbb{R}^{d_{model} \times d_v}$。通常 $d_k = d_v = d_{model} / h = 64$。

注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**缩放因子 $\sqrt{d_k}$ 的作用**：当 $d_k$ 较大时，点积结果的值域变大，导致 softmax 梯度进入极小区域。除以 $\sqrt{d_k}$ 使方差保持为 1，保证梯度稳定。

### 3.3 计算复杂度

- 矩阵乘法 $QK^T$：$O(n^2 \cdot d_k)$
- 空间复杂度：$O(n^2)$（存储注意力矩阵）
- 这是 Transformer 处理长序列的主要瓶颈

### 3.4 注意力可视化示例

对于输入句子 "The cat sat on the mat"，每个词的 Query 会与所有词的 Key 计算相似度，得到注意力权重矩阵：

```
        The   cat   sat   on   the   mat
The    [0.3, 0.1, 0.2, 0.1, 0.2, 0.1]
cat    [0.1, 0.4, 0.2, 0.1, 0.1, 0.1]
sat    [0.1, 0.1, 0.4, 0.2, 0.1, 0.1]
on     [0.0, 0.0, 0.2, 0.5, 0.1, 0.2]
the    [0.2, 0.0, 0.1, 0.1, 0.4, 0.2]
mat    [0.1, 0.0, 0.1, 0.2, 0.2, 0.4]
```

---

## 4. Multi-Head Attention

### 4.1 动机

单个注意力头只能关注一种模式。Multi-Head Attention 将 $Q, K, V$ 投影到 $h$ 个不同的子空间，并行计算注意力，使模型能同时关注不同位置的不同表示子空间。

### 4.2 数学形式

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

其中：
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$

### 4.3 实际意义

不同注意力头可以学习不同的关注模式：
- 头 1：关注相邻词（局部句法）
- 头 2：关注主语-谓语-宾语关系
- 头 3：关注指代关系（代词 → 先行词）
- 头 4：关注长距离依赖

---

## 5. Positional Encoding

### 5.1 为什么需要位置编码

Self-Attention 本身是**置换等变（permutation equivariant）**的——打乱输入顺序，输出也相应打乱。因此需要显式注入位置信息。

### 5.2 正弦位置编码

论文使用固定正弦函数：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中 $pos$ 是位置索引，$i$ 是维度索引。

**关键性质**：
- 对于固定偏移 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数（因为三角恒等式），这有助于模型学习相对位置关系
- 不同频率的正弦波覆盖了从短距离到长距离的位置模式
- 值域在 $[-1, 1]$ 之间，数值稳定

### 5.3 其他位置编码方案

| 方法 | 代表 | 特点 |
|------|------|------|
| 可学习位置编码 | BERT, GPT | 直接学习位置嵌入，简单但无法外推 |
| 相对位置编码 | Transformer-XL, T5 | 编码相对距离而非绝对位置 |
| 旋转位置编码 (RoPE) | LLaMA, Qwen, ChatGLM | 通过旋转矩阵注入相对位置信息；训练长度之外需配合 YaRN / NTK-aware / 位置插值等缩放方法外推 |
| ALiBi | BLOOM | 在注意力分数上加线性偏置，极强外推能力 |

---

## 6. Feed-Forward Network

每个 Encoder 和 Decoder 层都包含一个 Position-wise FFN：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

或使用 GELU 等激活函数：

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

**现代大模型多使用 GLU 变体（SwiGLU / GeGLU）**，将单个线性层替换为门控结构：

$$\text{SwiGLU}(x) = \underbrace{\text{Swish}(xW_1)}_{\text{gate}} \odot \underbrace{(xW_3)}_{\text{value}}, \qquad \text{FFN}_{LLaMA}(x) = \text{SwiGLU}(x)W_2$$

其中 $\text{Swish}(x) = x\sigma(x)$，$\odot$ 为逐元素乘。相比 ReLU FFN，GLU 变体在相近参数量下通常在语言建模上更优，是 LLaMA、Mistral、Qwen 等的主流选择。

- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- 中间维度 $d_{ff}$ 通常是 $d_{model}$ 的 4 倍
- FFN 可以看作两个 1×1 卷积，对每个位置独立应用

**FFN 的作用**：
- 提供非线性变换能力
- 存储大量知识（在大模型中，FFN 被视为知识存储的主要位置）
- 与 Self-Attention 互补：Attention 负责"检索"，FFN 负责"处理"

---

## 7. 残差连接与 Layer Normalization

### 7.1 残差连接

每个子层的输出：

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

残差连接使梯度可以直通底层，有效缓解深层网络的梯度消失问题。

### 7.2 Layer Normalization

与 Batch Normalization 不同，LayerNorm 在特征维度上归一化：

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu$ 和 $\sigma^2$ 在最后一个维度上计算。

**RMSNorm（LLaMA 等现代模型）**：去掉均值归一化，仅保留缩放，计算更省且训练稳定：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma$$

**Pre-Norm vs Post-Norm**：
- 原始论文使用 Post-Norm（先残差后归一化）
- 现代实践（GPT, LLaMA）多用 Pre-Norm（先归一化后残差），训练更稳定

---

## 8. Encoder 详解

```text
Input Embedding + Positional Encoding
    ↓
┌─────────────────────────────┐
│  Multi-Head Self-Attention  │
│  Add & LayerNorm            │
│  Feed-Forward Network       │
│  Add & LayerNorm            │
└─────────────────────────────┘  × 6
    ↓
Encoder Output
```

**Encoder Self-Attention 的特点**：
- 可以关注所有位置（双向）
- 每个位置都能看到整个输入序列
- 适合理解类任务（分类、序列标注、BERT 预训练）

---

## 9. Decoder 详解

```text
Output Embedding + Positional Encoding (shifted right)
    ↓
┌──────────────────────────────┐
│  Masked Multi-Head Self-Attn │  ← 只能看到当前位置及之前
│  Add & LayerNorm             │
│  Multi-Head Cross-Attention  │  ← Q 来自 Decoder, K,V 来自 Encoder
│  Add & LayerNorm             │
│  Feed-Forward Network        │
│  Add & LayerNorm             │
└──────────────────────────────┘  × 6
    ↓
Linear + Softmax → Output probabilities
```

### 9.1 Masked Self-Attention

通过上三角掩码矩阵将未来位置的注意力分数设为 $-\infty$，使 softmax 后权重为 0：

```
Attention(Q, K, V) = softmax(QK^T/√d_k + M)V

其中 M 为上三角矩阵，M[i,j] = -∞ if j > i else 0
```

### 9.2 Cross-Attention

Decoder 的 Cross-Attention 层中：
- Query 来自 Decoder 的上一子层输出
- Key 和 Value 来自 Encoder 的最终输出
- 允许 Decoder 在生成每个 token 时"参考"整个输入序列

### 9.3 自回归生成

训练时使用 Teacher Forcing（并行输入完整目标序列），推理时自回归逐 token 生成。

---

## 10. 训练与推理

### 10.1 训练配置（论文 base 模型）

| 超参数 | 值 |
|--------|-----|
| 优化器 | Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$) |
| 学习率调度 | Warmup + Inverse Square Root Decay |
| Warmup 步数 | 4000 |
| Dropout | 0.1 |
| Label Smoothing | 0.1 |
| Batch Size | 约 25,000 源/目标 tokens |
| 训练步数 | 100,000 |

### 10.2 学习率调度

$$lr = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})$$

即在 warmup 阶段线性增长，之后按步数的平方根倒数衰减。

### 10.3 推理

- 自回归解码（Autoregressive Decoding）
- Beam Search（论文使用 beam size = 4，长度惩罚 $\alpha = 0.6$）

---

## 11. Transformer 变体

### 11.1 Encoder-Only（BERT 系列）

- **BERT**：双向 Self-Attention + MLM 预训练
- **RoBERTa**：优化训练策略的 BERT
- **DeBERTa**：解耦注意力机制

### 11.2 Decoder-Only（GPT 系列）

- **GPT / GPT-2 / GPT-3 / GPT-4**：自回归语言模型
- **LLaMA / LLaMA 2 / LLaMA 3**：Meta 开源系列
- **Qwen / DeepSeek / Mistral**：社区开源系列

### 11.3 Encoder-Decoder（T5 系列）

- **T5**：将所有 NLP 任务统一为 text-to-text 格式
- **BART**：去噪自编码器

### 11.4 高效注意力变体

| 变体 | 复杂度 | 原理 |
|------|--------|------|
| Sparse Attention | $O(n\sqrt{n})$ | 只关注部分位置 |
| Linformer | $O(n)$ | 低秩投影 K, V |
| Reformer | $O(n\log n)$ | LSH 近似 |
| Flash Attention | 同复杂度，但 IO 最优 | 利用 GPU 内存层次结构 |
| Multi-Query Attention | 减少 KV 头 | 多个 Q 头共享一组 K, V |
| Grouped-Query Attention | 折中方案 | 将 Q 头分组，每组共享 K, V |

---


### 11.5 现代注意力与 KV 压缩（2023–2025）

| 技术 | 核心思想 | 收益 |
|------|---------|------|
| **Multi-head Latent Attention (MLA)** | 对 KV 做低秩压缩（DeepSeek-V2），推理时仅缓存压缩后的 latent 与少量投影 | KV Cache 大幅下降，推理成本显著降低 |
| **FlashDecoding** | 将长序列的 Query 分块并行，每块独立算 softmax 再合并 | 长序列 decode 吞吐提升 |
| **Ring Attention** | 将 KV 分块分布在多设备并循环传输，attention 计算与通信重叠 | 支持超长上下文（百万级 token） |
| **Sliding Window + 长上下文** | 局部窗口注意力 + 少量全局 token | 长序列显存可控 |

**MLA 示意**：设 $\mathbf{c}_t = W^{DKV} \mathbf{h}_t$ 为压缩的 KV latent，推理时仅缓存 $\mathbf{c}_t$（远小于完整 K/V），注意力时再上投影恢复 $\mathbf{k}_t, \mathbf{v}_t$。

### 11.6 线性注意力与状态空间模型（SSM）

标准 attention 的 $O(n^2)$ 复杂度是长序列的瓶颈。**线性注意力/SSM** 通过将注意力改写为可递推的线性形式，把复杂度降到 $O(n)$：

- **Mamba（S6）**：引入输入依赖的选择性扫描，隐藏状态 $h_t$ 按 $\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t \mathbf{x}_t$ 递推，$\bar{\mathbf{A}}, \bar{\mathbf{B}}$ 由输入生成。
- **Mamba-2 / SSD**：将选择性 SSM 与结构化矩阵乘法统一，可用 GPU 矩阵乘法高效实现。
- **RWKV / xLSTM / Gated DeltaNet**：RNN 式线性递推 + 门控，兼顾并行训练与线性推理。
- **Jamba / Samba**：SSM 与 attention 的混合架构，兼顾长程建模与内容检索。

这些模型在**超长上下文**与**低成本推理**场景正成为 Transformer 的重要补充方向。


## 12. 参考文献

[1] Vaswani, A., et al. Attention Is All You Need. NeurIPS, 2017.

[2] Devlin, J., et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019.

[3] Brown, T., et al. Language Models are Few-Shot Learners. NeurIPS, 2020.

[4] Touvron, H., et al. LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971, 2023.

[5] Su, J., et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864, 2021.

[6] Dao, T., et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS, 2022.

[7] Shazeer, N. Fast Transformer Decoding: One Write-Head Is All You Need. arXiv:1911.02150, 2019.

[8] Press, O., et al. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. ICLR, 2022.

[9] DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv:2405.04434, 2024. (MLA)

[10] Gu, A. & Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. COLM, 2024.

[11] Dao, T. & Gu, A. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML, 2024.

[12] Liu, Z., et al. Ring Attention with Blockwise Transformers for Near-Infinite Context. ICLR, 2024.

[13] Shazeer, N. GLU Variants Improve Transformer. arXiv:2002.05202, 2020.

[14] Zhang, B. & Sennrich, R. Root Mean Square Layer Normalization. NeurIPS, 2019.
