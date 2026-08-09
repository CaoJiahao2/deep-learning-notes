# 循环神经网络 (RNN / LSTM / GRU) 详解

> 序列建模的经典架构，从基础 RNN 到现代门控机制的完整解析。

---

## 目录

1. [RNN 基础](#1-rnn)
2. [LSTM 长短期记忆网络](#2-lstm)
3. [GRU 门控循环单元](#3-gru)
4. [双向 RNN](#4-rnn)
5. [深度 RNN 与堆叠](#5-rnn)
6. [序列到序列模型 (Seq2Seq)](#6-seq2seq)
7. [注意力机制](#7)
8. [RNN 的局限与 Transformer 的崛起](#8-rnn-transformer)
9. [参考文献](#9)

---

## 1. RNN 基础

### 1.1 核心思想

RNN 通过**隐藏状态**在时间步之间传递信息，实现对序列数据的建模。

### 1.2 数学形式

给定输入序列 $x_1, x_2, \ldots, x_T$，RNN 在每个时间步 $t$ 更新隐藏状态：

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

其中：
- $h_t \in \mathbb{R}^d$：时刻 $t$ 的隐藏状态
- $W_{hh} \in \mathbb{R}^{d \times d}$：隐藏状态到隐藏状态的权重
- $W_{xh} \in \mathbb{R}^{d \times d_{in}}$：输入到隐藏状态的权重
- $W_{hy} \in \mathbb{R}^{d_{out} \times d}$：隐藏状态到输出的权重

### 1.3 时间反向传播 (BPTT)

RNN 通过时间反向传播（Backpropagation Through Time）展开计算图进行梯度计算。

**梯度消失/爆炸问题**：

展开 $k$ 步后的梯度：

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{i=t}^{T-1} \frac{\partial h_{i+1}}{\partial h_i}$$

其中 $\frac{\partial h_{i+1}}{\partial h_i} = W_{hh}^T \cdot \text{diag}(\tanh'(...))$。

- 当 $W_{hh}$ 特征值 $< 1$：梯度指数衰减 → **梯度消失**
- 当 $W_{hh}$ 特征值 $> 1$：梯度指数增长 → **梯度爆炸**

**解决方案**：
- 梯度爆炸：梯度裁剪（Gradient Clipping）
- 梯度消失：门控机制（LSTM/GRU）

---

## 2. LSTM 长短期记忆网络

### 2.1 核心思想

LSTM (Hochreiter & Schmidhuber, 1997) 引入**细胞状态**（Cell State）和**门控机制**来控制信息流动。

### 2.2 LSTM 结构

```text
        遗忘门              输入门               输出门
    f_t = σ(W_f·[h_{t-1}, x_t] + b_f)    i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
    C̃_t = tanh(W_C·[h_{t-1}, x_t] + b_C)    o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
    
    细胞状态更新：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
    隐藏状态更新：h_t = o_t ⊙ tanh(C_t)
```

### 2.3 门控详解

**遗忘门（Forget Gate）** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
- 决定保留多少旧细胞状态信息
- $f_t \to 0$：遗忘；$f_t \to 1$：保留

**输入门（Input Gate）** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
- 决定将多少新信息写入细胞状态
- 与候选值 $\tilde{C}_t$ 配合使用

**候选细胞状态（Candidate Cell State）** $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
- 生成新的候选信息

**输出门（Output Gate）** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$
- 决定输出多少细胞状态信息到隐藏状态

### 2.4 为什么 LSTM 缓解梯度消失

细胞状态的更新是**线性加法**：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

当 $f_t = 1$ 时，$C_t = C_{t-1} + \Delta$，梯度可以通过细胞状态**无衰减地反向传播**。这被称为 Constant Error Carousel (CEC)。

### 2.5 参数量

LSTM 有 4 组参数（遗忘门、输入门、候选值、输出门），每组权重矩阵大小为 $d \times (d + d_{in})$：

$$\text{参数总数} = 4 \times d \times (d + d_{in}) + 4 \times d$$

---

## 3. GRU 门控循环单元

### 3.1 核心思想

GRU (Cho et al., 2014) 是 LSTM 的简化版本，将遗忘门和输入门合并为**更新门**，同时移除独立的细胞状态。

### 3.2 GRU 结构

```text
    重置门：r_t = σ(W_r·[h_{t-1}, x_t] + b_r)
    更新门：z_t = σ(W_z·[h_{t-1}, x_t] + b_z)
    
    候选隐藏状态：h̃_t = tanh(W_h·[r_t ⊙ h_{t-1}, x_t] + b_h)
    隐藏状态更新：h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

### 3.3 门控作用

- **重置门** $r_t$：控制忽略多少历史信息
  - $r_t \to 0$：忽略过去，只关注当前输入（适合检测新主题）
  - $r_t \to 1$：充分利用历史信息
- **更新门** $z_t$：控制保留多少历史 vs 新信息
  - $z_t \to 0$：保留全部历史（类似 LSTM 的 $f_t=1$）
  - $z_t \to 1$：完全使用新信息

### 3.4 LSTM vs GRU 对比

| 维度 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3（遗忘、输入、输出） | 2（重置、更新） |
| 状态 | 细胞状态 + 隐藏状态 | 仅隐藏状态 |
| 参数量 | $4d(d + d_{in})$ | $3d(d + d_{in})$ |
| 性能 | 任务相关，通常相当 | 任务相关，通常相当 |
| 训练速度 | 较慢 | 较快（约 20%） |
| 何时使用 | 需要精细化门控 | 数据不足或需快速训练 |

**经验法则**：没有绝对优势，建议两种都尝试。GRU 参数量少 25%，在小数据集上可能更好。

---

## 4. 双向 RNN

### 4.1 动机

许多序列任务中，当前位置的输出不仅依赖前文，也依赖后文（如命名实体识别、机器翻译）。

### 4.2 结构

$$\overrightarrow{h_t} = \text{RNN}_{\text{forward}}(x_t, \overrightarrow{h_{t-1}})$$

$$\overleftarrow{h_t} = \text{RNN}_{\text{backward}}(x_t, \overleftarrow{h_{t+1}})$$

$$y_t = W_y[\overrightarrow{h_t}; \overleftarrow{h_t}] + b_y$$

### 4.3 限制

- 需要完整序列才能计算，不适合流式/在线推理
- 不适用于自回归生成任务（Decoder 不能看到未来）

---

## 5. 深度 RNN 与堆叠

### 5.1 多层 RNN

将多个 RNN 层堆叠，每层的输出作为下一层的输入：

$$h_t^{(l)} = \text{RNN}(h_t^{(l-1)}, h_{t-1}^{(l)})$$

通常 2-4 层就足够，过深会导致梯度问题。

### 5.2 正则化技术

- **Dropout**：应用于层间连接（非循环连接），$\hat{h}_t^{(l)} = \text{Dropout}(h_t^{(l)})$
- **Recurrent Dropout**：应用于循环连接，对隐藏状态的更新进行 dropout
- **Layer Normalization**：在 RNN 内部添加 LayerNorm 稳定训练

---

## 6. 序列到序列模型 (Seq2Seq)

### 6.1 架构

Encoder-Decoder 结构：
- **Encoder**：将输入序列编码为上下文向量 $c$
- **Decoder**：从 $c$ 自回归生成输出序列

$$c = \text{Encoder}(x_1, \ldots, x_T)$$

$$P(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = \prod_{t=1}^{T'} P(y_t \mid y_{<t}, c)$$

### 6.2 瓶颈问题

所有输入信息被压缩到单个固定维度的向量 $c$ 中，长序列信息丢失严重。

---

## 7. 注意力机制

### 7.1 Bahdanau Attention (Additive Attention)

为解决 Seq2Seq 的瓶颈问题，Bahdanau et al. (2015) 提出注意力机制。

**对齐分数**：

$$e_{t,i} = v_a^T \tanh(W_a s_{t-1} + U_a h_i)$$

**注意力权重**：

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T} \exp(e_{t,j})}$$

**上下文向量**：

$$c_t = \sum_{i=1}^{T} \alpha_{t,i} h_i$$

### 7.2 Luong Attention (Multiplicative Attention)

三种对齐分数计算方式：

| 方式 | 公式 |
|------|------|
| Dot | $e_{t,i} = s_t^T h_i$ |
| General | $e_{t,i} = s_t^T W_a h_i$ |
| Concat | $e_{t,i} = v_a^T \tanh(W_a[s_t; h_i])$ |

### 7.3 关键区别

- Bahdanau：使用 $s_{t-1}$（上一时刻隐藏状态），先计算注意力，再生成当前输出
- Luong：使用 $s_t$（当前时刻隐藏状态），先生成当前输出，再计算注意力

### 7.4 注意力变体

| 变体 | 说明 |
|------|------|
| Global Attention | 关注所有 Encoder 位置 |
| Local Attention | 只关注一个窗口内的位置 |
| Self-Attention | 序列内部关注（Transformer 的基础） |
| Multi-Head Attention | 多个注意力头并行 |

---

## 8. RNN 的局限与 Transformer 的崛起

### 8.1 RNN 的根本局限

| 问题 | 说明 |
|------|------|
| 顺序计算 | 无法并行化，训练慢 |
| 长程依赖 | 即使 LSTM 也受限于梯度路径长度 |
| 内存瓶颈 | BPTT 需要保存所有中间激活值 |
| 数值不稳定 | 长序列仍存在梯度爆炸风险 |

### 8.2 从 RNN 到 Transformer

Transformer 的 Self-Attention 解决了这些问题：

- 任意两个位置间的路径长度恒为 $O(1)$
- 训练时完全并行化
- 更适合 GPU 计算模式

### 8.3 RNN 仍然适用的场景

- 在线流式处理（逐元素输入，无法预知序列长度）
- 极长序列（Transformer 的 $O(n^2)$ 复杂度是瓶颈）
- 嵌入式设备（LSTM 推理速度可能优于小型 Transformer）
- 时间序列预测（Mamba/SSM 等新架构兴起）

---

## 9. 参考文献

[1] Hochreiter, S. & Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.

[2] Cho, K., et al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP, 2014.

[3] Sutskever, I., et al. Sequence to Sequence Learning with Neural Networks. NeurIPS, 2014.

[4] Bahdanau, D., et al. Neural Machine Translation by Jointly Learning to Align and Translate. ICLR, 2015.

[5] Luong, M.-T., et al. Effective Approaches to Attention-based Neural Machine Translation. EMNLP, 2015.

[6] Vaswani, A., et al. Attention Is All You Need. NeurIPS, 2017.

[7] Graves, A. Generating Sequences With Recurrent Neural Networks. arXiv:1308.0850, 2013.

[8] Gal, Y. & Ghahramani, Z. A Theoretically Grounded Application of Dropout in Recurrent Neural Networks. NeurIPS, 2016.
