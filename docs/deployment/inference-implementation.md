# 大模型推理实现详解：从 Transformer Forward 到高性能 Serving

> 本文从系统实现视角解释现代大语言模型（LLM）推理是如何完成的，重点覆盖 Prefill、Decode、KV Cache、Continuous Batching、PagedAttention、FlashAttention、Speculative Decoding、量化、多 GPU 并行、MoE 推理和 Prefill-Decode 解耦等核心机制。

---

## 目录

1. [什么是“大模型推理”](#1)
2. [推理的两个核心阶段：Prefill 与 Decode](#2)
3. [Prefill 阶段](#3)
4. [Transformer 层在 Prefill 中做什么](#4)
5. [为什么需要 KV Cache](#5)
6. [KV Cache 的基本思想](#6)
7. [KV Cache 的显存占用](#7)
8. [MHA、MQA 与 GQA 为什么影响推理性能](#8)
9. [Decode 阶段](#9)
10. [Decode 的矩阵运算为什么很特殊](#10)
11. [为什么 LLM Decode 经常受显存带宽限制](#11)
12. [为什么需要 Batch](#12)
13. [Static Batching 的问题](#13)
14. [Continuous Batching](#14)
15. [Scheduler 是推理引擎的核心组件](#15)
16. [两个重要延迟指标](#16)
17. [KV Cache 的内存碎片问题](#17)
18. [PagedAttention](#18)
19. [Paged KV Cache 的优点](#19)
20. [Prefix Caching](#20)
21. [Radix Tree / Prefix Tree](#21)
22. [Chunked Prefill](#22)
23. [FlashAttention](#23)
24. [FlashAttention 的基本思想](#24)
25. [Kernel Fusion](#25)
26. [CUDA Graph](#26)
27. [Quantization 为什么对推理特别重要](#27)
28. [常见量化方式](#28)
29. [Speculative Decoding](#29)
30. [Speculative Decoding 的流程](#30)
31. [Draft Model](#31)
32. [Multi-Token Prediction / MTP](#32)
33. [Sampling](#33)
34. [Multi-GPU 推理](#34)
35. [Tensor Parallel](#35)
36. [Pipeline Parallel](#36)
37. [Data Parallel](#37)
38. [MoE 推理](#38)
39. [Expert Parallel](#39)
40. [Prefill-Decode Disaggregation](#40)
41. [为什么要做 PD 解耦](#41)
42. [PD 解耦最大的难点](#42)
43. [长上下文为什么难](#43)
44. [长上下文优化](#44)
45. [MLA 与 KV Cache](#45)
46. [KV Cache Quantization](#46)
47. [Offloading](#47)
48. [一次完整请求的生命周期](#48)
49. [一个更完整的 Serving 架构](#49)
50. [为什么现代推理引擎不是简单的 model.generate()](#50)
51. [vLLM 可以如何理解](#51)
52. [SGLang 可以如何理解](#52)
53. [TensorRT-LLM 可以如何理解](#53)
54. [推理性能的三个核心指标](#54)
55. [Latency 与 Throughput 的冲突](#55)
56. [Roofline 视角理解 LLM 推理](#56)
57. [Prefill 应该优化什么](#57)
58. [Decode 应该优化什么](#58)
59. [一个非常重要的直觉](#59)
60. [为什么 Batch 能提高 Decode 性能](#60)
61. [大模型推理优化的本质](#61)
62. [从研究角度建议的学习路线](#62)
63. [推荐建立的核心知识链](#63)
64. [最终总结](#64)
65. [关键词速查](#kw)

---

## 1. 什么是“大模型推理” {#1}

大语言模型推理（LLM Inference），本质上是：

1. 将用户输入文本转换为 token；
2. 使用 Transformer 对已有 token 序列做前向计算；
3. 得到下一个 token 的概率分布；
4. 按照 greedy、temperature、top-k、top-p 等策略选出下一个 token；
5. 将新 token 加入上下文；
6. 重复上述过程，直到生成 EOS、达到长度上限或触发其他停止条件。

对于一个 token 序列：

$$
x_1,x_2,\dots,x_t
$$

模型学习的是条件概率：

$$
P(x_{t+1}\mid x_1,x_2,\dots,x_t)
$$

因此，GPT 类大语言模型的生成过程天然是**自回归（autoregressive）**的。

例如：

```text
输入：
中国的首都是

模型预测：
北京

新的上下文：
中国的首都是北京

继续预测：
。
```

从算法层面看，这只是不断执行 Transformer Forward。

但从系统层面看，大模型推理的真正难点在于：

- 如何减少重复计算；
- 如何充分利用 GPU；
- 如何管理巨大的 KV Cache；
- 如何同时服务大量不同长度的请求；
- 如何减少显存带宽压力；
- 如何让多张 GPU 高效协作；
- 如何降低单 token 延迟并提高总体吞吐。

现代 LLM 推理引擎，例如 vLLM、SGLang、TensorRT-LLM，主要就在解决这些问题。

---

## 2. 推理的两个核心阶段：Prefill 与 Decode {#2}

现代 LLM 推理通常被划分为两个性质明显不同的阶段：

```text
Prompt
  ↓
Tokenizer
  ↓
┌─────────────────────┐
│       Prefill       │
│ 一次处理整个 Prompt │
│ 建立 KV Cache       │
└─────────────────────┘
          ↓
     第一个输出 token
          ↓
┌─────────────────────┐
│       Decode        │
│ 每轮生成一个 token  │
│ 复用历史 KV Cache   │
└─────────────────────┘
          ↓
     token → token → ...
```

这两个阶段虽然都运行同一个 Transformer，但它们对应的计算模式完全不同。

---

## 3. Prefill 阶段 {#3}

假设用户输入长度为 $L$：

```text
x_1, x_2, ..., x_L
```

Prefill 会一次性将整个 Prompt 输入模型。

如果隐藏层维度为 $d$，模型中的线性层通常会出现类似：

$$
[L,d]\times[d,4d]
$$

这样的矩阵乘法。

由于 $L$ 往往可以达到几百、几千甚至几万，GPU 可以在大量 token 上并行执行矩阵运算，因此 Prefill 阶段通常能够较充分地利用 Tensor Core。

所以：

> Prefill 一般更偏向 compute-bound。

也就是说，性能更容易受到 GPU 算力限制。

---

## 4. Transformer 层在 Prefill 中做什么 {#4}

一个现代 Decoder-only Transformer Layer 可以抽象为：

```text
Input
  ↓
RMSNorm
  ↓
Q / K / V Projection
  ↓
RoPE
  ↓
Causal Self-Attention
  ↓
Output Projection
  ↓
Residual
  ↓
RMSNorm
  ↓
MLP / MoE
  ↓
Residual
```

Attention 的基本计算是：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

然后：

$$
A=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_h}}
+\text{CausalMask}
\right)
$$

最后：

$$
O=AV
$$

其中 Causal Mask 保证第 $i$ 个 token 只能看到：

$$
x_1,\dots,x_i
$$

而不能看到未来 token。

---

## 5. 为什么需要 KV Cache {#5}

如果没有 KV Cache，每生成一个 token，都需要重新计算整个历史上下文。

例如已有：

```text
中国 的 首都 是 北京
```

生成下一个 token 时，最朴素的方法会再次计算：

```text
中国
中国 的
中国 的 首都
中国 的 首都 是
中国 的 首都 是 北京
```

这会产生巨大的重复计算。

Transformer Attention 中，历史 token 的：

$$
K_i,\quad V_i
$$

在后续 Decode 阶段不会发生变化。

因此可以把它们缓存起来：

```text
Layer 0:
K1 K2 K3 ... KL
V1 V2 V3 ... VL

Layer 1:
K1 K2 K3 ... KL
V1 V2 V3 ... VL

...

Layer N:
K1 K2 K3 ... KL
V1 V2 V3 ... VL
```

这就是：

### KV Cache

---

## 6. KV Cache 的基本思想 {#6}

假设已经有：

$$
x_1,\dots,x_t
$$

并缓存了：

$$
K_1,\dots,K_t
$$

$$
V_1,\dots,V_t
$$

现在只需要对新 token $x_{t+1}$ 计算：

$$
q_{t+1},k_{t+1},v_{t+1}
$$

随后：

$$
K_{\text{cache}}
=
[K_1,\dots,K_t,k_{t+1}]
$$

$$
V_{\text{cache}}
=
[V_1,\dots,V_t,v_{t+1}]
$$

新的 Attention 只需要执行：

$$
\operatorname{Attention}_{t+1}
=
\operatorname{softmax}
\left(
\frac{
q_{t+1}K_{\text{cache}}^T
}{
\sqrt{d_h}
}
\right)
V_{\text{cache}}
$$

这样就避免了重新计算历史 token 的 K 和 V。

---

## 7. KV Cache 的显存占用 {#7}

KV Cache 是 LLM Serving 中最重要的显存消耗来源之一。

粗略地说：

$$
M_{KV}
\approx
2
\times
N_{\text{layer}}
\times
L
\times
N_{\text{kv-head}}
\times
d_{\text{head}}
\times
B
$$

其中：

- 2：分别表示 K 和 V；
- $N_{\text{layer}}$：Transformer 层数；
- $L$：上下文 token 数；
- $N_{\text{kv-head}}$：KV Head 数量；
- $d_{\text{head}}$：每个 Attention Head 的维度；
- $B$：每个元素占用的字节数。

例如 FP16 / BF16：

$$
B=2
$$

当同时服务大量用户时：

$$
\text{Total KV Memory}
=
\sum_i M_{KV}^{(i)}
$$

所以并发数和上下文长度会迅速消耗 GPU 显存。

---

## 8. MHA、MQA 与 GQA 为什么影响推理性能 {#8}

传统 Multi-Head Attention：

```text
Q heads: 32
K heads: 32
V heads: 32
```

KV Cache 很大。

MQA（Multi-Query Attention）：

```text
Q heads: 32
K heads: 1
V heads: 1
```

KV Cache 大幅减少。

GQA（Grouped-Query Attention）：

```text
Q heads: 32
KV heads: 8
```

多个 Query Head 共享一个 KV Head。

因此现代 LLM 大量使用 GQA。

主要原因之一就是：

> 大幅降低 Decode 阶段的 KV Cache 占用和显存带宽压力。

---

## 9. Decode 阶段 {#9}

Prefill 完成后，模型开始逐 token 生成。

例如：

```text
Prompt:
中国的首都是

Decode #1:
北京

Decode #2:
，

Decode #3:
是

Decode #4:
中国

...
```

每一轮 Decode 只输入：

```text
1 个新 token
```

然后生成下一个 token。

---

## 10. Decode 的矩阵运算为什么很特殊 {#10}

Prefill 中可能执行：

$$
[4096,d]\times[d,4d]
$$

但是 Decode 中通常是：

$$
[1,d]\times[d,4d]
$$

于是矩阵的 M 维变成 1。

这意味着：

- 并行度下降；
- GPU Tensor Core 很难像 Prefill 那样高效利用；
- 每生成一个 token，都需要读取大量模型参数；
- 同时还要读取历史 KV Cache。

因此：

> Decode 通常更偏 memory-bound。

也就是瓶颈更多来自：

$$
\text{HBM Bandwidth}
$$

而不是理论 FLOPS。

---

## 11. 为什么 LLM Decode 经常受显存带宽限制 {#11}

以一个几十 GB 权重的大模型为例。

每生成一个 token，都需要执行大量线性层：

```text
QKV Projection
Output Projection
Gate Projection
Up Projection
Down Projection
...
```

这些运算都需要访问模型权重。

如果 batch 很小，则：

```text
权重被读取一次
↓
只服务少量 token
```

计算强度很低。

所以常见现象是：

```text
GPU 算力没有跑满
但 HBM 带宽已经接近瓶颈
```

这就是 LLM Decode 和训练非常不同的地方。

训练通常有较大的矩阵：

$$
[B\times L,d]
$$

因此更容易把 GEMM 算力吃满。

---

## 12. 为什么需要 Batch {#12}

如果只服务一个用户：

```text
User A:
token 1
token 2
token 3
token 4
...
```

每次 Decode 只有：

$$
M=1
$$

GPU 利用率很低。

如果同时服务：

```text
A 当前 token
B 当前 token
C 当前 token
D 当前 token
...
```

可以把这些 token 拼成 batch：

$$
[B,d]\times[d,4d]
$$

随着 $B$ 增大，矩阵乘法效率明显提高。

所以 Serving 系统的一个核心任务是：

> 尽可能将不同用户当前的 Decode token 放到同一个 GPU batch 中计算。

---

## 13. Static Batching 的问题 {#13}

传统 Batch：

```text
A ────────────────
B ─────
C ───────────
D ───
```

如果必须等 batch 中所有 request 都结束：

```text
A 还在生成
B 已经结束，但占着 slot
C 还在生成
D 已经结束，但占着 slot
```

GPU 会产生大量浪费。

此外，不同 request：

- Prompt 长度不同；
- 输出长度不同；
- 到达时间不同。

因此很难使用传统静态 batch。

---

## 14. Continuous Batching {#14}

现代 LLM 推理引擎一般使用：

### Continuous Batching

或者称：

- Inflight Batching；
- Iteration-level Scheduling。

其核心思想是：

> 每个 Decode step 都重新决定当前 batch 里有哪些 request。

例如：

```text
Step 1:
A B C D

Step 2:
A B C D

此时 B 完成
E 新到达

Step 3:
A E C D
```

这样，新请求不必等待旧 batch 全部结束。

因此可以显著提升：

- GPU 利用率；
- 吞吐；
- 并发能力。

---

## 15. Scheduler 是推理引擎的核心组件 {#15}

现代 Serving Engine 中通常有一个 Scheduler：

```text
Incoming Requests
       ↓
   Scheduler
       ↓
┌───────────────┐
│ Prefill Queue │
└───────────────┘
       ↓
┌───────────────┐
│ Decode Queue  │
└───────────────┘
       ↓
     GPU
```

Scheduler 需要同时考虑：

- GPU 显存；
- KV Cache 可用空间；
- Batch token 数；
- 请求等待时间；
- Prefill 和 Decode 的竞争；
- 最大上下文长度；
- 请求优先级；
- TTFT；
- TPOT；
- Throughput。

因此 LLM Serving 本质上也是一个复杂的调度系统问题。

---

## 16. 两个重要延迟指标 {#16}

LLM Serving 中常用两个指标。

### 16.1 TTFT

Time To First Token：

$$
TTFT
=
t_{\text{first token}}
-
t_{\text{request arrival}}
$$

它主要受：

- Queueing；
- Prefill；
- Prompt 长度

影响。

---

### 16.2 TPOT

Time Per Output Token：

$$
TPOT
\approx
\frac{
t_{\text{generation}}
}{
N_{\text{output token}}
}
$$

它主要反映 Decode 性能。

对聊天场景来说：

- TTFT 决定“多久开始回答”；
- TPOT 决定“回答得顺不顺”。

---

## 17. KV Cache 的内存碎片问题 {#17}

假设有请求：

```text
A: context = 300 tokens
B: context = 8000 tokens
C: context = 1200 tokens
D: context = 60000 tokens
```

如果为每个请求预分配固定长度连续 KV Cache：

```text
A: [================]
B: [================]
C: [================]
D: [================]
```

会导致：

- 大量未使用空间；
- 内部碎片；
- 外部碎片；
- 难以动态扩展；
- 并发数下降。

于是 vLLM 引入了非常经典的 PagedAttention 思想。

---

## 18. PagedAttention {#18}

PagedAttention 的思想类似操作系统分页内存。

不再要求一个 request 的 KV Cache 在物理显存中连续。

逻辑上：

```text
Request A:

Block 0
Block 1
Block 2
Block 3
```

物理 GPU Memory 中可能是：

```text
Physical Block 17
Physical Block 3
Physical Block 41
Physical Block 8
```

通过一个 Block Table：

```text
Logical Block 0 → Physical Block 17
Logical Block 1 → Physical Block 3
Logical Block 2 → Physical Block 41
Logical Block 3 → Physical Block 8
```

实现动态映射。

---

## 19. Paged KV Cache 的优点 {#19}

主要包括：

### 1. 减少内存碎片

不需要为最大可能 sequence length 提前预留整块空间。

### 2. 支持动态增长

生成新 token 时只需要继续申请新的 block。

### 3. 提高显存利用率

可以将更多请求放入同一 GPU。

### 4. 便于 KV Cache Sharing

例如多个请求具有相同 Prefix 时，可以共享前缀 Block。

这为 Prefix Caching 提供了基础。

---

## 20. Prefix Caching {#20}

现实中很多请求共享相同前缀。

例如：

```text
System Prompt:
你是一个专业的软件工程助手……
```

每个用户请求都包含同一个 System Prompt。

如果每次都重新 Prefill：

```text
System Prompt
↓
重复执行 Transformer
```

非常浪费。

Prefix Cache 可以缓存：

```text
System Prompt
→ 对应的 KV Cache
```

新的请求如果前缀相同：

```text
System Prompt + User Query
```

只需要复用已有 KV，再计算 User Query 部分。

因此可以显著降低：

- TTFT；
- Prefill FLOPs；
- GPU 使用量。

---

## 21. Radix Tree / Prefix Tree {#21}

SGLang 等系统会使用类似 Radix Tree 的数据结构管理共享前缀。

例如：

```text
                 System Prompt
                 /           \
          User Prefix A     User Prefix B
             /   \               |
           Q1    Q2              Q3
```

每个节点可以对应一段 KV Cache。

不同请求可以复用最长公共前缀。

这对：

- 多轮对话；
- Agent；
- Few-shot Prompt；
- RAG 固定模板；
- 多 sampling

尤其有效。

---

## 22. Chunked Prefill {#22}

Prefill 很容易一次占用大量计算资源。

例如一个请求有：

```text
100,000 tokens
```

如果一次性 Prefill，它可能长时间占用 GPU。

这会导致其他 Decode request：

```text
无法及时生成下一个 token
```

从而显著恶化 TPOT。

所以现代推理引擎经常使用：

### Chunked Prefill

把长 Prompt 分块：

```text
100k tokens

↓ split

Chunk 1: 4096
Chunk 2: 4096
Chunk 3: 4096
...
```

Scheduler 可以交替安排：

```text
Decode Batch
↓
Prefill Chunk
↓
Decode Batch
↓
Prefill Chunk
```

这样可以改善公平性和在线延迟。

---

## 23. FlashAttention {#23}

标准 Attention：

$$
S=QK^T
$$

$$
P=\operatorname{softmax}(S)
$$

$$
O=PV
$$

朴素实现会把中间矩阵：

$$
S\in\mathbb{R}^{L\times L}
$$

和：

$$
P\in\mathbb{R}^{L\times L}
$$

写入 HBM。

问题是 Attention 的瓶颈很多时候并不是算术，而是：

```text
HBM ↔ SRAM
```

之间的数据搬运。

FlashAttention 的核心目标就是：

> 通过 tiling 和 online softmax，减少 HBM 读写。

---

## 24. FlashAttention 的基本思想 {#24}

GPU 存储层次：

```text
Registers
   ↓
Shared Memory / SRAM
   ↓
HBM
```

越靠上：

- 更快；
- 容量更小。

FlashAttention 将 Q/K/V 分块：

```text
Q tile
K tile
V tile
```

尽量在 SRAM 中完成：

```text
QK^T
softmax
PV
```

避免把完整 Attention Matrix 写回 HBM。

因此，它本质上属于：

### IO-aware Attention Algorithm

优化的不是 Attention 数学公式本身，而是数据移动方式。

---

## 25. Kernel Fusion {#25}

普通 PyTorch 代码可能是：

```python
x = rms_norm(x)
q, k, v = qkv_proj(x)
q = rope(q)
k = rope(k)
attn = attention(q, k, v)
x = out_proj(attn)
x = x + residual
```

如果每一行都启动一个 CUDA Kernel：

```text
kernel launch
↓
write HBM
↓
kernel launch
↓
read HBM
↓
write HBM
...
```

会产生大量：

- Kernel Launch Overhead；
- HBM 流量；
- 中间 Tensor。

所以推理引擎通常进行：

```text
Fused RMSNorm
Fused RoPE
Fused QKV
Fused SwiGLU
Fused Add + Norm
```

减少 kernel 数量和中间数据搬运。

---

## 26. CUDA Graph {#26}

Decode 的计算图实际上高度重复：

```text
Decode Step 1
Decode Step 2
Decode Step 3
...
```

大量 CUDA Kernel 的启动顺序基本一致。

CPU 如果每次都：

```text
Launch Kernel A
Launch Kernel B
Launch Kernel C
...
```

会产生 CPU Launch Overhead。

CUDA Graph 可以预先捕获：

```text
Kernel A
↓
Kernel B
↓
Kernel C
↓
...
```

然后整个 Graph 一次启动。

因此对低延迟 Decode 很有价值。

---

## 27. Quantization 为什么对推理特别重要 {#27}

假设模型权重是 FP16。

一个模型可能需要：

```text
100 GB
```

如果量化到 FP8：

```text
约 50 GB
```

INT4：

```text
约 25 GB
```

量化有两个关键收益：

### 1. 减少显存占用

允许：

- 更大模型部署；
- 更多 KV Cache；
- 更高并发。

### 2. 减少 HBM Traffic

Decode 是 memory-bound。

模型权重从：

```text
FP16
→ FP8
```

每次读取的数据量大约减半。

因此即使计算量本身没有变化，也可能明显提高 token/s。

---

## 28. 常见量化方式 {#28}

大模型 Serving 中常见：

```text
FP16 / BF16
FP8
INT8
INT4
FP4
```

以及多种 Weight-only Quantization：

```text
W8A16
W4A16
```

含义：

```text
Weights = INT8 / INT4
Activations = FP16
```

也有：

```text
W8A8
```

表示权重和 Activation 都采用低精度。

量化方法需要在以下方面权衡：

- 精度损失；
- Kernel 支持；
- Calibration；
- Throughput；
- Latency；
- 显存占用。

---

## 29. Speculative Decoding {#29}

标准 Decode 必须：

```text
大模型 Forward
↓
生成 token 1
↓
大模型 Forward
↓
生成 token 2
↓
大模型 Forward
↓
生成 token 3
```

由于 token 之间存在自回归依赖，很难直接并行。

Speculative Decoding 的思路是：

> 使用更便宜的模型或预测头，一次先猜多个 token，然后让大模型并行验证。

---

## 30. Speculative Decoding 的流程 {#30}

Draft Model 先预测：

```text
A B C D
```

Target Model 一次验证：

```text
A ✓
B ✓
C ✓
D ✗
```

于是一次 Target Forward 可以接受：

```text
A B C
```

而不只是一个 token。

如果平均每次接受：

$$
k>1
$$

个 token，则有效 Decode 次数下降。

这可以降低：

- TPOT；
- 大模型 Forward 次数。

---

## 31. Draft Model {#31}

最常见做法是使用一个较小模型：

```text
Small Draft Model
       ↓
predict token 1
predict token 2
predict token 3
predict token 4
       ↓
Large Target Model
       ↓
verify
```

Draft Model 的要求是：

- 足够快；
- 与 Target Model 输出分布尽可能接近。

否则 Acceptance Rate 太低，就得不偿失。

---

## 32. Multi-Token Prediction / MTP {#32}

另一种方式是不单独部署 Draft Model。

可以让大模型附加多个预测头：

```text
Hidden State
   ├── Head 1 → t+1
   ├── Head 2 → t+2
   ├── Head 3 → t+3
   └── Head 4 → t+4
```

一次预测未来多个 token，再验证。

一些现代模型和推理系统已经大量使用类似机制提升 Decode 效率。

---

## 33. Sampling {#33}

模型最后输出 logits：

$$
z\in\mathbb{R}^{|V|}
$$

通过 softmax：

$$
P_i=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

然后决定下一 token。

---

### 33.1 Greedy

直接选择：

$$
x_{t+1}
=
\arg\max_i P_i
$$

优点：

- 确定；
- 简单；
- 快。

缺点：

- 输出容易缺乏多样性。

---

### 33.2 Temperature

使用：

$$
P_i
=
\operatorname{softmax}
\left(
\frac{z_i}{T}
\right)
$$

当：

$$
T<1
$$

分布更尖锐。

当：

$$
T>1
$$

分布更平缓。

---

### 33.3 Top-k

只保留概率最高的 $k$ 个 token。

例如：

```text
Vocabulary = 100000
Top-k = 50
```

只在最高概率的 50 个 token 中采样。

---

### 33.4 Top-p

选择最小 token 集合：

$$
S
$$

使得：

$$
\sum_{i\in S}P_i\ge p
$$

例如：

```text
top_p = 0.9
```

即只在累计概率达到 90% 的 token 集合中采样。

---

## 34. Multi-GPU 推理 {#34}

大型模型常常无法放入单张 GPU。

例如：

```text
Model weights = 300 GB
GPU HBM = 80 GB
```

需要多 GPU 并行。

主要策略包括：

```text
Tensor Parallel
Pipeline Parallel
Data Parallel
Expert Parallel
Context Parallel
```

---

## 35. Tensor Parallel {#35}

Tensor Parallel，简称 TP。

将单个矩阵乘法拆到多张 GPU。

例如：

$$
Y=XW
$$

将：

$$
W=[W_1,W_2,W_3,W_4]
$$

分别放到 4 张 GPU：

```text
GPU0 → XW1
GPU1 → XW2
GPU2 → XW3
GPU3 → XW4
```

然后通过通信聚合结果。

优点：

- 单个 Layer 被多 GPU 分担；
- 可以部署超出单卡显存的模型；
- 单 request 可以使用多卡算力。

缺点：

- 几乎每层都有跨 GPU 通信；
- 对 NVLink / NVSwitch / RDMA 非常敏感。

---

## 36. Pipeline Parallel {#36}

Pipeline Parallel，简称 PP。

将模型不同层放在不同 GPU：

```text
GPU0:
Layer 0-19

GPU1:
Layer 20-39

GPU2:
Layer 40-59

GPU3:
Layer 60-79
```

数据依次流过不同 GPU。

优点：

- 通信模式相对简单；
- 适合跨节点。

缺点：

- 容易产生 Pipeline Bubble；
- 单请求 latency 较高。

---

## 37. Data Parallel {#37}

每组 GPU 拥有完整模型：

```text
Replica 1 → Request Group A
Replica 2 → Request Group B
Replica 3 → Request Group C
```

用于横向扩展 Serving 吞吐。

实际系统通常是：

```text
TP × DP
```

例如：

```text
8 GPUs / replica
×
16 replicas
=
128 GPUs
```

---

## 38. MoE 推理 {#38}

Mixture-of-Experts 模型中，一个 token 不会经过所有 FFN Expert。

例如：

```text
Token
 ↓
Router
 ↓
Top-2 Expert
 ↓
Expert 7
Expert 31
```

如果总共有：

```text
256 experts
```

一个 token 可能只激活 8 个。

这样可以：

- 大幅增加模型参数量；
- 但不同比例增加每 token FLOPs。

---

## 39. Expert Parallel {#39}

MoE Serving 常使用 Expert Parallel。

例如：

```text
GPU0: Expert 0-31
GPU1: Expert 32-63
GPU2: Expert 64-95
GPU3: Expert 96-127
```

Router 决定 token 发往哪里。

因此会出现：

```text
GPU0 token
↓
Router
↓
Expert 位于 GPU3
↓
跨 GPU 发送
```

大量通信通常通过：

### All-to-All

完成。

所以 MoE 推理的瓶颈往往包括：

- Expert Load Balance；
- All-to-All；
- Network Bandwidth；
- Expert Hotspot；
- Token Dispatch；
- Token Combine。

---

## 40. Prefill-Decode Disaggregation {#40}

Prefill 和 Decode 的硬件特征差异明显：

```text
Prefill
→ compute-heavy

Decode
→ bandwidth / memory-heavy
```

因此大型 Serving 系统可能把它们部署在不同 GPU Pool。

例如：

```text
             Request
                ↓
        ┌──────────────┐
        │ Prefill GPU  │
        └──────────────┘
                ↓
           KV Cache
                ↓
        KV Transfer
                ↓
        ┌──────────────┐
        │ Decode GPU   │
        └──────────────┘
                ↓
             Tokens
```

称为：

### Prefill-Decode Disaggregation

或者：

```text
PD Disaggregation
```

---

## 41. 为什么要做 PD 解耦 {#41}

如果 Prefill 和 Decode 混在一起：

```text
GPU:
Prefill
Decode
Prefill
Decode
```

可能出现资源干扰。

长 Prompt 的 Prefill 会抢占大量算力，导致：

```text
Decode token 延迟突然升高
```

将二者分离后可以分别优化：

```text
Prefill Pool:
追求高 FLOPS

Decode Pool:
追求高 HBM Bandwidth
```

并独立做扩容。

---

## 42. PD 解耦最大的难点 {#42}

Prefill 生成的 KV Cache 必须传给 Decode 节点。

这意味着：

```text
几十 MB
甚至几百 MB KV
```

需要跨 GPU / 跨节点传输。

因此系统必须优化：

```text
NVLink
NVSwitch
PCIe
InfiniBand
RoCE
RDMA
```

否则 KV Transfer 本身就可能成为瓶颈。

---

## 43. 长上下文为什么难 {#43}

假设 Context Length：

$$
L=128K
$$

会带来多个问题。

### 1. Prefill Attention 成本

传统 Attention：

$$
O(L^2)
$$

长上下文迅速变贵。

### 2. KV Cache

KV 占用：

$$
O(L)
$$

随着 context 线性增长。

### 3. Decode Attention

每生成一个 token，都需要读取越来越长的 KV：

$$
q_tK_{1:t}^{T}
$$

因此 Decode 单 token 成本也会随上下文增加。

---

## 44. 长上下文优化 {#44}

常见方向包括：

```text
Paged KV Cache
KV Quantization
Sliding Window Attention
Sparse Attention
Grouped Query Attention
MLA
Context Parallelism
KV Offloading
Prefix Cache
```

目标都是降低：

- KV 显存；
- KV 带宽；
- Attention 成本；
- 长序列通信成本。

---

## 45. MLA 与 KV Cache {#45}

Multi-head Latent Attention（MLA）可以通过低维 latent 表示压缩 KV 信息。

传统 KV：

```text
K head 0
K head 1
...
V head 0
V head 1
...
```

MLA 思路则更接近：

```text
Hidden State
    ↓
Compressed Latent
    ↓
按需要恢复 Attention 所需表示
```

它可以显著减少 KV Cache。

这也是近年来高效大模型结构设计中特别关注的方向。

---

## 46. KV Cache Quantization {#46}

除了量化模型权重，也可以量化 KV Cache。

例如：

```text
KV BF16
↓
KV FP8
```

或者更低精度。

好处：

```text
KV Memory ↓
KV Bandwidth ↓
Max Concurrent Requests ↑
```

代价是：

- 精度误差；
- Quant / Dequant 开销；
- Kernel 复杂度。

---

## 47. Offloading {#47}

如果 GPU 显存不足，可以将：

```text
KV Cache
或者部分模型权重
```

放到：

```text
CPU RAM
```

甚至 SSD。

例如：

```text
GPU HBM
  ↕
CPU DRAM
```

但是 PCIe 带宽远小于 HBM。

因此 Offloading 通常属于：

```text
容量优化
```

而不是纯性能优化。

---

## 48. 一次完整请求的生命周期 {#48}

现代 LLM Serving 的完整流程可以表示为：

```text
HTTP / RPC Request
       ↓
Tokenizer
       ↓
Request Queue
       ↓
Scheduler
       ↓
Prefix Cache Lookup
       ↓
Prefill
       ↓
Create / Extend KV Cache
       ↓
Decode Scheduler
       ↓
Continuous Batching
       ↓
Attention + MLP
       ↓
Logits
       ↓
Sampling
       ↓
New Token
       ↓
Update KV Cache
       ↓
Stream Token to User
       ↓
是否结束？
   ↓         ↓
  否         是
   ↓         ↓
继续 Decode  Release KV Blocks
```

---

## 49. 一个更完整的 Serving 架构 {#49}

```text
             ┌─────────────────┐
             │ Client / API    │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │ Tokenizer       │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │ Request Queue   │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │ Scheduler       │
             └──────┬────┬─────┘
                    │    │
             Prefill│    │Decode
                    │    │
         ┌──────────▼┐  ┌▼────────────┐
         │ Prefill   │  │ Decode      │
         │ Engine    │  │ Engine      │
         └─────┬─────┘  └─────┬───────┘
               │              │
               └──────┬───────┘
                      ↓
             ┌─────────────────┐
             │ KV Cache Mgr    │
             │ Paged Blocks    │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │ CUDA Kernels    │
             │ FlashAttention  │
             │ GEMM / MoE      │
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │ GPU / GPU Mesh  │
             └─────────────────┘
```

---

## 50. 为什么现代推理引擎不是简单的 `model.generate()` {#50}

最简单的 Hugging Face 推理：

```python
model.generate(...)
```

主要关注：

```text
功能正确
```

而高性能 Serving Engine 必须同时处理：

```text
Thousands of requests
Dynamic batching
KV memory management
Prefix sharing
Quantization
Multi-GPU
Distributed communication
Fault tolerance
Streaming
Scheduling
Latency SLO
Throughput
```

所以本质上是：

> GPU Runtime + Distributed System + Memory Manager + Scheduler + Transformer Engine。

---

## 51. vLLM 可以如何理解 {#51}

可以把 vLLM 的核心思想概括为：

```text
Transformer
+
Paged KV Cache
+
Continuous Batching
+
Efficient Attention Kernels
+
Scheduler
+
Distributed Inference
```

其经典贡献是：

### PagedAttention

它极大改善了 KV Cache 的显存利用率和请求调度能力。

---

## 52. SGLang 可以如何理解 {#52}

SGLang 更强调：

```text
LLM program execution
+
Prefix reuse
+
Radix Cache
+
Scheduling
+
High-performance kernels
+
Distributed Serving
```

尤其适合：

- Agent；
- structured generation；
- multi-turn programs；
- repeated prefixes；
- complex LLM workflows。

---

## 53. TensorRT-LLM 可以如何理解 {#53}

TensorRT-LLM 更强调 NVIDIA GPU 上高度优化的部署：

```text
TensorRT
+
CUDA Kernels
+
Quantization
+
Inflight Batching
+
Multi-GPU
+
NVIDIA Hardware Optimization
```

通常非常重视：

- H100 / H200 / B200 等硬件特性；
- FP8 / FP4；
- Tensor Core；
- NCCL；
- NVLink；
- CUDA Graph。

---

## 54. 推理性能的三个核心指标 {#54}

### 54.1 Latency

单个请求需要多久。

包括：

```text
Queueing
Prefill
Decode
Communication
Sampling
```

---

### 54.2 Throughput

单位时间生成多少 token。

常见：

```text
tokens / second
```

或者：

```text
requests / second
```

---

### 54.3 Cost

例如：

```text
$/1M tokens
GPU-hours
Energy/token
```

商业 Serving 通常最终优化的是：

$$
\text{Cost per Token}
$$

而不仅仅是极限 tokens/s。

---

## 55. Latency 与 Throughput 的冲突 {#55}

为了更高 throughput，可以增加 batch size：

```text
Batch ↑
GPU Utilization ↑
Throughput ↑
```

但是：

```text
等待 Batch 的时间 ↑
单请求延迟 ↑
```

所以 Serving 系统需要在：

```text
Latency
↔
Throughput
```

之间权衡。

这也是 Scheduler 的核心职责之一。

---

## 56. Roofline 视角理解 LLM 推理 {#56}

硬件性能可以从：

$$
\text{Arithmetic Intensity}
=
\frac{
\text{FLOPs}
}{
\text{Bytes moved}
}
$$

理解。

如果 Arithmetic Intensity 高：

```text
Compute Bound
```

如果低：

```text
Memory Bound
```

通常：

```text
Prefill:
Arithmetic Intensity 高
→ Compute Bound

Decode:
Arithmetic Intensity 低
→ Memory Bound
```

因此优化方法完全不同。

---

## 57. Prefill 应该优化什么 {#57}

Prefill 重点优化：

```text
GEMM
Tensor Core Utilization
FlashAttention
Large Batch
FP8
Sequence Parallelism
Context Parallelism
Chunked Prefill
```

核心目标：

> 提高计算吞吐。

---

## 58. Decode 应该优化什么 {#58}

Decode 重点优化：

```text
HBM Bandwidth
Batch Size
KV Cache
PagedAttention
Weight Quantization
KV Quantization
Kernel Fusion
CUDA Graph
Speculative Decoding
```

核心目标：

> 每次从显存读取模型权重时，尽可能服务更多 token。

---

## 59. 一个非常重要的直觉 {#59}

假设模型权重：

```text
100 GB
```

GPU 显存带宽：

```text
3 TB/s
```

极粗略地看，如果生成一个 token 基本需要扫描一次权重，那么理论上单 batch=1 的极限可能受到：

$$
\frac{3000\ GB/s}{100\ GB}
\approx
30\ token/s
$$

这样的带宽上限约束。

当然真实模型还受到：

- Cache；
- Kernel；
- Tensor Parallel；
- Quantization；
- Layer 结构；
- KV；
- Communication

等影响。

但这个估算很好地解释了：

> 为什么 Decode 的性能与 HBM Bandwidth 高度相关。

---

## 60. 为什么 Batch 能提高 Decode 性能 {#60}

假设同样读取一次：

```text
100 GB weights
```

Batch=1：

```text
服务 1 个 token
```

Batch=64：

```text
服务 64 个 token
```

于是单位 token 分摊的权重读取成本显著降低。

所以：

```text
Batch Size ↑
↓
Arithmetic Intensity ↑
↓
GPU Efficiency ↑
```

这就是 Continuous Batching 的底层价值。

---

## 61. 大模型推理优化的本质 {#61}

从更高层总结，现代 LLM Inference 在解决四个问题。

### 1. 计算

```text
GEMM
Attention
MoE
```

优化：

```text
Tensor Core
FlashAttention
FP8
Kernel Fusion
```

### 2. 内存

```text
Weights
KV Cache
Activations
```

优化：

```text
Quantization
Paged KV
KV Compression
Prefix Cache
```

### 3. 调度

```text
Many Requests
Different Length
Different Arrival Time
```

优化：

```text
Continuous Batching
Chunked Prefill
Priority Scheduling
Admission Control
```

### 4. 通信

```text
Multi-GPU
Multi-Node
MoE
PD Disaggregation
```

优化：

```text
NVLink
NVSwitch
NCCL
RDMA
AllReduce
All-to-All
```

---

## 62. 从研究角度建议的学习路线 {#62}

如果希望系统理解 LLM Inference，建议按如下顺序学习。

### 第一阶段：Transformer 推理基础

掌握：

```text
Autoregressive Generation
Causal Attention
KV Cache
MHA / MQA / GQA
RoPE
RMSNorm
SwiGLU
```

理解：

$$
P(x_{t+1}\mid x_{\le t})
$$

以及 Decode 为什么只能逐 token 进行。

---

### 第二阶段：GPU 基础

掌握：

```text
GPU SM
Tensor Core
Warp
Registers
Shared Memory
HBM
Memory Bandwidth
CUDA Kernel
GEMM
```

尤其理解：

```text
Compute-bound
vs
Memory-bound
```

---

### 第三阶段：Attention Kernel

重点学习：

```text
FlashAttention
FlashAttention-2
FlashAttention-3
FlashInfer
PagedAttention
```

理解 IO-aware algorithm。

---

### 第四阶段：Serving

重点：

```text
Continuous Batching
Paged KV Cache
Prefix Cache
Chunked Prefill
Scheduler
TTFT
TPOT
Throughput
```

推荐阅读高性能 Serving Engine 的代码。

---

### 第五阶段：Decode 加速

重点：

```text
Quantization
CUDA Graph
Speculative Decoding
Multi-Token Prediction
KV Quantization
```

---

### 第六阶段：分布式推理

学习：

```text
Tensor Parallel
Pipeline Parallel
Data Parallel
Expert Parallel
Context Parallel
```

以及通信 primitive：

```text
AllReduce
AllGather
ReduceScatter
All-to-All
```

---

### 第七阶段：大规模 Serving

进一步研究：

```text
Prefill-Decode Disaggregation
KV Transfer
Disaggregated Serving
Load Balancing
Admission Control
Autoscaling
SLO-aware Scheduling
Fault Tolerance
```

这时问题已经不只是“深度学习”，而是：

```text
Distributed Systems
+
GPU Systems
+
Networking
+
Compiler
+
Machine Learning
```

---

## 63. 推荐建立的核心知识链 {#63}

最重要的一条链是：

```text
Naive Transformer Inference
        ↓
KV Cache
        ↓
GQA / MQA
        ↓
FlashAttention
        ↓
Continuous Batching
        ↓
PagedAttention
        ↓
Prefix Caching
        ↓
Chunked Prefill
        ↓
Quantization
        ↓
Speculative Decoding
        ↓
Tensor Parallel
        ↓
Expert Parallel
        ↓
Prefill-Decode Disaggregation
```

如果能把这条链完整理解，就已经具备比较完整的现代 LLM Inference Systems 知识框架。

---

## 64. 最终总结 {#64}

现代大模型推理并不只是：

```python
model(input)
```

真正的高性能推理系统可以抽象为：

```text
                 LLM Serving
                      │
      ┌───────────────┼───────────────┐
      │               │               │
   Compute          Memory        Scheduling
      │               │               │
 FlashAttention    KV Cache      Continuous Batch
 GEMM              Paged KV      Chunked Prefill
 FP8               Quantization  Prefix Cache
      │               │               │
      └───────────────┼───────────────┘
                      │
               Distributed
                      │
      TP / PP / DP / EP / CP
                      │
             NVLink / RDMA / NCCL
```

其中最关键的认知是：

### Prefill

```text
大量 token 并行
→ 大矩阵计算
→ 更偏 Compute Bound
```

### Decode

```text
每轮少量新 token
→ 频繁读取权重和 KV
→ 更偏 Memory Bandwidth Bound
```

因此现代 LLM 推理系统的核心目标不是简单增加 FLOPS，而是：

> 用更聪明的内存管理、批处理、调度、Kernel、量化和分布式通信，让昂贵的 GPU 每一次数据访问都服务尽可能多的 token。

从这个角度看，vLLM、SGLang、TensorRT-LLM 等推理框架，本质上都可以理解为：

> **围绕 Transformer 建立的一套 GPU 内存管理器、动态调度器、高性能 Kernel Runtime 和分布式执行系统。**

---

---

## 参考文献 {#ref}

[1] NVIDIA, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, 2023. arXiv:2307.08691.

[2] W. Kwon, Z. Li, S. Zhuang, et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP, 2023.

[3] G. Xiao, J. Lin, M. Seznec, et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS, 2022.

[4] C. Chen, S. Borgeaud, G. Irving, et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, arXiv preprint, 2023.

[5] A. M. Dai, et al., *Continuous Batching / In-flight Batching*, 2023-2024.

[6] L. Zheng, et al., *SGLang: Efficient Execution of Structured Language Model Programs*, NeurIPS, 2024.

[7] NVIDIA, *TensorRT-LLM: High-Performance LLM Inference with TensorRT*, 2024.

[8] Z. Li, et al., *DeepSeek-V2: MLA (Multi-head Latent Attention) for Efficient Inference*, 2024. arXiv:2405.04434.

[9] Y. Leviathan, M. Kalman, Y. Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML, 2023.

[10] B. Patel, et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, ISCA, 2024.


## 关键词速查 {#kw}

```text
Autoregressive Generation
Prefill
Decode
KV Cache
MHA
MQA
GQA
PagedAttention
Paged KV Cache
Continuous Batching
Inflight Batching
Chunked Prefill
Prefix Caching
Radix Cache
FlashAttention
FlashInfer
CUDA Graph
Kernel Fusion
FP8
INT8
INT4
KV Quantization
Speculative Decoding
Draft Model
Multi-Token Prediction
Tensor Parallel
Pipeline Parallel
Data Parallel
Expert Parallel
Context Parallel
MoE
AllReduce
All-to-All
PD Disaggregation
TTFT
TPOT
Throughput
HBM Bandwidth
Arithmetic Intensity
```
