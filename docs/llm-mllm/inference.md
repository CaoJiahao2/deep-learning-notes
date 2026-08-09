# LLM 推理优化详解

> KV Cache、Flash Attention、量化等大模型推理加速技术全景。

---

## 目录

1. [推理优化概述](#1)
2. [KV Cache](#2-kv-cache)
3. [Flash Attention](#3-flash-attention)
4. [模型量化](#4)
5. [推理框架](#5)
6. [投机解码 (Speculative Decoding)](#6-speculative-decoding)
7. [其他优化技术](#7)
8. [参考文献](#8)

---

## 1. 推理优化概述

### 1.1 推理的两个阶段

| 阶段 | 特点 | 瓶颈 |
|------|------|------|
| Prefill（预填充） | 一次性处理所有输入 token | 计算密集型 |
| Decode（解码） | 逐 token 自回归生成 | 内存带宽密集型 |

### 1.2 核心优化方向

- **内存优化**：减少 KV Cache 显存占用
- **计算优化**：加速 Attention 计算
- **带宽优化**：减少数据传输
- **算法优化**：减少解码步数

---

## 2. KV Cache

### 2.1 为什么需要 KV Cache

自回归生成时，每个新 token 的 Attention 需要所有历史 token 的 Key 和 Value。如不缓存，每次都需要重新计算，复杂度为 $O(n^2)$。

**KV Cache 方案**：将已计算的 Key 和 Value 缓存起来，新 token 只需要计算当前 token 的 Attention。

### 2.2 显存占用

$$\text{KV Cache 大小} = 2 \times \text{层数} \times n_{kv} \times \text{头维度} \times \text{序列长度} \times \text{精度字节数}$$

其中 $n_{kv}$ 是 **KV 头数**：MHA 中 $n_{kv} = n_{head}$，MQA 中 $n_{kv} = 1$，GQA 中 $1 \leq n_{kv} < n_{head}$。

以 LLaMA-7B 为例（32 层，32 头，头维度 128，seq_len=2048，FP16）：
- KV Cache = $2 \times 32 \times 32 \times 128 \times 2048 \times 2 = 1 \text{ GB}$

长序列时 KV Cache 成为瓶颈。seq_len=128K 时可达 ~64 GB。

### 2.3 KV Cache 优化

| 技术 | 原理 | 效果 |
|------|------|------|
| Multi-Query Attention (MQA) | 多个 Q 头共享一组 K, V | KV Cache 减少 $h$ 倍 |
| Grouped-Query Attention (GQA) | Q 头分组共享 K, V | 折中方案 |
| **Multi-head Latent Attention (MLA)** | 对 KV 做低秩压缩（DeepSeek-V2/V3），缓存压缩 latent | KV Cache 大幅下降（可达 90%+），推理成本显著降低 |
| PagedAttention | 分页管理 KV Cache | 减少碎片，提高利用率 |
| KV Cache 量化 | 将 KV Cache 量化为 INT8/INT4 | 显存减半或更多 |
| Sliding Window | 只保留最近窗口的 KV Cache | 固定显存占用 |

---

## 3. Flash Attention

### 3.1 核心问题

标准 Attention 实现中，$QK^T$ 矩阵 ($n \times n$) 需要写入 HBM（高带宽显存）再读回，导致大量 I/O。

### 3.2 Flash Attention 原理

**核心思想**：利用 GPU 内存层次结构，在 SRAM 中完成计算，避免中间结果写入 HBM。

**关键技术**：
1. **Tiling**：将 Q, K, V 切分为小块，在 SRAM 中逐块计算
2. **Online Softmax**：通过维护 running max 和 running sum，在分块中正确计算 softmax
3. **Recomputation**：反向传播时重新计算 Attention 矩阵，而非从 HBM 读取

### 3.3 Flash Attention 版本演进

| 版本 | 改进 | 加速比 |
|------|------|--------|
| Flash Attention 1 | 基础 Tiling + Online Softmax | 2-4× |
| Flash Attention 2 | 更好的并行策略，减少非矩阵乘法 | 2-3× (vs v1) |
| Flash Attention 3 | Hopper 架构优化，异步 FP8 | 1.5-2× (vs v2) |

### 3.4 使用方式

```python
# PyTorch 2.0+ 内置（FlashAttention / Memory-Efficient / Math 后端自动选择）
output = F.scaled_dot_product_attention(Q, K, V)

# 如需显式指定后端（PyTorch 2.1+，sdp_kernel 已废弃）
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(Q, K, V)
```

---

## 4. 模型量化

### 4.1 量化基础

将模型参数从 FP16/BF16 降低到 INT8/INT4 精度，减少显存和带宽需求。

| 精度 | 每参数 bits | 7B 模型大小 | 加速比 |
|------|-----------|------------|--------|
| FP32 | 32 | 28 GB | 基线 |
| FP16/BF16 | 16 | 14 GB | 2× |
| INT8 | 8 | 7 GB | 2-3× |
| INT4 | 4 | 3.5 GB | 3-4× |

### 4.2 量化方法分类

**训练后量化 (PTQ)**：
- 不需要重新训练
- 快速但精度可能损失

**量化感知训练 (QAT)**：
- 训练时模拟量化
- 精度更高但成本大

### 4.3 GPTQ

基于逐列最优脑手术（Optimal Brain Surgeon）的权重量化方法：

1. 对权重矩阵逐列量化
2. 量化误差通过 Hessian 逆矩阵传播到未量化列
3. 每一步对剩余权重进行补偿

**特点**：4-bit 量化精度损失极小，但需要校准数据。

### 4.4 AWQ (Activation-aware Weight Quantization)

**核心发现**：不是所有权重同等重要。权重的重要性与激活值的大小正相关。

**方法**：
1. 观察激活值分布，找到重要通道
2. 对重要通道的权重进行缩放保护
3. 量化到 INT4

**优势**：不需要反向传播，比 GPTQ 更简单高效。

### 4.5 GGUF / GGML

专为 CPU 推理设计的量化格式：

- 支持多种量化类型（Q4_0, Q4_1, Q5_0, Q8_0...）
- 可在 CPU 上运行大模型
- llama.cpp 生态的核心格式

### 4.6 BitsAndBytes

HuggingFace 集成的量化库，支持：
- 4-bit (NF4, FP4)
- 8-bit (LLM.int8())
- 双重量化
- QLoRA 训练

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)
```

---

## 5. 推理框架

### 5.1 主流框架对比

| 框架 | 核心技术 | 吞吐量 | 适用场景 |
|------|---------|--------|---------|
| vLLM | PagedAttention | 极高 | 高并发在线服务 |
| TensorRT-LLM | 深度图优化 | 极高 | NVIDIA GPU 部署 |
| llama.cpp | CPU 优化 | 中等 | 消费级硬件 |
| HuggingFace TGI | HF 生态集成 | 高 | 生产环境 |
| SGLang | RadixAttention | 高 | 结构化生成 |
| MLC-LLM | TVM 编译优化 | 高 | 跨平台部署 |

### 5.2 vLLM

**核心创新 — PagedAttention**：
- 将 KV Cache 按页（page）管理，类似操作系统虚拟内存
- 消除 KV Cache 碎片，提高显存利用率
- 支持动态批处理（Continuous Batching）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
outputs = llm.generate(prompts, SamplingParams(temperature=0.8))
```

### 5.3 Continuous Batching

传统静态批处理：必须等所有请求完成后才能释放资源。

Continuous Batching：每个 decode 步后，完成的请求立即释放资源，新请求可立即加入。

---

## 6. 投机解码 (Speculative Decoding)

### 6.1 核心思想

用小型快速模型（Draft Model）生成多个候选 token，然后让大模型（Target Model）一次性验证：

```text
1. Draft Model 生成 K 个候选 token
2. Target Model 并行验证 K 个候选
3. 接受所有匹配的 token，拒绝第一个不匹配的
4. 从该位置用 Target Model 生成一个 token
```

### 6.2 加速原理

目标模型每次验证 $K$ 个 token 只需一次前向传播。如果 draft model 准确率足够高，平均每步可接受 $\alpha K$ 个 token，加速比为 $\alpha K$。

### 6.3 常见实现

- **Medusa**：在原模型上加多个预测头，无需独立 draft model
- **Lookahead Decoding**：利用 Jacobi 迭代
- **Eagle**：基于特征预测的投机解码

---

### 6.4 MLA 与 PD 分离（2024–2025 前沿）

**MLA（DeepSeek-V2/V3）**：将每层的 K、V 压缩为低秩 latent $\mathbf{c}_t = W^{DKV}\mathbf{h}_t \in \mathbb{R}^{d_c}$（$d_c \ll d_{kv}$），推理时只需缓存 $\mathbf{c}_t$ 与少量解耦的旋转位置项，KV 显存可降低一个数量级，同时保持甚至超越 MHA 的效果。

**PD 分离（Prefill-Decode Disaggregation）**：由于 Prefill 阶段是计算密集型、Decode 阶段是显存/带宽密集型，将两者部署到不同机器：

```text
客户端 → [Prefill 实例（计算型）] → [KV 传输] → [Decode 实例（显存型）] → 输出
```

- Prefill 实例可用高算力 GPU 批量处理长 prompt；
- Decode 实例可长时间驻留 KV Cache，配合 continuous batching 提升吞吐；
- 典型实现：DeepSeek 在线服务的 Prefill/Decode 分离架构、vLLM 的分布式运行时。

---

## 7. 其他优化技术

### 7.1 算子融合

将多个操作合并为一个 kernel 调用，减少 kernel launch 开销：

- 残差连接 + LayerNorm + Dropout → 单个 kernel
- Attention + Softmax + Dropout → 单个 kernel

### 7.2 模型剪枝

移除不重要的权重或结构：

- **非结构化剪枝**：将单个权重置零
- **结构化剪枝**：移除整个通道/头/层
- **SparseGPT**：GPT 系列模型的稀疏化

### 7.3 模型蒸馏

用大模型（Teacher）指导小模型（Student）训练：

- 黑盒蒸馏：仅用 Teacher 的输出标签
- 白盒蒸馏：利用 Teacher 的中间表示和 logits

### 7.4 高效解码策略

| 策略 | 说明 |
|------|------|
| Beam Search | 保留多个候选序列 |
| Top-k Sampling | 只从 top-k 个 token 中采样 |
| Top-p (Nucleus) Sampling | 累积概率达到 p 的 token 集合中采样 |
| Temperature | 调节概率分布的陡峭度 |

---

## 8. 参考文献

[1] Dao, T., et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS, 2022.

[2] Dao, T. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR, 2024.

[3] Frantar, E., et al. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR, 2023.

[4] Lin, J., et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys, 2024.

[5] Dettmers, T., et al. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS, 2022.

[6] Kwon, W., et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP, 2023. (vLLM)

[7] Leviathan, Y., et al. Fast Inference from Transformers via Speculative Decoding. ICML, 2023.

[8] Shazeer, N. Fast Transformer Decoding: One Write-Head Is All You Need. arXiv:1911.02150, 2019. (MQA)

[9] Ainslie, J., et al. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP, 2023.
