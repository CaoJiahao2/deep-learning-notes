# LLM 微调技术详解

> 从全量微调到 PEFT，大模型高效微调技术全景。

---

## 目录

1. [微调范式概述](#1-微调范式概述)
2. [全量微调 (Full Fine-Tuning)](#2-全量微调-full-fine-tuning)
3. [LoRA 及其变体](#3-lora-及其变体)
4. [Adapter 系列](#4-adapter-系列)
5. [Prefix/Prompt Tuning](#5-prefixprompt-tuning)
6. [指令微调 (Instruction Tuning)](#6-指令微调-instruction-tuning)
7. [对齐技术 (RLHF/DPO)](#7-对齐技术-rlhfdpo)
8. [微调策略选择](#8-微调策略选择)
9. [参考文献](#9-参考文献)

---

## 1. 微调范式概述

### 1.1 微调的必要性

预训练大模型（如 LLaMA, Qwen）具有通用能力，但需要微调来适配特定领域或任务。

### 1.2 微调方法分类

| 方法 | 可训练参数量 | 显存需求 | 性能 |
|------|-------------|---------|------|
| 全量微调 | 100% | 极高 | 最佳 |
| LoRA | 0.1%-1% | 低 | 接近全量 |
| QLoRA | 0.1%-1% | 极低 | 接近 LoRA |
| Adapter | 1%-5% | 低 | 略低于 LoRA |
| Prefix Tuning | < 0.1% | 低 | 任务有限 |
| Prompt Tuning | < 0.01% | 极低 | 仅简单任务 |

---

## 2. 全量微调 (Full Fine-Tuning)

### 2.1 定义

在预训练权重基础上，对所有参数进行梯度更新。

### 2.2 资源需求

以 7B 模型为例：
- 参数：~14 GB (FP16)
- 梯度：~14 GB
- 优化器状态 (Adam)：~42 GB
- 激活值：~20 GB (batch=1, seq_len=2048)
- **总计：~90 GB** → 需要 2×A100 80GB

### 2.3 训练技巧

- 低学习率：通常 $10^{-5} \sim 5 \times 10^{-5}$（比预训练低 1-2 个数量级）
- Warmup 比例小：总步数的 3-10%
- 小 batch size + 梯度累积
- 使用 BF16 替代 FP16（动态范围更大）

---

## 3. LoRA 及其变体

### 3.1 LoRA (Low-Rank Adaptation)

**核心思想**：权重更新矩阵 $\Delta W$ 是低秩的，可以用两个小矩阵的乘积表示。

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$

$$h = W_0x + \Delta W x = W_0x + BAx$$

其中 $r \ll \min(d, k)$，通常 $r = 8 \sim 64$。

**可训练参数**：$r \times (d + k)$，相比原始 $d \times k$ 大幅减少。

**缩放因子**：通常将 $\Delta W x$ 乘以 $\alpha / r$ 控制更新幅度。

### 3.2 LoRA 配置参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| $r$ | 8, 16, 32, 64 | rank 越大能力越强，参数越多 |
| $\alpha$ | 16, 32 | 通常 $\alpha = 2r$ |
| target_modules | q_proj, v_proj | 或 qkv_proj + o_proj |
| dropout | 0.05 - 0.1 | LoRA dropout |

### 3.3 应用位置

```python
# 常见配置
target_modules = ["q_proj", "v_proj"]          # 最小配置
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 推荐
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]  # 全配置
```

### 3.4 QLoRA

在 LoRA 基础上引入量化技术：

- **4-bit NormalFloat (NF4)**：针对正态分布权重优化的 4-bit 量化格式
- **双重量化（Double Quantization）**：对量化常数进一步量化，节省 0.37 bit/参数
- **分页优化器（Paged Optimizer）**：利用 CPU 内存卸载优化器状态

**效果**：在 48GB GPU 上即可微调 65B 模型（全量微调需要 ~780GB）。

### 3.5 LoRA 变体

| 变体 | 改进 |
|------|------|
| LoRA+ | 为矩阵 A 和 B 使用不同的学习率 |
| DoRA | 将权重分解为幅度和方向，仅对方向做 LoRA |
| AdaLoRA | 自适应分配 rank，重要层给更高 rank |
| rsLoRA | 修正 scaling factor，使大 rank 训练稳定 |
| PiSSA | 用 SVD 分解初始化 LoRA 矩阵 |

---

## 4. Adapter 系列

### 4.1 标准 Adapter

在 Transformer 层中插入小型可训练模块：

```text
FFN(x) + Adapter(x) = FFN(x) + W_up(ReLU(W_down(x)))
```

其中 $W_{down} \in \mathbb{R}^{d \times m}$，$W_{up} \in \mathbb{R}^{m \times d}$，$m \ll d$。

### 4.2 Adapter 变体

| 变体 | 位置 | 特点 |
|------|------|------|
| Houlsby Adapter | FFN 后 + LayerNorm 后 | 两个 Adapter/层 |
| Pfeiffer Adapter | 仅 FFN 后 | 一个 Adapter/层 |
| Parallel Adapter | 与 FFN 并行 | 不影响推理延迟 |
| Compacter | 参数高效 | 低秩 + 参数共享 |
| LLaMA-Adapter | 仅 Attention 层 | 极轻量 |

---

## 5. Prefix/Prompt Tuning

### 5.1 Prefix Tuning

在每个 Transformer 层前添加可训练的 prefix tokens：

$$h = \text{Transformer}([P_k; x])$$

其中 $P_k$ 是可训练的 prefix 嵌入，$k$ 通常为 10-100。

### 5.2 Prompt Tuning

仅在输入层添加可训练的 soft prompt tokens：

$$h = \text{Transformer}([P; x])$$

极其轻量，但效果有限，主要用于简单任务。

### 5.3 P-Tuning v2

在每层都添加可训练的 soft prompts（类似 Prefix Tuning 的深度版本）。

---

## 6. 指令微调 (Instruction Tuning)

### 6.1 定义

使用 (指令, 回答) 对微调模型，使其能够遵循人类指令。

### 6.2 数据格式

```json
{
    "instruction": "将以下句子翻译成英文",
    "input": "今天天气真好",
    "output": "The weather is really nice today"
}
```

### 6.3 常用数据集

| 数据集 | 规模 | 说明 |
|--------|------|------|
| Alpaca | 52K | GPT-3.5 生成的指令数据 |
| Dolly | 15K | 人工标注的指令数据 |
| OpenOrca | ~1M | 大规模合成指令数据 |
| ShareGPT | ~90K | 与 ChatGPT 的真实对话 |

### 6.4 训练模板

```python
# LLaMA 格式
prompt = f"<s>[INST] {instruction} [/INST] {response}</s>"

# ChatML 格式
prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
```

---

## 7. 对齐技术 (RLHF/DPO)

### 7.1 RLHF (Reinforcement Learning from Human Feedback)

**三步流程**：

1. **监督微调 (SFT)**：用高质量数据进行指令微调
2. **奖励模型训练 (RM)**：人工标注偏好数据，训练奖励模型
3. **PPO 强化学习**：用 PPO 算法优化策略，最大化奖励

**挑战**：
- 需要训练和维护独立的奖励模型
- PPO 训练不稳定，四个模型同时加载（Policy, Reference, Reward, Value）
- 超参数敏感

### 7.2 DPO (Direct Preference Optimization)

直接利用偏好数据优化策略，无需显式训练奖励模型：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log\sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

其中 $y_w$ 是偏好回答，$y_l$ 是非偏好回答。

**DPO 优势**：
- 无需训练奖励模型
- 训练更稳定
- 计算成本更低

### 7.3 其他对齐方法

| 方法 | 特点 |
|------|------|
| KTO | 无需偏好对，只需二元反馈 |
| ORPO | 将 SFT 和对齐合并为一步 |
| SimPO | 以参考策略为基准，无需参考模型 |
| RLAIF | 用 AI 反馈替代人类反馈 |

---

## 8. 微调策略选择

### 8.1 决策流程

```
资源充足 (> 4×A100)? 
  ├─ 是 → 全量微调（最佳性能）
  └─ 否 → 单卡 (48GB+)?
           ├─ 是 → LoRA (r=16~64) + 全目标模块
           └─ 否 → QLoRA (r=8~16) + 基础目标模块
```

### 8.2 按场景推荐

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 领域适配 | LoRA (r=64) | 保留通用能力，高效注入领域知识 |
| 指令遵循 | 全量微调 | 影响所有层，效果最佳 |
| 特定任务 | LoRA (r=16) | 轻量灵活，可合并部署 |
| 消费级显卡 | QLoRA | 7B 模型仅需 ~8GB 显存 |
| 多任务切换 | LoRA 模块切换 | 不同任务加载不同 LoRA 权重 |

---

## 9. 参考文献

[1] Hu, E. J., et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022.

[2] Dettmers, T., et al. QLoRA: Efficient Finetuning of Quantized Language Models. NeurIPS, 2023.

[3] Houlsby, N., et al. Parameter-Efficient Transfer Learning for NLP. ICML, 2019.

[4] Li, X. L. & Liang, P. Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL, 2021.

[5] Lester, B., et al. The Power of Scale for Parameter-Efficient Prompt Tuning. EMNLP, 2021.

[6] Ouyang, L., et al. Training Language Models to Follow Instructions with Human Feedback. NeurIPS, 2022.

[7] Rafailov, R., et al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS, 2023.

[8] Liu, H., et al. Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning. NeurIPS, 2022. (IA3)

[9] Wang, Y., et al. DoRA: Weight-Decomposed Low-Rank Adaptation. ICML, 2024.
