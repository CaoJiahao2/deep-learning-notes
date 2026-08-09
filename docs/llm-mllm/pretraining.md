# LLM 预训练详解

> 预训练决定模型的「底座能力」；目标函数、数据质量与配比往往比单纯堆参数更关键。

---

## 目录

1. [预训练概述](#1)
2. [预训练目标](#2)
3. [数据工程](#3)
4. [Scaling Laws](#4-scaling-laws)
5. [训练策略](#5)
6. [训练监控与评测](#6)
7. [参考文献](#7)

---

## 1. 预训练概述

### 1.1 什么是预训练

预训练（Pretraining）是 LLM 训练流程的第一阶段，也是最核心的阶段。在数以万亿计的 token 上，模型通过自监督学习从海量文本中习得语言知识、世界知识和推理能力。预训练的质量直接决定了模型的「底座能力」——后续微调、对齐等技术只能在此基础上做增量调整，几乎无法弥补预训练的缺陷。

### 1.2 现代 LLM 预训练范式

现代生成式 LLM 几乎全部采用 **Decoder-only + CLM（Causal Language Modeling）** 架构：

| 模型 | 预训练目标 | 架构 | 数据规模 |
|------|-----------|------|---------|
| GPT-3 (Brown 2020) | CLM | Decoder-only | ~300B tokens |
| LLaMA (Touvron 2023) | CLM | Decoder-only | 1.0T (LLaMA) / 2.0T (LLaMA 2) |
| LLaMA 3 (2024) | CLM | Decoder-only | 15T+ tokens |
| Qwen 2.5 (2024) | CLM | Decoder-only | 18T tokens |
| DeepSeek-V3 (2024) | CLM | Decoder-only | 14.8T tokens |
| BERT (Devlin 2019) | MLM | Encoder-only | 16GB text |
| T5 (Raffel 2019) | Span 去噪 | Encoder-Decoder | 750GB text |

### 1.3 预训练的核心要素

预训练的成功取决于三大支柱：

1. **目标函数**：定义模型「学什么」——CLM、MLM 或去噪目标
2. **数据工程**：数据的质量、规模、配比——被广泛认为是最被低估的护城河
3. **Scaling 策略**：如何在给定计算预算下，最优分配模型参数与数据量

---

## 2. 预训练目标

### 2.1 目标函数对比

| 目标 | 全称 | 形式 | 代表模型 |
|------|------|------|---------|
| **CLM** | Causal Language Modeling | 自回归预测下一 token | GPT/LLaMA/Qwen/DeepSeek |
| **MLM** | Masked Language Modeling | 预测被遮掩的 token | BERT/RoBERTa |
| **Prefix LM** | Prefix Language Modeling | 前缀双向可见，后缀自回归预测 | GLM/PaLM（部分） |
| **Span 去噪** | Span Denoising | 随机遮盖连续 span，模型重建 | T5 |

### 2.2 CLM（因果语言模型）

CLM 是当前 LLM 的标准选择。给定输入序列 $\mathbf{x} = (x_1, x_2, \ldots, x_T)$，模型逐个预测下一个 token，训练目标为最大化对数似然：

$$\mathcal{L}_{CLM}(\theta) = \max_{\theta} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

其中 $x_{<t} = (x_1, \ldots, x_{t-1})$ 表示前 $t-1$ 个 token。在实现上，损失等价于对所有位置的交叉熵：

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

**因果掩码**：通过下三角 Attention Mask 确保每个 token 只能看到其之前的 token，实现并行训练。

```python
# 因果注意力掩码示意
# token 0: [0, -inf, -inf, -inf]  ← 只看自己
# token 1: [0,    0, -inf, -inf]  ← 看 0,1
# token 2: [0,    0,    0, -inf]  ← 看 0,1,2
# token 3: [0,    0,    0,    0]  ← 看全部
```

### 2.3 MLM（掩码语言模型）

BERT 引入的 MLM 随机遮盖输入中 15% 的 token，让模型预测被遮盖位置：

$$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i \mid \mathbf{x}_{\backslash \mathcal{M}})$$

其中 $\mathcal{M}$ 是被遮盖的位置集合，$\mathbf{x}_{\backslash \mathcal{M}}$ 是未被遮盖的 token。

由于 MLM 依赖双向上下文，天然适合理解任务，但不适合生成。MLM 的另一个局限是训练-推理不匹配（训练时看到 [MASK]，推理时没有）。

### 2.4 Prefix / Span 去噪

**Prefix LM**：前缀部分允许双向注意力，后缀部分使用因果注意力。GLM 系列采用此目标，试图兼顾理解与生成。

**Span 去噪（T5）**：随机遮盖多个连续 span（如 "for inviting"），替换为哨兵 token（如 `<X>`、`<Y>`），模型自回归地生成哨兵 token 及其对应原文：

```text
输入:  "Thank you <X> me to your party <Y> week."
输出:  "<X> for inviting <Y> last <Z>"
```

### 2.5 为何 Decoder-only CLM 胜出

| 因素 | 分析 |
|------|------|
| **统一格式** | 所有任务都可表达为「给定前缀，生成后续」，无需设计任务特定模板 |
| **零样本涌现** | 自回归训练与推理一致，规模足够大时涌现 zero-shot / few-shot 能力 |
| **可扩展性** | 因果掩码使训练高度并行，上下文长度可线性扩展 |
| **上下文学习** | 自然支持 in-context learning，无需额外设计 |
| **Scaling 验证** | OpenAI 的 Scaling Laws 研究最早在 CLM 上验证，路径最清晰 |

---

## 3. 数据工程

> 数据工程是预训练中最被低估的环节。高质量数据 + 合理配比，往往比单纯堆参数更有效。

### 3.1 数据流水线总览

```text
原始语料采集
    │
    ▼
去重（URL / 文档 / MinHash）
    │
    ▼
质量过滤（启发式规则 + 分类器）
    │
    ▼
去污染（移除评测集重叠）
    │
    ▼
领域配比 + 课程学习
    │
    ▼
分词（BPE / SentencePiece）
    │
    ▼
打包（Sample Packing）→ 训练
```

### 3.2 数据采集

预训练数据来源广泛，通常涵盖以下类别：

| 类别 | 来源 | 特点 |
|------|------|------|
| 网页 | CommonCrawl, C4, RefinedWeb | 规模最大，质量参差不齐 |
| 代码 | GitHub, StackExchange, The Stack | 提升推理与结构化能力 |
| 书籍 | Books3, Gutenberg, 电子书 | 高质量长文本，提升连贯性 |
| 学术论文 | arXiv, PubMed, S2ORC | 专业知识与科学推理 |
| 百科 | Wikipedia, Baidu Baike | 高质量结构化知识 |
| 对话/论坛 | Reddit, StackExchange | 多轮交互与问答风格 |

**多语言覆盖**：现代 LLM 通常覆盖中、英、日、韩、法、德、西等数十种语言，按语言比例采样。

### 3.3 去重

重复数据不但浪费训练预算，还可能导致模型记忆训练数据，增加隐私和版权风险。常用去重策略：

| 层级 | 方法 | 说明 |
|------|------|------|
| URL 去重 | 精确匹配 | 同一 URL 只保留一份 |
| 文档级去重 | n-gram 重叠 / MinHash | 近似去重，检出高相似度文档 |
| 子串级去重 | 精确子串匹配 | 移除跨文档重复的段落 |
| 跨源去重 | MinHash + LSH | 去除不同来源间的重复 |

**MinHash 原理**：将文档的 n-gram 集合通过多个哈希函数映射为签名，通过比较签名估计 Jaccard 相似度，配合 LSH（Locality-Sensitive Hashing）实现高效近似去重。

### 3.4 质量过滤

过滤低质量数据是提升模型性能的关键。常用方法：

**启发式规则**：
- 文本长度过短或过长 → 移除
- 特殊符号/乱码过多 → 移除
- 大写字母占比过高 → 移除
- 重复 n-gram 比例过高 → 移除
- 停用词比例异常 → 移除

**分类器过滤**：
- 训练轻量分类器（如 FastText）区分高质量与低质量文本
- 使用 Wikipedia 等高质文本为正例，随机网页为负例
- 对每条数据打分，保留分数高于阈值的数据

**模型过滤**（FineWeb 风格）：
- 使用已训练模型对数据质量打分
- 更精确但计算成本更高
- FineWeb 证明高质量过滤可大幅提升最终模型性能

### 3.5 去污染（Decontamination）

去污染的目标是移除训练集中与评测基准重叠的文本，防止数据泄漏导致的虚假性能提升。

**常见方法**：
- **n-gram 重叠检测**：在 13-gram 级别检查训练数据与评测集的重叠，移除高度重叠的文档
- **语义相似度**：使用嵌入模型检测语义层面相似的文本
- **基准覆盖**：针对 MMLU、GSM8K、HellaSwag、HumanEval 等常用基准逐一清理

### 3.6 数据配比

数据配比直接影响模型在不同能力上的表现。不同模型有不同的配比哲学：

**LLaMA 1 公开配比**：

| 数据源 | 占比 | 说明 |
|--------|------|------|
| CommonCrawl | 67.0% | 主力网页数据 |
| C4 | 15.0% | 经过过滤的网页 |
| GitHub | 4.5% | 代码能力 |
| Wikipedia | 4.5% | 百科知识 |
| Books | 4.5% | 书籍长文本 |
| ArXiv | 2.5% | 学术/科学 |
| StackExchange | 2.0% | 问答/技术讨论 |

**配比经验**：
- 代码占比过高可能损害自然语言流畅度，但能显著提升推理和数学能力
- 学术论文占比过高可能使模型风格过于「学术化」
- 多语言数据的配比需谨慎平衡，避免「多语诅咒」
- DeepSeek 等模型在训练后期会提高代码/数学比例（课程学习）

### 3.7 数据课程的启示

**RefinedWeb / FineWeb 的结论**：Penedo 等人（2023/2024）证明，仅凭 CommonCrawl 网页数据，经过严格的质量过滤，就能训练出接近甚至超越使用多源混合数据的模型。这颠覆了「必须精心配比多源数据」的传统认知，说明**数据过滤质量远重要于数据来源多样性**。

### 3.8 分词（Tokenization）

预训练前需要将原始文本转换为 token 序列。主流 LLM 使用 **BPE（Byte-Pair Encoding）** 或 **SentencePiece** 实现。

| 模型 | 分词器 | 词表大小 | 特殊设计 |
|------|--------|---------|---------|
| GPT-3 | BPE (tiktoken) | 50K | 空格感知 |
| LLaMA 1/2 | SentencePiece BPE | 32K | 字节回退 |
| LLaMA 3 | BPE (tiktoken) | 128K | 扩展词表，支持更多语言 |
| Qwen 2.5 | BPE | 152K | 多语言优化 |
| DeepSeek | BPE | 128K | 支持中英为主 |

> 分词详见 [分词与词向量](./tokenizer-embedding.md)。

---

## 4. Scaling Laws

### 4.1 Kaplan Scaling Laws（2020）

OpenAI 的 Kaplan 等人最早系统研究了 LLM 的 Scaling 行为。主要发现：

- 模型性能与**模型参数量**、**数据量**、**计算量**三个因素呈幂律关系
- 在当时条件下，结论是「参数优先」——给定计算预算，应优先增大模型而非数据
- 跨两个数量级的拟合表明，损失 $L$ 与各因素满足：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

其中 $N$ 是参数量，$N_c$ 和 $\alpha_N$ 是拟合常数。

### 4.2 Chinchilla Scaling Laws（Hoffmann 2022）

DeepMind 的 Hoffmann 等人对 Kaplan 的结论进行了重要修正：

**核心发现**：参数与数据应**同步增长**，而非参数优先。Kaplan 低估了数据的重要性，原因在于其只分析了少量训练步数。

**Chinchilla 最优**：给定计算预算 $C$（FLOPs），最优分配为：

- 参数 $N_{opt} \propto C^{0.5}$
- 数据 $D_{opt} \propto C^{0.5}$
- 约 **20 token / 参数**

**通用损失公式**：

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中：

- $E$：不可约损失（irreducible loss），反映数据分布的固有熵
- $A/N^\alpha$：模型容量有限带来的损失
- $B/D^\beta$：数据量有限带来的损失
- $\alpha \approx 0.34$，$\beta \approx 0.28$（Chinchilla 拟合结果）

**参数效率对照**：

| 模型 | 参数量 | Chinchilla 最优数据 | 实际数据 | 是否满足 |
|------|--------|---------------------|---------|---------|
| GPT-3 | 175B | 3.5T | 300B | 严重不足 |
| Chinchilla | 70B | 1.4T | 1.4T | 精确满足 |
| LLaMA 7B | 7B | 140B | 1T | 已超过 |
| LLaMA 70B | 70B | 1.4T | 1.4T | 精确满足 |

> 注意：LLaMA 系列故意在 Chinchilla 最优之上继续训练，以在固定推理成本下榨取更多性能——这是「**推理预算优先**」的设计哲学。

### 4.3 数据约束下的 Scaling（Muennighoff 2023）

当高质量数据逐渐枯竭，能否通过**重复训练**（多个 epochs）继续提升性能？

**核心发现**：
- 有限数据下，重复 4 个 epochs 内收益递减，但仍有明显边际收益
- 超过 4 epochs 后收益急剧衰减，但**不会为零**
- 不同数据来源对重复的敏感度不同：代码数据的重复收益高于自然语言

**数据约束下的修正损失**：

$$L(N, D, E) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}\cdot f(E)$$

其中 $E$ 是 epoch 数，$f(E)$ 是重复衰减函数——随 epoch 增加而上升，但渐近趋缓。

**实践启示**：
- 优先扩充数据来源，而非重复训练
- 在数据受限时，适量重复（2-4 epochs）+ 数据增强可缓解过拟合
- 「数据质量」比「数据独特性」更重要——提升过滤质量往往比新增来源更有效

### 4.4 涌现能力（Emergent Abilities）

Wei 等人（2022）观察到，某些能力在模型规模跨越特定阈值后突然出现，而非平滑增长：

- **涌现能力示例**：算术推理、多语言翻译、代码生成、思维链推理
- **涌现阈值**：通常在 10B-100B 参数量附近
- **争议**：部分研究认为涌现是评价指标不连续（如准确率 vs 对数概率）造成的假象（Schaeffer 2023）

---

## 5. 训练策略

### 5.1 优化器与超参数

现代 LLM 预训练普遍使用 **AdamW** 优化器：

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| 优化器 | AdamW | 带解耦权重衰减的 Adam |
| $\beta_1, \beta_2$ | 0.9, 0.95 | 动量参数 |
| 权重衰减 | 0.1 | 大模型常用较大权重衰减 |
| 梯度裁剪 | 1.0 | 按全局范数裁剪 |
| 峰值学习率 | 1e-4 ~ 3e-4（小模型）/ 6e-5 ~ 1.5e-4（大模型） | 模型越大，lr 越低 |
| 学习率调度 | Cosine decay | Warmup → 峰值 → Cosine 衰减至 10% |
| Warmup 步数 | 2000 | 约占总步数的 1-2% |
| Batch size | 4M tokens | 通常按 token 数而非样本数计算 |

**学习率调度曲线**：

```text
lr
 │
 │     ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │    ╱                                    ╲
 │   ╱                                      ╲
 │  ╱                                        ╲___
 │ ╱
 └────────────────────────────────────────────────→ steps
    warmup      cosine decay to 10% peak lr
```

### 5.2 混合精度训练

| 精度 | 用途 | 说明 |
|------|------|------|
| FP32 | 主权重 | 保持精度，用于参数更新 |
| BF16 | 前向 + 反向 | 计算梯度，动态范围与 FP32 一致 |
| FP16 | 前向 + 反向（旧方案） | 需 loss scaling 防下溢 |
| FP8 | 前向（超大模型） | 进一步压缩，需硬件支持（H100+） |

**BF16 优势**：与 FP32 有相同的指数位（8 位），动态范围一致，无需 loss scaling，是现代 LLM 训练的标配。

### 5.3 样本打包（Sample Packing）

传统做法是将每个样本 padding 到序列最大长度，但大量 padding token 浪费计算。**Sample Packing** 将多个短样本拼接为一条长序列：

```text
# 无打包（25%-50% 计算浪费）
[Doc A               ][PAD][PAD][PAD]...
[Doc B     ][PAD][PAD][PAD][PAD][PAD]...

# 有打包（接近 0 浪费）
[Doc A               | Doc B     | Doc C           ]...
```

实现要点：
- 使用特殊分隔符（如 `<EOS>`）分隔不同文档
- 修改 Attention Mask，使不同文档之间不可见（防止跨文档信息泄漏）
- 配合「文档边界」随机排列，增加训练多样性

### 5.4 序列长度课程

许多模型在训练过程中逐步增加序列长度：

| 阶段 | 序列长度 | 目的 |
|------|---------|------|
| 初始 | 2048 / 4096 | 大部分样本有效，训练速度最快 |
| 中期 | 8192 / 16384 | 引入长文本能力 |
| 后期 | 32768 / 131072 | 长上下文建模 |

**注意**：RoPE 等位置编码支持外推，但训练中逐步增加序列长度仍然是更保险的做法，可以避免长序列训练不稳定的问题。

### 5.5 Token 预算规划

遵循 Chinchilla 定律，给定参数量的 Token 预算：

| 参数量 | Chinchilla 最优 | 常见实践 | 说明 |
|--------|----------------|---------|------|
| 1B | 20B | 50-100B | 训练到收敛 |
| 7B | 140B | 1T-2T | 推理预算优先 |
| 13B | 260B | 1T-2T | 推理预算优先 |
| 70B | 1.4T | 1.4T-2T | 接近最优 |
| 405B (LLaMA 3.1) | 8.1T | 15T+ | 超量训练，极致性能 |

### 5.6 分布式训练

预训练需要在数千 GPU 上并行进行。常用并行策略：

| 策略 | 全称 | 原理 | 通信开销 |
|------|------|------|---------|
| DP | Data Parallelism | 每 GPU 持有完整模型，拆分 batch | 梯度 AllReduce |
| PP | Pipeline Parallelism | 按层切分，流水线执行 | 层间激活传递 |
| TP | Tensor Parallelism | 单层内矩阵切分，并行计算 | 高（同步通信） |
| ZeRO | Zero Redundancy Optimizer | 分片优化器状态/梯度/参数 | 中等 |
| SP | Sequence Parallelism | 长序列沿序列维度切分 | 中等 |

> 详见 [分布式训练](../training/distributed.md)。

---

## 6. 训练监控与评测

### 6.1 训练损失与困惑度

**困惑度（Perplexity, PPL）** 是预训练阶段最核心的监控指标：

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log P_\theta(x_i \mid x_{<i})\right)$$

PPL 越低越好。PPL = 10 意味着模型在每一步预测下一个 token 时，平均面对 10 个等可能的选择，即「平均每步有 10 个候选」。

**PPL 与损失的关系**：

$$\text{PPL} = \exp(\text{loss})$$

监控要点：
- 训练 loss 应持续下降，无明显震荡或平台
- 验证 loss 上升是过拟合信号——多 epochs 训练时尤其需要关注
- 不同数据源可分别监控 PPL（代码 PPL、数学 PPL 等）
- 梯度范数异常增大往往是训练不稳定的前兆

### 6.2 下游任务评测

预训练过程中，定期在标准基准上做 few-shot 评测，以追踪模型能力的增长：

| 基准 | 领域 | 评测方式 |
|------|------|---------|
| MMLU | 多学科知识 | 5-shot |
| GSM8K | 数学推理 | 8-shot / CoT |
| HellaSwag | 常识推理 | 10-shot |
| HumanEval / MBPP | 代码生成 | 0-shot / pass@k |
| PIQA / ARC | 物理/科学常识 | 0-shot |
| BBH | 复杂推理 | 3-shot |
| WinoGrande | 代词消歧 | 0-shot |

评测频率：通常每 N 个 step（如 1000）或每 N 个 token（如 10B）进行一次评测，计算量较大时适当降低频率。

### 6.3 能力涌现监控

**涌现现象**（Wei et al. 2022）：某些能力在模型规模跨越阈值后突然显现，而非平滑增长。

典型涌现模式：
- **小模型随机**：< 10B 参数时，某些任务（如多步推理）表现接近随机
- **阈值突破**：超过某个规模后，能力突然跃升
- **持续增长**：超过阈值后，能力随规模继续增长

**关键观察**：涌现意味着在小模型上做消融实验可能无法预测大模型行为，需要谨慎外推。

### 6.4 训练稳定性监控

| 指标 | 正常表现 | 异常信号 |
|------|---------|---------|
| Loss | 平滑下降 | 突然 spike 或长时间不下降 |
| 梯度范数 | 稳定或缓慢变化 | 急剧增大（可能 loss spike） |
| 学习率 | 按 schedule 变化 | 无 |
| 吞吐量 | 稳定 | 突然下降（硬件/网络问题） |
| GPU 利用率 | > 90% | 持续偏低（数据加载瓶颈） |

**Loss Spike 应对**：若出现 loss spike，通常回滚到最近的 checkpoint，跳过引起 spike 的数据 batch，或降低学习率后继续训练。

---

## 7. 参考文献

[1] Brown, T., et al. *Language Models are Few-Shot Learners*. NeurIPS, 2020. (GPT-3)

[2] Raffel, C., et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR, 2020. (T5)

[3] Hoffmann, J., et al. *Training Compute-Optimal Large Language Models*. NeurIPS, 2022. (Chinchilla Scaling Laws)

[4] Touvron, H., et al. *LLaMA: Open and Efficient Foundation Language Models*. arXiv:2302.13971, 2023.

[5] Touvron, H., et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288, 2023.

[6] Gao, L., et al. *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. arXiv:2101.00027, 2020.

[7] Penedo, G., et al. *The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only*. NeurIPS, 2023.

[8] Penedo, G., et al. *FineWeb: Decanting the Web for the Finest Text Data at Scale*. arXiv:2406.17557, 2024.

[9] Muennighoff, N., et al. *Scaling Data-Constrained Language Models*. NeurIPS, 2023.

[10] Kaplan, J., et al. *Scaling Laws for Neural Language Models*. arXiv:2001.08361, 2020.

[11] Wei, J., et al. *Emergent Abilities of Large Language Models*. TMLR, 2022.

[12] Schaeffer, R., et al. *Are Emergent Abilities of Large Language Models a Mirage?* NeurIPS, 2023.

[13] Devlin, J., et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL, 2019.

[14] Dubey, A., et al. *The Llama 3 Herd of Models*. arXiv:2407.21783, 2024.

[15] DeepSeek-AI. *DeepSeek-V3 Technical Report*. arXiv:2412.19437, 2024.