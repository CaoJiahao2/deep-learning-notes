# 分词与词嵌入详解

> LLM 不能直接吃"字"：文本先被切成 token，再映射为向量。分词（Tokenizer）与词嵌入（Embedding）是模型理解语言的入口，也直接决定推理成本（按 token 计费/计显存）。

---

## 目录

1. [概述](#1)
2. [主流分词算法](#2)
3. [词嵌入](#3)
4. [词表设计与训练策略](#4)
5. [词向量演进](#5)
6. [参考文献](#6)

---

## 1. 概述

### 1.1 为什么需要分词

自然语言由字符（characters）组成，但模型不能直接以字符为单位处理文本。原因有三：

- **全词表不可行**：自然语言有无限多的词，无法为每个词分配一个独立 ID。
- **未登录词（OOV, Out-of-Vocabulary）**：无论词表多大，总有没见过的词（新词、专有名词、拼写变体），全词表方法对此无能为力。
- **稀疏性**：长尾词出现频率极低，模型很难学到有效表示。

**子词（subword）**分词是当前主流方案：将词拆分为更小的有意义片段（如 "unhappiness" → "un" + "happiness" 或 "un" + "happi" + "ness"），在「词表大小」与「覆盖度」之间取得平衡。罕见词被拆成常见子词片段，彻底消除 OOV 问题。

### 1.2 分词与词嵌入的位置

在 LLM 的推理管线中，分词和词嵌入处于最前端：

```text
原始文本 → Tokenizer → [token ids] → Embedding Lookup → [dense vectors] → Transformer
```

每个 token 先被映射为一个整数 ID，再通过嵌入层（Embedding Layer）查找得到 $d$ 维稠密向量。这个向量是模型后续所有计算的起点，分词质量直接影响下游性能。

### 1.3 为什么关注分词

- **推理成本**：API 按 token 数计费，同一段文本用不同 tokenizer 得到的 token 数可能相差 30% 以上。
- **显存占用**：dense vectors 的序列长度 = token 数，直接影响 KV cache 和激活值大小。
- **多语言公平性**：中文、日文等语言通常被切成更多 token（同等语义下），导致推理成本更高、延迟更长。

---

## 2. 主流分词算法

### 2.1 算法对比总览

| 算法 | 核心思路 | 训练方向 | 代表模型 |
|------|---------|---------|---------|
| **BPE** | 合并最高频相邻符号对 | 自底向上（合并） | GPT-2/3/4、LLaMA |
| **WordPiece** | 类 BPE，按似然增益选合并对 | 自底向上（合并） | BERT、Electra |
| **Unigram** | 从全词表删词使似然最大 | 自顶向下（删减） | ALBERT、T5（SentencePiece） |
| **SentencePiece** | 语言无关，直接对原始文本建模（含空格） | 任意（BPE/Unigram） | T5、LLaMA、Qwen |

### 2.2 BPE（Byte-Pair Encoding）

**核心思想**：从字符级出发，反复将「同现频率最高」的相邻符号合并为新的 token，直到词表达到预设大小。

**训练过程**：

1. 将训练语料中的每个词拆成字符序列，末尾加 `</w>` 标记词边界。
2. 统计所有相邻符号对的频率。
3. 合并频率最高的符号对，生成新 token。
4. 重复步骤 2-3，直到词表大小达到预设值。

```text
BPE 合并过程示例（词表大小 = 3 次合并）：

初始状态（字符级）：
  "h u g"  → 频率 10
  "p u g"  → 频率 5
  "h u g s" → 频率 5
  "m u g s" → 频率 2

第 1 次合并：最高频对 "u" + "g" → "ug"
  "h ug"  "p ug"  "h ug s"  "m ug s"

第 2 次合并：最高频对 "h" + "ug" → "hug"
  "hug"  "p ug"  "hug s"  "m ug s"

第 3 次合并：最高频对 "ug" + "s" → "ugs"
  "hug"  "p ug"  "hug ugs"  "m ugs"
```

**推理时**：遇到未见过的词，从最短字符开始，逐步合并直到无法继续合并，最终得到一组子词 token。

**BPE 的特点**：

- 常见词整体保留为单个 token，罕见词拆成多个子词。
- 贪婪式合并可能导致次优分割（如 "hug" 合并后，"hugs" 只能用 "hug" + "s" 而非 "hug" + "s" 的理想组合）。
- 字节级 BPE（Byte-level BPE, BBPE）进一步将基础单元从字符扩展到字节，彻底消除未知字符问题（GPT-2 及后续）。

### 2.3 WordPiece

**核心思想**：与 BPE 同样是合并式算法，但选择合并对的标准不是频率，而是**语言模型似然增益**。

WordPiece 选择使训练语料似然 $P(\text{corpus})$ 提高最大的符号对进行合并。具体而言，对于候选对 $(a, b)$，计算：

$$score(a, b) = \frac{count(a, b)}{count(a) \times count(b)}$$

即这对符号的共现概率相对于独立概率的比值。合并得分最高的对。

**与 BPE 的关键区别**：

| 维度 | BPE | WordPiece |
|------|-----|-----------|
| 选择标准 | 频率 | 似然增益 |
| 拆分策略 | 基于合并规则 | 基于最长匹配（贪心） |
| 词内标记 | `</w>` 词尾 | `##` 前缀表示非首子词 |

```text
WordPiece 拆分示例：
  "unaffable" → ["un", "##aff", "##able"]
  "playing"   → ["play", "##ing"]
```

BERT 使用 WordPiece 词表，大小为 30,522。中文通常被切成单字或常见双字组合。

### 2.4 Unigram Language Model

**核心思想**：与 BPE/WordPiece 的「自底向上合并」相反，Unigram 采用「自顶向下删减」策略。

**训练过程**：

1. 初始化一个足够大的种子词表（如所有字符 + 常见子串）。
2. 固定词表，用 EM 算法估计每个 token 的 Unigram 概率 $p(x_i)$。
3. 对每个 token 计算「去掉它后似然损失多少」，损失最小的 token 被删除。
4. 重复步骤 2-3，直到词表缩小到预设大小。

在给定词表的条件下，一个句子 $X = (x_1, x_2, \ldots, x_M)$ 的概率为：

$$P(X) = \prod_{i=1}^{M} p(x_i)$$

其中分词方案取概率最大的 Viterbi 路径。

**Unigram 的特点**：

- 可以为同一句子生成多种分词候选，按概率选择。
- 天然支持**子词正则化**（subword regularization）：训练时随机采样不同分词方案，增强模型鲁棒性。
- 比 BPE 更灵活，但训练成本更高（需要 EM 迭代）。

### 2.5 SentencePiece

**核心思想**：SentencePiece 不是一种新算法，而是对 BPE 和 Unigram 的**统一实现框架**。它的核心创新是**语言无关**：直接将原始文本视为 Unicode 字符序列，不依赖任何预分词（pre-tokenization，如空格分词）。

**设计要点**：

- 把空格视为普通字符（用 `▁` / U+2581 表示），因此 "hello world" 被处理为 `▁hello ▁world`。
- 不需要语言特定的分词规则（如英文的空格分词、中文的分词工具），一份代码适用所有语言。
- 训练速度极快（C++ 实现），是当前 LLM 事实上的标准分词工具。

```text
SentencePiece 处理示例：
  "你好世界 Hello World"
  → ▁ 你好 世界 ▁ Hello ▁ World
```

**SentencePiece 的两种模式**：

| 模式 | 底层算法 | 代表模型 |
|------|---------|---------|
| `--model_type=bpe` | BPE | LLaMA 系列 |
| `--model_type=unigram` | Unigram + 子词正则化 | T5, ALBERT, Qwen 系列 |

### 2.6 多语言场景下的分词

SentencePiece 不分语言，直接将所有语料混合训练，这对多语言模型（如 Qwen、LLaMA 3）是理想选择。但不同语言在分词效率上存在显著差异：

```text
同一句话的分词对比（LLaMA tokenizer）：

英文 "The quick brown fox jumps over the lazy dog"
  → 9 tokens [The, quick, brown, fox, jumps, over, the, lazy, dog]

中文 "敏捷的棕色狐狸跳过了懒惰的狗"
  → 18 tokens [敏捷, 的, 棕, 色, 狐, 狸, 跳, 过, 了, 懒, 惰, 的, 狗]
```

中文通常被切得更细（每个汉字或双字组合），导致同等语义内容下 token 数偏多。这是多语言 LLM 在中文场景下推理成本更高、延迟更大的根本原因。部分中文优化模型（如 Qwen 系列）会扩充中文 token 提高压缩率。

---

## 3. 词嵌入

### 3.1 嵌入层

嵌入层（Embedding Layer）是一个可学习的查找表 $E \in \mathbb{R}^{V \times d}$，其中 $V$ 是词表大小，$d$ 是嵌入维度。它把每个 token ID 映射为一个 $d$ 维稠密向量：

$$e_i = E[i, :] \in \mathbb{R}^d$$

从实现角度看，嵌入层本质上是一个 `nn.Embedding` 模块，根据 token ID 做行索引：

```python
import torch
import torch.nn as nn

# 词表大小 32000，嵌入维度 4096
embed = nn.Embedding(num_embeddings=32000, embedding_dim=4096)

# 输入 token IDs: [batch=1, seq_len=3]
token_ids = torch.tensor([[1, 254, 998]])
vectors = embed(token_ids)  # shape: [1, 3, 4096]
```

### 3.2 用 tiktoken 观察分词

```python
import tiktoken

# 使用 GPT-4 的 tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# 中文分词示例
text = "大模型让AI更易用"
ids = enc.encode(text)
print(f"Token IDs: {ids}")
print(f"Token 数: {len(ids)}")

# 逐个解码查看每个 token
for i in ids:
    print(f"  {i}: '{enc.decode([i])}'")

# 输出示例：
# Token IDs: [16357, 255, 45380, 119, 119, 120, 22970, 8858, 222, 247, 29879]
# Token 数: 11（8 个中文字符拆成了 11 个 token）
```

### 3.3 嵌入层参数量

词表大小 $V$ 直接影响嵌入层的参数量：

$$N_{embed} = V \times d$$

同时，输出层（LM Head）通常共享嵌入层权重（weight tying），参数量为 $V \times d$。对于大词表，这是不可忽略的显存消耗：

| 模型 | 词表大小 $V$ | 嵌入维度 $d$ | 嵌入参数量 |
|------|-------------|-------------|-----------|
| GPT-2 | 50,257 | 768 | ~38.6M |
| LLaMA 2 | 32,000 | 4,096 | ~131M |
| LLaMA 3 | 128,000 | 4,096 | ~524M |
| Qwen 2 | 152,064 | 3,584 | ~545M |

LLaMA 3 和 Qwen 2 的大词表使嵌入参数达到 5 亿以上，约占总参数的 0.7%-1%，但训练时嵌入层梯度稀疏（每个 batch 仅激活少量 token），实际训练开销相对可控。

### 3.4 特殊 Token

每个 tokenizer 都预留了若干特殊 token，具有特定语义功能：

| Token | 作用 | 典型值 |
|-------|------|--------|
| `<bos>` / `<s>` | 序列开始（Beginning of Sequence） | ID=1 |
| `<eos>` / `</s>` | 序列结束（End of Sequence） | ID=2 |
| `<pad>` | 填充（Padding） | ID=0 |
| `<unk>` | 未知 token（Unknown） | ID=0 或 3 |
| `<sep>` | 分隔符（Separator, BERT 用） | ID=102 |

**对话模型中的角色标记**（ChatML 格式为例）：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的？<|im_end|>
```

这些特殊 token（如 `<|im_start|>`、`<|im_end|>`）在训练时被加入词表，模型通过它们区分对话角色和轮次。在推理时，这些 token 的嵌入同样参与计算，但通常不参与 loss 计算。

---

## 4. 词表设计与训练策略

### 4.1 词表大小的权衡

词表大小 $V$ 是分词器设计中最关键的参数：

| 词表倾向 | 优势 | 劣势 |
|----------|------|------|
| **大词表**（$V \geq 100K$） | 每个 token 涵盖更多信息，序列更短，推理更快 | 嵌入层参数量大，长尾 token 训练不充分，泛化差 |
| **小词表**（$V \leq 32K$） | 嵌入层轻量，罕见词也能由子词组合覆盖 | 序列变长，推理成本上升，信息密度低 |

当前主流 LLM 收敛到 $V = 32K \sim 152K$。LLaMA 3 将词表从 32K 扩到 128K 后，同等文本的 token 数减少了约 15%，在长上下文场景下节省显著。

### 4.2 中文与代码场景的词表优化

通用 tokenizer 在特定领域可能效率不佳：

- **中文**：BPE/SentencePiece 训练时中文语料占比不足，导致中文字符被切得过细。解决方案：提高中文语料比例、使用中文专属 tokenizer（如 Qwen 的 `qwen-tokenizer`）、在通用词表基础上扩增中文 token。
- **代码**：缩进、特殊符号（`{`, `(`, `->`）在通用 tokenizer 中可能被切碎。解决方案：使用代码专用 tokenizer（如 CodeLLaMA 的 tokenizer）、在训练语料中加入足够比例的代码数据。

### 4.3 词表扩张

当需要新加入特殊 token（如领域术语、格式标记）或扩增某种语言的 token 时，需要执行词表扩张：

```python
# 词表扩张伪代码
original_vocab_size = 32000
new_tokens = ["<|tool_call|>", "<|tool_response|>", "<|code|>", "</|code|>"]
num_new = len(new_tokens)

# 1. 扩展 tokenizer 词表
tokenizer.add_tokens(new_tokens)

# 2. 扩展嵌入层和输出层
old_embed = model.get_input_embeddings()  # [32000, 4096]
new_embed = nn.Embedding(32000 + num_new, 4096)
new_embed.weight[:32000] = old_embed.weight  # 保留原有权重

# 3. 新 token 随机初始化
new_embed.weight[32000:] = torch.randn(num_new, 4096) * 0.02

# 4. 替换模型层
model.set_input_embeddings(new_embed)
model.set_output_embeddings(new_embed)  # weight tying
```

**关键注意**：新 token 的嵌入是随机初始化的，必须经过少量继续训练（通常数千到数万步，仅训练新增参数或全模型小学习率训练）才能收敛。直接使用会严重损害模型输出质量。

### 4.4 分词对推理的全链路影响

分词选择影响远不止 token 数：

- **KV cache 大小**：token 数越多，KV cache 越大，长上下文场景下显存压力显著。
- **延迟（Latency）**：自回归解码每步生成一个 token，更短的序列意味着更少的解码步数。
- **跨模型迁移**：不同 tokenizer 的词表不通用。从一个模型的 tokenizer 切换到另一个时，需要重新分词或建立 token 映射表，且嵌入层无法直接复用。

**实际案例**（以 1000 字中文文本为例）：

| Tokenizer | Token 数 | 相对成本 |
|-----------|---------|---------|
| GPT-4 (cl100k_base) | ~2,200 | 1.00x |
| GPT-2 (p50k_base) | ~2,800 | 1.27x |
| LLaMA 2 (sentencepiece) | ~2,500 | 1.14x |
| 中文优化 tokenizer | ~1,200 | 0.55x |

---

## 5. 词向量演进

### 5.1 静态词向量时代

在 LLM 出现之前，词向量（Word Embedding）是 NLP 的基石。静态词向量为每个词分配一个固定向量，不考虑上下文。

**Word2Vec（Mikolov et al., 2013）**：

两种训练架构：

- **Skip-gram**：给定中心词 $w_t$，预测上下文词 $w_{t-k}, \ldots, w_{t+k}$。适合小数据集。
- **CBOW（Continuous Bag-of-Words）**：给定上下文词，预测中心词。训练更快，适合大数据集。

核心思想是通过预测任务学习词的分布式表示，语义相近的词在向量空间中距离更近。经典算术关系：$vec(\text{king}) - vec(\text{man}) + vec(\text{woman}) \approx vec(\text{queen})$。

**GloVe（Pennington et al., 2014）**：

Global Vectors 结合了全局矩阵分解和局部上下文窗口两种方法。基于词-词共现矩阵 $X$（$X_{ij}$ 表示词 $j$ 在词 $i$ 的上下文中出现的次数），优化目标：

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

其中 $f(X_{ij})$ 是加权函数，衰减高频词的影响。GloVe 在词类比和相似度任务上优于 Word2Vec。

**静态词向量的局限**：

- 一个词只有一个向量，无法处理一词多义（如 "bank" 可以是「银行」或「河岸」）。
- 无法捕捉句法结构、语义角色等上下文依赖信息。

### 5.2 上下文词嵌入

**ELMo（Peters et al., 2018）**：

Embeddings from Language Models 首次引入上下文感知的词表示。用双向 LSTM 在大规模语料上训练语言模型，词向量是对各层隐状态的加权组合。同一个词在不同句子中获得不同向量，解决了多义问题。

**BERT（Devlin et al., 2019）**：

通过 Masked Language Model（MLM）预训练，每个 token 的表示依赖于整个输入序列的上下文。BERT 的嵌入 = Token Embedding + Segment Embedding + Position Embedding。开启了「预训练 + 微调」范式，使上下文词嵌入成为 NLP 标配。

### 5.3 现代 LLM 的嵌入层

现代 LLM 的嵌入层回归简洁：一个 `nn.Embedding` 层，端到端训练，与 tokenizer 强耦合。相比早期静态词向量，有两个关键变化：

- **不再单独训练词向量**：嵌入层与整个 Transformer 一起预训练，token 的语义由模型内部表示而非静态向量承载。
- **Token 为基本单位**：不再是「词」向量，而是「token」向量。一个「词」可能被拆成多个 token，每个 token 有自己的嵌入，语义由 Transformer 层聚合。

**输入构造流程**：

```text
文本: "你好，世界"
  ↓ Tokenizer
Token IDs: [1, 12345, 678, 23456, 2]
  ↓ Embedding Lookup
Input Embeddings: [5, 4096] 矩阵
  ↓ + Positional Encoding (RoPE/ALiBi/learned)
Final Input: [5, 4096] → 送入 Transformer
```

---

## 6. 参考文献

[1] Sennrich, R., Haddow, B., & Birch, A. *Neural Machine Translation of Rare Words with Subword Units*. ACL, 2016. (BPE)

[2] Wu, Y., et al. *Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation*. arXiv:1609.08144, 2016. (WordPiece)

[3] Kudo, T. & Richardson, J. *SentencePiece: A Simple and Language Independent Subword Tokenizer and Detokenizer for Neural Text Processing*. EMNLP, 2018.

[4] Kudo, T. *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*. ACL, 2018. (Unigram)

[5] Mikolov, T., Chen, K., Corrado, G., & Dean, J. *Efficient Estimation of Word Representations in Vector Space*. ICLR Workshop, 2013. (Word2Vec)

[6] Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS, 2013. (Word2Vec + Negative Sampling)

[7] Pennington, J., Socher, R., & Manning, C. *GloVe: Global Vectors for Word Representation*. EMNLP, 2014.

[8] Peters, M. E., et al. *Deep Contextualized Word Representations*. NAACL, 2018. (ELMo)

[9] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL, 2019.

[10] Radford, A., et al. *Language Models are Unsupervised Multitask Learners*. OpenAI, 2019. (GPT-2, Byte-level BPE)