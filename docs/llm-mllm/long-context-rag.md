# 长上下文与检索增强生成

> 从位置编码外推到 Ring Attention，从稠密检索到 GraphRAG —— 两条让模型"看见更多"的技术路线。

---

## 目录

1. [概述](#1)
2. [长上下文的核心挑战](#2)
3. [位置编码外推](#3)
4. [高效注意力机制](#4)
5. [RAG 基础架构](#5-rag)
6. [高级检索技术](#6)
7. [长上下文 vs RAG：取舍与融合](#7-vs-rag)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1. 概述

大模型的知识受限于训练截止时间和上下文窗口。有两种互补的思路让模型在推理时获取更多信息：

- **长上下文（Long Context）**：扩展模型的上下文窗口，直接将更多文本塞入 prompt。从 GPT-4 的 8K 到 Claude 3 的 200K、Gemini 的 1M+ Token，长上下文能力快速演进。
- **检索增强生成（RAG, Retrieval-Augmented Generation）**：先从外部知识库检索相关片段，再将其拼入 prompt，让模型只看到与问题最相关的信息。

二者的核心权衡：

$$
\text{长上下文} \sim \text{广而浅的全量浏览} \quad \text{vs.} \quad \text{RAG} \sim \text{准而精的定向检索}
$$

---

## 2. 长上下文的核心挑战

### 2.1 位置编码外推

Transformer 的位置编码为训练长度内的位置设计。当推理序列长度超过训练长度时，位置编码需要外推（extrapolation），否则模型性能骤降。

### 2.2 注意力计算复杂度

标准自注意力的复杂度为 $O(n^2)$，其中 $n$ 为序列长度。当 $n$ 从 4K 扩展到 1M 时，计算量增加 $62500$ 倍。必须配合高效注意力或分布式注意力。

### 2.3 中间丢失（Lost in the Middle）

Liu et al. (2023) 发现，模型对位于上下文中间位置的信息利用率显著低于开头和结尾，即"U 形注意力曲线"。这意味着长上下文不等于有效利用上下文。

### 2.4 显存占用

KV Cache 随序列长度线性增长：

$$
\text{KV Cache} = 2 \times n_{\text{layers}} \times n \times d_{\text{head}} \times n_{\text{head}} \times b
$$

其中 $b$ 为 batch size。长上下文推理时 KV Cache 成为显存瓶颈。

---

## 3. 位置编码外推

### 3.1 位置插值（Position Interpolation, PI）

位置插值（Chen et al., 2023）将超出训练长度的位置线性压缩到训练范围内：

$$
p' = \frac{L_{\text{train}}}{L_{\text{target}}} \cdot p
$$

其中 $p$ 为原始位置，$p'$ 为插值后位置。PI 不引入新的位置编码，仅需少量微调即可将上下文窗口扩展 2-8 倍。

### 3.2 NTK-Aware 插值

NTK 插值改变 RoPE 的旋转基频，而非均匀缩放位置：

$$
\theta_i' = \theta_i \cdot s^{\frac{i}{d}}, \quad s = \frac{L_{\text{target}}}{L_{\text{train}}}
$$

这使得高频维度（编码局部位置信息）几乎不缩放，低频维度（编码全局位置信息）大幅缩放，在不微调的情况下即可实现外推。

### 3.3 YaRN

YaRN（Peng et al., 2023）结合 NTK 插值和温度缩放，并按维度区分插值策略：

- 低频维度使用 NTK 缩放。
- 高频维度保持不变。
- 中间频率维度使用渐进插值。

YaRN 仅需约 400 步微调即可将 LLaMA2 的上下文从 4K 扩展到 128K。

### 3.4 LongRoPE 与渐进式扩展

LongRoPE（Ding et al., 2024）引入两个关键改进：

- **非均匀位置插值**：通过搜索找到最优的各维度缩放因子，而非统一缩放。
- **渐进式扩展**：分阶段从 4K → 64K → 2048K，每阶段微调少量步。

LongRoPE 将 LLaMA2 扩展到 2M Token 上下文，且在短上下文任务上性能不下降。

### 3.5 其他方案

- **ALiBi（Press et al., 2022）**：不用绝对位置编码，而是在注意力分数上加入线性距离偏置，天然支持外推。
- **KERPLE**：将距离偏置从线性改为可学习的幂律/指数衰减。
- **NoPE**：部分研究表明在足够数据下，模型可隐式学习位置信息。

---

## 4. 高效注意力机制

### 4.1 Flash Attention

Flash Attention（Dao et al., 2022）通过 tiling 和重计算将注意力计算从 $O(n^2)$ 显存降为 $O(n)$，同时利用 GPU SRAM 减少 HBM 读写。FlashAttention-2 和 3 进一步优化了并行度和计算效率，是长上下文训练的基础算子。

### 4.2 Ring Attention

Ring Attention（Liu et al., 2023）将序列切分到多个设备上，每个设备只持有部分 KV Cache。设备按环形拓扑传递 KV 块，最终每个设备计算完整的注意力输出：

$$
\text{设备 } i \text{ 持有 } Q_i, \text{ 依次接收 } K_j, V_j \text{ 并累积 } \text{Attn}(Q_i, K_j, V_j)
$$

Ring Attention 使上下文长度可随设备数线性扩展，是 Gemini 等百万级上下文模型的技术基础。

### 4.3 稀疏注意力

- **Sliding Window Attention**：每个 Token 只关注固定窗口内的邻近 Token（Mistral、Longformer）。
- **Dilated Attention**：类似空洞卷积，以不同间隔关注远距离 Token。
- **Routing Attention / MoA**：学习一个路由函数，选择每个 Token 应关注的子集（Reformer、DeepSeek）。

### 4.4 KV Cache 压缩

- **GQA（Grouped Query Attention）**：多个 Query 头共享一组 KV 头，将 KV Cache 减少 $n_{\text{head}}/n_{\text{group}}$ 倍。
- **MLA（Multi-head Latent Attention）**：将 KV 投影到低维隐空间后压缩缓存（DeepSeek-V2）。
- **KV Cache 量化与驱逐**：对 KV Cache 进行低比特量化，或驱逐不重要的 KV 对（H2O、StreamingLLM）。
- **StreamingLLM**：保留注意力池（attention sink）+ 滑动窗口，支持无限长度流式推理，但不具备全局理解能力。

---

## 5. RAG 基础架构

### 5.1 RAG 流水线

标准 RAG 系统由三个阶段组成：

```text
查询 → 检索（Retrieval） → 拼接（Augmentation） → 生成（Generation）
```

**索引阶段（离线）**：

1. 文档采集与清洗。
2. 分块（Chunking）：将长文档切分为适合检索的片段。
3. 向量化（Embedding）：用编码器将每个块映射为稠密向量。
4. 存入向量数据库。

**查询阶段（在线）**：

1. 将用户查询编码为向量。
2. 在向量数据库中进行近邻搜索，返回 Top-$K$ 相关块。
3. 将检索结果与查询拼接为 prompt。
4. LLM 基于上下文生成回答。

### 5.2 分块策略（Chunking）

| 策略 | 说明 |
|------|------|
| 固定长度分块 | 按 Token/字符数切分，简单但可能截断语义 |
| 句子/段落分块 | 按自然语义边界切分，保留完整性 |
| 递归分块 | 优先按段落切，过长再按句子、字符切 |
| 结构化分块 | 按 Markdown 标题/HTML 标签/代码函数切分 |
| 父子分块 | 检索小块定位，返回大块提供上下文 |
| 语义分块 | 用嵌入相似度判断语义边界 |

分块大小是关键超参数：太小丢失上下文，太大稀释检索精度并增加 Token 成本。

### 5.3 稠密检索（Dense Retrieval）

双塔编码器将查询和文档分别编码为向量，用内积或余弦相似度匹配：

$$
\text{score}(q, d) = \text{sim}(E_q(q), E_d(d))
$$

代表模型包括 DPR（Karpukhin et al., 2020）、BGE、E5、GTE 等。关键训练方法：

- **对比学习**：正样本为相关文档，负样本为难负样本（hard negatives）。
- **负样本挖掘**：用 BM25 或上一阶段检索器找高相似度但不相关的文档。
- **知识蒸馏**：用跨编码器（Cross-Encoder）的分数监督双塔训练。

### 5.4 向量数据库

| 数据库 | 特点 |
|--------|------|
| FAISS | Meta 开源，库级使用，支持 GPU 加速 |
| Milvus | 分布式向量数据库，支持大规模生产 |
| Pinecone | 全托管云服务 |
| Weaviate | 支持混合检索和结构化过滤 |
| Qdrant | Rust 实现，高性能过滤检索 |
| pgvector | PostgreSQL 扩展，适合已有 PG 栈 |

核心索引算法包括 HNSW（层次化可导航小世界图）和 IVF（倒排文件索引）。

### 5.5 稀疏检索与混合检索

- **BM25**：基于词频和逆文档频率的经典稀疏检索，对关键词匹配效果好。
- **混合检索**：线性组合稠密检索和稀疏检索分数：

$$
\text{score}_{\text{hybrid}} = \alpha \cdot \text{score}_{\text{dense}} + (1-\alpha) \cdot \text{score}_{\text{sparse}}
$$

混合检索通常优于纯稠密检索，尤其在术语精确匹配（如产品名、错误代码）场景。

---

## 6. 高级检索技术

### 6.1 Rerank（重排序）

第一阶段检索（召回）追求高召回率，返回 Top-$K$（如 100）个候选。Reranker 对候选逐一精排，取 Top-$N$（如 5）：

- **Cross-Encoder Reranker**：将查询和文档拼接后输入 Transformer，输出相关性分数。精度高但速度慢，适合小规模精排。
- **Cohere Rerank / BGE-Reranker**：代表性预训练 Reranker 模型。

### 6.2 查询改写

- **HyDE（Hypothetical Document Embeddings）**：先让 LLM 生成一个假设性回答，用假设回答的向量去检索，而非直接用查询向量。假设回答在语义空间中更接近真实文档。
- **Multi-Query Retrieval**：用 LLM 生成同一查询的多个改写版本，分别检索后合并去重。
- **Step-back Prompting**：让 LLM 生成更抽象的上位问题，检索更广泛的背景知识。

### 6.3 GraphRAG

GraphRAG（Microsoft, 2024）将知识图谱引入 RAG：

1. 用 LLM 从文档中抽取实体和关系，构建知识图谱。
2. 对图进行社区检测（如 Leiden 算法），生成社区摘要。
3. 查询时，可在图中进行多跳推理，或根据查询相关的社区摘要回答全局性问题。

GraphRAG 的优势在于回答"总结整个数据集的主题"等聚合性问题——纯向量检索难以处理这类需要全局理解的查询。

### 6.4 Agentic RAG

Agentic RAG 将检索本身作为 Agent 可调用的工具，由 LLM 自主决定是否检索、如何改写查询、是否需要多轮检索。详见 `agents/agentic-rag.md`。

### 6.5 评估指标

| 指标 | 含义 |
|------|------|
| Hit Rate | 相关文档出现在 Top-K 中的比例 |
| MRR | 首个相关文档排名倒数的均值 |
| NDCG | 考虑排序位置的增益指标 |
| Context Precision | 检索结果中相关块的比例 |
| Faithfulness | 生成回答是否忠于检索上下文 |
| Answer Relevance | 回答与查询的相关程度 |

RAGAS 框架提供了自动化的 RAG 评估流水线。

---

## 7. 长上下文 vs RAG：取舍与融合

### 7.1 对比

| 维度 | 长上下文 | RAG |
|------|---------|-----|
| 信息覆盖 | 可塞入整个语料 | 仅检索相关片段 |
| 成本 | 高（Token 数和显存随长度线性/二次增长） | 低（仅处理 Top-K 片段） |
| 时效性 | 受上下文窗口限制，但不需重训 | 可随时更新知识库 |
| 信息定位 | "中间丢失"问题 | 检索质量决定上限 |
| 全局推理 | 可跨文档综合 | 受限于检索召回 |
| 隐私 | 所有数据都进入模型 | 可做细粒度权限控制 |

### 7.2 融合策略

实际系统通常结合二者：

- **RAG + 长上下文**：检索更多片段（Top-20/50），利用长上下文窗口提供更丰富的上下文。
- **分层 RAG**：先用粗检索定位相关文档组，再在长上下文窗口内做细粒度推理。
- **自适应检索**：简单问题直接回答，复杂问题触发 RAG。

---

## 8. 实践建议

### 8.1 选型建议

- 知识库 < 100K Token 且更新不频繁：长上下文直接塞入即可，无需 RAG。
- 知识库 > 1M Token 或需要频繁更新：必须使用 RAG。
- 需要多跳推理或全局摘要：考虑 GraphRAG。
- 对延迟敏感：用轻量级 Bi-Encoder 检索 + 小 K；对质量敏感：加 Reranker。

### 8.2 RAG 优化清单

- 分块前做文档清洗，去除页眉页脚、导航栏等噪声。
- 用混合检索（BM25 + 稠密），不要只用向量检索。
- 加 Reranker 是性价比最高的质量提升手段。
- 对检索结果做去重和冗余消除，避免相同信息重复占用 Token。
- 用 RAGAS 等工具定期评估检索和生成质量。
- 监控检索命中率，低于阈值时触发回退策略（如多查询改写）。

### 8.3 长上下文优化建议

- 长上下文模型必须用 YaRN/LongRoPE 等方案做位置编码外推，不能直接推理超长序列。
- 推理时使用 FlashAttention-3 和 PagedAttention 降低显存。
- 将关键信息放在上下文开头或结尾，缓解"中间丢失"。
- KV Cache 量化（如 INT4/FP8）可显著降低长上下文显存。

---

## 9. 参考文献

[1] S. Liu, H. Zhu, K. Li, et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL, 2024.

[2] S. Chen, Z. Zhu, X. Zhao, et al., *Extending Context Window of Large Language Models via Position Interpolation*, arXiv preprint, 2023.

[3] B. Peng, Q. Huang, D. Ahia, et al., *YaRN: Efficient Context Window Extension of Large Language Models*, arXiv preprint, 2023.

[4] Y. Ding, L. Zhang, C. Li, et al., *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens*, arXiv preprint, 2024.

[5] T. Dao, D. Y. Fu, S. Ermon, et al., *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, arXiv preprint, 2023.

[6] H. Liu, P. Abbeel, M. C. Gales, et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context*, ICLR, 2024.

[7] V. Karpukhin, B. Oğuz, S. Min, et al., *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP, 2020.

[8] P. Lewis, E. Perez, A. Piktus, et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS, 2020.

[9] L. Gao, X. Ma, J. Lin, et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, ACL, 2023.

[10] D. Edge, H. Trinh, N. Cheng, et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv preprint, 2024.
