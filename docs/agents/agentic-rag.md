# Agentic RAG：让 Agent 自主决定何时检索、如何检索

> 把检索本身作为一个工具，由 Agent 调用，自主决定检索时机、查询改写与多轮检索。

---

## 目录

1. [概述](#1)
2. [RAG 基础回顾](#2)
3. [从被动 RAG 到 Agentic RAG](#3)
4. [关键能力：查询改写与多轮检索](#4)
5. [Self-RAG 与反思式检索](#5)
6. [GraphRAG 与结构化知识](#6)
7. [实践模式](#7)
8. [参考文献](#8)

---

## 1 概述 {#1}

传统 RAG 是一条固定的流水线：用户提问 → 检索 → 重排 → 生成答案。在 Agentic RAG 中，Agent 把检索视为**一个可调用的工具**，可自行决定：

- 是否需要检索（还是直接回答）；
- 用什么 query 去检索（改写、分解、子查询）；
- 检索到结果后是否需要再检索一次（多轮迭代）；
- 检索结果是否相关、是否需要换工具或换数据源。

这种"以 Agent 为中心的检索"让 RAG 具备了适应复杂问题的灵活性。

---

## 2 RAG 基础回顾 {#2}

RAG（Retrieval-Augmented Generation）通过在生成前从外部知识库检索相关片段并拼入上下文，让模型"看着资料回答"。

```text
[离线建库]
  文档 → Chunking 切分 → Embedding → 写入向量库

[在线检索生成]
  Query → Embedding → 向量检索 → Reranking → 拼接上下文 → LLM 生成
```

主流相似度为 **cosine 相似度**：

$$\cos(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$$

其中 $\mathbf{a}\cdot\mathbf{b}$ 为向量内积，$\|\mathbf{a}\|$ 为向量 $L_2$ 范数。

常用向量数据库：FAISS、Milvus、Qdrant、Chroma、pgvector。

常用检索策略：Dense（向量）、BM25（稀疏）、Hybrid（融合）、Reranking（cross-encoder 精排）。

---

## 3 从被动 RAG 到 Agentic RAG {#3}

### 3.1 被动 RAG

被动 RAG 把"检索"与"生成"耦合在一个固定流程里：

```text
Query → [检索 → 生成]  ← 一次成型
```

它在简单问答场景下足够，但遇到复杂问题（多跳、子问题、上下文依赖）容易失败。

### 3.2 Agentic RAG

Agentic RAG 把检索视为 Agent 的工具之一：

```text
Question
  │
  ▼
┌──────────────────────────────────┐
│  Agent 循环                      │
│  • 是否需要检索？                  │
│  • 检索 query 怎么写？             │
│  • 检索结果够吗？要不要再查一次？   │
│  • 是否需要换工具/换源？           │
│  • 是否需要拆分多个子问题？        │
└──────────────────────────────────┘
  │
  ▼
Final Answer
```

### 3.3 优势

| 维度 | 被动 RAG | Agentic RAG |
|------|----------|-------------|
| 多跳问题 | 常失败 | 可分步检索、合并 |
| 子问题分解 | 不支持 | 自然支持 |
| 检索质量评估 | 不支持 | Agent 自评后重查 |
| 工具灵活性 | 单一检索 | 检索 + 搜索 + DB + API |

---

## 4 关键能力：查询改写与多轮检索 {#4}

### 4.1 查询改写 (Query Rewriting)

直接把用户原始 query 拿去检索，常常效果不佳。Agent 可以在检索前改写 query：

- **缩写与代词消解**：把"它""该公司"等替换为实体；
- **关键词提取**：去掉停用词，提炼核心词；
- **Query 扩展**：用 LLM 生成同义、近义、改述，扩大召回；
- **背景补全**：把对话上文的关键事实补进 query。

### 4.2 子问题分解 (Sub-question Decomposition)

复杂问题可拆为多个子问题，分别检索后合并：

```text
Question: 2024 年 Q2 公司营收同比增长多少？
  ├── 子问题 1: 2024 年 Q2 营收
  ├── 子问题 2: 2023 年 Q2 营收
  └── 子问题 3: 计算同比增长率
```

### 4.3 多轮检索

Agent 可以在第一轮检索后判断：

- "结果不够" → 改 query 再查；
- "结果矛盾" → 引入新工具比对；
- "结果够用" → 进入生成。

这相当于把"检索"变成了一个**可被反思的过程**。

---

## 5 Self-RAG 与反思式检索 {#5}

### 5.1 Self-RAG

**Self-RAG** 让模型自判断：

- 是否需要检索；
- 检索结果是否相关；
- 检索结果是否被忠实使用（防止幻觉）。

其训练目标是同时预测答案 token 与"反思 token"（Retrieve / IsRel / IsSup / IsUse）。

### 5.2 反思式检索 (Reflective RAG)

每次检索后插入一次"反思"步骤：

```text
检索 → 评估相关性 → 不相关则改 query → 再检索 → … → 相关则生成
```

### 5.3 FLARE

**FLARE**（Forward-Looking Active REtrieval）边生成边决定是否需要检索：当模型对下一个 token 不确定时触发检索，把检索结果注入生成。

---

## 6 GraphRAG 与结构化知识 {#6}

### 6.1 思路

传统 RAG 按"块-相似度"检索，丢失了文档中的实体关系。**GraphRAG** 把知识组织为图（实体-关系），检索时跨节点推理，适合多跳问答。

```text
文档 → 实体抽取 + 关系抽取 → 知识图谱
Query → 子图检索 → 跨节点推理 → 生成
```

### 6.2 优势

- 多跳关系：图天然支持；
- 社区发现：可在图上做社区检测，生成高层摘要；
- 跨文档整合：实体对齐后可串联多文档。

### 6.3 局限

- 实体抽取质量不稳定；
- 图构建成本高；
- 适合"全局性问题"（总结、趋势），未必优于向量检索用于"局部事实"。

---

## 7 实践模式 {#7}

### 7.1 检索作为工具的实现

```python
@tool("retrieve_docs")
def retrieve_docs(query: str, top_k: int = 5) -> list:
    """从内部文档库检索与 query 相关的片段。"""
    return vector_store.similarity_search(query, k=top_k)
```

Agent 决定何时调用此工具，传什么 query，并对结果做评估与重试。

### 7.2 多源路由

Agent 同时持有多个检索工具（内部文档 / Web / 学术 / DB），按 query 类型路由：

```text
"查询股价"      → 调用 stock_api
"查询内部规范"  → 调用 vector_db
"查询最新论文"  → 调用 semantic_scholar
```

### 7.3 与 ReAct 配合

Agentic RAG 通常在 ReAct 循环内执行：

```text
Thought: 用户问的是公司 Q2 营收，需要先检索财报
Action: retrieve_docs("2024 Q2 营收报告")
Observation: [片段 A、B、C]
Thought: 还差 2023 年同期数据，要再查一次
Action: retrieve_docs("2023 Q2 营收")
Observation: [片段 D、E]
Thought: 已有足够数据，生成答案
Action: Finish[计算并回答同比增长率]
```

### 7.4 性能与成本权衡

- 每次检索都有延迟与成本，多轮检索会显著放大；
- 建议设置最大检索轮数、超时与缓存；
- 可对简单问题跳过 Agentic 路径，仅在难问题上启用。

---

## 8 参考文献 {#8}

[1] P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS, 2020. arXiv:2005.11401.

[2] A. Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, ICLR, 2024. arXiv:2310.11511.

[3] Z. Jiang et al., *Active Retrieval Augmented Generation*, EMNLP, 2023. arXiv:2305.06983.

[4] S. Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR, 2023. arXiv:2210.03629.

[5] Microsoft, *GraphRAG: Unlocking LLM's Discovery Capabilities through Knowledge Graphs*, 2024. arXiv:2404.16130.

[6] S. Robertson and H. Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, Foundations and Trends in IR, 2009.

[7] S. Xiao et al., *C-Pack: Packed Resources For General Chinese Embeddings (BGE)*, 2023. arXiv:2309.07597.

[8] J. Johnson et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML, 2021. arXiv:2103.00020.