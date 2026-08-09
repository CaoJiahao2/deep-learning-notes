# AI Agent 详解

> 从 LLM 到会规划、会调用工具、会记忆的智能体：Agent = LLM + 工具调用 + 记忆 + 循环。

---

## 目录

1. [概述](#1)
2. [Agent 核心循环](#2-agent)
3. [规划范式](#3)
4. [工具使用与函数调用 (Tool Use & Function Calling)](#4-tool-use-function-calling)
5. [检索增强生成 (RAG)](#5-rag)
6. [记忆机制 (Memory)](#6-memory)
7. [多智能体 (Multi-Agent)](#7-multi-agent)
8. [实践建议与挑战](#8)
9. [参考文献](#9)

---

## 1. 概述

### 1.1 什么是 AI Agent

传统 LLM 应用是"单轮问答"：用户输入一个 prompt，模型输出一段文本，交互即结束。这种方式难以应对需要多步推理、外部信息获取、与现实世界交互的复杂任务。**AI Agent（智能体）** 则将 LLM 视为"大脑"，赋予它感知环境、自主规划、调用工具、维护记忆并循环执行的能力，从而完成"端到端"的多步任务。

一个简洁的定义：

$$\text{Agent} = \text{LLM} + \text{Tools} + \text{Memory} + \text{Planning} + \text{Loop}$$

### 1.2 核心组件

| 组件 | 作用 | 典型实现 |
|------|------|----------|
| 大脑 (LLM) | 推理、决策、自然语言理解 | GPT-4, Claude, Qwen |
| 工具 (Tools) | 与外部世界交互 | 函数调用、API、搜索、代码执行 |
| 记忆 (Memory) | 跨轮次保留上下文 | 短期上下文窗口、向量记忆 |
| 规划 (Planning) | 拆解任务、选择行动 | ReAct、Plan-and-Execute、ToT |
| 循环 (Loop) | 持续迭代直到完成 | Perception-Action 循环 |

### 1.3 "单轮问答" vs "自主多步任务"

```text
单轮问答：    用户 → Prompt → LLM → 回答
Agent：      用户 → LLM → [规划 → 行动 → 观察] × N → 最终回答
```

单轮问答中模型的知识与能力被压缩在一次前向推理里；Agent 则把"推理-行动-观察"展开为一个循环，使其能动态获取新信息、修正错误、逐步逼近目标。

---

## 2. Agent 核心循环

### 2.1 Perception-Planning-Action-Observation

Agent 的运行可抽象为一个不断重复的循环：

```text
while not done:
    perception   ← 读取用户输入 / 环境反馈
    planning     ← 基于当前状态决定下一步
    action       ← 调用工具或生成回复
    observation  ← 收集行动结果，更新状态
```

### 2.2 ReAct 范式

**ReAct**（Reasoning + Acting）是当前最主流的 Agent 范式。它要求模型在每一步交替输出三段：

- **Thought**：当前推理过程（自然语言）
- **Action**：选择并调用某个工具
- **Observation**：工具返回的结果

随后 Observation 进入下一轮的上下文，继续 Thought，如此迭代。

**与 Chain-of-Thought (CoT) 的关系**：CoT 是纯推理，模型只在内部"想"，不与外部世界交互；ReAct 把推理与外部行动交织——Thought 指导 Action，Observation 修正 Thought，从而能够访问 CoT 无法获取的实时/外部信息。

一个 ReAct 文字示例：

```text
Question: 2024 年奥斯卡最佳影片导演的母校是哪所大学？

Thought 1: 我需要先查 2024 年奥斯卡最佳影片是哪部。
Action 1: Search["2024 Oscar best picture winner"]
Observation 1: 《奥本海默》获 2024 年奥斯卡最佳影片，导演为 Christopher Nolan。
Thought 2: 导演是 Christopher Nolan，现在需要查他的母校。
Action 2: Search["Christopher Nolan alma mater"]
Observation 2: Christopher Nolan 本科毕业于伦敦大学学院 (UCL)，主修英国文学。
Thought 3: 已得到答案，可以回答用户。
Action 3: Finish["伦敦大学学院 (UCL)"]
```

### 2.3 形式化伪代码

```text
context = [system_prompt, user_query]
loop:
    output = LLM(context)
    if output contains Action:
        tool, args = parse_action(output)
        obs = run_tool(tool, args)
        context.append(output)        # Thought + Action
        context.append(f"Observation: {obs}")
    else:
        return final_answer(output)
```

---

## 3. 规划范式

ReAct 是"边推理边行动"的隐式规划。对于复杂任务，显式的规划策略能带来更好的结构性与可调试性。主流范式如下。

### 3.1 主流规划策略

- **ReAct**：Thought-Action-Observation 交替，规划隐含在每一步 Thought 中。
- **Plan-and-Execute / BabyAGI**：先让模型生成一个完整计划（任务列表），再逐项执行；执行过程中可根据观察更新计划。
- **Reflection / Reflexion**：在每轮（尤其失败轮）之后，让模型自我反思，把"经验教训"写入记忆，指导下一次尝试。
- **Tree of Thoughts (ToT) / Graph of Thoughts (GoT)**：把推理组织成树/图，对每个状态评估并搜索（BFS/DFS/beam search），允许回溯。

### 3.2 对比表

| 范式 | 适用场景 | 复杂度 | 特点 |
|------|----------|--------|------|
| ReAct | 通用、工具调用密集 | 低 | 简单稳健，易于实现 |
| Plan-and-Execute | 长程、多步任务 | 中 | 计划与执行解耦，便于调试 |
| Reflexion | 试错型、可复盘任务 | 中 | 自我反思，跨轮改进 |
| ToT / GoT | 推理路径多、需搜索 | 高 | 树/图搜索，可回溯 |
| Plan-Solve | 数学/组合优化 | 中 | 先整体规划再求解 |

### 3.3 示例：Plan-and-Execute

```text
Plan:
  1. 检索相关文档
  2. 提取关键实体
  3. 计算统计量
  4. 生成报告

Execute:
  step 1 → observation → (可能修订 plan)
  step 2 → observation → ...
  step 4 → final answer
```

---

## 4. 工具使用与函数调用 (Tool Use & Function Calling)

### 4.1 Function Calling 协议

现代 LLM 提供 **Function Calling** 能力：开发者用 JSON Schema 描述可用工具，模型在需要时输出结构化 JSON 调用，而不是自由文本。运行时由 Agent 框架解析该 JSON、执行真实函数，再把结果喂回模型。

一个工具的 JSON Schema 描述示例：

```json
{
  "name": "get_weather",
  "description": "获取指定城市的当前天气",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "城市名称，如 Beijing"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["city"]
  }
}
```

模型输出的一次调用示例：

```json
{
  "tool": "get_weather",
  "arguments": {"city": "Beijing", "unit": "celsius"}
}
```

### 4.2 MCP (Model Context Protocol)

**MCP** 是 Anthropic 于 2024 年提出的开放协议，旨在标准化 LLM 与外部工具/数据源的接入。它把"模型"与"工具"之间的连接抽象为统一的 client-server 协议，使任意支持 MCP 的 host（如 IDE、Agent 框架）都能复用同一套工具实现。

架构三件套：

- **Host**：运行 LLM 的应用（如 Claude Desktop、IDE 插件、Agent 框架）。
- **Client**：Host 内部与 Server 通信的协议端，每个 Server 对应一个 Client。
- **Server**：暴露 Tools / Resources / Prompts 的独立进程，可对接数据库、文件系统、API 等。

```text
+----------+      MCP       +----------+
|   Host   |  ←───────────→  |  Client  |
| (LLM App)|                  +----------+
+----------+                       |
                                   | MCP (JSON-RPC over stdio/HTTP/SSE)
                                   v
                            +-------------+
                            | MCP Server  |  → Tools / Resources / Prompts
                            +-------------+
```

MCP 的价值在于：工具实现一次，即可被所有 MCP 兼容的 Host 复用，避免每个 Agent 框架重复封装同一组 API。

### 4.3 工具选择、并行调用与错误处理

- **工具选择**：当工具数量多时，先做语义检索/路由筛选相关工具，再交给 LLM 决策，缓解上下文膨胀。
- **并行调用**：若多步行动间无依赖，模型可一次性输出多个 tool call，框架并行执行以降低延迟。
- **错误处理**：捕获工具异常，把错误信息以 Observation 形式回灌给 LLM，让其自纠（重试、换工具、改参数）。

```python
# 并行调用与错误处理的简化伪代码
calls = LLM.parse_tool_calls(prompt)          # 可能返回多个调用
results = parallel_run([safe_call(c) for c in calls])  # 并行执行
for c, r in zip(calls, results):
    if isinstance(r, Exception):
        context.append(f"Observation: 工具 {c.name} 失败: {r}; 请改用其它方式。")
    else:
        context.append(f"Observation: {r}")
```

---

## 5. 检索增强生成 (RAG)

### 5.1 为什么需要 RAG

LLM 的参数化知识存在三大局限：

- **知识时效性**：训练截止后的事件无法获知。
- **私有数据**：企业内部文档、个人笔记不在训练语料中。
- **幻觉 (Hallucination)**：对不确定问题，模型倾向于"编造"看似合理的事实。

**RAG**（Retrieval-Augmented Generation）通过在生成前从外部知识库检索相关片段并拼入上下文，让模型"看着资料回答"，显著缓解上述问题，并支持可追溯引用。

### 5.2 RAG 流水线

```text
[离线建库]
  文档 → Chunking 切分 → Embedding → 写入向量库

[在线检索生成]
  Query → Embedding → 向量检索 → Reranking → 拼接上下文 → LLM 生成
```

### 5.3 Embedding 模型

将文本映射为稠密向量，用于相似度检索。常见模型：

- **OpenAI text-embedding-3-small / large**：通用、多语言。
- **BGE 系列**（BAAI）：开源中文表现优秀，如 `bge-large-zh`、`bge-m3`（多语言/多功能）。
- **E5 / GTE / Jina** 等。

设文本 $t$ 的嵌入向量为 $\mathbf{e}(t) \in \mathbb{R}^d$，$d$ 为嵌入维度（如 768、1024、3072）。

### 5.4 向量数据库与相似度检索

主流向量数据库：

| 数据库 | 特点 |
|--------|------|
| FAISS | Meta 开源库，单机高效，适合原型与中等规模 |
| Milvus | 分布式、可扩展，适合生产环境 |
| Qdrant | Rust 实现，过滤能力强，云原生 |
| Chroma | 轻量，常用于原型 |
| pgvector | PostgreSQL 扩展，便于与业务库融合 |

最常用的相似度为 **cosine 相似度**：

$$\cos(\mathbf{a},\mathbf{b})=\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$$

其中 $\mathbf{a}\cdot\mathbf{b}$ 为向量内积，$\|\mathbf{a}\|$ 为向量 $L_2$ 范数。部分场景也用内积（向量已归一化时与 cosine 等价）或欧氏距离。

### 5.5 检索策略

- **稠密检索 (Dense)**：基于 Embedding 的语义相似，擅长捕捉同义/跨语言语义。
- **稀疏检索 (Sparse / BM25)**：基于词项匹配，对专有名词、ID、代码符号更精确。
- **Hybrid Search**：融合 Dense 与 Sparse（加权或 RRF 融合），兼顾语义与字面匹配。
- **Reranking**：用 cross-encoder（如 `bge-reranker`、Cohere Rerank）对召回的 top-K 候选重新打分，得到更精确的最终排序。cross-encoder 比 bi-encoder 慢但精度高，故只用于小规模重排。

### 5.6 Chunking 策略

切分质量直接影响检索效果：

| 策略 | 说明 |
|------|------|
| 固定大小 | 按字符/token 滑窗切分，简单但可能截断语义 |
| 按语义 | 按段落/标题/句子边界切分，保留语义完整性 |
| 父子分块 (Parent-Child) | 检索小块、返回大块，兼顾匹配精度与上下文完整 |
| 递归切分 | 层级化切分，适配不同粒度 |

### 5.7 评估指标

参考 **RAGAS** 等评测框架，常用指标：

- **Faithfulness（忠实度）**：回答是否仅基于检索上下文，无捏造。
- **Answer Relevancy（回答相关性）**：回答与问题的相关程度。
- **Context Recall / Precision**：是否召回了所有应召的上下文 / 召回中有多少是相关的。

### 5.8 进阶方向

- **GraphRAG**：把知识组织为图（实体-关系），检索时跨节点推理，适合多跳问答。
- **Self-RAG**：模型自判断是否需要检索、检索结果是否可信，并自我反思。
- **Agentic RAG**：把检索本身作为一个工具由 Agent 调用，Agent 自主决定检索时机、查询改写与多轮检索。

---

## 6. 记忆机制 (Memory)

### 6.1 短期 vs 长期记忆

- **短期记忆 (Short-term)**：即当前对话的上下文窗口（in-context）。容量受 token 上限约束，随对话增长会被截断或溢出。
- **长期记忆 (Long-term)**：跨会话持久化，通常以向量库/摘要形式存储，按需检索注入上下文。

### 6.2 记忆写入与检索

```text
写入：把关键事实/对话摘要 → Embedding → 写入长期记忆库
检索：当前问题 → Embedding → 相似度检索 → 取回相关历史 → 注入 prompt
```

常见长期记忆形态：

| 形态 | 说明 |
|------|------|
| 向量记忆 | 原文/摘要向量化，语义检索 |
| 摘要记忆 | 对历史滚动摘要，节省 token |
| 实体记忆 | 以结构化方式记录实体属性与关系 |
| 事件记忆 | 按时间序记录发生过的事件 |

### 6.3 MemGPT：分页式记忆管理

**MemGPT** 借鉴操作系统的虚拟内存与分页机制：把 LLM 上下文窗口视为主存，外部向量库视为磁盘，由 LLM 自主决定何时把哪些内容"换入"主存或"换出"到磁盘，从而在有限窗口内管理超长上下文。

```text
+--------------------------------------------------+
|  Main Context (LLM 上下文窗口 / "主存")          |
|  ┌────────────┐ ┌────────────┐ ┌──────────────┐ |
|  | System     | | Working    | | Recalled     | |
|  | Prompt     | | Memory     | | (从磁盘换入) | |
|  └────────────┘ └────────────┘ └──────────────┘ |
+--------------------------------------------------+
                ▲  recall (page-in)
                ▼  archive  (page-out)
+--------------------------------------------------+
|  External Memory ("磁盘": 向量库 / 摘要库)        |
+--------------------------------------------------+
```

---

## 7. 多智能体 (Multi-Agent)

### 7.1 为什么需要多智能体

单 Agent 在复杂任务中易出现职责混淆、上下文膨胀、推理漂移。多智能体把任务拆分给不同角色的 Agent，每个 Agent 专注一个子目标，通过通信协作完成整体任务。

### 7.2 角色分工与通信

常见角色：**Planner**（规划）、**Coder**（实现）、**Reviewer**（审查）、**Executor**（执行/运行）、**Critic**（批判）。通信协议通常基于自然语言消息传递，可叠加结构化字段（如状态、产物引用）。

### 7.3 框架范式对比

| 框架 | 范式 | 特点 |
|------|------|------|
| AutoGen | 对话式 | 多 Agent 通过消息对话协作，支持人在回路 |
| CrewAI | 角色 + 任务流 | 预定义 Crew/Role/Task，强调流程编排 |
| LangGraph | 显式状态图 | 节点 = Agent/函数，边 = 状态转移，可循环可分支 |

LangGraph 风格的状态图示意：

```text
            +--------+
            | Start  |
            +---+----+
                |
                v
        +--------------+     fail     +----------+
        |   Planner    +------------->| Reflector|
        +------+-------+             +-----+----+
               |                           |
               | ok                        | revised plan
               v                           |
        +--------------+                    |
        |    Coder     |                    |
        +------+-------+                    |
               |                            |
               v                            |
        +--------------+                    |
        |   Reviewer   +--------------------+
        +------+-------+  needs revision
               |
               | pass
               v
            +--------+
            |  Done  |
            +--------+
```

### 7.4 协作伪代码示例

```python
from langgraph.graph import StateGraph

def planner(state):
    state["plan"] = LLM.plan(state["task"])
    return state

def coder(state):
    state["code"] = LLM.code(state["plan"])
    return state

def reviewer(state):
    state["ok"] = LLM.review(state["code"], state["plan"])
    return state

def route(state):
    return "coder" if not state["ok"] else "done"

g = StateGraph(dict)
g.add_node("planner", planner)
g.add_node("coder", coder)
g.add_node("reviewer", reviewer)
g.set_entry_point("planner")
g.add_edge("planner", "coder")
g.add_edge("coder", "reviewer")
g.add_conditional_edges("reviewer", route, {"coder": "coder", "done": "__end__"})
app = g.compile()
```

---

## 8. 实践建议与挑战

### 8.1 常见挑战

| 挑战 | 表现 | 对策 |
|------|------|------|
| 幻觉与工具误用 | 调用不存在的工具、编造参数 | 严格 JSON Schema 校验、工具描述清晰、错误回灌自纠 |
| 长程规划漂移 | 多步后偏离原目标 | 引入显式计划、阶段复盘、Reflexion |
| 上下文窗口限制 | 长对话/大量工具结果溢出 | 摘要压缩、长期记忆、MemGPT 式分页 |
| 成本与延迟 | 多轮 + 多工具调用昂贵 | 缓存、并行调用、小模型路由、分级模型 |
| 评测困难 | 端到端任务难打分 | 使用 Agent 专用基准 + 人工评估 |

### 8.2 Agent 评测基准

- **AgentBench**：多场景（OS、DB、Web、KG）Agent 能力评测。
- **WebArena**：真实网站环境下的 web 交互任务。
- **SWE-bench**：从真实 GitHub 仓库 issue 中构造的软件工程任务，评估端到端修 bug 能力。
- **GAIA**：通用 Assistant 级多步问答基准。
- **ToolBench / API-Bank**：聚焦工具调用能力。

### 8.3 工程建议

- 工具描述要"自描述"：在 description 里写清用途、输入输出、副作用与失败模式。
- 默认失败可恢复：把异常当 Observation 喂回，给模型自纠机会。
- 设置最大迭代次数与超时，防止无限循环。
- 关键步骤加人在回路（human-in-the-loop）审批。
- 全链路日志（Thought/Action/Observation）便于调试与复盘。

---

## 9. 参考文献

[1] S. Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR, 2023. arXiv:2210.03629.

[2] N. Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS, 2023. arXiv:2303.11366.

[3] S. Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, NeurIPS, 2023. arXiv:2305.10601.

[4] T. Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, NeurIPS, 2023. arXiv:2302.04761.

[5] P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS, 2020. arXiv:2005.11401.

[6] S. Robertson and H. Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, Foundations and Trends in IR, 2009.

[7] J. Johnson et al., *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, ICML, 2021. arXiv:2103.00020.

[8] S. Xiao et al., *C-Pack: Packed Resources For General Chinese Embeddings (BGE)*, 2023. arXiv:2309.07597.

[9] Anthropic, *Model Context Protocol (MCP) Specification*, 2024. https://modelcontextprotocol.io.

[10] Q. Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*, 2023. arXiv:2308.08155.

[11] LangChain, *LangGraph: Building Stateful Multi-Actor Applications*, 2024. https://github.com/langchain-ai/langgraph.

[12] C. Packer et al., *MemGPT: Towards LLMs as Operating Systems*, 2023. arXiv:2310.08560.

[13] X. Liu et al., *AgentBench: Evaluating LLMs as Agents*, ICLR, 2024. arXiv:2308.03688.

[14] S. Zhou et al., *WebArena: A Realistic Web Environment for Building Autonomous Agents*, ICLR, 2024. arXiv:2307.13854.

[15] C. Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, ICLR, 2024. arXiv:2310.06770.

[16] S. Edalat et al., *GraphRAG: Unlocking LLM's Discovery Capabilities through Knowledge Graphs*, 2024. arXiv:2404.16130.
