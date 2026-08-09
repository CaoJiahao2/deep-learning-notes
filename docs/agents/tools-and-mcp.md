# 工具使用与 MCP 协议

> Function Calling 让 LLM 以结构化方式调用外部工具，MCP 让工具实现一次即可被所有 host 复用。

---

## 目录

1. [概述](#1)
2. [Function Calling 协议](#2)
3. [MCP (Model Context Protocol)](#3)
4. [工具选择、并行调用与错误处理](#4)
5. [工具生态](#5)
6. [实践建议](#6)
7. [参考文献](#7)

---

## 1 概述 {#1}

工具使用（Tool Use）是 Agent 与外部世界交互的桥梁。它让 LLM 不再局限于训练语料内的"参数化知识"，而能：

- 调用搜索引擎获取实时信息；
- 执行代码完成精确计算；
- 查询数据库、操作文件系统；
- 调用其他模型或第三方 API。

主流 LLM（GPT-4、Claude、Gemini、Qwen 等）都内置了工具调用能力，业界也通过 MCP 等开放协议把"工具"标准化为可复用资源。

---

## 2 Function Calling 协议 {#2}

### 2.1 基本流程

现代 LLM 提供 **Function Calling** 能力：开发者用 JSON Schema 描述可用工具，模型在需要时输出结构化 JSON 调用，而不是自由文本。运行时由 Agent 框架解析该 JSON、执行真实函数，再把结果喂回模型。

```text
┌────────┐    工具 Schema    ┌────────┐
│  开发   │ ───────────────▶ │  LLM   │
│  者    │                  │        │
└────────┘                  └────────┘
                                  │ JSON 调用
                                  ▼
                           ┌────────────┐
                           │   Runtime  │  执行函数
                           └────────────┘
                                  │
                                  ▼ 结果回灌给 LLM
```

### 2.2 JSON Schema 示例

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

### 2.3 工具描述的"自描述"原则

工具 description 不是简单的注释，而是 LLM 决策的关键依据。优质描述应包含：

- **用途**：什么时候用，解决什么问题；
- **输入**：参数含义、取值范围、约束；
- **输出**：返回结构与字段语义；
- **副作用**：是否修改外部状态（写文件、扣款）；
- **失败模式**：常见错误与替代方案。

---

## 3 MCP (Model Context Protocol) {#3}

### 3.1 什么是 MCP

**MCP** 是 Anthropic 于 2024 年提出的开放协议，旨在标准化 LLM 与外部工具/数据源的接入。它把"模型"与"工具"之间的连接抽象为统一的 client-server 协议，使任意支持 MCP 的 host（如 IDE、Agent 框架）都能复用同一套工具实现。

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

### 3.2 架构三件套

- **Host**：运行 LLM 的应用（如 Claude Desktop、IDE 插件、Agent 框架）。
- **Client**：Host 内部与 Server 通信的协议端，每个 Server 对应一个 Client。
- **Server**：暴露 Tools / Resources / Prompts 的独立进程，可对接数据库、文件系统、API 等。

MCP 提供三类原语：

| 原语 | 作用 | 示例 |
|------|------|------|
| Tools | 模型主动调用的函数 | 查天气、发邮件 |
| Resources | 模型只读的上下文资源 | 文档、文件 |
| Prompts | 预定义的提示模板 | 系统提示词模板 |

### 3.3 为什么需要 MCP

没有 MCP 时，每个 Agent 框架都要为同一工具（如 Slack）写一遍适配层。有了 MCP，工具实现一次，所有兼容 host 即可复用。这降低了：

- 工具开发者的适配成本（写一次，受众是所有 host）；
- 框架维护者的集成成本（按 MCP 规范接入新工具）。

### 3.4 MCP 与传统 Function Calling 的关系

Function Calling 是模型层的协议（模型怎么输出 JSON 调用），MCP 是传输层的协议（怎么把工具暴露给模型）。MCP 可以承载 Function Calling 的 payload，也可以承载 Resources/Prompts 等更丰富的语义。

---

## 4 工具选择、并行调用与错误处理 {#4}

### 4.1 工具选择

当工具数量多时，先做语义检索/路由筛选相关工具，再交给 LLM 决策，缓解上下文膨胀。常见做法：

- 给每个工具计算 embedding，按 query 相似度召回 top-k 个；
- 在系统 prompt 中只列出被召回的工具描述。

### 4.2 并行调用

若多步行动间无依赖，模型可一次性输出多个 tool call，框架并行执行以降低延迟：

```python
calls = LLM.parse_tool_calls(prompt)          # 可能返回多个调用
results = parallel_run([safe_call(c) for c in calls])  # 并行执行
for c, r in zip(calls, results):
    context.append(f"Observation: {r}")
```

并行调用是 Agent 框架的关键优化点，能把端到端延迟压缩到接近最慢的单步。

### 4.3 错误处理

捕获工具异常，把错误信息以 Observation 形式回灌给 LLM，让其自纠（重试、换工具、改参数）：

```python
for c, r in zip(calls, results):
    if isinstance(r, Exception):
        context.append(f"Observation: 工具 {c.name} 失败: {r}; 请改用其它方式。")
    else:
        context.append(f"Observation: {r}")
```

典型失败模式：

- 参数错误（type、enum 校验失败）；
- 网络/超时；
- 权限不足；
- 工具不存在（幻觉工具名）。

---

## 5 工具生态 {#5}

### 5.1 搜索与检索

- 通用 Web 搜索（Tavily、Serp、Bing Search）；
- 学术搜索（Semantic Scholar、PubMed）；
- 内部文档检索（见 [Agentic RAG](./agentic-rag.md)）。

### 5.2 代码与执行

- Python/JS REPL（本地或沙箱）；
- Docker 化执行（Code Interpreter 思路）；
- Jupyter 内核桥接。

### 5.3 文件与系统

- 读写文件（按权限分级）；
- Shell 命令（受限的 sandbox shell）；
- 浏览器自动化（Playwright、Selenium、Stagehand）。

### 5.4 第三方 API

- Slack、Notion、GitHub、Linear、Jira；
- 数据库（Postgres、BigQuery）；
- 邮件、日历、CRM。

### 5.5 GUI 操作

- 见 [GUI Agent 与 Computer Use](./gui-agent.md)。

---

## 6 实践建议 {#6}

1. **工具描述即文档**：description 直接决定 LLM 何时调用、写错参数；务必清晰、自描述、含失败模式。
2. **小工具优先**：工具粒度越细，越易组合，但调用次数增加；需在"细粒度"与"原子化"间权衡。
3. **失败可恢复**：把异常当 Observation 喂回，给模型自纠机会；不要在 framework 层静默吞错。
4. **超时与最大步数**：每个工具设置超时，整个 Agent 循环设置最大步数与总耗时，防止无限循环。
5. **人在回路审批**：危险操作（删库、发邮件、转账）必须有人工确认。
6. **全链路日志**：记录 Thought / Action / Observation / 时间戳，便于调试与复盘。
7. **路由与缓存**：相似 query 命中同一工具链时，直接复用历史结果，避免冗余调用。

---

## 7 参考文献 {#7}

[1] T. Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, NeurIPS, 2023. arXiv:2302.04761.

[2] OpenAI, *Function Calling and Other API Updates*, 2023. https://platform.openai.com.

[3] Anthropic, *Model Context Protocol (MCP) Specification*, 2024. https://modelcontextprotocol.io.

[4] Anthropic, *Introducing the Model Context Protocol*, 2024. https://www.anthropic.com/news/model-context-protocol.

[5] S. Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR, 2023. arXiv:2210.03629.

[6] P. Liang et al., *Holistic Evaluation of Language Models (ToolBench)*, 2023. arXiv:2211.09110.

[7] Microsoft Research, *TaskMatrix.AI: Completing Tasks by Connecting Foundation Models with the World*, 2023. arXiv:2303.16434.

[8] Google, *Function Calling with Gemini*, 2024. https://ai.google.dev.