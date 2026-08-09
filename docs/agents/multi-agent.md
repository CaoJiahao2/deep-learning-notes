# 多智能体协作

> 把复杂任务拆给多个角色 Agent，通过通信协作完成整体目标。

---

## 目录

1. [概述](#1)
2. [角色分工与通信](#2)
3. [主流框架对比](#3)
4. [状态图范式：LangGraph](#4)
5. [协作模式](#5)
6. [失败模式与对策](#6)
7. [参考文献](#7)

---

## 1 概述 {#1}

单 Agent 在复杂任务中易出现职责混淆、上下文膨胀、推理漂移。**多智能体（Multi-Agent）** 把任务拆分给不同角色的 Agent，每个 Agent 专注一个子目标，通过通信协作完成整体任务。

### 1.1 为什么需要多智能体

| 问题 | 单 Agent 表现 | 多 Agent 表现 |
|------|--------------|--------------|
| 职责混淆 | 一个 prompt 同时做规划 + 实现 + 审查 | 每个 Agent 单一职责 |
| 上下文膨胀 | 全部历史塞给一个 Agent | 各自维护局部上下文 |
| 推理漂移 | 长程任务易跑偏 | 阶段性角色切换锚定目标 |
| 评测粒度 | 只能端到端打分 | 可分阶段分角色评测 |

### 1.2 不是万能解

多 Agent 不一定优于单 Agent。它的代价：

- 通信开销：消息传递与状态同步；
- 错误传播：一个 Agent 错了会污染下游；
- 复杂度提升：拓扑设计、循环终止、冲突解决。

简单任务用单 Agent + 好工具就够，多 Agent 适合"明确需要分工"的场景。

---

## 2 角色分工与通信 {#2}

### 2.1 常见角色

| 角色 | 职责 |
|------|------|
| Planner | 拆解任务、生成计划 |
| Coder | 实现代码 |
| Reviewer | 审查产物、给出意见 |
| Executor | 执行/运行代码、调用外部工具 |
| Critic | 批判与反思 |
| Researcher | 信息检索与综合 |
| Summarizer | 摘要与整合 |

### 2.2 通信协议

通信通常基于自然语言消息传递，可叠加结构化字段（如状态、产物引用）。三种主流拓扑：

```text
1. 星型：        Coordinator → {A1, A2, A3}
                          ↑       ↓
                          └───────┘ (汇总)

2. 流水线：      A1 → A2 → A3 → Output

3. 群聊/议会：   A1 ↔ A2 ↔ A3 ↔ A1  (相互通信)
```

星型适合"统一调度"，流水线适合"线性加工"，群聊适合"博弈协商"。

---

## 3 主流框架对比 {#3}

| 框架 | 范式 | 特点 | 适用场景 |
|------|------|------|----------|
| AutoGen | 对话式 | 多 Agent 通过消息对话协作，支持人在回路 | 对话型协作、研究 |
| CrewAI | 角色 + 任务流 | 预定义 Crew/Role/Task，强调流程编排 | 业务流程自动化 |
| LangGraph | 显式状态图 | 节点 = Agent/函数，边 = 状态转移，可循环可分支 | 复杂工作流、生产化 |
| OpenAI Swarm | 极简 handoff | 轻量级 Agent 切换，无状态中心 | 教学、原型 |
| MetaGPT | 软件工程模拟 | 模拟软件公司角色（产品/架构/工程/QA） | 软件开发 |

### 3.1 AutoGen 风格示例

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config={...})
user_proxy = UserProxyAgent("user", code_execution_config={"work_dir": "coding"})

user_proxy.initiate_chat(
    assistant,
    message="写一个 Python 脚本，读取 CSV 并画图"
)
```

### 3.2 CrewAI 风格示例

```python
from crewai import Agent, Crew, Task

researcher = Agent(role="研究员", goal="检索资料", backstory="...")
writer     = Agent(role="作家",   goal="撰写报告", backstory="...")

t1 = Task(description="检索 AI Agent 最新进展", agent=researcher)
t2 = Task(description="撰写综述",            agent=writer, context=[t1])

crew = Crew(agents=[researcher, writer], tasks=[t1, t2])
crew.kickoff()
```

---

## 4 状态图范式：LangGraph {#4}

### 4.1 为什么是状态图

复杂 Agent 不只是"对话"，而是"流程"。把节点定义为 Agent/函数，把边定义为状态转移条件，就能用图显式描述流程：

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

### 4.2 代码示例

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

### 4.3 优势

- **可观测**：每一步的状态都可记录与回放；
- **可控**：用条件边实现循环、分支、并行；
- **可持久化**：内置 checkpoint、人在回路、时间旅行；
- **可生产化**：可挂载监控、限流、审计。

---

## 5 协作模式 {#5}

### 5.1 计划-执行模式

Planner 生成计划 → Executor 逐步执行 → 反馈给 Planner 更新。

### 5.2 辩论模式

两个 Agent 持不同观点辩论，Judge Agent 仲裁。适合需要多视角权衡的决策。

### 5.3 角色扮演模式

每个 Agent 扮演一个角色（产品、架构、QA），按软件工程流程协作。MetaGPT 是代表。

### 5.4 嵌套 Agent

主 Agent 调起子 Agent 解决子问题，子 Agent 完成后把结果返回。适合"分层任务分解"。

---

## 6 失败模式与对策 {#6}

| 失败模式 | 表现 | 对策 |
|----------|------|------|
| 通信死循环 | Agent A 与 B 反复推诿不收敛 | 设置最大轮数、强制裁决 |
| 错误传播 | 一个 Agent 错了，下游全错 | 引入 Critic / Reviewer、独立校验 |
| 上下文冲突 | 两个 Agent 对同一事实持不同看法 | 引入共享 memory / 单一信息源 |
| 角色漂移 | Planner 开始写代码、Coder 开始规划 | 严格 role prompt、权限控制 |
| 资源争用 | 多个 Agent 同时改同一文件 | 引入 Coordinator、锁机制 |

---

## 7 参考文献 {#7}

[1] Q. Wu et al., *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*, 2023. arXiv:2308.08155.

[2] CrewAI, *CrewAI: Role-Based Collaborative Agent Framework*, 2024. https://github.com/joaomdmoura/crewai.

[3] LangChain, *LangGraph: Building Stateful Multi-Actor Applications*, 2024. https://github.com/langchain-ai/langgraph.

[4] OpenAI, *Swarm: Lightweight Multi-Agent Orchestration*, 2024. https://github.com/openai/swarm.

[5] H. Zhao et al., *MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework*, ICLR, 2024. arXiv:2308.00352.

[6] S. Hong et al., *MetaGPT: Merging Large Language Models for Multi-Agent Collaboration*, 2023.

[7] D. Talebirad et al., *Multi-Agent Systems: A Survey of Foundations, Frameworks, and Applications*, 2023. arXiv:2310.14058.

[8] J. Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS, 2022. arXiv:2201.11903.