# AI Agent 基础与核心循环

> 从 LLM 到会规划、会调用工具、会记忆的智能体：Agent = LLM + 工具调用 + 记忆 + 循环。

---

## 目录

1. [概述](#1)
2. [核心循环与公式化](#2)
3. [ReAct 范式](#3)
4. [规划范式总览](#4)
5. [Agent 工程决策点](#5)
6. [失败模式与对策](#6)
7. [Agent 评测](#7)
8. [参考文献](#8)

---

## 1 概述 {#1}

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

## 2 核心循环与公式化 {#2}

### 2.1 Perception-Planning-Action-Observation

Agent 的运行可抽象为一个不断重复的循环：

```text
while not done:
    perception   ← 读取用户输入 / 环境反馈
    planning     ← 基于当前状态决定下一步
    action       ← 调用工具或生成回复
    observation  ← 收集行动结果，更新状态
```

形式化为：在每个时刻 $t$，策略 $\pi_\theta$ 接收观测 $o_t$，输出动作 $a_t$：

$$\pi_\theta(a_t \mid o_1, a_1, \dots, o_t)$$

其中观测包含对话历史、工具结果与记忆检索的内容。

### 2.2 形式化伪代码

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

其中关键决策由 LLM 完成，工具执行与状态维护由 runtime 负责。

### 2.3 状态机视角：感知—规划—行动—反思

把循环展开成显式状态机，每次迭代是一次状态转移，**反思（reflect）**是把失败转化为信息的元认知转移：

```text
          ┌─────────────────────────────────────────────┐
          ▼                                             │（未完成 / 发现错误）
┌───────────────┐   需要外部信息    ┌──────────────┐    │
│ PERCEIVE 感知  │ ───────────────► │ PLAN 规划     │    │
└───────────────┘                  └──────────────┘    │
      ▲                                  │             │
      │ observation                      ▼ action      │
┌───────────────┐  结果不可靠/偏离目标  ┌────────────┐  │
│ REFLECT 反思   │ ◄────────────────── │ ACT 行动    │──┘
└───────────────┘                    └────────────┘
      │ 结论经验证可靠
      ▼
   Finish（吸收态）
```

- **状态**不只是对话文本：$s_t$ = 上下文内容 + 任务进度 + 可用工具 + 剩余预算；
- **转移**由 LLM 输出驱动（输出 Action 即选择一条出边），runtime 执行工具、写回 Observation；
- **吸收态**（终止）= 显式 Finish 且结果通过校验，与 RL 中 episode 终止同构——整个循环可视为把策略 $\pi_\theta(a_t \mid o_1, \dots, o_t)$ 沿轨迹展开；
- 反思态可选但关键：没有它，Agent 只会"惯性前进"；有了它，失败轨迹成为下一轮的输入（Reflexion 的核心机制，见 §4.2）。

### 2.4 上下文工程（context engineering）

Agent 的上下文是运行时唯一的工作记忆，其组织质量直接决定上限。一个可复用的 system prompt 骨架（自上而下）：

```text
[角色与目标]  你是……，目标是……
[硬性约束]    禁止……；遇到 X 必须先 Y
[工具清单]    名称、参数 schema、适用时机、失败时的处理方式
[输出协议]    Thought / Action / Finish 的格式约定与示例
[环境状态]    当前时间、工作目录、任务清单、步数与预算（动态注入）
```

三个最影响成败的实践：

- **状态注入**：把动态状态（文件树、todo 清单、剩余步数）结构化地渲染进上下文，而不是让模型从几十轮对话里"考古"；任务目标与当前计划是全局信息，必须始终可见；
- **工具结果压缩**：原始返回（网页正文、日志、目录列表）先截断、摘要、抽取关键字段再入上下文——一次数万 token 的工具输出足以污染整个窗口的注意力；
- **预算分配**：为历史轨迹、工具结果、中间规划各设上限，超限做滚动摘要（rolling summary），永远保底保留：任务目标、当前计划、最近 $k$ 步。

工具协议与工具结果处理的工程细节见 [`agents/tools-and-mcp.md`](tools-and-mcp.md)。

---

## 3 ReAct 范式 {#3}

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

---

## 4 规划范式总览 {#4}

ReAct 是"边推理边行动"的隐式规划。对于复杂任务，显式的规划策略能带来更好的结构性与可调试性。主流范式如下。

| 范式 | 适用场景 | 复杂度 | 特点 |
|------|----------|--------|------|
| ReAct | 通用、工具调用密集 | 低 | 简单稳健，易于实现 |
| Plan-and-Execute / BabyAGI | 长程、多步任务 | 中 | 计划与执行解耦，便于调试 |
| Reflection / Reflexion | 试错型、可复盘任务 | 中 | 自我反思，跨轮改进 |
| ToT / GoT | 推理路径多、需搜索 | 高 | 树/图搜索，可回溯 |
| Plan-Solve | 数学/组合优化 | 中 | 先整体规划再求解 |

**Plan-and-Execute 示例**：

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

### 4.1 ReAct vs Plan-and-Execute：何时先规划

| 维度 | ReAct（边走边想） | Plan-and-Execute（先规划后执行） |
|------|------------------|--------------------------------|
| 规划时机 | 每步隐式规划一步 | 先产出显式计划，再逐项执行 |
| 上下文成本 | 每步携带全部历史 | 执行器只需当前步骤，历史可压缩 |
| 错误恢复 | 天然渐进修正 | 需要 replan 触发机制 |
| LLM 调用 | 多（每步一次强模型） | 少；执行可用小模型 |
| 适用 | 环境不可预测、探索型 | 结构清晰、可分解的长程任务 |

经验法则：任务能事先分解且步骤间弱依赖 → Plan-and-Execute（规划用强模型、执行用小模型还能省成本）；环境高度不可预测、每步依赖上一步观察 → ReAct。生产系统常混合使用：先 plan，逐步以 ReAct 方式执行，偏差超过阈值触发 replan。

### 4.2 Reflexion：语言化的强化学习

失败后不更新参数，而是让模型自我复盘，把"为什么失败"写成简短教训存入 episodic memory，下次尝试时注入上下文：

```text
attempt k ──失败──► self-reflect（生成教训）──► 存入记忆
attempt k+1（上下文携带历史教训）──► 更少重复同类错误 ──► …
```

与 RL 的对应：轨迹 → 自我反思 → 下一次尝试，相当于把 credit assignment 从梯度空间搬到语言空间（verbal reinforcement learning）。适用前提：任务有明确成败信号、允许多次尝试（代码单测是否通过、检索问答对错）。

### 4.3 树搜索：ToT 与 MCTS

ReAct/Reflexion 都在单条路径上贪心前进；ToT 把"中间思维状态"显式化为树节点，支持分支与回溯：

- **ToT**：对每个状态提出多个候选下一步 → LLM 为各状态打分（value）→ BFS/beam 搜索，坏分支可剪枝、可回溯，适合解空间大、单步可评估的推理题；
- **MCTS**：选择（UCB 平衡探索/利用）→ 扩展 → rollout 快速估计价值 → 回传更新，适合步数长、单步评估昂贵的决策（AlphaGo/AlphaZero 即此范式 + 神经网络价值头）；
- **代价意识**：树上每个节点的评估与模拟都是 LLM 调用，总成本与节点数成正比。只有"一步走错不可逆的代价高、且有调用预算"时才值得树搜索，否则 ReAct + 反思通常性价比更高。

---

## 5 Agent 工程决策点 {#5}

Agent 从 demo 到可靠系统，需要在几个关键点做显式决策。

### 5.1 模型选型

- 角色分层：planner / 推理用强模型（长上下文、指令遵循、工具调用准确率），executor / 摘要 / 路由用小模型；
- 硬指标优先于体感：函数调用 schema 遵从率、输出协议稳定性、长上下文中部信息利用率（lost in the middle）、成本与延迟；
- 动态路由：简单请求本地小模型，复杂规划升级大模型，按任务难度分配算力。

### 5.2 循环终止条件

Agent 的"刹车"必须显式设计，否则最小故障也会烧光预算：

- **完成信号**：模型显式 Finish，且结果通过校验器（不信任自声明）；
- **硬预算**：最大步数 / 最大 token / 最长墙钟时间 / 最大工具调用次数，任一触顶即强制收尾；
- **停滞检测**：连续 $k$ 步无新信息（重复相同的 action/observation）→ 注入反思提示或强制 replan；
- **人机协作**：高危动作（删除、支付、对外发送）前进入 human-in-the-loop 等待确认。

### 5.3 错误恢复

- 工具失败（超时、异常、schema 不符）：把错误信息作为 observation 回传让模型自纠，配重试 + 指数退避；连续失败多次就换路径，而非硬重试；
- 幻觉工具调用：在 runtime 侧做 schema 校验拦截，返回明确错误提示（"工具不存在，可用工具为……"），而不是崩溃或静默忽略；
- 全局兜底：预算耗尽时输出"部分结果 + 已完成/未完成清单"而不是空手而归；完整轨迹存档供复盘。

---

## 6 失败模式与对策 {#6}

| 失败模式 | 典型表现 | 对策 |
|----------|----------|------|
| 循环（looping） | 反复调用同一工具、Thought 复读 | 停滞检测、重复动作去重、强制 replan、预算上限 |
| 幻觉工具调用 | 编造不存在的工具/参数 | 强 schema 校验、工具名枚举、错误回传引导 |
| 上下文爆炸 | 一次工具结果塞满窗口，成本与延迟失控 | 结果截断/摘要、滚动压缩、子 Agent 隔离 |
| 过早终止 | 任务未完成即 Finish | 完成判据校验器、checklist 驱动 |
| 过度能动性 | 越权执行危险操作 | 权限分级、沙箱执行、HITL 确认 |

三点展开：

- **上下文爆炸**是最隐蔽的慢性失败：工具结果应视为外部记忆，按需检索式回读，而非全文堆叠；多 Agent 把长轨迹隔离在各自上下文中、只回传结论，是控制上下文规模的主流手段（见 [`agents/multi-agent.md`](multi-agent.md)）；
- **循环**常源于观察没有产生新信息——此时应改变的是计划而非重复行动，这也是反思态存在的意义（§2.3）；
- 失败模式的系统性度量（失败率分类、轨迹级评估）见 [`agents/agent-evaluation.md`](agent-evaluation.md)。

---

## 7 Agent 评测 {#7}

### 7.1 通用基准

| 基准 | 任务类型 | 难度 |
|------|----------|------|
| AgentBench | OS / DB / Web / KG 多场景 | 中 |
| WebArena | 真实网站 web 交互 | 中-高 |
| VisualWebArena | WebArena + 视觉元素 | 高 |
| SWE-bench | 真实 GitHub issue 修复 | 高 |
| GAIA | 通用 Assistant 多步问答 | 高 |
| τ-bench | 多轮对话 + 工具调用 | 中-高 |
| ToolBench / API-Bank | 工具调用能力 | 低-中 |

### 7.2 评测指标

- **Pass@k**：k 次尝试中至少 1 次成功的概率。
- **Pass^k**：k 次连续尝试全部成功的概率，衡量稳健性。
- **步骤成功率**：Agent 每步动作的可用性。
- **人在回路评估**：复杂任务常需人工打分（轨迹质量、最终结果）。

### 7.3 评测陷阱

- **过拟合基准**：模型针对特定 benchmark 调优，泛化能力被高估。
- **轨迹外观 ≠ 真实能力**：Thought 看似合理但 Action 错误的"幻觉式"轨迹难被自动评测捕获。
- **环境稳定性**：真实环境变化会破坏结果可比性，需要容器化/快照化环境。

---

> 相关文档：工具与 MCP 协议详见 [`agents/tools-and-mcp.md`](tools-and-mcp.md)，多 Agent 协作详见 [`agents/multi-agent.md`](multi-agent.md)，Agent 评测基准详见 [`agents/agent-evaluation.md`](agent-evaluation.md)，Agent 安全详见 [`agents/agent-security.md`](agent-security.md)，网页 Agent 见 [`agents/web-agents.md`](web-agents.md)。

## 8 参考文献 {#8}

[1] S. Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR, 2023. arXiv:2210.03629.

[2] N. Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS, 2023. arXiv:2303.11366.

[3] S. Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, NeurIPS, 2023. arXiv:2305.10601.

[4] X. Liu et al., *AgentBench: Evaluating LLMs as Agents*, ICLR, 2024. arXiv:2308.03688.

[5] S. Zhou et al., *WebArena: A Realistic Web Environment for Building Autonomous Agents*, ICLR, 2024. arXiv:2307.13854.

[6] C. Jimenez et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, ICLR, 2024. arXiv:2310.06770.

[7] Y. Yao et al., *τ-bench: A Benchmark for Tool-Agent-User Interaction*, 2024.

[8] J. Koh et al., *VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks*, COLM, 2024.

[9] Y. Shoham et al., *Multi-Agent Systems: Algorithmic, Game-Theoretic, and Logical Foundations*, Cambridge University Press, 2009.

[10] M. Packer et al., *MemGPT: Towards LLMs as Operating Systems*, 2023. arXiv:2310.08560.

[11] L. Wang et al., *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models*, ACL, 2023. arXiv:2305.04091.

[12] J. Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS, 2022. arXiv:2201.11903.