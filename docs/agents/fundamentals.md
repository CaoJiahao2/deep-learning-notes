# AI Agent 基础与核心循环

> 从 LLM 到会规划、会调用工具、会记忆的智能体：Agent = LLM + 工具调用 + 记忆 + 循环。

---

## 目录

1. [概述](#1)
2. [核心循环与公式化](#2)
3. [ReAct 范式](#3)
4. [规划范式总览](#4)
5. [Agent 评测](#5)
6. [参考文献](#6)

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

---

## 5 Agent 评测 {#5}

### 5.1 通用基准

| 基准 | 任务类型 | 难度 |
|------|----------|------|
| AgentBench | OS / DB / Web / KG 多场景 | 中 |
| WebArena | 真实网站 web 交互 | 中-高 |
| VisualWebArena | WebArena + 视觉元素 | 高 |
| SWE-bench | 真实 GitHub issue 修复 | 高 |
| GAIA | 通用 Assistant 多步问答 | 高 |
| τ-bench | 多轮对话 + 工具调用 | 中-高 |
| ToolBench / API-Bank | 工具调用能力 | 低-中 |

### 5.2 评测指标

- **Pass@k**：k 次尝试中至少 1 次成功的概率。
- **Pass^k**：k 次连续尝试全部成功的概率，衡量稳健性。
- **步骤成功率**：Agent 每步动作的可用性。
- **人在回路评估**：复杂任务常需人工打分（轨迹质量、最终结果）。

### 5.3 评测陷阱

- **过拟合基准**：模型针对特定 benchmark 调优，泛化能力被高估。
- **轨迹外观 ≠ 真实能力**：Thought 看似合理但 Action 错误的"幻觉式"轨迹难被自动评测捕获。
- **环境稳定性**：真实环境变化会破坏结果可比性，需要容器化/快照化环境。

---

## 6 参考文献 {#6}

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