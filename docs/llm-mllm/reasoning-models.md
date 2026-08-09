# 推理模型：CoT、o1 与 DeepSeek-R1

> 「推理模型」通过延长思考过程显著提升复杂任务表现，本质是把算力从训练期延伸到推理期（test-time compute）。

---

## 目录

1. [推理模型概述](#1)
2. [思维链 (Chain-of-Thought)](#2-chain-of-thought)
3. [Test-time Compute Scaling](#3-test-time-compute-scaling)
4. [o1 / R1 范式](#4-o1-r1)
5. [奖励设计：ORM vs PRM](#5-orm-vs-prm)
6. [训练策略](#6)
7. [评测与局限](#7)
8. [参考文献](#8)

---

## 1. 推理模型概述

### 1.1 什么是推理模型

「推理模型」（Reasoning Model）指通过**延长思考过程**显著提升复杂任务（数学、代码、逻辑推理）表现的 LLM。代表工作包括 **OpenAI o1/o3**、**DeepSeek-R1**。其核心思想是**把算力从训练期延伸到推理期（test-time compute）**，让模型在推理时花费更多 token 进行内部搜索、反思与回溯。

### 1.2 与普通 LLM 的核心区别

| 维度 | 普通 LLM | 推理模型 |
|------|----------|---------|
| 推理方式 | 一次前向，直接生成答案 | 多步推理，显式思考过程 |
| 算力分配 | 集中在训练阶段 | 训练 + 推理期双阶段算力 |
| Token 消耗 | 仅答案 token | 大量中间推理 token |
| 复杂任务表现 | 有限 | 显著提升（数学、代码等） |
| 代表模型 | GPT-4o, Llama-4 | o1, o3, DeepSeek-R1 |

### 1.3 发展脉络

| 年份 | 工作 | 关键贡献 |
|------|------|---------|
| 2022 | Chain-of-Thought (Wei et al.) | 发现「一步步思考」显著提升推理 |
| 2022 | Zero-shot CoT (Kojima et al.) | 仅提示 "Let's think step by step" 即生效 |
| 2022 | Self-Consistency (Wang et al.) | 多条路径投票，稳健性更强 |
| 2023 | Tree-of-Thought (Yao et al.) | 树搜索 + 状态评估，复杂规划 |
| 2024 | OpenAI o1 | 用 RL 内化长链推理 + 自我验证 |
| 2024 | GRPO (Shao et al., DeepSeekMath) | 组相对策略优化，省 Critic |
| 2025 | DeepSeek-R1 | 开源可复现的 RL 推理训练范式 |

---

## 2. 思维链 (Chain-of-Thought)

### 2.1 Few-shot CoT

**核心思想**：在提示中提供包含推理步骤的示例，让模型模仿逐步推理过程。

Wei et al. (2022) 发现，在 few-shot 示例中显式写出中间推理步骤，能让模型在数学、常识推理等任务上大幅提升。本质是**用示例激发模型已有的推理能力**，而非注入新知识。

**示例格式**：

```text
Q: 小明有 5 个苹果，吃了 2 个，又买了 3 个，现在有几个？
A: 小明开始有 5 个苹果。吃了 2 个，剩下 5 - 2 = 3 个。
   又买了 3 个，现在有 3 + 3 = 6 个。答案是 6。
```

### 2.2 Zero-shot CoT

Kojima et al. (2022) 发现，即使不提供推理示例，仅在提示末尾加上 **"Let's think step by step"** 就能触发模型一步步推理。这证明了模型在预训练中已经学到了推理模式，只需合适的提示即可激活。

**关键发现**：Zero-shot CoT 在数学推理任务（如 MultiArith、GSM8K）上效果显著，甚至与 few-shot CoT 持平，说明推理能力是潜在的而非需要示例教会。

### 2.3 Self-Consistency

Wang et al. (2022) 提出，对同一问题**采样多条推理路径**，然后对最终答案投票，取最多票数者。这利用了「条条大路通罗马」的直觉：正确的推理路径可能多种多样，但错误路径往往分散。

**流程**：

```text
问题 → 采样 N 条 CoT 路径 → 提取每条路径的最终答案 → 多数投票 → 最终答案
```

效果：$N$ 越大，投票越稳定，尤其在数学和逻辑推理任务上。通常 $N = 5 \sim 40$ 即可获得显著收益。

### 2.4 Tree-of-Thought

Yao et al. (2023) 将推理建模为**树搜索问题**：每个节点是推理的中间状态，分支是候选推理步骤，通过评估每个节点选出最优路径。

**对比示意图**：

```text
CoT (线性):           Tree-of-Thought (分支):
问题                   问题
  ↓                     ├─ 步骤 A1
  步骤 1                  │  ├─ 步骤 A2a → 答案 A
  ↓                     │  └─ 步骤 A2b (回溯)
  步骤 2                 │
  ↓                     ├─ 步骤 B1
  答案                    │  ├─ 步骤 B2a → 答案 B*
                          │  └─ 步骤 B2b (回溯)
                          └─ 步骤 C1 (放弃)
                             
  * 最终选择最优路径 B
```

**核心操作**：
- **分支 (Branching)**：在每个节点生成多个候选推理方向
- **评估 (Evaluation)**：对每个节点打分（可用 LLM 自身或启发式）
- **搜索 (Search)**：BFS/DFS 遍历，选择最有希望的路径

**适用场景**：需要规划和回溯的复杂任务，如 24 点游戏、创意写作、旅行规划等。CoT 的线性推理在需要「试错」的任务上力不从心，而 ToT 的搜索机制天然适合。

---

## 3. Test-time Compute Scaling

### 3.1 核心直觉

传统 LLM 的范式是：算力几乎全部投入训练，推理时一个答案一次前向。推理模型打破了这一范式：**在生成最终答案前，花费大量 token 进行内部搜索**。

如果用数学形式化：给定问题 $x$，传统 LLM 直接建模 $P(y|x)$，而推理模型先搜索一个推理路径 $z$，再给出答案 $P(y|x,z)$，其中 $z$ 是中间推理序列。

### 3.2 Scaling Law

OpenAI 在 o1 技术报告中提出 **test-time compute scaling law**：随着推理期算力（如推理 token 数、搜索宽度）增加，模型在困难任务上的准确率**持续提升**，而非快速饱和。

$$\text{Accuracy}(C_{\text{test}}) \propto \log(C_{\text{test}})$$

其中 $C_{\text{test}}$ 是推理期算力。这类似于训练期的 scaling law，但作用于推理阶段。

**关键含义**：面对难题时，与其训练更大的模型，不如让现有模型「多想一会儿」——这改变了算力分配的思路。

### 3.3 搜索形式

在答案空间上搜索的具体形式：

| 搜索方式 | 描述 | 示例 |
|----------|------|------|
| 树搜索 | 分支探索推理路径，回溯选优 | Tree-of-Thought, MCTS |
| 自我验证 | 生成答案后自我检查、修正 | Self-Refine, Self-Verification |
| 多候选排序 | 生成多个候选，用奖励模型排序 | Best-of-N, 重排序 |
| 反思链 | 显式生成「反思/修正」token | R1 的「啊哈」时刻 |

---

## 4. o1 / R1 范式

### 4.1 OpenAI o1

OpenAI 于 2024 年发布 o1 系列推理模型，核心是用 **强化学习（RL）** 训练模型学会「长链推理 + 自我验证」。

**关键特征**：
- **显式 Thinking 阶段**：推理时模型先进入隐藏的「思考」阶段，生成大量推理 token，再输出最终答案。用户看到的摘要思考过程经过过滤，原始 Reasoning Token 不公开。
- **RL 内化推理**：不是通过 prompt 工程引导推理，而是 RL 训练让模型**自发**产生推理行为，这类似于 AlphaGo 的「棋感」内化。
- **方法未完全公开**：OpenAI 未披露 RL 算法细节、奖励函数设计和训练数据，但推测涉及 PRM 和过程监督。

**表现**：o1 在 AIME 2024 上达到 74.4%（vs GPT-4o 的 13.4%），在 GPQA Diamond 上超越人类专家。

### 4.2 DeepSeek-R1（开源可复现范式）

DeepSeek 于 2025 年开源 R1，提供了**完整可复现**的推理模型训练范式。

#### R1-Zero：纯 RL 涌现推理

**训练流程**：直接在 DeepSeek-V3-Base 基座上用 **GRPO**（Group Relative Policy Optimization）进行 RL 训练，仅使用**基于规则的奖励**（答案正确性 + 格式），完全不使用人工偏好标注。

**关键现象**：
- **自发涌现长 CoT**：模型在 RL 过程中自动学会生成越来越长的推理链
- **「啊哈」时刻 (Aha Moment)**：模型学会在推理中途自我反思、修正错误，体现为在推理链中插入「Wait, let me reconsider...」等反思 token
- **无需人工演示**：纯 RL 即可让模型获得推理能力，证明推理能力是 RL 优化过程中涌现的

#### R1：冷启动 + 多阶段训练

在 R1-Zero 基础上改进：

1. **冷启动 SFT**：用数千条高质量可读 CoT 数据对基座进行 SFT，解决 R1-Zero 推理可读性差的问题
2. **推理 RL**：在冷启动模型上继续 GRPO 推理训练
3. **拒绝采样 + SFT**：用 RL 模型生成推理数据，结合通用 SFT 数据，进行新一轮 SFT（提升通用能力）
4. **全场景 RL**：最后阶段进行兼顾推理和通用能力的 RL

#### 蒸馏（Distillation）

将 R1 的推理能力蒸馏到小模型：

| 蒸馏模型 | 基座 | AIME 准确率 |
|----------|------|------------|
| DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-7B | 55.5% |
| DeepSeek-R1-Distill-Qwen-14B | Qwen2.5-14B | 69.7% |
| DeepSeek-R1-Distill-Qwen-32B | Qwen2.5-32B | 79.9% |
| DeepSeek-R1-Distill-Llama-70B | Llama-3.3-70B | 80.0% |

**蒸馏方法**：用 R1 生成的推理数据（CoT + 答案）对基座进行 SFT，无需 RL 即可获得强推理能力。

### 4.3 RL 训练循环

```text
         ┌──────────── 推理模型(策略 π) ────────────┐
         │  采样 rollout: 问题 → 长 CoT → 答案      │
         └───────────────┬──────────────────────────┘
                         │ 奖励 r
              ┌──────────┴──────────┐
              │ 规则 / ORM / PRM 打分 │
              └──────────┬──────────┘
                         │ 优势估计 (GRPO: 组内相对)
                         ▼
                 策略梯度更新 π (GRPO)
```

---

## 5. 奖励设计：ORM vs PRM

### 5.1 ORM (Outcome Reward Model)

**定义**：只对最终答案是否正确打分，不给中间步骤反馈。

`ORM`(x, y) = 1 如果答案正确，否则 0（规则奖励情况）

**优点**：
- 简单直接，无需标注中间步骤
- 对于可验证任务（数学、代码）天然适用
- 和 GRPO 配合好：组内答案正确性比较即可

**缺点**：
- 无过程监督，模型可能通过错误推理蒙对答案
- 长链推理中，奖励信号稀疏，优化困难
- 不能指导模型「哪一步错了」

### 5.2 PRM (Process Reward Model)

**定义**：对每一步推理分别打分，提供稠密的中间步骤奖励。

`PRM`(x, z_1, ..., z_t) 给出步骤 $z_t$ 的质量评分

**优点**：
- 提供稠密监督信号，加速 RL 收敛
- 能发现并惩罚错误中间步骤
- 适合需要精确推理链的任务

**缺点**：
- 标注成本极高：需要标注每一步的正确性
- 自动标注（如用模型判断）不够可靠
- PRM 本身可能被 reward hacking

**Lightman et al. (2023)** 对比了 ORM 和 PRM，发现 PRM 在数学推理上优于 ORM，但差距取决于标注质量和任务难度。

### 5.3 GRPO：组相对策略优化

GRPO（Group Relative Policy Optimization）是 DeepSeek-R1 的关键训练算法，由 Shao et al. (2024) 在 DeepSeekMath 中提出。

**核心思想**：对同一问题采样一组 $G$ 个 rollout，以组内平均奖励为基线计算优势，**免去独立 Critic 网络**。

优势估计：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

其中 $r_i$ 是第 $i$ 个 rollout 的奖励。

**GRPO 目标函数**：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[\min\left(\rho_i(\theta) A_i, \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) A_i\right)\right] - \beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

其中 $\rho_i(\theta) = \frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}$ 是重要性采样比。

**GRPO vs PPO**：

| 维度 | PPO | GRPO |
|------|-----|------|
| Critic 网络 | 需要独立 Value 网络 | 不需要，组内相对比较 |
| 显存 | 高（Critic 参数 + 梯度） | 低（无 Critic） |
| 优势估计 | 用 Critic 预测 | 组内标准化 |
| 基线 | 学习得到 | 组内均值 |
| 适用场景 | 通用 RL | 推理模型 RL |

**为什么 GRPO 适合推理模型**：推理任务天然可分组（同一问题多次采样），组内相对比较能有效区分好坏推理路径，且省去 Critic 网络大幅降低训练成本。

---

## 6. 训练策略

### 6.1 奖励设计策略

不同类型的任务采用不同的奖励策略：

| 任务类型 | 奖励方式 | 示例 |
|----------|---------|------|
| 数学（可验证） | 规则奖励：答案匹配即 1，否则 0 | MATH, AIME, GSM8K |
| 代码（可验证） | 规则奖励：通过测试用例即 1 | Codeforces, HumanEval |
| 逻辑推理 | 规则奖励 + 格式奖励 | 逻辑谜题 |
| 开放问答 | ORM / PRM 或偏好模型打分 | 创意写作、摘要 |
| 通用任务 | 多奖励组合（规则 + 格式 + 语言一致性） | 综合评测 |

**格式奖励**：R1 训练中要求模型用 `\boxed{}` 标记最终答案，用 `\n` 分隔推理步骤，格式正确可获得额外奖励。

### 6.2 冷启动（Cold Start）

R1 的训练经验表明，直接从基座做 RL（R1-Zero）虽然能涌现推理，但存在两个问题：
- 推理链可读性差，混合多种语言
- 格式不规范，难以解析

**冷启动策略**：先用数千条高质量 CoT 数据对基座做 SFT，让模型学会「什么样的推理是好的」，再进入 RL 阶段。

冷启动数据来源：
- 人工标注的高质量 CoT 示例
- 从其他推理模型蒸馏的 CoT
- 使用 few-shot prompt 引导基座生成，再人工筛选

### 6.3 训练数据

**核心数据**：以**可自动验证**的任务为主：

| 数据来源 | 类型 | 规模 |
|----------|------|------|
| MATH | 数学竞赛题 | 12,500 题 |
| AIME | 美国数学邀请赛 | 历年真题 |
| GSM8K | 小学数学应用题 | 8,500 题 |
| Codeforces | 竞赛编程 | 大量 |
| 合成数据 | 用已有模型生成推理数据 | 海量 |

**关键原则**：选择奖励信号明确的任务，让 RL 有明确的优化方向。可验证任务（数学、代码）天然适合——规则奖励明确、无歧义。

### 6.4 蒸馏策略

用大推理模型生成的 CoT 数据监督训练小模型，是成本最低、效果最好的推理能力迁移方式。

**蒸馏流程**：

```text
大推理模型 (R1 / o1)  →  生成带 CoT 的答案  →  筛选高质量样本  →  SFT 小模型
```

**关键发现**（来自 R1 论文）：
- 蒸馏效果往往优于直接在小模型上做 RL
- 推理能力可以跨模型架构迁移（如 R1 → Qwen → Llama）
- 蒸馏 + RL 组合效果最佳：先蒸馏获取基础推理能力，再用 RL 在特定任务上强化

### 6.5 训练注意事项

**长度与熵管理**：
- RL 训练中模型倾向于生成越来越长的推理链，可能产生「长度 Reward Hacking」——用冗长但无意义的推理骗取更高奖励
- 需要加入长度惩罚 / 格式正则，防止推理链无限膨胀
- 监控推理 token 平均长度，设置合理上限

**语言一致性**：
- 推理链的语言可能不稳定，中英文混杂
- 在 CoT 冷启动数据中强化目标语言的一致性
- R1 在奖励中加入语言一致性奖励

**计算资源**：
- GRPO 训练需要大规模采样（每组 $G$ 个 rollout）
- 推理阶段的极长序列（可达数万 token）对显存和训练速度有要求
- 推荐使用 FlashAttention 等长序列优化

---

## 7. 评测与局限

### 7.1 主流评测基准

| 基准 | 类型 | 说明 |
|------|------|------|
| MATH-500 | 数学竞赛 | 涵盖代数、几何、数论等 |
| AIME 2024 | 数学竞赛 | 美国数学邀请赛，高难度 |
| GPQA Diamond | 研究生级科学 | 物理、化学、生物博士级问答 |
| Codeforces | 竞赛编程 | 含推理和算法设计 |
| MMLU-Pro | 综合知识 | 比 MMLU 更难的升级版 |
| LiveCodeBench | 代码 | 实时更新防数据污染 |

**表现对比**（部分数据）：

| 模型 | AIME 2024 | MATH-500 | GPQA Diamond |
|------|-----------|----------|--------------|
| GPT-4o | 13.4% | 76.6% | 56.1% |
| o1-preview | 56.7% | 94.8% | 73.3% |
| o1 | 74.4% | 96.4% | 77.3% |
| DeepSeek-R1 | 71.0% | 97.3% | 71.5% |
| DeepSeek-R1-32B (蒸馏) | 72.6% | 94.3% | 62.1% |

**关键结论**：推理模型在数学和科学推理上大幅领先非推理模型，且蒸馏后的小模型也能保持强推理能力。

### 7.2 主要局限

**推理成本高**：
- 推理 token 可达数万甚至数十万，远超普通 LLM
- API 调用成本显著增加（按 token 计费）
- 延迟高：复杂问题推理可能需要数分钟

**过度思考**：
- 对简单问题也可能生成冗长的推理链
- 无法自适应推理长度（需要外部预算控制）
- 浪费计算资源

**幻觉放大**：
- 长推理链中错误可能被传播和放大
- 自我验证不一定可靠——模型可能确认自己的错误
- 在开放域问题上幻觉率没有显著降低

**可解释性有限**：
- o1 的完整 Reasoning Token 不公开，只展示过滤摘要
- R1 的推理链可读性经过训练改善，但仍有大量冗余
- 推理过程未必反映真实决策机制

**鲁棒性问题**：
- 对 prompt 格式敏感（如是否要求 `\boxed{}` 格式）
- 在某些分布外数据上退化明显
- 奖励 hacking 风险（优化奖励而非真实推理）

### 7.3 未来方向

- **推理效率**：自适应推理长度、推理 token 压缩、投机解码加速
- **多模态推理**：将测试时计算扩展到视觉、音频等多模态输入
- **推理与 Agent 结合**：推理模型作为 Agent 的「大脑」，规划 + 反思 + 工具调用
- **推理安全**：确保长链推理中的安全对齐，防止推理过程中绕过安全约束

---

## 8. 参考文献

[1] Wei, J., Wang, X., Schuurmans, D., et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS, 2022.

[2] Kojima, T., Gu, S. S., Reid, M., et al., *Large Language Models are Zero-Shot Reasoners*, NeurIPS, 2022.

[3] Wang, X., Wei, J., Schuurmans, D., et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, ICLR, 2023.

[4] Yao, S., Yu, D., Zhao, J., et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, NeurIPS, 2023.

[5] OpenAI, *Learning to Reason with LLMs* (o1 System Card), 2024.

[6] Shao, Z., Wang, P., Zhu, Q., et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024.

[7] DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025.

[8] Lightman, H., Kosaraju, V., Burda, Y., et al., *Let's Verify Step by Step* (PRM), ICLR, 2024.

[9] Snell, C., Lee, J., Xu, K., et al., *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters*, 2024.

[10] Zelikman, E., Wu, Y., Mu, J., et al., *STaR: Bootstrapping Reasoning With Reasoning*, NeurIPS, 2022.