# 测试时计算与推理扩展

> 从 Best-of-N 到 Tree of Thoughts，从搜索增强到推理时 Scaling Law —— 用更多计算换更强推理。

---

## 目录

1. [概述](#1)
2. [思维链与自洽性](#2)
3. [Best-of-N 与验证器](#3-best-of-n)
4. [树搜索与图搜索](#4)
5. [搜索增强推理](#5)
6. [推理时 Scaling Law](#6-scaling-law)
7. [实践建议](#7)
8. [参考文献](#8)

---

## 1. 概述

传统深度学习遵循"训练时扩展、推理时固定"的模式：模型参数量和训练数据决定能力，推理时一次前向传播给出答案。测试时计算（Test-time Compute）反其道而行：在推理阶段投入更多计算（多次采样、搜索、验证），显著提升模型在困难任务上的表现。

这一范式因 OpenAI o1 和 DeepSeek-R1 的成功而成为 2024-2025 年最活跃的研究方向。核心洞察是：对于推理密集型任务，增加推理计算量可以像增加训练计算量一样提升性能。

测试时计算的主要方法：

$$
\text{多次采样} \rightarrow \text{验证/选择} \rightarrow \text{结构化搜索} \rightarrow \text{学习式推理策略}
$$

与 `reasoning-models.md` 侧重训练侧（RL 训练推理模型）不同，本篇聚焦推理时的计算扩展机制，二者互补。

---

## 2. 思维链与自洽性

### 2.1 Chain-of-Thought

Chain-of-Thought（CoT, Wei et al., 2022）让模型在给出最终答案前生成中间推理步骤：

```text
问题：一个长方形长 8 米，宽 5 米，面积是多少？
思考：面积 = 长 × 宽 = 8 × 5 = 40 平方米。
答案：40 平方米。
```

CoT 将复杂问题分解为子步骤，每步只需简单推理。它不改变模型权重，而是通过提示格式引导模型分配更多计算到推理上。

### 2.2 Self-Consistency

Self-Consistency（Wang et al., 2022）对同一问题采样多条 CoT 推理路径，取多数投票答案：

1. 用温度 $T > 0$ 独立采样 $N$ 条推理链。
2. 每条链独立得出最终答案。
3. 选择出现次数最多的答案。

$$
\hat{a} = \arg\max_a \sum_{i=1}^{N} \mathbb{I}(a_i = a)
$$

Self-Consistency 是最简单有效的测试时计算方法，在数学推理上能带来 10-20% 的准确率提升。其有效性建立在"不同推理路径出错方式不同，但正确答案收敛"的假设上。

---

## 3. Best-of-N 与验证器

### 3.1 Best-of-N

Best-of-N 生成 $N$ 个候选回答，用某种评分函数选择最优：

- **多数投票**：选最常见答案（即 Self-Consistency）。
- **奖励模型打分**：用训练好的奖励模型对每个候选评分，选最高分。
- **规则验证**：代码任务运行测试用例，数学任务检查答案数值。
- **LLM-as-Judge**：用另一个（通常更强的）LLM 评判候选质量。

Best-of-N 的期望收益随 $N$ 增长但存在边际递减：

$$
P(\text{正确}) \approx 1 - (1-p)^N
$$

其中 $p$ 是单次采样正确的概率。当 $p=0.5$、$N=10$ 时，Best-of-N 正确率可达 99.9%。

### 3.2 验证器（Verifier）

验证器是一个独立模型，判断候选解答是否正确。训练流程：

1. 用生成模型对训练问题采样多条解答。
2. 标注每条解答正确与否（运行测试或人工标注）。
3. 训练验证器模型预测解答的正确性。
4. 推理时用验证器从 $N$ 个候选中选最优。

验证器可分为：

- **结果验证器（Outcome Verifier, OV）**：只判断最终答案是否正确。
- **过程验证器（Process Verifier, PRM）**：对推理每一步打分，提供更细粒度的信号。

过程奖励模型（Process Reward Model, PRM, Lightman et al., 2023）对每个推理步骤评分，能检测中间错误并引导搜索，在数学推理上显著优于结果验证。

### 3.3 可验证奖励（RLVR）

代码和数学任务天然可验证（测试通过/答案匹配），可验证奖励强化学习（RLVR）将验证信号用于训练时和测试时：

- 训练时：用验证结果作为 RL 奖励（见 `training/reinforcement-learning.md` 中 GRPO/RLVR）。
- 测试时：用验证器筛选候选，或在搜索中剪枝错误路径。

---

## 4. 树搜索与图搜索

### 4.1 Tree of Thoughts

Tree of Thoughts（ToT, Yao et al., 2023）将推理过程组织为树：

1. **分解**：将问题分解为多个思考阶段。
2. **生成**：每个状态生成多个候选"思考"（下一步推理）。
3. **评估**：用启发式方法（自评、规则、验证器）评估每个候选。
4. **搜索**：用 BFS 或 DFS 在树中搜索最优路径，剪枝低分支。

ToT 允许回溯——当某条路径走不通时，模型可以回到分叉点尝试其他方案，这是线性 CoT 无法做到的。

### 4.2 Graph of Thoughts

Graph of Thoughts（GoT, Besta et al., 2023）将推理结构从树扩展为有向图：

- 不同推理分支可以聚合（汇总多条线索）。
- 思考可以被细化和循环改进。
- 图结构支持更复杂的推理拓扑。

### 4.3 搜索策略

| 策略 | 特点 |
|------|------|
| 贪心解码 | 每步选概率最高，无回溯 |
| Beam Search | 维护 $k$ 条最优路径 |
| DFS + 回溯 | 深度探索，失败时回退 |
| BFS + 剪枝 | 层展探索，剪掉低分支 |
| MCTS | 蒙特卡洛树搜索，平衡探索与利用 |
| Lookahead Search | 向前模拟多步评估长期后果 |

AlphaProof/AlphaGeometry（DeepMind, 2024）将 MCTS 与 LLM 结合，在奥林匹克数学题上达到金牌水平。

---

## 5. 搜索增强推理

### 5.1 外部搜索

LLM 可以在推理过程中调用搜索引擎或知识库：

- **ReAct**：交替进行推理和行动（搜索/工具调用），观察结果后继续推理。
- **Self-Ask**：模型主动提出子问题并搜索答案，再组合成最终回答。
- **Toolformer**：模型学习在推理中自行决定何时调用工具。

外部搜索将模型的知识边界扩展到实时信息和海量文档。

### 5.2 代码执行增强推理

对于数学和逻辑问题，让模型生成并执行代码来验证或计算：

- **PAL（Program-Aided Language Models）**：模型生成 Python 代码来执行计算，而非在文本中心算。
- **Tool-integrated Reasoning**：推理过程中调用计算器、解释器、定理证明器。
- 代码执行避免了 LLM 的算术错误，将"计算"外包给确定性工具。

### 5.3 多 Agent 协作推理

多个 Agent 角色分工协作解决复杂问题：

- **辩论（Debate）**：多个 Agent 从不同角度论证，裁判模型综合判断。
- **角色分工**：规划者分解任务，执行者解决子问题，审查者验证结果。
- **Society of Minds**：大量专家 Agent 并行处理，聚合结果。

---

## 6. 推理时 Scaling Law

### 6.1 推理计算的缩放

Snell et al.（2024）系统研究了测试时计算的缩放规律：

- 在推理任务上，增加测试时计算量可以像增加模型参数量一样提升性能。
- 最优策略取决于任务难度：简单题用少计算，困难题投入更多搜索和验证。
- 存在"推理时计算最优分配"：将更多计算分配给更难的问题。

### 6.2 计算-性能曲线

测试时性能随计算量的增长模式：

$$
\text{Accuracy} \propto (\text{compute})^{-\alpha}
$$

其中 $\alpha$ 取决于任务类型和方法。Best-of-N 增长慢但稳定，搜索方法增长快但需要良好的验证器。

### 6.3 训练时 vs 推理时缩放

Wu et al.（2024）对比了两种缩放策略：

- **训练时缩放**：更大模型 + 更多数据，所有查询共享计算。
- **推理时缩放**：标准模型 + 更多推理计算，按查询难度分配。

推理时缩放的优势：

- 可按问题难度自适应分配计算，避免简单问题浪费。
- 可以用较小模型 + 搜索达到大模型的效果。
- 搜索和验证策略可以快速迭代，无需重新训练。

劣势：

- 推理成本和延迟增加。
- 需要有效的验证器或搜索启发式。
- 对非推理任务（如简单聊天）收益有限。

### 6.4 自适应计算

自适应测试时计算根据问题复杂度动态决定投入多少计算：

- **早停**：模型在高置信度时提前停止推理。
- **难度路由**：简单问题用贪心解码，困难问题启动树搜索。
- **元推理（Meta-Reasoning）**：模型自身判断是否需要进一步思考。
- **预算控制**：设定 Token 或时间预算，动态分配。

---

## 7. 实践建议

### 7.1 方法选型

- 简单事实性问答：直接回答，无需额外计算。
- 数学/逻辑推理：CoT + Self-Consistency（$N=5-10$），成本低收益高。
- 代码生成：Best-of-N + 测试用例验证，性价比最高。
- 高难度竞赛题：ToT/MCTS + PRM 验证器 + 代码执行。
- 需要实时知识：ReAct + 搜索引擎。

### 7.2 工程建议

- 设置计算预算上限（最大 Token 数/最大 API 调用次数），防止失控推理。
- 对 Best-of-N 的候选做去重，避免近似重复浪费计算。
- 验证器质量是搜索方法的天花板——投入在验证器上的回报最高。
- 批量采样和并行验证能有效降低延迟。
- 缓存重复子问题的结果（如不同路径共享的中间结论）。

### 7.3 评估建议

- 报告"计算量-性能"曲线而非单点性能，说明每点的 Token 成本。
- 区分训练时计算和推理时计算的总成本。
- 评估不同难度问题上的收益分布，确认方法不是只在简单题上有效。
- 与增加模型规模的 baseline 做成本效益对比。

---

## 8. 参考文献

[1] J. Wei, X. Wang, D. Schuurmans, et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS, 2022.

[2] X. Wang, J. Wei, D. Schuurmans, et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, ICLR, 2023.

[3] S. Yao, D. Yu, J. Zhao, et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*, NeurIPS, 2023.

[4] H. Lightman, V. Kosaraju, Y. Burda, et al., *Let's Verify Step by Step*, ICLR, 2024.

[5] M. Besta, N. Blach, A. Kubicek, et al., *Graph of Thoughts: Solving Elaborate Problems with Large Language Models*, AAAI, 2024.

[6] C. Snell, J. D. Swett, I. G. Xu, et al., *Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters*, arXiv preprint, 2024.

[7] Y. Wu, S. Zhao, H. Liu, et al., *Inference-Time Scaling for Generalist Reward Modeling*, arXiv preprint, 2024.

[8] G. Mialon, R. Fourrier, C. Wolf, et al., *Augmented Language Models: A Survey*, TMLR, 2023.

[9] T. Brown, B. Mann, N. Ryder, et al., *Language Models Are Few-Shot Learners*, NeurIPS, 2020.

[10] DeepMind, *AlphaProof and AlphaGeometry*, 2024.
