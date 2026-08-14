# 模型编辑与知识更新

> 从 ROME 定位-编辑到持续预训练，让大模型高效、精准地获取新知识。

---

## 目录

1. [概述](#1)
2. [知识存储定位](#2)
3. [ROME 与 MEMIT](#3-rome-memit)
4. [Adapter 式知识注入](#4-adapter)
5. [持续预训练](#5)
6. [知识编辑评估](#6)
7. [实践建议](#7)
8. [参考文献](#8)

---

## 1. 概述

大模型的知识具有时效性：预训练完成后，世界仍在变化。模型编辑（Model Editing）研究如何在不重新训练整个模型的前提下，精准修改模型中的特定知识或行为。

核心需求场景：

- **事实更新**：修正过时或错误的事实（如"美国总统是谁"）。
- **知识注入**：添加预训练时不存在的新知识（如新发布的 API 用法）。
- **偏见修正**：修改模型中的刻板印象或有害关联。
- **行为定制**：修改模型在特定输入下的输出风格或策略。

模型编辑的核心挑战是在修改目标知识的同时：
1. 不影响其他无关知识（特异性，specificity）。
2. 修改具有泛化能力（改写 "X 的首都是 Y" 也能正确回答 "Y 是 X 的首都"）。
3. 修改是持久的，不被后续对话遗忘。

---

## 2. 知识存储定位

### 2.1 中间层 MLP 作为知识存储

Meng et al. (2022) 通过因果追踪（Causal Tracing）证明，Transformer LLM 中的事实知识主要存储在中间层的 MLP 中：

1. 对事实提示（如 "The Eiffel Tower is located in"）进行前向传播。
2. 在各层添加噪声或恢复干净激活，观察对正确答案概率的影响。
3. 实验显示，恢复中间层（约 1/3 到 2/3 层）的 MLP 激活最能恢复正确答案。

这一发现表明 MLP 层充当了"键值记忆"（key-value memory）：前文上下文作为键，MLP 输出相关事实作为值。

### 2.2 注意力层的角色

注意力层主要负责信息路由和上下文推理，而非事实存储。但注意力头会影响知识检索——它们负责将"问题"与 MLP 中存储的"知识"关联起来。

---

## 3. ROME 与 MEMIT

### 3.1 ROME

ROME（Rank-One Model Editing, Meng et al., 2022）将 MLP 层视为线性联想记忆，通过秩为一的权重修改插入新知识。

对于第 $l$ 层 MLP 的第二线性层 $W_{\text{proj}}^{(l)}$，ROME 求解：

$$
W'_{\text{proj}} = W_{\text{proj}} + (\mathbf{v}^* - W_{\text{proj}} \mathbf{k}) \frac{\mathbf{k}^\top}{\mathbf{k}^\top \mathbf{k}}
$$

其中：
- $\mathbf{k}$ 是主体（subject，如"巴黎"）在该层的键表示。
- $\mathbf{v}^*$ 是目标客体（object，如"法国"）在最后一层的表示。
- 修改后的 $W'$ 满足 $W' \mathbf{k} = \mathbf{v}^*$，即当输入键为 $\mathbf{k}$ 时输出目标值。

ROME 一次编辑一条知识，且仅修改一个 MLP 层。

### 3.2 MEMIT

MEMIT（Meng et al., 2023）将 ROME 扩展为同时编辑数千条知识：

- 同时修改多个中间层（而非单层），分散存储修改。
- 使用正交化的键向量避免编辑间相互干扰。
- 可在一次编辑中修改上千条事实，且保持较高的特异性。

MEMIT 的批量编辑通过求解最小二乘问题实现：

$$
\min_{\Delta} \|\Delta K - (V^* - W K)\|_F^2 + \lambda \|\Delta\|_F^2
$$

其中 $K$ 是所有编辑键的矩阵，$V^*$ 是对应目标值矩阵。

### 3.3 其他编辑方法

| 方法 | 原理 | 特点 |
|------|------|------|
| MEND | 训练超网络将单样本梯度转化为编辑 | 需额外训练 |
| SERAC | 训练独立的反事实模型处理修改 | 不修改原模型 |
| IKE | 用上下文示例实现编辑（In-Context） | 无需训练但占用上下文 |
| T-Patcher | 在 MLP 中添加神经元补丁 | 增量式、可组合 |
- **AlphaEdit**：使用 DPO 信号做无训练参数编辑，提升泛化性。

### 3.4 局限

- **涟漪效应（Ripple Effect）**：编辑一条知识可能意外影响相关知识（如修改了"A 是 B 的首都"也改变了"B 的人口"）。
- **泛化不可控**：编辑可能不覆盖所有转述形式，也可能过度泛化。
- **规模限制**：编辑超过一定数量（~10K）后性能下降。
- **不适合程序性知识**：ROME/MEMIT 主要适合事实性知识，不适合修改推理能力或风格。

---

## 4. Adapter 式知识注入

### 4.1 LoRA 用于知识更新

LoRA（Low-Rank Adaptation）在冻结的权重旁添加低秩更新矩阵：

$$
W' = W + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}
$$

用新知识数据训练 LoRA 适配器，可实现参数高效的知识注入。优势是模块化——不同领域知识可训练不同 LoRA，按需加载。

### 4.2 K-Adapter

K-Adapters（Wang et al., 2021）为每种新知识训练独立的 Adapter，通过可学习的融合层注入模型。多个 K-Adapter 可插拔组合，互不干扰。

### 4.3 检索增强作为"软编辑"

RAG 可视为非参数化的知识编辑：

- 不改模型权重，通过检索最新文档更新知识。
- 适合高频更新的知识（新闻、文档、代码库）。
- 不适合需要模型内化的程序性知识（推理模式、风格）。

RAG 与模型编辑互补：时效性强、可验证的知识用 RAG；稳定的、需要泛化的知识用参数编辑。

---

## 5. 持续预训练

### 5.1 什么是持续预训练

持续预训练（Continued Pre-training / Domain-Adaptive Pre-training, DAPT）在已有基座模型上用新领域或新时间的数据继续预训练：

$$
\mathcal{L}_{\text{CPT}} = -\sum_{i} \log P(x_i | x_{<i}; \theta_0 + \Delta\theta)
$$

通常以较小学习率（如基座训练的 1/10）在新数据上训练 1-3 个 epoch。

### 5.2 应用场景

- **时间更新**：用最新日期的语料刷新知识截止时间。
- **领域适配**：在医疗、法律、金融等领域语料上继续训练。
- **语言扩展**：在新语言数据上训练以增强多语言能力。
- **能力增强**：在代码、数学等高质量数据上训练提升推理能力。

### 5.3 持续预训练 vs 微调

| 维度 | 持续预训练 | SFT/微调 |
|------|-----------|---------|
| 数据 | 大规模无标注文本 | 小规模指令数据 |
| 学习率 | 小（1e-5 量级） | 较大（1e-4 量级） |
| 目标 | 更新知识和语言能力 | 对齐输出格式和风格 |
| 灾难性遗忘 | 风险较高 | 风险较低（通常加通用数据混合） |
| 计算量 | 大 | 小 |

### 5.4 缓解灾难性遗忘

- **数据混合**：新数据中混入 5%-30% 的通用预训练数据，保持旧能力。
- **弹性权重巩固（EWC）**：对重要参数施加正则化，防止大幅偏移：

$$
\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}} + \lambda \sum_i F_i (\theta_i - \theta_i^*)^2
$$

其中 $F_i$ 是 Fisher 信息矩阵的对角线元素。
- **LoRA / 参数高效方法**：只更新少量参数，降低遗忘风险。
- **回放（Replay）**：在训练中定期回放旧数据子集。

### 5.5 知识注入的特殊技术

- **知识增量预训练**：将结构化知识（知识图谱三元组）转为自然语言句子，加入持续预训练。
- **合成数据注入**：用 LLM 生成关于新知识的问答和解释数据，用于 CPT 或 SFT。
- **时间感知训练**：在数据中标注时间戳，让模型学习"知识随时间变化"的概念。

---

## 6. 知识编辑评估

### 6.1 核心评估维度

| 维度 | 含义 | 指标 |
|------|------|------|
| 编辑成功率 | 编辑后是否正确回答目标知识 | Efficacy Score |
| 泛化能力 | 是否正确回答转述和相关问题 | Paraphrase / Neighborhood Score |
| 特异性 | 是否不影响无关知识 | Specificity Score |
| 持久性 | 编辑效果是否在多轮对话后保持 | Retention Score |
| 流利性 | 编辑后模型语言能力是否受损 | Fluency / Perplexity |

### 6.2 评测基准

- **CounterFact**：ROME/MEMIT 使用的标准评测集，包含反事实事实及其转述。
- **MQuAKE**：多跳问题编辑评测，测试知识编辑对多跳推理的影响。
- **KnowEdit**：大规模知识编辑基准，覆盖多种知识类型和编辑方法。
- **WikiBio / 时间敏感 QA**：评估时间性知识更新效果。

### 6.3 评估陷阱

- 编辑成功率高不等于实际效果好——模型可能只是记住了特定问答对。
- 需在编辑后测试通用能力（MMLU 等）是否下降。
- 多跳推理中，编辑一条中间知识可能影响整条推理链，需仔细评估。

---

## 7. 实践建议

### 7.1 方案选型

| 场景 | 推荐方案 |
|------|---------|
| 修改少量事实（< 100 条） | ROME / MEMIT |
| 批量领域知识更新 | 持续预训练 + 通用数据混合 |
| 时效性知识（新闻、文档） | RAG（不改权重） |
| 领域适配（医疗/法律/代码） | 领域 CPT + 领域 SFT |
| 多领域可切换 | LoRA 适配器（每领域一个） |
| 临时修改/原型验证 | In-Context 编辑（IKE） |

### 7.2 工程建议

- ROME/MEMIT 适合一次性批量编辑，但不适合高频更新（每次编辑需重新计算）。
- 持续预训练前备份原始模型，便于回滚。
- 编辑后务必运行通用能力 benchmark 回归测试，确保无意外退化。
- RAG + 模型编辑可组合：RAG 处理高频变化信息，模型编辑内化稳定知识。
- 对编辑操作建立审计日志：记录修改了什么、何时修改、谁授权，便于追溯和回滚。

### 7.3 研究趋势

- 从单事实编辑向多跳推理编辑发展（MQuAKE 等）。
- 从事实编辑向行为、风格、价值观编辑扩展。
- 编辑方法与对齐技术融合——安全对齐本身可视为一种行为编辑。
- 终生学习（Lifelong Learning）：模型在部署后持续、安全地更新知识的完整框架。

---

## 8. 参考文献

[1] K. Meng, D. Bau, A. Andonian, Y. Belinkov, *Locating and Editing Factual Associations in GPT*, NeurIPS, 2022.

[2] K. Meng, A. S. Sharma, A. Andonian, et al., *Mass-Editing Memory in a Transformer*, ICLR, 2023.

[3] E. Mitchell, C. Lin, A. Bosselut, et al., *MEND: Fast Model Editing at Scale*, ICLR, 2022.

[4] S. Wang, L. Li, X. Zheng, et al., *K-Adapter: Infusing Knowledge into Pre-Trained Models with Adapters*, Findings of ACL, 2021.

[5] Z. Zheng, F. Mi, Y. Xu, et al., *IKE: In-Context Knowledge Editing*, arXiv preprint, 2023.

[6] N. Zhang, Y. Xu, T. Qiu, et al., *Evaluating Knowledge-Editing Methods on a Multilingual Benchmark*, TACL, 2024.

[7] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, et al., *Overcoming Catastrophic Forgetting in Neural Networks*, PNAS, 2017.

[8] K. Hase, Z. Huo, X. Hu, et al., *Can Language Models Learn to Edit Knowledge?*, ICLR, 2024.

[9] C. Zhong, Y. Chen, X. Xie, et al., *A Survey on Knowledge Editing for Large Language Models*, arXiv preprint, 2024.

[10] M. Cohen, M. Zhou, S. Wies, et al., *Knowledge Editing of Large Language Models: A Survey*, arXiv preprint, 2024.
