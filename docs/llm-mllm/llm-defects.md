# 大模型典型缺陷

> 幻觉、灾难性遗忘、模型坍塌 —— 大模型落地的三大核心缺陷，统一采用「原理 → 评估 → 应对」三段式分析。

---

## 目录

1. [概述](#1)
2. [幻觉 (Hallucination)](#2-hallucination)
3. [灾难性遗忘 (Catastrophic Forgetting)](#3-catastrophic-forgetting)
4. [模型坍塌 (Model Collapse)](#4-model-collapse)
5. [实践建议](#5)
6. [参考文献](#6)

---

## 1. 概述

大模型在落地过程中面临三大核心缺陷，分别对应生成质量、持续学习能力、数据生态三个维度：

- **幻觉（Hallucination）**：生成内容与事实或给定上下文不一致，影响可信度。
- **灾难性遗忘（Catastrophic Forgetting）**：学习新任务 / 新数据后旧任务性能骤降，影响通用能力。
- **模型坍塌（Model Collapse）**：用模型生成的数据递归训练导致输出质量退化、多样性丧失，威胁数据生态。

这三个缺陷常作为面试中的「大模型缺陷三连问」，按「定义 → 评估指标 → 应对策略」三步作答，既显体系又显深度。

---

## 2. 幻觉 (Hallucination)

### 2.1 定义与分类

**幻觉**是指模型生成的内容与事实（factuality）、给定来源（faithfulness）或用户指令不一致，表现为「自信地胡说八道」。

幻觉的分类维度：

| 分类维度 | 类型 | 含义 |
|---------|------|------|
| 按内容 | 事实性幻觉（Factuality Hallucination） | 与真实世界事实冲突 |
| 按内容 | 忠实性幻觉（Faithfulness Hallucination） | 与提供的上下文 / 指令冲突（RAG 场景尤甚） |
| 按来源 | 内在幻觉（Intrinsic） | 与上下文内信息自相矛盾 |
| 按来源 | 外在幻觉（Extrinsic） | 超出上下文 / 外部知识，无依据编造 |

### 2.2 成因分析

幻觉的产生贯穿模型的全生命周期，可从以下六个层面理解：

**数据层面**：训练语料含噪声、信息过时、存在偏见、重复内容过多。模型在预训练阶段学到的不准确信息会被记忆并放大。

**训练目标层面**：语言模型仅最大化似然（next-token prediction），不强制「真」。模型学会的是「生成看似合理的文本」而非「陈述事实」。训练目标本身没有真值信号。

**知识边界层面**：模型对自身知识的边界缺乏感知，面对未知问题时仍「硬答」而非拒绝回答。这源于模型没有内在的「知道 / 不知道」校准机制。

**解码策略层面**：采样温度过高引入随机性，导致生成偏离；贪婪解码则可能固化某些错误模式。Top-$p$ / Top-$k$ 采样虽然提升多样性，但也可能引入幻觉。

**上下文利用层面**：长上下文场景下存在「中间丢失」（lost-in-the-middle）现象，模型对上下文中间位置的检索片段关注度不足，导致忠实性幻觉。

**推理层面**：模型在推理链中可能引入错误假设，且错误会沿推理链传播和放大。

### 2.3 评估指标

| 基准 / 指标 | 测什么 | 评测形式 |
|------------|--------|---------|
| **TruthfulQA** (Lin 2021) | 是否传播常见误解 | 选择题（真假判断） |
| **HaluEval** (Li 2023) | 幻觉倾向检测 | 有 / 无幻觉二分类 |
| **FEVER** | 事实核查 | 蕴含 / 中立 / 矛盾 |
| **FactScore** (Min 2023) | 实体级事实分解 | 逐原子事实支持率 |
| **RAGAS** (Es 2023) | RAG 忠实度 / 上下文利用 | faithfulness、context precision |
| **SelfCheckGPT** (Manakul 2023) | 无参考自一致性 | 多采样一致性估不可靠度 |
| **FaithEval** | 忠实性细粒度评估 | 声明级支撑判断 |

**核心指标定义**：

忠实性（Faithfulness）：生成内容中有证据支撑的声明所占比例。通常用 NLI 模型或 LLM-judge 判断每个声明是否被上下文支撑。

$$F = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(\text{claim}_i\text{ 被上下文支撑})$$

事实性（Factuality）：与权威知识库一致的断言所占比例。

一致性（Consistency）：多次采样结果是否稳定。SelfCheckGPT 通过比较多次采样的不一致程度来估计幻觉风险。

### 2.4 应对策略

**RAG（检索增强生成）**：将可查证的外部来源拼接进上下文，为模型提供事实依据，显著降低事实性幻觉。RAG 是当前最直接有效的幻觉缓解手段，但需注意检索质量本身也会引入忠实性幻觉。

**事实性对齐**：在 RLHF / DPO 训练中加入「事实性奖励」信号。例如，RLHF-V 使用事实校验信号作为奖励，factuality-aware DPO 在偏好数据中标注事实性维度。目标是让模型在最大化人类偏好的同时内化「真」的约束。

**自反思与自一致性**：

- **Self-Consistency**：对同一问题采样多条推理路径，取多数投票结果。
- **SelfCheckGPT**：多次采样后比较各份结果，识别不可靠段落。
- **Chain-of-Verification (CoVe)**：先生成初始回答，再规划并执行验证问题，最后修正。

**引用溯源（Citation）**：要求模型输出时附带出处，便于人工或自动校验。Self-RAG 框架通过 `IsSupported` 等特殊令牌实现生成时自检。

**受控解码（Controlled Decoding）**：通过 Contrastive Decoding 或 Constrained Decoding 降低「高概率但错误」词的概率，引导模型生成更可靠的内容。

**事后校验**：接入知识图谱、计算器、搜索引擎等工具，对生成内容中的关键事实进行二次验证。

---

## 3. 灾难性遗忘 (Catastrophic Forgetting)

### 3.1 定义与本质

**灾难性遗忘**是指模型在学习新任务或新数据后，旧任务上的性能大幅下降。其本质是神经网络参数共享导致的「稳定性-可塑性」困境（Stability-Plasticity Dilemma）：

- **稳定性（Stability）**：保留旧知识的能力；
- **可塑性（Plasticity）**：学习新知识的能力。

两者天然矛盾：要学新（需要可塑性）就不能太僵（需要稳定性）。全参微调时新梯度覆盖旧知识，遗忘最为严重。

在 LLM 场景中，灾难性遗忘主要体现在：

- **SFT 微调过度**：领域微调后通用对话能力下降；
- **持续学习（Continual Learning）**：依次学习多个任务时，后学任务覆盖先学任务；
- **对齐训练**：RLHF 对齐后部分能力退化（对齐税）。

### 3.2 评估指标

**遗忘量（Forgetting）**：任务 $k$ 上相对历史最佳性能的平均下降。

$$F_k = \max_{t < k} a_{k,t} - a_{k,k}$$

其中 $a_{k,t}$ 表示在学习完任务 $t$ 后，任务 $k$ 上的性能。

**后向迁移（Backward Transfer, BWT）**：学完所有任务后，新任务对旧任务的整体影响（正值表示正向迁移）。

$$\mathrm{BWT} = \frac{1}{T-1}\sum_{j=1}^{T-1} (a_{T,j} - a_{j,j})$$

**平均精度保留率**：旧任务微调后的精度除以微调前的初始精度，衡量通用能力的保留程度。

### 3.3 应对策略

| 路线 | 核心思想 | 代表方法 |
|------|---------|---------|
| **回放（Replay）** | 混合旧任务样本再训练 | Experience Replay (Robins 1995) |
| **正则化（Regularization）** | 保护重要参数不被大幅修改 | EWC (Kirkpatrick 2017)、L2SP (Li 2018)、SI (Zenke 2017) |
| **参数隔离（Parameter Isolation）** | 冻结主干，只训练旁路 | LoRA、Adapter、Prefix Tuning |
| **知识蒸馏（Distillation）** | 旧模型输出约束新模型 | LwF (Li & Hoiem 2017) |
| **Prompt 方法** | 可训前缀隔离任务 | L2P、DualPrompt (Wang 2022) |

**EWC（弹性权重巩固）**：核心思想是用 Fisher 信息矩阵 $F$ 估计每个参数对旧任务的重要性，重要的参数在后续训练中改动受罚。

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2}\sum_i F_i (\theta_i - \theta_i^*)^2$$

其中 $\theta_i^*$ 是旧任务训练后的最优参数，$F_i$ 是 Fisher 信息矩阵的对角元素，$\lambda$ 控制遗忘惩罚强度。

**LLM 微调实践建议**：首选 **PEFT（LoRA）+ 小学习率 + 混合原分布数据 + 少量回放样本**。LoRA 等参数高效方法天然具有参数隔离效果，能最大程度保留预训练能力。

---

## 4. 模型坍塌 (Model Collapse)

### 4.1 定义与机制

**模型坍塌**是指在模型自生成数据上递归训练后，生成分布逐渐偏离真实分布，导致尾部分布信息丢失、方差坍缩、多样性退化的现象。

Shumailov et al. (2023, 2024) 在 Nature 上系统阐述了这一现象的机制：

1. **统计近似误差**：每次用模型生成数据来估计真实分布时，都会引入统计近似误差。
2. **误差累积**：递归训练使得误差在代际间累积，低概率事件被逐步遗忘。
3. **分布窄化**：长程递归下，分布不断变窄，最终生成内容高度同质化，尾部（稀有事件）完全消失。

模型坍塌与以下概念的区别：

- **模式崩溃（Mode Collapse）**：GAN 训练中的经典问题，指生成器只产生少数几种模式 —— 是单次训练的优化问题，而非代际递归退化。
- **数据污染（Data Contamination）**：训练数据中包含测试数据造成的评估偏差 —— 是评测问题，而非生成质量退化。

### 4.2 评估指标

- **分布距离**：与原始数据分布的 KL 散度 / JS 散度上升，衡量整体分布偏移。
- **长尾保留率**：稀有事件在生成数据中的占比下降幅度，衡量尾部信息丢失。
- **多样性指标**：$n$-gram 多样性、生成长度方差、嵌入层表示的多样性。
- **困惑度与人类评价**：可读性（perplexity）可能仍高甚至更优，但信息量、新颖度退化 —— 这是模型坍塌的隐蔽性所在。

### 4.3 应对策略

**保留真实数据**：合成数据仅作为增强手段，真实人类数据始终占训练数据的主体。这是最根本的防线。

**混合比例控制**：设定合成数据与真实数据的上限比例（如合成数据不超过 10%-30%），具体阈值取决于任务和领域。

**合成数据质量管控**：

- 去重：去除合成数据中的重复内容。
- 质量过滤：用分类器或评分模型过滤低质量生成。
- 多样性筛选：确保合成数据覆盖足够的语义多样性，避免同质化生成物入库。

**水印与溯源（Watermarking）**：给机器生成的内容打上水印，在构建训练数据时识别并降权或剔除合成内容。

**持续注入新鲜真实数据**：构建「真实数据闭环」，定期用新采集的真实人类语料刷新训练集，打破递归退化循环。

---

## 5. 实践建议

### 5.1 三大缺陷的权衡

| 场景 | 主要风险 | 建议策略 |
|------|---------|---------|
| 领域微调（如医疗、法律） | 遗忘 > 幻觉 | LoRA + 小学习率 + 通用数据混合 |
| 知识密集型问答 | 幻觉 > 遗忘 | RAG + 事实性对齐 + 引用溯源 |
| 合成数据训练 | 坍塌 > 幻觉 > 遗忘 | 真实数据为主 + 水印过滤 + 多样性控制 |
| 持续学习（多任务依序） | 遗忘 + 坍塌 | EWC + 回放 + 参数隔离 |

### 5.2 部署前评测清单

在模型上线前，建议对以下维度进行系统评测：

- **幻觉评测**：用 TruthfulQA / HaluEval 测事实性，用 RAGAS 测 RAG 场景忠实度。
- **遗忘评测**：对比微调前后通用能力 benchmark（如 MMLU、C-Eval）的分数变化。
- **坍塌评测**：检查合成数据训练后长尾生成能力、输出多样性是否退化。

### 5.3 综合防御策略

最优实践是「三管齐下」：

1. **RAG**：降低幻觉，提供可验证的证据来源。
2. **对齐**：通过 RLHF / DPO 内化真实性约束，减少有害输出。
3. **数据策略**：保持真实数据占比，控制合成数据比例，防御模型坍塌。

三者不是替代关系，而是互补关系：RAG 解决「不知道」，对齐解决「不乱说」，数据策略解决「不退化」。

---

## 6. 参考文献

[1] L. Huang, W. Yu, W. Ma, et al., *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions*, arXiv preprint, 2023.

[2] Z. Ji, N. Lee, R. Frieske, et al., *Survey of Hallucination in Natural Language Generation*, ACM Computing Surveys, 2023.

[3] P. Manakul, A. Liusie, M. J. F. Gales, *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models*, EMNLP, 2023.

[4] S. Min, K. Krishna, X. Lyu, et al., *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*, EMNLP, 2023.

[5] S. Es, J. James, L. Espinosa-Anke, S. Schockaert, *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, arXiv preprint, 2023.

[6] S. Lin, J. Hilton, O. Evans, *TruthfulQA: Measuring How Models Mimic Human Falsehoods*, ACL, 2022.

[7] M. McCloskey, N. J. Cohen, *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem*, Psychology of Learning and Motivation, 1989.

[8] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, et al., *Overcoming Catastrophic Forgetting in Neural Networks*, PNAS, 2017.

[9] A. Robins, *Catastrophic Forgetting, Rehearsal and Pseudorehearsal*, Connection Science, 1995.

[10] I. Shumailov, Z. Shumaylov, Y. Zhao, et al., *The Curse of Recursion: Training on Generated Data Makes Models Forget*, arXiv preprint, 2023.

[11] I. Shumailov, Z. Shumaylov, Y. Zhao, et al., *AI Models Collapse When Trained on Recursively Generated Data*, Nature, 2024.

[12] S. Alemohammad, J. Casco-Rodriguez, L. Luzi, et al., *Self-Consuming Generative Models Go MAD*, NeurIPS, 2024.

[13] Z. Li, D. Hoiem, *Learning without Forgetting*, IEEE TPAMI, 2017.

[14] F. Zenke, B. Poole, S. Ganguli, *Continual Learning Through Synaptic Intelligence*, ICML, 2017.

[15] Z. Wang, Z. Zhang, C.-Y. Lee, et al., *Learning to Prompt for Continual Learning*, CVPR, 2022.