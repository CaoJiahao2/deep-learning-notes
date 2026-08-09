# LLM 评测体系详解

> 大模型能力需系统化度量：从自动基准到 LLM-as-Judge，从数据污染到基准饱和，构建可靠的评测体系。

---

## 目录

1. [概述](#1)
2. [能力维度与代表基准](#2)
3. [LLM-as-Judge](#3-llm-as-judge)
4. [评测框架与工具](#4)
5. [评测陷阱](#5)
6. [实践建议](#6)
7. [参考文献](#7)

---

## 1. 概述

### 1.1 为什么需要系统化评测

大模型的能力边界不断扩展，但「能力强」是一个模糊的概念。系统化评测的核心目标是将模型能力转化为可量化、可比较、可复现的指标，从而回答三个关键问题：

- **能力有多强？** 在特定任务上，模型的准确率 / 得分是多少？
- **和谁比？** 与竞品模型、上一版本、人类的差距有多大？
- **可信吗？** 评测结果是否真实反映模型能力，还是被数据污染、基准过拟合等噪声干扰？

### 1.2 评测的分类

评测方法按自动化程度可分为两大类：

| 类型 | 特点 | 代表 |
|------|------|------|
| **自动基准** | 客观、可复现、规模化 | MMLU, GSM8K, HumanEval |
| **人工 / LLM 裁判** | 处理主观、开放任务 | MT-Bench, Chatbot Arena |

**自动基准** 依赖固定数据集和标准化指标（准确率、F1、pass@k 等），适合大规模横向对比。**人工 / LLM 裁判** 面向难以用固定答案衡量的任务（如创意写作、多轮对话、指令遵循），通常由人类标注者或强模型作为裁判进行打分。

### 1.3 核心原则

- **多基准交叉印证**：不只看单一分数，模型可能在某个基准上表现好但泛化差。需要覆盖知识、推理、代码、对齐等多个维度，交叉验证。
- **区分「能力」与「对齐」指标**：能力指标衡量模型「知道什么、能做什么」（如 MMLU、GSM8K），对齐指标衡量模型「回答得好不好、是否符合人类偏好」（如 MT-Bench、Arena Elo）。两者不可混淆。
- **评测设置透明化**：prompt、few-shot 示例、temperature、解码策略等都会显著影响分数，报告时必须完整记录。

---

## 2. 能力维度与代表基准

### 2.1 能力-基准全景表

| 维度 | 代表基准 | 任务形式 | 指标 | 说明 |
|------|---------|---------|------|------|
| **知识** | MMLU | 4 选 1 选择题 | Accuracy | 57 个学科，从初等到专业 |
| | C-Eval | 4 选 1 选择题 | Accuracy | 中文多学科，52 个领域 |
| | CMMLU | 4 选 1 选择题 | Accuracy | 中文多学科，67 个主题 |
| | TriviaQA | 问答 | F1 / Exact Match | 百科知识型问答 |
| **推理** | GSM8K | 数学应用题 | 最终答案准确率 | 小学数学，8.5K 题，需 CoT 推理 |
| | MATH | 竞赛数学 | 答案准确率 | AMC/AIME 级别，需符号推理 |
| | BBH | 多选题 + 生成 | Accuracy | 23 个 BIG-Bench 难任务 |
| | LogiQA | 选择题 | Accuracy | 逻辑推理，中英双语 |
| | GPQA | 选择题 | Accuracy | 研究生级科学推理，专家验证 |
| **代码** | HumanEval | 函数补全 | pass@k | 164 题，单测验证 |
| | MBPP | 函数补全 | pass@k | 974 题，含描述 |
| | LiveCodeBench | 编程题 | pass@k | 定期更新，防污染 |
| | DS-1000 | 数据分析 | pass@k | Pandas/NumPy 等库 |
| **对齐** | MT-Bench | 多轮对话 | LLM-Judge 1-10 分 | GPT-4 裁判，8 类问题 |
| | AlpacaEval | 单轮指令 | Win Rate vs GPT-4 | 长度控制版 LC |
| | Arena-Hard | 单轮指令 | Win Rate | 500 个高难度 prompt |
| | Chatbot Arena | 人工对战 | Elo 分 | 众包比较，真实偏好 |
| **真实性** | TruthfulQA | 选择题 + 生成 | Accuracy / Truthfulness | 测「不说谎」，对抗常见误解 |
| | HaluEval | 二元分类 | Accuracy | 幻觉检测数据集 |
| | RealTimeQA | 问答 | Accuracy | 实时知识，防过时记忆 |
| **多模态** | MMBench | 选择题 | Accuracy | 20 个能力维度 |
| | MME | 是/否判断 | Accuracy | 14 个感知+认知子任务 |
| | POPE | 是/否判断 | Accuracy / F1 | 测视觉幻觉 |
| | MMMU | 选择题 | Accuracy | 多学科多模态理解 |

### 2.2 知识维度

MMLU（Massive Multitask Language Understanding）是当前最广泛使用的大模型知识基准，覆盖 57 个学科领域，从初等数学到专业法律。每个领域约 100-200 题，总题量约 14K。评测方式为 4 选 1 选择题，指标为 Accuracy。

C-Eval 和 CMMLU 是中文知识评测的代表。C-Eval 覆盖 52 个学科，包含中学、大学、职业考试三个难度级别；CMMLU 则覆盖 67 个主题，更侧重中国文化和语境下的知识。

### 2.3 推理维度

GSM8K 是小学数学应用题的经典基准，约 8.5K 道题，考察模型的多步推理能力。评测时通常使用 Chain-of-Thought (CoT) prompting，让模型逐步推导后再给出答案。最终答案准确率是核心指标。

MATH 难度更高，题目来自 AMC、AIME 等数学竞赛，需要较强的符号推理和公式推导能力。它与 GSM8K 互为补充，前者测基础推理，后者测高阶数学能力。

BBH（BIG-Bench Hard）从 BIG-Bench 的 200+ 任务中筛选出 23 个对当时模型来说「困难」的任务，涵盖逻辑推理、算法、数学等方向。

GPQA（Graduate-Level Google-Proof Q&A）是近年来更难的推理基准，题目由领域专家（PhD 级别）编写，经过「Google-Proof」验证——即无法通过搜索引擎简单检索到答案，防止模型死记硬背。

### 2.4 代码维度

HumanEval 是代码生成的标准基准，包含 164 个 Python 编程题，每个题目提供函数签名和 docstring，要求模型补全函数体。评测指标为 pass@k：对每个题目生成 k 个候选，只要有 1 个通过所有单测即算正确。

```text
pass@k = E[1 - C(n-c, k) / C(n, k)]

其中 n = 总生成数, c = 正确数, C = 组合数
```

MBPP 规模更大，包含 974 个入门级 Python 题，带有自然语言描述。LiveCodeBench 则通过定期从 LeetCode、AtCoder 等平台收集新题，缓解数据污染问题。

### 2.5 对齐维度

对齐评测关注模型回答是否符合人类偏好：有用性、无害性、诚实性。MT-Bench 设计了 8 类共 80 个多轮对话问题，由 GPT-4 作为裁判对回答进行 1-10 分打分。每个问题进行两轮对话，考察模型的上下文理解和持续对话能力。

Chatbot Arena（LMSYS）采用众包对战模式：用户随机向两个匿名模型提问并投票，胜率通过 Elo 评分系统聚合。至今已收集数百万次人类偏好投票，是目前最接近「真实用户体验」的评测方式。

AlpacaEval 和 Arena-Hard 则是基于 LLM-Judge 的自动化对齐评测，前者用 GPT-4 与参考模型对比，后者筛选了 500 个高难度 prompt 以提升区分度。

### 2.6 真实性与多模态

TruthfulQA 专门测试模型是否会产生「以假乱真」的错误回答，题目设计基于人类常见的误解和谬误。HaluEval 则聚焦于幻觉检测，要求模型判断给定回答中是否存在虚构内容。

多模态评测方面，MMBench 和 MME 是综合感知 + 认知能力的基准，POPE 专门检测视觉幻觉（如图像中存在但模型编造出不存在物体），MMMU 则覆盖 30 个学科的大学级多模态推理。

---

## 3. LLM-as-Judge

### 3.1 核心思想

对于开放式生成任务，无法用固定答案判断对错。LLM-as-Judge 使用一个强模型（通常是 GPT-4 或 Claude）作为裁判，对候选回答进行打分或比较。这已成为 LLM 评测的主流范式，MT-Bench、AlpacaEval、Arena-Hard 等均采用此方法。

常见的评判模式：

```text
1. 单回答打分（Single Answer Grading）
   Judge 对单个回答给出 1-10 分的评分

2. 成对比较（Pairwise Comparison）
   Judge 比较两个回答，输出胜/负/平

3. 参考对比（Reference-based）
   Judge 比较候选回答与参考回答的质量
```

### 3.2 已知偏置与缓解方法

LLM-as-Judge 虽高效，但存在多种系统性偏置，需在评测设计中主动应对：

**位置偏置（Position Bias）**：Judge 倾向于偏好某个位置的回答（通常是第一位）。缓解方法：交换 A/B 顺序再评，取两次结果的平均。MT-Bench 默认采用此策略。

**长度偏置（Length Bias）**：Judge 倾向给更长的回答更高分，即使内容质量相同。AlpacaEval 2.0 引入长度控制（Length-Controlled）版本，通过回归模型消除长度影响。

**风格偏置（Style Bias）**：Judge 偏好更礼貌、更正式或更「自信」的表述风格，而非内容准确性。解决方案是优化 prompt 模板，明确要求 Judge 忽略风格，聚焦内容质量。

**自我增强偏置（Self-Enhancement Bias）**：Judge 倾向偏好自己生成的回答（或同家族模型）。缓解方法是使用不同模型家族的 Judge 交叉验证。

### 3.3 校准与验证

Judge 分数必须与人工评分进行校准，以确保其有效性：

```text
Pearson / Spearman 相关系数
  衡量 Judge 打分与人类打分的相关性

Human Agreement
  以人类之间的一致性作为上界，Judge 与人类的
  一致性应接近此上界
```

MT-Bench 的实验表明，GPT-4 作为 Judge 与人类的一致性超过 80%，接近人类之间的一致性（约 82%），验证了 LLM-as-Judge 的可行性。但这一结论在不同领域和 Judge 模型上可能不成立，使用前需验证。

### 3.4 Chatbot Arena 与 Elo 评分

Chatbot Arena 是 LMSYS 组织维护的众包对战平台。用户输入任意 prompt，系统随机返回两个匿名模型的回答，用户投票选择更好的一个。所有对战结果通过 Elo 评分系统聚合：

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}
$$

其中 $R_A$ 和 $R_B$ 是两个模型的 Elo 分，$E_A$ 是模型 A 的预期胜率。实际对战结果更新分数的公式为：

$$
R_A' = R_A + K \cdot (S_A - E_A)
$$

其中 $S_A$ 为实际结果（胜=1, 平=0.5, 负=0），$K$ 为更新系数。Arena 还引入了 Bradley-Terry 模型和置信区间，提供更稳健的排名估计。

Chatbot Arena 的优势在于：覆盖真实用户分布、多样化 prompt、持续更新。局限在于：用户群体偏技术，可能不代表所有使用场景；且无法控制 prompt 难度分布。

---

## 4. 评测框架与工具

### 4.1 主流评测框架

| 框架 | 维护方 | 特点 |
|------|--------|------|
| **lm-eval-harness** | EleutherAI | 最广泛使用，支持 200+ 基准，HuggingFace 集成 |
| **OpenCompass** | 上海 AI Lab | 中文评测全面，社区活跃，可视化报告 |
| **HELM** | Stanford CRFM | 多维度全景评测，透明化方法论 |
| **BigCode-Eval-Harness** | BigCode | 代码评测专用，多语言支持 |
| **AlpacaEval** | Stanford | 自动化 LLM-judge 评测，长度控制 |
| **MT-Bench** | LMSYS | 多轮对话评测，GPT-4 裁判 |

### 4.2 lm-eval-harness 使用示例

```python
# 安装
# pip install lm-eval

# 命令行评测
# lm_eval --model hf \
#   --model_args pretrained=mistralai/Mistral-7B-v0.1 \
#   --tasks mmlu,gsm8k,hellaswag \
#   --batch_size auto \
#   --output_path ./results

# Python API 评测
from lm_eval import simple_evaluate

results = simple_evaluate(
    model="hf",
    model_args="pretrained=mistralai/Mistral-7B-v0.1",
    tasks=["mmlu", "gsm8k"],
    num_fewshot=5,
    batch_size="auto",
)
print(results["results"]["mmlu"]["acc,none"])
```

### 4.3 HELM 全景评测

HELM（Holistic Evaluation of Language Models）由 Stanford CRFM 提出，核心理念是透明化评测。它不仅报告最终分数，还公开：

- 完整的评测 prompt 和场景配置
- 每个样本的预测结果
- 模型可能存在的偏置和局限
- 多维度指标（准确率、校准、公平性、效率、偏置等）

HELM 的评测维度矩阵：

```text
场景（Scenario）→ 指标（Metric）
  ├─ 知识       → 准确率, 校准
  ├─ 推理       → 鲁棒性, 公平性
  ├─ 生成       → 效率, 偏置
  └─ 交互       → 毒性, 信息危害
```

### 4.4 数据污染检测

数据污染（Data Contamination）是指评测集样本出现在训练数据中，导致模型「作弊」而非真正推理。检测方法包括：

**n-gram 重叠检测**：将评测样本按 13-gram 分词，检索训练集中是否存在完全相同或高度重叠的片段。简单高效，但无法检测改写或翻译后的污染。

**困惑度异常检测**：被污染的样本在模型下的困惑度（perplexity）通常显著低于同类干净样本。通过比较评测样本与类似难度样本的困惑度分布，可发现异常。

**留痕探测（Canary Probing）**：在训练数据中故意插入特殊字符串（canary），评测时检测模型是否「记住」了这些字符串。这是最直接的检测手段，但需要在训练数据构建时预埋。

**推荐实践**：发布评测结果时，同时报告数据污染分析结果。使用多个检测方法交叉验证，并说明污染检测的阈值和细节。

---

## 5. 评测陷阱

### 5.1 数据污染

这是 LLM 评测中最严重的系统性风险。当评测集被混入训练数据时，模型可以在测试时「背诵」答案而非进行推理，导致分数虚高。

典型表现：
- 模型在某个基准上突然大幅提升，但真实体验无变化
- 模型在改写后的相同题目上表现骤降
- 困惑度分析显示模型对评测样本异常「熟悉」

案例：多个模型在 GSM8K 上的表现曾因数据污染被质疑，后经 n-gram 检测发现部分测试样本确实出现在训练集中。

### 5.2 过拟合基准

为在特定基准上刷分，部分团队会进行针对性优化：选择特定 prompt 模板、调参 few-shot 示例、甚至针对基准进行微调。这种「应试」优化可能导致模型在其他任务上泛化性能下降。

```text
Goodhart 定律：当一个指标成为目标，它就不再是一个好的指标。
```

缓解方法：使用多个独立基准交叉验证；关注模型在未见过的评测集上的表现；使用「留出集」（held-out set）进行最终验证。

### 5.3 基准饱和

随着模型能力提升，部分基准已被「饱和」——顶级模型表现接近天花板，无法进一步区分模型优劣。

| 基准 | 饱和状态 | 替代方案 |
|------|---------|---------|
| MMLU | GPT-4 已达 ~86.4%，接近人类专家 | GPQA, MMMU |
| GSM8K | 多个模型已达 95%+ | MATH, FrontierMath |
| HumanEval | 多个模型已达 90%+ pass@1 | LiveCodeBench, SWE-bench |
| HellaSwag | 饱和 | 不再作为核心区分指标 |

饱和后的基准仍有价值（作为「及格线」检测），但不应作为核心区分指标。评测社区需要持续开发更难、更多样的基准。

### 5.4 LLM-Judge 系统性偏置

除 3.2 节讨论的位置、长度、风格偏置外，还需注意：

- **领域偏置**：Judge 在特定领域（如数学、代码）的判断能力可能不足，导致评分不可靠
- **语言偏置**：非英语评测中，Judge 的可靠性可能下降
- **prompt 敏感性**：Judge 对不同 prompt 模板可能给出不同结果，需固定模板并报告

### 5.5 评测复现性

即使使用相同基准和模型，不同团队的评测结果也可能不一致，原因包括：

- prompt 模板差异（少量措辞变化可能影响结果）
- few-shot 示例的选择和顺序
- 解码参数（temperature, top-p, top-k）
- 评测框架的实现差异（如 normalization 方式、评测指标计算）

建议参考 HELM 的透明化方法论，完整记录评测配置，鼓励社区采用统一的评测协议。

---

## 6. 实践建议

### 6.1 多基准交叉印证

```text
推荐的最小评测矩阵：

知识  → MMLU + C-Eval（中文）
推理  → GSM8K + MATH + BBH
代码  → HumanEval + MBPP
对齐  → MT-Bench + AlpacaEval 2.0 LC
真实  → TruthfulQA
```

不要只看一个分数——模型在 GSM8K 上 95% 但 MATH 上 30%，说明基础推理强但高阶推理弱。这种「能力剖面」比单一分数更有信息量。

### 6.2 报告评测设置

完整报告以下信息以确保可复现：

```text
- 模型版本和 checkpoint
- 评测框架和版本
- Prompt 模板（完整文本）
- Few-shot 示例数量和内容
- 解码参数（temperature, top-p, top-k, max_tokens）
- 评测指标（acc, pass@k, F1, etc.）
- 数据污染检测结果
- 是否使用 CoT / self-consistency
```

### 6.3 关注评测 vs 实际体验的差距

评测分数高不代表实际体验好。可能的原因：

- 评测集覆盖不全，忽略了模型的实际短板
- 基准过拟合导致评测分数虚高
- 评测场景（如选择题）与真实使用场景（如开放式对话）有差异

建议结合定量评测和定性分析：在评测之外，进行人工 A/B 测试、构建典型场景的测试用例、收集用户反馈。

### 6.4 构建评测体系

对于需要持续跟踪模型能力的团队，建议建立分层评测体系：

```text
第一层：快速评测（每次训练后）
  MMLU + HumanEval + GSM8K（10 分钟内完成）

第二层：完整评测（每周/关键 checkpoint）
  全维度基准 + MT-Bench + AlpacaEval

第三层：深度评测（发布前）
  全维度 + 数据污染分析 + 人工评估 + A/B 测试
```

### 6.5 前沿趋势

- **Agentic 评测**：随着 AI Agent 的兴起，评测从单轮问答转向多步任务完成（如 SWE-bench、WebArena）。
- **动态评测**：基准定期更新以避免数据污染（LiveCodeBench, RealTimeQA）。
- **多模态评测**：从纯文本扩展到图文、视频、音频等多模态评测。
- **安全性评测**：红队测试、越狱检测、有害内容生成等安全维度日益重要。

---

## 7. 参考文献

[1] Hendrycks, D., et al. *Measuring Massive Multitask Language Understanding*. ICLR, 2021. (MMLU)

[2] Cobbe, K., et al. *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168, 2021. (GSM8K)

[3] Chen, M., et al. *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374, 2021. (HumanEval)

[4] Lin, S., et al. *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. ACL, 2022.

[5] Zheng, L., et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS, 2023. (MT-Bench / Chatbot Arena)

[6] Liang, P., et al. *Holistic Evaluation of Language Models*. TMLR, 2023. (HELM)

[7] Srivastava, A., et al. *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models*. TMLR, 2023. (BIG-Bench)

[8] Suzgun, M., et al. *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them*. ACL Findings, 2023. (BBH)

[9] Rein, D., et al. *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*. arXiv:2311.12022, 2023.

[10] Li, X., et al. *AlpacaEval: An Automatic Evaluator of Instruction-following Models*. GitHub, 2023.

[11] Tian, Y., et al. *Arena-Hard: A Better Automatic Evaluator for LLM Chatbots*. arXiv:2406.11939, 2024.

[12] Huang, Y., et al. *C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models*. NeurIPS, 2023.

[13] Li, H., et al. *CMMLU: Measuring Massive Multitask Language Understanding in Chinese*. arXiv:2306.09212, 2023.

[14] Jain, N., et al. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code*. arXiv:2403.07974, 2024.

[15] Yue, X., et al. *MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI*. CVPR, 2024.
