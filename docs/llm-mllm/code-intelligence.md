# 代码大模型与程序推理

> 从 Codex 到 SWE-bench，从 Fill-in-the-Middle 到 Agentic Coding —— 代码是 LLM 商业化最成熟的落地场景。

---

## 目录

1. [概述](#1)
2. [代码预训练](#2)
3. [代码补全与 Fill-in-the-Middle](#3-fill-in-the-middle)
4. [代码指令微调与推理](#4)
5. [仓库级理解](#5)
6. [代码 Agent 与工具使用](#6-agent)
7. [评测基准](#7)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1. 概述

代码智能是大语言模型最早实现商业化落地的领域。从 2021 年 GitHub Copilot 基于 Codex 推出，到 2024-2025 年 Cursor、Devin、Claude Code 等 Agentic Coding 工具爆发，代码大模型已从"自动补全"演进为"自主解决软件工程任务"。

代码智能的独特性在于：

- **可验证性**：代码正确性可通过测试用例、编译、Lint 自动验证，为 RLVR 提供天然奖励信号。
- **结构化**：代码有明确的语法结构（AST）、类型系统和执行语义，便于形式化推理。
- **工具链丰富**：编译器、调试器、测试框架、版本控制构成完整的反馈闭环。
- **长上下文需求**：仓库级理解需要处理跨文件依赖，推动了长上下文和代码检索技术。

代码能力的技术演进路线：

$$
\text{代码补全} \rightarrow \text{对话式编程} \rightarrow \text{仓库级理解} \rightarrow \text{自主 Agent 编程}
$$

---

## 2. 代码预训练

### 2.1 代码数据

代码预训练数据主要来自公开代码仓库（GitHub），需经过：

- **去重**：Fork 仓库、模板代码、自动生成代码大量重复，需精确和近似去重。
- **许可证过滤**：移除 GPL 等强传染性许可证代码，避免法律风险。
- **质量过滤**：根据 Star 数、提交活跃度、测试覆盖率等指标过滤低质量仓库。
- **语言分布**：以 Python、JavaScript/TypeScript、Java、C/C++、Go 为主，合理配比。
- **多模态代码数据**：代码-注释对、Commit 信息、Issue、PR 描述、文档。

代表性代码数据集包括 The Stack（Kocetkov et al., 2022）、StarCoderData、CommitPack。

### 2.2 预训练目标

代码模型的预训练目标在标准因果语言建模基础上有特殊设计：

- **因果语言建模（CLM）**：与文本 LLM 相同，预测下一个 Token。代码数据约占总语料的 5%-20%。
- **代码感知分词器**：对缩进、括号、操作符等代码特有符号优化分词，减少 Token 数量并提升语义表示。
- **多语言代码训练**：在多种编程语言上训练，跨语言迁移提升整体代码能力。
- **注释-代码对齐**：利用文档字符串和注释建立自然语言与代码的对齐。

### 2.3 代表性代码模型

| 模型 | 机构 | 特点 |
|------|------|------|
| Codex | OpenAI | GPT-3 在代码上微调，Copilot 初代引擎 |
| CodeLlama | Meta | LLaMA2 代码特化版本，支持补全和对话 |
| StarCoder | HuggingFace | 6K 上下文、80+ 语言、FIM |
| DeepSeek-Coder | DeepSeek | 128K 上下文、仓库级预训练、Fill-in-Middle |
| CodeQwen | 阿里 | 多语言代码、长上下文 |
| Phi-3/Mini | Microsoft | 教科书级合成代码数据，小模型强代码能力 |

---

## 3. 代码补全与 Fill-in-the-Middle

### 3.1 代码补全任务

代码补全（Code Completion）是最基础的代码智能任务：给定光标前的代码（prefix），预测光标后的代码（suffix）。这与文本生成的关键区别在于：补全是"中间填空"而非"续写"——光标之后已有代码，补全内容需要与前后文一致。

### 3.2 Fill-in-the-Middle（FIM）

FIM（Bavarian et al., 2022）将代码补全形式化为填空任务：将代码在光标处拆分为 prefix 和 suffix，用特殊标记拼接后让模型填充中间部分（middle）：

```text
<prefix> 光标前代码
<suffix> 光标后代码
<middle> 模型预测的补全
```

训练时随机选择分割点将代码切分为 PSM（Prefix-Suffix-Middle）格式，让模型学习根据前后文填空。FIM 训练在不影响左到右生成能力的前提下，显著提升补全质量。

FIM 的三种格式：

- **PSM**：Prefix → Suffix → Middle（标准格式）。
- **PSM+SPM**：随机交换 PSM 和 SPM（Suffix → Prefix → Middle）增强鲁棒性。
- **CodeFill**：多光标填充，同时填充多个位置。

### 3.3 补全质量评估

- **Exact Match / Edit Similarity**：补全与参考代码的精确匹配率或编辑相似度。
- **Pass@$k$**：生成 $k$ 个补全，至少一个通过测试的概率：

$$
\text{Pass@}k = 1 - \binom{n-c}{k} / \binom{n}{k}
$$

其中 $n$ 为总采样数，$c$ 为通过测试的样本数。
- **用户接受率**：实际使用中补全被采纳的比例，是最核心的产品指标。

---

## 4. 代码指令微调与推理

### 4.1 代码 SFT

代码指令微调将基础代码模型对齐为代码助手：

- **指令数据**：编程问题与解答（如 APPS、CodeAlpaca）、Bug 修复、代码解释、重构。
- **执行反馈**：运行生成代码，用执行结果（编译错误、测试失败）作为反馈修正答案。
- **Chain-of-Thought 代码推理**：让模型先写解题思路和伪代码，再实现，提升复杂问题正确率。

### 4.2 测试时计算（Test-time Compute）

代码是测试时计算最自然的应用场景——生成多个候选方案，用编译和测试结果筛选正确答案：

- **Best-of-N**：生成 $N$ 个候选程序，运行测试，选择全部通过的。
- **Self-Debugging**：模型运行自己的代码，根据错误信息自行修复。
- **Reflexion（Shinn et al., 2023）**：将执行失败的反馈加入上下文，让模型反思并重试。
- **CodeR（Zhang et al., 2024）**：将代码执行与迭代修复集成到 RL 循环中。

测试时计算将代码生成从"一次生成"变为"生成-执行-修复"循环，大幅提升复杂任务通过率。

### 4.3 代码强化学习

代码任务的可验证性使其成为 RLVR（Reinforcement Learning with Verifiable Rewards）的理想场景：

- **编译器奖励**：代码能编译通过获得奖励。
- **测试用例奖励**：通过单元测试获得奖励。
- **代码风格奖励**：通过 Lint 检查获得额外奖励。
- **Edge case 奖励**：通过边界测试获得更高奖励。

DeepSeek-Coder、SWE-RL 等工作证明，在代码任务上执行 RL 能同时提升代码正确率和通用推理能力。

---

## 5. 仓库级理解

### 5.1 挑战

真实软件开发不是单文件任务，而是跨多个文件、模块和依赖的仓库级任务：

- **跨文件依赖**：理解函数定义、类继承、类型声明可能分散在不同文件中。
- **全局上下文**：代码修改需考虑项目约定、配置文件、构建系统。
- **检索定位**：在大型仓库中找到需要理解和修改的相关代码。

### 5.2 技术方案

- **仓库级预训练**：DeepSeek-Coder 等模型在预训练时将同一仓库的文件拼接训练（用仓库级分隔符），学习跨文件依赖。
- **代码 RAG**：用嵌入检索相关代码片段、类定义、文档。工具包括 Sourcegraph、LlamaIndex 的代码加载器。
- **代码知识图谱**：构建调用图、依赖图、类型继承图，支持结构化导航。
- **LSP 集成**：利用 Language Server Protocol 获取实时的类型信息、定义跳转、引用查找，为模型提供精确的代码结构信息。
- **长上下文窗口**：128K+ Token 窗口可直接塞入中型项目的核心文件。

### 5.3 仓库级补全与编辑

- **跨文件补全**：基于其他文件的定义和上下文补全当前文件。
- **代码编辑（Code Editing）**：模型输出 diff/patch 而非完整重写，精确修改代码。
- **Whole-context Completion**：将相关文件通过检索拼接进上下文，提供全局感知的补全。

---

## 6. 代码 Agent 与工具使用

### 6.1 从补全到 Agent

代码 Agent 将 LLM 从"建议者"升级为"执行者"，能够：

- 自主读取和浏览代码库。
- 运行命令和测试。
- 根据错误信息迭代修复。
- 创建分支、提交代码、发起 PR。

### 6.2 核心能力

| 能力 | 工具 | 作用 |
|------|------|------|
| 文件操作 | read/write/edit | 读取、创建、修改代码文件 |
| 命令执行 | bash/terminal | 编译、测试、运行脚本 |
| 代码搜索 | grep/rg/LSP | 定位相关代码和定义 |
| 版本控制 | git | 分支管理、提交、PR |
| 浏览器/文档 | web search/API docs | 查询文档和解决方案 |
| 调试器 | debugger | 断点、单步、变量检查 |

### 6.3 代表性系统

- **Cursor / GitHub Copilot Workspace**：IDE 集成的 AI 编程助手，支持多文件编辑和代码库问答。
- **Devin（Cognition, 2024）**：自主软件工程师，能端到端完成开发任务。
- **SWE-agent（Yang et al., 2024）**：专为 SWE-bench 设计的 Agent，自定义计算机交互界面。
- **OpenHands（原 OpenDevin）**：开源的自主软件工程 Agent 平台。
- **Claude Code / Codex CLI**：终端原生的 Agentic 编程工具，支持多轮编辑和命令执行。

### 6.4 Agent 设计模式

- **ReAct 循环**：推理 → 行动 → 观察 → 再推理，逐步推进。
- **计划-执行（Plan-and-Execute）**：先制定修复计划，再逐步执行，减少上下文混乱。
- **Map-Reduce 式修复**：将大任务拆解为子问题并行处理，最后合并。
- **Sandbox 执行**：在 Docker/VM 中安全执行代码，防止误操作。

---

## 7. 评测基准

| 基准 | 评测内容 | 形式 |
|------|---------|------|
| HumanEval | Python 函数级编程 | Pass@k |
| MBPP | 基础 Python 编程 | Pass@k |
| APPS | 竞赛级编程题 | 准确率 |
| CodeContests | Google 竞赛题 | Pass@k |
| DS-1000 | 数据科学代码 | Pass@k |
| MultiPL-E | 多语言代码 | Pass@k |
| SWE-bench | 真实 GitHub Issue 修复 | 解决率 |
| SWE-bench Lite | 精选 300 个高价值任务 | 解决率 |
| LiveCodeBench | 实时更新的竞赛题 | Pass@k（防污染） |
| Aider Polyglot | 多语言仓库级编辑 | 测试通过率 |
| BigCodeBench | 复杂代码生成（多库调用） | Pass@k |

**SWE-bench**（Jimenez et al., 2024）是当前最具挑战性的代码评测：模型需要在真实开源仓库中理解 Issue、定位相关代码、生成补丁并通过测试。它衡量的是端到端软件工程能力而非孤立的函数编写，已成为代码 Agent 的核心排行榜。

---

## 8. 实践建议

### 8.1 代码模型选型

- 单文件补全：轻量代码模型（CodeQwen、DeepSeek-Coder Base）+ FIM。
- 对话式编程助手：代码 SFT 模型（DeepSeek-Coder Chat、CodeLlama-Instruct）。
- 仓库级任务：长上下文模型 + 代码 RAG 检索。
- Agentic Coding：强推理模型（Claude/GPT-4 级别）+ 工具使用能力。

### 8.2 代码 Agent 工程建议

- 始终在沙箱中执行代码，限制文件系统和网络访问。
- 给 Agent 提供结构化的错误信息（编译错误、测试失败、堆栈跟踪），而非原始日志。
- 限制最大迭代轮数和 Token 预算，防止无限循环。
- 将大任务拆分为可验证的小步骤，每步运行测试确认进展。
- 使用 patch/diff 格式提交修改，便于审查和回滚。

### 8.3 数据与训练建议

- FIM 训练应同时用于基础模型预训练和微调，覆盖率建议 50%-80%。
- 代码 SFT 数据应包含修复迭代样本（错误代码 + 错误信息 + 修复代码），教会模型利用执行反馈。
- 代码 RL 的奖励函数应分层次：编译通过 → 部分测试通过 → 全部通过 → 代码质量。
- 关注许可证合规，商用代码模型需仔细过滤训练数据。

---

## 9. 参考文献

[1] M. Chen, J. Tworek, H. Jun, et al., *Evaluating Large Language Models Trained on Code*, arXiv preprint, 2021.

[2] B. Bavarian, H. Jun, N. Tezak, et al., *Efficient Training of Language Models to Fill in the Middle*, arXiv preprint, 2022.

[3] D. Kocetkov, R. Li, L. Ben Allal, et al., *The Stack: 3 TB of Permissively Licensed Source Code*, arXiv preprint, 2022.

[4] R. Li, L. Ben Allal, Y. Zi, et al., *StarCoder: May the Source Be with You!*, arXiv preprint, 2023.

[5] D. Guo, Q. Zhuo, Z. Liu, et al., *DeepSeek-Coder: When the Large Language Model Meets Programming*, arXiv preprint, 2024.

[6] C. E. Jimenez, J. Yang, A. Wettig, et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, ICLR, 2024.

[7] N. Shinn, F. Cassano, E. Berman, et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS, 2023.

[8] J. Yang, C. E. Jimenez, A. Wettig, et al., *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*, NeurIPS, 2024.

[9] A. M. Dai, A. M. Saxe, D. D. Johnson, et al., *CodeRL: Mastering Code Generation Through Pretrained Models and Deep Reinforcement Learning*, NeurIPS, 2022.

[10] F. F. Xu, U. Alon, G. Neubig, V. J. Hellendoorn, *A Systematic Evaluation of Large Language Models of Code*, ICLR, 2022.
