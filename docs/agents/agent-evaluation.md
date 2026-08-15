# Agent 评测与基准

> 从 AgentBench 到 τ-bench，从 SWE-bench 到 OSWorld —— 如何衡量一个 Agent 真的"会做事"。

---

## 目录

1. [概述](#1)
2. [评测维度](#2)
3. [通用 Agent 基准](#3)
4. [代码 Agent 基准](#4)
5. [工具调用与业务 Agent 基准](#5)
6. [Web / 浏览器 Agent 基准](#6)
7. [GUI / 设备 Agent 基准](#7)
8. [评测方法与陷阱](#8)
9. [实践建议](#9)
10. [参考文献](#10)

---

## 1 概述 {#1}

Agent 评测是 Agent 工程中最困难也最关键的环节。与单轮 LLM 评测（MMLU、GSM8K）不同，Agent 评测需要评估**多步决策、工具使用、环境交互与长期目标达成**——答案不是一个字符串，而是一段轨迹（trajectory）与最终结果。

Agent 评测的独特挑战：

- **环境依赖**：需要可执行的环境（代码仓库、浏览器、手机模拟器、桌面系统）；
- **成本高**：每次评测都要真实运行多步交互，昂贵且慢；
- **非确定性**：同一模型多次运行结果可能不同，需多次采样；
- **奖励信号**：过程正确 vs 结果正确，二者需分别衡量。

核心问题：**我们到底要衡量"模型想得对不对"还是"Agent 做得成不成事"？** 现代 Agent 评测越来越偏向后者的端到端任务完成率。

---

## 2 评测维度 {#2}

| 维度 | 衡量内容 | 典型指标 |
|------|---------|---------|
| 任务成功率 | 是否达成最终目标 | Success Rate、Pass@k |
| 步骤正确性 | 每一步是否合理 | Step Accuracy、工具调用正确率 |
| 效率 | 用多少步/多少 token 完成 | 步数、成本、延迟 |
| 鲁棒性 | 不同初始条件下的稳定性 | 多次运行方差、成功率均值 |
| 安全合规 | 是否触犯边界 | 违规次数、越权率 |
| 泛化性 | 未见过任务上的表现 | 新任务成功率 |

---

## 3 通用 Agent 基准 {#3}

### 3.1 AgentBench

AgentBench（Liu et al., 2023）是最早的通用 Agent 评测之一，覆盖 8 个真实环境：

- **操作系统**：终端命令操作（OS）
- **数据库**：SQL 查询与运维（DB）
- **知识图谱**：KGQA 与查询（KG）
- **数字卡牌游戏**：长期策略决策（CARD）
- **横向思考谜题**：Lateral Thinking Puzzles
- **网页浏览**：网页问答与导航（Web）
- **代码**：仓库级代码任务（Code）
- **家用**：交互式家居环境（House）

AgentBench 的结论：当时最强模型在需要长期规划、错误恢复的环境上表现普遍不佳，暴露了 LLM Agent 的"长程任务"短板。

### 3.2 GAIA

GAIA（Mialon et al., 2023）面向**通用助手能力**，设计 466 个"对人类容易、对 AI 困难"的问题，涉及：

- 多模态推理（图片、PDF、表格）；
- 信息检索与整合（搜索、查表、交叉验证）；
- 多步工具调用与推理链。

GAIA 强调"真实生产力任务"，答案需要从多个来源拼接。GAIA 全名 *General AI Assistants*，被视为衡量"通用数字助手"潜力的风向标。

### 3.3 τ-bench（Tau-Bench）

τ-bench（Yao et al., 2024）专门评测**面向用户的业务 Agent**（客服、购物、银行业务）：

- 模拟真实商业环境：用户、工具（数据库查询、下单、退款）、约束；
- 引入**价格敏感性**（用户在意价格）与**多目标**（正确、省时、合规）；
- 指标：任务成功率 + 代价（cost）+ 超额尝试次数；
- 强调 Agent 需要"主动澄清需求"而非机械执行。

τ²-bench（2025）进一步引入跨工具依赖与更真实的业务流。

---

## 4 代码 Agent 基准 {#4}

### 4.1 SWE-bench

SWE-bench（Jimenez et al., 2024）是代码 Agent 事实标准：给定真实 GitHub Issue，让 Agent 在完整仓库中定位问题、生成 patch，并通过隐藏测试验证。

- **SWE-bench Lite**：精选 300 个较易任务，广泛用于快速对比；
- **SWE-bench Verified**：OpenAI 人工复核的 500 个任务；
- 指标：Pass@1（一次尝试成功解决率）、解决率（Resolved %）。

### 4.2 SWE-bench Multimodal / Multilingual

- **SWE-bench Multimodal**（2024）：引入前端、可视化相关 Issue，需要模型理解图片/渲染效果；
- **SWE-bench Multilingual**：扩展到 Java、C++、Rust、TypeScript 等语言。

### 4.3 其他代码基准

- **HumanEval / MBPP**：函数级生成（见 `code-intelligence.md`）；
- **Aider Polyglot**：多语言仓库级编辑；
- **Terminal-Bench**：终端命令执行的 Linux 任务；
- **BigCodeBench**：复杂代码生成 + 多库调用。

---

## 5 工具调用与业务 Agent 基准 {#5}

### 5.1 ToolBench

ToolBench（Qin et al., 2023）构建大规模真实 API 工具集（RapidAPI），评测 Agent 的多工具选择、参数填充与多步组合能力。

### 5.2 API-Bank / BFCL

- **API-Bank**：从检索 API 到组合 API 的分级任务；
- **BFCL（Berkeley Function Calling Leaderboard）**：标准化函数调用评测，覆盖多模型、多工具、并行调用、错误处理。

### 5.3 企业/行业 Agent 基准

前沿已出现面向垂直行业的评测：金融（FinEvo-Bench）、GIS（GISAgentBench）、医疗、市场（Business Arena）等，强调**专业工具 + 领域知识 + 多步工作流**的复合能力。

---

## 6 Web / 浏览器 Agent 基准 {#6}

### 6.1 Mind2Web

Mind2Web（Deng et al., 2023）包含 2000+ 真实网站任务，评测从自然语言到网页操作的**泛化能力**（跨站、跨域）。任务分为三个子任务：动作候选检索、元素选择、动作序列生成。

### 6.2 WebArena

WebArena（Zhou et al., 2024）提供自托管的可交互网站环境（电商、论坛、CMS、GitLab），评测端到端任务完成率：

- 400+ 个真实任务；
- 需要登录、购物、发帖、代码管理等完整交互；
- 自动判分：状态检查 + 人工复核。

### 6.3 WebVoyager

WebVoyager（He et al., 2024）面向真实公开网站（不限于自托管），用视觉 + 文本双模态驱动浏览器操作，并引入**人工 + 模型混合判分**。

### 6.4 BrowserGym / VBS

- **BrowserGym**：统一 Web Agent 实验环境（Lightweight Web Arena）；
- **Visual WebBench（VWB）**：视觉驱动的网页操作评测，强调"像人一样看屏幕"。

---

## 7 GUI / 设备 Agent 基准 {#7}

### 7.1 OSWorld

OSWorld（Xie et al., 2024）面向完整操作系统（Ubuntu/Windows/macOS）的桌面 Agent：

- 369 个任务，覆盖文件、办公、浏览、编程；
- 需要 GUI 交互、键盘鼠标控制、多应用协作；
- 结论：当时的 GUI Agent 成功率普遍很低（< 20%），是开放难题。

### 7.2 AndroidWorld / AndroidArena

- **AndroidWorld**：在 Android 模拟器中评测移动端 Agent（发消息、订票、设置）；
- **AndroidArena**：多应用移动任务基准。

### 7.3 Mobile-Agent / 其他

- **Mobile-Agent**：中文开源移动端 Agent，支持安卓自动化；
- **τ-bench 的移动变体**、Llama-Mobile 等新兴基准不断涌现。

详见 `gui-agent.md` 中的系统介绍。

---

## 8 评测方法与陷阱 {#8}

### 8.1 判分方式

| 方式 | 优点 | 缺点 |
|------|------|------|
| 自动状态检查 | 客观、可复现 | 需精心设计状态表示 |
| 隐藏测试 | 代码/结果验证强 | 难覆盖开放任务 |
| LLM-as-Judge | 灵活、可扩展 | 可能有偏见、不稳定 |
| 人工评测 | 最可靠 | 昂贵、不可扩展 |

### 8.2 常见陷阱

- **数据污染**：模型可能见过基准任务（尤其 SWE-bench 类）；
- **奖励设计偏差**：过度优化成功率导致"作弊"（如输出空 patch 骗过简单检查）；
- **方差被忽略**：单次运行无法反映真实能力，需多次采样；
- **环境差异**：不同运行环境导致结果不可比；
- **只报最好结果**：应报告 mean ± std 而非只挑最优采样。

---

## 9 实践建议 {#9}

- **分层评测**：单元级（函数调用/单步）→ 任务级（端到端）→ 系统级（成本+安全+体验）；
- **多基准组合**：代码用 SWE-bench Verified，通用用 GAIA + AgentBench，GUI 用 OSWorld，业务用 τ-bench；
- **建立内部评测集**：把线上真实失败案例沉淀为回归测试；
- **报告完整统计**：成功率、方差、步数、成本、安全违规，而非单一数字；
- **警惕污染**：避免用公开基准训练时直接调优该基准；
- **成本纳入考量**：同等成功率下，更少步数/更低成本的 Agent 更优。

---

## 10 参考文献 {#10}

[1] X. Liu, H. Yu, H. Zhang, et al., *AgentBench: Evaluating LLMs as Agents*, ICLR, 2024.

[2] G. Mialon, R. Fourrier, C. Swift, et al., *GAIA: A Benchmark for General AI Assistants*, ICLR, 2024.

[3] S. Yao, N. Shinn, A. Razavi, K. Narasimhan, *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*, 2024. arXiv:2406.12045.

[4] C. E. Jimenez, J. Yang, A. Wettig, et al., *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*, ICLR, 2024.

[5] X. Deng, Y. Gu, B. Zheng, et al., *Mind2Web: Towards a Generalist Agent for the Web*, NeurIPS, 2023.

[6] S. Zhou, F. F. Xu, H. Zhu, et al., *WebArena: A Realistic Web Environment for Building Autonomous Agents*, ICLR, 2024.

[7] H. He, W. Yao, K. Ma, et al., *WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models*, ACL, 2024.

[8] T. Xie, D. Zhang, J. Chen, et al., *OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments*, NeurIPS, 2024.

[9] Y. Qin, S. Liang, Y. Ye, et al., *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*, ICLR, 2024.

[10] S. Yao, D. Yang, et al., *Function-Calling Benchmarks: BFCL*, 2023-2025.
