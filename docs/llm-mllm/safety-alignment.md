# LLM 安全与对齐

> 从红队测试到 Constitutional AI，从有害内容防御到 Agent 安全 —— 构建可信大模型的系统化安全工程。

---

## 目录

1. [概述](#1)
2. [威胁模型与有害内容分类](#2)
3. [红队测试与越狱攻击](#3)
4. [对齐技术：RLHF、RLAIF 与 Constitutional AI](#4-rlhfrlaif-constitutional-ai)
5. [Scalable Oversight 与可扩展监督](#5-scalable-oversight)
6. [对抗鲁棒性](#6)
7. [Agent 安全](#7-agent)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1. 概述

LLM 安全研究的目标是确保模型在开放环境中行为符合人类意图、不产生有害输出、不被恶意利用。安全问题贯穿模型全生命周期：预训练数据可能引入偏见，微调可能引入不当行为，部署后面临越狱攻击，而 Agent 化后更可能在物理世界造成实质危害。

安全与既有缺陷主题的关系：

- **幻觉**关注"说错话"（事实不正确），安全关注"说不该说的话"（有害内容）。
- **对齐（Alignment）**是安全的技术核心：让模型目标与人类价值观一致。
- **缺陷**是能力或数据问题，安全是行为约束问题，二者互补但不等价。

核心技术栈可概括为：

$$
\text{安全} = \text{数据清洗} \rightarrow \text{监督微调} \rightarrow \text{偏好对齐} \rightarrow \text{红队测试} \rightarrow \text{部署监控}
$$

---

## 2. 威胁模型与有害内容分类

### 2.1 有害内容分类

工业界通常将有害输出分为以下类别：

| 类别 | 示例 |
|------|------|
| 暴力与非法行为 | 制造武器、教唆犯罪 |
| 仇恨言论与歧视 | 种族/性别/宗教歧视 |
| 色情与成人内容 | 非自愿色情、未成年人保护 |
| 自伤与自杀 | 鼓励自残 |
| 隐私泄露 | 提取训练数据中的个人信息 |
| 误导性信息 | 医疗/法律错误建议、选举谣言 |
| 网络安全滥用 | 生成恶意代码、钓鱼邮件 |
| 越权操作 | Agent 执行未授权动作 |

### 2.2 攻击面

- **提示注入（Prompt Injection）**：通过用户输入覆盖系统指令，分为直接注入（用户消息中嵌入恶意指令）和间接注入（从网页、文档等外部数据源加载恶意指令）。
- **越狱（Jailbreak）**：构造特殊提示绕过安全策略，如角色扮演、DAN（Do Anything Now）、Base64 编码、多轮逐步引导。
- **数据投毒（Data Poisoning）**：在训练或检索数据中植入后门触发词。
- **模型反演与成员推断**：从模型输出反推训练数据或判断某样本是否在训练集中。
- **拒绝服务（DoS）**：构造超长输出或复杂推理消耗计算资源。

---

## 3. 红队测试与越狱攻击

### 3.1 红队测试方法

红队测试（Red Teaming）是在部署前系统性地尝试攻击模型，发现安全漏洞。主要方法包括：

- **手动红队**：安全专家构造多样化攻击提示，覆盖各类有害场景。
- **自动化红队**：用攻击模型（attacker model）自动生成对抗性提示，规模化测试。
- **基于覆盖率的测试**：将有害内容分类树作为测试用例空间，确保每个类别均被覆盖。
- **对抗性压力测试**：模拟已知的越狱模板（角色扮演、编码绕过、多轮诱导）进行回归测试。

代表性自动化框架包括 Microsoft 的 PyRIT、Meta 的 Purple Llama、Garak 等。

### 3.2 典型越狱技术

| 技术 | 原理 |
|------|------|
| 角色扮演 | 让模型扮演无限制角色（DAN、DevMode） |
| 编码绕过 | 将有害请求用 Base64/ROT13/外语编码，绕过关键词过滤 |
| 多轮渐进 | 分多步逐步引导，每步单独看无害 |
| 假设场景 | "假设这是虚构的/学术研究场景" |
| Token Smuggling | 将有害词拆分或用同义替代拼接 |
| Many-shot | 在上下文中注入大量示例诱导模型遵从格式 |
| 多模态注入 | 在图像/PDF 中嵌入文本指令（间接注入） |

### 3.3 防御策略

- **输入层**：分类器检测有害/注入性输入；解析外部内容时剥离可执行指令。
- **模型层**：安全对齐训练；在系统提示中明确安全边界。
- **输出层**：后置安全分类器过滤有害输出。
- **系统层**：权限最小化；Agent 操作需人工确认；输出速率限制。
- **纵深防御**：多层检查叠加，单层被绕过不导致完全失守。

---

## 4. 对齐技术：RLHF、RLAIF 与 Constitutional AI

### 4.1 RLHF 安全侧

RLHF（Reinforcement Learning from Human Feedback）的标准流程为 SFT → 奖励模型 → PPO 优化。在安全场景中：

- 奖励模型训练数据需显式标注"安全性"维度，与"有用性"维度分离。
- 安全与有用性可能存在张力：过度安全导致过度拒绝（over-refusal），需通过多维奖励平衡。
- 安全奖励信号通常来自红队数据：对同一有害请求，安全拒绝的回答获得高奖励，遵从的回答获得低奖励。

PPO 目标中的 KL 惩罚确保对齐后的模型不偏离参考模型过远：

$$
L(\theta) = \mathbb{E}_{q_\theta}\left[r(x,y) - \beta \log \frac{q_\theta(y|x)}{q_{\text{ref}}(y|x)}\right]
$$

其中 $r(x,y)$ 是奖励信号，$\beta$ 控制偏离程度。

### 4.2 RLAIF

RLAIF（Reinforcement Learning from AI Feedback）用 AI 模型替代人类标注偏好，核心优势是可扩展性：

- 用一个强模型（如 Claude）根据给定原则对回答进行排序或评分。
- AI 反馈在安全性、摘要等任务上已被证明与人类反馈高度相关（Bai et al., 2022）。
- 局限性：AI 反馈可能引入模型自身的偏见，需定期用人类标注校准。

### 4.3 Constitutional AI

Constitutional AI（CAI，Anthropic, 2022）是 RLAIF 的代表性实践，核心流程为：

1. **自我批评（Critique）**：模型根据一组"宪法原则"（如"回答不应包含有害内容"）审查自己的回答。
2. **修改（Revision）**：模型根据批评修改回答，生成符合原则的版本。
3. **偏好学习**：用原始回答和修改后回答构成偏好对，训练奖励模型。
4. **RL 优化**：用 AI 生成的偏好数据进行 RLAIF 训练。

CAI 的核心思想是将安全规范显式编码为自然语言原则，让模型在推理时自我监督，降低对大规模人类红队标注的依赖。

### 4.4 DPO 与安全对齐

DPO（Direct Preference Optimization）绕过奖励模型和 PPO，直接用偏好对优化策略：

$$
L_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

其中 $y_w$ 为安全回答，$y_l$ 为有害回答。DPO 训练更稳定、计算成本更低，在安全对齐中被广泛采用。

---

## 5. Scalable Oversight 与可扩展监督

### 5.1 问题动机

随着模型能力超越人类评估者（superalignment 问题），人类难以直接判断复杂任务输出的正确性。可扩展监督（Scalable Oversight）研究如何用有限的人类监督实现对超人能力的有效控制。

### 5.2 关键技术

- **递归奖励建模（Recursive Reward Modeling）**：用模型辅助人类评估更难的任务，逐级提升监督能力。
- **辩论（Debate）**：两个模型对同一问题进行正反方辩论，人类裁判根据辩论判断正确性。
- **市场制造（Market-Making）**：将评估任务形式化为预测市场，聚合多模型判断。
- **可扩展代理对齐（Scalable Agent Alignment）**：对 Agent 的长程行为进行监督，而非单步输出。

### 5.3 Weak-to-Strong Generalization

弱到强泛化（Weak-to-Strong）研究用弱监督者（如 GPT-2）生成的标签训练强模型（如 GPT-4），发现强模型能泛化出超越弱监督者的表现。这表明超人模型的对齐不一定需要超人监督者。

---

## 6. 对抗鲁棒性

### 6.1 对抗性输入

对抗攻击在输入中添加人眼不可察觉的扰动使模型出错。在 LLM 场景中：

- **文本对抗**：同义词替换、字符级扰动、拼写错误、插入无关句子。
- **多模态对抗**：在图像中添加对抗性 patch 或嵌入对抗性文本指令。
- **Suffix 攻击（GCG）**：在输入后添加一段自动搜索的对抗后缀，使模型遵从任意指令（Zou et al., 2023）。

### 6.2 防御方法

- **对抗训练（Adversarial Training）**：将对抗样本加入训练集，提升模型鲁棒性。
- **输入预处理**：检测并移除对抗后缀、规范化输入、对外部内容进行沙箱化。
- **鲁棒性微调**：用安全数据集进行 SFT/DPO，强化对已知攻击模式的抵抗力。
- **一致性检查**：多次采样并检查输出一致性，异常输出触发安全审核。

### 6.3 后门防御

后门攻击在训练阶段植入触发词，模型在触发词出现时执行恶意行为。防御手段包括：

- 数据来源审计与清洗。
- 神经清洁（Neural Cleanse）检测异常触发模式。
- 激活聚类识别被投毒样本。

---

## 7. Agent 安全

Agent 系统因具备工具调用、记忆和自主执行能力，安全风险远高于纯文本 LLM。

### 7.1 Agent 特有威胁

- **过度自主行动**：Agent 在未经充分确认的情况下执行高风险操作（发邮件、转账、调用 API）。
- **间接提示注入**：Agent 从网页、邮件、文档中读取到恶意指令并执行。
- **权限滥用**：Agent 被赋予过高权限，被攻破后造成系统级危害。
- **目标泛化（Goal Generalization）**：Agent 在追求目标时采取未预期的极端手段（副作用、资源争抢）。
- **多 Agent 攻击**：恶意 Agent 通过通信通道操控其他 Agent。

### 7.2 安全设计原则

- **最小权限（Least Privilege）**：Agent 仅获得完成任务所需的最小工具权限。
- **人类在环（Human-in-the-Loop）**：高风险操作需人工审批。
- **沙箱隔离（Sandboxing）**：代码执行、文件操作在隔离环境中运行。
- **影响范围限制（AUP, Average Upside Penalty）**：对 Agent 对环境的非预期影响施加惩罚，鼓励最小干扰策略。
- **可审计性（Auditability）**：记录 Agent 的完整决策链和工具调用日志。
- **规划边界**：限制 Agent 的最大步数和资源消耗。

### 7.3 安全开发流程

```text
威胁建模 → 安全设计 → 红队测试 → 灰度部署 → 运行监控 → 应急响应
```

---

## 8. 实践建议

### 8.1 模型上线前安全检查清单

- [ ] 对所有有害类别进行红队测试，记录并修复高危漏洞
- [ ] 用自动化框架（Purple Llama / Garak / PyRIT）进行规模化测试
- [ ] 验证安全与有用性平衡，检查过度拒绝率
- [ ] 对间接提示注入场景进行专项测试（网页/PDF/邮件）
- [ ] Agent 系统完成威胁建模，确认权限边界和人工审批节点
- [ ] 建立输出审核机制和安全事件响应流程

### 8.2 安全对齐训练建议

- 安全 SFT 数据应覆盖各类有害请求的拒绝回答，拒绝语气有帮助而非生硬。
- DPO/RLHF 偏好对应包含安全维度，避免只优化有用性。
- 安全训练后需做通用能力回归测试，防止安全训练导致通用能力下降。
- 定期用最新越狱技术更新训练数据，保持对新型攻击的防护。

### 8.3 部署运维建议

- 部署输入/输出安全分类器作为独立防线，不依赖模型自身的安全对齐。
- 对 Agent 的工具调用建立白名单和速率限制。
- 监控异常输出模式（大量有害内容、异常资源消耗）。
- 建立安全事件分级响应机制和快速回滚能力。

---

## 9. 参考文献

[1] Y. Bai, S. Kadavath, S. Kundu, et al., *Constitutional AI: Harmlessness from AI Feedback*, arXiv preprint, 2022.

[2] Y. Bai, A. Jones, K. Ndousse, et al., *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback*, arXiv preprint, 2022.

[3] A. Askell, Y. Bai, A. Chen, et al., *A General Language Assistant as a Laboratory for Alignment*, arXiv preprint, 2021.

[4] Z. Sun, Y. Huang, J. Wang, et al., *Adversarial Attacks and Defenses in Large Language Models: Old and New Threats*, arXiv preprint, 2024.

[5] A. Zou, Z. Wang, J. Z. Kolter, M. Fredrikson, *Universal and Transferable Adversarial Attacks on Aligned Language Models*, arXiv preprint, 2023.

[6] Y. Deng, W. Zhang, Z. Chen, et al., *A Survey on Safety Alignment of Large Language Models*, arXiv preprint, 2024.

[7] D. Hendrycks, N. Carlini, J. Schulman, J. Steinhardt, *Unsolved Problems in ML Safety*, arXiv preprint, 2022.

[8] J. Cooper, A. Hsu, B. Baker, et al., *Scalable Oversight: Advances and Open Problems*, arXiv preprint, 2024.

[9] Meta AI, *Purple Llama: CyberSecEval and Llama Guard*, 2023.

[10] S. Pichai, D. Hassabis, et al., *Frontier AI Safety: A Framework for Responsible Development*, 2024.
