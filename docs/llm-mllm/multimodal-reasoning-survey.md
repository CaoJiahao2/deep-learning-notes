# 大型多模态推理模型（LMRM）综述

> 论文：**Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models**  
> arXiv: **2505.04921v2**  
> 版本日期：**2025-07-06**

这篇综述讨论的是一个很明确的问题：

> **多模态大模型如何从“能看懂”走向“能推理”，再从“能推理”走向“能长期规划、调用工具并与环境交互”？**

作者将这一方向概括为 **Large Multimodal Reasoning Models（LMRMs，大型多模态推理模型）**，并进一步提出未来的 **Native Large Multimodal Reasoning Models（N-LMRMs，原生大型多模态推理模型）**。

如果只记住整篇论文的一条主线，可以记成：

```text
Perception
    ↓
Multimodal CoT
    ↓
Search / Long Reasoning
    ↓
Reinforcement Learning
    ↓
Agentic Reasoning
    ↓
Native Multimodal Reasoning
```

也就是：

> **Perception → CoT → Search → RL → Agent**

---

# 1. 论文到底在研究什么？

传统多模态大模型更多关注：

```text
图像 / 视频 / 音频
        ↓
      理解
        ↓
      回答
```

而这篇论文关注的是更高一级的能力链：

```text
Perception
   ↓
Reasoning
   ↓
Thinking
   ↓
Planning
   ↓
Action
```

LMRM 的目标不只是识别“图里有什么”，而是希望模型能够完成：

- 逻辑推理；
- 因果推理；
- 类比推理；
- 长程思考；
- 多步骤任务分解；
- 规划；
- 工具调用；
- 与真实或虚拟环境持续交互。

因此，作者认为多模态模型当前的核心瓶颈，已经开始从单纯的 **perception / understanding** 转向：

- **reasoning depth**：推理深度；
- **generalization**：跨任务、跨模态泛化；
- **agentic behavior**：智能体式行为。

---

# 2. 整篇论文的核心路线图

论文正文主要组织为三个已发展的阶段，并进一步提出一个未来范式。

| 阶段 | 核心思想 | 代表技术 |
|---|---|---|
| Stage 1 | 感知驱动、隐式推理 | Representation、Alignment、Fusion、Task-specific reasoning modules |
| Stage 2 | 语言中心的短推理 | MCoT、Prompt、Structural Reasoning、Tools / RAG |
| Stage 3 | 语言中心的长推理 | Cross-modal reasoning、Long CoT、O1-like、R1-like |
| Future | Native Multimodal Reasoning | Omni-modal、Agent、Environment Interaction |

可以把能力演进理解为：

```text
Stage 1
Perception → Fusion → Prediction

Stage 2
Perception → Short Reasoning → Answer

Stage 3
Perception → Long Thinking → Planning → Answer

Future N-LRM
Perception ↔ Reasoning ↔ Generation ↔ Action ↔ Environment
```

一个值得注意的小细节：论文摘要把路线称为“四阶段”，而引言正文又说是“三阶段”，随后单独把 N-LMRM 作为未来方向展开。理解时最好把它看成：

> **三阶段演进 + 一个未来原生范式。**

---

# 3. Stage 1：感知驱动的模块化推理

早期多模态模型主要解决三个问题：

```text
Representation
     ↓
Alignment
     ↓
Fusion
     ↓
Prediction
```

也就是：

1. 不同模态怎么表示；
2. 图像和文本怎么对齐；
3. 多模态信息怎么融合。

早期模型通常采用：

- CNN；
- LSTM；
- Attention；
- Memory Network；
- Task-specific reasoning module；
- 后来逐渐过渡到 Transformer-based VLM。

这一时期所谓的“推理”，多数仍然是**隐式推理**：

```text
视觉特征 + 文本特征
        ↓
      融合
        ↓
     直接预测
```

模型并不会显式执行：

```text
先分析 A
→ 再推导 B
→ 检查 C
→ 得出答案
```

因此，Stage 1 更准确地说是在解决：

> **怎么看懂多模态信息。**

而不是：

> **怎么深入思考。**

---

# 4. Stage 2：Language-Centric Short Reasoning

Stage 2 的关键变化是：

> **不再专门设计 reasoning module，而是让 LLM 本身承担推理。**

典型范式变成：

```text
Vision Encoder
      ↓
     LLM
      ↓
Reasoning + Answer
```

此时最重要的技术就是：

# Multimodal Chain-of-Thought（MCoT）

传统：

```text
Image → Answer
```

MCoT：

```text
Image
  ↓
Intermediate Reasoning
  ↓
Answer
```

作者把 Stage 2 的方法分成三类：

1. Prompt-based MCoT；
2. Structural Reasoning；
3. Externally Augmented Reasoning。

---

## 4.1 Prompt-based MCoT：告诉模型“你要思考”

这是最轻量的方法。

基本不修改模型，只通过 Prompt 诱导模型输出显式推理过程。

例如：

```text
See
 ↓
Think
 ↓
Confirm
 ↓
Answer
```

可以理解成：

> “先观察图像，再一步一步分析，最后回答问题。”

优点：

- 不需要额外复杂训练；
- 成本低；
- 可解释性较好。

问题：

- 推理能力高度依赖 backbone 原本会什么；
- 对 Prompt 比较敏感；
- reasoning path 不稳定。

因此，它更像：

> **Reasoning Elicitation：把模型已有推理能力诱导出来。**

而不是：

> **Reasoning Learning：真正训练出新的推理能力。**

---

## 4.2 Structural Reasoning：规定模型“怎么思考”

Prompt-based CoT 太自由，因此作者进一步讨论结构化推理。

核心思想：

> **把自由生成的 CoT 变成标准化 reasoning workflow。**

例如：

```text
Question Decomposition
        ↓
Visual Grounding
        ↓
Caption / Observation
        ↓
Reasoning
        ↓
Verification
        ↓
Answer
```

TextCoT 的思路可以简化成：

```text
Image Overview
      ↓
Coarse Localization
      ↓
Fine-grained Observation
      ↓
Answer
```

作者把 Structural Reasoning 分为三类：

### 1. Rationale Construction

让模型学习生成更高质量、更细粒度的 rationale。

### 2. Defined Reasoning Procedure

人为定义明确推理流程：

```text
A → B → C → D
```

### 3. Modality-specific Structural Reasoning

针对不同模态设计不同推理结构，比如：

- 图像；
- 音频；
- 视频；
- Embodied input。

Structural Reasoning 的优势是：

- consistency 更高；
- interpretability 更好；
- reasoning 更可控。

但问题也很明显：

> **复杂问题未必符合人工预定义的固定流程。**

因此模型的 adaptability 仍然有限。

---

## 4.3 Externally Augmented Reasoning：把部分推理交给外部系统

这一步已经开始出现 Agent 的味道。

核心思想：

> **不要求 backbone model 什么都自己完成。**

模型可以调用：

- Search；
- RAG；
- Tool；
- Vision Expert；
- specialized multimodal processor。

例如：

```text
LLM Planning
    ↓
Object Detector
    ↓
Retrieve Knowledge
    ↓
Continue Reasoning
    ↓
Call Another Tool
    ↓
Answer
```

作者大致分为：

1. Search-enhanced MCoT；
2. Tool-based augmentation；
3. Retrieval-Augmented Reasoning；
4. Multimodal enhancing modules。

普通 CoT 往往只有一条路线：

```text
Thought 1
   ↓
Thought 2
   ↓
Thought 3
   ↓
Answer
```

搜索增强以后：

```text
                 → Path A
Problem → Search → Path B → Evaluate → Answer
                 → Path C
```

也就是说：

> **模型开始在 reasoning space 中搜索。**

这一变化非常重要，因为它为后面的 O1-like Reasoning Model 奠定了基础。

---

# 5. Stage 2 的本质：System-1

虽然 MCoT、结构化流程和工具增强都让模型“更像在推理”，但作者认为 Stage 2 仍然属于：

> **System-1 Reasoning**

特点：

- 短链；
- 局部；
- 快速；
- 反应式；
- 适合熟悉或边界明确的问题。

模型通常还是：

```text
输入问题
  ↓
推几步
  ↓
回答
```

但复杂任务真正需要的是：

- 自主分解任务；
- 选择不同 strategy；
- 发现错误；
- 回退；
- 重新观察；
- 修改计划；
- 长时间持续推理。

这就进入 Stage 3。

---

# 6. Stage 3：Language-Centric Long Reasoning

Stage 3 是整篇论文技术上最核心的部分。

作者借用了 System-2 的概念：

> **更慢、更审慎、更系统、更依赖规划的推理。**

核心能力包括：

```text
Long Reasoning
+
Problem Decomposition
+
Planning
+
Search
+
Reflection
+
Cross-modal Reasoning
+
Reinforcement Learning
```

Stage 3 被分为三个方向：

1. Cross-Modal Reasoning；
2. Multimodal-O1；
3. Multimodal-R1。

---

# 7. Cross-Modal Reasoning：不仅“看图”，还要“用图思考”

很多所谓 Multimodal CoT，实际上仍然是：

```text
Image
  ↓
Visual Encoder
  ↓
Language Representation
  ↓
Text CoT
```

也就是说：

> **视觉只是输入，真正的 reasoning space 仍然是文字。**

这并不是真正意义上的 multimodal reasoning。

作者认为 Cross-Modal Reasoning 更理想的形态是：

```text
Text
 ↕
Image
 ↕
Visual Region
 ↕
Tool
 ↕
Text
```

或者：

```text
Image
  ↓
Text Reasoning
  ↓
重新观察局部视觉区域
  ↓
继续推理
  ↓
生成 visual subgoal
  ↓
下一步行动
```

即：

> **不同模态本身成为 reasoning carrier。**

论文讨论了三类方式：

### External Tools

根据 reasoning state 动态调用：

- OCR；
- Grounding；
- CV Expert；
- Program Execution。

### External Algorithms

例如：

- fast / slow thinking 切换；
- 视觉与文本 reasoning step 交替；
- Image-of-Thought；
- visual subgoal generation。

### Model-Intrinsic

让模型自身学会：

- 生成视觉中间表示；
- 更新 visual-text representation；
- 使用视觉对象或 visual patch 参与推理。

核心结论：

> **真正的 multimodal reasoning，不应该只是 multimodal input + textual reasoning，而应该让 multimodal information 本身进入 reasoning loop。**

---

# 8. Multimodal-O1：用 Long CoT + Search 把推理做深

O1-like 方法可以粗略概括为：

> **高质量 CoT 数据 + Long Reasoning + Test-Time Search**

## 训练阶段

使用高质量 rationale / Long-CoT 数据进行 fine-tuning。

普通文本 reasoning：

```text
Thinking
   ↓
Answer
```

多模态 reasoning 可能进一步拆成：

```text
Summary / Rationale
       ↓
Caption
       ↓
Thinking
       ↓
Answer
```

先处理视觉信息，再进入更深 reasoning。

## 推理阶段

重点是：

> **Inference-Time Scaling**

即给模型更多推理时计算资源。

代表技术：

### Best-of-N

```text
Generate N Answers
      ↓
Select Best
```

### Beam Search

在每个中间步骤保留最有希望的候选。

### MCTS

把 reasoning 看成一棵搜索树：

```text
Problem
  ├── Thought A
  │    ├── A1
  │    └── A2
  │
  ├── Thought B
  │    ├── B1
  │    └── B2
  │
  └── Thought C
```

模型不断：

```text
探索
→ 评分
→ 扩展
→ 再探索
```

所以可以把 O1-like 路线简化为：

> **SFT 教模型“好的推理长什么样” + Search 在 inference time 找更好的 reasoning trajectory。**

---

# 9. Multimodal-R1：用 RL 学习“应该怎么推理”

R1-like 更进一步。

它不再完全依赖人工提供完整 Long-CoT，而是：

```text
Policy
  ↓
Sample Reasoning Trajectories
  ↓
Reward
  ↓
RL Optimization
  ↓
Updated Policy
```

核心思想：

> **不一定告诉模型完整正确的思考过程，而是通过 reward 告诉它“哪种推理更有效”。**

论文重点讨论：

- DPO；
- GRPO。

---

## DPO

使用 preference data：

```text
Reasoning A
vs.
Reasoning B
```

告诉模型：

> 哪一个 reasoning / answer 更好。

---

## GRPO

模型一次生成多个 reasoning trajectories：

```text
Prompt
 ├── Reasoning A → wrong
 ├── Reasoning B → correct
 ├── Reasoning C → partially correct
 └── Reasoning D → correct
```

再使用：

- rule-based reward；
- reward model；
- multimodal feedback；

进行评分和 policy optimization。

随着训练进行，模型可能逐渐学会：

- decomposition；
- verification；
- reflection；
- backtracking；
- strategy switching。

这就是 R1-style 方法真正有吸引力的地方：

> **模型不只是模仿别人写出来的 CoT，而是在 optimization 中自己探索有效 reasoning strategy。**

---

# 10. O1 与 R1 的关键区别

这是理解当前 Reasoning Model 路线最重要的一点。

| 维度 | O1-like | R1-like |
|---|---|---|
| 核心手段 | Long CoT + Search | Reinforcement Learning |
| 主要依赖 | 高质量 reasoning data | Reward / Preference Signal |
| 推理提升 | 搜索更好的 reasoning path | 学习更好的 reasoning policy |
| 主要计算发生在哪 | Inference Time | Training Time |
| 典型技术 | Best-of-N、Beam Search、MCTS | DPO、GRPO |
| 目标 | 想得更深、更充分 | 学会怎么想得更好 |

一句话：

> **O1：Search over reasoning trajectories。**  
> **R1：Learn the reasoning policy。**

可以进一步理解为：

```text
SFT
模仿怎么想
   ↓
Search
探索不同想法
   ↓
RL
学习什么想法更有效
```

二者并不冲突。

一个强 reasoning model 完全可以：

```text
R1-style RL
先学到好的 reasoning policy
       ↓
O1-style Search
Inference 时继续扩大计算
```

---

# 11. Multimodal-R1 当前最大的瓶颈

论文指出：

> **当前 Multimodal-R1 学到的 long reasoning ability 仍然高度 task-specific。**

例如：

模型在视觉数学 / 几何任务中通过 GRPO 学会了长 CoT，

并不意味着这种能力能自然迁移到：

- 普通视觉理解；
- 视频推理；
- 音频推理；
- GUI Agent；
- Embodied Agent。

因此，当前最重要的问题之一仍然是：

> **Reasoning Generalization。**

---

# 12. 为什么还需要 Native LMRM？

即使 O1 / R1 已经具有 Long CoT、Search 和 RL，作者认为仍然存在一个根本限制：

> **Language-Centric Architecture**

大多数系统本质上仍然是：

```text
Image / Video / Audio
        ↓
      Encoder
        ↓
       LLM
        ↓
  Text Reasoning
```

也就是说：

> **语言仍然是主要 reasoning space。**

作者认为这带来两个问题。

---

## 12.1 Omni-modal 能力不足

现实世界不仅有：

- Text；
- Image；
- Video；

还有：

- Audio；
- Tactile；
- Sensor Streams；
- Temporal Signals；
- 3D；
- Action State。

当前模型在模态数量增加时性能往往明显下降。

说明：

> **模型还没有真正学会构造统一的 omni-modal reasoning path。**

---

## 12.2 Agentic 能力不足

真实世界任务需要：

```text
Perception
   ↓
Planning
   ↓
Tool Use
   ↓
Action
   ↓
Environment Feedback
   ↓
Re-planning
```

但当前模型在以下方面仍然不足：

- real-world grounding；
- long-horizon planning；
- tool integration；
- dynamic adaptation；
- cross-domain robustness。

因此作者提出：

# Native Large Multimodal Reasoning Model（N-LMRM）

---

# 13. 什么叫“Native”？

Native 并不是：

> “这个模型支持图片、视频、音频输入。”

真正的含义是：

> **Understanding、Generation、Reasoning、Planning、Action 从模型设计上原生统一。**

不是：

```text
Image Encoder + Audio Encoder + LLM
```

然后把一切都转成语言再思考。

而是希望：

```text
Text ───┐
Image ──┤
Video ──┤
Audio ──┼→ Unified Representation ↔ Reasoning ↔ Generation
Sensor ─┤
Action ─┤
3D ─────┘
```

不同模态能够：

> **共同参与 reasoning。**

---

# 14. N-LMRM 的两根核心支柱

作者认为未来 N-LMRM 需要把目前相对独立的两条研究路线真正统一。

---

## 14.1 Multimodal Agentic Reasoning

目标从：

> 回答问题

变成：

> 完成目标。

理想过程：

```text
Goal
  ↓
Hierarchical Task Decomposition
  ↓
Planning
  ↓
Perception
  ↓
Tool / Action
  ↓
Environment Feedback
  ↓
Re-planning
  ↓
...
```

关键能力包括：

### Hierarchical Task Decomposition

把复杂目标拆成长程子任务。

### Dynamic Adaptation

根据环境反馈实时修改策略。

### Embodied Learning

通过真实或模拟环境 interaction 学习，而不是只依赖静态 dataset。

因此：

> **R1 是“学怎么推理”，N-LMRM 进一步要求“学怎么行动，并根据行动结果继续推理”。**

---

# 15. Omni-Modal Understanding + Generative Reasoning

第二根支柱解决的是：

> **模型到底应该在什么空间里思考？**

未来模型可能不再是：

```text
Vision → LLM
Audio → LLM
Video → LLM
```

而是：

```text
Text
Image
Video
Audio
Sensor
Action
   ↓
Unified Representation
   ↕
Reasoning
   ↕
Generation
```

甚至出现：

```text
看视频
  ↓
听声音
  ↓
文字推理
  ↓
生成中间图像
  ↓
重新观察
  ↓
调用工具
  ↓
执行动作
```

所以：

> **Omni-modal ≠ 支持很多输入格式。**

真正重要的是：

> **这些模态是否能在同一个 reasoning process 中相互作用。**

---

# 16. o3 / o4-mini 为什么被作者特别关注？

作者把 o3 / o4-mini 视为向 N-LMRM 迈进的早期案例。

一个关键变化是：

> **Think with Images**

也就是模型可以在 reasoning 过程中：

- crop；
- zoom；
- flip；
- enhance；
- 重新观察图片；

再继续 reasoning。

传统：

```text
Image → Text → Reasoning
```

开始变成：

```text
Image
  ↓
Reason
  ↓
Crop / Inspect
  ↓
Reason
  ↓
...
```

这已经出现了：

> **Interleaved Multimodal Reasoning**

的雏形。

不过作者同时发现问题：

- 语言先验可能干扰真实视觉输入；
- 文件处理和多媒体生成仍不可靠；
- reasoning trace 可能不忠实；
- complex tool use 仍然不稳定。

所以这些模型还不是作者理想中的完整 N-LMRM。

---

# 17. N-LMRM 的四条关键技术路线

论文第 4.3 节给出了非常值得关注的研究方向。

---

## 17.1 Unified Representation and Cross-Modal Fusion

第一个问题：

> Text、Image、Audio、Video、Sensor 如何进入同一个模型？

传统方法：

```text
Image Encoder
Audio Encoder
Video Encoder
    ↓
Connector
    ↓
LLM
```

未来希望进一步走向统一表示。

但存在一个问题：

> **不同模态可能发生 negative interference。**

某个模态过强可能破坏其他模态表示。

因此论文提到 MoE 是潜在路线之一：

```text
            Modality Expert A
           /
Core Model ─ Modality Expert B
           \
            Modality Expert C
```

即：

> **共享核心 intelligence + modality-specific experts。**

---

# 18. Interleaved Multimodal Long Chain-of-Thought

这是第四章技术上非常关键的概念。

传统 R1：

```text
Text
 ↓
Text Reasoning
 ↓
Text Reasoning
 ↓
Answer
```

未来：

```text
Text
 ↓
Visual Reasoning
 ↓
Crop Image
 ↓
Text
 ↓
Audio
 ↓
Tool
 ↓
Action
 ↓
...
```

也就是：

> **Interleaved Multimodal Long CoT**

这意味着 test-time scaling 出现一个新的维度。

过去：

```text
更多 token
+
更长 CoT
+
更多 Search Branch
```

未来：

```text
更多 token
+
更多 visual observation
+
更多 tool call
+
更多 modality interaction
```

因此 reasoning compute 不再只是“生成更多文字”。

---

# 19. Learning and Evolving from World Experiences

传统训练：

```text
Dataset
  ↓
Train
  ↓
Deploy
```

N-LMRM 希望变成：

```text
Environment
     ↓
Interaction
     ↓
Experience
     ↓
Learning
     ↓
New Policy
     ↓
Environment
     ↺
```

也就是：

> **模型通过持续 interaction 积累 world experience。**

这与：

- World Model；
- Online RL；
- Lifelong Learning；
- Agent Learning；

高度相关。

作者认为未来模型需要逐渐：

- 记录 interaction trajectory；
- 总结 transferable experience；
- 更新 decision strategy；
- 自主探索 tool combination。

最终从：

> Static Knowledge Model

走向：

> Self-improving Interactive System。

---

# 20. Data Synthesis：未来真正缺什么数据？

普通 Image-Text Pair 已经不够。

N-LMRM 真正需要的数据可能是：

```text
Multimodal Observation
        ↓
Task Decomposition
        ↓
Reasoning
        ↓
Tool Calling
        ↓
Visual Operation
        ↓
Environment Feedback
        ↓
Re-planning
        ↓
Continue Action
```

也就是完整：

> **Multimodal Interactive Trajectory**

论文认为当前尤其缺：

- 三种及以上模态联合数据；
- multimodal interactive CoT；
- dynamic environment 中的 multi-step planning；
- multi-tool calling；
- parallel tool use；
- multimodal generation + reasoning trajectory。

---

# 21. 第五章：Dataset and Benchmark

第五章主要回答：

> **LMRM 应该测什么？怎么测？**

作者把能力分成四类：

```text
Understanding
    ↓
Generation
    ↓
Reasoning
    ↓
Planning
```

---

# 22. Multimodal Understanding

核心问题：

> **模型能不能看懂 / 听懂？**

包括：

### Visual-Centric Understanding

- VQA；
- OCR；
- DocVQA；
- Video Understanding；
- Comprehensive Multimodal Understanding。

代表 benchmark：

- VQA；
- DocVQA；
- MMBench；
- MMMU；
- MM-Vet。

### Audio-Centric Understanding

包括：

- ASR；
- Speech Translation；
- Emotion Recognition；
- Audio Understanding；
- Music Understanding。

这一层主要评估：

> **Perception + Understanding**

而不是复杂的长程 reasoning。

---

# 23. Multimodal Generation

分成两类。

## Cross-modal Generation

一个模态输入，另一个模态输出：

```text
Text → Image
Text → Video
Text → Speech
```

核心：

- alignment；
- semantic consistency；
- generation quality。

## Joint Multimodal Generation

同时生成多个模态：

```text
Text
  ↓
Text + Image + Audio + Video
```

重点不只是单独生成质量，还要保证：

> **不同模态之间 coherent / aligned。**

这与 N-LMRM 的 generative reasoning 非常相关。

---

# 24. Multimodal Reasoning

这一层不只是识别内容，而是要求模型：

> **综合多种信息进行 inference 和 problem solving。**

论文分为：

## General Visual Reasoning

```text
视觉理解
+
常识
+
逻辑
+
知识
```

例如：

- VCR；
- PhysBench；
- VideoPhy。

## Domain-specific Reasoning

评估特定领域的高难 reasoning：

- MathVista；
- MATH-Vision；
- ChartQA；
- ScienceQA；
- Robotics；
- Physical Reasoning；
- Web Reasoning。

这类 benchmark 更能真正测试：

> **模型是不是“推出来的”。**

---

# 25. Multimodal Planning

这是第五章里与未来 Agent 最直接相关的一类。

Planning benchmark 不再只是：

```text
Question → Answer
```

而是：

```text
Observe
  ↓
Plan
  ↓
Act
  ↓
Receive Feedback
  ↓
Continue
```

作者分为两类。

---

## 25.1 GUI Navigation

让模型操作：

- Web；
- Desktop；
- Mobile App；
- GUI。

代表 benchmark：

- WebArena；
- Mind2Web；
- OSWorld；
- Windows Agent Arena；
- VisualAgentBench。

能力要求：

```text
Visual Grounding
+
Multi-step Reasoning
+
Planning
+
Action
```

例如：

> 打开网站 → 搜索信息 → 找按钮 → 填表 → 下载文件。

这已经不是传统 VQA。

---

## 25.2 Embodied / Simulated Environment

进一步进入：

- Robot；
- Minecraft；
- Simulated Home；
- Autonomous Driving；
- Interactive Game。

代表环境：

- MineDojo；
- BEHAVIOR-1K；
- Habitat 3.0；
- SAPIEN；
- DrivingDojo。

这里要求模型真正形成：

```text
Language
+
Vision
+
Action
+
Environment Feedback
```

闭环。

---

# 26. Evaluation Method：到底怎么评分？

作者总结四类主流评价方法。

---

## 26.1 Exact / Fuzzy Match

模型答案与标准答案进行直接或模糊匹配。

优点：

- 客观；
- 自动；
- 成本低。

问题：

- 不适合开放式回答。

---

## 26.2 Option Matching

把问题转成选择题。

优点：

- 评分稳定；
- 自动化容易。

问题：

- 与真实开放问题差距较大。

---

## 26.3 LLM / MLLM Scoring

使用另一个强模型做 Judge：

```text
Question
+
Reference
+
Model Answer
    ↓
Judge LLM / MLLM
    ↓
Score
```

更适合：

- Open-ended QA；
- Multimodal Output；
- Generation；
- Reasoning。

---

## 26.4 Agentic Evaluation

评价器本身也可以：

- 调工具；
- 多 Agent 协作；
- 多 Agent 对抗；
- 检查复杂多模态输出。

评价范式也开始发生变化：

```text
Static Metric
     ↓
LLM-as-a-Judge
     ↓
Agent-based Evaluation
```

---

# 27. 整篇论文的统一逻辑

从模型能力角度：

```text
Understanding
    ↓
Reasoning
    ↓
Thinking
    ↓
Planning
    ↓
Acting
```

从技术角度：

```text
Representation / Fusion
        ↓
MCoT
        ↓
Structural Reasoning
        ↓
Tools / Search
        ↓
Long CoT
        ↓
O1-like Search
        ↓
R1-like RL
        ↓
Agentic Reasoning
        ↓
Native LMRM
```

从评价角度：

```text
VQA Accuracy
    ↓
Reasoning Benchmark
    ↓
Open-ended Evaluation
    ↓
Planning Benchmark
    ↓
Agentic Evaluation
```

三个方向实际上是一致的。

---

# 28. 我认为最值得记住的 5 个关键词

如果几个月后只允许记住五个词：

## 1. Perception

先让模型看懂世界。

## 2. CoT

把隐式 reasoning 显式化。

## 3. Search

让模型探索多个 reasoning trajectories。

## 4. RL

让模型学习“什么样的 reasoning strategy 更好”。

## 5. Agent

让 reasoning 与 action、tools、environment feedback 形成闭环。

所以最简路线就是：

> **Perception → CoT → Search → RL → Agent**

---

# 29. O1、R1、N-LMRM 三者关系

可以这样理解：

```text
O1
↓
解决“怎么在推理时想得更深”

R1
↓
解决“怎么在训练中学会更好的推理策略”

N-LMRM
↓
解决“怎么让多模态感知、推理、生成、行动和环境交互原生统一”
```

更进一步：

```text
O1 = Test-Time Scaling
R1 = Learning Reasoning Policy
N-LMRM = Native Multimodal Agentic Intelligence
```

---

# 30. 论文最终想表达什么？

这篇综述真正的核心观点并不是：

> “多模态模型越来越强。”

而是：

> **AI 的能力中心正在从“理解输入并生成答案”，迁移到“在多模态世界中持续感知、推理、规划、行动，并根据反馈自我调整”。**

当前模型已经从：

```text
Perception
```

进入：

```text
Reasoning
```

下一阶段则是：

```text
Reasoning
   ↓
Planning
   ↓
Action
   ↓
Environment Interaction
```

最终目标不是一个“更会回答问题的 VLM”，而是一个：

> **能够在真实世界中进行原生多模态推理、长期规划和自主行动的智能系统。**

---

# 31. 阅读建议

如果只是为了快速掌握这篇 91 页综述，不建议平均用力。

推荐优先级：

```text
Introduction
   ↓
Section 2：理解整体演化路线
   ↓
Section 3.2：理解 MCoT
   ↓
Section 3.3：重点读 O1 / R1 / Cross-modal Reasoning
   ↓
Section 4：重点读 Native LMRM
   ↓
Section 5：按需查 Benchmark
```

最值得精读的内容：

- 3.2 Multimodal CoT；
- 3.3 Cross-Modal Reasoning；
- 3.3.2 Multimodal-O1；
- 3.3.3 Multimodal-R1；
- 4.2 Capability of N-LMRMs；
- 4.3 Technical Prospects。

第五章更适合作为：

> **数据集 / Benchmark 工具手册**

使用，不必逐个背诵。

---

# 32. 最终速记版

```text
Stage 1
感知驱动
Representation / Alignment / Fusion
        ↓

Stage 2
显式短推理
MCoT
Prompt → Structure → Tool
        ↓

Stage 3
System-2 长推理
Cross-modal
+ Long CoT
+ Search
+ O1
+ R1 / RL
        ↓

Future
Native LMRM
Omni-modal
+ Generation
+ Reasoning
+ Planning
+ Action
+ Environment
```

一句话收尾：

> **多模态智能正在从“看懂并回答”，走向“感知—推理—思考—规划—行动”的完整闭环。**

---

## 参考论文

**Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models**  
arXiv:2505.04921v2, 2025.

