# 大模型后训练（Post-Training）综述

> 基于论文 **A Survey on Post-Training of Large Language Models** 的内容整理。  
> 目标：快速建立大模型 Post-Training 的完整技术地图，并抓住 PPO / RLHF / DPO / GRPO / Reasoning RL 等关键概念。

---

## 0. 一句话理解整篇论文

整篇论文可以压缩成一条主线：

$$
\text{Base LLM}
\rightarrow
\text{SFT}
\rightarrow
\text{Alignment}
\rightarrow
\text{Reasoning}
\rightarrow
\text{Efficiency}
\rightarrow
\text{Integration \& Adaptation}
$$

分别对应：

- **SFT**：让模型学会“按照要求回答”
- **Alignment**：让模型学会“什么回答更符合人类偏好”
- **Reasoning RL**：让模型学会“如何探索出正确答案”
- **Efficiency**：让模型训练、推理、部署更便宜
- **Integration & Adaptation**：让模型接入多模态、专业知识和多个模型能力

---

# 1. Introduction：为什么需要 Post-Training

Pre-training 解决的是：

> 模型“知道什么”、具备哪些基础语言与知识能力。

Post-training 解决的是：

> 模型“如何使用这些能力”，以及如何适应用户、任务、领域和推理需求。

因此：

$$
\boxed{
\text{Pre-training}
\rightarrow
\text{General Capability}
}
$$

$$
\boxed{
\text{Post-training}
\rightarrow
\text{Useful / Aligned / Reasoning / Specialized Capability}
}
$$

论文将 Post-training 统一为五个方向：

1. Fine-Tuning
2. Alignment
3. Reasoning
4. Efficiency
5. Integration and Adaptation

---

# 2. Overview：Post-Training 的演化与数学基础

## 2.1 历史演化

可以压缩为：

$$
\text{BERT/GPT}
\rightarrow
\text{Fine-Tuning}
\rightarrow
\text{RLHF}
\rightarrow
\text{DPO}
\rightarrow
\text{Reasoning RL}
\rightarrow
\text{DeepSeek-R1}
$$

同时还有两条并行路线：

$$
\text{Dense Model}
\rightarrow
\text{MoE}
$$

以及：

$$
\text{Text-only}
\rightarrow
\text{Multimodal}
$$

核心趋势：

> LLM 的研究重点逐渐从“把 Base Model 预训练得更大”，转向“如何在预训练之后塑造模型行为、偏好、推理和适应能力”。

---

## 2.2 PPO

PPO 的核心是：**提高高 Advantage 动作的概率，但限制策略更新幅度。**

Advantage：

$$
\boxed{
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
}
$$

策略概率比：

$$
\boxed{
r_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
}
}
$$

PPO clipped objective：

$$
\boxed{
L^{\mathrm{CLIP}}(\theta)
=
\hat{\mathbb E}_t
\left[
\min
\left(
r_t(\theta)\hat A_t,\;
\operatorname{clip}
(r_t(\theta),1-\epsilon,1+\epsilon)
\hat A_t
\right)
\right]
}
$$

直觉：

$$
\boxed{
\text{好动作：增加概率，但别增加得过猛}
}
$$

传统 PPO 通常还需要 Value / Critic Model：

$$
\boxed{
\phi_{k+1}
=
\arg\min_{\phi}
\mathbb E
\left[
(V_\phi(s_t)-R(s_t))^2
\right]
}
$$

---

## 2.3 RLHF

RLHF：

$$
\boxed{
\text{Human Preference}
\rightarrow
\text{Reward Model}
\rightarrow
\text{RL/PPO}
}
$$

语言模型自回归概率：

$$
\boxed{
\rho(x_0\cdots x_{n-1})
=
\prod_{0\le k<n}
\rho(x_k\mid x_0\cdots x_{k-1})
}
$$

RLHF 的目标：

$$
\boxed{
\pi^*
=
\arg\max_\pi
\mathbb E_{
x\sim D,\,
y\sim\pi(\cdot|x)
}
[R(x,y)]
}
$$

核心：

> 调整模型概率分布，使高 Reward 回答的生成概率增加。

---

## 2.4 DPO

DPO 的目标是绕开显式 Reward Model + PPO。

传统 RLHF：

$$
\text{Preference}
\rightarrow
\text{Reward Model}
\rightarrow
\text{PPO}
\rightarrow
\text{Policy}
$$

DPO：

$$
\boxed{
\text{Preference}
\rightarrow
\text{Policy}
}
$$

KL 约束下最优策略：

$$
\boxed{
\pi_r(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp
\left(
\frac{1}{\beta}r(x,y)
\right)
}
$$

DPO 的核心操作：

- 提高 preferred / chosen response 的相对概率
- 降低 rejected response 的相对概率

---

## 2.5 GRPO

GRPO 是 PPO 的变体。

核心变化：

$$
\boxed{
\text{GRPO 不需要 Critic Model}
}
$$

PPO：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

通常需要 Value Model。

GRPO 则：

$$
\boxed{
\text{同一道题采样多个回答}
\rightarrow
\text{组内比较 Reward}
\rightarrow
\text{构造 Relative Advantage}
}
$$

所以可以粗略记为：

$$
\boxed{
\text{PPO}
=
\text{Policy}
+
\text{Critic}
}
$$

$$
\boxed{
\text{GRPO}
=
\text{Policy}
+
\text{Group Statistics}
}
$$

这也是 GRPO 适合大规模 Reasoning RL 的关键原因之一。

---

# 3. PoLMs for Fine-Tuning

本章分成：

$$
\boxed{
\text{SFT}
+
\text{Adaptive Fine-Tuning}
+
\text{Reinforcement Fine-Tuning}
}
$$

---

## 3.1 Supervised Fine-Tuning

基本模式：

$$
\text{Pretrained LLM}
+
\text{Task Data}
\rightarrow
\text{Fine-tuned LLM}
$$

SFT 本质上仍是标准监督学习。

分类场景中的交叉熵：

$$
\boxed{
L_{\text{fine-tune}}(\theta)
=
-\frac{1}{N}
\sum_{i=1}^{N}
\sum_{j=1}^{C}
y_{ij}\log P(y_j\mid x_i;\theta)
}
$$

全参数微调更新：

$$
\boxed{
\theta_{t+1}
=
\theta_t
-
\eta\nabla_\theta L(\theta_t)
}
$$

重点：

> SFT 不只是“数据越多越好”，数据质量、覆盖度和筛选同样关键。

---

## 3.2 Instruction Tuning

普通 SFT：

$$
x\rightarrow y
$$

Instruction Tuning：

$$
(\text{Instruction},\text{Input})
\rightarrow
\text{Output}
$$

本质：

$$
\boxed{
\text{让模型学习“如何按照自然语言指令做任务”}
}
$$

---

## 3.3 Prefix / Prompt Tuning

核心：

$$
\boxed{
\text{冻结 Base Model，只训练少量额外参数}
}
$$

Prompt-Tuning：

$$
\text{主要在输入层加入 Soft Prompt}
$$

Prefix-Tuning：

$$
\text{在 Transformer 多层加入 Trainable Prefix}
$$

---

## 3.4 Reinforcement Fine-Tuning

传统 SFT：

$$
\text{Human reasoning}
\rightarrow
\text{Model imitation}
$$

ReFT：

$$
\boxed{
\text{Model-generated reasoning}
\rightarrow
\text{Reward}
\rightarrow
\text{Policy Improvement}
}
$$

核心变化：

> 不再只模仿固定 CoT，而是让模型探索多条可能的推理路径。

---

# 4. PoLMs for Alignment

三条核心路线：

$$
\boxed{
\text{RLHF}
\rightarrow
\text{RLAIF}
\rightarrow
\text{DPO}
}
$$

---

## 4.1 RLHF

标准流程：

$$
\boxed{
\text{LLM Outputs}
\rightarrow
\text{Human Preference}
\rightarrow
\text{Reward Model}
\rightarrow
\text{RL}
}
$$

Reward 通常还需要限制模型不要偏离 Reference Model 太远：

$$
\boxed{
r_\theta(x,y)
=
r(x,y)
-
\beta
\log
\frac{\pi(y|x)}{\rho(y|x)}
}
$$

核心矛盾：

$$
\text{Maximize Human Reward}
$$

同时：

$$
\text{Stay close to Reference Model}
$$

---

## 4.2 RLAIF

RLAIF：

$$
\boxed{
\text{RLHF pipeline}
+
\text{AI-generated preference}
}
$$

区别只在 Feedback 来源：

RLHF：

$$
\text{Human}
\rightarrow
\text{Preference}
$$

RLAIF：

$$
\text{AI / LLM Judge}
\rightarrow
\text{Preference}
$$

优势：便宜、可扩展。

风险：AI Judge 的偏差可能被进一步放大。

---

## 4.3 DPO

DPO 直接从偏好数据：

$$
(x,y_w,y_l)
$$

训练 Policy。

核心 loss：

$$
\boxed{
\mathcal L_{\mathrm{DPO}}
=
-
\mathbb E
\left[
\log\sigma
\left(
\beta
\log
\frac{\pi_\theta(y_w|x)}
{\pi_{\mathrm{ref}}(y_w|x)}
-
\beta
\log
\frac{\pi_\theta(y_l|x)}
{\pi_{\mathrm{ref}}(y_l|x)}
\right)
\right]
}
$$

理解成一句话：

$$
\boxed{
\text{提高 chosen，相对降低 rejected}
}
$$

优势：

- 不需要显式 Reward Model
- 不需要 PPO rollout
- 训练系统更简单、更稳定

---

# 5. PoLMs for Reasoning

这是整篇论文最关键的章节之一。

两条路线：

$$
\boxed{
\text{Self-Refine}
+
\text{Reinforcement Learning for Reasoning}
}
$$

---

## 5.1 Self-Refine

统一模式：

$$
\text{Generate}
\rightarrow
\text{Evaluate}
\rightarrow
\text{Feedback}
\rightarrow
\text{Revise}
$$

作者从两个维度分类：

1. 是否使用 External Tool
2. 是否 Fine-Tuning

得到：

| | 不训练参数 | 训练参数 |
|---|---|---|
| 不用外部工具 | Intrinsic | Fine-tuned Intrinsic |
| 使用外部工具 | External | Fine-tuned External |

核心：

> 让模型通过自检、外部工具、执行器、搜索或训练后的纠错能力改善已有推理。

---

## 5.2 Reasoning as MDP

把推理看成：

$$
\boxed{
M=(S,A,P,R,\gamma)
}
$$

也就是 Markov Decision Process。

状态：

$$
s_t=(x,a_1,\ldots,a_{t-1},\text{context})
$$

动作：

$$
a_t=\text{next reasoning move}
$$

目标：

$$
\boxed{
J(\theta)
=
\mathbb E_{\pi_\theta}
\left[
\sum_{t=1}^{T}
\gamma^tR(s_t,a_t)
\right]
}
$$

核心思想：

$$
\boxed{
\text{Reasoning}
\approx
\text{Sequential Decision Making}
}
$$

---

## 5.3 Reasoning Reward

常见四类：

1. **Binary / Outcome Reward**
   - 最终答案对：1
   - 错：0

2. **Step-wise Reward**
   - 对每个推理步骤打分

3. **Self-Consistency Reward**
   - 多条 reasoning path 达成一致时加分

4. **Preference-Based Reward**
   - 通过 Human / AI Reward Model 评价 reasoning quality

核心矛盾：

$$
\boxed{
\text{Sparse but Reliable}
\quad\text{vs}\quad
\text{Dense but Noisy}
}
$$

---

## 5.4 DeepSeek-R1 / R1-Zero

R1-Zero：

$$
\boxed{
\text{Base Model}
\rightarrow
\text{GRPO}
\rightarrow
\text{Reasoning Model}
}
$$

关键思想：

> 不依赖大量人工 CoT，直接通过 RL 让模型自己探索 reasoning behavior。

DeepSeek-R1 的完整路线则更接近：

$$
\text{Base Model}
\rightarrow
\text{Cold-start CoT SFT}
\rightarrow
\text{Reasoning RL}
\rightarrow
\text{Reasoning Data}
\rightarrow
\text{SFT}
\rightarrow
\text{Second RL}
$$

核心意义：

$$
\boxed{
\text{“对齐模型”}
\rightarrow
\text{“训练模型进行推理”}
}
$$

---

# 6. PoLMs for Efficiency

三条主线：

$$
\boxed{
\text{Model Compression}
+
\text{PEFT}
+
\text{Knowledge Distillation}
}
$$

---

## 6.1 Quantization

把高精度数值转换为低 bit：

$$
\text{FP16/FP32}
\rightarrow
\text{INT8/INT4/INT2}
$$

核心：

$$
\boxed{
\text{参数数量不变，但每个参数更便宜}
}
$$

常见类别：

- Weight-only Quantization
- Weight-Activation Quantization
- KV-Cache Quantization

---

## 6.2 Pruning

核心：

$$
\boxed{
\text{直接删除不重要参数或结构}
}
$$

两类：

- Unstructured Pruning：删单个 weight
- Structured Pruning：删 neuron / channel / attention head / block

Structured Pruning 更容易获得真实硬件加速。

---

## 6.3 LoRA / PEFT

LoRA：

$$
\Delta W=W_{\text{up}}W_{\text{down}}
$$

因此：

$$
\boxed{
h_{\text{out}}
=
W_0h_{\text{in}}
+
\alpha W_{\text{up}}W_{\text{down}}h_{\text{in}}
}
$$

核心：

$$
\boxed{
\text{不重新学习整个大矩阵，只学习低秩增量}
}
$$

---

## 6.4 Knowledge Distillation

基本结构：

$$
\boxed{
\text{Teacher}
\rightarrow
\text{Knowledge}
\rightarrow
\text{Student}
}
$$

经典 KD：

$$
\boxed{
L_{\mathrm{KD}}
=
\alpha L_{\mathrm{CE}}
+
(1-\alpha)L_{\mathrm{KL}}
}
$$

DeepSeek-R1 reasoning distillation：

$$
\boxed{
\text{Large Model RL}
\rightarrow
\text{Reasoning Data}
\rightarrow
\text{Small Model SFT}
}
$$

也就是：

> 大模型负责“探索”，小模型负责“模仿”。

---

# 7. PoLMs for Integration and Adaptation

三部分：

$$
\boxed{
\text{Multimodal}
+
\text{Domain Adaptation}
+
\text{Model Merging}
}
$$

---

## 7.1 Multimodal Integration

典型结构：

$$
\boxed{
\text{Modality Encoder}
\rightarrow
\text{Connector}
\rightarrow
\text{LLM}
}
$$

Connector 三类：

- Projection-based：直接映射
- Query-based：通过 learnable query 提取信息
- Fusion-based：在模型内部进行深层跨模态融合

可以记成：

$$
\boxed{
\text{Projection：映射进去}
}
$$

$$
\boxed{
\text{Query：主动提取}
}
$$

$$
\boxed{
\text{Fusion：内部深度融合}
}
$$

---

## 7.2 Knowledge Editing vs RAG

Knowledge Editing：

$$
\boxed{
\theta'=\theta+\Delta\theta
}
$$

目标：

$$
L(\theta';D_{\text{new}})\rightarrow\min
$$

同时约束：

$$
L(\theta';D_{\text{old}})
\le
L(\theta;D_{\text{old}})+\epsilon
$$

也就是：

$$
\boxed{
\text{更新新知识，但尽量不破坏旧知识}
}
$$

RAG：

$$
\boxed{
\text{Query}
\rightarrow
\text{Retriever}
\rightarrow
\text{Documents}
\rightarrow
\text{LLM}
}
$$

区别：

$$
\boxed{
\text{Knowledge Editing：知识写进参数}
}
$$

$$
\boxed{
\text{RAG：知识保留在外部数据库}
}
$$

---

## 7.3 Model Merging

统一形式：

$$
\boxed{
M'
=
F_{\text{merge}}
(M_1,M_2,\ldots,M_n)
}
$$

三层：

1. Weight-Level
2. Output-Level
3. Model-Level / MoE Routing

Weight-Level：

$$
\boxed{
\theta'
=
\sum_{k=1}^{n}\alpha_k\theta_k
}
$$

Task Vector：

$$
\boxed{
\tau_t
=
\Theta^{(t)}-\Theta^{(0)}
}
$$

可将 Fine-tuned Model 理解成：

$$
\boxed{
\text{Base Model}
+
\text{Task Vector}
}
$$

---

# 8. Datasets

Post-training 数据按来源分成三类：

$$
\boxed{
\text{Human-Labeled}
+
\text{Distilled}
+
\text{Synthetic}
}
$$

---

## 8.1 Human-Labeled

特点：

- 质量高
- 能表达细粒度偏好
- 成本高
- 扩展性差

适合：

- SFT
- Preference Data
- RLHF

---

## 8.2 Distilled Data

核心：

$$
\boxed{
\text{大量原始交互}
\rightarrow
\text{筛选 / 提炼}
\rightarrow
\text{高价值训练数据}
}
$$

例如 ShareGPT、HC3。

---

## 8.3 Synthetic Data

核心：

$$
\boxed{
\text{Seed Data}
\rightarrow
\text{LLM}
\rightarrow
\text{Large Synthetic Dataset}
}
$$

例如：

- Self-Instruct
- Alpaca
- Magpie

优势：

$$
\text{低成本}+\text{可扩展}+\text{隐私友好}
$$

风险：

$$
\text{Bias Propagation}
+
\text{Model Errors}
+
\text{Low Diversity}
$$

因此更合理的是：

$$
\boxed{
\text{Human Data}
+
\text{Synthetic Data}
}
$$

---

# 9. Applications

作者把应用分成三类：

$$
\boxed{
\text{Professional Domains}
+
\text{Technical Reasoning}
+
\text{Understanding \& Interaction}
}
$$

---

## 9.1 Professional Domains

主要包括：

- Legal
- Healthcare
- Finance
- Mobile / GUI Agents

共同模式：

$$
\boxed{
\text{General LLM}
+
\text{Domain Data / RAG / SFT / Alignment}
\rightarrow
\text{Domain-specific LLM}
}
$$

---

## 9.2 Technical Reasoning

主要是：

$$
\boxed{
\text{Mathematics}
+
\text{Code}
}
$$

Math：

$$
\text{SFT}
+
\text{GRPO / RL}
$$

Code：

$$
\text{Code Data}
+
\text{SFT}
+
\text{Reasoning}
+
\text{Execution Feedback}
$$

---

## 9.3 Understanding and Interaction

包括：

- Recommendation
- Speech Conversation
- Video Understanding

本质是：

$$
\boxed{
\text{Domain Adaptation}
+
\text{Multimodal Integration}
+
\text{Alignment}
}
$$

---

# 10. Open Problems and Future Directions

论文提出七个主要开放问题。

## 10.1 Beyond Large-Scale RL

当前 RL 仍过度依赖：

- binary reward
- sparse reward
- 人工反馈
- benchmark-oriented optimization

未来需要：

$$
\boxed{
\text{Multi-objective RL}
+
\text{Self-supervised Consistency}
+
\text{Domain Priors}
}
$$

---

## 10.2 Scalability

问题：

$$
\boxed{
\text{Large-Scale RL 太贵}
}
$$

方向：

- lightweight RL
- 更高效 GRPO
- PEFT for RL
- Distillation
- distributed / federated post-training

---

## 10.3 Ethical Alignment and Bias

问题：

- Reward / Preference Data 自带 Bias
- 不同文化、领域的“正确偏好”不完全一致
- 过度安全会牺牲 Utility
- 安全不足又会产生风险

因此需要：

$$
\boxed{
\text{Fairness-aware RL}
+
\text{Multi-stakeholder Preference}
+
\text{Adaptive Safety}
}
$$

---

## 10.4 Multimodal Reasoning

真正的多模态 reasoning 不能只靠：

$$
\text{Vision Encoder}
+
\text{LLM}
$$

而需要：

$$
\boxed{
\text{Text}
+
\text{Image}
+
\text{Audio}
+
\text{Video}
\rightarrow
\text{Unified Reasoning}
}
$$

---

## 10.5 Context-Adaptive Trustworthiness

未来 Alignment 不应该是一套固定规则。

而应该动态调整：

$$
\boxed{
\text{Safety}
\leftrightarrow
\text{Utility}
}
$$

例如医疗和创作任务显然不应该采用完全相同的安全策略。

---

## 10.6 Democratization

最先进 Post-training 技术成本太高。

未来需要：

- Open-source tools
- 更便宜的 RL
- Synthetic Data
- PEFT
- Distillation

使小团队也能使用先进 Post-training 方法。

---

## 10.7 Creative Intelligence + System 2 Thinking

当前 reasoning model 很擅长：

$$
\boxed{
\text{Step-by-step Logical Analysis}
}
$$

但仍不擅长：

- 产生真正新颖的假设
- 跨领域组合概念
- 开放式科学发现
- 创造性战略规划

因此未来的重要问题是：

$$
\boxed{
\text{Logical Reasoning}
+
\text{Creative Intelligence}
}
$$

---

# 11. Conclusion

论文最终将 Post-training 概括为：

$$
\boxed{
\text{Fine-Tuning}
+
\text{Alignment}
+
\text{Reasoning}
+
\text{Efficiency}
+
\text{Integration \& Adaptation}
}
$$

作者认为 LLM 的发展已经从：

$$
\text{Instruction Following / Alignment}
$$

逐渐转向：

$$
\boxed{
\text{Reasoning-centric Post-training}
}
$$

但未来不能只追求更强 Reasoning，还需要同时优化：

$$
\boxed{
\text{Reasoning}
+
\text{Efficiency}
+
\text{Ethical Robustness}
+
\text{Adaptability}
}
$$

---

# 12. 最终技术地图

整篇论文可以用下面这一条链条记忆：

$$
\boxed{
\text{Base LLM}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{SFT：学会按照指令回答}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{RLHF / DPO：学会什么回答更符合偏好}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{Reasoning RL / GRPO：学会如何探索出正确答案}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{PEFT / Quantization / Distillation：降低成本}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{RAG / Knowledge Editing / Multimodal / Model Merging：扩展能力边界}
}
$$

最终得到：

$$
\boxed{
\text{Useful}
+
\text{Aligned}
+
\text{Reasoning}
+
\text{Efficient}
+
\text{Domain/Multimodal LLM}
}
$$

---

# 13. 快速复习：最值得记住的概念

| 概念 | 一句话理解 |
|---|---|
| Pre-training | 学通用知识与基础能力 |
| SFT | 模仿高质量答案 |
| Instruction Tuning | 学会按照自然语言指令做任务 |
| RLHF | Human Preference → Reward → RL |
| RLAIF | 用 AI Feedback 替代/补充 Human Feedback |
| DPO | 不显式训练 Reward Model，直接从 Preference 优化 Policy |
| PPO | 通过 clipped update 稳定策略优化 |
| GRPO | 用组内相对 reward 替代大型 Critic |
| Self-Refine | Generate → Check → Revise |
| Reasoning RL | 把推理建模为 Sequential Decision Making |
| LoRA | 只学习低秩参数增量 |
| Quantization | 用更低 bit 表示参数/激活 |
| Distillation | 大模型教小模型 |
| RAG | 把知识保留在模型外部并动态检索 |
| Knowledge Editing | 直接修改模型内部知识 |
| Model Merging | 把多个专长模型合并成统一模型 |
| Synthetic Data | 用 LLM 自动生成后训练数据 |

---

# 14. 阅读优先级

如果以后重新复习这篇论文，建议优先级为：

**第一优先级：**
- PPO
- RLHF
- DPO
- GRPO
- Reasoning as MDP
- Reward Design
- DeepSeek-R1 / R1-Zero

**第二优先级：**
- SFT
- LoRA / PEFT
- Quantization
- Knowledge Distillation
- RAG

**第三优先级：**
- Multimodal Connector
- Knowledge Editing
- Model Merging
- Dataset taxonomy
- Applications

---

## 最后一条记忆

如果只能记一句：

$$
\boxed{
\text{Post-training 的本质，是把“有能力的 Base Model”塑造成“真正可用的模型”。}
}
$$

---

## 参考文献

[1] Gu, Y., Zhang, P., et al. *A Survey on Post-Training of Large Language Models*. arXiv preprint, 2025.
