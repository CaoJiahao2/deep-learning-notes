# SayCan：把语言模型 grounding 到机器人可供性

> SayCan 回答了一个关键问题：LLM 知道“能做什么”，机器人知道“适合做什么”，两者相乘才能变成可执行的物理任务。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (2022) |
| 机构 | Google Robotics / Everyday Robots / UC Berkeley 等 |
| 关键词 | LLM 规划、affordance、value function、任务分解 |
| 定位 | VLA 出现前的经典分层具身语言系统 |

---

## 1 背景与问题

在 SayCan 之前，LLM 已经能理解“帮我清理桌面”，但直接让它输出机器人动作会遇到两类错误：

1. **语义幻觉**：LLM 可能提出机器人不会做的技能；
2. **物理不可行**：即使技能存在，当前物体位置和场景状态也可能不允许执行。

SayCan 的核心观察是：语言模型擅长给出**语义上合理的候选技能**，机器人策略与价值函数擅长判断**物理上成功的概率**。因此不要让 LLM 直接控制电机，而让它在一个受约束的技能库中选择。


---

## 2 核心贡献

1. 提出 **LLM as high-level planner + learned skill policies as executor** 的分层范式；
2. 用 affordance value function 给每个候选技能打分，把概率乘进 LLM 的 token 概率；
3. 将自然语言目标转换为可执行技能链，是后来 PaLM-E、RT-2、Helix、π0.5 等高阶规划的直接前身；
4. 说明“通用语言智能 + 可学习的物理可供性”可以互补，而不必一开始就端到端。


---

## 3 核心原理与方法详解

设技能库为 $\mathcal{S}=\{s_1,\dots,s_N\}$，LLM 根据历史与任务 $\ell$ 给出第 $i$ 个技能的语言概率：

$$
P_{\text{LLM}}(s_i\mid \ell, h_t)
$$

技能策略对应的价值函数估计该技能在当前观测 $o_t$ 下的成功概率：

$$
P_{\text{afford}}(s_i\mid o_t)
$$

SayCan 在每步选择：

$$
s_t^*
=\operatorname*{arg\,max}_{s_i\in\mathcal{S}}
P_{\text{LLM}}(s_i\mid \ell,h_t)
\cdot P_{\text{afford}}(s_i\mid o_t).
$$

选择后执行该技能直到终止，再把结果写入历史 $h_{t+1}$，重复上述过程直到任务完成。


---

## 4 数据与训练

- **技能库**：抓取、放置、移动到位置、开门等离散技能，每个技能由一个模仿学习策略实现；
- **LLM**：采用 few-shot prompting，将任务描述、当前场景与历史输入模型；
- **affordance function**：由强化学习价值函数或成功分类器估计；
- **终止条件**：LLM 可输出终止符号，或技能链达到最大长度；
- **训练重点**：SayCan 不训练 LLM 本体，而是训练技能与可供性，因此工程边界清晰。


---

## 5 实验、结果与解读

论文在真实厨房场景中执行长程任务，例如“拿一杯可乐”。LLM 能规划到冰箱取可乐，但不会直接输出关节角度；价值函数会过滤掉当前不可达、无法成功或机器人没有学过的动作。

关键结果：

- 相比只用 LLM 选择动作，任务可行性显著提升；
- 相比只按启发式排序，更符合自然语言意图；
- 系统对同义表达和未见物品描述有一定泛化，但泛化范围仍受技能库限制。


---

## 6 精读要点与实践建议

- **读法建议**：先把 SayCan 理解为概率乘积式任务 grounding，而不是“LLM 机器人控制器”；
- **工程启示**：当低层技能可靠但组合复杂时，LLM 做调度是低成本方案；
- **数学本质**：将语义先验与物理先验解耦，再通过乘法融合；
- **与现代 VLA 的关系**：RT-2 把这条链压进端到端模型，π0.5 / Helix 又在推理层面重新引入分层；
- **适用判断**：技能库稳定、任务组合开放的仓库和家庭场景最适合。


---

## 7 局限与后续影响

- 技能库需要人工定义和分别训练；
- 语言概率与成功概率的量纲不同，乘法融合依赖经验校准；
- 无法自动创建新技能，只能组合已有技能；
- 长程错误会累积，缺少显式世界模型验证；
- 后续 PaLM-E 尝试把多模态状态放入 LLM，RT-2 则进一步把动作 token 化并端到端训练。


---

## 相关阅读

- [RT-1 论文精读](./rt-1.md)
- [PaLM-E 论文精读](./palm-e.md)
- [具身智能前沿专题](../frontier.md)

---

## 参考文献

1. Ahn et al. *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*, CoRL 2022.
2. Brohan et al. *RT-1: Robotics Transformer for Real-World Control at Scale*, 2022.
3. Driess et al. *PaLM-E: An Embodied Multimodal Language Model*, ICML 2023.

