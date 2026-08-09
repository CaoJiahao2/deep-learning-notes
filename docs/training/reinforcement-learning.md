# 强化学习与对齐技术详解

> 从贝尔曼方程到 GRPO 与推理时强化学习，强化学习在深度学习中的完整演进。

---

## 目录

1. [概述](#1)
2. [MDP 基础](#2-mdp)
3. [Q-learning](#3-q-learning)
4. [策略梯度](#4)
5. [PPO](#5-ppo)
6. [RLHF 全流程](#6-rlhf)
7. [DPO](#7-dpo)
8. [新进展：GRPO 与 RLVR](#8-grpo-rlvr)
9. [实践建议](#9)
10. [参考文献](#10)

---

## 1. 概述

强化学习（Reinforcement Learning, RL）研究智能体在与环境的交互中通过试错最大化长期累积奖励的问题。它不同于监督学习：没有标注的"正确动作"，只有延迟的标量奖励信号。强化学习在深度学习中的角色经历了三次重要转向：

- **游戏智能**：DQN（Mnih et al., 2015）在 Atari 游戏上达到人类水平，AlphaGo（Silver et al., 2016）击败围棋世界冠军。这一阶段证明深度网络可以逼近值函数或策略，解决高维状态空间问题。
- **人类偏好对齐**：RLHF（Ouyang et al., 2022, InstructGPT）将人类偏好建模为奖励信号，再用 PPO 微调语言模型，使大语言模型（LLM）的行为与人类意图对齐，是 ChatGPT 成功的关键技术之一。
- **推理时强化学习**：RLVR（DeepSeek-R1, 2025）用可验证奖励（数学题答案、代码测试通过）替代人类标注，催生了 o1/R1 类"长思维链推理"模型。

本文的叙事线索为：MDP → Q-learning → 策略梯度 → PPO → RLHF → DPO → GRPO/RLVR。每一节都建立在前一节的概念之上，最终串起从经典控制到现代对齐的完整图景。

---

## 2. MDP 基础

### 2.1 马尔可夫决策过程

马尔可夫决策过程（Markov Decision Process, MDP）由五元组 $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ 定义：

- $\mathcal{S}$：状态空间
- $\mathcal{A}$：动作空间
- $P(s'|s,a)$：状态转移概率，满足马尔可夫性 $P(s_{t+1}|s_t,a_t,s_{t-1},\dots)=P(s_{t+1}|s_t,a_t)$
- $r(s,a,s')$：奖励函数
- $\gamma \in [0,1)$：折扣因子

策略 $\pi(a|s)$ 是状态到动作分布的映射。给定策略，回报（return）为：

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

折扣因子 $\gamma$ 既保证无限长回报收敛，也反映对未来奖励的不确定性偏好。与之对应的有效视野（effective horizon）约为 $1/(1-\gamma)$：若 $\gamma=0.99$，智能体约关注未来 $100$ 步内的奖励。

### 2.2 状态值与动作值

状态值函数 $V^\pi(s)$ 与动作值函数 $Q^\pi(s,a)$ 定义为期望回报：

$$V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right]$$

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]$$

### 2.3 贝尔曼方程

通过对 $G_t = r_{t+1} + \gamma G_{t+1}$ 取条件期望，可得贝尔曼期望方程：

$$V^\pi(s)=\sum_a \pi(a|s)\sum_{s'}P(s'|s,a)\left[r(s,a,s')+\gamma V^\pi(s')\right]$$

$$Q^\pi(s,a)=\sum_{s'}P(s'|s,a)\left[r(s,a,s')+\gamma \sum_{a'}\pi(a'|s')Q^\pi(s',a')\right]$$

两者的关系为：

$$V^\pi(s)=\sum_a \pi(a|s)Q^\pi(s,a)$$

### 2.4 贝尔曼最优方程

最优策略 $\pi^*$ 满足 $V^{\pi^*}(s) \ge V^\pi(s), \forall \pi$。对应贝尔曼最优方程：

$$V^*(s)=\max_a \sum_{s'}P(s'|s,a)\left[r(s,a,s')+\gamma V^*(s')\right]$$

$$Q^*(s,a)=\sum_{s'}P(s'|s,a)\left[r(s,a,s')+\gamma \max_{a'}Q^*(s',a')\right]$$

最优策略可由 $Q^*$ 贪心给出 $\pi^*(a|s)=1$ 当 $a=\arg\max_{a'}Q^*(s,a')$。这是值迭代、Q-learning 等方法的根基。

---

## 3. Q-learning

### 3.1 TD 学习与 off-policy

Q-learning（Watkins, 1989）是时序差分（Temporal Difference, TD）学习的代表。它直接学习最优 $Q^*$，且为 off-policy：用任意行为策略采集的样本更新目标策略。更新规则为：

$$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma \max_{a'}Q(s',a')-Q(s,a)\right]$$

其中括号内项称为 TD 误差 $\delta = r+\gamma \max_{a'}Q(s',a')-Q(s,a)$。注意目标 $r+\gamma \max_{a'}Q(s',a')$ 已用 $\max$，而非当前策略的动作，这正是 off-policy 的体现。

### 3.2 DQN

当状态空间连续或高维时（如 Atari 像素），用神经网络 $Q_\theta(s,a)$ 近似 $Q$。DQN（Mnih et al., 2015, Nature）引入两项关键技术稳定训练：

- **经验回放（Experience Replay）**：将转移 $(s,a,r,s')$ 存入回放缓冲区，训练时随机采样批量，打破样本间时间相关性，提高数据利用率。
- **目标网络（Target Network）**：用一份延迟更新的网络 $Q_{\theta^-}$ 计算 TD 目标 $r+\gamma \max_{a'}Q_{\theta^-}(s',a')$，避免自举中目标随 $\theta$ 同步抖动导致的发散。

```python
# DQN 训练伪代码
for step in range(num_steps):
    # epsilon-greedy 采样
    if random() < eps:
        a = sample_action(env)
    else:
        a = argmax(Q_theta(state))
    s_next, r, done = env.step(a)
    replay_buffer.add((s, a, r, s_next, done))

    # 从回放缓冲区采样
    batch = replay_buffer.sample(batch_size)
    s_b, a_b, r_b, s_next_b, done_b = batch
    # 目标网络计算 TD 目标（不参与梯度）
    y = r_b + gamma * (1 - done_b) * max_a Q_theta_minus(s_next_b, a)
    loss = mse(Q_theta(s_b, a_b), y)
    loss.backward()
    optimizer.step()

    # 每 C 步同步目标网络
    if step % C == 0:
        theta_minus = theta
```

---

## 4. 策略梯度

### 4.1 直接优化策略

值函数方法（Q-learning）在动作空间离散且不大时有效，但对连续动作或巨大动作空间（如 LLM 词表）困难。策略梯度方法直接参数化策略 $\pi_\theta(a|s)$，对期望回报做梯度上升。

### 4.2 REINFORCE 推导

设轨迹 $\tau=(s_0,a_0,r_1,s_1,a_1,\dots)$，目标函数：

$$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t \gamma^t r_t\right]$$

利用对数求导（得分函数估计），可得：

$$\nabla_\theta J(\theta)=\mathbb{E}_{\tau}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t\right]$$

直观解释：$\nabla_\theta \log\pi_\theta(a_t|s_t)$ 指向增大 $a_t$ 概率的方向，乘以回报 $G_t$ 后，高回报的动作概率被提升，低回报的被压低。可用 Monte Carlo 样本估计，得到 REINFORCE 算法（Williams, 1992）。

### 4.3 基线与方差减少

REINFORCE 方差极大。引入基线 $b(s_t)$（与动作无关）不改变期望但降低方差，因为 $\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a|s_t)b(s_t)]=b(s_t)\nabla_\theta\sum_a\pi(a|s_t)=0$：

$$\nabla_\theta J(\theta)=\mathbb{E}_{\tau}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\,(G_t - b(s_t))\right]$$

自然选择 $b(s_t)=V(s_t)$，差值 $A_t = G_t - V(s_t)$ 称为优势（advantage）。用学得的 $V_\phi$ 作基线，并用网络估计 $A_t$，即得到 Actor-Critic 架构（Konda & Tsitsiklis）：actor 更新策略，critic 评估状态值。其同步/异步变体即 A2C/A3C（Mnih et al., 2016）。

---

## 5. PPO

### 5.1 信任区域的动机

策略梯度是 on-policy 方法：样本必须来自当前策略。一旦策略更新，旧样本即失效，样本效率低。若更新步长过大，策略急剧变化，目标估计失真甚至发散。TRPO（Schulman et al., 2015）用 KL 约束限制更新步长（信任区域），但需二阶优化，工程复杂。PPO（Schulman et al., 2017）以一阶裁剪近似达到同样目的，简洁稳定，成为 RLHF 的事实标准。

### 5.2 重要性采样与裁剪目标

设新旧策略分别为 $\pi_\theta, \pi_{\theta_{old}}$，重要性采样比：

$$r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

PPO 的 Clipped 目标：

$$L^{CLIP}(\theta)=\mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\; \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)\right]$$

裁剪机理：当 $A_t>0$（动作优于平均），$r_t$ 被允许增大到 $1+\epsilon$ 后封顶，防止过度利用；当 $A_t<0$，$r_t$ 被压到 $1-\epsilon$ 为止，防止过度抑制。$\min$ 取两者较小者形成下界，使目标对过大更新保持悲观估计，从而限制策略步长。

### 5.3 完整损失

PPO 实践中将三项组合训练：

```text
L_total = L_CLIP                        # 策略目标（裁剪）
        - c1 * L_VF                     # 价值函数回归损失
        - c2 * S[pi]                    # 策略熵奖励，鼓励探索

L_VF   = (V_theta(s) - G_hat)^2
S[pi]  = - sum_a pi(a|s) log pi(a|s)
```

策略项驱动策略向高优势动作倾斜；价值项保证 critic 准确估计回报；熵项防止策略过早坍缩为确定性策略，保留探索。为何 RLHF 选 PPO：on-policy + 裁剪保证策略漂移可控、训练稳定，且天然支持奖励模型作为回报来源，工程上成熟可靠。

---

## 6. RLHF 全流程

### 6.1 三阶段概览

RLHF（Ouyang et al., 2022, InstructGPT）分三阶段，把"人类偏好"转化为可控的微调信号：

```text
[阶段1 SFT]   预训练模型 --(高质量指令数据监督微调)--> SFT 模型 pi_SFT
[阶段2 RM]    偏好数据 (x, y_w, y_l) --(Bradley-Terry 训练)--> 奖励模型 r_phi
[阶段3 PPO]   pi_SFT 作为初值与参考, 用 r_phi 作奖励,
              KL(pi_theta || pi_ref) 约束, PPO 优化 --> 对齐模型 pi_RLHF
```

### 6.2 SFT

监督微调（Supervised Fine-Tuning）用高质量指令-回复对训练模型，使其先学会遵循指令的格式与基本能力，作为后续 RL 的起点。不经过 SFT 直接做 RL 通常难以收敛。

### 6.3 奖励模型

人类偏好数据以成对比较形式 $(x, y_w, y_l)$ 给出，$y_w$ 为偏好（win）回复，$y_l$ 为非偏好（lose）。基于 Bradley-Terry 模型，假设人类偏好概率仅取决于奖励差：

$$P(y_w \succ y_l \mid x) = \sigma\left(r(x,y_w)-r(x,y_l)\right)$$

RM 损失为负对数似然：

$$L_{RM}=-\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(r(x,y_w)-r(x,y_l)\right)\right]$$

### 6.4 PPO 对齐与 KL 约束

用 RM 给出的 $r_\phi(x,y)$ 作奖励，PPO 优化策略，同时用参考模型 $\pi_{ref}$（通常即 SFT 模型）做 KL 约束防止策略漂移：

$$\max_{\pi_\theta}\mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta}\left[r_\phi(x,y)-\beta\,D_{KL}\left(\pi_\theta(\cdot|x)\,\|\,\pi_{ref}(\cdot|x)\right)\right]$$

### 6.5 讨论

- **KL 系数 $\beta$**：过小则策略漂移过大，可能输出 RM 的高奖励但低质量文本（reward hacking）；过大则退化为 SFT，无法学到偏好信号。需谨慎调节。
- **RM 局限**：RM 是对人类偏好的有损近似，易被过度优化（reward over-optimization），尤其在 PPO 后期奖励持续上升但真实质量下降；需结合 RM 训练分布与对 KL 的强约束缓解。
- **数据成本**：偏好数据昂贵且标注者间一致性有限。

---

## 7. DPO

### 7.1 核心洞察

DPO（Rafailov et al., 2023）发现：RLHF 中的 KL 约束最优策略有闭式解。对 6.4 中的目标做变分推导，最优策略满足：

$$\pi^*(y|x)\propto \pi_{ref}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)$$

于是可反解出奖励：

$$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}$$

### 7.2 直接损失

将上式代入 Bradley-Terry 损失，奖励被策略直接表达，无需显式训练 RM：

$$L_{DPO}=-\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

直觉：DPO 直接增大偏好回复相对参考的对数比、压低非偏好回复，等价于隐式地优化奖励并满足 KL 约束。

### 7.3 DPO vs PPO 对比

| 维度 | PPO (RLHF) | DPO |
| --- | --- | --- |
| 是否需要 RM | 需要 | 不需要 |
| 在线/离线 | 在线采样 | 离线偏好数据 |
| 训练稳定性 | 需调参，易抖动 | 简洁稳定，分类式训练 |
| 计算成本 | 高（多次 rollout） | 低 |
| 探索能力 | 强，可发现 RM 外的分布 | 弱，受限于偏好数据 |

DPO 以极简实现获得接近 PPO 的效果，是当前离线对齐的主流基线；但当需要在线探索或任务奖励可程序化验证时，PPO 类方法仍占优。

---

## 8. 新进展：GRPO 与 RLVR

### 8.1 GRPO

GRPO（Group Relative Policy Optimization, Shao et al., 2024, DeepSeekMath）针对 PPO 在大模型上维护 value 网络开销巨大的问题。对同一 prompt $x$ 采样一组 $G$ 个回复 $\{y_i\}$，得到奖励 $\{r_i\}$，用组内相对优势代替 critic：

$$A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}$$

目标类似 PPO 的裁剪形式，但无 value 网络：

$$L^{GRPO}(\theta)=\mathbb{E}_{x,\{y_i\}}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(r_i(\theta)A_i,\,\text{clip}(r_i(\theta),1-\epsilon,1+\epsilon)A_i\right)\right] - \beta\,D_{KL}(\pi_\theta\|\pi_{ref})$$

其中 $r_i(\theta)=\pi_\theta(y_i|x)/\pi_{\theta_{old}}(y_i|x)$。组内归一化作为 baseline 起到方差减少作用。省去 critic 网络显著降低显存与工程复杂度，使其特别适合百亿至千亿参数大模型的 RL 微调。

### 8.2 RLVR

RLVR（Reinforcement Learning with Verifiable Rewards）的关键转变：奖励不再来自人类或 RM，而来自可验证规则——数学题答案与标准解比对、代码通过单元测试等。DeepSeek-R1（2025）以 GRPO + 规则奖励训练，模型自发涌现长思维链（long chain-of-thought）、自我反思与回溯行为。

这一范式的意义有三：

1. **奖励精确**：可验证奖励无歧义、无 RM 过度优化问题，训练信号干净。
2. **数据成本极低**：无需人类标注偏好，只需带标准答案的题目集。
3. **test-time compute scaling**：模型学会在推理时生成更长更深的思考链，性能随"思考预算"近似可预测地提升，开辟了除参数规模外的新扩展维度。

### 8.3 方法对比

| 方法 | 奖励来源 | 在线/离线 | 显存 | 适合任务 |
| --- | --- | --- | --- | --- |
| SFT | 无（监督标签） | 离线 | 低 | 格式与基础能力 |
| RLHF (PPO) | 人类偏好 + RM | 在线 | 高（actor+critic+ref+RM） | 开放域对齐 |
| DPO | 人类偏好（隐式） | 离线 | 低 | 离线偏好对齐 |
| GRPO | RM 或规则 | 在线 | 中（无 critic） | 大模型数学/代码 RL |
| RLVR | 可验证规则 | 在线 | 中 | 推理任务 |

---

## 9. 实践建议

### 9.1 选型指南

- **追求开放域人类对齐、有偏好数据**：优先 DPO，工程简单、稳定；若需在线探索再考虑 PPO。
- **推理任务（数学、代码）且奖励可验证**：GRPO + 规则奖励（RLVR）为当前最优实践。
- **仅打基础、改格式**：SFT 即可，不必引入 RL 复杂度。

### 9.2 建议清单

1. **先 SFT 再 RL**：未经 SFT 的预训练模型直接做 RL 几乎不可收敛，SFT 提供良好初始化与参考策略。
2. **课程学习**：由易到难安排任务，避免早期奖励稀疏导致策略崩塌；RLVR 中按难度分桶采样。
3. **奖励设计**：可验证奖励优先；用 RM 时关注分布偏移，定期用新数据更新 RM，并对 KL 系数 $\beta$ 做扫描。
4. **评估不只看 reward**：reward 上升未必代表真实质量提升，需用 held-out 评测集、人类评估或独立基准，警惕 reward hacking。
5. **监控 KL**：训练中持续跟踪 $D_{KL}(\pi_\theta\|\pi_{ref})$，漂移过大通常是异常信号。
6. **采样温度与探索**：GRPO/PPO 阶段保持足够采样温度与组数 $G$，确保优势估计有区分度。

---

## 10. 参考文献

[1] R. Bellman, *Dynamic Programming*, Princeton University Press, 1957.

[2] C. J. C. H. Watkins, *Learning from Delayed Rewards*, PhD Thesis, University of Cambridge, 1989.

[3] V. Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015.

[4] R. J. Williams, *Simple statistical gradient-following algorithms for connectionist reinforcement learning*, Machine Learning, 1992.

[5] V. R. Konda and J. N. Tsitsiklis, *Actor-Critic Algorithms*, NeurIPS, 2000.

[6] V. Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning (A3C)*, ICML, 2016.

[7] J. Schulman et al., *Trust Region Policy Optimization (TRPO)*, ICML, 2015.

[8] J. Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347, 2017.

[9] L. Ouyang et al., *Training language models to follow instructions with human feedback (InstructGPT)*, NeurIPS, 2022.

[10] R. Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS, 2023.

[11] Z. Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)*, arXiv:2402.03300, 2024.

[12] DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025.
