# 策略学习：从行为克隆到离线 RL 与生成式策略

> 模仿学习把"专家经验"蒸馏进策略；扩散与 Flow Matching 在动作空间上自然地建模多模态分布；离线 RL 与 RL 后训练则回答"没有新交互数据时如何继续变强"。

---

## 目录

1. [行为克隆 (BC)](#1)
2. [Diffusion Policy](#2)
3. [Flow Matching Policy](#3)
4. [3D Diffusion Policy (DP3)](#4)
5. [离线强化学习](#5)
6. [人干预 RL (HIL-SERL)](#6)
7. [BC + RL 混合范式与 VLA 后训练](#7)
8. [参考文献](#8)

---

## 1 行为克隆 (BC) {#1}

BC 从专家演示数据集 $\mathcal{D} = \{(o_i, a_i)\}$ 中监督学习策略。

### 1.1 损失函数

连续动作用回归损失：

$$\mathcal{L}_{\text{BC}} = \mathbb{E}_{(o,a)\sim\mathcal{D}}\left[\|\pi_\theta(o)-a\|^2\right]$$

离散化动作用交叉熵：

$$\mathcal{L}_{\text{CE}} = -\mathbb{E}_{(o,a)\sim\mathcal{D}}\left[\log \pi_\theta(a \mid o)\right]$$

### 1.2 两大顽疾

- **分布坍缩 (Distribution Collapse)**：均方误差回归会平均多模态动作，导致输出趋向不同合理动作的均值（例如抓杯子既可从左也可从右，均值是"中间"——一个无效动作）；
- **协变量偏移 (Covariate Shift)**：训练时观测来自专家轨迹，推理时策略一旦偏离就会进入从未见过的状态，误差雪崩式累积。常用对策：DAgger（在策略自身产生的状态下询问专家标签）。

---

## 2 Diffusion Policy {#2}

### 2.1 思路

Diffusion Policy 把动作序列建模为去噪扩散过程，直接在动作空间上拟合多模态分布，天然规避分布坍缩。

### 2.2 训练

给定观测 $o$，对动作轨迹 $a_0$（即 $a_{t:t+H}$）加噪得到 $a_k$，训练网络 $\epsilon_\theta$ 预测注入的噪声：

$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{\epsilon, k, a_0}\left[\|\epsilon - \epsilon_\theta(a_k, k, o)\|^2\right]$$

推理时从噪声起逐步去噪采样得到一条动作轨迹。

### 2.3 优势

- **多模态性**：扩散分布可表示多峰动作分布，不再被迫取均值；
- **时序一致性**：对整条动作块联合建模，避免逐帧独立预测的不连贯；
- **稳定性**：训练目标简单（噪声预测），比 GMM/Mixture 更稳。

### 2.4 局限

- 推理需多步去噪，延迟较高；
- 训练数据量大；
- 离散动作不便直接建模（需要做离散扩散）。

---

## 3 Flow Matching Policy {#3}

### 3.1 思路

π0 / π0.5 把 Flow Matching 引入动作空间，用 ODE 直接回归"速度场"，采样步数比 Diffusion Policy 更少：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, a_0, a_1, a_t}\left[\|v_\theta(a_t, t, o) - (a_1 - a_0)\|^2\right]$$

其中 $a_0$ 是专家动作分布，$a_1$ 是噪声，$a_t = (1-t) a_0 + t a_1$。推理时从 $t=1$ 用 ODE 求解器积分到 $t=0$。

### 3.2 与 Diffusion Policy 的对比

| 维度 | Diffusion Policy | Flow Matching Policy |
|------|------------------|----------------------|
| 训练目标 | 预测噪声 | 预测速度场 |
| 训练稳定性 | 良好 | 通常更好 |
| 采样步数 | 通常 10–100 | 通常 5–30 |
| 灵活性 | 多种噪声调度 | 线性插值路径天然简洁 |
| 工程实现 | 成熟 | 新，需要 ODE solver |

### 3.3 代表工作

- π0 / π0.5（Physical Intelligence）；
- 学术界的多个 Flow Matching 操作策略实现。

---

## 4 3D Diffusion Policy (DP3) {#4}

### 4.1 动机

2D 视觉对"物体在空间中的几何"不敏感。DP3 把**点云**作为输入：

- 用预训练 3D 编码器把点云压为紧凑特征；
- 把 3D 特征与 2D 图像、文本指令拼在一起，Diffusion Policy 生成动作。

### 4.2 效果

- 几何感知显著优于纯 2D；
- 对新物体、新视角更鲁棒；
- 训练数据需求略高。

### 4.3 衍生工作

- **3D Diffuser Actor**：进一步把 3D 信息注入 actor head；
- 多份开源 DP3 实现，已成为几何感知操作策略的常用基线。

---

## 5 离线强化学习 {#5}

### 5.1 设定与动机

离线 RL（Batch RL）只使用**固定数据集** $\mathcal{D} = \{(o_i, a_i, r_i, o_i')\}$ 学习策略，不与环境交互。动机：

- 真机探索代价高且危险，但历史数据（含失败数据）可以复用；
- 跨实体数据（OXE）本质上是离线数据，天然适合离线 RL 设定。

核心困难是**分布偏移**：数据中没有的动作 $a \notin \text{supp}(\pi_\beta)$，其 Q 值 $\hat{Q}(o, a)$ 被外推得过高，策略贪心追高后会执行"数据里没见过"的坏动作。

### 5.2 CQL：保守 Q 学习

CQL（Conservative Q-Learning）直接压低数据分布外动作的 Q 值：

$$\mathcal{L}_{\text{CQL}} = \alpha\, \mathbb{E}_{o}\left[\log \sum_a \exp(\hat{Q}(o,a)) - \mathbb{E}_{a \sim \mathcal{D}} \hat{Q}(o,a)\right] + \mathcal{L}_{\text{Bellman}}$$

第一项拉高所有动作的 log-sum-exp、压低数据内动作——等效于对 Q 做"悲观下界"，从而抑制外推误差。

### 5.3 IQL：期望回归

IQL（Implicit Q-Learning）避开对数据外动作的查询，用**期望分位回归**学习值函数：

$$\mathcal{L}_{V} = \mathbb{E}_{(o,a)\sim\mathcal{D}}\left[\ell_2^\tau\big(V(o) - \hat{Q}(o, a)\big)\right]$$

其中 $\ell_2^\tau$ 是期望 $\tau > 0.5$ 的非对称分位损失——$V$ 被拟合为"数据内最好动作"的 Q 值上分位。策略再通过优势加权回归（AWR）从 $Q - V$ 提取：

$$\mathcal{L}_{\pi} = -\mathbb{E}\left[\exp\big(\beta (Q - V)\big) \log \pi_\theta(a \mid o)\right]$$

IQL 实现简单、无需环境交互，是机器人离线 RL 的默认基线（HIL-SERL 等工作的 Critic 即用 IQL 式更新）。

### 5.4 序列建模视角：Decision Transformer

把 RL 视为条件序列建模：以"回报-to-go"为条件，自回归生成 $[\hat{R}, o_1, a_1, o_2, a_2, \dots]$。省去 Bellman 回归，直接用 Transformer 拟合轨迹分布。在机器人上，这引出了"策略即序列模型"的路线，与 VLA 的自回归范式天然亲和。

### 5.5 离线 RL 在具身智能中的位置

| 设定 | 数据 | 方法 | 适用 |
|---|---|---|---|
| BC | 演示 $(o, a)$ | 监督回归 | 有专家、任务明确 |
| 离线 RL | 交互数据 $(o,a,r,o')$ | CQL / IQL / DT | 有奖励或可标注成败 |
| 在线 RL | 环境交互 | PPO / SAC | 仿真或安全真机 |
| 人干预 RL | 在线 + 人类纠正 | HIL-SERL | 真机、高价值任务 |

---

## 6 人干预 RL (HIL-SERL) {#6}

### 6.1 思路

模仿学习的演示数据有限，BC 训练到一定精度后难以继续提升。**HIL-SERL** 在 BC 初始化之上加入 RL：

- 策略在线执行；
- 人类在策略"即将失败"时接管，给出纠正动作；
- 纠正信号进入 Critic（IQL 式）与策略更新。

### 6.2 优势

- 用少量人类干预把成功率从 ~70% 提升到 90%+；
- 避免完全从零 RL 的样本效率与安全问题。

### 6.3 工程挑战

- 人类干预接口需要低延迟（共享主从臂或力反馈手柄）；
- 干预次数需要控制（避免人类疲劳）；
- 安全护栏必须到位（力/速度限幅、急停）。

---

## 7 BC + RL 混合范式与 VLA 后训练 {#7}

### 7.1 经典混合管线

```text
BC 预训练   →  得到合理初值 π_θ
   │
   ▼
RL 微调     →  在仿真或真机环境继续提升
   ├─ 无人类干预 → 标准 SAC/PPO
   ├─ 人类干预  → HIL-SERL / SERL
   └─ 离线 RL   → IQL/CQL，从已采集数据中学习
```

### 7.2 VLA 的 RL 后训练（2024–2025 前沿）

VLA 出现后，"预训练 VLA + RL 后训练"成为继 SFT 后训练之后的新主线：

- **iRe-VLA / VLA-RL 思路**：在 RL 更新中交替"冻结骨干、只调动作头"与"解冻骨干联合更新"，避免 RL 不稳定破坏预训练表征；
- **PA-RL**（Physical Intelligence 系）：以 π0 为基座，用"AUX 辅助任务 + 人类反馈奖励 + 有效信息瓶颈"做在线 RL 微调，在折叠衣物等长程任务上显著超越 SFT；
- **RIPT-VLA**：以可验证奖励（任务成功/失败）做交互式策略优化，奖励来自环境判据而非人类；
- **SimpleVLA-RL**：把 GRPO 式组相对优化搬到 VLA 动作生成上，用极少的真机交互样本提升长程成功率。

### 7.3 奖励来源的谱系

RL 后训练的关键是奖励从哪来：

```text
手工奖励 → 环境判据（成功/失败检测器）
        → VLM 打分（vision-language reward）
        → 人类偏好/纠正（HIL）
        → 世界模型 rollout 估计（model-based）
```

趋势：**环境判据 + VLM 奖励模型**组合覆盖面最广——成功检测用 VLM（"物体是否在碗里"），失败边界数据反哺错误检测器，与数据飞轮（见[数据集与数据采集](./data.md)）闭环。

---

## 8 参考文献 {#8}

[1] C. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS, 2023. arXiv:2303.04137.

[2] Y. Ze et al., *3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Sparse 3D Representation*, RSS, 2024.

[3] Physical Intelligence, *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. arXiv:2410.24164.

[4] J. Luo et al., *HIL-SERL: Human-in-the-Loop Sample-Efficient Reinforcement Learning for Robot Manipulation*, RSS, 2024.

[5] S. Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger), AISTATS, 2011.

[6] A. Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations*, CoRL, 2023.

[7] I. Kostrikov et al., *Offline Reinforcement Learning with Implicit Q-Learning* (IQL), ICLR, 2022. arXiv:2110.06169.

[8] A. Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning* (CQL), NeurIPS, 2020.

[9] L. Chen et al., *Decision Transformer: Reinforcement Learning via Sequence Modeling*, NeurIPS, 2021.

[10] C. Chi et al., *Universal Manipulation Interface* (ACT/ALOHA 系), RSS, 2024.

[11] Physical Intelligence, *PA-RL: Reinforcement Fine-Tuning of Vision-Language-Action Models*, 2025.

[12] RIPT-VLA Team, *RIPT-VLA: Interactive Post-Training with Verifiable Rewards for Vision-Language-Action Models*, 2025.

[13] SimpleVLA-RL Team, *SimpleVLA-RL: Scaling VLA Training with Reinforcement Learning*, 2025.

[14] S. Nair et al., *R3M: A Universal Visual Representation for Robot Manipulation*, CoRL, 2022.
