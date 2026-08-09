# 策略学习：从行为克隆到 Diffusion Policy 与 Flow Matching Policy

> 模仿学习把"专家经验"蒸馏进策略；扩散与 Flow Matching 在动作空间上自然地建模多模态分布。

---

## 目录

1. [行为克隆 (BC)](#1)
2. [Diffusion Policy](#2)
3. [Flow Matching Policy](#3)
4. [3D Diffusion Policy (DP3)](#4)
5. [人干预 RL (HIL-SERL)](#5)
6. [BC + RL 混合范式](#6)
7. [参考文献](#7)

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

π0 / π0.5 把 Flow Matching 引入动作空间，用 ODE 直接回归"速度场"，采样步数比 Diffusion Policy 更少。

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

- π0 / π0.5（Physical Intelligence）
- FlowMatching Policy in Robot Manipulation（多个学术实现）

---

## 4 3D Diffusion Policy (DP3) {#4}

### 4.1 动机

2D 视觉对"物体在空间中的几何"不敏感。DP3 把**点云**作为输入：

- 用预训练 3D 编码器（如 DP3 encoder）把点云压为紧凑特征；
- 把 3D 特征与 2D 图像、文本指令拼在一起，Diffusion Policy 生成动作。

### 4.2 效果

- 几何感知显著优于纯 2D；
- 对新物体、新视角更鲁棒；
- 训练数据需求略高。

### 4.3 衍生工作

- **3D Diffuser Actor**：进一步把 3D 信息注入 actor head；
- **DP3 的开源实现**：已有多份开源代码。

---

## 5 人干预 RL (HIL-SERL) {#5}

### 5.1 思路

模仿学习的演示数据有限，BC 训练到一定精度后难以继续提升。**HIL-SERL** 在 BC 初始化之上加入 RL：

- 策略在线执行；
- 人类在策略"即将失败"时接管，给出纠正动作；
- 把纠正动作作为奖励信号训练 Critic。

### 5.2 优势

- 用少量人类干预把成功率从 ~70% 提升到 90%+；
- 避免完全从零 RL 的样本效率问题。

### 5.3 工程挑战

- 人类干预接口需要低延迟；
- 干预次数需要控制（避免人类疲劳）；
- 安全护栏必须到位（机器人执行错误动作可能导致损坏）。

---

## 6 BC + RL 混合范式 {#6}

实践中常见 "BC 预训练 + RL 微调" 的混合范式：

```text
BC 预训练   →  得到合理初值 π_θ
   │
   ▼
RL 微调     →  在仿真或真机环境探索
   │
   ├─ 无人类干预 → 标准 SAC/PPO
   ├─ 人类干预  → HIL-SERL / SERL
   └─ 离线 RL   → IQL/CQL，从已采集数据中学习
```

数据需求与样本效率对比：

- 模仿学习需要**大量高质量演示**（数十到数千条轨迹），但无需奖励工程，样本利用直接且安全；
- 强化学习只需定义奖励即可自动探索，但真机探索代价高、样本效率低，常需在仿真中预训练再迁移。

---

## 7 参考文献 {#7}

[1] C. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS, 2023. arXiv:2303.04137.

[2] Y. Ze et al., *3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Sparse 3D Representation*, CoRL, 2024.

[3] Physical Intelligence, *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. arXiv:2410.24164.

[4] J. Luo et al., *HIL-SERL: Human-in-the-Loop Sample-Efficient Reinforcement Learning for Robot Manipulation*, 2024.

[5] S. Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*, AISTATS, 2011.

[6] A. Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations*, CoRL, 2023.

[7] A. Kumar et al., *Implicit Q-Learning (IQL)*, 2021. arXiv:2110.06169.

[8] A. Nair et al., *Behavior Transformer (BeT): Behavior Cloning with Action Chunking and Transformer*, 2023.