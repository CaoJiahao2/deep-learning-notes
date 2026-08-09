# 世界模型 (World Model)

> 学习环境动力学，在"脑内"推演未来以辅助规划或想象训练。

---

## 目录

1. [概念与目标](#1)
2. [Dreamer 系列](#2)
3. [视频生成作为世界模型](#3)
4. [动作条件视频生成](#4)
5. [世界模型与决策](#5)
6. [局限与开放问题](#6)
7. [参考文献](#7)

---

## 1 概念与目标 {#1}

世界模型学习环境的动力学 $\hat{p}(s_{t+1} \mid s_t, a_t)$，使智能体能够"在脑中推演"——给定当前状态与候选动作，预测未来状态，从而进行规划或想象训练，无需每一步都真实交互。

自回归世界模型目标（最大似然下一状态）：

$$\mathcal{L}_{\text{world}} = -\mathbb{E}\left[\log \hat p_\theta(s_{t+1}\mid s_{\le t}, a_{\le t})\right]$$

世界模型的核心价值：

- **规划**：在隐空间 rollout 多个候选动作序列，选回报最大的执行；
- **想象训练**：在"想象"轨迹上做 RL，避免昂贵的真实交互；
- **数据效率**：把少量真实交互扩展为海量"想象"数据。

---

## 2 Dreamer 系列 {#2}

### 2.1 RSSM

**Recurrent State-Space Model**（Hafner et al.）是 Dreamer 的核心：

- **隐状态** $h_t$：循环更新，承载长期记忆；
- **随机状态** $s_t$：捕捉随机性；
- **观测模型** $p(o_t \mid h_t, s_t)$：从隐状态重建观测；
- **转移模型** $q(s_t \mid h_t, a_{t-1}, s_{t-1})$：预测下一随机状态。

```text
o_t ──► encoder ──► 拼接 ──► RSSM ──► h_t ──► decoder ──► ô_t
                     ▲          │
                     │          ▼
                   a_{t-1}     s_t  ─► reward / value head
```

### 2.2 DreamerV1 → V2 → V3 演进

- **DreamerV1**（ICLR 2020）：在 Atari 与 DeepMind Control 上首次证明"完全在隐空间做 RL"可行；
- **DreamerV2**（ICLR 2022）：离散隐变量 + Transformer，大幅提升图像重建质量；
- **DreamerV3**（2023）：统一的多任务超参配置，在 Minecraft、Atari、Control Suite、机器人等多个领域同时达到 SOTA。

### 2.3 想象训练

在 Dreamer 框架下：

1. 用真实数据训练 RSSM + 策略 / 价值网络；
2. 在 RSSM 隐空间 rollout 长轨迹（"想象"）；
3. 用想象轨迹更新策略与价值。

这样一次真实交互可以产生大量"想象"样本，显著提升样本效率。

---

## 3 视频生成作为世界模型 {#3}

### 3.1 趋势

Sora、Wan 2.1、Genie 等大规模视频生成模型引发争论：**视频生成模型是否可作通用世界模拟器？**

### 3.2 正方观点

- 大规模视频预测隐式学到了物理规律（重力、碰撞、流体）；
- 能预测未来 $N$ 帧，并保持视觉一致性；
- 在可控条件下可作"游戏引擎"或"仿真器"。

### 3.3 反方观点

- **预测下一帧像素 ≠ 理解物理因果**——模型可能只是记住表面纹理与运动模式；
- 对长程因果、反事实推理、精确物理（碰撞、形变、摩擦）无能为力；
- 缺少显式的**动作条件**接口，难以做规划。

### 3.4 共识

视频生成是不错的"视觉先验"与"短程动态"模型，但作为可执行动作的世界模型仍差关键一环——显式的动作条件与可控性。

---

## 4 动作条件视频生成 {#4}

为了把视频模型用作世界模型，需要给它**动作条件**。代表工作：

### 4.1 GAIA-1 (Wayve, 2024)

自动驾驶场景下的视频生成模型，以文本、过去视频帧、自车动作作为条件，预测未来驾驶视频。

### 4.2 UniSim (Google, 2023)

通用动作条件视频生成器，目标是"仿真任意场景"。动作通过编码后注入扩散条件。

### 4.3 iVideoGPT (2024)

把 Transformer 与视频扩散结合，构建可交互的隐空间世界模型。

### 4.4 GameNGen (Google, 2024)

实时交互式游戏环境生成，用扩散模型在推理时生成游戏帧，被宣传为"第一个由神经网络实时驱动的游戏引擎"。

### 4.5 Genie（Google DeepMind, 2024）

从无标注视频中学到可交互的世界模型，用户可通过（隐式或显式）动作控制生成视频中的对象行为，展示了无监督学习得到可玩环境控制器的可能性。

---

## 5 世界模型与决策 {#5}

### 5.1 Model-Based RL

经典 MBRL（Dyna、SAC+MBPO）把世界模型作为规划或想象工具：

- **规划**：在隐空间 rollout 多个动作序列，按回报选最优；
- **想象训练**：在隐空间做策略更新。

### 5.2 World-Model as Simulator

把世界模型作为"数据生成器"，为 RL 提供海量仿真样本，对机器人/自动驾驶这类**真实交互昂贵**的场景尤其有价值。

### 5.3 Hierarchical World Model

高层世界模型做"任务级规划"（"打开柜门 → 取杯子 → 放桌上"），低层世界模型做"动作级推演"（关节轨迹）。与分层 VLA 自然契合。

---

## 6 局限与开放问题 {#6}

| 局限 | 说明 |
|------|------|
| 长期一致性 | 长程 rollout 容易出现累积误差，物理一致性下降 |
| 多模态动作 | 同一状态下多种合理动作，世界模型需支持 |
| 评估困难 | 缺乏统一的"世界模型质量"评估指标 |
| 数据需求 | 学习高质量世界模型需要大量多模态交互数据 |
| 可控性 | 视频生成模型的动作接口与可控性仍有限 |

---

## 7 参考文献 {#7}

[1] D. Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination*, ICLR, 2020.

[2] D. Hafner et al., *Mastering Atari with Discrete World Models*, ICLR, 2022. arXiv:2010.02193.

[3] D. Hafner et al., *Mastering Diverse Domains through World Models*, Nature, 2023.

[4] J. Bruce et al., *Genie: Generative Interactive Environments*, ICML, 2024.

[5] M. Yang et al., *UniSim: Learning Interactive Real-World Simulators*, 2023. arXiv:2310.06114.

[6] A. Hu et al., *GAIA-1: A Generative World Model for Autonomous Driving*, 2024. arXiv:2309.17080.

[7] OpenAI, *Sora: Video Generation Models as World Simulators*, 2024.

[8] J. Valevski et al., *GameNGen: Diffusion Models are Real-Time Game Engines*, 2024.

[9] J. Wu et al., *iVideoGPT: Interactive Video Prediction with World Model*, 2024.