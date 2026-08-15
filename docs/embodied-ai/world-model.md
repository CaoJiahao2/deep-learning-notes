# 世界模型 (World Model)

> 学习环境动力学，在"脑内"推演未来以辅助规划或想象训练——从 Dreamer 到 Genie/Sora，从隐空间 RL 到与 VLA 融合。

---

## 目录

1. [概念与目标](#1)
2. [Dreamer 系列](#2)
3. [视频生成作为世界模型](#3)
4. [动作条件视频生成](#4)
5. [大规模世界模型：Genie / Sora / Cosmos / World Labs](#5)
6. [世界模型与决策](#6)
7. [世界模型与 VLA 融合](#7)
8. [评测与开放问题](#8)
9. [参考文献](#9)

---

## 1 概念与目标 {#1}

世界模型学习环境的动力学 $\hat{p}(s_{t+1} \mid s_t, a_t)$，使智能体能够"在脑中推演"——给定当前状态与候选动作，预测未来状态，从而进行规划或想象训练，无需每一步都真实交互。

自回归世界模型目标（最大似然下一状态）：

$$\mathcal{L}_{\text{world}} = -\mathbb{E}\left[\log \hat p_\theta(s_{t+1}\mid s_{\le t}, a_{\le t})\right]$$

世界模型的核心价值：

- **规划**：在隐空间 rollout 多个候选动作序列，选回报最大的执行；
- **想象训练**：在"想象"轨迹上做 RL，避免昂贵的真实交互；
- **数据效率**：把少量真实交互扩展为海量"想象"数据；
- **真实世界仿真**：为机器人/自动驾驶提供低成本、可并行的训练场。

世界模型的发展经历了三条并行主线：**隐空间 MBRL（Dreamer）**、**视频预测/生成（Genie、Sora）**、**面向机器人的可交互仿真（UniSim、Cosmos、Genesis）**。三者正在走向统一——目标是通用可交互世界模拟器。

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
- **DreamerV3**（Nature, 2023）：统一的多任务超参配置，在 Minecraft、Atari、Control Suite、机器人等多个领域同时达到 SOTA，无需为每任务调参。

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

视频生成是不错的"视觉先验"与"短程动态"模型，但作为可执行动作的世界模型仍差关键一环——显式的动作条件与可控性。2024-2025 年的大规模世界模型正围绕"可交互性"补上这一环。

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

## 5 大规模世界模型：Genie / Sora / Cosmos / World Labs {#5}

2024-2025 年，世界模型从学术实验走向"基础模型"级建设，主要沿四个方向展开。

### 5.1 Genie 2 / Genie 3（Google DeepMind）

- **Genie 2**（2024）：单张图像生成可玩的 3D 世界，支持视角控制、长时程一致性、NPC 与物体交互，展示"prompt 世界"的实时交互生成。
- **Genie 3**（2025）：面向智能体的可交互世界模型，可从第一人称视频学动作，为训练"在可交互世界中行动的智能体"提供基准环境。

Genie 系列的核心贡献是把"可玩"（playable）作为世界模型的第一性目标——不仅预测未来，还支持用户在生成的世界中采取动作并看到因果响应。

### 5.2 Sora 2 与 World Simulator 叙事

- **Sora（OpenAI, 2024）**：文/图生视频，以其 3D 一致性与物理合理震惊业界。
- **Sora 2（OpenAI, 2025）**：从"生成视频"转向"世界模拟"——支持编辑现实视频、可交互的实时世界模拟、以及"anything-to-anything"的创作能力，明确把世界模拟作为目标。

### 5.3 NVIDIA Cosmos 平台

- **Cosmos World Foundation Models**（NVIDIA, 2025）：面向物理 AI 与机器人开发的开源世界基础模型平台。
- 提供视频生成、世界预测、可交互仿真与合成数据生成能力；
- 支持把生成的多样化世界数据用于训练机器人（数据飞轮）；
- 与 Omniverse / Isaac 结合，成为"生成式物理 AI 训练场"。

### 5.4 World Labs（李飞飞团队）

- **World Labs**（2024 成立）聚焦"空间智能"（Spatial Intelligence），目标是让 AI 具备三维感知、生成与交互能力；
- 发布 **World Models** 与 3D 生成技术，强调"理解三维世界、预测交互后果"而非仅仅像素预测；
- 与自监督 3D 表示、NeRF/3DGS 重建（见 `architectures/3d-generation.md`）深度结合。

### 5.5 自监督世界模型：V-JEPA / i-JEPA

- **V-JEPA**（Meta, 2024）：视觉联合嵌入预测架构，在特征空间做未来预测（而非像素空间），更高效、更具物理一致性；
- 代表"预测式世界模型"路线：预测抽象特征而非逐像素，缓解长期预测累积误差。

---

## 6 世界模型与决策 {#6}

### 6.1 Model-Based RL

经典 MBRL（Dyna、SAC+MBPO）把世界模型作为规划或想象工具：

- **规划**：在隐空间 rollout 多个动作序列，按回报选最优；
- **想象训练**：在隐空间做策略更新。

### 6.2 World-Model as Simulator

把世界模型作为"数据生成器"，为 RL 提供海量仿真样本，对机器人/自动驾驶这类**真实交互昂贵**的场景尤其有价值。

### 6.3 Hierarchical World Model

高层世界模型做"任务级规划"（"打开柜门 → 取杯子 → 放桌上"），低层世界模型做"动作级推演"（关节轨迹）。与分层 VLA 自然契合。

### 6.4 Imagination-Augmented Agents

- 在决策时先用世界模型 rollout 若干候选策略，评估后再执行；
- 降低探索风险：先在"脑中"试错，再在真实环境执行；
- 与 LLM 规划结合：LLM 生成高层计划，世界模型验证可行性。

---

## 7 世界模型与 VLA 融合 {#7}

世界模型正成为 VLA（Vision-Language-Action）体系的关键组件，二者形成互补。

### 7.1 为什么 VLA 需要世界模型

- VLA 直接"观测→动作"映射，缺乏对未来后果的推演；
- 真实机器人数据稀缺昂贵，世界模型能合成训练数据（数据增强）；
- 世界模型可提供"想象"预演，提升策略的安全性与泛化。

### 7.2 融合范式

| 范式 | 说明 | 代表 |
|------|------|------|
| 世界模型作为数据引擎 | 用世界模型生成多任务合成轨迹，扩充 VLA 训练集 | Cosmos + GR00T、Genie 3 |
| 世界模型作为规划器 | 用世界模型 rollout 评估候选动作序列，选最优执行 | 分层 VLA + WM |
| 世界模型作为正则器 | 用未来预测任务辅助 VLA 表征学习，提升泛化 | 预测式预训练 |
| 可交互仿真 | 在生成的世界中训练/评测策略，避免真实风险 | UniSim、Genesis |

### 7.3 代表性方向

- **GR00T + Cosmos**：NVIDIA 用 Cosmos 生成多样化世界数据，训练人形机器人基础模型；
- **Genesis（2024）**：通用物理引擎 + 生成式数据平台，可生成四维动态世界用于机器人训练；
- **预测式 VLA 预训练**：先用未来帧预测任务做视觉-动作联合表征预训练，再微调下游策略；
- **World Model RL 的机器人应用**：在隐空间世界模型中用 RL 学到的策略，直接迁移到真实机器人。

### 7.4 数据飞轮

世界模型 + 机器人形成数据飞轮：

```text
真实采集少量轨迹 → 世界模型合成海量数据 → 训练 VLA
        ↑                                        │
        └──────── 真机验证反馈 ←──────────────────┘
```

合成数据能扩大数据规模与多样性，但真实验证仍是不可或缺的一环，需防范"仿真与现实差距"（见 `sim2real.md`）。

---

## 8 评测与开放问题 {#8}

### 8.1 评测维度

| 维度 | 说明 | 指标 |
|------|------|------|
| 预测精度 | 未来状态/帧与真实的一致程度 | FVD、PSNR、隐状态重构误差 |
| 交互一致性 | 动作是否产生因果一致的响应 | 可控性、因果干预准确率 |
| 长期一致性 | 长程 rollout 是否漂移 | 长时程 FVD、物理违反率 |
| 规划价值 | 用于决策时能否提升回报 | 规划回报、任务成功率 |
| 数据增益 | 合成数据对下游训练的提升 | 下游 VLA 成功率 |

### 8.2 开放问题

| 局限 | 说明 |
|------|------|
| 长期一致性 | 长程 rollout 容易出现累积误差，物理一致性下降 |
| 多模态动作 | 同一状态下多种合理动作，世界模型需支持 |
| 评估困难 | 缺乏统一的"世界模型质量"评估指标 |
| 数据需求 | 学习高质量世界模型需要大量多模态交互数据 |
| 可控性 | 视频生成模型的动作接口与可控性仍有限 |
| 因果 vs 相关 | 像素预测不等于因果理解，反事实推理仍薄弱 |
| 仿真到现实 | 合成世界与真实物理仍有差距，需 sim2real 弥合 |

### 8.3 趋势判断

- 世界模型将从"预测工具"走向"通用可交互模拟器"，成为具身智能的训练场与规划器；
- 隐空间 MBRL、视频生成、物理引擎三条路线正在融合（如 Genesis、Cosmos + Isaac）；
- 与 VLA、Sim-to-Real 的结合将决定其在机器人领域的落地价值；
- 评测基准（交互一致性、规划价值）是当下最稀缺的基础设施。

---

## 9 参考文献 {#9}

[1] D. Hafner et al., *Dream to Control: Learning Behaviors by Latent Imagination*, ICLR, 2020.

[2] D. Hafner et al., *Mastering Atari with Discrete World Models*, ICLR, 2022. arXiv:2010.02193.

[3] D. Hafner et al., *Mastering Diverse Domains through World Models*, Nature, 2023.

[4] J. Bruce et al., *Genie: Generative Interactive Environments*, ICML, 2024.

[5] J. Parker-Holder et al., *Genie 2: A Large-Scale Foundation World Model*, Google DeepMind, 2024.

[6] J. Valevski et al., *GameNGen: Diffusion Models are Real-Time Game Engines*, 2024.

[7] OpenAI, *Sora: Video Generation Models as World Simulators*, 2024.

[8] OpenAI, *Sora 2: World Simulators*, 2025.

[9] NVIDIA, *Cosmos: World Foundation Models for Physical AI*, 2025.

[10] M. Yang et al., *UniSim: Learning Interactive Real-World Simulators*, 2023. arXiv:2310.06114.

[11] A. Hu et al., *GAIA-1: A Generative World Model for Autonomous Driving*, 2024. arXiv:2309.17080.

[12] J. Wu et al., *iVideoGPT: Interactive Video Prediction with World Model*, 2024.

[13] A. Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video*, ICLR, 2024.

[14] B. Huang et al., *Genesis: A Generative and Universal Physics Engine*, 2024.
