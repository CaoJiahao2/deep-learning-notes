# 具身智能前沿专题

> 涵盖分层规划、双系统 VLA、视觉导航、Humanoid、灵巧操作与评测基准。

---

## 目录

1. [分层规划 (Hierarchical Planning)](#1)
2. [双系统 VLA](#2)
3. [视觉语言导航 (VLN)](#3)
4. [Humanoid 与全身控制](#4)
5. [灵巧操作](#5)
6. [评测基准](#6)
7. [挑战与未来方向](#7)
8. [参考文献](#8)

---

## 1 分层规划 (Hierarchical Planning) {#1}

### 1.1 动机

"做早餐"这种长程任务，需要：

- **常识**：知道做早餐要先开冰箱找食材；
- **规划**：拆成"开冰箱 → 取鸡蛋 → 煎蛋 → 摆盘"等子目标；
- **执行**：每个子目标由专门的 VLA 策略完成。

单一端到端模型难以同时承担这些职责，于是出现**分层**结构：

```text
   高层：LLM / VLM（Task Planner）
        │
        │ 子目标序列 s_1, s_2, ..., s_N
        ▼
   低层：专用 VLA 策略（Skill Executor）
        │
        │ 动作序列 a_t:t+H
        ▼
       机器人
```

### 1.2 SayCan（Google, 2022）

最早的代表性工作：用 LLM 给出候选技能，用 affordance value function 估计物理成功率，二者相乘选择下一步技能。单篇精读见 [SayCan](./papers/saycan.md)。

### 1.3 PaLM-E / RT-2 Chain

[PaLM-E](./papers/palm-e.md) 把图像与机器人状态作为多模态 token 注入 LLM；[RT-2](./papers/rt-2.md) 进一步把动作离散为 token，与网页图文任务共同微调。它们把“具身推理链”和“端到端动作输出”连接起来。

### 1.4 HiRT（Hierarchical Robot Transformer）

把"高层规划"与"低层控制"分别建模为 Transformer：

- 高层推理：每 ~1s 一次，给出未来子目标；
- 低层控制：每 10–50Hz 一次，输出具体动作。

兼顾长程推理与高频控制。

### 1.5 π0.5：子任务与异构数据共训练

[π0.5](./papers/pi-0-5.md) 同时预测高层子任务和低层动作块，把机器人数据、网页多模态数据和口头指令放进同一训练配方，使模型能进入未见家庭执行 10–15 分钟家务。

### 1.6 π0.7：可操控上下文与世界模型子目标

[π0.7](./papers/pi-0-7.md) 进一步把 prompt 扩展为：

```text
总任务 + 当前子任务 + 视觉子目标 + episode metadata + 控制模式
```

其中视觉子目标由轻量世界模型生成，metadata 描述速度、质量和是否出错。这个设计让模型能吸收失败与次优自主轨迹，并通过上下文选择期望行为。

### 1.7 Xiaomi-Robotics-1：从状态转移到指令对齐

[Xiaomi-Robotics-1](./papers/xiaomi-robotics-1.md) 先用超过 100k 小时 UMI 轨迹和自动 state-transition caption 学习“如何改变世界”，再用约 10k 小时跨实体数据把模型对齐到“人类命令机器人做事”的 imperative 指令形式。

---

## 2 双系统 VLA {#2}

### 2.1 思路

借鉴 Kahneman 的"系统 1 / 系统 2"：

- **系统 1（快）**：低层 VLA，直接从观测输出动作，反应快；
- **系统 2（慢）**：高层推理模型（LLM/VLM），做规划、反思、错误恢复。

```text
       ┌──────────────────┐
       │  系统 2（慢思考） │
       │  LLM/VLM 推理    │
       └────────┬─────────┘
                │ 任务分解 / 错误恢复
                ▼
       ┌──────────────────┐
       │  系统 1（快执行） │
       │  VLA 策略         │
       └────────┬─────────┘
                │
                ▼
              动作
```

### 2.2 代表工作

- **iManip / HiRT / RoboBrain**：学术上的分层与双系统设计；
- **[Helix](./papers/helix.md)**：System 2 为 7B VLM（7–9 Hz），System 1 为 80M 反应策略（200 Hz），控制人形完整上半身；
- **[GR00T N1](./papers/groot-n1.md)**：Eagle-2 VLM 作为 System 2，DiT / Flow Matching 动作模块作为 System 1；
- **[Gemini Robotics](./papers/gemini-robotics.md)**：云侧 Gemini backbone + 本地 action decoder，端到端约 250ms，动作块内有效 50Hz。

### 2.3 优势

- 长程任务分解更稳健；
- 错误恢复机制更可解释；
- 高层推理可借鉴 LLM 的世界知识。

---

## 3 视觉语言导航 (VLN) {#3}

### 3.1 任务定义

智能体在 3D 环境中，根据自然语言指令自主导航到目标位置。代表性基准：

- **R2R**（Room-to-Room）：室内导航，给定"走到卧室的白色桌子旁"；
- **REVERIE**：远程物体定位（不光走到位置，还要找具体物体）；
- **VLN-CE**：连续环境版本（更接近真实机器人）。

### 3.2 关键技术

- **场景图**：维护对环境的结构化记忆；
- **动作空间**：前进 / 左转 / 右转 / 停止；
- **跨模态对齐**：指令中的实体与场景图节点对齐。

### 3.3 代表模型

- **NaviLLM**：用 LLM 做导航推理；
- **NaVid**：视频 VLM 直接输出导航动作；
- **MapGPT**：把语言地图作为规划媒介。

### 3.4 评测指标

- 成功率 (Success Rate)；
- 路径长度加权 SPL；
- 导航误差 (NE)。

---

## 4 Humanoid 与全身控制 {#4}

### 4.1 行业焦点

2024–2025 年人形机器人迎来爆发：

- **Figure**（Helix）：双系统 VLA；
- **Tesla Optimus**：自研 VLA；
- **Unitree H1 / H2**：硬件 + 开源控制器；
- **Agility Digit**：仓储物流；
- **Boston Dynamics Atlas**：新版本基于 RL 训练。

### 4.2 全身控制 (Whole-Body Control)

人形机器人有 20+ 自由度，需要同时：

- **维持平衡**（零力矩点 ZMP）；
- **完成操作**（末端轨迹）；
- **协调运动**（步态 + 手臂）。

代表方法：

- **基于模型**：MPC + WBC；
- **基于学习**：RL + Sim-to-Real。

### 4.3 VLA for Humanoid

NVIDIA **GR00T** 是通用人形机器人的 VLA 基础模型，强调：

- 接受语言指令 + 多视角 RGB；
- 输出全身关节指令；
- 在 Isaac Lab 仿真预训练 + 少量真机数据微调。

### 4.4 挑战

- 硬件稳定性（摔倒保护、关节耐久）；
- 长程任务分解（"铺床"需要几十步精细操作）；
- 安全护栏（与人近距离协作）。

---

## 5 灵巧操作 {#5}

### 5.1 任务

灵巧手（Dexterous Hand）有 16–20 自由度，可完成：

- 双手协作（穿针、拧螺丝）；
- 工具使用（锤子、剪刀）；
- 物体在手中的重新定位（in-hand manipulation）。

### 5.2 代表工作

- **DexCap**：便携式灵巧手数据采集；
- **DiffusionDex**：Diffusion Policy 在灵巧手上的应用；
- **Humanoid-Grasp**：通用抓取数据集；
- **Open-Diffusion-VLA**：开源多指手 VLA。

### 5.3 关键挑战

- **数据极稀缺**：多指手遥操作远比夹爪困难；
- **控制频率高**：20+ DoF 的 RL 训练样本量爆炸；
- **跨形态迁移**：不同手形态共享策略困难。

---

## 6 评测基准 {#6}

### 6.1 模拟器基准

| 基准 | 任务 | 特点 |
|------|------|------|
| **RLBench** | 100+ 操作任务 | Franka Panda + CoppeliaSim |
| **Meta-World** | 50+ 操作任务 | Sawyer + MuJoCo |
| **ManiSkill2** | 大规模操作 | GPU 并行 + 多机器人 |
| **Behavior** | 长程家务 | iGibson + 高层任务 |
| **Habitat** | 室内导航 | 高效 3D 仿真 |

### 6.2 真机基准

- **Open X-Embodiment Benchmark**：跨实体任务评测；
- **DROID Benchmark**：真实家庭任务；
- **AgiBot-World**：大规模真实场景。

### 6.3 指标

- **任务成功率**：二元成功 / 失败；
- **完成时间**：完成任务的步数 / 时间；
- **人类偏好**：与人类执行轨迹对比。

---

## 7 挑战与未来方向 {#7}

| 挑战 | 现状与方向 |
|------|-----------|
| 数据规模 | UMI、人类视频、失败轨迹、部署 rollout 与自动标注流水线共同扩大有效数据 |
| 长程任务 | 分层规划 + 记忆 + 子目标图像 + 世界模型校验 |
| 通用泛化 | 跨实体 / 跨场景 / 跨任务 / 反事实指令的组合泛化评估 |
| 经验学习 | π*0.6 类 RL、coaching、advantage conditioning 与自动 reward |
| 安全 | 标准化安全评估、护栏设计、人在回路 |
| 实时性 | 模型蒸馏 + 量化 + 边缘部署 |
| 可解释性 | 推理链可视化、失败归因 |
| Sim-to-Real | 可微仿真 + Real-to-Sim + 域随机化 |
| 多模态融合 | 触觉、力觉、听觉的整合 |

> 相关文档：单篇精读见[具身智能论文地图](./papers/index.md)；评测基准详见 [evaluation.md](evaluation.md)，操作与灵巧操控详见 [manipulation.md](manipulation.md)，世界模型详见 [world-model.md](world-model.md)。

---

## 8 参考文献 {#8}

[1] M. Ahn et al., *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan)*, CoRL, 2022.

[2] Driess et al., *PaLM-E: An Embodied Multimodal Language Model*, ICML, 2023.

[3] S. Belkhale et al., *HiRT: Hierarchical Robot Transformer*, 2024.

[4] Y. Hong et al., *iManip: Skill-Based Manipulation with Dual-System VLA*, 2024.

[5] NVIDIA, *GR00T: Foundation Models for Humanoid Robots*, 2025.

[6] P. Anderson et al., *Vision-and-Language Navigation (R2R)*, CVPR, 2018.

[7] J. Krantz et al., *NaviLLM: Vision-and-Language Navigation with LLM Reasoning*, 2024.

[8] S. Y. Gadre et al., *NaVid: Video-based VLM for Navigation*, 2024.

[9] A. Mandlekar et al., *RoboCasa: Large-Scale Simulation for Everyday Robot Manipulation*, 2024.

[10] S. James et al., *RLBench: The Robot Learning Benchmark & Learning Environment*, RA-L, 2020.

[11] T. Yu et al., *Meta-World: A Benchmark and an Environment for Learning*, CoRL, 2020.

[12] J. Gu et al., *ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills*, ICLR, 2023.
[13] Physical Intelligence, *π0.5 / π*0.6 / π0.7*, 2025–2026.

[14] Xiaomi Robotics Team, *Xiaomi-Robotics-1*, 2026.

[15] Figure, *Helix: A Vision-Language-Action Model for Generalist Humanoid Control*, 2025.
