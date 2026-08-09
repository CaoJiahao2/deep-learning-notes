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

最早的代表性工作：用 LLM 把指令分解为原子技能序列，每步选当前最可能成功的技能执行。

### 1.3 PaLM-E / RT-2 Chain

把 LLM 直接与低层动作头结合，可端到端推理；适合"中等长度"任务。

### 1.4 HiRT（Hierarchical Robot Transformer）

把"高层规划"与"低层控制"分别建模为 Transformer：

- 高层推理：每 ~1s 一次，给出未来子目标；
- 低层控制：每 10–50Hz 一次，输出具体动作。

兼顾长程推理与高频控制。

### 1.5 π0.5 的跨实体通用

π0.5 进一步引入"跨实体共享表征"，让高层规划器能在不同机器人形态之间迁移。

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

- **iManip**：双层 VLA；
- **HiRT**：上文已提及；
- **RoboBrain**：把"任务级大脑"作为独立模块，跨任务共享。

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
| 数据规模 | 数据仍远小于语言；自监督视频预训练是突破点 |
| 长程任务 | 分层规划 + 记忆 + 反思的组合 |
| 通用泛化 | 跨实体 / 跨场景 / 跨任务的统一基础模型 |
| 安全 | 标准化安全评估、护栏设计、人在回路 |
| 实时性 | 模型蒸馏 + 量化 + 边缘部署 |
| 可解释性 | 推理链可视化、失败归因 |
| Sim-to-Real | 可微仿真 + Real-to-Sim + 域随机化 |
| 多模态融合 | 触觉、力觉、听觉的整合 |

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