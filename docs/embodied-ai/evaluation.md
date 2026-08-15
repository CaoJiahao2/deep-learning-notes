# 具身智能评测基准

> 从 CALVIN 到 SimplerEnv，从 RLBench 到 HumanoidBench —— 如何客观衡量"机器人真的会干活"。

---

## 目录

1. [概述](#1)
2. [评测的挑战](#2)
3. [仿真操作基准](#3)
4. [真机评测](#4)
5. [人形与全身控制基准](#5)
6. [世界模型 / 预测类评测](#6)
7. [评测协议与方法](#7)
8. [实践建议](#8)
9. [参考文献](#9)

---

## 1 概述 {#1}

具身智能（Embodied AI）评测衡量智能体在真实或仿真环境中**执行物理任务**的能力。与 LLM 评测不同，它必须评估"身体"与环境交互的完整闭环：感知 → 决策 → 动作 → 反馈。

评测的层次：

```text
单元能力（抓取、单步控制）
→ 任务能力（多步操作、任务完成率）
→ 系统能力（长期自主、泛化、安全性）
```

评测是具身智能最稀缺的基础设施之一——**没有可信评测，就无法判断算法进步**。

---

## 2 评测的挑战 {#2}

| 挑战 | 说明 |
|------|------|
| 环境差异 | 不同仿真器、不同真机导致结果不可比 |
| 任务定义不一 | 同一"倒水"在不同基准中规格不同 |
| 随机性 | 初始状态、物体位姿、物理引擎随机性 |
| 真机成本 | 大规模真机评测昂贵且不可复现 |
| 评测 vs 训练泄露 | 训练数据包含评测任务，造成虚高 |
| 奖励 vs 成功率 | 仿真可用稠密奖励，真机只有稀疏任务信号 |

---

## 3 仿真操作基准 {#3}

### 3.1 CALVIN

CALVIN（Mees et al., 2022）是长程操作任务的代表基准：

- 连续 5 个子任务的长程操作（拿、放、开抽屉、按压）；
- ABC-D 设置：前三个环境训练，第四个环境评测（跨环境泛化）；
- 指标：长程成功率（依次完成 1-5 个连续任务的比例）；
- 常被 VLA / Diffusion Policy 用作评估基准。

### 3.2 RLBench

RLBench（James et al., 2020）基于 CoppeliaSim + PyRep，包含 100+ 任务：

- 覆盖堆叠、开门、插孔、放置等常见操作；
- 提供多视角视觉、深度、关节状态；
- 常用作"大规模多任务"学习的评测，如 PerAct、RVT。

### 3.3 ManiSkill

ManiSkill（Mu et al., 2021）基于 SAPIEN 物理引擎，强调**接触与物理交互**：

- 任务：插销、倒水、装配、移动等接触密集操作；
- 提供点云、视觉、关节观测；
- 支持人类演示数据，适合模仿学习与 RL 评测。

### 3.4 MetaWorld / RoboSuite

- **MetaWorld**：50 个桌面操作任务（推、抓、放、组装）；
- 任务多样、接口统一，常作为多任务训练基线；
- 固定 200 个演示的评测协议，方便对比。

### 3.5 FrankaKitchen / Adroit

- **FrankaKitchen**：真实厨房场景仿真，长程任务（开柜、转开关）；
- **Adroit**：灵巧手操作（开门、转笔、拾物），评测灵巧操控。

---

## 4 真机评测 {#4}

### 4.1 SimplerEnv

SimplerEnv（Li et al., 2024）把仿真评测标准化，聚焦**真实-仿真对齐**：

- 在仿真中复刻真实硬件（Go-1、UR5、WidowX 等）与任务；
- 用"仿真预测 vs 真机表现"的一致性衡量基准可信度；
- 指出：很多在仿真上刷分的策略，真机表现远差于仿真，揭示 sim2real gap。

### 4.2 开源真机基准

- **DROID（Google, 2024）**：大规模真机遥操作数据集 + 训练评测；
- **Open X-Embodiment（RT-X）**：跨机构聚合数据，跨本体泛化评测；
- **BridgeData / BridgeData V2**：低成本真机数据，任务多样；
- **FMB（机器人基础模型基准）**：统一多个真机环境的评测套件。

### 4.3 真机评测协议

真机评测需要固定协议以保证可比性：

- 固定初始布局与物体位姿；
- 统一成功率定义（严格/宽松判据）；
- 多次运行取统计（均值 ± 方差）；
- 报告"第一次尝试"与"允许重试"两种结果。

---

## 5 人形与全身控制基准 {#5}

### 5.1 HumanoidBench

HumanoidBench（Sferrazza et al., 2024）面向人形机器人的全身控制：

- 基于 MuJoCo 的人形模型（Unitree、SMPL 等）；
- 任务：行走、站立恢复、搬箱、开门、爬坡等；
- 评测 RL 算法在"高维全身控制 + 稀疏奖励"下的表现。

### 5.2 运动控制基准

- **DeepMind Control / DM Humanoid**：经典连续控制；
- **Isaac Gym / Isaac Lab 人形任务**：大规模并行 RL 评测；
- **HumanoidPlus**：增加真实人形本体与接触任务的扩展。

### 5.3 灵巧手基准

- **DexBench / DexGraspNet**：灵巧手抓取与操作；
- **Shadow Hand 任务**：Adroit 中的灵巧操作；
- 灵巧操控评测仍缺乏统一标准，是开放问题。

---

## 6 世界模型 / 预测类评测 {#6}

随世界模型（`world-model.md`）成为热点，出现专门评测：

- **动作条件预测**：给定过去帧 + 动作，预测未来帧（FVD、PSNR）；
- **交互一致性**：动作是否引起因果一致的场景响应；
- **规划价值**：用世界模型 rollout 做规划，评测任务成功率提升；
- **GigaWorld / WorldSimProbe 等 2025 工作**：专门诊断仿真器/世界模型的"忠实度"——世界模型预测是否与真实物理一致。

---

## 7 评测协议与方法 {#7}

### 7.1 成功率定义

统一成功率定义至关重要：

```text
严格：任务所有子步骤完全满足才算成功
宽松：达到主要目标即算成功
分步：报告每一步的完成率
```

### 7.2 统计与可复现性

- 多次随机种子运行，报告 mean ± std；
- 固定初始状态种子，确保对比公平；
- 评测集与训练集隔离，防止泄露；
- 公开评测脚本与 checkpoint，保证可复现。

### 7.3 多维度报告

除任务成功率外，还应报告：

- 平均步数 / 平均耗时（效率）；
- 抓取成功率、碰撞率（单元能力）；
- 跨任务/跨环境的泛化率；
- 仿真 vs 真机的差距（sim2real gap）。

---

## 8 实践建议 {#8}

- **仿真先行，真机复验**：先在标准化仿真基准（CALVIN、RLBench、ManiSkill）快速迭代，再用 SimplerEnv 类协议验证真机可行性；
- **警惕刷分**：仿真刷分不等于真机可用，优先关注 sim2real 一致的基准；
- **多基准组合**：操作（CALVIN/RLBench）+ 接触（ManiSkill）+ 长程（FrankaKitchen）+ 人形（HumanoidBench）；
- **建立自研评测**：沉淀公司/实验室自有的真机任务集，避免依赖公开基准；
- **评测与训练隔离**：确保训练数据不含评测任务，防止泄露虚高；
- **报告完整统计**：成功率、方差、步数、真机差距一起呈现。

---

## 9 参考文献 {#9}

[1] O. Mees, L. Hermann, E. Rosete-Beas, W. Burgard, *CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks*, RA-L, 2022.

[2] S. James, Z. Ma, D. R. Arrojo, A. J. Davison, *RLBench: The Robot Learning Benchmark & Learning Environment*, RA-L, 2020.

[3] T. Mu, Z. Ling, F. Xiang, et al., *ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations*, NeurIPS, 2021.

[4] X. B. Peng, E. Coumans, T. Zhang, et al., *MetaWorld: A Benchmark and Evaluation for Robotic Manipulation*, ICRA, 2022.

[5] X. Li, K. Hsu, J. Gu, et al., *Evaluating Real-World Robot Manipulation Policies in Simulation*, RSS, 2024.

[6] A. Khazatsky, K. Pertsch, S. Nair, et al., *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, ICRA, 2025.

[7] C. Sferrazza, D. M. Huang, X. Lin, et al., *HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation*, 2024. arXiv:2403.10506.

[8] A. Padalkar, A. Pooley, A. Jain, et al., *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, 2023.

[9] Google DeepMind, *DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset*, 2024.

[10] P. Mees, et al., *FRANKA Kitchen Dataset and Benchmark*, 2021.
