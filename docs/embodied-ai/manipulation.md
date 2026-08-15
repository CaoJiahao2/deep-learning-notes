# 操作学习：灵巧操控与移动操作

> 从桌面抓放到灵巧手，从移动操作到世界动作模型 —— 让机器人在真实世界"手到擒来"。

---

## 目录

1. [概述](#1)
2. [操作的层次](#2)
3. [桌面操作（Tabletop Manipulation）](#3)
4. [接触丰富的操作（Contact-Rich）](#4)
5. [灵巧操控（Dexterous Manipulation）](#5)
6. [移动操作（Mobile Manipulation）](#6)
7. [世界动作模型（WAM）](#7)
8. [数据与训练](#8)
9. [评测与开放问题](#9)
10. [参考文献](#10)

---

## 1 概述 {#1}

操作（Manipulation）是具身智能的核心能力：让机器人通过末端执行器改变环境状态。从最简单的"抓起杯子"到复杂的"装配零件""倒水""穿针"，操作任务横跨机械、物理、感知与决策。

操作学习的关键特性：

- **接触物理**：力、摩擦、形变、多体接触难以精确建模；
- **多模态输入**：视觉、触觉、力觉、本体感受需融合；
- **长程任务**：常需多步分解与子目标；
- **泛化**：新物体、新位姿、新场景下的迁移能力。

技术路线从"脚本/专家系统"走向"数据驱动的学习"（模仿学习 + RL + 世界模型），并朝**基础模型化**演进。

---

## 2 操作的层次 {#2}

| 层次 | 能力 | 示例 |
|------|------|------|
| 单元级 | 单一动作 | 抓取、推、按压 |
| 技能级 | 组合动作序列 | 拿起并放置、倒水 |
| 任务级 | 长程多步目标 | 收拾餐桌、装配 |
| 移动+操作 | 具身移动与操作结合 | 走到桌边拿起杯子 |

---

## 3 桌面操作 Tabletop Manipulation {#3}

桌面操作是机器人学习最成熟的子领域：机械臂 + 相机从上方/第三视角操作桌面物体。

### 3.1 抓取（Grasping）

- **基于分析的抓取**：几何/力闭合计算（GraspIt）；
- **数据驱动抓取**：从深度图/点云预测抓取位姿（GG-CNN、GraspNet）；
- **语言/语义引导抓取**：按自然语言指令选择目标物体（Grasp-Anything、Semantic Grasp）。

### 3.2 策略学习

- **行为克隆**：从演示学习（见 `policy-learning.md`）；
- **Diffusion Policy / Flow Matching Policy**：在动作空间建模多模态分布；
- **3D 扩散策略（DP3）**：直接以点云为输入，鲁棒于视觉变化；
- **RL 微调**：从 BC 初始化再 RL 优化（如 SERL、HIL-SERL）。

### 3.3 典型任务

堆叠、放置、插孔、开门、抽屉、倒水、舀取、旋转等，常用 CALVIN / RLBench / MetaWorld 评测（见 `evaluation.md`）。

---

## 4 接触丰富的操作 Contact-Rich {#4}

接触丰富操作（Contact-Rich Manipulation）依赖与环境的持续物理接触，建模难度高：

- **插孔/装配**：需要毫米级精度与力控制；
- **倒水/搅拌**：流体与软体的物理；
- **打磨/擦拭**：连续力交互；
- **穿线/打结**：柔性物体操作。

关键难点：

- **接触建模**：摩擦、形变、多体接触难以解析建模；
- **观测不足**：视觉无法感知接触力，需**触觉传感**；
- **不稳定动态**：操作过程存在多个接触切换（contact switching）。

应对技术：

- **触觉学习**：用触觉传感器数据训练（Tactile + vision）；
- **模仿学习**：从人/专家演示学习接触策略；
- **RL + 仿真**：在物理仿真中大规模训练后 sim2real 迁移；
- **阻抗/力控**：结合底层力控制与高层学习策略。

---

## 5 灵巧操控 Dexterous Manipulation {#5}

灵巧操控指用**多指灵巧手**（而非简单夹爪）完成精细操作，是操作学习的圣杯问题。

### 5.1 挑战

- 高自由度（每指 3-4 个关节，整手 20+ DOF）；
- 接触丰富、连续切换；
- 触觉信息至关重要但传感器昂贵；
- 数据获取困难（人手操作 vs 灵巧手遥操作）。

### 5.2 灵巧手平台

- **Shadow Hand**：经典 24-DOF 灵巧手，常用于研究；
- **Allegro Hand**：低成本 16-DOF，学术常用；
- **Inspire / DexHand**：国产灵巧手；
- **LEAP / Digit（触觉）**：触觉传感手指。

### 5.3 学习方法

- **灵巧操作模仿**：用人类手部姿态估计（如 AGRoL、DexMV）迁移到灵巧手；
- **触觉增强**：融合视觉 + 触觉的灵巧操作策略；
- **RL + 仿真**：在仿真中用大量随机化训练灵巧操作（如 DexterousHands benchmark）；
- **遥操作数据**：用灵巧手遥操作系统采集真实数据（见 `data.md`）。

### 5.4 关键工作

- **OpenAI 灵巧手 RL**（2018）：仿真中学习灵巧旋转魔方；
- **DexMV**：视觉 + 仿真 + 人类演示；
- **AGRoL / DexRep**：从人手视频学习灵巧操作；
- **H2R-Bench**（2025）：人类到机器人的操作视频生成评测。

---

## 6 移动操作 Mobile Manipulation {#6}

移动操作（Mobile Manipulation）让机器人**先移动到目标位置，再执行操作**，把导航（locomotion）与操作（manipulation）统一到单一任务中。

### 6.1 为什么重要

- 现实任务（家务、仓储、服务）几乎都需要移动 + 操作；
- 单独桌面上操作无法覆盖真实场景；
- 是"家庭助手""服务机器人"的落地核心。

### 6.2 技术挑战

- **长时间跨度**：导航 + 操作 + 多步推理；
- **视角变化**：从导航到操作时视角剧烈变化；
- **物体搜索**：需要在环境中寻找目标物体；
- **长程规划**：高层任务规划与底层运动控制协同。

### 6.3 代表工作

- **EmbodiedGPT / PaLM-E**：长程具身规划；
- **MOO / Navigation + Manipulation 联合**；
- **MobileWAM / Chain-of-Foresight**（2025）：把世界动作模型扩展到移动操作；
- **家庭服务基准**：BEHAVIOR、ALFRED、Habitat + 操作。

---

## 7 世界动作模型 WAM {#7}

世界动作模型（World-Action Model, WAM）是 2025 年新兴的融合范式：把**世界模型**（预测未来，见 `world-model.md`）与**动作模型**（输出动作，见 `vla.md`）统一到一个模型。

### 7.1 核心思想

传统 VLA 直接"观测 → 动作"，缺乏对未来后果的推演。WAM 让模型同时：

- 预测未来状态（世界模型能力）；
- 输出当前动作（动作模型能力）；
- 用"想象未来"指导"当前决策"。

$$\text{WAM} = \text{World Model} + \text{Action Model} = \text{预测未来} + \text{决定动作}$$

### 7.2 代表工作

| 系统 | 说明 |
|------|------|
| π0.5 | Physical Intelligence 的世界动作模型 |
| WSA-1 | 3D 为中心的世界-空间-动作模型 |
| MobileWAM | 世界动作模型 + 移动操作，Chain-of-Foresight |
| ω-0 | 潜在预测世界动作模型，人形移动操作 |
| AtlasVLA | 世界-自我状态建模的 VLA |
| Surgical WAM | 手术机器人世界动作模型 |

### 7.3 价值

- **更高效**：用世界模型想象训练/规划，减少真机数据需求；
- **更稳健**：对未来后果的预测使策略更能规避错误；
- **更通用**：统一"理解环境"与"作用于环境"的能力。

WAM 是"具身基础模型"的重要演进方向，与 `world-model.md` 第 7 节的 VLA 融合互补。

---

## 8 数据与训练 {#8}

### 8.1 数据来源

- **遥操作**：人类/系统遥操作采集（见 `data.md`）；
- **仿真数据**：大规模并行仿真生成（Isaac Lab、Genesis）；
- **人类视频**：网络人类操作视频（学习技能先验）；
- **合成数据**：世界模型/生成式仿真合成轨迹。

### 8.2 训练范式

- **模仿学习**：BC → Diffusion Policy / Flow Matching（见 `policy-learning.md`）；
- **BC + RL 混合**：先 BC 后 RL（HIL-SERL）；
- **预训练 + 微调**：基础模型预训练，任务微调；
- **RLVR 扩展**：用可验证信号做机器人 RL（Reward Foundation Models）。

### 8.3 数据飞轮

真实采集 → 训练策略 → 仿真/世界模型扩充数据 → 再训练，形成闭环（见 `world-model.md` 数据飞轮）。

---

## 9 评测与开放问题 {#9}

### 9.1 评测

见 `evaluation.md`：CALVIN（长程）、RLBench（多任务）、ManiSkill（接触）、HumanoidBench（人形）、SimplerEnv（真机对齐）。

### 9.2 开放问题

| 问题 | 说明 |
|------|------|
| 泛化 | 新物体、新场景、新任务下的迁移 |
| 长程 | 长时间任务的累积误差与状态漂移 |
| 接触 | 接触丰富任务的物理一致性与鲁棒性 |
| 数据 | 高质量真实数据稀缺昂贵 |
| 评测 | 缺乏统一、可信、可复现的基准 |
| 安全 | 高自由度机器人在真实环境的安全性 |
| 硬件 | 灵巧手、触觉传感成本与可靠性 |

---

## 10 参考文献 {#10}

[1] OpenAI, I. Akkaya, M. Andrychowicz, et al., *Solving Rubik's Cube with a Robot Hand*, 2019. arXiv:1910.07113.

[2] T. Mu, Z. Ling, F. Xiang, et al., *ManiSkill: Generalizable Manipulation Skill Benchmark with Large-Scale Demonstrations*, NeurIPS, 2021.

[3] X. Li, K. Hsu, J. Gu, et al., *Evaluating Real-World Robot Manipulation Policies in Simulation*, RSS, 2024.

[4] Physical Intelligence, *π0.5: Scaling Robot Learning with World-Action Models*, 2025.

[5] W. Huang, C. Wang, et al., *WSA-1: A 3D-Centric World-Spatial-Action Model for Generalizable Robot Control*, 2025.

[6] K. Chen, et al., *MobileWAM: Bridging World Action Models to Mobile Manipulation with Chain-of-Foresight*, 2025.

[7] T. Z. Zhao, V. Kumar, S. Levine, C. Finn, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS, 2023.

[8] C. Chi, S. Feng, Y. Du, et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS, 2023.

[9] A. Khazatsky, K. Pertsch, S. Nair, et al., *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, ICRA, 2025.

[10] R. Qin, et al., *H2R-Bench: Benchmarking Human-to-Robot Manipulation Video Generation*, 2025.
