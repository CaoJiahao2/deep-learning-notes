# 数据集与数据采集

> 大规模、高质量、跨实体的机器人数据，是 VLA / 世界模型训练的命脉。

---

## 目录

1. [数据需求总览](#1)
2. [真机遥操作 (Teleoperation)](#2)
3. [开源数据集](#3)
4. [仿真数据生成](#4)
5. [网络视频数据](#5)
6. [数据工程实践](#6)
7. [参考文献](#7)

---

## 1 数据需求总览 {#1}

| 数据来源 | 真实度 | 规模 | 多样性 | 标注成本 |
|---|---|---|---|---|
| 真机遥操 | 高 | 小-中 | 受限于平台 | 高（人工） |
| 仿真生成 | 中（有 gap） | 大 | 高（可随机化） | 低（自动） |
| 网络视频 | 低（无动作） | 极大 | 极高 | 低（弱监督） |

不同来源的数据扮演不同角色：

- **真机数据**：训练 VLA 主干，最真实；
- **仿真数据**：扩展规模与多样性，需配合 Sim-to-Real；
- **网络视频**：学习视觉先验、物体可供性、物理常识，作为 VLM 预训练来源。

---

## 2 真机遥操作 (Teleoperation) {#2}

人类通过 VR 手柄、空间鼠标或主从臂操控机器人完成任务，记录关节状态与图像，得到高质量演示。代表性便携采集装置：

### 2.1 UMI（Universal Manipulation Interface, 2024）

手持式夹爪 + GoPro + 智能手机，把人手操作转化为机器人可执行的末端轨迹，无需机器人本体即可采集。

优势：低成本、便携、可大规模部署到非实验室环境。

### 2.2 DexCap

便携式灵巧手数据采集手套，捕获人手高自由度动作。

### 2.3 Gello

低成本 3D 打印主控臂，与从臂同构，直觉操作。

### 2.4 ALOHA / ALOHA 2

双手机器人遥操作平台，强调双手协作任务（如拧螺丝、穿针）。

### 2.5 跨实体数据采集

不同实验室、不同机器人形态的演示数据往往格式不一。要联合训练，需要：

- **统一状态空间**：把关节角度统一为末端 6DoF + 夹爪；
- **统一时间步**：重采样到固定频率；
- **统一标注格式**：RLDS、Open X-Embodiment 标准。

---

## 3 开源数据集 {#3}

### 3.1 Open X-Embodiment (RT-X, 2023)

由 Google DeepMind 牵头、多家机构联合发布的开源机器人数据集，覆盖 **22 种机器人形态**、百万级 episodes，统一为 RLDS 格式，是训练跨实体通用策略的关键数据基础。OpenVLA、Octo 等模型都在此数据上训练。

### 3.2 DROID

Stanford 等机构开源的分布式机器人数据，强调"日常家庭场景"的丰富度。

### 3.3 RoboNet

大规模多机器人形态数据，包含 1000+ 物体、12 个机器人。

### 3.4 BridgeData / BridgeData V2

UC Berkeley 团队，强调"真实厨房环境 + 真实任务"。

### 3.5 AgiBot / AgiBot-World

国内机器人公司发布的大规模具身数据集，覆盖工业 + 家庭场景。

### 3.6 Physical Intelligence 数据

π0 / π0.5 训练数据为闭源，但其论文披露了数据规模与组成（多实体、多任务）。

---

## 4 仿真数据生成 {#4}

### 4.1 MimicGen

从少量人类演示出发，自动生成大量"语义等效"的仿真轨迹：

- 把演示轨迹中的物体用仿真中的其他物体替换；
- 重新计算机械臂轨迹以完成新物体的任务。

可在没有新数据采集的情况下，把演示数据扩展 10x–100x。

### 4.2 仿真器中的大规模生成

借助 Isaac Sim、Isaac Lab、Genesis 等 GPU 并行仿真器，可在短时间内生成百万级轨迹。优势：

- 成本低（GPU 时间）；
- 多样性高（域随机化）；
- 自动标注（关节、位姿、力矩）。

局限：存在 Sim-to-Real gap，需配合域随机化、Real-to-Sim 标定等技术。

---

## 5 网络视频数据 {#5}

### 5.1 价值

网络视频虽无动作标签，但蕴含丰富的：

- 物体可供性（"人可以怎么用这个物体"）；
- 物理常识（"重物落地会下沉"）；
- 长程人类行为（"做早餐的步骤"）。

可作为 VLM 视觉-语言 backbone 的预训练来源。

### 5.2 代表工作

- **GR-1 / GR-2**（ByteDance, 2024）：在大规模视频上预训练视频生成 + 动作预测模型；
- **HUMANISE**：用视频训练"人在做什么"的常识模型；
- **SuSIE / Something-Something**：人类操作视频数据集。

### 5.3 局限

- 没有动作标签，只能做"预训练"或"弱监督微调"；
- 视角、相机参数、机器人本体不可见；
- 噪声大，需要严格清洗。

---

## 6 数据工程实践 {#6}

### 6.1 数据清洗

- 去除失败轨迹（夹爪空抓、目标未到达）；
- 过滤异常值（关节限位、图像模糊）；
- 标注质量检查（演示者熟练度、任务一致性）。

### 6.2 数据增强

- **视觉增强**：随机裁剪、颜色抖动；
- **空间增强**：随机平移、旋转、镜像（注意任务对称性）；
- **时序增强**：随机时间扰动（注意动作块长度）。

### 6.3 数据平衡

不同任务、不同物体的轨迹数量差异巨大，需要：

- 按任务类型做重采样；
- 按物体类别做加权；
- 训练时做任务/物体平衡采样。

### 6.4 数据格式

主流格式：

- **RLDS**（Reinforcement Learning Datasets）：Google 推出，适合基于 TF/Torch 的 RL 训练；
- **HDF5**：通用结构化格式，灵活性高；
- **Parquet + WebDataset**：大规模数据常用；
- **LeRobot Dataset**：Hugging Face 的机器人数据标准。

---

> 相关文档：数据如何驱动操作学习见 [`embodied-ai/manipulation.md`](manipulation.md)，数据基准见 [`embodied-ai/evaluation.md`](evaluation.md)，真机数据与仿真见 [`embodied-ai/sim2real.md`](sim2real.md)。

## 7 参考文献 {#7}

[1] Open X-Embodiment Collaboration, *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, ICRA, 2024. arXiv:2310.08864.

[2] A. Mandlekar et al., *MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations*, CoRL, 2023.

[3] Data et al., *On the Design of Universal Manipulation Interface*, RSS, 2024.

[4] F. Torabi et al., *RoboNet: A Large-Scale Multi-Robot Manipulation Dataset*, 2019.

[5] H. Walke et al., *BridgeData: Scaling Robot Learning with Multi-Modal Demonstration Data*, 2023.

[7] A. O'Neill et al., *Open-X-Embodiment: The Next Generation of Robot Learning Datasets*, RSS, 2024.

[8] C. Luo et al., *Gr-1: Video Generation Model as World Model for Robot Manipulation*, 2024.

[9] AgiBot Team, *AgiBot-World: A Large-Scale Robotic Manipulation Dataset*, 2025.