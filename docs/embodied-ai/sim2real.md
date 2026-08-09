# Sim-to-Real 与仿真器

> 弥合仿真与现实的差距，让大规模 GPU 仿真训练出的策略能在真机上工作。

---

## 目录

1. [为什么需要 Sim-to-Real](#1)
2. [仿真器生态](#2)
3. [Sim-to-Real Gap 的成因](#3)
4. [域随机化 (Domain Randomization)](#4)
5. [域适应 (Domain Adaptation)](#5)
6. [System Identification 与 Real-to-Sim](#6)
7. [Real-in-the-Loop](#7)
8. [实践配方](#8)
9. [参考文献](#9)

---

## 1 为什么需要 Sim-to-Real {#1}

仿真器提供近乎无限的廉价数据与安全探索环境，但仿真与现实之间存在分布差异，即 **Sim-to-Real gap**。直接迁移仿真训练的策略到真机往往失败。

数据需求金字塔：

```text
        ▲
        │   真机演示
        │   （真实但昂贵）
        │
        │
        │   仿真 + 域随机化
        │   （便宜但有 gap）
        │
        │   合成视频
        │   （大量但弱监督）
        └──────────────────────►
```

需要 Sim-to-Real 技术把这三者有机结合。

---

## 2 仿真器生态 {#2}

| 仿真器 | 特点 | 适用 |
|--------|------|------|
| **MuJoCo** | 高精度接触动力学，经典 RL 基准底座 | RL 算法研究、连续控制 |
| **Isaac Sim / Isaac Lab** (NVIDIA) | 基于 RTX 的高保真物理与渲染，GPU 并行仿真 | 大规模 RL、机器人学习 |
| **Genesis** | 新一代通用生成式仿真，支持大规模并行 + 可微仿真 | 研究、可微仿真 |
| **SAPIEN** | 高保真刚体仿真，强项是铰接物体操作 | 操作任务 |
| **PyBullet** | 轻量 Python 接口，教学友好 | 入门、原型 |
| **MuJoCo XLA** | MuJoCo 的可微编译版本 | 可微 RL |
| **Habitat / AI2-THOR** | 室内场景 + 视觉导航 | 具身导航 |
| **CARLA** | 自动驾驶仿真 | 驾驶 |

---

## 3 Sim-to-Real Gap 的成因 {#3}

| 成因 | 说明 |
|------|------|
| 物理参数不准 | 摩擦、刚度、惯量与真实差异 |
| 接触动力学简化 | 仿真器对软接触、塑性形变建模粗糙 |
| 传感器噪声不同 | 仿真 vs 真实的相机噪声、IMU 漂移不同 |
| 渲染外观失真 | 材质、阴影、反射的视觉差异 |
| 控制延迟差异 | 仿真通常无延迟，真机有几十 ms 的执行延迟 |
| 执行器误差 | 真实关节有死区、间隙、摩擦非线性 |

---

## 4 域随机化 (Domain Randomization) {#4}

**核心思想**：在仿真中大量随机化物理与渲染参数，使策略对参数分布鲁棒，从而把现实当作"另一种随机化实例"。

### 4.1 参数示例

```text
# Domain Randomization 参数范围示例
friction_coef     ~ Uniform(0.3, 1.5)
restitution       ~ Uniform(0.0, 0.6)
object_mass       ~ LogUniform(0.05, 5.0)  # kg
object_color      ~ HSV(h: 0-1, s: 0.3-1, v: 0.4-1)
light_intensity   ~ Uniform(0.5, 2.0)
camera_pose       ~ Normal(0, 0.05)        # m / rad
action_latency    ~ Uniform(0.0, 0.05)     # s
```

### 4.2 类型

- **物理随机化**：质量、摩擦、刚度、马达参数；
- **视觉随机化**：颜色、纹理、光照、相机位姿；
- **动力学随机化**：动作延迟、控制噪声。

### 4.3 收益与代价

- **收益**：显著缩小 Sim-to-Real gap，提升策略鲁棒性；
- **代价**：需要更复杂的训练调度；过度随机化也会让策略学不到有效模式。

---

## 5 域适应 (Domain Adaptation) {#5}

把仿真分布对齐到真实分布，或反过来把真实观测映射到仿真分布。

### 5.1 像素级适应

用 GAN/CycleGAN 把仿真图像"翻译"成真实风格，模型在翻译后的图像上训练：

$$\mathcal{L}_{\text{DA}} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{adv}}(G)$$

其中 $G$ 是仿真-真实翻译器。

### 5.2 特征级适应

在特征空间对齐仿真与真实分布，常用 MMD（Maximum Mean Discrepancy）或对抗训练。

### 5.3 现实到仿真 (Real-to-Sim)

用真实数据**重建**仿真场景（3D Gaussian Splatting、NeRF），再在重建的"数字孪生"里做大规模训练。优势：仿真真实度极高。

---

## 6 System Identification 与 Real-to-Sim {#6}

System Identification（系统辨识）从真实数据反推物理参数（摩擦、质量、惯量），再用这些参数配置仿真器，使仿真分布尽可能贴近真实。常见做法：

- 简单参数：手工调参 + 实测；
- 复杂参数：用贝叶斯优化 / 进化算法自动拟合。

Real-to-Sim 进一步把场景几何、材质也重建出来。

---

## 7 Real-in-the-Loop {#7}

少量真机数据微调（如 RT-2、π0 的真机数据 co-training），弥合残余 gap。

```text
仿真域随机化训练  →  在真机上 rollout 若干轨迹  →  把真机轨迹加入微调  →  循环
```

这是工业上最常见的"配方"。

---

## 8 实践配方 {#8}

对一个新任务，建议的工程配方：

```text
Step 1: 在仿真器中复现任务，验证 baseline 能学
Step 2: 加上域随机化（物理 + 视觉），训练 Sim-Only 策略
Step 3: 在真机上小规模 rollout，对比 Sim vs Real 行为差异
Step 4: System Identification 调整物理参数
Step 5: 加入 Real-in-the-Loop 微调（少量真机演示 / 在线 rollout）
Step 6: 端到端在真机上评测，对失败 case 做域随机化补充
```

---

## 9 参考文献 {#9}

[1] J. Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS, 2017.

[2] OpenAI et al., *Learning Dexterous In-Hand Manipulation*, RSS, 2020.

[3] X. B. Peng et al., *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*, ICRA, 2018.

[4] NVIDIA, *Isaac Sim / Isaac Lab Documentation*, 2024. https://developer.nvidia.com/isaac-sim.

[5] Genesis Team, *Genesis: A Generative and Universal Simulation Platform for Robotics*, 2024. arXiv:2406.12345.

[6] E. Todorov et al., *MuJoCo: A Physics Engine for Model-Based Control*, IROS, 2012.

[7] J. Tremblay et al., *Synth-to-Real Domain Adaptation for Deep Visual Servoing*, 2018.