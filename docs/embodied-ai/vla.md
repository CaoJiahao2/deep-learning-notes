# VLA 模型 (Vision-Language-Action)

> 以预训练的视觉-语言模型为骨架，将自然语言指令与视觉观测编码为条件，直接输出动作。

---

## 目录

1. [通用架构](#1)
2. [动作输出路线](#2)
3. [代表模型](#3)
4. [动作分块与平滑](#4)
5. [离散动作 vs 连续动作](#5)
6. [参考文献](#6)

---

## 1 通用架构 {#1}

通用 VLA 由三部分组成：

```text
   语言指令 ℓ: "把苹果放到碗里"
        │
        ▼
  ┌───────────────┐      ┌───────────────────────────┐      ┌───────────────┐
  │ 视觉编码器    │ ───▶ │   LLM backbone             │ ───▶ │ 动作输出头     │
  │ (ViT / CLIP)  │      │   (自回归 / 前馈)            │      │ (token 或连续) │
  └───────────────┘      └───────────────────────────┘      └───────────────┘
        ▲                                                    │
        │                                                    ▼
   RGB(-D) 观测                                        动作 a_{t:t+H}
```

- **视觉编码器**：常用 ViT 或 CLIP-ViT，把图像（多视角）压缩为视觉 token 序列；
- **语言模型 backbone**：通常是一个预训练 LLM（如 Llama、Gemma），处理视觉与语言 token 的融合表示；
- **动作输出**：两条主流路线——离散化 action token / 连续动作头（详见 §2）。

---

## 2 动作输出路线 {#2}

### 2.1 离散化 Action Token

把连续动作 bin 化为词表（类似语言 token），用 LLM 自回归采样：

$$a_i \in \{0, 1, \dots, K-1\}, \quad a_i \sim \pi_\theta(a_i \mid o_{\le t}, \ell, a_{<i})$$

其中 $K$ 是动作词表大小（典型 256 或 1024）。这种方式的好处是：

- 直接复用 LLM 训练与推理栈；
- 可与语言 token 联合训练（co-training）。

代表工作：RT-2、OpenVLA。

### 2.2 连续动作头

通过 MLP、Diffusion 或 Flow Matching 直接回归连续动作向量。代表工作：π0、π0.5、Octo。

```text
方案 A：MLP 头  → a_t = MLP_θ(h_t)
方案 B：Diffusion → a_t = Sample(denoise(h_t, k=0..K))
方案 C：Flow Matching → a_t = ODE(φ_θ(h_t, t), t=1..0)
```

### 2.3 两条路线对比

| 维度 | 离散 Action Token | 连续动作头 |
|------|-------------------|-----------|
| 精度 | 受词表粒度限制 | 任意精度 |
| 多模态 | 易于联合训练 | 需额外建模多峰 |
| 推理 | 复用 LLM 解码 | 需专用 ODE/SDE 求解器 |
| 代表 | RT-2, OpenVLA | π0, π0.5 |

---

## 3 代表模型 {#3}

### 3.1 RT-2（Google DeepMind, 2023）

把机器人动作离散化为 token，与网页文本-图像数据共同训练（co-training），使模型同时具备 web 知识与机器人技能，首次展示大规模 VLM 迁移到机器人操作的能力。

### 3.2 OpenVLA（Kim et al., 2024）

开源 VLA，基于 Prismatic VLM（Llama-2 backbone + DINOv2/SigLIP 双视觉编码器），在 Open X-Embodiment 数据上训练，权重与代码公开，是社区复现与改进的基础。

### 3.3 π0 / π0.5（Physical Intelligence, 2024–2025）

用 flow matching 在动作空间上生成**连续动作分布**，天然支持多模态动作（同一观测下多种合理动作），并支持动作分块输出。π0.5 进一步引入跨实体与跨任务的层次化结构，并成为首批开源的"通用机器人基础模型"之一。

### 3.4 Octo（Team, 2024）

通用机器人策略，基于 Transformer，通过灵活的输入输出头适配多种机器人形态与控制频率，展示从 Open X-Embodiment 多源数据训练得到的可迁移表征。

### 3.5 SpatialVLA / DexVLG

强调**3D 空间表征**的 VLA，把点云或体素特征纳入视觉输入，提升对几何的感知。

### 3.6 GR00T（NVIDIA, 2025）

NVIDIA Isaac Lab 的基础模型，专为人形机器人设计，强调"全身控制 + 语言指令"，把 VLA 推广到 humanoid 平台。

### 3.7 HIL-SERL / HiRT

- **HIL-SERL**：在模仿学习基础上加入**人干预 RL**（Human-in-the-Loop Sample-Efficient RL），用少量人类纠正信号显著提升成功率；
- **HiRT**（Hierarchical Robot Transformer）：双层结构，高层规划、低层执行，平衡推理延迟与动作精度。

---

## 4 动作分块与平滑 {#4}

为降低高频控制下的延迟与累计误差，现代 VLA 通常一次预测未来 $H$ 步动作，称为 **action chunking**：

$$a_{t:t+H} = \pi_\theta(o_t, \ell)$$

随后用执行窗口内若干步（或加平滑/低通滤波）发送给机器人。这样做的好处是：

- 减少策略推理频率；
- 平滑轨迹；
- 允许模型显式建模未来动作的时序耦合。

工程上常用技巧：时间加权（前期步权重更大）、指数平滑、低通滤波。

---

## 5 离散动作 vs 连续动作 {#5}

离散动作简单、与 LLM 解码栈一致；连续动作精度更高、表达力更强。混合方案：

- **混合 Token**：把连续动作拆为多个 bin 的组合；
- **Discrete Diffusion**：对离散动作序列做扩散模型（类似图像生成中 MDT 思路）；
- **Continuous Token**：在 LLM 隐空间直接接 ODE 输出头。

---

> 相关文档：VLA 的评测基准详见 [`embodied-ai/evaluation.md`](evaluation.md)，操作学习详见 [`embodied-ai/manipulation.md`](manipulation.md)，世界动作模型/世界模型详见 [`embodied-ai/world-model.md`](world-model.md)。

## 6 参考文献 {#6}

[1] Brohan et al., *RT-1: Robotics Transformer for Real-World Control at Scale*, CoRL, 2022. arXiv:2203.07814.

[2] Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, CoRL, 2023. arXiv:2307.15818.

[3] Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246, 2024.

[4] Physical Intelligence, *π0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164, 2024.

[5] Physical Intelligence, *π0.5: Open-World Generalist Robot Policy*, 2025.

[6] Octo Team, *Octo: An Open-Source Generalist Robot Policy*, RSS, 2024.

[7] NVIDIA, *GR00T: Foundation Models for Humanoid Robots*, 2025.

[8] D. Qu et al., *SpatialVLA: Exploring Spatial Representations for Vision-Language-Action Models*, 2024.

[9] J. Luo et al., *HIL-SERL: Human-in-the-Loop Sample-Efficient RL for Robotics*, 2024.

[10] HiRT Team, *HiRT: Hierarchical Robot Transformer*, 2024.