# Xiaomi-Robotics-1：10 万小时真实轨迹上的 VLA Scaling

> Xiaomi-Robotics-1 用 UMI 设备采集超过 10 万小时真实操作，先学“状态如何改变”，再后训练成“听指令控制机器人”，把 VLA scaling 推向新量级。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories (2026) |
| 机构 | Xiaomi Robotics |
| 关键词 | VLA scaling、UMI、auto-labeling、Mixture-of-Transformers、flow matching |
| 定位 | 截至 2026 年公开报告中真实操作轨迹规模最大的 VLA 之一 |

---

## 1 背景与问题

机器人 VLA 的主要瓶颈是数据。真机遥操作慢、贵且硬件绑定；互联网视频虽多但没有动作。Xiaomi-Robotics-1 选择中间路线：用 UMI 手持设备采集大规模真实操作，再通过跨实体后训练对齐到机器人。

这使数据既有真实物理交互，又能扩展到大量环境与任务。


---

## 2 核心贡献

1. 使用 UMI 设备采集超过 100k 小时真实操作轨迹；
2. 构建自动标注流水线，用 VLM 为轨迹片段生成自然语言 state-transition 描述；
3. 提出两阶段训练：预训练学习动作生成，post-training 对齐机器人本体与 imperative instruction；
4. 使用 Mixture-of-Transformers：Qwen3-VL 骨干 + DiT 连续动作生成；
5. 在 2B / 5B / 10B 变体上验证数据规模与模型规模 scaling；
6. 在 RoboCasa、RoboCasa365、VLABench、RoboDojo 上达到当时 SOTA；
7. 论文承诺开源代码与模型 checkpoint。


---

## 3 核心原理与方法详解

模型学习：

$$
a_{t:t+H}
\sim
\pi_\theta(a\mid o_t, s_t, \ell).
$$

**Mixture-of-Transformers**：

- VLM 分支：Qwen3-VL 处理图像、语言与观测；
- DiT 分支：较小 hidden size，与 VLM 层数对应，生成动作块；
- DiT 读取 VLM KV cache、机器人状态和噪声动作；
- 用 adaLN 注入 Flow Matching timestep。

训练损失：

$$
\mathcal{L}
=
\mathcal{L}_{\text{Flow}}
+
\mathcal{L}_{\text{Regression}}
+
\lambda \mathcal{L}_{\text{NTP}},
\quad \lambda=0.1.
$$

其中 Regression 分支用 Choice Policies 在 VLM 内预测候选动作块与分数，为 DiT 提供辅助监督；NTP 保持视觉语言能力。为防止 DiT 抄 VLM 动作候选，DiT attention 排除 action-related tokens。


---

## 4 数据与训练

**预训练**：

- 数据为超过 100k 小时 UMI 真实操作；
- 轨迹切段后用 Qwen3.5-27B 自动 caption 场景状态转移；
- 语言形式是“把夹爪/物体从状态 A 变到状态 B”；
- 混合视觉语言数据，比例约 1:9；
- 使用高吞吐 producer-consumer 标注流水线，100k 小时约两周完成标注。

**后训练**：

- 约 10k 小时跨实体数据；
- 包含 7.2k 小时移动操作与双臂机器人、1k 小时指令标注 UMI、Bridge V2 / RT-1 / DROID 等开源数据；
- 从 state-transition 描述转为人类 imperative 指令；
- 使用相对末端位姿、底盘速度、腰部增量等统一动作表示；
- 提供 2B / 5B / 10B 模型变体，推理时 5 步 Euler 积分生成动作块。


---

## 5 实验、结果与解读

论文结果体现明显 scaling：

- 预训练数据从 12.5% 到 100% 时，动作验证误差和后训练成功率持续提升；
- 无动作预训练的 5B 后训练整体成功率约 26%，使用完整预训练数据约 75%；
- 在 RoboCasa365 达到 57.4%–57.6% 成功率，超过此前最佳 46.6%；
- 在 RoboDojo 平均 20.07，显著超过此前 13.07；
- VLABench 和 RoboCasa 也取得 SOTA；
- 少量演示可微调复杂灵巧任务；
- 相比 π0.5、X-VLA、Spatial Forcing 等基线在多基准上更强。


---

## 6 精读要点与实践建议

- **数据规模仍是 VLA 最直接的杠杆**，但数据必须有多样环境和准确语言标注；
- **UMI 是解耦硬件与数据规模的有效设备**：先学通用操作，再对齐机器人；
- **自动标注流水线决定 scaling 可行性**，人工 caption 无法扩展；
- **state-transition pre-training + imperative post-training** 是清晰的语义对齐两段式；
- **MoT 架构**让 VLM 语义与 DiT 动作生成共享上下文但保留容量差异；
- **实践建议**：入库时保存末端坐标系、本体状态、场景 caption 和语言指令，便于跨阶段训练。


---

## 7 局限与后续影响

- UMI 夹爪与真实机器人动力学、本体感知和延迟仍有差距；
- 100k 小时数据的隐私、版权与场景覆盖治理需要关注；
- 自动 caption 可能产生错误或歧义；
- benchmark SOTA 不完全等价于任意家庭任务可靠性；
- 论文发布时的开源范围和工程复现成本仍需验证；
- 长程规划和安全恢复仍是瓶颈。


---

## 相关阅读

- [π0.7 论文精读](./pi-0-7.md)
- [π0 论文精读](./pi-0.md)
- [OpenVLA 论文精读](./openvla.md)
- [数据集与数据采集](../data.md)

---

## 参考文献

1. Xiaomi Robotics Team. *Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories*, 2026. [arXiv:2607.15330](https://arxiv.org/abs/2607.15330)
2. Black et al. *π0.5: a Vision-Language-Action Model with Open-World Generalization*, 2025.
3. Chi et al. *Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots*, 2024.
4. Qwen Team. *Qwen3-VL*, 2025.

