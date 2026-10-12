# 具身智能经典论文与新模型精读

> 本页是具身智能论文阅读地图：按「RT 系列 → π 系列 → 最新 VLA → 世界模型」组织。每篇论文单独一页，覆盖背景、核心贡献、架构、数据、实验、工程启示、局限与参考文献。

---

## 阅读路线

### 路线 A：VLA 主线

```text
SayCan → PaLM-E → RT-1 → RT-2 → RT-X
      → OpenVLA → π0 → RDT-1B
      → π0.5 → Helix / GR00T N1 / Gemini Robotics
      → π*0.6 → π0.7 → Xiaomi-Robotics-1
```

推荐先读：

1. [SayCan](saycan.md)：理解 LLM + affordance 的分层思想；
2. [RT-1](rt-1.md)：看真机数据规模化；
3. [RT-2](rt-2.md)：看 VLM 知识如何迁移成 action token；
4. [π0](pi-0.md)：看 VLM + Flow Matching 动作专家；
5. [π0.7](pi-0-7.md)：看上下文条件化与组合泛化；
6. [Xiaomi-Robotics-1](xiaomi-robotics-1.md)：看 10 万小时级真实轨迹 scaling。

### 路线 B：世界模型主线

```text
World Models → DreamerV3
            → Genie → Genie 3 / Sora 2
            → Cosmos → V-JEPA 2
            → GE-Sim 2.0 → GE-Act 2.0
```

推荐先读：

1. [World Models](world-models-2018.md)：建立 V / M / R 直觉；
2. [DreamerV3](dreamer-v3.md)：看 latent world model 如何做 RL；
3. [Genie](genie.md)：看无标注视频如何学潜在动作；
4. [Cosmos](cosmos.md)：看物理 AI 世界基础模型平台；
5. [V-JEPA 2](v-jepa-2.md)：看非生成式 latent planning；
6. [GE-Sim 2.0](ge-sim-2.md) / [GE-Act 2.0](ge-act-2.md)：看 2026 年闭环模拟与 world-action scaling。

---

## RT 系列与早期奠基

| 年份 | 页面 | 一句话定位 |
|---:|---|---|
| 2022 | [SayCan](saycan.md) | LLM 规划 + affordance 选择技能 |
| 2022 | [RT-1](rt-1.md) | 130k 真机 episodes 验证多任务 Transformer |
| 2023 | [PaLM-E](palm-e.md) | 连续多模态观测注入 LLM |
| 2023 | [RT-2](rt-2.md) | 确立 Vision-Language-Action 范式 |
| 2023 | [RT-X / Open X-Embodiment](rt-x.md) | 22 种机器人数据统一与跨实体正迁移 |
| 2023 | [RT-Trajectory](rt-trajectory.md) | 用轨迹草图增强任务泛化 |
| 2024 | [RT-H](rt-h.md) | 用语言化中间动作分层 |

## π 系列

| 年份 | 页面 | 核心变化 |
|---:|---|---|
| 2024 | [π0](pi-0.md) | VLM + Flow Matching 动作专家 |
| 2025 | [π0.5](pi-0-5.md) | 异构数据与网页知识带来开世界泛化 |
| 2025 | [π*0.6](pi-star-0-6.md) | RECAP 真实机器人 RL 与人类纠错 |
| 2026 | [π0.7](pi-0-7.md) | 多样上下文、视觉子目标与组合泛化 |

## 最新与代表性 VLA

| 年份 | 页面 | 关键路线 |
|---:|---|---|
| 2024 | [OpenVLA](openvla.md) | 开源 7B VLA 与高效微调 |
| 2025 | [RDT-1B](rdt-1b.md) | 双臂 Diffusion Transformer |
| 2025 | [GR00T N1](groot-n1.md) | 人形机器人开源双系统 VLA |
| 2025 | [Gemini Robotics](gemini-robotics.md) | Gemini 2.0 驱动的通用 VLA |
| 2025 | [Helix](helix.md) | Figure 人形上半身 7–9Hz / 200Hz 双系统 |
| 2026 | [Xiaomi-Robotics-1](xiaomi-robotics-1.md) | 100k+ 小时真实轨迹与 2B/5B/10B scaling |

## 世界模型

| 年份 | 页面 | 关键路线 |
|---:|---|---|
| 2018 | [World Models](world-models-2018.md) | VAE + MDN-RNN + controller |
| 2023 | [DreamerV3](dreamer-v3.md) | RSSM 想象训练 + 单一配置跨域 RL |
| 2024 | [Genie](genie.md) | 无标注视频学潜在动作与交互环境 |
| 2025 | [Cosmos](cosmos.md) | Physical AI 世界基础模型平台 |
| 2025 | [V-JEPA 2](v-jepa-2.md) | 表征空间预测与零样本机器人规划 |
| 2025 | [Genie 3](genie-3.md) | 720p / 24FPS 实时可交互世界 |
| 2025 | [Sora 2](sora-2.md) | 视频生成作为世界模拟器 |
| 2026 | [GE-Sim 2.0](ge-sim-2.md) | 闭环视频世界模拟器 |
| 2026 | [GE-Act 2.0](ge-act-2.md) | World-Action Model 从零预训练与 scaling |

---

## 技术演进速查

| 维度 | 2022 | 2023 | 2024–2025 | 2026 |
|---|---|---|---|---|
| 语义来源 | 语言条件 + 可供性 | VLM / 网页知识 | VLM 骨干 + 网页/视频 co-training | 多模态 prompt + 世界模型子目标 |
| 动作输出 | 离散 token | 离散 action token | Flow Matching / Diffusion | Flow Matching + 记忆 + RTC |
| 数据来源 | 单平台真机 | 跨实体数据集 | OXE / DROID / 自采异构数据 | UMI 100k 小时、失败轨迹、人类视频 |
| 训练目标 | imitation | VQA + action co-training | 多任务模仿 + 后训练 | imitation + RL + context steering |
| 世界模型 | latent dynamics | 视频生成雏形 | 世界基础模型平台 | 闭环模拟、world judge、WAM |
| 主要短板 | 数据贵、控制粗 | 推理慢、动作离散 | 泛化与可靠性 | 安全、成本、长期自主性 |

---

## 论文精读方法

读具身智能论文时建议固定问以下 10 个问题：

1. 输入观测有哪些：图像、深度、状态、触觉、历史？
2. 输出动作是离散 token、连续 chunk，还是技能/子任务？
3. 预训练数据与后训练数据分别是什么？
4. embodiment 如何统一：动作空间、token、投影、adapter？
5. 高层与低层如何分层，频率各是多少？
6. 训练目标是 imitation、RL、生成、预测还是混合？
7. 有没有世界模型或中间未来表示？
8. 评测是否覆盖 unseen instruction / scene / object / embodiment？
9. 报告的是成功率、吞吐、任务进度，还是人类偏好？
10. 失败模式是什么，系统如何恢复或继续学习？

---

## 与其他章节的关系

- [VLA 模型](../vla.md)：先看横向综述，再进入单篇精读；
- [世界模型](../world-model.md)：先看概念与分类，再读论文细节；
- [数据集与数据采集](../data.md)：理解 OXE、DROID、AgiBot、UMI 与数据飞轮；
- [Sim-to-Real 与仿真器](../sim2real.md)：理解世界模型、合成数据与真实部署的关系；
- [具身智能评测基准](../evaluation.md)：理解成功率、SPL、任务进度与泛化协议。

---

## 参考文献

1. Open X-Embodiment Collaboration et al. *Open X-Embodiment: Robotic Learning Datasets and RT-X Models*, 2023/2025.
2. Black et al. *π0 / π0.5 / π*0.6 / π0.7*, 2024–2026.
3. NVIDIA et al. *Cosmos World Foundation Model Platform for Physical AI*, 2025.
4. Assran et al. *V-JEPA 2*, 2025.
5. AgiBot Research Team. *GE-Sim 2.0 / GE-Act 2.0*, 2026.
