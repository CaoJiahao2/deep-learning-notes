# GR00T N1：通用人形机器人的双系统开源基础模型

> GR00T N1 用 System 2 VLM 做视觉语言理解，用 System 1 Diffusion Transformer 高频输出动作，为人形机器人提供开源基础模型与数据金字塔训练方案。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots (2025) |
| 机构 | NVIDIA |
| 关键词 | humanoid VLA、dual system、diffusion transformer、flow matching、synthetic data |
| 定位 | 人形机器人基础模型与 Isaac 生态的代表工作 |

---

## 1 背景与问题

人形机器人面临“身体复杂、数据稀缺”的双重挑战：

- 全身与双手自由度高，安全约束强；
- 单一人形平台真实数据很少；
- 任务需要语义理解、空间推理和实时运动控制；
- Sim-to-Real 与跨实体迁移不可回避。

NVIDIA 的路线是全栈：硬件 + Isaac 仿真 + 合成数据 + 真机数据 + 开源基础模型。


---

## 2 核心贡献

1. 提出 GR00T N1，一个面向通用人形机器人的开源 VLA foundation model；
2. 采用 System 2 / System 1 双系统：Eagle-2 VLM 与 Diffusion Transformer 动作模块端到端联合训练；
3. 使用真实轨迹、人类视频和合成生成的异构数据金字塔；
4. 在多 embodiment 仿真基准上超过 imitation learning 基线；
5. 部署到 Fourier GR-1 人形机器人执行语言条件双臂任务；
6. 开源 GR00T-N1-2B checkpoint、数据和仿真基准。


---

## 3 核心原理与方法详解

GR00T N1 的策略为：

$$
a_{t:t+H}
\sim
\pi_\theta(a\mid o_t, q_t, \ell, e),
$$

其中 $q_t$ 是本体状态，$e$ 是 embodiment token，用于区分机器人形态。

**System 2**：

- Eagle-2 VLM 编码图像与语言；
- 每帧约 64 个 image tokens；
- 输出视觉语言 latent tokens；
- 频率较低，负责语义、空间和任务理解。

**System 1**：

- DiT 动作模块处理噪声动作与状态；
- 通过 cross-attention 读取 VLM tokens；
- 使用 Flow Matching 训练连续动作；
- 论文报告可输出高频动作（约 120 Hz）。

训练目标包含 flow matching 损失和动作细节预测损失，以提升平滑性和精度。


---

## 4 数据与训练

**数据金字塔**：

- **Web 数据与人类视频**：数量最大，提供语义、视觉与人类运动知识；
- **合成数据**：Isaac 仿真、神经视频、动作生成，扩大任务覆盖；
- **真机数据**：量少但 embodiment-specific，用于最后对齐。

**训练流程**：

1. 用人类视频提取 latent action 或表示学习信号；
2. 在仿真和跨实体数据上训练通用 VLA；
3. 用 GR-1 遥操作与仿真数据 post-training；
4. 部署并评测真实双臂操作；
5. 使用 embodiment token 和可学习 state/action embedding 处理异构自由度。


---

## 5 实验、结果与解读

- 在标准仿真 benchmark 上超过 Octo、OpenVLA 等 generalist 与 imitation 基线；
- 真实 GR-1 24 个桌面任务和双臂任务上，GR00T-N1-2B 明显优于 Diffusion Policy；
- 预训练 + 后训练比仅用少量真机数据 from scratch 更具数据效率；
- 语言条件任务、单臂、双臂和 humanoid embodiment 可共享一个模型；
- 论文显示后训练数据规模和合成数据质量对最终成功率影响显著。


---

## 6 精读要点与实践建议

- **人形机器人需要数据金字塔**：不可能只靠真机遥操作；
- **System 1 / System 2 是工程必然**：VLM 低频理解与 DiT 高频控制解耦；
- **embodiment token** 是统一多机器人动作空间的轻量方案；
- **合成数据不是替代真机**，而是与真机数据分层对齐；
- **生态价值**：GR00T 与 Isaac、Cosmos、Omniverse 连接，形成 Physical AI 开发闭环。


---

## 7 局限与后续影响

- 仿真到真实仍有 domain gap；
- 人形机器人全身平衡与操作联合控制仍难；
- 开源版本主要覆盖 2B 模型，训练数据和完整工程链仍有限；
- 双手高自由度操作数据稀缺；
- 安全评估和长期可靠性需要更多真实部署验证。


---

## 相关阅读

- [Helix 论文精读](./helix.md)
- [Gemini Robotics 论文精读](./gemini-robotics.md)
- [Cosmos 世界模型精读](./cosmos.md)
- [具身智能前沿专题](../frontier.md)

---

## 参考文献

1. NVIDIA et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, 2025. [arXiv:2503.14734](https://arxiv.org/abs/2503.14734)
2. Chi et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, 2023.
3. NVIDIA et al. *Cosmos World Foundation Model Platform for Physical AI*, 2025.

