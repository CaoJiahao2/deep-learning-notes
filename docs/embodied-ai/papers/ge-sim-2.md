# GE-Sim 2.0：面向机器人操作的闭环视频世界模拟器

> GE-Sim 2.0 认为只会生成视频还不够：它加入状态专家、world judge 和加速框架，把 action-conditioned 视频模型变成可闭环训练与评测的机器操作模拟器。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | GE-Sim 2.0: A Roadmap Towards Comprehensive Closed-loop Video World Simulators for Robotic Manipulation (2026) |
| 机构 | AgiBot Research Team / Genie Envisioner |
| 关键词 | closed-loop simulator、multi-view video rollout、proprioceptive state expert、VLM world judge |
| 定位 | 2026 年机器人世界模拟平台的最新代表之一 |

---

## 1 背景与问题

许多机器人视频世界模型能根据动作生成未来画面，但距离可用模拟器仍有三块缺口：

1. 策略需要本体状态（关节角、夹爪状态），仅靠视频或 commanded action 会漂移；
2. 训练和评测需要机器可读的成功 / 失败信号，不能只看视频；
3. chunk-wise 大规模 rollout 要求高吞吐，逐步扩散模型太慢。

GE-Sim 2.0 把这三点显式补齐。


---

## 2 核心贡献

1. 基于动作条件多视角视频生成器，输入长时历史和动作轨迹，输出机器人执行视频；
2. 加入 proprioceptive state expert，从视频 latents 解码双臂关节角与夹爪状态；
3. 加入 VLM world judge，将 rollout 与任务指令对比并输出评分 / 奖励信号；
4. 构建加速框架，在保持保真度的情况下提升 rollout 吞吐；
5. 使用真实遥操作、策略部署和丰富物体交互数据训练；
6. 在 WorldArena 公开榜上以约 2B 参数取得领先，超过多个专用机器人世界模型与闭源视频生成器；
7. 闭环评测显示模拟结果与真实机器人结果在总体与逐 episode 层面较一致。


---

## 3 核心原理与方法详解

GE-Sim 2.0 的生成目标：

$$
\hat{x}_{t+1:t+L}^{1:V}
\sim
G_\theta(x_{\le t}^{1:V}, a_{t:t+L}),
$$

其中 $V$ 是相机视角数量。它不仅生成视频，还输出：

$$
\hat{s}_{t+1:t+L}
=
D_\phi(\text{latents}(\hat{x})),
$$

state expert $D_\phi$ 解码关节与夹爪状态。

World judge：

$$
r, \text{feedback}
=
J_\psi(\hat{x}, \ell),
$$

用 VLM 判断 rollout 是否完成语言任务。策略训练时可用 $\hat{x}$ 和 $\hat{s}$ 作为下一状态，用 $r$ 作为奖励，从而实现闭环。


---

## 4 数据与训练

- 训练数据包含数千小时真实机器人数据：遥操作、策略部署、接触丰富物体交互；
- 多视角历史帧和 end-effector 校准后的动作轨迹作为条件；
- 保留 GE-Sim 的 action-conditioned video backbone，并扩展真实部署与失败数据；
- state expert 从视频 latent 而不是原始像素解码，降低计算量；
- world judge 基于 VLM，可输出成功、进度和失败原因；
- 加速框架支持并行 chunk rollout 和多任务多 seed 采样；
- 评测覆盖 PSNR、SSIM、LPIPS、FID、FVD 与闭环策略一致性。


---

## 5 实验、结果与解读

论文报告：

- 在 WorldArena 公开榜排名第一，参数量约 2B；
- 超过 Ctrl-World、DreamDojo、GigaWorld、ABot 等机器人世界模型；
- 与 Sora、Veo 等闭源通用视频生成器相比，在机器人操作 rollout 指标上更强；
- 六类代表操作任务上 replay 指标逐任务领先；
- state expert 能给策略提供更可靠的下一 chunk 状态；
- world judge 使失败样本和奖励可以被自动识别；
- 闭环结果与真实机器人在 aggregate 与 episode 层面有一致性。


---

## 6 精读要点与实践建议

- **闭环 = 视频 + 状态 + 评分**，三者缺一就只是生成器；
- **proprioceptive state expert** 解决“视频看起来对但机器人状态不可用”的问题；
- **world judge 是自动 reward / evaluator**，也是 VLA 后训练的数据引擎；
- **多视角对 contact 很重要**：正面视角难以判断夹爪遮挡与双臂协调；
- **吞吐是训练基础设施的一部分**，不是后处理优化；
- **实践建议**：使用世界模型训练策略时，必须保留 failure rollout 与状态一致性检查。


---

## 7 局限与后续影响

- 视频模拟仍可能遗漏接触力、摩擦和柔性物体细节；
- state expert 依赖训练分布，真实传感器漂移会伤害闭环一致性；
- VLM judge 可能被视觉相似性误导；
- 高吞吐推理仍需要大量 GPU；
- 物理误差长期累积，需要真实数据校准；
- 机器人安全关键任务不能只依赖视频世界模拟器验收。


---

## 相关阅读

- [GE-Act 2.0 论文精读](./ge-act-2.md)
- [Cosmos 世界模型精读](./cosmos.md)
- [操作学习](../manipulation.md)
- [世界模型总览](../world-model.md)

---

## 参考文献

1. Liao et al., AgiBot Research Team. *GE-Sim 2.0: A Roadmap Towards Comprehensive Closed-loop Video World Simulators for Robotic Manipulation*, 2026. [arXiv:2605.27491](https://arxiv.org/abs/2605.27491)
2. NVIDIA et al. *Cosmos World Foundation Model Platform for Physical AI*, 2025.
3. Shang et al. *WorldArena: A Unified Benchmark for Evaluating Embodied World Models*, 2026.

