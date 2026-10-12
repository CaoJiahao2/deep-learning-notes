# Helix：Figure 的人形上半身双系统 VLA

> Helix 用 7B VLM 以 7–9 Hz 慢思考，80M 反应式策略以 200 Hz 快控制，首次用一套权重控制人形机器人完整上半身并支持双机器人协作。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 技术报告 | Helix: A Vision-Language-Action Model for Generalist Humanoid Control (2025) |
| 机构 | Figure |
| 关键词 | humanoid、System 1/System 2、whole upper-body、onboard GPU、multi-robot collaboration |
| 定位 | 商业化人形机器人 VLA 的代表系统 |

---

## 1 背景与问题

家庭环境物体长尾、姿态多样、任务开放。传统机器人技能需要人工编程或大量演示，难以扩展到“任意小 household object”。

Figure 的目标是让 VLM 的语义泛化直接转成高频连续控制，并且所有推理都在机器人机载低功耗 GPU 上完成，以满足商业部署。


---

## 2 核心贡献

1. 首个直接控制人形完整上半身的 VLA：手腕、躯干、头部和 individual fingers；
2. 首个在两台机器人上同时运行并协作完成长程操作的 VLA；
3. 一套神经网络权重覆盖收纳、抽屉、冰箱、多机器人交接等行为，不做任务专用微调；
4. System 2 与 System 1 端到端通信；
5. 机载嵌入式低功耗 GPU 推理，具备商业部署条件；
6. 约 500 小时高质量监督数据即可训练出较强物体泛化。


---

## 3 核心原理与方法详解

Helix 的双系统结构：

$$
z_t = f_{S2}(I_t, \ell, h_t),
$$

$$
a_{t:t+H} = f_{S1}(I_t, s_t, z_t).
$$

- **System 2**：7B internet-pretrained VLM，7–9 Hz；处理语言、场景语义、目标选择与高层意图；
- **System 1**：80M visuomotor policy，200 Hz；读取 S2 latent，输出连续上半身控制；
- 两个系统端到端训练，使 latent 语义向量成为稳定接口；
- S1 可快速反应，例如协作中调整另一台机器人变化带来的接触与姿态；
- 不需要任务专用 action head 或 per-task fine-tuning。


---

## 4 数据与训练

- 训练数据约 500 小时高质量监督数据，规模远小于许多 VLA 数据集；
- 覆盖完整上半身高维动作：头、躯干、双腕、手指；
- 单一权重学习全部行为；
- 所有推理在机载低功耗 GPU 上执行；
- 训练目标强调从 VLM latent 到连续控制的映射；
- 双机器人实验使用同一模型同时部署，不做显式中央脚本控制。


---

## 5 实验、结果与解读

Figure 报告的关键表现：

- 可按自然语言抓取数千个未见 household objects；
- 对抽象语言有泛化，例如“拿沙漠物品”能识别玩具仙人掌；
- 能操作抽屉、冰箱和容器；
- 两台 Figure 机器人可协作收纳从未见过的食品杂货；
- S1 在协作中保持 200 Hz 反应，S2 维持语义目标；
- 新技能可通过语言即时指定，而不需要新演示集。


---

## 6 精读要点与实践建议

- **双频率接口**是 humanoid VLA 的关键工程创新：慢系统提供语义，快系统保证反应；
- **latent communication** 比文本接口更快，也比直接共享参数更清晰；
- **机载推理**决定商业可行性，不能假设始终有云端 GPU；
- **统一权重**降低部署与维护复杂度；
- **数据质量重于数量**：500 小时高质量数据已能支撑强泛化；
- **评测要看真实物体长尾**，而不只看固定 benchmark。


---

## 7 局限与后续影响

- 技术报告缺少完整论文级实验细节与第三方复现；
- 家庭任务的持续成功率、失败恢复和安全边界未完全公开；
- VLM 语义错误会传导到 S1；
- 200 Hz 上半身控制不等于全身平衡与行走问题解决；
- 双机器人协作仍处于演示与有限任务阶段；
- 训练数据分布和测试物体分布难以独立审计。


---

## 相关阅读

- [GR00T N1 论文精读](./groot-n1.md)
- [Gemini Robotics 论文精读](./gemini-robotics.md)
- [具身智能应用与产业版图](../applications.md)

---

## 参考文献

1. Figure. *Helix: A Vision-Language-Action Model for Generalist Humanoid Control*, 2025. [项目页面](https://www.figure.ai/news/helix)
2. NVIDIA et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, 2025.
3. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024.

