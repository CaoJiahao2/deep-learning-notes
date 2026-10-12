# GE-Act 2.0：从零预训练与 scaling 世界-动作模型

> GE-Act 2.0 把可生成未来视觉状态的世界模型与 inverse dynamics action model 拆开预训练，再用 KASO 对齐，使 world-action model 的 scaling 能被系统验证。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | GE-Act 2.0: Pretraining and Scaling a World-Action Model for Robotic Manipulation (2026) |
| 机构 | AgiBot Research Team |
| 关键词 | world-action model、CoAE、single-step visual planner、inverse dynamics、KASO |
| 定位 | 2026 年 world-action model pretraining 与 scaling 代表作 |

---

## 1 背景与问题

VLA 直接学习 observation-action 映射，但不显式要求理解场景将如何变化。World-action model（WAM）则尝试同时学：

- 未来状态如何演化；
- 给定当前状态与目标未来，应执行什么动作。

现有 WAM 多继承预训练视频生成器，导致视觉生成、动作反演和共训练的 scaling 关系不清楚。GE-Act 2.0 选择从操作数据中从头预训练所有生成与动作组件。


---

## 2 核心贡献

1. 提出 CoAE（control-oriented autoencoder），在激进压缩下保留动作和指令相关信息；
2. 提出 single-step visual planner（SVP），一次可微前向生成完整未来视觉 latent；
3. 提出 inverse dynamics model（IDM），从当前与未来状态预测动作；
4. SVP 可用 action-free video 预训练，IDM 可用 instruction-free robot trajectories 预训练；
5. 提出 KASO（knowledge-aligned selective optimization），只选择与记录动作行为兼容的预测未来做监督；
6. 在 100 个任务、20 个技能组上做无 per-task fine-tuning 的 zero-shot OOD 评测；
7. 验证 300h 到 30000h co-training 数据的 scaling 规律。


---

## 3 核心原理与方法详解

系统由三部分组成：

**1. CoAE**

$$
z=E_{\text{CoAE}}(x),
\quad
\hat{x}=D_{\text{CoAE}}(z).
$$

采用 64 倍空间下采样，比常见 8×/16× tokenizer 更激进；用多教师语义对齐和动作可恢复性约束保留控制信息。

**2. SVP**

$$
\hat{z}_{t+1}
=
S_\theta(z_t, \ell).
$$

基于单步流生成，一次前向输出未来多视角 latents。

**3. IDM**

$$
\hat{a}_{t:t+H}
=
I_\phi(z_t, \hat{z}_{t+1}).
$$

从当前状态与预测未来状态反推动作块。

**4. KASO**

预测的未来可能有多个合理结果，但录制动作只对应一个模式。KASO 采样多个候选未来，用当前 IDM 判断哪些与记录动作兼容，只对兼容候选施加端到端监督，避免模式平均。


---

## 4 数据与训练

- SVP pretraining 使用约 39,000 小时 instruction-video 数据，可包含 action-free 人类视频；
- IDM pretraining 使用约 32,000 小时 action-labeled 机器人轨迹，可包含失败和部署 rollout；
- co-training 使用约 30,000 小时完全标注数据；
- SVP flow generator 约 2.51B，IDM 约 0.56B，CoAE 与 VLM 冻结；
- 评测任务包含 held-out scenes、backgrounds、lighting 和 object instances；
- 不做 per-task fine-tuning，直接评估 pretrained checkpoint；
- 比较 G1-OP 与 G2-90D 两种 embodiment。


---

## 5 实验、结果与解读

关键结果：

- co-training 从 300h 增加到 30,000h，G1-OP zero-shot OOD 成功率从 17.1% 提升到 44.1%；
- G2-90D 从 13.4% 提升到 31.1%；
- G2-90D 在 co-training 中占比不足 2%，但提升 17.7 points，说明跨 embodiment 正迁移；
- 提升覆盖 G1-OP 的 19/20 和 G2-90D 的 18/20 技能组；
- 技能覆盖度与 OOD 成功率强相关（Pearson $r=0.80$，Spearman $\rho=0.85$）；
- 对 object、color、shape、position 指代至少 90% trials 能正确 grounding；
- 能遵循与已承诺行为或常规场景关联相冲突的显式指令。


---

## 6 精读要点与实践建议

- **WAM 的 scaling 需要拆组件**：视觉 planner 与 inverse dynamics 可以利用互补数据源；
- **控制导向 tokenizer**：压缩率高不等于好，关键是 latents 能否恢复动作和指令；
- **single-step future**降低生成与动作模块的耦合成本，也便于可微连接；
- **KASO 解决 validity gap**：视觉上合理的未来不一定与训练动作一致；
- **失败与部署数据有价值**，尤其适合 IDM 预训练；
- **实践建议**：zero-shot OOD 评测应固定任务、场景、光照和物体实例，避免 fine-tuning 掩盖预训练能力。


---

## 7 局限与后续影响

- zero-shot 成功率仍远低于专用微调模型；
- 单步未来生成可能牺牲长时动态细节；
- CoAE 需要多教师与动作探针，训练复杂；
- KASO 增加采样和选择成本；
- 主要验证机器人操作，导航与全身人形控制待扩展；
- 世界视觉计划可检查，但物理误差仍需真实闭环校准。


---

## 相关阅读

- [GE-Sim 2.0 世界模型精读](./ge-sim-2.md)
- [V-JEPA 2 论文精读](./v-jepa-2.md)
- [π0 论文精读](./pi-0.md)
- [世界模型总览](../world-model.md)

---

## 参考文献

1. AgiBot Research Team. *GE-Act 2.0: Pretraining and Scaling a World-Action Model for Robotic Manipulation*, 2026. [arXiv:2609.05588](https://arxiv.org/abs/2609.05588)
2. AgiBot Research Team. *GE-Sim 2.0*, 2026.
3. Assran et al. *V-JEPA 2*, 2025.
4. Black et al. *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024.

