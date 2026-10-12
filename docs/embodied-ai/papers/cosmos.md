# Cosmos：Physical AI 的世界基础模型平台

> Cosmos 把视频数据治理、视频 tokenizer、预训练世界基础模型和 post-training 流程打包成开源平台，目标是让物理 AI 先在数字世界中训练。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | Cosmos World Foundation Model Platform for Physical AI (2025) |
| 机构 | NVIDIA |
| 关键词 | world foundation model、physical AI、video tokenizer、diffusion / autoregressive、synthetic data |
| 定位 | 机器人与自动驾驶世界模型基础设施的代表平台 |

---

## 1 背景与问题

Physical AI 的难点是交互数据昂贵且有风险。真实机器人试错可能损坏硬件、环境甚至人。NVIDIA 提出的解决路径是：

```text
Physical AI
    ├── policy model：数字孪生中的智能体
    └── world model：物理世界的数字孪生
```

Cosmos 的目标不是单一 demo，而是可微调的世界基础模型平台。


---

## 2 核心贡献

1. 定义 World Foundation Model（WFM）作为可 post-training 的通用世界模型；
2. 构建从 20M 小时视频筛出约 100M 个 2–60 秒片段的数据治理流水线；
3. 提供连续与离散视频 tokenizer；
4. 发布 diffusion 与 autoregressive 两条路线的预训练 WFM；
5. 展示 text2world、image2world、video2world 和 action-conditioned post-training；
6. 与 Omniverse / Isaac 生态结合，用于自动驾驶、机器人和物理 AI 数据生成；
7. 以开放权重和宽松许可发布 Cosmos-Predict 等模型。


---

## 3 核心原理与方法详解

Cosmos 将世界建模为：

$$
x_{t+1:t+L}
\sim
p_\theta(x_{t+1:t+L}\mid x_{\le t}, c),
$$

其中条件 $c$ 可以是文本、图像、视频、结构化布局或动作。

**预训练 / 后训练范式**：

```text
20M 小时原始视频
    ↓ 质量过滤 / 切分 / VLM caption
100M 高动态 clip
    ↓ tokenizer + 大规模生成训练
预训练 WFM
    ↓ 物理场景数据 post-training
定制世界模型
```

**模型路线**：

- Diffusion WFM：在 latent space 去噪生成未来视频；
- Autoregressive WFM：逐 token 预测未来；
- 视频 tokenizer 决定压缩率、时间一致性和可控性；
- 后训练可加入相机位姿、动作、机器人状态等控制信号。


---

## 4 数据与训练

- 数据 curation 强调动态丰富、视觉质量、物理相关性和字幕质量；
- 使用 VLM 对每 256 帧生成 caption；
- Ray 编排异构模型，GPU 硬编解码加速视频处理；
- 提供 tokenizer 消融，研究压缩率与生成质量的权衡；
- Cosmos-Predict 面向世界生成，Cosmos-Transfer 支持多控制生成；
- Cosmos-Reason 进一步加入物理常识与空间推理；
- post-training 示例覆盖自动驾驶、机器人操作等 Physical AI 场景。


---

## 5 实验、结果与解读

论文报告：

- 筛选后的视频数据能提升动态与物理表现；
- 不同 tokenizer 在速度、时间一致性、细节和训练稳定性上各有优势；
- diffusion 模型视觉质量强，autoregressive 模型便于扩展和条件化；
- 后训练后的 WFM 可生成与特定机器人 / 驾驶场景一致的未来；
- 可用于策略训练、评测和合成数据增强；
- 开放权重降低了世界模型研究门槛。


---

## 6 精读要点与实践建议

- **世界模型要平台化**：数据、tokenizer、基础模型、后训练工具缺一不可；
- **视频数据治理是核心**：不是任意视频都有益，动态、质量与 caption 决定可用性；
- **post-training 比 zero-shot 更现实**：通用 WFM 需要场景动作数据对齐；
- **与世界模型融合的三种方式**：合成数据、想象规划、策略评测；
- **实践建议**：机器人团队应保存多相机视频、动作、位姿、接触事件和任务结果，便于 Cosmos 式后训练。


---

## 7 局限与后续影响

- 视频生成的物理一致性仍不完美，可能出现不可实现动作；
- 缺少闭环本体状态和力反馈，仅靠视觉难以仿真接触细节；
- 高分辨率长 rollout 成本高；
- Sim-to-Real 差异仍存在；
- 需要严格治理合成数据偏差与安全风险；
- 后来的 GE-Sim 2.0 等工作专门补齐闭环状态和 world judge。


---

## 相关阅读

- [GR00T N1 论文精读](./groot-n1.md)
- [Genie 论文精读](./genie.md)
- [GE-Sim 2.0 世界模型精读](./ge-sim-2.md)
- [世界模型总览](../world-model.md)

---

## 参考文献

1. NVIDIA et al. *Cosmos World Foundation Model Platform for Physical AI*, 2025. [arXiv:2501.03575](https://arxiv.org/abs/2501.03575)
2. NVIDIA et al. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*, 2025.
3. Bruce et al. *Genie: Generative Interactive Environments*, 2024.

