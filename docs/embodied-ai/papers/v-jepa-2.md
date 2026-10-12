# V-JEPA 2：非生成式视频世界模型与零样本机器人规划

> V-JEPA 2 在 100 万小时互联网视频上做联合嵌入预测，再用不到 62 小时机器人视频训练 action-conditioned predictor，实现零样本 Franka 抓放规划。

---

## 论文速览

| 项目 | 内容 |
|---|---|
| 论文 | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning (2025) |
| 机构 | Meta FAIR / Mila |
| 关键词 | JEPA、latent prediction、action-conditioned world model、video QA、robot planning |
| 定位 | 抽象表征世界模型路线的最新代表 |

---

## 1 背景与问题

像素生成式世界模型视觉质量高，但用于规划有明显缺点：

- 每个候选动作都要生成视频，计算昂贵；
- 很多与决策无关的细节占用模型容量；
- 生成误差随时间累积；
- FVD 好不等于规划好。

V-JEPA 2 的路线是：在表征空间预测未来，不逐像素重构画面。


---

## 2 核心贡献

1. 在超过 1M 小时互联网视频和 1M 图像上预训练 V-JEPA 2；
2. 使用 masked video prediction 的 joint-embedding predictive objective；
3. 在动作理解、动作 anticipation 和 video QA 上取得强结果；
4. 冻结视频编码器，训练 V-JEPA 2-AC action-conditioned predictor；
5. 只用少于 62 小时无标签 DROID 机器人视频，不采集目标实验室数据；
6. 在两个实验室的 Franka 上零样本完成 reach、grasp、pick-and-place；
7. 相比视频生成世界模型，规划速度大幅提升。


---

## 3 核心原理与方法详解

V-JEPA 2 的预训练目标是在表征空间预测被遮挡区域：

$$
\mathcal{L}
=
\left\|
\phi(x_y)
-
g_\theta(\phi(x_x), m_y)
\right\|,
$$

其中 $x_x$ 是可见上下文，$m_y$ 指定待预测的时空区域，预测器 $g_\theta$ 输出目标表征。

**V-JEPA 2-AC** 加入动作条件：

$$
\hat{z}_{t+1}
=
P_\phi(z_t, a_t),
$$

冻结 $z_t=\phi(o_t)$，训练 predictor $P_\phi$。

规划时给定目标图像表征 $z_g$，搜索动作序列使未来表征接近目标：

$$
a_{0:T}^*
=
\operatorname*{arg\,min}_{a_{0:T}}
\left\|
P_\phi^{(T)}(z_0,a_{0:T})
-
z_g
\right\|_1.
$$

使用 model predictive control 反复重规划。


---

## 4 数据与训练

- 预训练数据：1M+ 小时互联网视频，1M 图像；
- 视频编码器可扩展到 ViT-g / 1B 量级；
- 可与 LLM 对齐做 video QA；
- V-JEPA 2-AC 使用 DROID 中少于 62 小时无标签视频；
- 不需要目标实验室数据、任务奖励或任务专用训练；
- predictor 约 300M 参数、24 层 Transformer；
- 部署在两个实验室的 Franka + RobotiQ 夹爪上，使用图像目标指定任务。


---

## 5 实验、结果与解读

论文报告：

- Something-Something v2 motion understanding 达到 77.3 top-1；
- Epic-Kitchens-100 action anticipation recall@5 达 39.7，超过此前专用模型；
- 8B LLM 对齐后在 PerceptionTest、TempCompass 等 video QA 上取得 SOTA；
- 零样本 Franka 任务中，V-JEPA 2-AC 在两个实验室均能完成 reach、grasp、pick-and-place；
- 在规划性能上优于视频生成世界模型基线；
- 相比 Cosmos 采样规划，V-JEPA 2-AC 每个动作规划约 16 秒，而论文中视频生成基线需约 4 分钟。


---

## 6 精读要点与实践建议

- **预测抽象特征比预测像素更适合决策**；
- **冻结视频编码器 + 小动作 predictor** 是高效的世界模型后训练范式；
- **零样本跨实验室部署**说明表征泛化能力；
- **视频理解与机器人规划可以共享底座**；
- **实践建议**：若目标是从图像目标生成动作，优先训练 latent action-conditioned dynamics，再考虑视频可视化；
- **与 Dreamer 的区别**：V-JEPA 2 从互联网视频预训练，不依赖任务 reward；与 Genie 的区别是不生成像素。


---

## 7 局限与后续影响

- 输出不是可解释视频，调试依赖表征探针或额外 decoder；
- 长程规划仍受限，复杂非贪心任务需要中间子目标；
- 依赖 DROID 的动作分布和相机设定，跨实体泛化仍需验证；
- 图像目标指定不如语言灵活；
- 与语言模型和高层规划的深度集成仍在发展。


---

## 相关阅读

- [DreamerV3 论文精读](./dreamer-v3.md)
- [Genie 论文精读](./genie.md)
- [π0.7 论文精读](./pi-0-7.md)
- [世界模型总览](../world-model.md)

---

## 参考文献

1. Assran et al. *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
2. Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video*, 2024.
3. Ha & Schmidhuber. *World Models*, 2018.

